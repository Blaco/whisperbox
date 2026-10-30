"""Whisper environment process. stdin: JSON lines; stdout: tagged JSON event lines."""
import argparse
import base64
import io
import collections
import json
import queue
import sys
import threading
import time
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

commands = queue.Queue()
settings = {}
def event(kind, **kw):
    print('EVENT:' + json.dumps({'kind': kind, **kw}), flush=True)

def read_commands():
    for line in sys.stdin:
        try:
            commands.put(json.loads(line))
        except Exception as exc:
            event('protocol_error', message=str(exc))
    commands.put({'action': 'quit'})
threading.Thread(target=read_commands, daemon=True).start()

def get(name, default):
    return settings.get(name, default)

def process_audio(vad, whisper, frames, speech_frames, samp, minimum, trailing_silence_frames=0):
    if speech_frames * 512 / 16000 < minimum or not frames:
        event('transcript', text='', audio_b64='', outcome='too_short',
              raw_segments=0, accepted_segments=0)
        return
    # Automatic VAD endings retain only a small margin after the last voiced block.
    if get('trim_trailing_silence', True) and trailing_silence_frames > 0:
        margin = max(0, round(float(get('trailing_margin', .20)) / .032))
        excess = max(0, trailing_silence_frames - margin)
        if excess:
            frames = frames[:-excess]
    if not frames:
        event('transcript', text='', audio_b64='', outcome='trimmed_empty',
              raw_segments=0, accepted_segments=0)
        return
    audio = np.concatenate(frames).astype(np.float32)
    if get('input_pitch', 0) or get('input_formant', 1.) != 1. or get('input_cleanup', False):
        try:
            from input_effects import process
            audio = process(audio, samp, get('input_pitch', 0), get('input_formant', 1.),
                            get('input_cleanup_strength', 1.0) if get('input_cleanup', False) else 0.)
        except Exception as exc:
            event('input_effect_warning', message='Input effects skipped: ' + str(exc))
    event('latency', stage='vad_endpoint', seconds=round(trailing_silence_frames*512/samp,4),
          audio_seconds=round(len(audio)/samp,3))
    event('busy', state=True, phase='transcribing')
    event('transcription_started', audio_seconds=round(len(audio) / samp, 3),
          speech_seconds=round(speech_frames * 512 / 16000, 3))
    try:
        kwargs = dict(language='en', beam_size=max(1, min(5, int(get('whisper_beam_size', 3)))), vad_filter=False)
        if get('whisper_filter', False):
            kwargs.update(
                no_speech_threshold=float(get('no_speech_threshold', .6)),
                log_prob_threshold=float(get('log_prob_threshold', -1.0)),
                compression_ratio_threshold=float(get('compression_ratio_threshold', 2.4)),
            )
        segments, _ = whisper.transcribe(audio, **kwargs)
        # Materialize the segments once. Whisper generation is lazy.
        parts = list(segments)
        raw_count = len(parts)
        if get('whisper_filter', False):
            # Extra filtering ONLY when BOTH no-speech probability is high and
            # token confidence is low; avoids dropping quiet valid speech based
            # solely on a single threshold.
            parts = [seg for seg in parts if not (
                seg.no_speech_prob > float(get('no_speech_threshold', .6))
                and seg.avg_logprob < float(get('log_prob_threshold', -1.0))
            )]
        text = ' '.join(s.text.strip() for s in parts).strip()
        # Only transmit audio to GUI if saving of input recordings is enabled.
        audio_b64 = ''
        if get('save_input', False):
            buffer = io.BytesIO()
            sf.write(buffer, audio, samp, format='WAV', subtype='PCM_16')
            audio_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
        outcome = ('recognized' if text else 'no_segments' if raw_count == 0
                   else 'filtered_all' if not parts else 'blank_segments')
        event('transcript', text=text, audio_b64=audio_b64, outcome=outcome,
              raw_segments=raw_count, accepted_segments=len(parts))
    except Exception as exc:
        event('error', message='Transcription: ' + str(exc))

def main():
    torch.set_num_threads(1)
    event('status', message='Loading Silero and Whisper...')
    vad = load_silero_vad(onnx=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    args = parser.parse_args()
    model_dir = Path(__file__).resolve().parent.parent / 'models' / 'whisper-small.en'
    if not (model_dir / 'model.bin').is_file():
        raise FileNotFoundError('Local Whisper weights missing: {}. Run Download_Models.bat while online.'.format(model_dir))
    whisper = WhisperModel(str(model_dir), device=args.device,
        compute_type='float16' if args.device == 'cuda' else 'int8', local_files_only=True)
    event('ready')
    stream = None
    recording = False
    frames = []
    pre = collections.deque(maxlen=12)
    start_count = silence_count = voiced = 0
    muted = False
    listening = False
    blocked = False
    conversion_capture = False
    conversion_token = None
    sample_rate = 16000
    gain_prev = 1.0
    last_meter = 0.0
    # Emit a small, bounded sample of VAD statistics after startup. This
    # distinguishes a live but silent input from speech below the VAD threshold.
    probe_started = time.monotonic()
    probe_last = probe_started
    probe_max_rms = 0.0
    probe_max_vad = 0.0
    probe_frames = 0
    probe_above_threshold = 0
    next_open_retry = 0.0
    last_open_error = None
    listener_session = None
    def reset():
        nonlocal frames, recording, pre, start_count, silence_count, voiced, gain_prev
        frames = []
        recording = False
        pre = collections.deque(maxlen=max(1, round(float(get('pre_roll', .35)) / .032), int(get('start_frames', 3))))
        start_count = silence_count = voiced = 0
        gain_prev = 1.0
        vad.reset_states()
    def stop_stream():
        nonlocal stream
        if stream is not None:
            if hasattr(stream, 'stop'):
                stream.stop()
            stream.close()
            stream = None
    def open_stream():
        nonlocal stream
        stop_stream()
        device = get('mic_device', 'default')
        if get('wasapi_enabled', False):
            import soundcard as sc
            if not str(device).startswith('loopback:'):
                raise ValueError('Select a WASAPI loopback capture device in the Audio tab.')
            identifier = str(device).split(':', 1)[1]
            if identifier == 'default':
                speaker = sc.default_speaker()
                if speaker is None:
                    raise RuntimeError('No default Windows playback device for loopback')
                microphone = next((m for m in sc.all_microphones(include_loopback=True) if m.isloopback and m.id == speaker.id), None)
                if microphone is None:
                    raise RuntimeError('Default playback device has no available WASAPI loopback endpoint')
            else:
                microphone = next((m for m in sc.all_microphones(include_loopback=True) if m.isloopback and str(m.id) == identifier), None)
                if microphone is None:
                    raise RuntimeError('Selected loopback device is no longer available; refresh Audio devices')
            if not microphone.isloopback:
                raise RuntimeError('Selected WASAPI source is not a loopback device')
            # SoundCard's Windows backend can have trouble capturing a single
            # channel; capture stereo then downmix to mono for Silero/Whisper.
            recorder = microphone.recorder(samplerate=16000, channels=2, blocksize=512)
            recorder.__enter__()
            class LoopbackStream:
                def read(self, size):
                    samples = np.asarray(recorder.record(numframes=size), dtype=np.float32)
                    if samples.ndim == 1:
                        samples = samples[:, None]
                    return samples.mean(axis=1, keepdims=True), False
                def close(self):
                    recorder.__exit__(None, None, None)
            stream = LoopbackStream()
        else:
            if device == 'default':
                device = None
            stream = sd.InputStream(device=device, samplerate=16000, blocksize=512, channels=1, dtype='float32')
            stream.start()
    while True:
        try:
            while True:
                cmd = commands.get_nowait()
                action = cmd.get('action')
                if action == 'quit':
                    stop_stream()
                    return
                if action == 'initialize':
                    # One startup command atomically applies the listening
                    # configuration before any capture device is opened.
                    listener_session = cmd.get('session')
                    settings.update(cmd.get('settings', {}))
                    listening = True
                    muted = bool(cmd.get('muted', False))
                    blocked = False
                    conversion_capture = False
                    stop_stream()
                    reset()
                    next_open_retry = 0.0
                    last_open_error = None
                    event('listener_initialized', session=listener_session,
                          muted=muted, device=str(get('mic_device', 'default')),
                          wasapi=bool(get('wasapi_enabled', False)))
                    # The main loop will arm the device and acknowledge
                    # capture_ready only after the stream actually opens.
                elif action == 'config':
                    old_device = (get('mic_device', 'default'), bool(get('wasapi_enabled', False)))
                    settings.update(cmd.get('settings', {}))
                    # Device changes wait until the current recording ends.
                    if (stream and listening and not muted and not recording and not blocked
                            and (get('mic_device', 'default'), bool(get('wasapi_enabled', False))) != old_device):
                        stop_stream()
                        reset()
                        event('capture_paused', reason='device_changed', session=listener_session)
                elif action == 'hold':
                    conversion_capture = False
                    blocked = True
                    stop_stream()
                    event('capture_paused', reason='hold', session=listener_session)
                elif action == 'mute':
                    muted = bool(cmd.get('value', False))
                    if muted:
                        reset()
                        stop_stream()
                        event('capture_paused', reason='muted', session=listener_session)
                elif action == 'listen':
                    listening = bool(cmd.get('value', False))
                    if not listening:
                        reset()
                        stop_stream()
                    # Only the main capture loop opens the stream, after
                    # settings have been applied. Avoid competing opens.
                elif action == 'conversion_record_start' and listening and not blocked and not muted:
                    reset()
                    conversion_capture = True
                    conversion_token = cmd.get('token')
                    recording = True
                elif action == 'conversion_record_stop' and conversion_capture and cmd.get('token') == conversion_token:
                    conversion_capture = False
                    recorded = np.concatenate(frames).astype(np.float32) if frames else np.zeros(0, dtype=np.float32)
                    reset()
                    if len(recorded):
                        buffer = io.BytesIO()
                        sf.write(buffer, recorded, 16000, format='WAV', subtype='PCM_16')
                        event('conversion_recorded', audio_b64=base64.b64encode(buffer.getvalue()).decode('ascii'), token=conversion_token)
                    else:
                        event('conversion_recorded', audio_b64='', token=conversion_token)
                    conversion_token = None
                elif action == 'resume':
                    # One deterministic unblock after transcription/playback.
                    # Repeated resume commands cannot reset active VAD state.
                    if blocked:
                        blocked = False
                        reset()
                        next_open_retry = 0.0
                        event('listener_resumed', session=listener_session,
                              muted=muted, listening=listening)
        except queue.Empty:
            pass
        except Exception as exc:
            # Audio device errors must not kill the loaded Whisper model.
            event('input_error', message=str(exc))
            try:
                stop_stream()
            except Exception:
                pass
            # Preserve the requested listening state. A transient device
            # error must not silently disarm the microphone until Echo changes.
            next_open_retry = time.monotonic() + 1.0
        if not listening or muted or blocked:
            time.sleep(.04)
            continue
        if stream is None:
            if time.monotonic() < next_open_retry:
                time.sleep(.04)
                continue
            try:
                open_stream()
                last_open_error = None
                next_open_retry = 0.0
                event('capture_ready', session=listener_session)
            except Exception as exc:
                detail = str(exc)
                if detail != last_open_error:
                    event('input_error', message=detail)
                    last_open_error = detail
                next_open_retry = time.monotonic() + 1.0
                time.sleep(.04)
                continue
        try:
            block, overflow = stream.read(512)
            audio = block[:, 0].copy()
            input_gain = float(get('input_gain', 1.0))
            audio = np.clip(audio * input_gain, -1, 1)
            # Soft noise gate: below-threshold signal fades toward attenuation level.
            threshold = float(get('gate_threshold', 0.0))
            gate_floor = float(get('gate_floor', 0.10))
            attack = max(float(get('gate_attack', .020)), .001)
            release = max(float(get('gate_release', .150)), .001)
            level = float(np.sqrt(np.mean(audio * audio)))
            target = 1.0 if threshold <= 0 or level >= threshold else gate_floor
            tau = attack if target > gain_prev else release
            gain_prev += (target - gain_prev) * (-np.expm1(-.032 / tau))
            audio *= gain_prev
            if overflow:
                event('status', message='Microphone overflow')
            if conversion_capture:
                frames.append(audio)
                if len(frames) >= 5 * 60 * 16000 // 512:
                    conversion_capture = False
                    recorded = np.concatenate(frames).astype(np.float32)
                    reset()
                    buffer = io.BytesIO()
                    sf.write(buffer, recorded, 16000, format='WAV', subtype='PCM_16')
                    event('conversion_recorded', audio_b64=base64.b64encode(buffer.getvalue()).decode('ascii'), token=conversion_token)
                    conversion_token = None
                continue
            p = float(vad(torch.from_numpy(audio), 16000).item())
            now = time.monotonic()
            if now - probe_started < 35.0:
                probe_frames += 1
                probe_max_rms = max(probe_max_rms, float(np.sqrt(np.mean(audio * audio))))
                probe_max_vad = max(probe_max_vad, p)
                if p >= float(get('vad_threshold', .50)):
                    probe_above_threshold += 1
                if now - probe_last >= 3.0:
                    event('vad_probe', frames=probe_frames,
                          max_rms=round(probe_max_rms, 5),
                          max_vad=round(probe_max_vad, 4),
                          above_threshold=probe_above_threshold,
                          threshold=float(get('vad_threshold', .50)),
                          device=str(get('mic_device', 'default')),
                          wasapi=bool(get('wasapi_enabled', False)))
                    probe_last = now
                    probe_frames = 0
                    probe_max_rms = 0.0
                    probe_max_vad = 0.0
            if time.monotonic()-last_meter >= .14:
                last_meter=time.monotonic()
                event('meter',level=min(1.0,float(np.sqrt(np.mean(audio*audio)))*5),vad=p)
            # Hysteresis: begin speech at the configured threshold, but allow
            # quiet syllables to remain part of an ongoing utterance.
            start_threshold = float(get('vad_threshold', .50))
            end_threshold = max(0.0, start_threshold - 0.15)
            yes = p >= (end_threshold if recording else start_threshold)
            if not recording:
                pre.append(audio)
                start_count = start_count + 1 if yes else 0
                if start_count >= int(get('start_frames', 3)):
                    recording = True
                    frames = list(pre)
                    voiced = start_count
                    silence_count = 0
                    event('recording')
                continue
            frames.append(audio)
            if yes:
                voiced += 1
                silence_count = 0
            else:
                silence_count += 1
            silence_frames = max(1, round(float(get('end_silence', 1.00)) / .032))
            max_frames = max(1, round(float(get('max_record', 45)) / .032))
            if silence_count >= silence_frames or len(frames) >= max_frames:
                collected = frames[:]
                count = voiced
                silence_count_at_end = silence_count if silence_count >= silence_frames else 0
                minimum = float(get('min_speech', .15))
                blocked = True
                stop_stream()
                event('capture_paused', reason='transcription', session=listener_session)
                reset()
                if count * .032 < minimum:
                    # No inference cycle is pending; resume capture directly.
                    blocked = False
                    continue
                process_audio(vad, whisper, collected, count, 16000, minimum,
                              trailing_silence_frames=silence_count_at_end)
        except Exception as exc:
            # An interrupted recording cannot be stitched across a device
            # failure. Explicitly resolve its UI state before retrying capture.
            if conversion_capture:
                captured = np.concatenate(frames).astype(np.float32) if frames else np.zeros(0, dtype=np.float32)
                token = conversion_token
                conversion_capture = False
                conversion_token = None
                reset()
                output = io.BytesIO()
                if captured.size:
                    sf.write(output, captured, 16000, format='WAV', subtype='PCM_16')
                event('conversion_recorded', audio_b64=base64.b64encode(output.getvalue()).decode('ascii'), token=token)
            elif recording:
                reset()
                event('capture_interrupted')
            event('input_error', message=str(exc))
            stop_stream()
            next_open_retry = time.monotonic() + 1.0

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        event('error', message=str(exc))
        raise
