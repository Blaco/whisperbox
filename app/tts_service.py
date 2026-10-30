"""WhisperBox private loopback TTS service."""
import argparse
import faulthandler
import os
import sys
import time
from datetime import datetime
from uuid import uuid4
import threading
import io
import base64
import tempfile
import json
import traceback
import random
import struct
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tts_longform import MAX_TEXT_CHARACTERS, split_speech, checked_audio_segments
import numpy as np
import soundfile as sf
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--model', choices=['original', 'turbo'], default='original')
parser.add_argument('--port', type=int, default=18765)
args = parser.parse_args()

if args.model == 'original':
    from chatterbox.tts import ChatterboxTTS as Model
else:
    from chatterbox.tts_turbo import ChatterboxTurboTTS as Model

# Offline-first: install/download steps populate these folders in advance.
ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / 'logs' / 'tts_diagnostics.jsonl'
FAULTS = ROOT / 'logs' / 'tts_faults.log'
DIAG.parent.mkdir(parents=True, exist_ok=True)
_fault_stream = FAULTS.open('a', encoding='utf-8', buffering=1)
faulthandler.enable(file=_fault_stream, all_threads=True)
_diag_lock = threading.Lock()

def trace(event, **fields):
    try:
        line = json.dumps({'timestamp': datetime.now().astimezone().isoformat(timespec='milliseconds'),
                           'source': 'tts_backend', 'event': event, 'pid': os.getpid(),
                           **fields}, default=str) + '\n'
        with _diag_lock:
            if DIAG.exists() and DIAG.stat().st_size > 5_000_000:
                rotated = DIAG.with_suffix('.jsonl.1')
                rotated.unlink(missing_ok=True)
                DIAG.replace(rotated)
            with DIAG.open('a', encoding='utf-8') as out:
                out.write(line)
    except Exception:
        pass

trace('backend_started', model=args.model, device=args.device,
      torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
      cuda_device=torch.cuda.get_device_name(0) if args.device == 'cuda' and torch.cuda.is_available() else None)
CKPT = ROOT / 'models' / ('chatterbox-original' if args.model == 'original' else 'chatterbox-turbo')
if not CKPT.is_dir() or not any(CKPT.glob('*.safetensors')):
    raise FileNotFoundError('Local Chatterbox weights missing: {}. Run Download_Models.bat while online.'.format(CKPT))
model_load_started=time.perf_counter()
trace('model_load_start', model=args.model, local_weights=True)
print('Loading Chatterbox', args.model, 'from', CKPT, flush=True)
model = Model.from_local(CKPT, device=args.device)
vc_model = None  # Loaded only when the user requests voice conversion.
# Conditioning is attached to the active model, with a single hot cache entry.
_cond_key = None
def ensure_conditioning(voice, settings):
    global _cond_key
    stat = voice.stat()
    normalize = bool(settings.get('turbo_norm_loudness', True)) if args.model == 'turbo' else None
    exaggeration = float(settings.get('exaggeration', .5)) if args.model == 'original' else None
    cache_enabled = bool(settings.get('voice_conditioning_cache', True))
    key = (str(voice), stat.st_size, stat.st_mtime_ns, normalize, exaggeration)
    begin = time.perf_counter()
    hit = cache_enabled and key == _cond_key and model.conds is not None
    if not hit:
        with torch.inference_mode():
            if args.model == 'turbo':
                model.prepare_conditionals(str(voice), norm_loudness=normalize)
            else:
                model.prepare_conditionals(str(voice), exaggeration=float(settings.get('exaggeration', .5)))
        _cond_key = key
    trace('conditioning', cache_hit=hit, elapsed_seconds=round(time.perf_counter()-begin, 4),
          voice_extension=voice.suffix.lower())
    return hit

trace('model_load_complete', model=args.model, elapsed_seconds=round(time.perf_counter()-model_load_started,4))
print('READY', flush=True)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/health':
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')

    def do_POST(self):
        global vc_model
        if self.path == '/shutdown':
            trace('graceful_shutdown_requested')
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'ok')
            self.wfile.flush()
            # HTTPServer.shutdown() must execute in a separate thread.
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        request_id = uuid4().hex[:12]
        started = time.perf_counter()
        trace('request_received', request_id=request_id, endpoint=self.path)
        if self.path not in ('/speak', '/speak-stream', '/convert'):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > 165_000_000:
                self.send_error(413)
                return
            obj = json.loads(self.rfile.read(length))
            voice = Path(obj['voice']).resolve()
            voices_root = Path(__file__).resolve().parent.parent / 'models' / 'voices'
            if not voice.is_file() or voice.suffix.lower() not in ('.wav','.mp3','.ogg') or voices_root.resolve() not in voice.parents:
                self.send_error(400, 'Invalid target voice audio')
                return
            if self.path == '/convert':
                trace('conversion_start', request_id=request_id, source_type='memory' if obj.get('audio_b64') else Path(obj.get('source','')).suffix.lower(), target_type=voice.suffix.lower())
                audio_bytes = base64.b64decode(obj['audio_b64'], validate=True) if obj.get('audio_b64') else None
                source = Path(obj.get('source','')).resolve() if audio_bytes is None else None
                if audio_bytes is None and (not source.is_file() or source.suffix.lower() not in ('.wav','.mp3','.ogg')):
                    self.send_error(400, 'Invalid source audio')
                    return
                if audio_bytes is not None and (len(audio_bytes) > 120_000_000 or len(audio_bytes) < 44):
                    self.send_error(400, 'Invalid audio size')
                    return
                if vc_model is None:
                    from chatterbox.vc import ChatterboxVC
                    original_weights = ROOT / 'models' / 'chatterbox-original'
                    if not (original_weights/'s3gen.safetensors').is_file():
                        raise FileNotFoundError('VC weights missing. Run Download_Models.bat.')
                    trace('vc_model_load_start', request_id=request_id)
                    print('Loading voice conversion model on demand...', flush=True)
                    vc_model = ChatterboxVC.from_local(original_weights, device=args.device)
                    trace('vc_model_load_complete', request_id=request_id)
                    print('VC READY', flush=True)
                if audio_bytes is not None:
                    # The voice-conversion library accepts file paths only. Stage
                    # input within our own data directory for this request and
                    # delete it immediately; no user-profile temp folder is used.
                    staging = ROOT / 'data'
                    staging.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix='vc-', dir=staging) as staging_dir:
                        source = Path(staging_dir) / 'input.wav'
                        source.write_bytes(audio_bytes)
                        with torch.inference_mode():
                            wav = vc_model.generate(audio=str(source), target_voice_path=str(voice))
                else:
                    with torch.inference_mode():
                        wav = vc_model.generate(audio=str(source), target_voice_path=str(voice))
                audio = np.squeeze(wav.detach().float().cpu().numpy())
                f = io.BytesIO()
                sf.write(f, audio, vc_model.sr, format='WAV')
                result = f.getvalue()
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', str(len(result)))
                self.end_headers()
                self.wfile.write(result)
                trace('conversion_complete', request_id=request_id, elapsed_seconds=round(time.perf_counter()-started, 3), wav_bytes=len(result))
                return
            text = obj['text'].strip()
            if not text or len(text) > MAX_TEXT_CHARACTERS:
                self.send_error(400, 'Invalid text')
                return
            trace('generation_start', request_id=request_id, model=args.model, text_characters=len(text), voice_extension=voice.suffix.lower())
            settings = obj.get('settings', {})
            opts = {'text': text, 'audio_prompt_path': None}
            prefix = 'original_' if args.model == 'original' else 'turbo_'
            seed = int(settings.get(prefix+'seed', 0))
            if seed:
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            if args.model == 'original':
                opts.update(exaggeration=float(settings.get('exaggeration', .5)),
                            cfg_weight=float(settings.get('cfg', .5)),
                            temperature=float(settings.get('original_temperature', .8)),
                            min_p=float(settings.get('original_min_p', .05)),
                            top_p=float(settings.get('original_top_p', 1.0)),
                            repetition_penalty=float(settings.get('original_repetition_penalty', 1.2)))
            else:
                opts.update(temperature=float(settings.get('turbo_temperature', .8)),
                            top_p=float(settings.get('turbo_top_p', .95)),
                            top_k=int(settings.get('turbo_top_k', 1000)),
                            repetition_penalty=float(settings.get('turbo_repetition_penalty', 1.2)),
                            norm_loudness=bool(settings.get('turbo_norm_loudness', True)))
            ensure_conditioning(voice, settings)
            pieces = split_speech(text, bool(obj.get('chunked', False)))
            trace('generation_plan', request_id=request_id, chunks=len(pieces),
                  safety_segmentation=(len(pieces) > 1 and not bool(obj.get('chunked', False))))
            streaming = self.path == '/speak-stream'
            headers_sent = False
            streamed_bytes = 0
            completed_chunks = 0
            output = io.BytesIO() if not streaming else None
            output_wav = None
            if not streaming:
                output_wav = sf.SoundFile(output, mode='w', samplerate=model.sr,
                                          channels=1, format='WAV', subtype='PCM_16')
            last_generate_seconds = [0.0]
            def synthesize_piece(piece):
                opts['text'] = piece
                generate_started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**opts)
                last_generate_seconds[0] = time.perf_counter() - generate_started
                return np.asarray(generated.detach().float().cpu().numpy()).reshape(-1)
            def log_retry(piece, depth, seconds, replacements):
                trace('generation_segment_retry', request_id=request_id,
                      characters=len(piece), depth=depth, audio_seconds=round(seconds, 3),
                      replacement_segments=replacements)
            try:
                for piece, audio, depth in checked_audio_segments(
                        pieces, synthesize_piece, model.sr, on_retry=log_retry):
                    if streaming:
                        if not headers_sent:
                            self.send_response(200)
                            self.send_header('Content-Type', 'application/x-whisperbox-wav-frames')
                            self.end_headers()
                            headers_sent = True
                        f = io.BytesIO()
                        sf.write(f, audio, model.sr, format='WAV', subtype='PCM_16')
                        part = f.getvalue()
                        streamed_bytes += len(part)
                        self.wfile.write(struct.pack('!I', len(part)))
                        self.wfile.write(part)
                        self.wfile.flush()
                    else:
                        output_wav.write(audio)
                        part = None
                    trace('generation_chunk', request_id=request_id, index=completed_chunks,
                          characters=len(piece), retry_depth=depth,
                          generation_seconds=round(last_generate_seconds[0], 4),
                          audio_seconds=round(len(audio) / model.sr, 3),
                          wav_bytes=len(part) if part is not None else None)
                    completed_chunks += 1
            finally:
                if output_wav is not None:
                    output_wav.close()
            if streaming:
                self.wfile.write(struct.pack('!I', 0))
                self.wfile.flush()
                total_bytes = streamed_bytes
            else:
                result = output.getvalue()
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', str(len(result)))
                self.end_headers()
                self.wfile.write(result)
                total_bytes = len(result)
            trace('generation_complete', request_id=request_id, elapsed_seconds=round(time.perf_counter()-started, 3), wav_bytes=total_bytes, chunks=completed_chunks)
        except Exception as exc:
            trace('request_failed', request_id=request_id, endpoint=self.path, elapsed_seconds=round(time.perf_counter()-started,3), error_type=type(exc).__name__, error=str(exc)[:1500])
            traceback.print_exc()
            print('ERROR:', repr(exc), flush=True)
            if self.path == '/speak-stream' and locals().get('headers_sent', False):
                self.close_connection = True  # No final zero frame: client rejects incomplete audio.
            else:
                self.send_error(500, 'Generation failed; see terminal log')

    def log_message(self, *args):
        pass

trace('server_listening', address='127.0.0.1', port=args.port)
server = HTTPServer(('127.0.0.1', args.port), Handler)
try:
    server.serve_forever(poll_interval=0.1)
finally:
    trace('backend_shutting_down')
    server.server_close()
    trace('backend_stopped')
    _fault_stream.flush()
    _fault_stream.close()
