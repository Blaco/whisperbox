"""WhisperBox: Windows PySide6 desktop frontend, no LLM dependencies."""
import io
_APP_IMPORT_STARTED = __import__("time").perf_counter()
import re
import zipfile
import platform
import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import sounddevice as sd
import soundfile as sf
from control_state import ControlSnapshot, control_availability
from tts_longform import MAX_SEGMENT_CHARACTERS, MAX_TEXT_CHARACTERS, join_wav_frames
from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QThread, Signal, QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtGui import QAction, QIcon, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QSystemTrayIcon,
    QTabWidget, QVBoxLayout, QWidget, QStackedWidget, QScrollArea, QProgressBar, QFrame, QGridLayout, QSizePolicy, QLayout
)

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / 'app'
CONFIG = ROOT / 'config' / 'settings.json'
DATA = ROOT / 'data'
VOICES = ROOT / 'models' / 'voices'
WHISPER_PY = (ROOT / 'runtime' / 'whisper' / 'Scripts' / 'python.exe' if (ROOT / 'runtime' / 'whisper' / 'Scripts' / 'python.exe').exists() else ROOT / 'runtime' / 'whisper' / 'python.exe')
CHATTERBOX_PY = (ROOT / 'runtime' / 'chatterbox' / 'Scripts' / 'python.exe' if (ROOT / 'runtime' / 'chatterbox' / 'Scripts' / 'python.exe').exists() else ROOT / 'runtime' / 'chatterbox' / 'python.exe')
URL = 'http://127.0.0.1:18765/speak'
CONVERT_URL = 'http://127.0.0.1:18765/convert'
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg'}

DIAG_LOG = ROOT / 'logs' / 'diagnostics.jsonl'
_DIAG_LOCK = threading.Lock()

def diagnostic(event, **fields):
    """Append a privacy-conscious, machine-readable diagnostic event."""
    try:
        DIAG_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {'timestamp': datetime.now().astimezone().isoformat(timespec='milliseconds'),
                 'source': 'gui', 'event': event, **fields}
        line = json.dumps(entry, ensure_ascii=True, default=str) + '\n'
        with _DIAG_LOCK:
            if DIAG_LOG.exists() and DIAG_LOG.stat().st_size > 5_000_000:
                backup = DIAG_LOG.with_suffix('.jsonl.1')
                backup.unlink(missing_ok=True)
                DIAG_LOG.replace(backup)
            with DIAG_LOG.open('a', encoding='utf-8') as handle:
                handle.write(line)
    except Exception:
        pass  # Diagnostics must never prevent inference or closing.


def _query_gpu_snapshot():
    """Best-effort memory usage; never block UI waiting for nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
             '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=1.5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode != 0:
            return {'available': False}
        snapshot = {'available': True, 'compute_processes': result.stdout.strip()[:1200]}
        # Windows WDDM frequently reports N/A for per-process GPU memory.
        # Aggregate GPU usage is still useful for confirming resource release.
        try:
            total = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                timeout=1.5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if total.returncode == 0:
                snapshot['gpus_memory_mib'] = total.stdout.strip()[:600]
        except (OSError, subprocess.TimeoutExpired):
            pass
        return snapshot
    except (OSError, subprocess.TimeoutExpired):
        return {'available': False}


_GPU_STATE = {'available': False, 'pending': True}
_GPU_STATE_LOCK=threading.Lock()
def gpu_snapshot():
    """Instantaneous cached snapshot; never launches a subprocess on the GUI thread."""
    with _GPU_STATE_LOCK:
        return dict(_GPU_STATE)

def _gpu_poll_loop():
    while True:
        latest=_query_gpu_snapshot()
        with _GPU_STATE_LOCK:
            _GPU_STATE.clear()
            _GPU_STATE.update(latest)
        time.sleep(8)

threading.Thread(target=_gpu_poll_loop, name='whisperbox-gpu-poll', daemon=True).start()

def format_exit_code(code):
    return '0x{:08X}'.format(code & 0xffffffff)

DEFAULTS = dict(compute_device='cpu', echo_enabled=True, model='original', voice='', conversion_audio='',
    mic_device='default', output_device='default', wasapi_enabled=False, input_gain=1.0, output_gain=1.0,
    input_pitch=0, input_formant=1.0, input_cleanup=False, input_cleanup_strength=1.0, playback_speed=1.0, vad_threshold=.50, start_frames=3, end_silence=1.00,
    pre_roll=.35, min_speech=.15, max_record=45.0,
    trim_trailing_silence=True, trailing_margin=.20, whisper_filter=False,
    no_speech_threshold=.60, log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
    gate_threshold=0.0,
    gate_floor=.10, gate_attack=.020, gate_release=.150,
    exaggeration=.50, cfg=.50, original_temperature=.8, original_min_p=.05, original_top_p=1.0, original_repetition_penalty=1.2, original_seed=0,
    turbo_temperature=.8, turbo_top_p=.95, turbo_top_k=1000, turbo_repetition_penalty=1.2, turbo_norm_loudness=True, turbo_seed=0, save_input=False, save_text=False, save_output=False,
    recordings_dir='', transcripts_dir='', generated_audio_dir='',
    tray_enabled=False, cuda_validation={}, whisper_beam_size=3, chunked_tts=True, voice_conditioning_cache=True, streaming_playback=True)

def load_settings():
    try:
        saved = json.loads(CONFIG.read_text(encoding='utf-8'))
        # Retired listening modes are never restored from older settings files.
        saved.pop('manual', None)
        saved.pop('mode', None)
        return {**DEFAULTS, **saved}
    except (OSError, ValueError):
        return DEFAULTS.copy()

def save_settings(values):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG.with_suffix('.tmp')
    tmp.write_text(json.dumps(values, indent=2), encoding='utf-8')
    tmp.replace(CONFIG)

SAVE_LOCATIONS = {
    'recordings': ('recordings_dir', 'recordings'),
    'transcripts': ('transcripts_dir', 'transcripts'),
    'generated_audio': ('generated_audio_dir', 'generated_audio'),
}

def saved_directory(config, kind):
    """Use the original data folder unless the user explicitly selects another."""
    setting, default_folder = SAVE_LOCATIONS[kind]
    custom = str(config.get(setting, '') or '').strip()
    return Path(custom).expanduser() if custom else DATA / default_folder

def voice_files():
    return sorted((p for p in VOICES.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS), key=lambda p: (0 if p.name == 'Chatterbox_Default.wav' else 1, str(p).lower())) if VOICES.exists() else []

class Synthesis(QThread):
    ready = Signal(bytes, str)
    chunk = Signal(bytes, int)
    stream_complete = Signal(str)
    failed = Signal(str)
    def __init__(self, text, path, model, settings):
        super().__init__()
        self.text = text
        self.path = path
        self.model = model
        self.settings = settings.copy()
    def run(self):
        try:
            chunked=bool(self.settings.get('chunked_tts', False))
            streaming=chunked and bool(self.settings.get('streaming_playback', False))
            if len(self.text.strip()) > MAX_TEXT_CHARACTERS:
                raise ValueError('TTS text exceeds the supported {} characters'.format(MAX_TEXT_CHARACTERS))
            # Long requests always use the framed service protocol to avoid
            # leaving a single blocking HTTP response open for the full book.
            # This does not turn on user-facing streaming playback.
            framed=streaming or len(self.text.strip()) > MAX_SEGMENT_CHARACTERS
            payload = json.dumps(dict(text=self.text, voice=self.path,
                settings=self.settings, chunked=chunked)).encode('utf-8')
            req = urllib.request.Request(URL + ('-stream' if framed else ''), data=payload, method='POST',
                headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=600) as response:
                if framed:
                    import struct
                    def exact(n):
                        chunks=[]
                        while n:
                            b=response.read(n)
                            if not b:raise EOFError('Incomplete TTS audio stream')
                            chunks.append(b); n-=len(b)
                        return b''.join(chunks)
                    frames=[]
                    index=0
                    while True:
                        size=struct.unpack('!I',exact(4))[0]
                        if size==0:break
                        if size>32_000_000:raise ValueError('Invalid TTS audio frame')
                        frame=exact(size)
                        if streaming:self.chunk.emit(frame,index)
                        else:frames.append(frame)
                        index+=1
                    if streaming:
                        self.stream_complete.emit(self.text)
                    else:
                        self.ready.emit(join_wav_frames(frames), self.text)
                else:
                    self.ready.emit(response.read(), self.text)
        except Exception as exc:
            self.failed.emit(str(exc))

class Conversion(QThread):
    ready = Signal(bytes, str)
    failed = Signal(str)

    def __init__(self, source, target, effects=(0., 1., 0.)):
        super().__init__()
        self.source, self.target, self.effects = source, target, effects

    def run(self):
        try:
            if isinstance(self.source, bytes):
                wav = self.source
                source_name = '<recording>'
            else:
                audio, rate = sf.read(self.source, dtype='float32', always_2d=False)
                if audio.ndim == 2: audio = audio.mean(axis=1)
                data = io.BytesIO()
                sf.write(data, audio, rate, format='WAV', subtype='PCM_16')
                wav = data.getvalue()
                source_name = self.source
            if self.effects != (0., 1., 0.):
                from input_effects import process
                audio, rate = sf.read(io.BytesIO(wav), dtype='float32')
                processed = process(audio, rate, *self.effects)
                result = io.BytesIO()
                sf.write(result, processed, rate, format='WAV', subtype='PCM_16')
                wav = result.getvalue()
            payload = json.dumps({'audio_b64': base64.b64encode(wav).decode('ascii'), 'voice': self.target}).encode('utf-8')
            req = urllib.request.Request(CONVERT_URL, data=payload, method='POST', headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as response:
                self.ready.emit(response.read(), source_name)
        except Exception as exc:
            self.failed.emit(str(exc))


class VoiceDropdown(QComboBox):
    """Refresh before opening, so new reference files appear without a button."""
    def showPopup(self):
        selected = self.currentData()
        self.blockSignals(True)
        self.clear()
        for p in voice_files():
            self.addItem(str(p.relative_to(VOICES)), str(p))
        index = self.findData(selected)
        if index >= 0:
            self.setCurrentIndex(index)
        self.blockSignals(False)
        super().showPopup()


class ConversionWaveform(QWidget):
    seek_requested = Signal(float)
    def __init__(self):
        super().__init__()
        self.peaks = np.zeros(0, dtype=np.float32)
        self.progress = 0.0
        self.setMinimumHeight(72)
        self.setMaximumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_audio(self, audio):
        data = np.asarray(audio, dtype=np.float32)
        if data.ndim > 1: data = data.mean(axis=1)
        if data.size:
            bins = min(520, max(1, data.size // 128))
            edges = np.linspace(0, data.size, bins + 1, dtype=int)
            self.peaks = np.array([np.max(np.abs(data[edges[i]:max(edges[i]+1,edges[i+1])])) for i in range(bins)], dtype=np.float32)
            peak = float(self.peaks.max())
            if peak > 0: self.peaks /= peak
        else:
            self.peaks = np.zeros(0, dtype=np.float32)
        self.progress = 0.0
        self.update()

    def set_progress(self, value):
        self.progress = max(0.0, min(1.0, float(value)))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.peaks.size:
            fraction = max(0.0, min(1.0, event.position().x() / max(1, self.width())))
            self.set_progress(fraction)
            self.seek_requested.emit(fraction)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#19212a'))
        width, height = self.width(), self.height()
        mid = height / 2
        p.setPen(QPen(QColor('#51606e'), 1))
        p.drawLine(0, int(mid), width, int(mid))
        if self.peaks.size:
            count = len(self.peaks)
            for i, value in enumerate(self.peaks):
                x = int((i + .5) * width / count)
                amplitude = max(1, int(value * (mid - 6)))
                p.setPen(QPen(QColor('#4bd5ad') if i / count <= self.progress else QColor('#71a4cd'), max(1, width // count)))
                p.drawLine(x, int(mid - amplitude), x, int(mid + amplitude))
            p.setPen(QPen(QColor('#ffffff'), 2))
            x = int(self.progress * max(0, width - 1))
            p.drawLine(x, 0, x, height)
        else:
            p.setPen(QColor('#93a1af'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Record or load audio to preview')
        p.end()


def playback_audio_for_device(audio, sample_rate, device):
    """Convert playback samples to a rate accepted by the selected output endpoint.

    Loopback capture can be 16 kHz while WASAPI output devices often only
    accept their shared-mode mix rate (typically 48 kHz).
    """
    import math
    import scipy.signal
    data = np.asarray(audio, dtype=np.float32)
    index = None if device == 'default' else int(device)
    try:
        sd.check_output_settings(device=index, samplerate=sample_rate,
                                 channels=(1 if data.ndim == 1 else data.shape[1]),
                                 dtype='float32')
        return data, int(sample_rate)
    except sd.PortAudioError:
        info = sd.query_devices(index, 'output')
        rate = int(round(info['default_samplerate']))
        if rate <= 0:
            raise ValueError('Output device does not report a valid playback rate')
        divisor = math.gcd(rate, int(sample_rate))
        data = scipy.signal.resample_poly(data, rate // divisor,
                                          int(sample_rate) // divisor, axis=0)
        sd.check_output_settings(device=index, samplerate=rate,
                                 channels=(1 if data.ndim == 1 else data.shape[1]),
                                 dtype='float32')
        return np.ascontiguousarray(data, dtype=np.float32), rate


class ConversionPreview(QThread):
    progressed = Signal(float)
    stopped = Signal()
    failed = Signal(str)
    def __init__(self, audio, rate, device, effects=(0.,1.,0.)):
        super().__init__()
        self.effects = effects
        self.audio = np.asarray(audio, dtype=np.float32)
        self.rate = rate
        self.device = device
        self.pause_requested = threading.Event()
        self.stop_requested = threading.Event()
        self._seek_lock = threading.Lock()
        self._seek_fraction = 0.0
        self._seek_pending = False

    def seek(self, fraction):
        with self._seek_lock:
            self._seek_fraction = max(0.0, min(1.0, float(fraction)))
            self._seek_pending = True

    def _take_seek(self):
        with self._seek_lock:
            if not self._seek_pending:
                return None
            self._seek_pending = False
            return self._seek_fraction

    def run(self):
        try:
            device = None if self.device == 'default' else int(self.device)
            data = self.audio.mean(axis=1) if self.audio.ndim == 2 else self.audio
            if self.effects != (0., 1., 0.):
                from input_effects import process
                data = process(data, self.rate, *self.effects)
            data, playback_rate = playback_audio_for_device(data, self.rate, self.device)
            data = data.reshape(-1, 1)
            with sd.OutputStream(samplerate=playback_rate, channels=data.shape[1], device=device, dtype='float32') as stream:
                offset = 0
                while not self.stop_requested.is_set():
                    seek = self._take_seek()
                    if seek is not None:
                        # Seek is independent of Play/Pause, including while paused.
                        offset = min(max(0, int(seek * len(data))), len(data))
                        self.progressed.emit(offset / len(data) if len(data) else 0.)
                    if offset >= len(data):
                        break
                    if self.pause_requested.is_set():
                        time.sleep(.025)
                        continue
                    part = data[offset:offset + 512]
                    stream.write(part)
                    offset += len(part)
                    self.progressed.emit(offset / len(data))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.stopped.emit()

class Playback(QThread):
    finished_playback = Signal()
    failed = Signal(str)
    def __init__(self, wav, device, gain, speed):
        super().__init__()
        self.wav, self.device, self.gain, self.speed = wav, device, gain, speed
    def run(self):
        try:
            audio, sample_rate = sf.read(io.BytesIO(self.wav), dtype='float32')
            # Speed change by resampling playback rate; also changes pitch.
            # Keep playback output sample rate native; interpolate samples instead.
            if abs(self.speed - 1) > .001:
                positions = np.arange(0, len(audio), self.speed)
                x = np.arange(len(audio))
                if audio.ndim == 1:
                    audio = np.interp(positions, x, audio).astype('float32')
                else:
                    audio = np.column_stack([np.interp(positions, x, audio[:,i]) for i in range(audio.shape[1])]).astype('float32')
            audio = np.clip(audio * self.gain, -1, 1)
            device = None if self.device == 'default' else int(self.device)
            audio, playback_rate = playback_audio_for_device(audio, sample_rate, self.device)
            sd.play(audio, samplerate=playback_rate, device=device, blocking=True)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished_playback.emit()

class StreamPlayback(QThread):
    started_playback=Signal()
    finished_playback=Signal()
    failed=Signal(str)
    def __init__(self,device,gain,speed):
        super().__init__()
        import queue
        self.frames=queue.Queue()
        self.device,self.gain,self.speed=device,gain,speed
        self.stop_requested=threading.Event()
    def add_chunk(self,wav):
        self.frames.put(wav)
    def finish_input(self):
        self.frames.put(None)
    def halt(self):
        self.stop_requested.set()
        self.frames.put(None)
        sd.stop()
    def run(self):
        try:
            device=None if self.device=='default' else int(self.device)
            with_stream=None
            try:
                while not self.stop_requested.is_set():
                    wav=self.frames.get()
                    if wav is None:break
                    audio,sample_rate=sf.read(io.BytesIO(wav),dtype='float32')
                    if abs(self.speed-1)>.001:
                        positions=np.arange(0,len(audio),self.speed)
                        x=np.arange(len(audio))
                        if audio.ndim==1:audio=np.interp(positions,x,audio).astype('float32')
                        else:audio=np.column_stack([np.interp(positions,x,audio[:,i]) for i in range(audio.shape[1])]).astype('float32')
                    audio=np.clip(audio*self.gain,-1,1).astype('float32')
                    audio, playback_rate = playback_audio_for_device(audio, sample_rate, self.device)
                    channels=1 if audio.ndim==1 else audio.shape[1]
                    if with_stream is None:
                        with_stream=sd.OutputStream(samplerate=playback_rate,channels=channels,device=device,dtype='float32')
                        with_stream.start()
                        self.started_playback.emit()
                    for offset in range(0,len(audio),1024):
                        if self.stop_requested.is_set():break
                        frame=audio[offset:offset+1024]
                        if frame.ndim==1:frame=frame[:,None]
                        with_stream.write(frame)
            finally:
                if with_stream is not None:
                    with_stream.stop()
                    with_stream.close()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished_playback.emit()


def perform_startup_checks(compute_device):
    """Blocking filesystem and subprocess checks; safe to call from a worker thread."""
    import importlib.util
    errors=[];notes=[]
    for name,path in (('Whisper',WHISPER_PY),('Chatterbox',CHATTERBOX_PY)):
        (notes if path.is_file() else errors).append(name+' runtime: '+('OK' if path.is_file() else 'MISSING'))
    model_files=[('Whisper weights',ROOT/'models'/'whisper-small.en'/'model.bin'),
                 ('Original model',ROOT/'models'/'chatterbox-original'/'s3gen.safetensors'),
                 ('Turbo model',ROOT/'models'/'chatterbox-turbo'/'s3gen.safetensors')]
    for name,path in model_files:
        (notes if path.is_file() and path.stat().st_size > 1024 else errors).append(name+': '+('OK' if path.is_file() and path.stat().st_size > 1024 else 'MISSING / EMPTY — run Download_Models.bat'))
    # One additional startup check covers SoundCard in both the GUI and Whisper runtimes.
    try:
        import soundcard
        gui_soundcard=True
    except (ImportError, OSError):
        gui_soundcard=False
    whisper_soundcard=False
    if WHISPER_PY.is_file():
        try:
            checked=subprocess.run([str(WHISPER_PY),'-c','import soundcard'],
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,timeout=8,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            whisper_soundcard=(checked.returncode==0)
        except (OSError,subprocess.TimeoutExpired):
            pass
    if gui_soundcard and whisper_soundcard:
        notes.append('SoundCard / WASAPI: GUI and Whisper runtimes OK')
    else:
        missing=[]
        if not gui_soundcard:missing.append('GUI')
        if not whisper_soundcard:missing.append('Whisper')
        errors.append('SoundCard / WASAPI: missing in '+', '.join(missing)+' runtime; rerun Install_WhisperBox.bat')
    # Dependency checks prove that the actual worker runtimes can import their entry-point dependencies.
    for name, py, imports in (
        ('Whisper', WHISPER_PY, 'import faster_whisper, silero_vad, torch, soundcard'),
        ('Chatterbox', CHATTERBOX_PY, 'import chatterbox, torch, torchaudio, perth')):
        if py.is_file():
            try:
                probe=subprocess.run([str(py), '-c', imports], capture_output=True,
                    text=True, timeout=45, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if probe.returncode:
                    errors.append(name+' dependency import failed: '+probe.stderr[-500:])
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(name+' dependency import failed: '+str(exc))
    voices=voice_files()
    (notes if voices else errors).append('Reference voices: '+str(len(voices))+' found')
    # CUDA readiness is a property of the installed runtimes, not the currently
    # selected inference mode. CPU-only installs must never display a false GPU PASS.
    cuda_ready = True
    cuda_failures = []
    for backend, python in (('Whisper', WHISPER_PY), ('Chatterbox', CHATTERBOX_PY)):
        if not python.is_file():
            cuda_ready = False
            cuda_failures.append(backend + ' Python runtime missing')
            continue
        try:
            probe = subprocess.run([str(python), '-c',
                'import torch; assert torch.version.cuda is not None, "CPU-only PyTorch build"; '
                'assert torch.cuda.is_available(), "CUDA device unavailable"; '
                'x=torch.ones(1,device="cuda"); assert x.item()==1'],
                capture_output=True, text=True, timeout=45,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if probe.returncode:
                cuda_ready = False
                lines = (probe.stderr or probe.stdout or 'CUDA initialization failed').strip().splitlines()
                cuda_failures.append(backend + ': ' + (lines[-1] if lines else 'CUDA initialization failed'))
        except (OSError, subprocess.TimeoutExpired) as exc:
            cuda_ready = False
            cuda_failures.append(backend + ': ' + str(exc))
    cuda_detail = ('Both inference runtimes initialized CUDA successfully' if cuda_ready
                   else '; '.join(cuda_failures))
    if cuda_ready:
        notes.append('GPU: CUDA ready — ' + cuda_detail)
    else:
        errors.append('GPU: CUDA unavailable — ' + cuda_detail)

    return notes, errors, cuda_ready, cuda_detail



def cuda_runtime_fingerprint():
    """Hash installed Python/Torch artifacts, not just hardware name or a prior boolean.

    A changed, missing, or replaced runtime invalidates the remembered CUDA PASS.
    Live CUDA probing still runs asynchronously on every launch.
    """
    evidence=[]
    for label, executable in (('whisper', WHISPER_PY), ('chatterbox', CHATTERBOX_PY)):
        env_root=executable.parent.parent if executable.parent.name.lower()=='scripts' else executable.parent
        torch_root=env_root/'Lib'/'site-packages'/'torch'
        version_file=torch_root/'version.py'
        native_modules = sorted(torch_root.glob('_C*.pyd'))
        if not native_modules:
            return None
        native_module = native_modules[0]
        if not all(path.is_file() for path in (executable,version_file,native_module)):
            return None
        evidence.append(label)
        for path in (executable, version_file, native_module):
            stat=path.stat()
            evidence.extend((str(path.resolve()).lower(),str(stat.st_size),str(stat.st_mtime_ns)))
        # The Torch version file indicates +cpu versus +cu126; include actual bytes.
        evidence.append(hashlib.sha256(version_file.read_bytes()).hexdigest())
    return hashlib.sha256('\n'.join(evidence).encode('utf-8')).hexdigest()


class StartupValidationWorker(QThread):
    completed = Signal(list, list, bool, str)

    def __init__(self, compute_device, parent=None):
        super().__init__(parent)
        self.compute_device = compute_device

    def run(self):
        try:
            notes, errors, cuda_ready, cuda_detail = perform_startup_checks(self.compute_device)
        except Exception as exc:
            notes, errors, cuda_ready, cuda_detail = [], ['Startup checks failed: ' + str(exc)], False, str(exc)
        self.completed.emit(notes, errors, cuda_ready, cuda_detail)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_settings()
        self._cuda_fingerprint = cuda_runtime_fingerprint()
        remembered = self.cfg.get('cuda_validation', {})
        self._cuda_cached = (isinstance(remembered, dict) and
            remembered.get('verified') is True and
            bool(self._cuda_fingerprint) and
            remembered.get('runtime_fingerprint') == self._cuda_fingerprint)
        self._cuda_live_checked = False
        self._cuda_ready = bool(self._cuda_cached)
        self._cuda_reason = ('Previously validated; refreshing CUDA checks in the background...'
                             if self._cuda_cached else
                             'Checking CUDA support in both installed inference runtimes...')
        self._saved_cuda_requested = self.cfg.get('compute_device') == 'cuda' and not self._cuda_cached
        self._cuda_auto_select_pending = not self._cuda_cached
        if self._saved_cuda_requested:
            # Don't trust a CUDA preference if the runtimes changed since validation.
            self.cfg['compute_device'] = 'cpu'
        if remembered and not self._cuda_cached:
            self.cfg['cuda_validation'] = {}
            diagnostic('cuda_cache_invalidated', reason='Installed runtimes changed or missing')
        self.workers = []
        self.whisper = None
        self.tts = None
        self.model_ready = False
        self.tts_ready = False
        self.running = False
        self.listening = False
        self.muted = False
        self.busy = False
        self.most_recent_wav = None
        self.audio_last = None
        self.session = None
        self.pending_speech = None
        self.pending_conversion = None
        self.playback_task = None
        self.streaming_chunks=[]
        self.streaming_complete=False
        self.streaming_started_at=None
        self.echo_cycle_started_at=None
        self._generation_epoch = 0
        self.mic_level = 0
        self.mic_vad = 0
        # Model-ready, configuration-applied and capture-ready are distinct.
        # Do not infer microphone readiness from model startup or Echo state.
        self._listener_session = 0
        self._listener_initialized = False
        self._listener_capture_ready = False
        self._listener_first_frame = False
        self._vad_last_report = None
        self.setWindowTitle('WhisperBox — Whisper + Chatterbox')
        self.setWindowIcon(QIcon(str(ROOT / 'assets' / 'WhisperBox.ico')))
        self.last_tts_lines=[]
        self.tts_buf=''
        self.shutting_down=False
        self.process_intents = {}
        self.process_started = {}
        self.current_operation = 'idle'
        self.last_tts_intent = None
        self.tts_active_model = None
        self.switch_requested = False
        self.device_restart_pending = False
        self.recording_active = False
        self.conversion_recording = False
        self.conversion_finalizing = False
        self._conversion_token = None
        self._conversion_serial = 0
        self._conversion_record_timer = QTimer(self)
        self._conversion_record_timer.setSingleShot(True)
        self._conversion_record_timer.setInterval(30000)
        self._conversion_record_timer.timeout.connect(self.conversion_record_timeout)
        self.conversion_recording_wav = None
        self.conversion_source_kind = "file"
        self.conversion_audio_data = None
        self.conversion_audio_rate = None
        self.conversion_preview = None
        self._retiring_previews = []
        self.device_restart_in_progress = False
        # Batch rapid slider edits without postponing UI state updates.
        self._settings_flush_timer = QTimer(self)
        self._settings_flush_timer.setSingleShot(True)
        self._settings_flush_timer.setInterval(180)
        self._settings_flush_timer.timeout.connect(self.flush_settings)
        self._settings_dirty = False
        self.resize(900, 920)
        # Horizontal resizing may clip dense tab controls, but the default size stays unchanged.
        self.setMinimumSize(320, 0)
        # Keep one status display visible below every tab; do not duplicate it.
        self.central_area = QWidget(self)
        self.central_layout = QVBoxLayout(self.central_area)
        self.central_layout.setContentsMargins(6, 6, 6, 6)
        self.central_layout.setSpacing(6)
        self.central_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.tabs = QTabWidget(self.central_area)
        self.central_layout.addWidget(self.tabs, 1)
        self.status_frame = QFrame(self.central_area)
        self.status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.status_frame.setFrameShadow(QFrame.Shadow.Sunken)
        self.status_layout = QHBoxLayout(self.status_frame)
        self.status_layout.setContentsMargins(10, 5, 10, 5)
        self.status_label = QLabel('Status:', self.status_frame)
        self.status = QLabel('Models stopped — compute resources released.', self.status_frame)
        self.status.setWordWrap(True)
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addWidget(self.status, 1)
        self.central_layout.addWidget(self.status_frame)
        self.setCentralWidget(self.central_area)
        self.numeric_sliders = {}
        self.create_session()
        self.create_audio()
        self.create_vad()
        self.create_logs()
        self.create_settings()
        self.install_tray()
        self.install_help()
        self.refresh_voices()
        self.refresh_devices()
        if self.conversion_source.text().strip() and Path(self.conversion_source.text().strip()).is_file():
            self.load_conversion_preview(self.conversion_source.text().strip())
        self.changed()
        self.update_last_output_button()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_services)
        self.timer.start(600)
        diagnostic('app_started', device=self.cfg['compute_device'], model=self.cfg['model'], python=sys.version.split()[0])

    def make_widget(self):
        w=QWidget(); v=QVBoxLayout(w); return w,v
    def row(self, *items):
        h=QHBoxLayout()
        for item in items:
            if isinstance(item,str): h.addWidget(QLabel(item))
            else: h.addWidget(item)
        return h
    def selector(self, opts, key):
        w=QComboBox(); w.addItems(opts)
        idx=w.findText(str(self.cfg[key]))
        if idx>=0:w.setCurrentIndex(idx)
        w.currentTextChanged.connect(self.changed)
        return w
    def _update_cuda_selector(self):
        """CUDA is selectable only after both installed runtimes pass a live CUDA test."""
        cuda_index = self.compute.findText('cuda')
        if cuda_index < 0:
            return
        item = self.compute.model().item(cuda_index)
        if item is not None:
            item.setEnabled(bool(self._cuda_ready))
        self.compute.setItemData(cuda_index,
            ('CUDA is available in both Whisper and Chatterbox runtimes.' if self._cuda_ready
             else 'CUDA unavailable: ' + self._cuda_reason),
            Qt.ItemDataRole.ToolTipRole)
        self.compute.setToolTip('Inference device. ' +
            ('CUDA is available.' if self._cuda_ready else 'CUDA disabled: ' + self._cuda_reason))
        if not self._cuda_ready and self.compute.currentText() == 'cuda':
            self.compute.setCurrentText('cpu')

    def number(self,key,lo,hi,step,dec=2):
        w=QDoubleSpinBox(); w.setRange(lo,hi); w.setDecimals(dec); w.setSingleStep(step)
        w.setValue(float(self.cfg[key])); w.valueChanged.connect(self.changed)
        return w
    def integer(self,key,lo,hi):
        w=QSpinBox();w.setRange(lo,hi);w.setValue(int(self.cfg[key]));w.valueChanged.connect(self.changed);return w
    def numeric_row(self, form, label, spin):
        """Keep an editable spinbox at right and add a synchronized slider at left."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        slider = QSlider(Qt.Orientation.Horizontal)
        minimum = spin.minimum()
        maximum = spin.maximum()
        if isinstance(spin, QDoubleSpinBox):
            decimals = spin.decimals()
            scale = 10 ** decimals
            slider.setRange(round(minimum * scale), round(maximum * scale))
            slider.setSingleStep(max(1, round(spin.singleStep() * scale)))
            slider.setPageStep(max(1, round((maximum-minimum) * scale / 10)))
            slider.setValue(round(spin.value() * scale))
            slider.valueChanged.connect(lambda value, box=spin, k=scale: box.setValue(value / k))
            spin.valueChanged.connect(lambda value, bar=slider, k=scale: bar.setValue(round(value * k)))
        else:
            slider.setRange(minimum, maximum)
            slider.setSingleStep(1)
            slider.setPageStep(max(1, (maximum-minimum)//10))
            slider.setValue(spin.value())
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
        spin.setFixedWidth(90)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        slider.setToolTip(spin.toolTip())
        layout.addWidget(slider, 1)
        layout.addWidget(spin, 0)
        # Reset only this setting. Always use the application's factory defaults,
        # not the user's previously saved value.
        key = next((name for name, widget in self.__dict__.items() if widget is spin), None)
        # numeric_row is invoked immediately after assigning the widget to self.
        reset = QPushButton('↺')
        reset.setFixedSize(30, 28)
        reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset.setToolTip('Reset this setting to its factory default.')
        if key is not None:
            setting_key = {
                'speed': 'playback_speed', 'threshold': 'vad_threshold',
                'silence': 'end_silence', 'preroll': 'pre_roll',
                'minimum': 'min_speech', 'maximum': 'max_record',
                'cfg_weight': 'cfg', 'pitch_input': 'input_pitch',
                'formant_input': 'input_formant', 'cleanup_input': 'input_cleanup_strength',
            }.get(key, key)
            # Clicking triggers the existing settings-save
            # and slider synchronization mechanisms via spin.valueChanged.
            if setting_key in DEFAULTS:
                reset.clicked.connect(lambda checked=False, box=spin, value=DEFAULTS[setting_key]: box.setValue(value))
            else:
                reset.setEnabled(False)
        else:
            reset.setEnabled(False)
        layout.addWidget(reset, 0)
        form.addRow(label, row)
        self.numeric_sliders[spin] = slider
        return row

    def flag(self,label,key):
        w=QCheckBox(label);w.setChecked(bool(self.cfg[key]));w.toggled.connect(self.changed);return w

    def create_session(self):
        w,v=self.make_widget()
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(w)
        self.tabs.addTab(scroll,'Generate')
        f=QFormLayout()
        # Keep inference hardware and voice model on the same row.
        self.compute=self.selector(['cpu','cuda'],'compute_device')
        self._update_cuda_selector()
        self.model=self.selector(['original','turbo'],'model')
        device_model_row=QHBoxLayout()
        device_model_row.addWidget(QLabel('Inference device:'))
        device_model_row.addWidget(self.compute, 1)
        device_model_row.addSpacing(12)
        device_model_row.addWidget(QLabel('Chatterbox model:'))
        device_model_row.addWidget(self.model, 1)
        f.addRow(device_model_row)
        self.echo=self.flag('Echo transcriptions','echo_enabled')
        self.voice=VoiceDropdown();self.voice.currentIndexChanged.connect(self.changed)
        first_row=QHBoxLayout()
        first_row.addWidget(self.echo)
        first_row.addSpacing(12)
        first_row.addWidget(QLabel('Reference audio:'))
        first_row.addWidget(self.voice, 1)
        f.addRow(first_row)
        self.conversion_source=QLineEdit()
        self.conversion_source.setReadOnly(True)
        self.conversion_source.mousePressEvent = self.conversion_source_clicked
        self.conversion_source.setPlaceholderText('No conversion audio selected')
        self.conversion_source.setText(self.cfg.get('conversion_audio',''))
        self.conversion_source.setMinimumWidth(170)
        self.conversion_source.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.conversion_record_button=QPushButton('RECORD AUDIO')
        self.conversion_record_button.setToolTip('Record a microphone clip for Voice Conversion. Click again to stop. This is separate from microphone listening and does not transcribe or trigger Echo. The clip stays in memory and replaces the conversion source until you browse another file.')
        self.conversion_record_button.clicked.connect(self.toggle_conversion_record)
        self.convert_button=QPushButton('CONVERT VOICE')
        self.convert_button.clicked.connect(self.convert_voice)
        self.browse_conversion=QPushButton('Browse...')
        self.browse_conversion.clicked.connect(self.pick_conversion_audio)
        conversion_row=QHBoxLayout()
        conversion_row.addWidget(self.conversion_record_button, 1)
        conversion_row.addWidget(self.convert_button, 1)
        conversion_row.addWidget(QLabel('Load audio file:'))
        conversion_row.addWidget(self.conversion_source, 1)
        conversion_row.addWidget(self.browse_conversion)
        f.addRow(conversion_row)
        waveform_row=QHBoxLayout()
        self.conversion_play=QPushButton('PLAY')
        self.conversion_play.setFixedWidth(72)
        self.conversion_play.clicked.connect(self.toggle_conversion_preview)
        self.conversion_play.setEnabled(False)
        self.conversion_wave=ConversionWaveform()
        self.conversion_wave.setToolTip('Click anywhere on the waveform to seek. Seeking does not start or pause playback; Play/Pause controls playback separately.')
        self.conversion_wave.seek_requested.connect(self.seek_conversion_preview)
        waveform_row.addWidget(self.conversion_play)
        waveform_row.addWidget(self.conversion_wave, 1)
        f.addRow(waveform_row)
        v.addLayout(f)
        self.model_settings=QStackedWidget()
        original_group=QGroupBox('Original — generation parameters')
        of=QFormLayout(original_group)
        self.exaggeration=self.number('exaggeration',.25,1.5,.05)
        self.cfg_weight=self.number('cfg',0,1,.05)
        self.original_temperature=self.number('original_temperature',.2,1.5,.05)
        self.original_min_p=self.number('original_min_p',0,.3,.01)
        self.original_top_p=self.number('original_top_p',.5,1,.01)
        self.original_repetition_penalty=self.number('original_repetition_penalty',1,1.8,.05)
        self.original_seed=self.integer('original_seed',0,1000)
        for label,widget in [('Exaggeration',self.exaggeration),('CFG / pacing',self.cfg_weight),('Temperature',self.original_temperature),('Min-P',self.original_min_p),('Top-P',self.original_top_p),('Repetition penalty',self.original_repetition_penalty),('Seed (0 = random)',self.original_seed)]:
            self.numeric_row(of, label+':', widget)
        turbo_group=QGroupBox('Turbo — generation parameters')
        tf=QFormLayout(turbo_group)
        self.turbo_temperature=self.number('turbo_temperature',.2,1.5,.05)
        self.turbo_top_p=self.number('turbo_top_p',.5,1,.01)
        self.turbo_top_k=self.integer('turbo_top_k',1,1000)
        self.turbo_repetition_penalty=self.number('turbo_repetition_penalty',1,1.8,.05)
        self.turbo_norm_loudness=self.flag('Normalize reference loudness','turbo_norm_loudness')
        self.turbo_seed=self.integer('turbo_seed',0,1000)
        for label,widget in [('Temperature',self.turbo_temperature),('Top-P',self.turbo_top_p),('Top-K',self.turbo_top_k),('Repetition penalty',self.turbo_repetition_penalty),('Seed (0 = random)',self.turbo_seed)]:
            self.numeric_row(tf, label+':', widget)
        tf.addRow(self.turbo_norm_loudness)
        self.model_settings.addWidget(original_group)
        self.model_settings.addWidget(turbo_group)
        v.addWidget(self.model_settings)
        self.global_button=QPushButton('START MODELS');self.global_button.clicked.connect(self.toggle_models)
        self.record_button=QPushButton('MUTE MICROPHONE');self.record_button.clicked.connect(self.toggle_listening)
        v.addLayout(self.row(self.global_button,self.record_button))
        v.addWidget(QLabel('Type text to generate cloned speech without microphone:'))
        self.typed=QPlainTextEdit();self.typed.setMaximumHeight(95);v.addWidget(self.typed)
        self.speak_button=QPushButton('GENERATE SPEECH'); self.speak_button.clicked.connect(self.speak_typed)
        v.addWidget(self.speak_button)
        v.addWidget(QLabel('Most recent generated audio:'))
        self.play_button=QPushButton('PLAY LAST');self.play_button.clicked.connect(self.play_last)
        self.open_button=QPushButton('OPEN FILE LOCATION');self.open_button.clicked.connect(self.open_last)
        v.addLayout(self.row(self.play_button,self.open_button))
        # Status is shown in the persistent footer below all tabs.
        self.cancel_button=QPushButton('CANCEL GENERATION')
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip('Cancel the current generation. If Chatterbox cannot interrupt immediately, only its service is restarted; Whisper stays loaded.')
        self.cancel_button.clicked.connect(self.cancel_generation)
        v.addWidget(self.cancel_button)
        self.model_state_label=QLabel('Whisper: Stopped   |   Chatterbox: Stopped   |   Microphone: Off   |   Voice conversion: Idle')
        v.addWidget(self.model_state_label)
        v.addWidget(QLabel('Live microphone input (audio level / speech detection):'))
        self.mic_meter=QProgressBar(); self.mic_meter.setRange(0,100); self.mic_meter.setValue(0)
        self.mic_meter.setFormat('Microphone level: %p%'); v.addWidget(self.mic_meter)
        self.vad_state_label=QLabel('Silero: waiting for microphone'); v.addWidget(self.vad_state_label)
        v.addStretch()

    def create_audio(self):
        w,v=self.make_widget(); self.tabs.addTab(w,'Audio')
        f=QFormLayout()
        self.microphone=QComboBox();self.microphone.currentIndexChanged.connect(self.changed)
        self.output=QComboBox();self.output.currentIndexChanged.connect(self.changed)
        self.wasapi_enabled=QCheckBox('WASAPI system audio (loopback)')
        self.wasapi_enabled.setToolTip('Off: record a microphone as usual. On: record audio playing through a Windows output device. WhisperBox echo playback can be captured too, so avoid enabling Echo with system audio.')
        self.wasapi_enabled.setChecked(bool(self.cfg.get('wasapi_enabled', False)))
        self.wasapi_enabled.toggled.connect(self.on_wasapi_changed)
        refresh=QPushButton('Refresh audio devices'); refresh.clicked.connect(self.refresh_devices)
        devices=QHBoxLayout()
        devices.addWidget(QLabel('Input Device:'))
        devices.addWidget(self.microphone, 1)
        devices.addWidget(QLabel('Output Device:'))
        devices.addWidget(self.output, 1)
        devices.addWidget(self.wasapi_enabled)
        f.addRow(devices)
        f.addRow(refresh)
        self.input_gain=self.number('input_gain',0,3,.05)
        self.output_gain=self.number('output_gain',0,2,.05)
        self.speed=self.number('playback_speed',.5,2,.05)
        self.numeric_row(f, 'Input gain:', self.input_gain)
        self.numeric_row(f, 'Output volume:', self.output_gain)
        self.numeric_row(f, 'Playback speed (changes pitch):', self.speed)
        self.gate_threshold=self.number('gate_threshold',0,.2,.01,2)
        self.gate_floor=self.number('gate_floor',0,1,.05)
        self.gate_attack=self.number('gate_attack',.01,1,.01,2)
        self.gate_release=self.number('gate_release',.01,2,.05,2)
        self.numeric_row(f, 'Noise gate threshold (0 = off):', self.gate_threshold)
        self.numeric_row(f, 'Closed gate volume (0 = silence):', self.gate_floor)
        self.numeric_row(f, 'Gate attack (seconds):', self.gate_attack)
        self.numeric_row(f, 'Gate release/falloff (seconds):', self.gate_release)
        v.addLayout(f)
        effects_group=QGroupBox('Input Audio Effects')
        ef=QFormLayout(effects_group)
        self.clean_input=self.flag('Enable input noise cleanup','input_cleanup')
        self.cleanup_input=self.number('input_cleanup_strength',0,2,.05)
        self.pitch_input=self.number('input_pitch',-12,12,1,0)
        self.formant_input=self.number('input_formant',.7,1.3,.05,2)
        ef.addRow(self.clean_input)
        self.numeric_row(ef, 'Cleanup strength:', self.cleanup_input)
        self.numeric_row(ef, 'Pitch shift (semitones):', self.pitch_input)
        self.numeric_row(ef, 'Formant shift (1.00 = neutral):', self.formant_input)
        v.addWidget(effects_group)
        v.addWidget(QLabel('Effects apply to completed input recordings and conversion previews; live VAD uses raw audio.'))
        v.addWidget(QLabel('The gate reduces sound below the threshold; it is not full noise cancellation.'))
        v.addWidget(QLabel('Device swaps take effect after any current recording/playback finishes.'))
        v.addStretch()

    def create_vad(self):
        w,v=self.make_widget();self.tabs.addTab(w,'Transcription')
        f=QFormLayout()
        self.threshold=self.number('vad_threshold',.05,.95,.05)
        self.start_frames=self.integer('start_frames',1,20)
        self.silence=self.number('end_silence',.1,5,.05)
        self.preroll=self.number('pre_roll',.05,2,.05)
        self.minimum=self.number('min_speech',0,2,.05)
        self.maximum=self.number('max_record',2,300,1,0)
        for lbl,obj in [('Threshold (higher = less sensitive)',self.threshold),('Speech start frames',self.start_frames),('End silence seconds',self.silence),('Pre-roll seconds',self.preroll),('Minimum speech seconds',self.minimum),('Max utterance seconds',self.maximum)]:self.numeric_row(f, lbl, obj)
        v.addLayout(f)
        whisper_group=QGroupBox('Whisper hallucination controls')
        wf=QFormLayout(whisper_group)
        self.trim_trailing=self.flag('Trim trailing VAD silence before transcription','trim_trailing_silence')
        self.trailing_margin=self.number('trailing_margin',0,1,.05)
        self.whisper_filter=self.flag('Enable Whisper low-confidence / no-speech filtering','whisper_filter')
        self.no_speech_threshold=self.number('no_speech_threshold',0,1,.05)
        self.log_prob_threshold=self.number('log_prob_threshold',-3,0,.1)
        self.compression_ratio_threshold=self.number('compression_ratio_threshold',1,5,.1)
        wf.addRow(self.trim_trailing)
        self.numeric_row(wf, 'Trailing speech margin (seconds):', self.trailing_margin)
        wf.addRow(self.whisper_filter)
        self.numeric_row(wf, 'No-speech probability threshold:', self.no_speech_threshold)
        self.numeric_row(wf, 'Log-probability threshold:', self.log_prob_threshold)
        self.numeric_row(wf, 'Compression ratio threshold:', self.compression_ratio_threshold)
        wf.addRow(QLabel('Whisper filtering is optional. Start with trimming enabled and filtering disabled.'))
        v.addWidget(whisper_group)
        v.addStretch()

    def create_settings(self):
        # Three compact groups inspired by the user-provided layout reference.
        # Keep the standard 900x850 window, with no Settings scroll container.
        content=QWidget()
        layout=QVBoxLayout(content)
        layout.setContentsMargins(10,8,10,8)
        layout.setSpacing(11)
        self.tabs.addTab(content,'Settings')

        performance=QGroupBox('Performance and processing')
        pf=QVBoxLayout(performance)
        pf.setContentsMargins(10,9,10,9)
        pf.setSpacing(5)
        performance_row=QHBoxLayout()
        performance_row.setSpacing(8)
        beam_tip=('Whisper decoding speed versus accuracy: beam 1 is fastest but may miss words; '
                  '3 is balanced (default); 5 explores more candidates and may be slower. '
                  'Changes apply to subsequent transcriptions.')
        beam_label=QLabel('Whisper beam size:')
        beam_label.setToolTip(beam_tip)
        beam_label.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Preferred)
        self.whisper_beam_size=self.integer('whisper_beam_size',1,5)
        self.whisper_beam_size.setFixedWidth(75)
        self.whisper_beam_size.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.whisper_beam_size.setToolTip(beam_tip)
        beam_slider=QSlider(Qt.Orientation.Horizontal)
        beam_slider.setRange(1,5)
        beam_slider.setValue(self.whisper_beam_size.value())
        beam_slider.setToolTip(beam_tip)
        beam_slider.setMinimumWidth(75)
        beam_slider.setSingleStep(1)
        beam_slider.setPageStep(1)
        beam_slider.valueChanged.connect(self.whisper_beam_size.setValue)
        self.whisper_beam_size.valueChanged.connect(beam_slider.setValue)
        beam_reset=QPushButton('↺')
        beam_reset.setToolTip('Reset Whisper beam size to the balanced default (3).')
        beam_reset.setFixedSize(30,28)
        beam_reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        beam_reset.clicked.connect(lambda: self.whisper_beam_size.setValue(3))
        performance_row.addWidget(beam_label)
        performance_row.addWidget(beam_slider,1)
        performance_row.addWidget(self.whisper_beam_size)
        performance_row.addWidget(beam_reset)

        self.voice_conditioning_cache=self.flag('Cache voice conditioning','voice_conditioning_cache')
        self.voice_conditioning_cache.setToolTip('Reuse reference-voice conditioning to avoid processing the same voice each time. Changing the voice or conditioning settings refreshes the cache. Turn off when troubleshooting voice quality.')
        self.chunked_tts=self.flag('Chunked TTS','chunked_tts')
        self.chunked_tts.setToolTip('On by default: generate speech in shorter, sentence-aware segments for more reliable long responses. Off: prefer single-pass speech for short text; long text is still split internally to avoid model limits. Either mode can produce a complete WAV.')
        self.streaming_playback=self.flag('Streaming playback','streaming_playback')
        self.streaming_playback.setToolTip('On by default: start playing each generated segment while the remaining text is synthesized. Requires Chunked TTS; unavailable while it is off. Turn off to wait for the complete combined WAV before playback.')
        self.chunked_tts.toggled.connect(self.update_streaming_playback_availability)
        self.update_streaming_playback_availability()
        for toggle in (self.voice_conditioning_cache,self.chunked_tts,self.streaming_playback):
            performance_row.addWidget(toggle)
        pf.addLayout(performance_row)
        layout.addWidget(performance)

        saving=QGroupBox('Saving and storage')
        save_v=QVBoxLayout(saving)
        save_v.setContentsMargins(10,7,10,9)
        save_v.setSpacing(7)
        save_info=QLabel('Leave a location blank to use the corresponding data folder.')
        save_v.addWidget(save_info)
        self.create_saving(save_v)
        save_footnote=QLabel('Transcript history stays in the Transcript Log tab; unsaved audio stays in memory.')
        save_v.addWidget(save_footnote)
        layout.addWidget(saving)

        tools=QGroupBox('Tools and maintenance')
        tools_v=QVBoxLayout(tools)
        tools_v.setContentsMargins(10,8,10,9)
        tools_v.setSpacing(6)
        self.create_polish_tab(tools_v)
        layout.addWidget(tools)
        layout.addStretch(1)

    def update_streaming_playback_availability(self, *args):
        # Chunking is required for early playback. Do not uncheck the control:
        # retain the user's saved preference and restore access when re-enabled.
        self.streaming_playback.setEnabled(self.chunked_tts.isChecked())

    def create_saving(self,v):
        self.save_input=self.flag('Save input microphone WAV files','save_input')
        self.save_text=self.flag('Save transcripts as TXT files','save_text')
        self.save_output=self.flag('Save cloned output WAV files','save_output')
        # Blank locations use the default folder; each input has its own placeholder.
        self.directory_fields = {}
        grid=QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(1,1)
        for row,(kind,check,title) in enumerate((('recordings',self.save_input,'Input recordings'),
                                                   ('transcripts',self.save_text,'Transcripts'),
                                                   ('generated_audio',self.save_output,'Generated audio'))):
            key,_=SAVE_LOCATIONS[kind]
            grid.addWidget(check,row,0)
            field=QLineEdit(str(self.cfg.get(key,'') or ''))
            field.setPlaceholderText(str(DATA/kind))
            field.setToolTip('Choose where these saved files go. Blank uses the default folder.')
            field.editingFinished.connect(self.changed)
            self.directory_fields[kind]=field
            grid.addWidget(field,row,1)
            browse=QPushButton('Browse...')
            browse.setToolTip('Select a folder for '+title.lower()+'.')
            browse.clicked.connect(lambda checked=False,k=kind:self.browse_save_directory(k))
            grid.addWidget(browse,row,2)
        v.addLayout(grid)
        # Saved-file cleanup only affects the selected file category.
        empty_input=QPushButton('EMPTY INPUT WAVS')
        empty_text=QPushButton('EMPTY TRANSCRIPTS')
        empty_output=QPushButton('EMPTY CLONED OUTPUT WAVS')
        empty_input.clicked.connect(lambda:self.empty_saved('recordings','.wav'))
        empty_text.clicked.connect(lambda:self.empty_saved('transcripts','.txt'))
        empty_output.clicked.connect(lambda:self.empty_saved('generated_audio','.wav'))
        v.addLayout(self.row(empty_input,empty_text,empty_output))

    def browse_save_directory(self, kind):
        current = saved_directory(self.cfg, kind)
        selected = QFileDialog.getExistingDirectory(self, 'Select save folder', str(current))
        if selected:
            self.directory_fields[kind].setText(selected)
            self.changed()

    def create_polish_tab(self,v):
        v.addWidget(QLabel('Settings presets — save or load a named configuration:'))
        self.preset_selector=QComboBox()
        self.preset_selector.setEditable(True)
        self.preset_selector.setPlaceholderText('Type a preset name or select a saved preset')
        v.addWidget(self.preset_selector)
        save=QPushButton('SAVE PRESET');save.clicked.connect(self.save_preset)
        load=QPushButton('LOAD PRESET');load.clicked.connect(self.load_preset)
        delete=QPushButton('DELETE PRESET');delete.clicked.connect(self.delete_preset)
        v.addLayout(self.row(save,load,delete))
        reset=QPushButton('RESET ALL SETTINGS TO DEFAULTS');reset.clicked.connect(self.reset_all)
        v.addWidget(reset)

        v.addWidget(QLabel('Startup checks — verify available hardware, runtimes and local model weights:'))
        validate=QPushButton('RUN STARTUP CHECKS')
        validate.clicked.connect(lambda:self.validate_startup(show_dialog=True))
        v.addWidget(validate)
        # Compact 4-column, 2-row results; each cell provides the full result on hover.
        self.startup_result_label=QLabel('Startup checks have not run yet.')
        self.startup_result_label.setToolTip('Summary of startup checks; detailed results appear in the Run Startup Checks dialog.')
        self.startup_result_label.hide()  # The eight results show status without an extra vertical line.
        self.startup_check_labels=[]
        results=QGridLayout()
        results.setContentsMargins(3,4,3,3)
        results.setHorizontalSpacing(8)
        results.setVerticalSpacing(3)
        for i in range(8):
            # Each column has one stable left edge for PASS/FAIL in both rows.
            # Separate labels keep status color independent from the description.
            cell=QWidget()
            pair=QHBoxLayout(cell)
            pair.setContentsMargins(0,0,0,0)
            pair.setSpacing(4)
            state_label=QLabel('—')
            state_label.setFixedWidth(33)
            state_label.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
            name_label=QLabel('Not checked')
            name_label.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
            name_label.setMinimumWidth(0)
            # Center an identical-width two-label group within each grid column.
            # Both rows share the same PASS/FAIL x-position, even when names differ.
            name_label.setFixedWidth(147)
            pair.addStretch(1)
            pair.addWidget(state_label)
            pair.addWidget(name_label)
            pair.addStretch(1)
            cell.setToolTip('Run startup checks to inspect this component.')
            results.addWidget(cell,i//4,i%4)
            self.startup_check_labels.append((cell,state_label,name_label))
            results.setColumnStretch(i%4,1)
        v.addLayout(results)
        export=QPushButton('EXPORT DIAGNOSTIC REPORT ZIP')
        export.setToolTip('Export diagnostic logs without private transcripts or audio files.')
        export.clicked.connect(self.export_diagnostics)
        v.addWidget(export)
        self.refresh_presets()
        # Begin checks only after the top-level window has been painted.
        # Calling blocking subprocess checks during initial show can prevent foreground activation.
        self._startup_checks_scheduled = True

    def preset_folder(self):
        d=ROOT/'config'/'presets';d.mkdir(parents=True,exist_ok=True);return d

    def refresh_presets(self):
        name=self.preset_selector.currentText()
        self.preset_selector.clear()
        self.preset_selector.addItems(sorted(p.stem for p in self.preset_folder().glob('*.json')))
        self.preset_selector.setEditText(name)

    def selected_preset(self):
        name=self.preset_selector.currentText().strip()
        if not name or not re.fullmatch(r'[A-Za-z0-9 _-]{1,48}',name):
            QMessageBox.warning(self,'Preset name','Use 1–48 letters, digits, spaces, hyphens or underscores.')
            return None
        return self.preset_folder()/(name+'.json')

    def save_preset(self):
        path=self.selected_preset()
        if path is None:return
        # Read every numeric spinbox, checkbox, device and voice selection from
        # the UI. Sliders are synchronized with their corresponding spinboxes.
        self.changed()
        values={k:self.cfg[k] for k in DEFAULTS}
        if path.is_file():
            if QMessageBox.question(self,'Replace preset',
                'Overwrite the existing preset "'+path.stem+'"?',
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return
        try:
            temporary=path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(values,indent=2),encoding='utf-8')
            temporary.replace(path)
            self.refresh_presets()
            self.status.setText('Preset saved: '+path.stem)
        except OSError as exc:
            QMessageBox.warning(self,'Preset error',str(exc))

    def load_preset(self):
        path=self.selected_preset()
        if path is None:return
        if not path.is_file():
            QMessageBox.warning(self,'Preset not found','Save this preset before loading it.')
            return
        try:
            settings=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(settings,dict):
                raise ValueError('Preset must contain a settings object.')
            self.apply_configuration({**DEFAULTS,**{k:v for k,v in settings.items() if k in DEFAULTS}})
            self.status.setText('Preset loaded: '+path.stem)
        except (OSError,ValueError,TypeError) as exc:
            QMessageBox.warning(self,'Preset error',str(exc))

    def delete_preset(self):
        path=self.selected_preset()
        if path is None:return
        if not path.is_file():
            QMessageBox.warning(self,'Preset not found','The selected preset does not exist.')
            return
        if QMessageBox.question(self,'Delete preset',
            'Permanently delete the preset "'+path.stem+'"?\n\nThis will not change your current settings.',
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return
        try:
            path.unlink()
            self.preset_selector.setEditText('')
            self.refresh_presets()
            self.status.setText('Preset deleted: '+path.stem)
        except OSError as exc:
            QMessageBox.warning(self,'Preset error',str(exc))

    def reset_all(self):
        if QMessageBox.question(self,'Reset all settings','Reset all WhisperBox settings to factory values?',
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)==QMessageBox.StandardButton.Yes:
            self.apply_configuration(DEFAULTS.copy())

    def apply_configuration(self, settings):
        # Apply every control as one transaction. Qt normally emits a change
        # signal for EACH setValue/setChecked; without this guard, intermediate
        # states overwrite settings that have not been restored yet.
        previous=self.cfg.copy()
        self._applying_configuration=True
        try:
            self.cfg.update({key: value for key, value in settings.items() if key not in ('manual', 'mode')})
            self.cfg.pop('manual', None)
            self.cfg.pop('mode', None)
            if self.cfg.get('compute_device') == 'cuda' and not self._cuda_ready:
                self.cfg['compute_device'] = 'cpu'
                diagnostic('cuda_selection_rejected', source='preset', reason=self._cuda_reason)
            self.compute.setCurrentText(str(self.cfg['compute_device']))
            self.model.setCurrentText(str(self.cfg['model']))
            for widget,key in ((self.echo,'echo_enabled'),
                (self.save_input,'save_input'),(self.save_text,'save_text'),
                (self.save_output,'save_output'),(self.trim_trailing,'trim_trailing_silence'),
                (self.whisper_filter,'whisper_filter'),(self.chunked_tts,'chunked_tts'),
                (self.voice_conditioning_cache,'voice_conditioning_cache'),(self.streaming_playback,'streaming_playback'),
                (self.turbo_norm_loudness,'turbo_norm_loudness'),
                (self.tray_setting,'tray_enabled'),(self.wasapi_enabled,'wasapi_enabled')):
                widget.setChecked(bool(self.cfg[key]))
            mapping={'speed':'playback_speed','threshold':'vad_threshold',
                     'silence':'end_silence','preroll':'pre_roll',
                     'minimum':'min_speech','maximum':'max_record','cfg_weight':'cfg'}
            for name,widget in list(vars(self).items()):
                key=mapping.get(name,name)
                if key in self.cfg and isinstance(widget,(QSpinBox,QDoubleSpinBox)):
                    widget.setValue(self.cfg[key])
            for kind,(key,_) in SAVE_LOCATIONS.items():
                self.directory_fields[kind].setText(self.cfg.get(key,'') or '')
            self.stop_conversion_preview()
            self.conversion_source_kind = 'file'
            self.conversion_source.setStyleSheet('')
            self.conversion_source.setText(self.cfg.get('conversion_audio',''))
            if self.conversion_source.text().strip() and Path(self.conversion_source.text().strip()).is_file():
                self.load_conversion_preview(self.conversion_source.text().strip())
            # Re-populate selectors using the target settings, not stale state.
            self.refresh_voices()
            self.refresh_devices()
        finally:
            self._applying_configuration=False
        # Compare with the actual previous configuration so model/device changes
        # trigger their normal restart logic exactly once.
        self.cfg=previous
        self.changed()

    def validate_startup(self, show_dialog=False):
        """Schedule checks off the GUI thread; never block the first paint."""
        worker = getattr(self, '_startup_validation_worker', None)
        if worker is not None and worker.isRunning():
            if show_dialog:
                self._startup_validation_dialog_pending = True
            return
        self._startup_validation_dialog_pending = bool(show_dialog)
        self.startup_result_label.setText('Checking startup requirements...')
        diagnostic('startup_validation_started')
        worker = StartupValidationWorker(self.cfg.get('compute_device', 'cpu'), self)
        self._startup_validation_worker = worker
        worker.completed.connect(self._complete_startup_validation)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _complete_startup_validation(self, notes, errors, cuda_ready, cuda_detail):
        show_dialog = bool(getattr(self, '_startup_validation_dialog_pending', False))
        self._startup_validation_dialog_pending = False
        self._startup_validation_worker = None
        self._cuda_live_checked = True
        self._cuda_ready = bool(cuda_ready)
        self._cuda_reason = cuda_detail
        fingerprint = cuda_runtime_fingerprint() if cuda_ready else None
        if cuda_ready and fingerprint:
            self.cfg['cuda_validation'] = {'verified': True, 'runtime_fingerprint': fingerprint}
            self._cuda_fingerprint = fingerprint
            self._cuda_cached = True
        else:
            self._cuda_ready = False
            self.cfg['cuda_validation'] = {}
            self._cuda_cached = False
            if cuda_ready:
                self._cuda_reason = 'Runtime files changed during CUDA validation; run checks again.'
        self._update_cuda_selector()
        # Restore a previously selected CUDA preference once reverified, and
        # choose CUDA automatically on the first successful installation check.
        use_cuda = (self._cuda_ready and
                    (self._saved_cuda_requested or self._cuda_auto_select_pending) and
                    self.cfg.get('compute_device') == 'cpu')
        self._saved_cuda_requested = False
        self._cuda_auto_select_pending = False
        if use_cuda:
            self.compute.setCurrentText('cuda')  # routes through changed() and saves config
        else:
            self.cfg['compute_device'] = self.compute.currentText()
        save_settings(self.cfg)
        diagnostic('cuda_capability', available=self._cuda_ready, cached=self._cuda_cached,
                   detail=self._cuda_reason[:700])
        passed=len(notes)
        failed=len(errors)
        summary=(f'Startup checks complete: {passed} passed, {failed} need attention.'
                 if failed else f'Startup checks passed: all {passed} checks are OK.')
        report=summary+'\n\n'+'\n'.join(('PASS  '+item for item in notes))
        if errors:
            report+='\n\n'+'\n'.join('NEEDS ATTENTION  '+item for item in errors)
        # Compact fixed grid; never introduce a nested result textbox or scrolling.
        labels=['Whisper runtime','Chatterbox runtime','Whisper weights','Original model',
                'Turbo model','SoundCard / WASAPI','Reference voices','GPU']
        for i,(display,(cell,state_label,name_label)) in enumerate(zip(labels,self.startup_check_labels)):
            relevant=[line for line in notes+errors if line.startswith(display+':')]
            if display=='SoundCard / WASAPI':
                relevant=[line for line in notes+errors if line.startswith('SoundCard / WASAPI:')]
            if display=='GPU':
                relevant=[line for line in notes+errors if line.startswith('GPU:')]
            detail=relevant[0] if relevant else 'Not checked'
            good=not any(detail == issue for issue in errors)
            state='PASS' if good else 'FAIL'
            accent='#43b481' if good else '#ef7474'
            state_label.setText(state)
            state_label.setStyleSheet(f'color: {accent}; font-weight: 600;')
            name_label.setText(display)
            name_label.setStyleSheet('')
            cell.setToolTip(detail)
        self.startup_result_label.setText(summary)
        self.status.setText(summary)
        diagnostic('startup_validation',passed=passed,issues=failed)
        if show_dialog:
            if failed:
                QMessageBox.warning(self,'Startup checks — attention needed',report)
            else:
                QMessageBox.information(self,'Startup checks passed',report)

    def export_diagnostics(self):
        out,_=QFileDialog.getSaveFileName(self,'Export diagnostic report',
            str(ROOT/'logs'/('WhisperBox_diagnostics_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')),'ZIP (*.zip)')
        if not out:return
        try:
            with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
                meta={'app_version':'1.0.0-rc3','platform':platform.platform(),'python':sys.version.split()[0],
                      'device':self.cfg['compute_device'],'model':self.cfg['model'],
                      'whisper_ready':self.model_ready,'tts_ready':self.tts_ready}
                z.writestr('system_summary.json',json.dumps(meta,indent=2))
                for filename in ('diagnostics.jsonl','diagnostics.jsonl.1','tts_diagnostics.jsonl',
                                 'tts_diagnostics.jsonl.1','tts_faults.log','chatterbox.log'):
                    f=ROOT/'logs'/filename
                    if f.is_file():z.write(f,arcname='logs/'+filename)
            self.status.setText('Diagnostics exported (audio and transcripts excluded).')
        except OSError as exc:QMessageBox.warning(self,'Export failed',str(exc))

    def cancel_generation(self):
        if not self.busy and self.pending_speech is None and self.pending_conversion is None:return
        diagnostic('generation_cancel_requested',operation=self.current_operation)
        self.pending_speech=None;self.pending_conversion=None
        if isinstance(self.playback_task,StreamPlayback):self.playback_task.halt()
        # Invalidate every in-flight request; late callbacks from the retired
        # model session must not overwrite the next session's audio or state.
        self._generation_epoch += 1
        self.cancel_button.setEnabled(False)
        # Chatterbox inference is not cooperatively cancellable with the upstream API.
        # Restart only Chatterbox, leaving Whisper, Silero and the GUI running.
        old=self.tts
        if old is not None and old.state()!=QProcess.ProcessState.NotRunning:
            # Old worker callbacks are identity-guarded and cannot affect its replacement.
            self.terminate_process(old,'cancel_generation')
            old.deleteLater()
        self.tts=None;self.tts_ready=False;self.tts_active_model=None
        self.finish_cycle()
        self.refresh_controls()
        self.status.setText('Generation canceled; Whisper remains loaded.')
        diagnostic('generation_cancel_completed')

    def create_logs(self):
        w,v=self.make_widget();self.tabs.addTab(w,'Transcript Log')
        self.entries=QListWidget();v.addWidget(self.entries)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);self.details.setMaximumHeight(135)
        self.entries.currentItemChanged.connect(lambda current,previous:self.details.setPlainText(current.data(Qt.ItemDataRole.UserRole) if current else ''))
        v.addWidget(self.details)
        copy=QPushButton('Copy selected text');copy.clicked.connect(self.copy_selected_transcript)
        clear=QPushButton('Clear displayed history');clear.clicked.connect(self.entries.clear)
        v.addLayout(self.row(copy,clear))
        v.addWidget(QLabel('Only transcripts appear here; technical status messages appear below every tab.'))

    def install_tray(self):
        self.tray=QSystemTrayIcon(QIcon(str(ROOT / 'assets' / 'WhisperBox.ico')),self)
        # Embed the tray-close option in the blank right-hand end of the tab bar.
        # There is deliberately no main-window menu bar.
        self.tray_setting=QCheckBox('Close window to system tray')
        self.tray_setting.setChecked(bool(self.cfg['tray_enabled']))
        self.tray_setting.toggled.connect(self.changed)
        tray_corner=QWidget(self.tabs)
        tray_corner_layout=QHBoxLayout(tray_corner)
        tray_corner_layout.setContentsMargins(8,0,8,0)
        tray_corner_layout.addWidget(self.tray_setting)
        self.tabs.setCornerWidget(tray_corner, Qt.Corner.TopRightCorner)
        # Keep Restore and Exit available from the actual system-tray icon.
        restore=QAction('Restore window',self);restore.triggered.connect(self.restore_window)
        quit_action=QAction('Exit and stop models',self);quit_action.triggered.connect(self.quit_app)
        from PySide6.QtWidgets import QMenu
        traymenu=QMenu()
        traymenu.addAction(restore);traymenu.addAction(quit_action)
        self.tray.setContextMenu(traymenu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.messageClicked.connect(self.restore_window)
        self.tray.show()

    def restore_window(self):
        """Restore and focus a hidden/minimized tray window on Windows."""
        diagnostic('gui_restore_requested', hidden=self.isHidden(), minimized=self.isMinimized())
        # showNormal() alone does not consistently surface a hidden Qt window
        # on Windows (it may remain behind another application's window).
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        # A brief topmost pulse helps Win32 foreground restrictions when
        # activation originates from the notification area.
        if sys.platform == 'win32':
            try:
                import ctypes
                hwnd = int(self.winId())
                user32 = ctypes.windll.user32
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)
                user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)
                user32.SetForegroundWindow(hwnd)
            except (OSError, ValueError, AttributeError):
                pass
        QTimer.singleShot(0, self.activateWindow)

    def on_tray_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick,
                      QSystemTrayIcon.ActivationReason.MiddleClick):
            self.restore_window()

    def install_help(self):
        """Practical hover descriptions for GUI parameters, checkboxes and actions."""
        tips = {
            'compute': 'CUDA uses a compatible NVIDIA GPU and is faster; CPU works without an NVIDIA GPU but may be much slower.',
            'model': 'Original is generally the higher-quality Chatterbox voice generator. Turbo is optimized for quicker speech generation.',
            'echo': 'When checked, Chatterbox reads your Whisper transcription aloud in the selected reference voice. Does not change typed Text-to-Speech or Voice Conversion.',
            'voice': 'Choose the voice you want Chatterbox to imitate. Add WAV, MP3 or OGG recordings under models/voices and open the voice dropdown to refresh. Generally a clean, 20 second voice sample.',
            'conversion_source': 'Recorded audio or the loaded file that you want to transform into the selected reference voice. Voice Conversion tries to keep the original words and timing; it does not use transcription.',
            'browse_conversion': 'Load a source recording to convert. WAV is the most reliable; MP3 and OGG depend on available audio decoders.',
            'convert_button': 'Convert the selected source speech into the selected reference voice using Chatterbox Voice Conversion. This is different from Echo, which first transcribes speech with Whisper.',
            'exaggeration': 'Higher values make emotional expression more dramatic; lower values usually sound calmer. The default 0.5 is a good starting point.',
            'cfg_weight': 'Adjusts how strongly speech generation follows its conditioning and pacing. At extreme values the voice can sound less natural; start at 0.5.',
            'original_temperature': 'Higher temperature makes generation more varied and less predictable; lower values are more consistent but may sound stiff.',
            'original_min_p': 'Filters out very unlikely next sound tokens. Higher values restrict variation more; lower values allow more possibilities.',
            'original_top_p': 'Limits token choices to a cumulative probability pool. 1.0 allows the widest pool; lower values make speech more conservative.',
            'original_repetition_penalty': 'Higher values discourage repeated sounds or fragments. Too high can produce odd pronunciations.',
            'original_seed': 'Set 0 to use a fresh random seed each generation; reuse a nonzero number to make results more repeatable, although exact output may still vary.',
            'turbo_temperature': 'Higher temperature allows more variation in how the model speaks. Lower values are more consistent.',
            'turbo_top_p': 'Reducing this limits token choices to the more likely options, making results more conservative.',
            'turbo_top_k': 'Maximum number of next-token candidates. Smaller values narrow choices; larger values allow more variety.',
            'turbo_repetition_penalty': 'Discourages the model from getting stuck repeating sounds, syllables or words. Excessively high values can distort speech.',
            'turbo_norm_loudness': 'Make the reference recording loudness more consistent before using it to clone the voice. Useful when voice examples are recorded at different volumes.',
            'turbo_seed': '0 chooses a random seed. A fixed nonzero seed helps repeat a similar generation with the same input and settings.',
            'global_button': 'Start Models loads Whisper/Silero into memory and starts microphone capture if unmuted. Stop Models shuts down inference workers and releases their model memory, including GPU VRAM.',
            'record_button': 'Mute or unmute automatic microphone listening. Speech is detected by Silero and transcribed by Whisper; Echo can optionally read recognized text. Use RECORD AUDIO to capture a source clip for voice conversion.',
            'typed': 'Type the exact words you want Chatterbox to speak in your selected voice. No microphone or transcription is needed.',
            'speak_button': 'Generate speech from the typed text using the currently selected Chatterbox model and reference voice.',
            'play_button': 'Replay the most recently generated or converted audio. It works even when that audio has only been kept in RAM.',
            'open_button': 'If the latest generated audio was saved to disk, reveal it in Windows Explorer. If it is only in RAM, this button offers to save it first.',
            'microphone': 'When WASAPI is off, choose a microphone. When WASAPI is on, choose the Windows playback device whose sound you want to transcribe. Changes take effect after an active recording ends.',
            'output': 'Choose the speakers or headphones for generated speech. WASAPI mode shows Windows WASAPI output devices only.',
            'input_gain': 'Adjust microphone signal strength before speech detection and transcription. 1.0 means unchanged; lower is quieter, higher amplifies both your voice and background noise.',
            'output_gain': 'Change generated speech loudness without changing your Windows master volume. 1.0 is normal; values over 1.0 can clip loud peaks.',
            'pitch_input': 'Shift pitch of recorded or loaded speech by -12 to +12 semitones without changing duration. Zero is neutral; reset restores zero. Large shifts can cause artifacts.',
            'formant_input': 'Change vocal resonance independently of pitch (0.70–1.30). Values below 1.00 sound deeper; above 1.00 sound brighter. Reset restores neutral 1.00.',
            'clean_input': 'Enable or disable input noise reduction (off by default). The Cleanup strength slider only takes effect while enabled. Applies to completed recordings and conversion preview, not live VAD.',
            'cleanup_input': 'Set noise-reduction strength when input cleanup is enabled. 0 = none, 1 = normal (reset default), 2 = strongest. High values may soften quiet speech.',
            'speed': 'Change playback speed. 1.0 is normal, 0.8 slower, 1.2 faster. This simple method ALSO changes pitch, so fast playback sounds higher.',
            'gate_threshold': 'Soft noise gate: audio quieter than this level is reduced. 0 turns the gate off. It helps between phrases but does not remove fan noise while you speak.',
            'gate_floor': 'How much sound remains while the noise gate is closed. 0 makes very quiet audio silent; 0.1 leaves 10% of its original level, avoiding abrupt cuts.',
            'gate_attack': 'How quickly the noise gate opens when you begin speaking. Shorter reacts faster; too short may sound abrupt.',
            'gate_release': 'How slowly the noise gate fades back down after speech becomes quiet. Longer keeps the ends of words from being chopped; too long leaves more background noise.',
            'threshold': 'How sure Silero must be that it hears speech. Higher = fewer false starts from noise, but it may miss quiet speech. Lower = easier detection, but more noise triggers.',
            'start_frames': 'How many consecutive tiny audio chunks must sound like speech before recording starts. Higher helps ignore clicks and bumps; lower reacts sooner to short words.',
            'silence': 'How long Silero waits in silence before deciding you finished talking. Increase if it cuts off pauses; decrease for quicker transcription after you finish.',
            'preroll': 'Keeps a short bit of audio from before Silero declares speech started. Increase it if the beginnings of words are missing.',
            'minimum': 'Ignore recordings containing less than this much detected speech. Raise it to reject little clicks and coughs; lower it for very short utterances.',
            'maximum': 'Safety limit for one automatic recording, in seconds. After this many seconds WhisperBox processes the captured audio even if you are still speaking.',
            'trim_trailing': 'Remove extra quiet audio after your last detected word before sending the recording to Whisper. This can reduce made-up text at the end of a transcript.',
            'trailing_margin': 'When trimming trailing silence, keep this much audio after the final detected speech. Increase if endings of words are being cut.',
            'whisper_filter': 'Apply Whisper low-confidence and silence checks to reject suspect segments. It may reduce invented phrases, but can also discard real quiet speech.',
            'no_speech_threshold': 'How likely Whisper must consider a segment to contain no speech before silence filtering may reject it. Tune only if it misses speech or invents text during silence.',
            'log_prob_threshold': 'A threshold for unusually low-confidence transcription. Raising it can flag more uncertain words, but may also reject legitimate speech.',
            'compression_ratio_threshold': 'Flag unusually repetitive text, such as Whisper repeating a phrase. Lower catches more repetition but risks rejecting legitimate repeated words.',
            'save_input': 'When checked, save microphone input WAV recordings to your configured input recordings folder. Unchecked keeps unsaved input audio in memory only.',
            'save_text': 'Save each finished Whisper transcription as a TXT file in your configured transcripts folder. Transcripts still appear in the GUI log if this is unchecked.',
            'save_output': 'Save each generated TTS or Voice Conversion WAV in your configured generated audio folder. When unchecked, only the latest output is kept in RAM for Play Last.',
            'tray_setting': 'Checked: clicking X hides WhisperBox in the system tray and keeps any loaded models running. Unchecked: X exits the program and stops the models to release RAM/VRAM.',
            'entries': 'Select a transcription here to read its full text below. These entries are a convenient in-app history, not necessarily saved files.',
            'details': 'Full text of the selected transcript. Use Copy Selected Text to place it on your clipboard.',
        }
        for name, explanation in tips.items():
            widget = getattr(self, name, None)
            if isinstance(widget, QWidget):
                widget.setToolTip(explanation)
                widget.setToolTipDuration(20000)

        # Numeric rows use a slider and spinbox. Apply the same explanation
        # to both, including their containing row widget.
        for spin, slider in self.numeric_sliders.items():
            if spin.toolTip():
                slider.setToolTip(spin.toolTip())
                slider.setToolTipDuration(20000)
                # Reset buttons have their own default-value explanation.

        # Include the QFormLayout labels for model parameters, whose short
        # captions (for example Temperature) occur in both model panels.
        for form in self.findChildren(QFormLayout):
            for name, hint in tips.items():
                control = getattr(self, name, None)
                if isinstance(control, QWidget):
                    caption = form.labelForField(control.parentWidget() if control in self.numeric_sliders else control)
                    if isinstance(caption, QWidget):
                        caption.setToolTip(hint)
                        caption.setToolTipDuration(20000)

        # Text-button explanations, including locally created controls.
        button_tips = {
            'Refresh audio devices': 'Look again for available microphones and speakers, including newly plugged-in USB devices.',
            'Browse...': tips['browse_conversion'],
            'CONVERT VOICE': tips['convert_button'],
            'EMPTY INPUT WAVS': 'Permanently delete saved input WAV files from your configured input recordings folder. A confirmation appears first. This does not delete reference voices.',
            'EMPTY TRANSCRIPTS': 'Permanently delete saved transcript TXT files from data/transcripts after confirmation. This does not automatically clear the currently displayed history.',
            'EMPTY CLONED OUTPUT WAVS': 'Permanently delete saved generated WAV files from data/generated_audio after confirmation. This does not affect model files or voice references.',
            'Copy selected text': 'Copy the full selected transcript (not just the short row title) to the Windows clipboard.',
            'Clear displayed history': 'Remove transcript entries from the current GUI display. Does not delete saved TXT files from your disk.',
            'START MODELS': tips['global_button'],
            'STOP MODELS / FREE VRAM': tips['global_button'],
            'MUTE MICROPHONE': tips['record_button'],
            'UNMUTE MICROPHONE': tips['record_button'],
            'GENERATE SPEECH': tips['speak_button'],
            'PLAY LAST': tips['play_button'],
            'OPEN FILE LOCATION': tips['open_button'],
            'SAVE LAST GENERATION': tips['open_button'],
        }
        for button in self.findChildren(QPushButton):
            hint = button_tips.get(button.text())
            if hint:
                button.setToolTip(hint)
                button.setToolTipDuration(20000)

        # Make the descriptive labels hoverable, too, not just their inputs.
        label_tips = {
            'Inference device': tips['compute'], 'Chatterbox model': tips['model'],
            'Reference audio': tips['voice'], 'Conversion audio': tips['conversion_source'],
            'Input Device': tips['microphone'], 'Output Device': tips['output'],
            'Input gain': tips['input_gain'], 'Playback volume': tips['output_gain'],
            'Playback speed': tips['speed'], 'Noise gate threshold': tips['gate_threshold'],
            'Closed gate volume': tips['gate_floor'], 'Gate attack': tips['gate_attack'],
            'Gate release/falloff': tips['gate_release'], 'Threshold (higher': tips['threshold'],
            'Speech start frames': tips['start_frames'], 'End silence seconds': tips['silence'],
            'Pre-roll seconds': tips['preroll'], 'Minimum speech seconds': tips['minimum'],
            'Max utterance seconds': tips['maximum'], 'Trailing speech margin': tips['trailing_margin'],
            'No-speech probability threshold': tips['no_speech_threshold'],
            'Log-probability threshold': tips['log_prob_threshold'],
            'Compression ratio threshold': tips['compression_ratio_threshold'],
            'Cleanup strength': tips['cleanup_input'],
        }
        for label in self.findChildren(QLabel):
            caption = label.text().strip().rstrip(':')
            for prefix, hint in label_tips.items():
                if caption.startswith(prefix):
                    label.setToolTip(hint)
                    label.setToolTipDuration(20000)
                    break
        for index, hint in enumerate((
            'Microphone recording, voice cloning, and model start/stop controls.',
            'Choose recording and playback devices, gain, playback speed, and noise-gate settings.',
            'Control speech detection and optional Whisper hallucination protections.',
            'View and copy recognized speech without technical debug messages.',
            'Performance settings, save locations, presets, startup checks, and diagnostic tools.',
        )):
            self.tabs.setTabToolTip(index, hint)

    def flush_settings(self):
        """Write one snapshot and send the latest settings to the listener."""
        if not self._settings_dirty:
            return
        self._settings_dirty = False
        save_settings(self.cfg)
        if self.whisper and self.whisper.state() == QProcess.ProcessState.Running:
            self.command({'action': 'config', 'settings': self.cfg})

    def changed(self,*args):
        if getattr(self,'_applying_configuration',False):
            return
        if not hasattr(self,'tray_setting') or not hasattr(self,'microphone') or not hasattr(self,'threshold') or not hasattr(self,'voice'):
            return
        pairs=dict(compute_device=self.compute.currentText(),echo_enabled=self.echo.isChecked(),model=self.model.currentText(),
            voice=self.voice.currentData() or '',conversion_audio=(self.conversion_source.text().strip() if self.conversion_source_kind == 'file' else ''),mic_device=self.microphone.currentData() or 'default',
            output_device=self.output.currentData() or 'default',wasapi_enabled=self.wasapi_enabled.isChecked(),input_gain=self.input_gain.value(),
            output_gain=self.output_gain.value(),input_pitch=self.pitch_input.value(),input_formant=self.formant_input.value(),
            input_cleanup=self.clean_input.isChecked(),input_cleanup_strength=self.cleanup_input.value(),playback_speed=self.speed.value(),
            vad_threshold=self.threshold.value(),start_frames=self.start_frames.value(),
            end_silence=self.silence.value(),pre_roll=self.preroll.value(),
            min_speech=self.minimum.value(),max_record=self.maximum.value(),
            trim_trailing_silence=self.trim_trailing.isChecked(),
            trailing_margin=self.trailing_margin.value(),
            whisper_filter=self.whisper_filter.isChecked(),whisper_beam_size=self.whisper_beam_size.value(),chunked_tts=self.chunked_tts.isChecked(),
            voice_conditioning_cache=self.voice_conditioning_cache.isChecked(),streaming_playback=self.streaming_playback.isChecked(),
            no_speech_threshold=self.no_speech_threshold.value(),
            log_prob_threshold=self.log_prob_threshold.value(),
            compression_ratio_threshold=self.compression_ratio_threshold.value(),
            gate_threshold=self.gate_threshold.value(),gate_floor=self.gate_floor.value(),
            gate_attack=self.gate_attack.value(),gate_release=self.gate_release.value(),
            exaggeration=self.exaggeration.value(),cfg=self.cfg_weight.value(),
            original_temperature=self.original_temperature.value(),original_min_p=self.original_min_p.value(),
            original_top_p=self.original_top_p.value(),original_repetition_penalty=self.original_repetition_penalty.value(),
            original_seed=self.original_seed.value(),turbo_temperature=self.turbo_temperature.value(),
            turbo_top_p=self.turbo_top_p.value(),turbo_top_k=self.turbo_top_k.value(),
            turbo_repetition_penalty=self.turbo_repetition_penalty.value(),
            turbo_norm_loudness=self.turbo_norm_loudness.isChecked(),turbo_seed=self.turbo_seed.value(),
            save_input=self.save_input.isChecked(),save_text=self.save_text.isChecked(),
            save_output=self.save_output.isChecked(),tray_enabled=self.tray_setting.isChecked(),
            recordings_dir=self.directory_fields['recordings'].text().strip(),
            transcripts_dir=self.directory_fields['transcripts'].text().strip(),
            generated_audio_dir=self.directory_fields['generated_audio'].text().strip())
        previous=self.cfg.copy()
        if pairs['compute_device'] == 'cuda' and not self._cuda_ready:
            pairs['compute_device'] = 'cpu'
            self.compute.blockSignals(True)
            self.compute.setCurrentText('cpu')
            self.compute.blockSignals(False)
            self.status.setText('CUDA unavailable: ' + self._cuda_reason)
            diagnostic('cuda_selection_rejected', source='ui', reason=self._cuda_reason)
        settings_changed = any(previous.get(key) != value for key, value in pairs.items())
        if settings_changed:
            self.cfg.update(pairs)
            self._settings_dirty = True
            self._settings_flush_timer.start()
        self.refresh_controls()
        self.model_settings.setCurrentIndex(0 if self.cfg['model']=='original' else 1)
        # Switching CPU/CUDA needs both workers restarted: Whisper and Chatterbox.
        # Defer while microphone capture, speech generation or playback is active.
        if self.running and previous.get('compute_device') != self.cfg['compute_device']:
            self.device_restart_pending = True
            self.switch_requested = False
            self.status.setText('Inference device change queued until idle...')
            QTimer.singleShot(0, self.apply_device_restart)
            return
        if self.device_restart_pending:
            return
        # Original/Turbo changes restart just Chatterbox, not Whisper.
        if self.running and previous.get('model') != self.cfg['model']:
            self.switch_requested = True
            self.status.setText('Switching Chatterbox to ' + self.cfg['model'] + ' when idle...')
            QTimer.singleShot(0, self.apply_tts_switch)
        if settings_changed and previous.get('echo_enabled') != self.cfg['echo_enabled']:
            diagnostic('echo_state_changed', enabled=self.cfg['echo_enabled'],
                       whisper_ready=self.model_ready,
                       listener_initialized=self._listener_initialized,
                       capture_ready=self._listener_capture_ready,
                       busy=self.busy, operation=self.current_operation)
        if self.running and self.cfg['echo_enabled'] and not previous.get('echo_enabled'):
            self.ensure_tts()
        # Disabling Echo only stops automatic playback. Keep Chatterbox loaded
        # for typed speech and voice conversion; explicit Stop Models frees VRAM.

    def stop_conversion_preview(self):
        player = self.conversion_preview
        if player is not None:
            # The DSP worker may still be compiling/processing when asked to stop.
            # Never destroy a running QThread: retain it until finished.
            player.stop_requested.set()
            player.pause_requested.clear()
            if player.isRunning() and not player.wait(350):
                if player not in self._retiring_previews:
                    self._retiring_previews.append(player)
                    player.finished.connect(lambda p=player: self._release_retired_preview(p))
            self.conversion_preview = None
        self.conversion_play.setText('PLAY')
        self._conversion_seek_position = 0.0
        self.conversion_wave.set_progress(0)

    def _release_retired_preview(self, player):
        if player in self._retiring_previews:
            self._retiring_previews.remove(player)

    def conversion_source_clicked(self, event):
        # Clicking the field restores the latest recording after browsing a file.
        if self.conversion_recording_wav is not None and self.conversion_source_kind == 'file':
            self.use_current_recording()
        QLineEdit.mousePressEvent(self.conversion_source, event)

    def use_current_recording(self):
        if not self.conversion_recording_wav or self.conversion_recording or self.busy:
            return
        self.stop_conversion_preview()
        self.conversion_source_kind = 'recording'
        self.conversion_source.setText('Using current recording...')
        self.conversion_source.setStyleSheet('color: #89949f; font-style: italic;')
        self.load_conversion_preview_bytes(self.conversion_recording_wav)
        self.changed()

    def load_conversion_preview_bytes(self, wav):
        self._conversion_seek_position = 0.0
        try:
            audio, rate = sf.read(io.BytesIO(wav), dtype='float32', always_2d=False)
            self.conversion_audio_data = audio
            self.conversion_audio_rate = rate
            self.conversion_wave.set_audio(audio)
            self.refresh_controls()
        except Exception as exc:
            self.conversion_audio_data = None
            self.conversion_audio_rate = None
            self.conversion_wave.set_audio([])
            self.refresh_controls()
            self.status.setText('Audio preview unavailable: ' + str(exc))

    def load_conversion_preview(self, filename):
        try:
            with open(filename, 'rb') as handle:
                data = handle.read()
            self.load_conversion_preview_bytes(data)
        except (OSError, ValueError) as exc:
            self.status.setText('Audio preview unavailable: ' + str(exc))

    def current_effects(self):
        return (self.pitch_input.value(), self.formant_input.value(), self.cleanup_input.value() if self.clean_input.isChecked() else 0.)

    def seek_conversion_preview(self, fraction):
        if self.conversion_audio_data is None:
            return
        self._conversion_seek_position = max(0.0, min(1.0, float(fraction)))
        player = self.conversion_preview
        if player is not None and player.isRunning():
            player.seek(self._conversion_seek_position)

    def toggle_conversion_preview(self):
        if self.recording_active or self.conversion_recording or self.conversion_finalizing or self.busy:
            self.status.setText('Finish current recording or processing before preview playback')
            return
        if self.conversion_audio_data is None:
            return
        if self.conversion_preview and self.conversion_preview.isRunning():
            paused = not self.conversion_preview.pause_requested.is_set()
            if paused: self.conversion_preview.pause_requested.set()
            else: self.conversion_preview.pause_requested.clear()
            self.conversion_play.setText('PLAY' if paused else 'PAUSE')
            return
        if self.playback_task is not None and self.playback_task.isRunning():
            self.status.setText('Finish existing playback before previewing input audio')
            return
        position = getattr(self, '_conversion_seek_position', 0.0)
        self.conversion_wave.set_progress(position)
        player = ConversionPreview(self.conversion_audio_data, self.conversion_audio_rate, self.cfg.get('output_device','default'), self.current_effects())
        if position:
            player.seek(position)
        self.conversion_preview = player
        player.progressed.connect(lambda progress, p=player: self.conversion_wave.set_progress(progress) if self.conversion_preview is p else None)
        player.failed.connect(lambda message: self.status.setText('Preview playback: ' + message))
        player.stopped.connect(lambda p=player: self.conversion_preview_finished(p))
        self.conversion_play.setText('PAUSE')
        player.start()
        self.refresh_controls()

    def conversion_preview_finished(self, player):
        if self.conversion_preview is not player:
            return
        self._conversion_seek_position = 0.0
        self.conversion_wave.set_progress(0)
        self.conversion_play.setText('PLAY')
        self.refresh_controls()

    def toggle_conversion_record(self):
        if self.conversion_recording:
            if self.conversion_finalizing:
                return
            self.conversion_finalizing = True
            self.command({'action':'conversion_record_stop', 'token':self._conversion_token})
            self._conversion_record_timer.start()
            self.refresh_controls()
            self.status.setText('Finalizing conversion recording...')
            return
        if (not self.running or not self.model_ready or self.busy or self.recording_active or
                self.muted or self.device_restart_pending or self.device_restart_in_progress or
                self.switch_requested or self.shutting_down or self.conversion_finalizing):
            self.status.setText('Microphone must be ready and idle to record conversion audio')
            return
        self.stop_conversion_preview()
        self.conversion_recording = True
        self._conversion_serial += 1
        self._conversion_token = self._conversion_serial
        self.conversion_record_button.setText('STOP RECORDING')
        # The Whisper listener switches to capture-only: no VAD, transcription or echo.
        self.command({'action':'conversion_record_start', 'token':self._conversion_token})
        self.refresh_controls()
        self.status.setText('Recording conversion source (inference paused)...')

    def receive_conversion_recording(self, audio_b64, token):
        if token is None or token != self._conversion_token or not self.conversion_recording:
            diagnostic('ignored_late_conversion_recording', token=token)
            return
        self._conversion_token = None
        self._conversion_record_timer.stop()
        self.conversion_finalizing = False
        self.conversion_recording = False
        self._conversion_token = None
        self.conversion_record_button.setText('RECORD AUDIO')
        self.refresh_controls()
        if not audio_b64:
            self.status.setText('No conversion audio was recorded')
            self.changed()
            return
        try:
            wav = base64.b64decode(audio_b64, validate=True)
            if len(wav) > 120_000_000:
                raise ValueError('Recording exceeds 120 MB limit')
            self.conversion_recording_wav = wav
            self.use_current_recording()
            self.status.setText('Conversion recording ready in memory')
        except (OSError, ValueError) as exc:
            self.status.setText('Could not prepare conversion recording: ' + str(exc))
        finally:
            self.refresh_controls()

    def pick_conversion_audio(self):
        if self.conversion_recording or self.busy:
            self.status.setText('Finish recording or conversion before loading another audio file')
            return
        source, _ = QFileDialog.getOpenFileName(
            self, 'Choose speech recording for voice conversion',
            str(Path(self.cfg.get('conversion_audio','')).parent) if self.cfg.get('conversion_audio') else str(ROOT),
            'Audio files (*.wav *.mp3 *.ogg);;All files (*)')
        if source:
            self.stop_conversion_preview()
            self.conversion_source_kind = 'file'
            self.conversion_source.setStyleSheet('')
            self.conversion_source.setText(source)
            self.load_conversion_preview(source)
            self.changed()

    def convert_voice(self):
        if not self.running or self.busy or self.pending_speech is not None or self.pending_conversion is not None or self.recording_active or self.device_restart_pending or self.switch_requested:
            self.status.setText('Start models and finish current processing before voice conversion')
            return
        if self.conversion_recording or self.conversion_finalizing:
            self.status.setText('Stop conversion recording before converting')
            return
        in_memory = self.conversion_source_kind == 'recording' and self.conversion_recording_wav is not None
        source = Path(self.conversion_source.text().strip()) if not in_memory else None
        target = Path(self.cfg.get('voice',''))
        if not in_memory and (not source.is_file() or source.suffix.lower() not in AUDIO_EXTENSIONS):
            QMessageBox.warning(self,'Source audio required','Choose an existing WAV, MP3 or OGG speech recording.')
            return
        if not target.is_file():
            QMessageBox.warning(self,'Reference audio required','Select a valid target voice in Reference audio.')
            return
        self.stop_conversion_preview()
        self.busy=True
        self.command({'action':'hold'})
        self.refresh_controls()
        if not self.tts_ready:
            self.pending_conversion = (self.conversion_recording_wav if in_memory else str(source), str(target))
            if not self.ensure_tts():
                self.pending_conversion = None
                self.finish_cycle()
            else:
                self.status.setText('Loading Chatterbox before voice conversion...')
            return
        self.start_conversion(self.conversion_recording_wav if in_memory else str(source),str(target))

    def start_conversion(self, source, target):
        self.status.setText('Converting voice (may load VC weights on first use)...')
        self.current_operation='voice_conversion'
        diagnostic('generation_requested', kind='vc', model=self.cfg['model'], source_extension=('.wav' if isinstance(source, bytes) else Path(source).suffix.lower()), gpu=gpu_snapshot())
        self.refresh_controls()
        task = Conversion(source,target,self.current_effects())
        task._generation_epoch = self._generation_epoch
        self.workers.append(task)
        task.ready.connect(self.conversion_ready)
        task.failed.connect(self.speech_error)
        task.finished.connect(lambda: self.workers.remove(task) if task in self.workers else None)
        task.start()

    def conversion_ready(self, wav, source):
        if self._stale_generation_result():
            return
        self.speech_ready(wav, 'Voice conversion: '+('current recording' if source == '<recording>' else Path(source).name))

    def update_last_output_button(self):
        saved = self.most_recent_wav is not None and self.most_recent_wav.is_file()
        self.open_button.setText('OPEN FILE LOCATION' if saved else 'SAVE LAST GENERATION')
        self.open_button.setEnabled(self.audio_last is not None)

    def refresh_voices(self):
        desired=self.cfg.get('voice','')
        self.voice.blockSignals(True);self.voice.clear()
        for p in voice_files():
            self.voice.addItem(str(p.relative_to(VOICES)),str(p))
        idx=self.voice.findData(desired)
        if idx>=0:self.voice.setCurrentIndex(idx)
        elif not desired and self.voice.count():self.voice.setCurrentIndex(0)
        self.voice.blockSignals(False)
        self.changed()

    def on_wasapi_changed(self, checked):
        if getattr(self, '_applying_configuration', False):
            return
        # Retain the current selections whenever possible. An invalid selection
        # is replaced with an actual available device, rather than a stale ID.
        self.refresh_devices()

    def refresh_devices(self):
        """Show only usable capture/playback devices for the selected mode."""
        loopback = bool(self.wasapi_enabled.isChecked())
        desired_input = str(self.cfg.get('mic_device', 'default'))
        desired_output = str(self.cfg.get('output_device', 'default'))
        if loopback:
            inputs = []
            discovery_errors = []
            # Enumerate actual loopback microphones; a playback speaker lookup
            # can fail even when its loopback endpoint exists.
            def enumerate_loopbacks():
                import soundcard as sc
                devices = sc.all_microphones(include_loopback=True)
                loopbacks = [d for d in devices if getattr(d, 'isloopback', False)]
                default = sc.default_speaker()
                return [(str(d.name), str(d.id)) for d in loopbacks], (str(default.id) if default else '')
            try:
                candidates, default_id = enumerate_loopbacks()
                diagnostic('wasapi_discovery_gui', count=len(candidates), default_id=bool(default_id))
            except Exception as exc:
                candidates, default_id = [], ''
                discovery_errors.append('GUI: ' + str(exc))
                diagnostic('wasapi_discovery_gui_error', error=repr(exc))
            # Existing installations may have SoundCard only in Whisper's
            # isolated environment. Query that same environment instead of
            # falsely claiming that Windows has no audio devices.
            if not candidates and WHISPER_PY.is_file():
                try:
                    probe = subprocess.run(
                        [str(WHISPER_PY), '-u', str(APP/'audio_devices.py')],
                        capture_output=True, text=True, timeout=12,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    payload = json.loads(probe.stdout.strip().splitlines()[-1])
                    if payload.get('error'):
                        discovery_errors.append('Whisper: ' + str(payload['error']))
                    candidates = [(str(x['name']), str(x['id'])) for x in payload.get('loopbacks', [])]
                    default_id = str(payload.get('default_id') or '')
                    diagnostic('wasapi_discovery_whisper', count=len(candidates), error=payload.get('error', ''))
                except Exception as exc:
                    discovery_errors.append('Whisper: ' + str(exc))
                    diagnostic('wasapi_discovery_whisper_error', error=repr(exc))
            seen = set()
            for name, identifier in candidates:
                if identifier and identifier not in seen:
                    seen.add(identifier)
                    inputs.append((name + ' (loopback)', 'loopback:' + identifier))
            if inputs and default_id and default_id in seen:
                inputs.insert(0, ('System default output (loopback)', 'loopback:default'))
            if not inputs:
                reason = '; '.join(discovery_errors) if discovery_errors else 'No loopback endpoints returned by Windows'
                self.status.setText('WASAPI unavailable: ' + reason)
                diagnostic('wasapi_no_devices', reason=reason)
        else:
            inputs = [('Windows default microphone', 'default')]
        outputs = [] if loopback else [('Windows default output', 'default')]
        try:
            hosts = sd.query_hostapis()
            for i, dev in enumerate(sd.query_devices()):
                host = hosts[int(dev['hostapi'])]['name']
                is_wasapi = 'WASAPI' in host.upper()
                if loopback and not is_wasapi:
                    continue
                if not loopback and 'WASAPI' in host.upper():
                    # Normal mode keeps a single predictable Windows device
                    # list rather than duplicate WASAPI endpoints.
                    continue
                if not loopback and int(dev['max_input_channels']) > 0:
                    inputs.append((f'{dev["name"]} ({host})', str(i)))
                if int(dev['max_output_channels']) > 0:
                    outputs.append((f'{dev["name"]} ({host})', str(i)))
        except Exception as exc:
            self.status.setText('Device discovery: ' + str(exc))
        for combo, items, desired in ((self.microphone, inputs, desired_input),
                                      (self.output, outputs, desired_output)):
            combo.blockSignals(True)
            combo.clear()
            for label, value in items:
                combo.addItem(label, value)
            selected = combo.findData(desired)
            combo.setCurrentIndex(selected if selected >= 0 else (0 if items else -1))
            combo.setEnabled(bool(items))
            combo.blockSignals(False)
        # Preserve the detailed discovery failure shown above rather than overwriting it.
        self.changed()

    def command(self,obj):
        if self.whisper and self.whisper.state()==QProcess.ProcessState.Running:
            self.whisper.write((json.dumps(obj)+'\n').encode('utf-8'))

    def make_process(self,python,filename,args=None):
        p=QProcess(self)
        p.setProgram(str(python))
        p.setArguments(['-u', str(APP/filename)] + (args or []))
        p.setWorkingDirectory(str(ROOT))
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env=QProcessEnvironment.systemEnvironment()
        # Conda prefix environments require their own DLL search paths even
        # when python.exe is invoked directly (without conda activate).
        runtime=Path(python).parent
        if runtime.name.lower()=='scripts':
            runtime=runtime.parent
        dll_dirs=[runtime, runtime/'Library'/'bin', runtime/'DLLs', runtime/'Scripts']
        inherited=env.value('PATH','')
        env.insert('PATH', os.pathsep.join(str(folder) for folder in dll_dirs if folder.is_dir()) + os.pathsep + inherited)
        env.insert('CONDA_PREFIX',str(runtime))
        env.insert('PYTHONUNBUFFERED', '1')
        env.insert('HF_HOME', str(ROOT / 'models' / 'cache' / 'huggingface'))
        env.insert('XDG_CACHE_HOME', str(ROOT / 'models' / 'cache'))
        env.insert('WHISPERBOX_ROOT', str(ROOT))
        # No model-repository traffic during normal inference.
        env.insert('HF_HUB_OFFLINE', '1')
        env.insert('TRANSFORMERS_OFFLINE', '1')
        env.insert('HF_HUB_DISABLE_TELEMETRY', '1')
        p.setProcessEnvironment(env)
        return p

    def attach_process_diagnostics(self, process, label):
        pid_key = id(process)
        self.process_intents[pid_key] = None
        self.process_started[pid_key] = time.monotonic()
        process.started.connect(lambda p=process, l=label: diagnostic(
            'process_started', service=l, pid=int(p.processId()),
            program=p.program(), arguments=p.arguments(), gpu=gpu_snapshot()))
        process.errorOccurred.connect(lambda error, p=process, l=label: diagnostic(
            'process_error', service=l, pid=int(p.processId()),
            qt_error=str(error), qt_error_string=p.errorString(),
            intended_reason=self.process_intents.get(id(p)),
            operation=self.current_operation))
        process.finished.connect(lambda code, status, p=process, l=label: self.log_process_exit(p, l, code, status))

    def log_process_exit(self, process, label, code, status):
        key = id(process)
        reason = self.process_intents.pop(key, None)
        started = self.process_started.pop(key, None)
        expected = bool(reason)
        diagnostic('process_exit', service=label, pid=int(process.processId()),
                   exit_code=code, exit_hex=format_exit_code(code),
                   qt_exit_status=str(status), intentional=expected,
                   reason=reason, operation=self.current_operation,
                   model=self.cfg['model'], device=self.cfg['compute_device'],
                   elapsed_seconds=round(time.monotonic()-started, 2) if started else None,
                   gpu=gpu_snapshot())

    def terminate_process(self, process, reason):
        if process is None:
            return
        self.process_intents[id(process)] = reason
        if process is self.tts:
            self.last_tts_intent = reason
        diagnostic('process_stop_requested', reason=reason,
                   service='Chatterbox' if process is self.tts else 'Whisper',
                   pid=int(process.processId()), operation=self.current_operation)
        if process.state() == QProcess.ProcessState.NotRunning:
            # A finished process may have already delivered its exit signal.
            # Do not retain dead-process IDs in the diagnostics maps.
            self.process_intents.pop(id(process), None)
            self.process_started.pop(id(process), None)
            return
        if process is self.whisper:
            self.command({'action': 'quit'})
            # QProcess writes are buffered. Flush the quit command before
            # waiting; otherwise the worker may never receive it during this
            # synchronous wait, freezing the GUI until force-kill.
            process.waitForBytesWritten(350)
            if process.waitForFinished(1250):
                diagnostic('process_graceful_exit', service='Whisper', reason=reason)
                return
        elif process is self.tts:
            try:
                req = urllib.request.Request('http://127.0.0.1:18765/shutdown',
                                             data=b'{}', method='POST',
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=1.5) as response:
                    response.read()
                diagnostic('process_graceful_shutdown_requested', service='Chatterbox', reason=reason)
                if process.waitForFinished(6000):
                    diagnostic('process_graceful_exit', service='Chatterbox', reason=reason)
                    return
            except Exception as exc:
                diagnostic('process_graceful_shutdown_failed', service='Chatterbox',
                           reason=reason, error_type=type(exc).__name__, error=str(exc)[:300])
        diagnostic('process_terminate_fallback', reason=reason, pid=int(process.processId()))
        process.terminate()
        # Bound GUI stalls when a native audio backend ignores termination.
        timeout = 850 if process is self.whisper else 3000
        if not process.waitForFinished(timeout):
            diagnostic('process_force_kill', reason=reason, pid=int(process.processId()))
            process.kill()
            process.waitForFinished(900)

    def toggle_models(self):
        if self.running:
            self.stop_models()
        else:
            self.start_models()

    def start_models(self):
        if not WHISPER_PY.exists():
            QMessageBox.critical(self, 'Missing WhisperBox runtime', 'Whisper: ' + str(WHISPER_PY))
            return
        if self.cfg['echo_enabled'] and (not self.cfg['voice'] or not Path(self.cfg['voice']).is_file()):
            QMessageBox.warning(self,'No voice','Select a valid reference WAV or disable Echo.')
            return
        if self.cfg['echo_enabled'] and not CHATTERBOX_PY.exists():
            QMessageBox.critical(self, 'Missing WhisperBox runtime', 'Chatterbox: ' + str(CHATTERBOX_PY))
            return
        self.shutting_down=False
        self.current_operation='starting_models'
        # A matching, previously successful CUDA validation is sufficient to
        # start immediately. Fresh/changed runtimes must still pass the live
        # background check before CUDA model startup is permitted.
        if (self.cfg.get('compute_device') == 'cuda' and
                not self._cuda_live_checked and not self._cuda_cached):
            QMessageBox.information(self, 'Verifying CUDA',
                'CUDA has not yet passed validation for these runtimes.\n'
                'The current background verification must finish before starting models.')
            return
        if self.cfg.get('compute_device') == 'cuda' and not self._cuda_ready:
            QMessageBox.warning(self, 'CUDA unavailable',
                'CUDA has not passed the startup checks in both runtimes.\n' + self._cuda_reason)
            return
        diagnostic('models_start_requested', model=self.cfg['model'], device=self.cfg['compute_device'], echo=self.cfg['echo_enabled'], gpu=gpu_snapshot())
        self.running=True
        self.global_button.setText('STOP MODELS / FREE VRAM')
        self.status.setText('Starting Whisper/Silero...')
        self.model_ready=False
        self.tts_ready=False
        self._listener_initialized = False
        self._listener_capture_ready = False
        self._listener_first_frame = False
        self.whisper=self.make_process(WHISPER_PY,'listen_worker.py',['--device', self.cfg['compute_device']])
        self.attach_process_diagnostics(self.whisper, 'Whisper')
        self.whisper.readyReadStandardOutput.connect(lambda p=self.whisper: self.read_whisper(p))
        self.whisper.finished.connect(lambda *args, p=self.whisper: self.whisper_exited(p, *args))
        self.whisper.start()
        if self.cfg['echo_enabled']:
            self.ensure_tts()
        self.changed()

    def ensure_tts(self):
        if self.tts and self.tts.state() != QProcess.ProcessState.NotRunning:
            if self.tts_active_model != self.cfg['model']:
                self.switch_requested = True
                QTimer.singleShot(0, self.apply_tts_switch)
                return True
            return True
        if not CHATTERBOX_PY.exists():
            self.status.setText('Chatterbox runtime Python not found')
            return False
        if not self.cfg['voice'] or not Path(self.cfg['voice']).is_file():
            self.status.setText('Select a valid reference WAV for Chatterbox')
            return False
        self.tts_ready=False
        self.tts_active_model = self.cfg['model']
        self.last_tts_intent = None
        self.tts=self.make_process(CHATTERBOX_PY,'tts_service.py',['--model',self.cfg['model'], '--device', self.cfg['compute_device']])
        self.last_tts_lines=[]
        self.tts_buf=''
        self.attach_process_diagnostics(self.tts, 'Chatterbox')
        self.tts.readyReadStandardOutput.connect(lambda p=self.tts: self.read_tts(p))
        self.tts.finished.connect(lambda *args, p=self.tts: self.tts_exited(p, *args))
        self.tts.errorOccurred.connect(lambda e, p=self.tts: self.log_backend('chatterbox', ('Expected process exit: ' if self.process_intents.get(id(p)) else 'QProcess error: ') + str(e)))
        self.tts.start()
        self.status.setText('Loading Chatterbox...')
        return True

    def apply_device_restart(self):
        """Restart workers on the selected device, without cutting off active audio."""
        if not self.device_restart_pending or self.device_restart_in_progress:
            return
        if not self.running or self.shutting_down:
            self.device_restart_pending = False
            return
        if (self.busy or self.recording_active or self.pending_speech is not None
                or self.conversion_recording or self.conversion_finalizing):
            self.status.setText('Device change queued until current recording/playback finishes...')
            return
        self.device_restart_in_progress = True
        diagnostic('device_switch_requested', from_device=self.cfg.get('compute_device'), operation=self.current_operation)
        target = self.cfg['compute_device']
        try:
            self.status.setText('Restarting inference services on ' + target.upper() + '...')
            self.stop_models()
            self.start_models()
            if self.running:
                self.status.setText('Starting inference services on ' + target.upper() + '...')
        finally:
            self.device_restart_pending = False
            self.device_restart_in_progress = False

    def apply_tts_switch(self):
        # A live generation/playback must finish before the old model is stopped.
        if not self.running or not self.switch_requested or self.shutting_down or self.device_restart_pending:
            return
        if self.busy or self.pending_speech is not None or self.pending_conversion is not None:
            self.status.setText('Chatterbox switch queued until current speech finishes...')
            return
        target = self.cfg['model']
        if self.tts_active_model == target and self.tts is not None and self.tts.state() != QProcess.ProcessState.NotRunning:
            self.switch_requested = False
            self.changed()
            return
        old = self.tts
        if old is not None:
            # Process-bound event handlers ignore any late events from retired workers.
            self.terminate_process(old, 'tts_model_switch')
            old.deleteLater()
        self.tts = None
        self.tts_ready = False
        self.tts_active_model = None
        self.switch_requested = False
        diagnostic('tts_model_switch', target_model=target, gpu=gpu_snapshot())
        self.status.setText('Loading Chatterbox ' + target + '...')
        # If Echo is off, delay loading until Generate Speech is clicked.
        if self.cfg['echo_enabled']:
            self.ensure_tts()
        else:
            self.status.setText('Chatterbox set to ' + target + '; loads on next TTS request')
        self.changed()

    def stop_models(self):
        self._conversion_record_timer.stop()
        self.conversion_finalizing = False
        self.stop_conversion_preview()
        self.conversion_recording = False
        self.conversion_record_button.setText('RECORD AUDIO')
        if self.playback_task is not None and self.playback_task.isRunning():
            if isinstance(self.playback_task,StreamPlayback):self.playback_task.halt()
            else:sd.stop()
        diagnostic('models_stop_requested', operation=self.current_operation, gpu=gpu_snapshot())
        self._generation_epoch += 1
        self.shutting_down=True
        self.listening=False
        self.busy=False
        self.running=False
        self.pending_speech=None
        self.pending_conversion=None
        self.switch_requested=False
        self.device_restart_pending=False
        self.recording_active=False
        self.tts_active_model=None
        sd.stop()
        for p in (self.whisper,self.tts):
            self.terminate_process(p, 'stop_models')
            if p is not None:
                p.deleteLater()
        self.whisper=self.tts=None
        self.whisper_buf=''
        self._listener_initialized = False
        self._listener_capture_ready = False
        self._listener_first_frame = False
        self.model_ready=self.tts_ready=False
        self.global_button.setText('START MODELS')
        self.status.setText('Models stopped. GPU/CPU memory released after processes exit.')
        self.changed()
        self.shutting_down=False
        self.current_operation='idle'
        self.refresh_controls()
        diagnostic('models_stopped', gpu=gpu_snapshot())

    def log_backend(self, name, text):
        folder=ROOT/'logs'
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/(name+'.log')).open('a',encoding='utf-8') as f:
            f.write('[{}] {}\n'.format(datetime.now().isoformat(timespec='seconds'),text.rstrip()))

    def tts_exited(self, process, exit_code, exit_status):
        if process is not self.tts:
            return
        if self.shutting_down or not self.running or self.last_tts_intent:
            self.last_tts_intent = None
            return
        self.tts_ready=False
        self.tts_active_model=None
        process.deleteLater()
        self.tts = None
        last=' | '.join(self.last_tts_lines[-5:])[-600:] or 'No backend error details captured'
        self.status.setText('Chatterbox exited (code {}). See logs/chatterbox.log'.format(exit_code))
        self.log_backend('chatterbox','Unexpected exit code={} hex={} status={} operation={} last={}'.format(exit_code,format_exit_code(exit_code),exit_status,self.current_operation,last))
        if self.busy or self.pending_speech is not None or self.pending_conversion is not None:
            self.pending_speech = None
            self.pending_conversion = None
            self.finish_cycle()
        self.refresh_controls()
        QMessageBox.warning(self,'Chatterbox stopped unexpectedly',
            'The Chatterbox process exited (code {}).\n\n{}\n\nSee: {}'
            .format(exit_code,last,ROOT/'logs'/'chatterbox.log'))

    def read_tts(self, process=None):
        if process is not None and process is not self.tts:
            return
        if self.tts is None:
            return
        chunk=bytes(self.tts.readAllStandardOutput()).decode('utf-8','replace')
        if chunk:
            self.log_backend('chatterbox',chunk)
            self.last_tts_lines.extend(line for line in chunk.splitlines() if line.strip())
            self.last_tts_lines=self.last_tts_lines[-40:]

        self.tts_buf += chunk
        lines=self.tts_buf.split('\n')
        self.tts_buf=lines.pop()
        if any(line.strip() == 'READY' for line in lines):
            diagnostic('tts_ready', model=self.tts_active_model, pid=int(self.tts.processId()), gpu=gpu_snapshot())
            if not self.busy and self.pending_speech is None and self.pending_conversion is None:
                self.current_operation='idle'
            if self.switch_requested or self.tts_active_model != self.cfg['model']:
                QTimer.singleShot(0, self.apply_tts_switch)
                return
            self.tts_ready=True
            self.status.setText('Chatterbox ready — ' + ('microphone muted' if self.muted else 'microphone listening') if self.listening else 'Chatterbox ready')
            self.refresh_controls()
            if self.pending_conversion is not None:
                source,target=self.pending_conversion
                self.pending_conversion=None
                self.start_conversion(source,target)
            elif self.pending_speech is not None:
                text=self.pending_speech
                self.pending_speech=None
                self.synthesize(text)
        elif 'ERROR:' in chunk:
            self.status.setText('Chatterbox failed; see logs/chatterbox.log')

    def read_whisper(self, process=None):
        if process is not None and process is not self.whisper:
            return
        if self.whisper is None:
            return
        data=bytes(self.whisper.readAllStandardOutput()).decode('utf-8','replace')
        if not hasattr(self,'whisper_buf'):self.whisper_buf=''
        self.whisper_buf+=data
        while '\n' in self.whisper_buf:
            line,self.whisper_buf=self.whisper_buf.split('\n',1)
            if not line.startswith('EVENT:'):
                if line.strip():
                    try:
                        with (ROOT/'logs'/'whisper.log').open('a', encoding='utf-8') as logfile:
                            logfile.write(line + '\n')
                    except OSError:
                        pass
                continue
            try: ev=json.loads(line[6:])
            except ValueError: continue
            kind=ev.get('kind')
            if kind=='ready':
                self.model_ready=True
                self._listener_session += 1
                self._listener_initialized = False
                self._listener_capture_ready = False
                self._listener_first_frame = False
                diagnostic('whisper_ready', pid=int(self.whisper.processId()),
                           listener_session=self._listener_session, gpu=gpu_snapshot())
                # Send one atomic initialization request. The worker opens its
                # capture stream only after acknowledging applied settings.
                self.listening=True
                self.command({'action': 'initialize', 'session': self._listener_session,
                              'settings': self.cfg, 'muted': self.muted})
                self.status.setText('Microphone muted' if self.muted else 'Initializing microphone...')
                self.refresh_controls()
            elif kind=='listener_initialized':
                if ev.get('session') == self._listener_session:
                    self._listener_initialized = True
                    diagnostic('listener_config_applied', session=self._listener_session,
                               muted=ev.get('muted'), device=ev.get('device'),
                               wasapi=ev.get('wasapi'))
            elif kind=='capture_ready':
                if ev.get('session') == self._listener_session:
                    self._listener_capture_ready = True
                    diagnostic('microphone_stream_opened', session=self._listener_session)
                    if not self.busy and not self.muted and not self.recording_active:
                        self.status.setText('Listening')
            elif kind=='capture_paused':
                if ev.get('session') == self._listener_session:
                    self._listener_capture_ready = False
                    diagnostic('listener_capture_paused', reason=ev.get('reason'),
                               session=self._listener_session)
            elif kind=='listener_resumed':
                if ev.get('session') == self._listener_session:
                    diagnostic('listener_resumed', session=self._listener_session,
                               muted=ev.get('muted'), listening=ev.get('listening'))
            elif kind=='busy':
                if ev.get('state'):
                    self.busy = True
                    self.recording_active = False
                    self.current_operation = 'transcription'
                    diagnostic('listener_busy', phase=ev.get('phase', 'transcribing'))
            elif kind=='transcription_started':
                diagnostic('whisper_transcription_started', audio_seconds=ev.get('audio_seconds'),
                           speech_seconds=ev.get('speech_seconds'))
            elif kind=='conversion_recorded':
                self.receive_conversion_recording(ev.get('audio_b64',''), ev.get('token'))
            elif kind=='recording':
                self.recording_active=True
                self.status.setText('Recording speech...')
            elif kind=='transcript':
                self.recording_active=False
                diagnostic('whisper_transcription_result', outcome=ev.get('outcome', 'unknown'),
                           text_characters=len(ev.get('text','')),
                           raw_segments=ev.get('raw_segments'),
                           accepted_segments=ev.get('accepted_segments'),
                           echo_enabled=self.cfg['echo_enabled'])
                self.handle_transcript(ev.get('text',''),ev.get('audio_b64',''))
            elif kind=='resume_required':
                self.recording_active=False
                self.finish_cycle()
            elif kind=='capture_interrupted':
                self.recording_active = False
                diagnostic('microphone_recording_interrupted')
            elif kind=='input_effect_warning':
                self.status.setText(ev.get('message', 'Input effects unavailable'))
                diagnostic('input_effect_warning', detail=ev.get('message', '')[:500])
            elif kind=='protocol_error':
                self.status.setText('Listener command rejected: ' + ev.get('message',''))
                diagnostic('listener_protocol_error', detail=ev.get('message','')[:500])
            elif kind=='input_error':
                self._listener_capture_ready = False
                self.status.setText('Audio input unavailable (retrying): ' + ev.get('message',''))
                diagnostic('microphone_input_error', detail=ev.get('message','')[:500])
            elif kind=='error':
                active_operation = self.busy or self.recording_active
                self.recording_active = False
                self.status.setText(ev.get('message','Worker error'))
                if active_operation or ev.get('message', '').startswith('Transcription:'):
                    self.finish_cycle()
            elif kind=='latency':
                diagnostic('latency_stage', **{k:v for k,v in ev.items() if k!='kind'})
            elif kind=='vad_probe':
                # Compact signal statistics; never export or log microphone samples.
                diagnostic('microphone_vad_probe', **{k:v for k,v in ev.items() if k != 'kind'})
                self._vad_last_report = ev
            elif kind=='meter':
                if not self._listener_first_frame:
                    self._listener_first_frame = True
                    diagnostic('microphone_frames_received', session=self._listener_session)
                self.mic_level=float(ev.get('level',0))
                self.mic_vad=float(ev.get('vad',0))
                self.mic_meter.setValue(max(0,min(100,round(self.mic_level*100))))
                self.vad_state_label.setText('Silero: ' + ('speech detected' if self.mic_vad>=float(self.cfg.get('vad_threshold',.5)) else 'listening / no speech'))
            elif kind=='status':self.status.setText(ev.get('message',''))
            if kind in ('ready', 'listener_initialized', 'capture_ready', 'capture_paused', 'listener_resumed', 'busy', 'conversion_recorded', 'recording', 'transcript', 'resume_required', 'error', 'input_error', 'protocol_error', 'capture_interrupted'):
                self.refresh_controls()

    def refresh_controls(self):
        """Render the single authoritative control policy; never infer state from labels."""
        if not hasattr(self, 'record_button'):
            return
        snapshot = ControlSnapshot(
            running=self.running, whisper_ready=self.model_ready,
            listening=self.listening, muted=self.muted,
            busy=self.busy, speech_recording=self.recording_active,
            conversion_recording=self.conversion_recording,
            conversion_finalizing=self.conversion_finalizing,
            pending_request=(self.pending_speech is not None or self.pending_conversion is not None),
            switching=(self.switch_requested or self.device_restart_pending or
                       self.device_restart_in_progress),
            shutting_down=self.shutting_down,
            waveform_available=self.conversion_audio_data is not None,
            preview_playing=bool(self.conversion_preview and self.conversion_preview.isRunning()),
            output_playing=bool(self.playback_task and self.playback_task.isRunning()),
        )
        controls = control_availability(snapshot)
        self.record_button.setText('UNMUTE MICROPHONE' if self.muted else 'MUTE MICROPHONE')
        for widget, name in (
            (self.record_button, 'microphone'),
            (self.conversion_record_button, 'record_audio'),
            (self.speak_button, 'generate'),
            (self.convert_button, 'convert'),
            (self.cancel_button, 'cancel'),
            (self.conversion_play, 'preview'),
            (self.global_button, 'models'),
        ):
            widget.setEnabled(controls[name])
        self.conversion_record_button.setText(
            'STOP RECORDING' if self.conversion_recording else 'RECORD AUDIO')

    def conversion_record_timeout(self):
        if not self.conversion_finalizing:
            return
        diagnostic('conversion_record_finalize_timeout')
        token = self._conversion_token
        self._conversion_token = None
        self.conversion_recording = False
        self.conversion_finalizing = False
        self.command({'action': 'conversion_record_stop', 'token': token})
        self.status.setText('Recording finalization timed out. You can record again.')
        self.refresh_controls()

    def whisper_exited(self, process=None, *args):
        if process is not None and process is not self.whisper:
            return
        if self.shutting_down or not self.running:
            return
        diagnostic('whisper_unexpected_exit', operation=self.current_operation)
        if process is not None:
            process.deleteLater()
        self.whisper = None
        self.model_ready = False
        self.listening = False
        self._listener_initialized = False
        self._listener_capture_ready = False
        self._listener_first_frame = False
        self.recording_active = False
        self.conversion_recording = False
        self.conversion_finalizing = False
        self._conversion_token = None
        self._conversion_record_timer.stop()
        self.pending_conversion = None
        self.pending_speech = None
        self.busy = False
        self.refresh_controls()
        self.status.setText('Whisper stopped unexpectedly. Stop Models to reset.')

    def toggle_listening(self):
        """One microphone button: mute or unmute automatic speech detection."""
        if (not self.running or not self.model_ready or not self.listening or
                self.device_restart_pending or self.device_restart_in_progress or
                self.shutting_down or self.conversion_recording or self.conversion_finalizing):
            return
        self.muted = not self.muted
        self.command({'action': 'mute', 'value': self.muted})
        self._listener_capture_ready = False if self.muted else self._listener_capture_ready
        if not self.muted:
            self._listener_first_frame = False
        self.status.setText('Microphone muted' if self.muted else 'Listening')
        self.refresh_controls()

    def handle_transcript(self,text,audio_b64):
        self.busy=True
        self.refresh_controls()
        self.echo_cycle_started_at=time.perf_counter()
        self.session=datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid4().hex[:6]
        if text:
            from PySide6.QtWidgets import QListWidgetItem
            item=QListWidgetItem('%s  %s'%(datetime.now().strftime('%H:%M:%S'),text[:110]))
            item.setData(Qt.ItemDataRole.UserRole,text)
            self.entries.insertItem(0,item)
            self.entries.setCurrentItem(item)
        if self.cfg['save_input'] and audio_b64:
            try:
                dest=saved_directory(self.cfg, 'recordings')/('input_'+self.session+'.wav')
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(base64.b64decode(audio_b64,validate=True))
            except (OSError,ValueError) as exc:
                self.status.setText('Could not save input WAV: '+str(exc))
        if self.cfg['save_text'] and text:
            dest=saved_directory(self.cfg, 'transcripts')/('transcript_'+self.session+'.txt')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(text+'\n',encoding='utf-8')
        if not text or not self.cfg['echo_enabled']:
            diagnostic('echo_generation_skipped',
                       reason='empty_transcript' if not text else 'echo_disabled',
                       echo_enabled=self.cfg['echo_enabled'],
                       listener_initialized=self._listener_initialized,
                       capture_ready=self._listener_capture_ready)
            self.finish_cycle()
        else:
            diagnostic('echo_generation_handoff', text_characters=len(text))
            self.synthesize(text)

    def speak_typed(self):
        if self.busy or self.pending_speech is not None or self.pending_conversion is not None or self.recording_active or self.conversion_recording:
            self.status.setText('Finish the current operation before generating more speech.')
            return
        if self.device_restart_pending:
            self.status.setText('Inference device is switching. Try again when ready.')
            return
        if self.switch_requested:
            self.status.setText('Chatterbox is switching models. Try again when ready.')
            return
        text=self.typed.toPlainText().strip()
        if not text or not self.running:
            return
        self.command({'action':'hold'})
        self.synthesize(text)
        self.refresh_controls()


    def synthesize(self,text):
        self.stop_conversion_preview()
        if not self.running:
            self.status.setText('Start models first')
            self.finish_cycle()
            return
        if self.switch_requested or (self.tts and self.tts_active_model != self.cfg['model']):
            self.pending_speech=text
            self.switch_requested=True
            self.refresh_controls()
            QTimer.singleShot(0, self.apply_tts_switch)
            return
        if not self.tts_ready:
            if self.ensure_tts():
                self.pending_speech=text
                self.refresh_controls()
                self.status.setText('Loading Chatterbox for speech generation...')
            else:
                self.finish_cycle()
            return
        if not self.cfg['voice'] or not Path(self.cfg['voice']).is_file():
            self.status.setText('Select a valid reference WAV')
            self.finish_cycle()
            return
        self.busy=True
        self.status.setText('Generating cloned speech...')
        self.current_operation='tts_generation'
        diagnostic('generation_requested', kind='tts', model=self.cfg['model'], text_characters=len(text), beam_size=self.cfg.get('whisper_beam_size',3), chunked=self.cfg.get('chunked_tts',False), streaming=self.cfg.get('streaming_playback',False), voice_cache=self.cfg.get('voice_conditioning_cache',True), gpu=gpu_snapshot())
        self.refresh_controls()
        task=Synthesis(text,self.cfg['voice'],self.cfg['model'],self.cfg)
        task._generation_epoch = self._generation_epoch
        self.workers.append(task)
        self.streaming_chunks=[]
        self.streaming_complete=False
        self.streaming_started_at=time.perf_counter()
        task.chunk.connect(self.speech_chunk)
        task.stream_complete.connect(self.speech_stream_complete)
        task.ready.connect(self.speech_ready)
        task.failed.connect(self.speech_error)
        task.finished.connect(lambda:self.workers.remove(task) if task in self.workers else None)
        task.start()

    def _stale_generation_result(self):
        task = self.sender()
        return task is not None and getattr(task, '_generation_epoch', -1) != self._generation_epoch

    def speech_chunk(self,wav,index):
        if self._stale_generation_result() or not self.running:return
        self.streaming_chunks.append(wav)
        diagnostic('tts_chunk_received',index=index,bytes=len(wav),
                   elapsed_since_request=round(time.perf_counter()-self.streaming_started_at,4))
        if index==0:
            self.command({'action':'hold'})
            task=StreamPlayback(self.cfg['output_device'],self.cfg['output_gain'],self.cfg['playback_speed'])
            self.playback_task=task
            self.play_button.setText('STOP PLAYING')
            self.workers.append(task)
            task.failed.connect(lambda error:self.status.setText('Playback: '+error))
            task.started_playback.connect(lambda: diagnostic('audio_playback_started',
                elapsed_since_tts_request=round(time.perf_counter()-self.streaming_started_at,4),
                elapsed_since_transcription=round(time.perf_counter()-self.echo_cycle_started_at,4) if self.echo_cycle_started_at else None))
            task.finished_playback.connect(lambda t=task:self.stream_playback_finished(t))
            task.finished.connect(lambda:self.workers.remove(task) if task in self.workers else None)
            task.start()
            self.status.setText('Playing speech while generating remaining segments...')
        if isinstance(self.playback_task,StreamPlayback):
            self.playback_task.add_chunk(wav)

    def speech_stream_complete(self,text):
        if self._stale_generation_result():return
        try:
            wav=join_wav_frames(self.streaming_chunks)
            self.audio_last=wav
            self.most_recent_wav=None
            if self.cfg['save_output']:
                dest=saved_directory(self.cfg,'generated_audio');dest.mkdir(parents=True,exist_ok=True)
                name='output_'+(self.session or datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid4().hex[:6])+'.wav'
                self.most_recent_wav=dest/name
                self.most_recent_wav.write_bytes(wav)
            self.update_last_output_button()
            diagnostic('tts_stream_complete',chunks=len(self.streaming_chunks),wav_bytes=len(wav),
                       elapsed_seconds=round(time.perf_counter()-self.streaming_started_at,4))
        except Exception as exc:
            self.status.setText('Audio assembly failed: '+str(exc))
            diagnostic('tts_stream_assembly_failed',error=str(exc))
        self.streaming_complete=True
        if isinstance(self.playback_task,StreamPlayback):
            if self.playback_task.isRunning():self.playback_task.finish_input()
            else:self.finish_cycle()
        else:self.finish_cycle()

    def stream_playback_finished(self,task):
        if self.playback_task is not task:
            return  # A replaced playback worker must not finish a newer request.
        self.playback_done(task)
        if self.streaming_complete and self.current_operation=='tts_generation':
            self.finish_cycle()

    def speech_ready(self,wav,text):
        if self._stale_generation_result(): return
        diagnostic('generation_completed', operation=self.current_operation, wav_bytes=len(wav), gpu=gpu_snapshot())
        if not self.running:
            return
        # Keep only the latest generated WAV in RAM for immediate replay.
        self.audio_last=wav
        self.most_recent_wav=None
        if self.cfg['save_output']:
            try:
                dest=saved_directory(self.cfg, 'generated_audio')
                dest.mkdir(parents=True,exist_ok=True)
                name='output_'+(self.session or datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid4().hex[:6])+'.wav'
                self.most_recent_wav=dest/name
                self.most_recent_wav.write_bytes(wav)
            except OSError as exc:
                self.most_recent_wav=None
                self.status.setText('Could not save output WAV: '+str(exc))
        self.update_last_output_button()
        self.status.setText('Speech generated. Playing...')
        self.play_audio(wav, resume=True)

    def speech_error(self,error):
        if self._stale_generation_result(): return
        if isinstance(self.playback_task,StreamPlayback):self.playback_task.halt()
        diagnostic('generation_failed', operation=self.current_operation, error=str(error)[:1200], gpu=gpu_snapshot())
        self.status.setText('TTS error: '+error)
        self.finish_cycle()

    def play_audio(self,wav,resume=False):
        if self.playback_task is not None and self.playback_task.isRunning():
            if isinstance(self.playback_task,StreamPlayback):self.playback_task.halt()
            else:sd.stop()
            if not isinstance(self.playback_task,StreamPlayback):self.playback_task.wait(1200)
        task=Playback(wav,self.cfg['output_device'],self.cfg['output_gain'],self.cfg['playback_speed'])
        self.playback_task=task
        self.play_button.setText('STOP PLAYING')
        self.workers.append(task)
        task.failed.connect(lambda e:self.status.setText('Playback: '+e))
        if resume:task.finished_playback.connect(self.finish_cycle)
        task.finished_playback.connect(lambda t=task:self.playback_done(t))
        task.finished.connect(lambda:self.workers.remove(task) if task in self.workers else None)
        task.start()

    def playback_done(self, task):
        if self.playback_task is task:
            self.playback_task = None
            self.play_button.setText('PLAY LAST')
        self.refresh_controls()

    def play_last(self):
        if self.playback_task is not None and self.playback_task.isRunning():
            if isinstance(self.playback_task,StreamPlayback):self.playback_task.halt()
            else:sd.stop()
            self.status.setText('Playback stopped. Last completed clip remains available.')
            return
        if self.audio_last is not None:
            self.play_audio(self.audio_last)
        else:
            self.status.setText('No synthesized audio yet')

    def open_last(self):
        if not self.audio_last:
            self.status.setText('No synthesized audio yet')
            return
        if self.most_recent_wav and self.most_recent_wav.is_file():
            subprocess.Popen(['explorer.exe','/select,',str(self.most_recent_wav)])
            return
        # No unsaved files are written silently. Save only on explicit request.
        dest=saved_directory(self.cfg, 'generated_audio')
        dest.mkdir(parents=True,exist_ok=True)
        filename='output_'+datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid4().hex[:6]+'.wav'
        path,_=QFileDialog.getSaveFileName(self,'Save latest generated audio',str(dest/filename),'WAV files (*.wav)')
        if not path:
            return
        try:
            path=Path(path)
            if path.suffix.lower()!='.wav':
                path=path.with_suffix('.wav')
            path.write_bytes(self.audio_last)
            self.most_recent_wav=path
            self.update_last_output_button()
            subprocess.Popen(['explorer.exe','/select,',str(path)])
        except OSError as exc:
            QMessageBox.warning(self,'Unable to save audio',str(exc))

    def finish_cycle(self):
        self.cancel_button.setEnabled(False)
        diagnostic('cycle_finished', operation=self.current_operation)
        self.current_operation='idle'
        self.busy=False
        self.session=None
        self.refresh_controls()
        self.command({'action':'resume'})
        self.status.setText('Listening' if self._listener_capture_ready and not self.muted else ('Microphone muted' if self.muted else 'Resuming microphone...'))
        if self.device_restart_pending:
            QTimer.singleShot(0, self.apply_device_restart)
        elif self.switch_requested:
            QTimer.singleShot(0, self.apply_tts_switch)

    def check_services(self):
        # The worker owns opening/retrying the device. GUI timers only report
        # readiness; they never inject competing capture-reset commands.
        if hasattr(self,'model_state_label'):
            whisper = 'Ready' if self.model_ready else ('Loading' if self.running else 'Stopped')
            tts = 'Ready' if self.tts_ready else ('Loading' if self.tts is not None and self.tts.state()!=QProcess.ProcessState.NotRunning else 'Stopped')
            mic = 'Muted' if self.muted else ('Recording' if self.recording_active else ('Listening' if self._listener_capture_ready else ('Arming' if self.listening else 'Off')))
            vc = 'Converting' if self.current_operation=='voice_conversion' else 'Idle'
            self.model_state_label.setText(f'Whisper: {whisper}   |   Chatterbox: {tts}   |   Microphone: {mic}   |   Voice conversion: {vc}')
        if not self.running:return
        for label,p in (('Whisper',self.whisper),('Chatterbox',self.tts)):
            if p is not None and p.state()==QProcess.ProcessState.NotRunning:
                self.status.setText(label+' stopped unexpectedly. Stop Models to reset.')

    def copy_selected_transcript(self):
        item=self.entries.currentItem()
        if item is not None:
            QApplication.clipboard().setText(item.data(Qt.ItemDataRole.UserRole) or '')
            self.status.setText('Selected transcription copied to clipboard')
        else:
            self.status.setText('Select a transcription in the log first')

    def empty_saved(self, folder, suffix):
        directory=saved_directory(self.cfg, folder)
        files=[p for p in directory.glob('*') if p.is_file() and p.suffix.lower()==suffix] if directory.is_dir() else []
        if not files:
            QMessageBox.information(self,'Nothing to delete','No matching saved files were found.')
            return
        answer=QMessageBox.question(self,'Delete saved files',
            'Permanently delete {} {} files from\n{}?'.format(len(files),suffix,str(directory)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        failures=[]
        for file in files:
            try:
                file.unlink()
                if self.most_recent_wav == file:
                    self.most_recent_wav=None
            except OSError as exc:
                failures.append(str(exc))
        self.status.setText('{} saved files deleted{}'.format(len(files)-len(failures),
            ' ({} failed)'.format(len(failures)) if failures else ''))
        if failures:
            QMessageBox.warning(self,'Some files could not be deleted','\n'.join(failures[:5]))

    def quit_app(self):
        self._settings_flush_timer.stop()
        self.flush_settings()
        diagnostic('app_exit_requested', source='tray')
        self.cfg['tray_enabled']=False
        self.stop_conversion_preview()
        self.stop_models()
        self.tray.hide()
        QApplication.instance().quit()

    def closeEvent(self,event):
        if self.tray_setting.isChecked() and self.tray.isVisible():
            event.ignore()
            self.hide()
            self.tray.showMessage('WhisperBox','Running in tray. Use Exit to release GPU memory.')
        else:
            self._settings_flush_timer.stop()
            self.flush_settings()
            self.stop_conversion_preview()
            # QApplication is configured not to quit merely because the last
            # window closes (to support tray mode), so quit explicitly here.
            self.stop_models()
            self.tray.hide()
            event.accept()
            QTimer.singleShot(0, QApplication.instance().quit)

def start_single_instance(app):
    # The server is held for the lifetime of QApplication. Qt cleans up its
    # local endpoint on normal exit; retry after removing a stale endpoint.
    name = 'WhisperBox-' + os.environ.get('USERNAME', 'local')

    def activate_existing():
        client = QLocalSocket()
        client.connectToServer(name)
        if not client.waitForConnected(500):
            return False
        client.write(b'ACTIVATE\n')
        client.flush()
        if not client.waitForBytesWritten(1500):
            client.disconnectFromServer()
            return False
        # Keep the new process alive until the running instance receives the
        # activation request. Otherwise Windows may drop the foreground grant
        # when the transient pythonw process exits immediately.
        acknowledged = client.waitForReadyRead(2000)
        answer = bytes(client.readAll()) if acknowledged else b''
        client.disconnectFromServer()
        diagnostic('duplicate_instance_activation', acknowledged=(b'OK' in answer))
        return b'OK' in answer

    if activate_existing():
        return None
    server = QLocalServer(app)
    if not server.listen(name):
        # Stale endpoint from a crash; never delete one if another instance responds.
        if activate_existing():
            return None
        QLocalServer.removeServer(name)
        if not server.listen(name):
            QMessageBox.warning(None, 'WhisperBox',
                'Could not establish the single-instance lock.\n' + server.errorString())
            return None
    return server


if __name__=='__main__':
    _gui_init_start = time.perf_counter()
    diagnostic('gui_import_complete', import_ms=round((_gui_init_start-_APP_IMPORT_STARTED)*1000,1))
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    server = start_single_instance(app)
    if server is None:
        sys.exit(0)
    _main_start = time.perf_counter()
    window = Main()
    diagnostic('gui_init_timing', main_ms=round((time.perf_counter()-_main_start)*1000,1), total_ms=round((time.perf_counter()-_APP_IMPORT_STARTED)*1000,1))
    diagnostic('gui_window_created')

    def restore_from_duplicate():
        while server.hasPendingConnections():
            connection = server.nextPendingConnection()
            # Acknowledge before the new process exits, and restore on the GUI
            # loop only after the socket message has been consumed.
            if connection.waitForReadyRead(150):
                command = bytes(connection.readAll()).strip()
            else:
                command = b'ACTIVATE'  # compatibility with older launchers
            if command.startswith(b'ACTIVATE'):
                connection.write(b'OK\n')
                connection.flush()
                connection.waitForBytesWritten(300)
                diagnostic('duplicate_instance_restore', command='ACTIVATE')
                QTimer.singleShot(0, window.restore_window)
            connection.disconnectFromServer()
            connection.deleteLater()

    server.newConnection.connect(restore_from_duplicate)
    window.show()
    diagnostic('gui_shown', total_ms=round((time.perf_counter()-_APP_IMPORT_STARTED)*1000,1))
    # Queue background checks after the event loop starts and Windows paints the first frame.
    QTimer.singleShot(0, window.restore_window)
    QTimer.singleShot(200, lambda: window.validate_startup(show_dialog=False))
    sys.exit(app.exec())
