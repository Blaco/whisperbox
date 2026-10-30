"""Enumerate system-audio loopback endpoints from Whisper's Python runtime."""
import json
try:
    import soundcard as sc
    devices = sc.all_microphones(include_loopback=True)
    loopbacks = [{'id': str(d.id), 'name': str(d.name)} for d in devices if getattr(d, 'isloopback', False)]
    speaker = sc.default_speaker()
    print(json.dumps({'loopbacks': loopbacks, 'default_id': str(speaker.id) if speaker else None}))
except Exception as exc:
    print(json.dumps({'loopbacks': [], 'default_id': None, 'error': type(exc).__name__ + ': ' + str(exc)}))
