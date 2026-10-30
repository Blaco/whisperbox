"""Single control-availability policy for WhisperBox's automatic listening mode.

Keep Qt button state and lifecycle decisions separate: the GUI supplies a
snapshot of worker/capture states, and the policy returns button availability.
This module does not import Qt or start inference services.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlSnapshot:
    running: bool = False
    whisper_ready: bool = False
    listening: bool = False
    muted: bool = False
    busy: bool = False
    speech_recording: bool = False
    conversion_recording: bool = False
    conversion_finalizing: bool = False
    pending_request: bool = False
    switching: bool = False
    shutting_down: bool = False
    waveform_available: bool = False
    preview_playing: bool = False
    output_playing: bool = False


def control_availability(s: ControlSnapshot) -> dict[str, bool]:
    """Calculate every model/capture button from the same consistent snapshot.

    Listening is always automatic. Muting does not require Chatterbox/Echo.
    The active conversion-capture Stop button remains clickable, even when
    another operation could otherwise block starting a new recording.
    """
    switching = s.switching or s.shutting_down
    ready = s.running and s.whisper_ready and s.listening and not switching
    capturing = s.speech_recording or s.conversion_recording or s.conversion_finalizing
    working = s.busy or s.pending_request
    new_work = s.running and not switching and not capturing and not working
    return {
        'microphone': bool(ready and not s.conversion_recording and not s.conversion_finalizing),
        'record_audio': bool(
            (s.conversion_recording and not s.conversion_finalizing) or
            (ready and not s.muted and not capturing and not working)
        ),
        'generate': bool(new_work),
        'convert': bool(new_work),
        'cancel': bool(s.running and working and not s.shutting_down),
        'preview': bool(
            s.waveform_available and not capturing and not working and not switching and
            (s.preview_playing or not s.output_playing)
        ),
        'models': bool(not s.shutting_down),
    }
