"""Bounded text-to-speech segments and WAV assembly for long-form Chatterbox output.

No Chatterbox or GUI imports: this is shared by the local service and its client.
"""
import io
import re
import wave
from collections import deque
from itertools import chain

# A conservative text budget per Chatterbox generate() call. The model's audio
# token limit is independent of the total requested text length.
MAX_SEGMENT_CHARACTERS = 160
CHUNKED_SEGMENT_CHARACTERS = 120
MAX_TEXT_CHARACTERS = 100_000
MAX_SEGMENT_AUDIO_SECONDS = 38.0


def split_speech(text, chunked=False, max_chars=None):
    """Return bounded, naturally separated speech segments without dropping words.

    Chunked controls the preferred segment size, not whether model safety
    segmentation is permitted. Whitespace is normalized for spoken text.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    limit = max_chars if max_chars is not None else (
        CHUNKED_SEGMENT_CHARACTERS if chunked else MAX_SEGMENT_CHARACTERS)
    if not isinstance(limit, int) or limit < 2:
        raise ValueError('TTS segment size must be at least two characters')
    if len(text) <= limit or (max_chars is None and len(text) <= MAX_SEGMENT_CHARACTERS):
        return [text]

    pieces = []
    while len(text) > limit:
        window = text[:limit + 1]
        minimum = max(1, limit // 3)
        # End clauses on natural boundaries when possible, but avoid very
        # small leading segments. A space is the fallback for long sentences.
        natural = [m.end() for m in re.finditer(r'[.!?;:,](?=\s)', window)
                   if m.end() >= minimum]
        if natural:
            end = natural[-1]
        else:
            end = window.rfind(' ', minimum, limit + 1)
            if end <= 0:
                end = limit  # e.g. unbroken identifiers or languages without spaces
        piece = text[:end].strip()
        if not piece:
            raise ValueError('Unable to segment TTS input')
        pieces.append(piece)
        text = text[end:].lstrip()
    if text:
        pieces.append(text)
    return pieces


def checked_audio_segments(pieces, synthesize, sample_rate, on_retry=None):
    """Yield (text, audio, retry_depth), regenerating token-limit-sized clips.

    A near-40-second clip from a short text is suspicious for Chatterbox EOS
    failure. Split that text further instead of returning likely truncated or
    looping speech. After bounded attempts, raise instead of saving bad audio.
    """
    if sample_rate <= 0:
        raise ValueError('Invalid TTS sample rate')
    pending = deque((piece, 0) for piece in pieces)
    while pending:
        piece, depth = pending.popleft()
        audio = synthesize(piece)
        frames = len(audio)
        if not frames:
            raise RuntimeError('Chatterbox generated empty audio')
        duration = frames / sample_rate
        if duration >= MAX_SEGMENT_AUDIO_SECONDS:
            if depth >= 5 or len(piece) <= 8:
                raise RuntimeError('Chatterbox reached the audio generation limit even after reducing the text segment')
            halves = split_speech(piece, max_chars=max(2, len(piece) // 2))
            if len(halves) < 2:
                midpoint = len(piece) // 2
                halves = [piece[:midpoint], piece[midpoint:]]
            if on_retry is not None:
                on_retry(piece, depth, duration, len(halves))
            pending.extendleft((part, depth + 1) for part in reversed(halves))
            continue
        yield piece, audio, depth


def join_wav_frames(wav_frames):
    """Combine PCM WAV frames without repeated float32 concat/encode passes."""
    wav_frames = iter(wav_frames)
    first = next(wav_frames, None)
    if first is None:
        raise ValueError('No TTS audio received')
    output = io.BytesIO()
    count = 0
    with wave.open(output, 'wb') as destination:
        for frame in chain((first,), wav_frames):
            with wave.open(io.BytesIO(frame), 'rb') as source:
                params = source.getparams()
                if params.comptype != 'NONE' or params.sampwidth != 2:
                    raise ValueError('Expected 16-bit PCM WAV audio frames')
                format_key = (params.nchannels, params.sampwidth, params.framerate)
                if count == 0:
                    destination.setnchannels(params.nchannels)
                    destination.setsampwidth(params.sampwidth)
                    destination.setframerate(params.framerate)
                elif format_key != expected_format:
                    raise ValueError('TTS segments have inconsistent audio formats')
                expected_format = format_key
                destination.writeframes(source.readframes(params.nframes))
                count += 1
    return output.getvalue()
