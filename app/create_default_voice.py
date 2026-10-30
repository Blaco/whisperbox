"""Create a reusable reference WAV using Chatterbox's built-in default voice."""
from pathlib import Path
import argparse
import torch
import soundfile as sf

p=argparse.ArgumentParser()
p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
args=p.parse_args()
root=Path(__file__).resolve().parent.parent
output=root/'models'/'voices'/'Chatterbox_Default.wav'
if output.is_file() and output.stat().st_size > 4096:
    print('Existing default reference preserved:',output)
    raise SystemExit(0)
from chatterbox.tts import ChatterboxTTS
checkpoint=root/'models'/'chatterbox-original'
if not checkpoint.is_dir():
    raise RuntimeError('Original Chatterbox model missing; download models first')
model=ChatterboxTTS.from_local(checkpoint,device=args.device)
with torch.inference_mode():
    wav=model.generate('Hello, this is the default Chatterbox reference voice. I am ready to read your text aloud.')
output.parent.mkdir(parents=True,exist_ok=True)
audio=wav.detach().cpu().float().numpy().squeeze()
if audio.size < 16000:
    raise RuntimeError('Chatterbox produced an unexpectedly short default voice sample')
tmp=output.with_suffix('.tmp.wav')
sf.write(str(tmp),audio,model.sr)
tmp.replace(output)
print('Created Chatterbox default reference:',output)
