"""Download required model snapshots into local WhisperBox folders (online installer only)."""
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--component', required=True, choices=['chatterbox','whisper'])
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
MODELS.mkdir(parents=True, exist_ok=True)

if args.component == 'chatterbox':
    from huggingface_hub import snapshot_download
    for repo, folder in [
        ('ResembleAI/chatterbox', 'chatterbox-original'),
        ('ResembleAI/chatterbox-turbo', 'chatterbox-turbo'),
    ]:
        output = MODELS / folder
        print('Downloading', repo, 'to', output, flush=True)
        snapshot_download(repo_id=repo, local_dir=str(output),
                          allow_patterns=['*.safetensors','*.json','*.txt','*.pt','*.model'],
                          max_workers=2)
        if not any(x.stat().st_size > 100000 for x in output.glob('*.safetensors')):
            raise RuntimeError('No weights downloaded to {}'.format(output))
        print('Downloaded', repo, flush=True)
else:
    from faster_whisper.utils import download_model
    output = MODELS / 'whisper-small.en'
    print('Downloading small.en to', output, flush=True)
    download_model('small.en', output_dir=str(output))
    if not (output / 'model.bin').is_file() or (output / 'model.bin').stat().st_size < 100000:
        raise RuntimeError('Missing model.bin in {}'.format(output))
    print('Downloaded small.en', flush=True)
