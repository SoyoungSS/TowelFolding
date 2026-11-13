"""Resolve the locally validated default ACT checkpoint without loading torch."""
import hashlib
import json


def load_deployment(root):
    path = root / 'training/act/deployment.json'
    if not path.exists():
        return None
    selected = json.loads(path.read_text())
    checkpoint = root / selected['checkpoint']
    digest = hashlib.sha256()
    with (checkpoint / 'model.safetensors').open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != selected['checkpoint_sha256']:
        raise ValueError('Validated checkpoint has changed; rerun closed-loop acceptance before deploying it')
    return dict(selected, checkpoint=checkpoint)
