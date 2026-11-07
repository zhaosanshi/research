"""Read real E1 checkpoints; inject an extra seed only into a temporary copy."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

import torch

root = Path('/workspace/Truth_Decoupling')
code = Path(__file__).resolve().parents[3] / 'from_claude/code'
sys.path[:0] = [str(code / 'scripts'), str(code)]
import e26_matched_coverage as e

records = []
for model in ('llama3.1-8b', 'qwen3-8b'):
    sweep = root / 'results_das_tfce/E23' / model / 'knob_sweep_das_tfce_full.json'
    original = root / 'results_das_tfce/E1' / model
    n_layers = e.read(original / 'summary.json')['num_layers']
    before = {p.name: e.digest(p) for p in original.glob('*.pt')}
    identity = e.verify_tf_sweep_source(sweep, original, range(n_layers))
    # A legacy sweep cannot prove what the directory held historically. Verify
    # the directions actually frozen into the completed E26 bank as well.
    run = root / 'results_e26_full_20261106' / model / 'main'
    bank_meta, e26_bank, _ = e.load_run(SimpleNamespace(run_dir=run, model=model))
    vector_differences = {}
    for l in range(n_layers):
        source_w, _ = e.load_das_direction(original, 'E', l, variant='tf-ce-bidir',
                                          align_to=e26_bank['tf_md_E_L%d' % l])
        diff = float(np.max(np.abs(source_w - e26_bank['tf_das_E_L%d' % l])))
        assert diff < 1e-10, (model, l, diff)
        vector_differences[str(l)] = diff
    layer = min(map(int, identity['loader_files']))
    first_name = identity['loader_files'][str(layer)][0]
    with tempfile.TemporaryDirectory(prefix='extra_seed_', dir=Path(__file__).parent) as tmp:
        relocated = Path(tmp)
        for name in identity['checkpoint_sha256']:
            shutil.copyfile(original / name, relocated / name)
        old_w, _ = e.load_das_direction(relocated, 'E', layer, variant='tf-ce-bidir')
        extra_name = 'das_E_L%d_r1_s999999_tf-ce-bidir.pt' % layer
        checkpoint = torch.load(relocated / first_name, map_location='cpu', weights_only=False)
        checkpoint['seed'] = 999999
        w = torch.as_tensor(checkpoint['W']).clone().float()
        w[0, 0] += .01
        checkpoint['W'] = w / w.norm(dim=0)
        torch.save(checkpoint, relocated / extra_name)
        new_w, loaded = e.load_das_direction(relocated, 'E', layer, variant='tf-ce-bidir')
        changed = float(torch.as_tensor(new_w - old_w).norm())
        assert extra_name in loaded['files'] and changed > 0
        try:
            e.verify_tf_sweep_source(sweep, relocated, range(n_layers))
        except ValueError as exc:
            assert 'loader checkpoint set' in str(exc)
            rejection = str(exc)
        else:
            raise AssertionError('extra loader seed was not rejected')
        negative = {'extra_checkpoint': extra_name, 'sha256': e.digest(relocated / extra_name),
                    'actual_loader_used_files': loaded['files'], 'direction_change_l2': changed,
                    'rejected': True, 'reason': rejection}
    after = {p.name: e.digest(p) for p in original.glob('*.pt')}
    assert before == after
    records.append({'model': model, 'verified_layers': n_layers,
                    'verified_checkpoint_count': len(identity['checkpoint_sha256']),
                    'sweep_sha256': e.digest(sweep), 'identity': identity,
                    'existing_e26_bank_fingerprint': bank_meta['fingerprint'],
                    'existing_e26_bank_vector_max_abs_differences': vector_differences,
                    'temporary_copy_negative': negative, 'production_checkpoints_unchanged': True})
out = {'records': records, 'all_passed': True, 'entry_sha256': e.digest(e.__file__)}
path = Path(__file__).with_name('server_source_check.json')
path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps([{k: v for k, v in r.items() if k != 'identity'} for r in records], indent=2))
