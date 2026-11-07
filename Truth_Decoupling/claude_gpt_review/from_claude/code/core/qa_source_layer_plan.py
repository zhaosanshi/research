"""Validate the finite, declared layer-alignment follow-up; no open-ended sweep."""
import json
from pathlib import Path
import shutil

from .qa_generation_adapter import digest


def resolve_layer_plan(args, meta, default_layer, source_system, stored_layer=None):
    requested = getattr(args, 'layer', None)
    layer = requested if requested is not None else (stored_layer if stored_layer is not None else default_layer)
    if type(layer) is not int or layer not in meta['das_candidate_layers']:
        raise ValueError('requested layer has no verified original-DAS checkpoint')
    plan_path = getattr(args, 'layer_plan', None)
    run_dir = getattr(args, 'run_dir', None)
    if plan_path is None and run_dir is not None and (Path(run_dir) / 'layer_plan.json').exists():
        plan_path = Path(run_dir) / 'layer_plan.json'
    if plan_path is None:
        if layer != default_layer:
            raise ValueError('a new generation layer requires the frozen --layer-plan')
        return layer, None
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    model = plan.get('models', {}).get(meta['model'], {})
    if (plan.get('protocol') != 'source_layer_alignment_plan_v1'
            or model.get('num_layers') != meta['num_layers'] or model.get('reference_layer') != default_layer
            or layer not in model.get('additional_layers', [])
            or plan.get('source_system') != source_system
            or plan.get('generation', {}).get('alpha') != 1.0
            or plan['generation'].get('max_new_tokens') != meta['config']['max_new_tokens']
            or plan['generation'].get('batch_size') != 1
            or plan['generation'].get('fixed_center') is not False
            or plan['generation'].get('live_same_prefix_source') is not True
            or plan.get('training', {}).get('include_eos') is not False
            or plan['training'].get('base_model_frozen') is not True):
        raise ValueError('layer/model/source policy differs from the finite frozen layer plan')
    expected_depths = [int(meta['num_layers'] * fraction) - 1 for fraction in (.5, .75)]
    if model['additional_layers'] != expected_depths:
        raise ValueError('only the declared 50%/75% depth layers are allowed')
    if getattr(args, 'phase', None) == 'train':
        target = plan['training']; pilot = getattr(args, 'pilot', None)
        epochs = plan['pilot']['epochs'] if pilot is not None else target['epochs']
        if (args.epochs != epochs or args.lr != target['lr'] or args.clip != target['clip']
                or args.seed != target['seed'] or args.init != target['init']
                or (pilot is not None and pilot != plan['pilot']['pairs'])):
            raise ValueError('training request changes a variable outside the frozen layer-only follow-up')
    return layer, {'sha256': digest(plan_path), 'plan': plan, 'source_file': str(plan_path.resolve())}


def freeze_plan(run_dir, record):
    if record is None: return
    path = Path(run_dir) / 'layer_plan.json'; path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != record['sha256']:
            raise ValueError('run directory already contains a different layer plan')
    else:
        shutil.copyfile(record['source_file'], path)


def plan_identity(record):
    return None if record is None else {'sha256': record['sha256'], 'plan': record['plan']}
