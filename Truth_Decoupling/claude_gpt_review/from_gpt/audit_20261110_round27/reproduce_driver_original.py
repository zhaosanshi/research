"""Replay the reviewed 44b57ac driver on synthetic data; no API/model requests."""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_driver_bindings import DriverBindingsTests, D


def main():
    path = HERE / 'reviewed_e43_driver.py.txt'
    loader = importlib.machinery.SourceFileLoader('e43_reviewed_44b57ac', str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    original = importlib.util.module_from_spec(spec);loader.exec_module(original)
    fixture = DriverBindingsTests();fixture.setUp()
    out = {'scope':'Original 44b57ac functions with deterministic offline stubs; not observed scientific data pollution',
           'new_calls':0, 'findings':{}}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, sig = original.run_identity(fixture.meta, fixture.systems, fixture.args, [9])
            changed = dict(fixture.meta, model_config=dict(fixture.meta['model_config'],dtype='bfloat16'))
            _, sig2 = original.run_identity(changed, fixture.systems, fixture.args, [9])
            assert sig == sig2
            out['findings']['model_dtype_not_bound'] = {'same_identity_after_dtype_change':True}
            fixture.freeze()
            called=[]
            with patch.object(original.pd,'generate_prompts',side_effect=lambda *a,**k:called.append(True)):
                original.run_freeze(fixture.args,fixture.meta,fixture.prepared,fixture.bundle,fixture.systems,
                                    {'different':'identity'},'different-signature')
            assert not called
            out['findings']['freeze_skip_ignores_identity']={'old_completed_file_silently_retained':True,'generation_calls':0}
            fixture.args.splits=['train'];fixture.args.pair_ids=['q1'];fixture.freeze()
            (fixture.run/'frozen_train.json').unlink()
            fixture.args.pair_ids=['q2']
            frozen,_=original.load_frozen(fixture.args,'train',fixture.sig,allow_pilot=True)
            assert frozen['question_ids']==['q1']
            out['findings']['pilot_subset_collision']={'requested':['q2'],'returned':frozen['question_ids']}
            fixture.args.pair_ids=None;fixture.args.splits=['train','tune'];fixture.freeze()
            outdir=fixture.run/'dirs';outdir.mkdir(exist_ok=True)
            (outdir/'E_L0.pt').write_text('not even a checkpoint')
            original.run_train(fixture.args,fixture.meta,fixture.prepared,fixture.bundle,fixture.systems,fixture.ident,fixture.sig)
            out['findings']['train_skip_ignores_checkpoint_and_parameters']={'invalid_checkpoint_silently_skipped':True}
            (outdir/'E_L0.pt').unlink();fixture.train()
            fixture.args.arms=['none']
            def drift(bundle,prompts,**kwargs):
                rows=fixture.generate(bundle,prompts,**kwargs)
                for row in rows:row.update(text='Different answer',token_ids=[12])
                return rows
            with patch.object(original.pd,'generate_prompts',side_effect=drift):
                original.run_generate(fixture.args,fixture.meta,fixture.prepared,fixture.bundle,fixture.systems,fixture.ident,fixture.sig)
            result=D.read(fixture.run/'gen/E_L0_none_tune.json')
            assert result['summary']['changed']==1 and result['completed']
            out['findings']['zero_write_drift_published_complete']={'changed':1,'completed':True}
    finally:
        fixture.doCleanups()
    (HERE/'driver_original_reproductions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
