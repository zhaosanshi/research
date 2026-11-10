"""Execute the real E43 stages against deterministic offline model stubs."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'from_claude/code'
sys.path[:0] = [str(CODE), str(CODE / 'scripts')]
import torch
import e43_qa_prompt_das as D


def question(index):
    return {'pair_id': 'q%d' % index, 'question': 'Question %d?' % index,
            'best_answer': 'Gold', 'gold': ['Gold'], 'incorrect': ['Bad']}


class DriverBindingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.source = self.root / 'source'; self.source.mkdir()
        self.run = self.root / 'run'; self.run.mkdir()
        self.prepared = {'splits': {'train': [question(1), question(2)], 'tune': [question(3)]},
                         'e_train': [question(1), question(2)]}
        self.meta = {'model': 'toy', 'model_config': {'path': str(self.root / 'model'), 'name': 'toy', 'dtype': 'float32'},
                     'hidden_size': 2, 'num_layers': 2, 'config': {'qa_system': 'base', 'qa_user': '{question}'}}
        self.source.joinpath('bank.json').write_text(json.dumps(self.meta))
        self.source.joinpath('questions.json').write_text(json.dumps(self.prepared))
        self.args = SimpleNamespace(source_run=self.source, run_dir=self.run, pair_ids=None, n_check=0,
            max_new_tokens=48, max_target_tokens=24, batch_size=2, output_projection='float32',
            seed=42, epochs=1, lr=.001, clip=1., overwrite=False, splits=['train','tune'],
            split='tune', family='E', layers=[0], arms=['none','main'], include_pilot=False)
        self.systems = {n: n for n in D.PROMPT_NAMES}
        self.bundle = SimpleNamespace(hidden_size=2)
        self.changed_plain = False
        self.patches = [
            patch.object(D.pd, 'turn_terminator_ids', return_value=[9]),
            patch.object(D.pd, 'render_prompt', side_effect=self.render),
            patch.object(D.pd, 'build_target', side_effect=self.target),
            patch.object(D.pd, 'generate_prompts', side_effect=self.generate),
            patch.object(D, 'attach_sources', side_effect=self.sources),
            patch.object(D.pd, 'train_direction', return_value=(torch.tensor([1.,0.]), {'history': [], 'epochs': 1})),
            patch.object(D.pd, 'baseline_ce', side_effect=lambda b,l,w,ps: [{'ce_plus':1.,'ce_minus':1.} for p in ps]),
        ]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        self.ident, self.sig = D.run_identity(self.meta, self.systems, self.args, [9])

    def render(self, bundle, system, template, text):
        return {'prefix_text':system+' '+text, 'prefix_ids':[1,2], 'prompt_length':2,
                'content_start':1, 'boundary_clean':True}

    def target(self, bundle, prompt, answer_text=None, answer_ids=None, **kwargs):
        return {'prefix_ids':[1,2], 'answer_ids':[11,9], 'full_ids':[1,2,11,9], 'prompt_length':2,
                'answer_length':2, 'content_start':1, 'answer_body_tokens':1,
                'truncated':False, 'terminator_appended':True, 'source_kind':'text' if answer_text else 'ids',
                'answer_text':answer_text or 'Answer'}

    def generate(self, bundle, prompts, **kwargs):
        rows = []
        for p in prompts:
            r={'text':'answer' if self.changed_plain and 'layer' not in kwargs else 'Answer',
               'token_ids':[11], 'finished':True, 'stop_id':9, 'n_tokens':1}
            if 'layer' in kwargs:r['write']={'prefill_rel':.1,'prefill_gap_abs':.2,'decode_rel_mean':.1}
            rows.append(r)
        return rows

    def sources(self, bundle, pairs, layer, batch_size):
        for p in pairs:
            p['plus']['source_hidden']=torch.tensor([2.,1.]);p['minus']['source_hidden']=torch.tensor([0.,1.])
        return pairs

    def freeze(self):
        D.run_freeze(self.args,self.meta,self.prepared,self.bundle,self.systems,self.ident,self.sig)

    def train(self):
        self.freeze();D.run_train(self.args,self.meta,self.prepared,self.bundle,self.systems,self.ident,self.sig)

    def test_source_configuration_and_data_identity(self):
        self.freeze()
        changed=copy.deepcopy(self.meta);changed['model_config']['dtype']='bfloat16'
        _,signature=D.run_identity(changed,self.systems,self.args,[9]);self.assertNotEqual(signature,self.sig)
        self.args.max_new_tokens=49
        ident,signature=D.run_identity(self.meta,self.systems,self.args,[9])
        with self.assertRaises(SystemExit):
            D.run_freeze(self.args,self.meta,self.prepared,self.bundle,self.systems,ident,signature)
        new=copy.deepcopy(self.prepared);new['splits']['train'][0]['question']='Different question'
        self.source.joinpath('questions.json').write_text(json.dumps(new))
        _,signature=D.run_identity(self.meta,self.systems,self.args,[9]);self.assertNotEqual(signature,self.sig)

    def test_local_model_content_changes_fingerprint(self):
        p=self.root/'model';p.mkdir();f=p/'model.safetensors';f.write_bytes(b'weights-a')
        a=D._local_model_files(p);f.write_bytes(b'weights-b');b=D._local_model_files(p)
        self.assertNotEqual(a,b)

    def test_pilot_cohort_cannot_silently_resume_another_subset(self):
        self.args.splits=['train'];self.args.pair_ids=['q1'];self.freeze()
        self.args.pair_ids=['q2']
        with self.assertRaisesRegex(ValueError,'cohort'):
            self.freeze()

    def test_duplicate_and_changed_frozen_questions_rejected(self):
        self.freeze();f=D.read(self.run/'frozen_train.json')
        changed=copy.deepcopy(f);changed['questions'][0]['question']='New question'
        with self.assertRaisesRegex(ValueError,'different question'):
            D.validate_frozen(D.seal(changed),self.prepared['splits']['train'],self.sig,True)
        duplicate=copy.deepcopy(f);duplicate['questions'][1]=copy.deepcopy(duplicate['questions'][0])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            D.validate_frozen(D.seal(duplicate),self.prepared['splits']['train'],self.sig,True)

    def test_training_parameter_change_refuses_existing_checkpoint(self):
        self.train();self.args.epochs=2
        with self.assertRaisesRegex(ValueError,'different seed, parameters'):
            D.run_train(self.args,self.meta,self.prepared,self.bundle,self.systems,self.ident,self.sig)

    def test_zero_write_case_change_is_fatal_and_no_generation_is_published(self):
        self.train();self.changed_plain=True;self.args.arms=['main']
        with self.assertRaisesRegex(RuntimeError,'zero-write'):
            D.run_generate(self.args,self.meta,self.prepared,self.bundle,self.systems,self.ident,self.sig)
        self.assertFalse((self.run/'gen/E_L0_main_tune.json').exists())

    def test_generation_resume_checks_direction_bytes_and_current_sources(self):
        self.train();D.run_generate(self.args,self.meta,self.prepared,self.bundle,self.systems,self.ident,self.sig)
        frozen=D.read(self.run/'frozen_tune.json');generation=D.read(self.run/'gen/E_L0_main_tune.json')
        D.validate_generation(generation,frozen,self.prepared['splits']['tune'],self.run)
        path=self.run/'dirs/E_L0.pt';record=torch.load(path,weights_only=False)
        record['seconds']+=1;torch.save(D.seal(record),path)
        with self.assertRaisesRegex(ValueError,'source file changed'):
            D.run_generate(self.args,self.meta,self.prepared,self.bundle,self.systems,self.ident,self.sig)

    def test_generation_seal_detects_changed_answer(self):
        self.train();D.run_generate(self.args,self.meta,self.prepared,self.bundle,self.systems,self.ident,self.sig)
        frozen=D.read(self.run/'frozen_tune.json');generation=D.read(self.run/'gen/E_L0_main_tune.json')
        generation['rows'][0]['text']='Changed'
        with self.assertRaisesRegex(ValueError,'unbound or changed'):
            D.validate_generation(generation,frozen,self.prepared['splits']['tune'],self.run)

    def test_wrong_layer_checkpoint_is_rejected(self):
        self.train();(self.run/'dirs/E_L1.pt').write_bytes((self.run/'dirs/E_L0.pt').read_bytes());self.args.layers=[1]
        with self.assertRaisesRegex(ValueError,'family/layer/pilot'):
            D.run_generate(self.args,self.meta,self.prepared,self.bundle,self.systems,self.ident,self.sig)

    def test_summary_and_all_stages_run_then_reject_mixed_identity(self):
        self.train();D.run_generate(self.args,self.meta,self.prepared,self.bundle,self.systems,self.ident,self.sig)
        D.run_summarize(self.args)
        summary=D.read(self.run/'summary.json');self.assertEqual(len(summary['generation']),2)
        self.assertEqual(summary['generation'][1]['changed'],0)
        f=self.run/'frozen_tune.json';d=D.read(f);d['questions'][0]['answers']['x']['text']='changed';D.save(f,D.seal(d))
        with self.assertRaises(ValueError):D.run_summarize(self.args)

    def test_unknown_or_repeated_requested_ids_fail(self):
        self.args.pair_ids=['no-such-id']
        with self.assertRaises(ValueError):D.pick_questions(self.prepared,'train',self.args)
        self.args.pair_ids=['q1','q1']
        with self.assertRaises(ValueError):D.pick_questions(self.prepared,'train',self.args)

    def test_literal_translation_best_answer_is_not_discarded_as_refusal(self):
        self.assertEqual(D.gold_target_text({'question':'Translate Je ne sais pas into English.', 'best_answer':"I don't know."}),
                         ("I don't know.",None))
        self.assertEqual(D.gold_target_text({'question':'What is the answer?', 'best_answer':'I have no comment'})[1],
                         'best_answer_personal_refusal')


if __name__=='__main__':unittest.main()
