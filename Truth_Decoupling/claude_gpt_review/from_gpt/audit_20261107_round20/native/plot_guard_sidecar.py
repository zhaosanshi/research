"""Export separately named guarded figures; preserve original-label figures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REVIEW=HERE.parents[2]
sys.path[:0]=[str(HERE),str(REVIEW/'from_claude/code')]
from audit_native import read_artifacts,native,plot,MODELS,SOURCE
from core.qa_scoring import full_personal_refusal_decision,FULL_PERSONAL_REFUSAL_POLICY


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original={p.name:sha(p) for p in HERE.glob('native_commitment_*.*') if not p.name.startswith('native_commitment_guarded_')}
    dataset={};sources={};checked=0
    for model in MODELS:
        files,hashes=read_artifacts(None,model);scored=files['native_scored.json'];data=json.loads((SOURCE/model/'data.json').read_text())
        questions={r['pair_id']:r['question'] for r in data['qa']['tune']}
        corrected=json.loads((HERE.parent/'refusal_guard/sidecars'/('native_'+model)/'report.json').read_text())
        dataset[model]={};sources[model]=hashes
        for task,arms in scored['tasks'].items():
            changed=deepcopy(arms)
            if task=='qa':
                for name,rows in changed.items():
                    for row in rows:
                        if full_personal_refusal_decision(questions[row['pair_id']],row['output'])['apply']:row['outcome']='abstain'
                    actual=native(rows);expected=corrected['corrected_native_points'][name]
                    assert all(actual[k]==expected[k] for k in ('n','answered','correct','actual_coverage'))
                    checked+=1
            dataset[model][task]=(changed,scored['plan'])
    plot(dataset,'native_commitment_guarded','\nE28 原方向库；QA 用拒答守卫侧表，TF 不变；非新 TF 主表')
    assert original=={p.name:sha(p) for p in HERE.glob('native_commitment_*.*') if not p.name.startswith('native_commitment_guarded_')}
    manifest={'outcome_policy':FULL_PERSONAL_REFUSAL_POLICY,'source_artifact_hashes':sources,
              'policy_source_sha256':sha(REVIEW/'from_claude/code/core/qa_scoring.py'),
              'plotting_source_sha256':sha(HERE/'audit_native.py'),'entry_source_sha256':sha(Path(__file__)),
              'guarded_QA_points_matched_sidecar':checked,'original_figure_hashes_unchanged':original,
              'guarded_figures':{p.name:sha(p) for p in HERE.glob('native_commitment_guarded_*.*')}}
    (HERE/'guarded_figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('guarded figure points verified',checked,'original figures preserved')


if __name__=='__main__':main()
