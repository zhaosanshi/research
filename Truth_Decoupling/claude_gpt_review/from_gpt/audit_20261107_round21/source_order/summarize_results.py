"""Summarize the predeclared ten-arm E37 run without selecting a winning layer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result={'models':{},'scope':'192 previously exposed tuning questions; all ten arms, both C signs/depths retained'}
    lines=['# 固定 E 前后放置 C：完整自由问答结果','',
        '所有结果来自原先已经使用过的192道调参题。两模型各十组全部重新生成；E权重、模型、层位和正负两档事前固定。Llama上限48词元，Qwen上限256词元，不在模型之间直接相减。',
        '层编号从0起：Llama的E在15层、C在7/23层；Qwen的E在17层、C在8/26层。before/after指C在E之前/之后，m1/p1指固定负/正号，EC是组合。符号没有看效果后重定义。',
        'C只写基础模型原提示末位，E每一步执行真实来源交换；来源没有C。前后C层使用同一训练问题集校准相对实际写入中位量，没有依据回答效果调力度。','',
        '统一采用完整个人拒答守卫。同问同完整答案另经同预算父七组核对：Llama1899条、Qwen1818条标签全部一致，reference_locked最终副本与v1数值及字节相同。Qwen原模型51正确而不是旧原判52，来自旧“我不知道”误判的统一规则修正，不是生成变化。','',
        '表内25%/50%/75%分别交付48/96/144个实质回答；使用同一个未干预模型的平均词元对数概率排序，正确和错误回答同样有资格入选。无法填足配额记为不可行。','']
    for model in ('llama3.1-8b','qwen3-8b'):
        run=args.inputs/model/'full192';rawpath=run/'tune.json';gradepath=run/'tune_scored_items.json';reportpath=run/'tune_matched_coverage.json'
        raw,grade,report=read(rawpath),read(gradepath),read(reportpath)
        assert raw['completed'] and grade['completed'] and grade['grading_completed'] and len(raw['evaluations'])==192
        assert grade['grading_source_generation_sha256']==sha(rawpath)
        assert report['source_files']['tune.json']==sha(rawpath) and report['source_files']['tune_scored_items.json']==sha(gradepath)
        names=raw['conditions'];rows=grade['evaluations'];stats={}
        for name in names:
            counts=Counter(r['outcome'][name] for r in rows);control='E_source' if name.startswith('EC_') else 'baseline'
            changes=Counter((r['outcome'][control]+'->'+r['outcome'][name]) for r in rows)
            common=[r for r in rows if r['outcome'][control]!='abstain' and r['outcome'][name]!='abstain']
            cnorm=[r['trace'][name][0]['c_writes'][0]['stored_delta_over_base'] for r in raw['evaluations'] if r['trace'][name][0]['c_writes']]
            gate_delta=[r['gate_readout'][name]['yes_minus_no']-r['gate_readout'][control]['yes_minus_no'] for r in raw['evaluations']]
            values=[]
            for point in report['matched_coverage']['coverages']:
                arm=point['arms'][name];values.append(arm['correct'] if arm['feasible'] else None)
            stats[name]={'natural':dict(counts),'matched_correct':values,
                'unfinished':sum(not r['finished'][name] for r in rows),'replacement_character':sum('\ufffd' in r['outputs'][name] for r in rows),
                'text_changes_from_no_C':sum(r['outputs'][name]!=r['outputs'][control] for r in rows),
                'outcome_transitions_from_no_C':dict(changes),'common_answered':len(common),
                'correct_change_on_common_answered':sum(int(r['outcome'][name]=='correct')-int(r['outcome'][control]=='correct') for r in common),
                'C_relative_write':None if not cnorm else {'minimum':min(cnorm),'median':statistics.median(cnorm),'maximum':max(cnorm)},
                'gate_margin_delta':{'minimum':min(gate_delta),'median':statistics.median(gate_delta),'maximum':max(gate_delta),
                                     'positive':sum(x>0 for x in gate_delta),'negative':sum(x<0 for x in gate_delta),'zero':sum(x==0 for x in gate_delta)},
                'new_unfinished_ids_vs_no_C':[r['pair_id'] for r in rows if not r['finished'][name] and r['finished'][control]]}
        result['models'][model]={'source_sha256':{p.name:sha(p) for p in (rawpath,gradepath,reportpath)},'conditions':stats}
        lines+=['## '+model,'','| 方法 | 自然答对/实答 | 25%答对 | 50%答对 | 75%答对 | 未结束 |','|---|---:|---:|---:|---:|---:|']
        for name in names:
            row=stats[name];c=row['natural'];cells=['不可行' if x is None else str(x) for x in row['matched_correct']]
            lines.append('| %s | %d/%d | %s | %s | %s | %d |'%(name,c.get('correct',0),c.get('correct',0)+c.get('wrong',0),*cells,row['unfinished']))
        lines+=['','组合相对固定E的逐题状态变化：','',
                '| 组合 | 错→对 | 对→错 | 弃答→对 | 对→弃答 | 共同实答题上的正确净变化 | 新增未结束题数 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for name in names:
            if not name.startswith('EC_'):continue
            row=stats[name];t=row['outcome_transitions_from_no_C']
            lines.append('| %s | %d | %d | %d | %d | %+d | %d |'%(name,t.get('wrong->correct',0),t.get('correct->wrong',0),
                  t.get('abstain->correct',0),t.get('correct->abstain',0),row['correct_change_on_common_answered'],len(row['new_unfinished_ids_vs_no_C'])))
        lines+=['','C-only能力读数方向检查（含E的读数属于跨提示诊断，另见原件）：','',
                '| C条件 | Yes−No增加/减少/不变 | 改变量中位数 | 回答首步相对实际写入中位量 |',
                '|---|---:|---:|---:|']
        for name in names:
            if not name.startswith('C_'):continue
            row=stats[name];g=row['gate_margin_delta']
            lines.append('| %s | %d/%d/%d | %+.5f | %.4f%% |'%(name,g['positive'],g['negative'],g['zero'],g['median'],100*row['C_relative_write']['median']))
        lines+=['']
    lines+=['## 解释限制','',
        '前后C的物理层、方向本身都不同；训练校准只对齐未干预训练状态上的相对写入中位量，不保证每道新题、每个E改写后的状态或C语义作用一样。这不是同两个算子的纯换序实验。',
        '自然总正确数不能代替同作答率比较；各自生成置信、未结束/乱码敏感性和预先声明的多重比较校正在配套报告中统一给出，不挑某个读出、符号或层位当最终赢家。']
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
