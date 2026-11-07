#!/usr/bin/env python3
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
font=Path('/System/Library/Fonts/Hiragino Sans GB.ttc')
if font.exists():
 fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
out=json.loads((ROOT/'full_independent_recount.json').read_text())
rows=[r for m in out.values() for r in m['comparisons']]
keys=list(dict.fromkeys((r['model'],r['method'],r['c_layer'],r['c_alpha'],r['e_layer']) for r in rows))
assert len(keys)==16
for selector,label in [('simple','简单筛题'),('commitment','能力筛题')]:
 fig,axes=plt.subplots(1,3,figsize=(14,8),sharey=True,constrained_layout=True)
 for ax,q in zip(axes,['0.25','0.5','0.75']):
  rs=[r for r in rows if r['selector']==selector and r['coverage']==q]
  for r in rs:
   key=(r['model'],r['method'],r['c_layer'],r['c_alpha'],r['e_layer']);y=keys.index(key)
   lo,hi=r['question_bootstrap_reselect_ci95_correct_count'];delta=r['EC_minus_E']
   color='#2e6da4' if r['order']=='C_before_E' else '#b8562b'
   ax.plot([lo,hi],[y,y],color=color,alpha=.7,lw=1.6)
   ax.scatter([delta],[y],color=color,s=32,zorder=3)
  ax.axvline(0,color='#606060',lw=1,ls='--');ax.grid(axis='x',alpha=.15)
  ax.set_title(f'实际作答率 {int(float(q)*100)}% / {int(float(q)*384)} 句')
  ax.set_xlabel('组合相对固定 E 多答对的句数')
  ax.spines[['top','right']].set_visible(False)
 axes[0].set_yticks(range(len(keys)),[f'{"Llama" if m.startswith("llama") else "Qwen"} {method.upper()}  E{e}/C{c} {alpha:+g}' for m,method,c,alpha,e in keys]);axes[0].invert_yaxis()
 fig.suptitle(f'E38 真假判断：{label}，全部预定层位与符号',fontsize=15)
 fig.legend(handles=[Line2D([0],[0],color='#2e6da4',marker='o',label='先 C 后 E'),Line2D([0],[0],color='#b8562b',marker='o',label='先 E 后 C')],loc='outside lower center',ncol=2,frameon=False,title='线段为按原问题重抽、重新筛题的 95% 区间；已见调参题，不是确认集')
 fig.savefig(ROOT/f'{selector}_all_points.png',dpi=180);fig.savefig(ROOT/f'{selector}_all_points.svg');plt.close(fig)
print('Saved both readouts, all 96 comparisons, PNG and SVG')
