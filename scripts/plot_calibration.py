"""Export the recorded Phase 22 calibration and CPU measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/benchmarks'


def main():
    read=lambda name:json.loads((OUT/('calibration_phase22_'+name+'.json')).read_text())
    fit=read('fit')['fit'];con=read('convergence');truth=read('truth');deep=read('refinement');s=read('summary')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axs[0,0]
    ax.semilogy(range(1,len(fit['history'])+1),[100*h['price_rmse'] for h in fit['history']],marker='o',color='#176485')
    ax.axhline(5,color='#ae5131',linestyle='--',label='Synthetic quote scale: 5 cents')
    ax.set(xlabel='L-BFGS closure evaluation',ylabel='Training RMSE (cents)',title='Joint network and initial-variance fit')
    ax.legend(fontsize=9)
    ax=axs[0,1]
    levels=['coarse','calibration','space','refined']
    lookup=lambda data:{r['mesh']:r['rmse']*100 for r in data['convergence']}
    for label,data,last,color in [('Heston numerical control',truth,deep['heston_control']['rmse']*100,'#ae5131'),
        ('Frozen fitted network',con,next(r['rmse']*100 for r in deep['results'] if r['mesh']=='fine' and not r['heldout']),'#176485')]:
        table=lookup(data)
        ax.semilogy(range(5),[table[k] for k in levels]+[last],marker='o',label=label,color=color)
    ax.set_xticks(range(5),['21 x 9\n12/yr','41 x 17\n24/yr','81 x 17\n24/yr','81 x 33\n48/yr','161 x 65\n96/yr'])
    ax.set(ylabel='Nine-quote RMSE (cents)',xlabel='Spatial mesh and rollback steps per year',title='Fixed models across grids; no refitting')
    ax.legend(fontsize=9)
    ax=axs[1,0]
    cases=[('scaling',1),('three_dimensional',1),('three_dimensional',2)]
    for j,(mode,color) in enumerate([('full','#ae5131'),('adjoint','#176485')]):
        values=[next(r['seconds'] for r in s['timings'] if r['mesh']==mesh and r['latent_dimensions']==m and r['mode']==mode
                     and r['execution']=='streamed' and r['contracts']==9) for mesh,m in cases]
        bars=ax.bar(np.arange(3)+(j-.5)*.35,values,.35,label=mode,color=color)
        ax.bar_label(bars,fmt='%.2f',fontsize=9)
    ax.set_xticks(range(3),['21 x 9\n1 latent, 12/yr','13 x 5\n1 latent, 6/yr','13 x 5 x 5\n2 latent, 6/yr'])
    ax.set(ylabel='Median seconds',title='Nine-contract objective and gradients, streamed')
    ax.margins(y=.2);ax.legend()
    ax=axs[1,1]
    for j,(mode,color) in enumerate([('full','#ae5131'),('adjoint','#176485')]):
        values=[next(r['peak_growth_mb'] for r in s['memory'] if r['mesh']=='diagnostic' and r['mode']==mode
                     and r['execution']==execution and r['contracts']==9) for execution in ['stacked','streamed']]
        bars=ax.bar(np.arange(2)+(j-.5)*.35,values,.35,label=mode,color=color)
        ax.bar_label(bars,fmt='%.1f',fontsize=9)
    ax.set_xticks(range(2),['Retain all contract graphs','Backward per contract'])
    ax.set(ylabel='Peak process growth (decimal MB)',title='Nine contracts on 9 x 5 diagnostic mesh')
    ax.margins(y=.2);ax.legend()
    fig.suptitle('QuantLibTorch Phase 22: synthetic multi-maturity calibration\nCPU float64; timing, memory and profiler workers measured separately',fontsize=14)
    fig.savefig(OUT/'calibration_phase22.png')
    fig.savefig(OUT/'calibration_phase22.pdf')
    plt.close(fig)


if __name__=='__main__':main()
