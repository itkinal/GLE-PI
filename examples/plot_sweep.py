"""Plot a saved sweep: python examples/plot_sweep.py results/size_duration.json"""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

path=Path(sys.argv[1])
data=json.loads(path.read_text())
for sign in (-1,1):
    durations=sorted({x['T'] for x in data['bands'] if x['sign']==sign})
    if not durations: continue
    fig,axes=plt.subplots(2,len(durations),figsize=(4*len(durations),7),squeeze=False)
    for column,T in enumerate(durations):
        for row in data['bands']:
            if row['sign']!=sign or row['T']!=T: continue
            V,I=np.array(row['sizes']),np.array(row['impacts'])
            axes[0,column].loglog(V,I,label=row['variant'])
            axes[1,column].semilogx(V,np.array([np.nan if x is None else x for x in row['exponents']]),label=row['variant'])
        axes[0,column].set_title(f'T = {T:g}; sign = {sign}')
        axes[1,column].axhspan(0.4,0.6,color='grey',alpha=0.15)
        axes[1,column].set_xlabel('Order-size magnitude')
        for ax in axes[:,column]: ax.grid(alpha=0.2)
    axes[0,0].set_ylabel('Signed terminal impact')
    axes[1,0].set_ylabel('Local exponent')
    axes[0,0].legend()
    fig.tight_layout()
    fig.savefig(path.with_name(f'size_duration_sign_{sign}.png'),dpi=180)
    plt.close(fig)
