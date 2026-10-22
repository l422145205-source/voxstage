"""Plot an actual exported dialogue and its reserved cue slots, not a UI mockup."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from voxstage.audio import read

folder=ROOT/'examples/saved/spaceport-ae0d9eeccdd40041'
scene=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
timeline=json.loads((folder/'export/timeline.json').read_text(encoding='utf-8'))
values,rate=read(folder/'export/dialogue.wav')
fig,(ax,lanes)=plt.subplots(2,1,figsize=(13,4.8),sharex=True,
    gridspec_kw={'height_ratios':[2,1]},facecolor='#f3f5f4')
colors={'pilot':'#426c5e','commander':'#5b7892'}
for a in [ax,lanes]:
    a.set_facecolor('#ffffff');a.spines[['top','right']].set_visible(False)
    a.spines[['left','bottom']].set_color('#c9d4cf');a.tick_params(colors='#52645c',labelsize=9)
step=max(1,len(values)//6000)
ax.plot(np.arange(0,len(values),step)/rate,values[::step],color='#426c5e',linewidth=.6)
ax.set_title('VoxStage  |  real two-character dialogue and cue slots',loc='left',fontsize=15,pad=18,color='#21392d')
ax.set_ylabel('Amplitude');ax.set_ylim(-.55,.55);ax.grid(axis='x',alpha=.15)
actors={role['id']:i for i,role in enumerate(scene['cast'])}
for cue,placed in zip(scene['cues'],timeline['cues']):
    y=actors[cue['character']]
    lanes.barh(y,cue['end']-cue['start'],left=cue['start'],height=.58,color='#e6ece8')
    lanes.barh(y,placed['end']-placed['start'],left=placed['start'],height=.34,color=colors[cue['character']])
    lanes.text(cue['start']+.08,y,cue['id'],va='center',fontsize=9,color='white')
lanes.set_yticks(range(len(scene['cast'])),[r['speaker'] for r in scene['cast']])
lanes.set_xlabel('Seconds   |   light: reserved slot     color: placed audio')
lanes.set_xlim(0,len(values)/rate);lanes.set_ylim(-.6,len(scene['cast'])-.4)
fig.tight_layout();fig.savefig(ROOT/'examples/production_timeline.png',dpi=130,bbox_inches='tight');plt.close(fig)
print('rendered real waveform')
