"""Offline per-window tactile reward illustration, not recorded online RL return."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from tactile_force_envelope.common import load_config
from tactile_force_envelope.model import TransformerEnvelopeExpert
from tactile_force_envelope.stable_contact_experiment import _gates_for_rows
from tactile_force_envelope.transformer_train import TransformerWindowDataset, _predict


def main():
    torch.set_num_threads(4)
    config = load_config(ROOT/'configs/experiments/pi05_jax_zrl_ink_beaker.yaml')
    workspace = Path(config['data']['output_dir'])
    out = ROOT/'outputs/videos/ink_episode18_tactile'
    folder = workspace/'sustained_contact_seed_44/stable_evaluation'
    cal = json.loads((folder/'bilateral_sigma_calibration.json').read_text())
    ckpt_path = folder/'stable_contact_calibrated_checkpoint.pt'
    assert hashlib.sha256(ckpt_path.read_bytes()).hexdigest() == cal['checkpoint_sha256']
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    condition = torch.load(workspace/'condition_tokens.pt', map_location='cpu', weights_only=False)
    history = torch.load(workspace/'causal_tactile_history.pt', map_location='cpu', weights_only=False)
    digest = hashlib.sha256((workspace/'windows.jsonl').read_bytes()).hexdigest()
    assert condition['windows_sha256'] == history['windows_sha256'] == digest
    rows = [json.loads(l) for l in (workspace/'windows.jsonl').open()]
    rows = sorted([r for r in rows if r['episode_index']==18], key=lambda r:r['start_timestamp'])
    video_dataset = Path(config['vision']['local_v3_dataset'])
    video_rows = pq.read_table(video_dataset/'data/chunk-000/file-000.parquet',filters=[('episode_index','=',18)]).to_pylist()
    video_states = {r['frame_index']:r['observation.state'] for r in video_rows}
    assert all(np.allclose(r['state'],video_states[r['anchor_frame_index']],atol=1e-5) for r in rows), 'Episode/frame mapping differs'
    gates = _gates_for_rows(rows,history['history'],history['mask'],config).all(axis=1)
    cache_ok = np.array([bool(condition['condition_mask'][r['window_index']].any()) for r in rows])
    selected = [r for r,v in zip(rows,cache_ok) if v]
    assert selected, 'No cached features for episode'
    d = checkpoint['dimensions']
    model = TransformerEnvelopeExpert(condition_dim=d['condition_dim'],state_dim=d['state_dim'],
        action_dim=d['action_dim'],action_steps=d['action_steps'],config=checkpoint['model_config'],
        history_dim=d.get('history_dim',0),history_steps=d.get('history_steps',0))
    model.load_state_dict(checkpoint['model_state_dict'])
    dataset = TransformerWindowDataset(selected,condition['condition_tokens'],condition['condition_mask'],
        checkpoint['normalization'],config,history['history'],history['mask'])
    pred = _predict(model,DataLoader(dataset,batch_size=64),torch.device('cpu'),checkpoint['normalization'])
    positions = {r['window_index']:i for i,r in enumerate(selected)}
    records = []
    for r,gate,cached in zip(rows,gates,cache_ok):
        per = r['tactile']['per_sensor']
        y = .5*(per['left_tip']['filtered_top2_mean']+per['right_tip']['filtered_top2_mean'])
        peak = max(per[s]['raw_max'] for s in ('left_tip','right_tip'))
        safety = float(peak>cal['raw_peak_threshold']['threshold_N'])
        mu=scale=lower=upper=float('nan')
        soft = 0. if not gate else float('nan')
        if cached:
            i = positions[r['window_index']]
            mu = float(.5*(pred['left']['mu'][i]+pred['right']['mu'][i]))
            scale = max(1e-6,float(.5*(pred['left']['sigma'][i]+pred['right']['sigma'][i])))
            lower,upper = mu+cal['lower_alpha']*scale, mu+cal['upper_alpha']*scale
            soft = float(gate)*(np.clip(max(0,lower-y)/2,0,1)**2+np.clip(max(0,y-upper)/2,0,1)**2)
        start = r['anchor_frame_index']/30
        records.append(dict(window_index=r['window_index'],start_s=start,end_s=start+.3,
            gate=bool(gate),cached=bool(cached),target_avg_N=y,mu_avg_N=mu,scale_N=scale,
            lower_N=lower,upper_N=upper,soft_cost=soft,peak_cost=safety,
            reward=-min(1.,.6*soft+.4*safety) if np.isfinite(soft) else float('nan')))
    with (out/'offline_reward_windows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report=dict(method='Offline demonstration-window illustration; not logged online RL reward',
        checkpoint=str(ckpt_path),calibration=str(folder/'bilateral_sigma_calibration.json'),split=rows[0]['split'],
        formula='r_window = -min(1, 0.6 * gate * (clip(low/2,0,1)^2 + clip(high/2,0,1)^2) + 0.4 * peak)',
        gate='Both sides: at least 3 of last 4 valid samples in 200ms history exceed 5N; fixed at anchor',
        timing='Reward plotted at window end; 300ms Top2Mean labels, not instantaneous force',
        caveats=['Uses demonstrated future actions and per-anchor cached z_rl, not an online policy rollout.',
                 'No E18 pair averaging or episode budget applied: curve shows local weighted window reward.',
                 'Missing build windows or active windows without cached features remain gaps.',
                 'Training episode: visualization only, not held-out performance evidence.'],
        windows=len(records),cached_windows=int(cache_ok.sum()),gated_windows=int(gates.sum()),
        missing_active_predictions=int((gates & ~cache_ok).sum()),
        penalized_windows=int(sum(r['reward']<0 for r in records)),
        minimum_window_reward=float(np.nanmin([r['reward'] for r in records])),
        peak_penalty_windows=sum(r['peak_cost']>0 for r in records))
    (out/'offline_reward_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fig=Figure(figsize=(12.8,3.2),dpi=100,facecolor='white');canvas=FigureCanvasAgg(fig)
    ax=fig.add_axes([.08,.20,.90,.56]);ax.set(xlim=(0,22.3),ylim=(-1.02,.04),xlabel='Time (s)',ylabel='Tactile reward')
    ax.tick_params(labelsize=12);ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(14)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#dbe1e6')
    fig.text(.08,.89,'Offline window reward (300 ms)',fontsize=15)
    status=fig.text(.98,.89,'',ha='right',fontsize=13)
    x=[];y=[]
    for r in records:
        # A reward is known only after its observation window has completed.
        x.extend([r['end_s'],r['end_s']+.03,float('nan')]);y.extend([r['reward'],r['reward'],float('nan')])
    x,y=np.array(x),np.array(y)
    line,=ax.plot([],[],color='#aa3655',lw=2,marker='o',ms=3)
    cursor=ax.axvline(0,color='#9099a0',ls='--',lw=1)
    destination=out/'ink_episode_018_task_tactile_reward.mp4'
    with av.open(str(out/'ink_episode_018_task_and_tactile.mp4')) as source, av.open(str(destination),'w',options={'movflags':'+faststart'}) as output:
        stream=output.add_stream('libx264',rate=30);stream.width=1280;stream.height=1400;stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'fast'};stream.codec_context.thread_count=4
        for i,frame in enumerate(source.decode(video=0)):
            now=i/30;line.set_data(np.where(x<=now,x,np.nan),y);cursor.set_xdata([now,now])
            done=[r for r in records if r['end_s']<=now]
            status.set_text(f"Last window: {done[-1]['reward']:.3f}" if done else 'Awaiting first window')
            canvas.draw();rgb=np.vstack([frame.to_ndarray(format='rgb24'),np.asarray(canvas.buffer_rgba())[:,:,:3]])
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):output.mux(packet)
            if i==600:cv2.imwrite(str(out/'reward_preview.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        for packet in stream.encode():output.mux(packet)
    line.set_data(x,y);cursor.set_visible(False);status.set_text('')
    fig.savefig(out/'tactile_reward_time.png',dpi=180)
    with av.open(str(destination)) as check:count=sum(1 for _ in check.decode(video=0))
    assert count==669
    print(json.dumps(report,indent=2));print(destination, count,flush=True)


if __name__=='__main__':main()
