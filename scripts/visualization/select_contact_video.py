"""Audit feature availability and rank held-out episodes for video illustration."""
import json
from collections import defaultdict

from render_ink_envelope_reward import (
    ROOT, Path, np, torch, load_config, _gates_for_rows,
    TransformerEnvelopeExpert, TransformerWindowDataset, DataLoader, _predict,
)


def main():
    torch.set_num_threads(4)
    output = ROOT/'outputs/videos/contact_video_selection'
    output.mkdir(exist_ok=True)
    report = {}
    for task, filename in [('ink', 'pi05_jax_zrl_ink_beaker.yaml'),
                           ('pastry', 'pi05_jax_zrl_pastry_A298.yaml')]:
        cfg = load_config(ROOT/'configs/experiments'/filename)
        work = Path(cfg['data']['output_dir'])
        hist = torch.load(work/'causal_tactile_history.pt', weights_only=False)
        cache = torch.load(work/'condition_tokens.pt', weights_only=False)
        rows = [json.loads(line) for line in (work/'windows.jsonl').open()]
        rows = [r for r in rows if r['split'] == 'test']
        gates = _gates_for_rows(rows, hist['history'], hist['mask'], cfg).all(axis=1)
        selected = [r for r in rows if cache['condition_mask'][r['window_index']].any()]
        folder = work/'sustained_contact_seed_44/stable_evaluation'
        cal = json.loads((folder/'bilateral_sigma_calibration.json').read_text())
        ckpt = torch.load(folder/'stable_contact_calibrated_checkpoint.pt', weights_only=False)
        d = ckpt['dimensions']
        model = TransformerEnvelopeExpert(condition_dim=d['condition_dim'], state_dim=d['state_dim'],
            action_dim=d['action_dim'], action_steps=d['action_steps'], config=ckpt['model_config'],
            history_dim=d.get('history_dim', 0), history_steps=d.get('history_steps', 0))
        model.load_state_dict(ckpt['model_state_dict'])
        ds = TransformerWindowDataset(selected, cache['condition_tokens'], cache['condition_mask'],
            ckpt['normalization'], cfg, hist['history'], hist['mask'])
        pred = _predict(model, DataLoader(ds,batch_size=64), torch.device('cpu'), ckpt['normalization'])
        lookup = {r['window_index']: i for i,r in enumerate(selected)}
        episodes = defaultdict(list)
        for row, gate in zip(rows, gates):
            index = row['window_index']
            item = dict(window=index, time=row['anchor_frame_index']/30, gate=bool(gate), cached=index in lookup)
            y = np.mean([row['tactile']['per_sensor'][s]['filtered_top2_mean'] for s in ['left_tip','right_tip']])
            item['y'] = float(y)
            if index in lookup:
                i = lookup[index]
                mu = .5*(pred['left']['mu'][i]+pred['right']['mu'][i])
                scale = .5*(pred['left']['sigma'][i]+pred['right']['sigma'][i])
                lo, hi = mu+cal['lower_alpha']*scale, mu+cal['upper_alpha']*scale
                item.update(mu=float(mu), error=float(abs(y-mu)), violation=float(max(lo-y,0)+max(y-hi,0)))
            episodes[row['episode_index']].append(item)
        ranking=[]
        for ep, items in episodes.items():
            active=[r for r in items if r['gate'] and r['cached']]
            if not active: continue
            ranking.append(dict(episode=ep, windows=len(items), stable=len(active),
                start=active[0]['time'], end=active[-1]['time']+.3,
                mae=float(np.mean([r['error'] for r in active])),
                violations=sum(r['violation']>0 for r in active),
                max_violation=max(r['violation'] for r in active),
                internal_missing=sum(not (r['gate'] and r['cached']) for r in items if active[0]['time']<=r['time']<=active[-1]['time'])))
        ranking.sort(key=lambda r:(r['internal_missing'], -r['stable']))
        report[task] = dict(tau_on=cfg['stable_contact']['tau_on'], ranking=ranking, episodes=dict(episodes))
        print(task, json.dumps(ranking[:15], indent=2), flush=True)
    (output/'selection.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
