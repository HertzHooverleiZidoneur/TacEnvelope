"""Check visualization data against reward equations and original caches."""
import csv
import json

from add_ink_reward_video import ROOT, Path, np, torch, load_config, av


def main():
    reports = {}
    for task, episode, filename in [
        ('ink',95,'pi05_jax_zrl_ink_beaker.yaml'),
        ('pastry',48,'pi05_jax_zrl_pastry_A298.yaml'),
    ]:
        out = ROOT/f'outputs/videos/{task}_source{episode}_envelope_reward_early_predictions_continuous_presentation_t0_full_task'
        rows = list(csv.DictReader((out/'windows.csv').open()))
        summary = json.loads((out/'summary.json').read_text())
        cfg = load_config(ROOT/'configs/experiments'/filename)
        cache = torch.load(Path(cfg['data']['output_dir'])/'condition_tokens.pt',weights_only=False)
        missing = [r for r in rows if not cache['condition_mask'][int(r['window_index'])].any()]
        for r in rows:
            y, mu, lo, hi, scale = [float(r[k]) for k in ('observed_avg_N','predicted_avg_N','lower_N','upper_N','scale_N')]
            assert np.isfinite([y,mu,lo,hi,scale]).all() and lo <= hi
            assert np.isclose(lo, mu+summary['alpha05']*scale)
            assert np.isclose(hi, mu+summary['alpha95']*scale)
            gate = r['gate'] == 'True'
            soft = gate*(np.clip(max(lo-y,0)/2,0,1)**2+np.clip(max(y-hi,0)/2,0,1)**2)
            expected = -min(1,.6*soft+.4*float(r['peak_cost']))
            assert np.isclose(float(r['reward']), expected)
            if not gate:
                assert float(r['soft_cost']) == 0
        with av.open(str(out/f'{task}_task_envelope_reward.mp4')) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        assert decoded == summary['verified_decoded_frames']
        reports[task] = dict(episode=episode, duration_s=summary['duration_s'], windows=len(rows),
            newly_extracted_windows=len(missing), previously_missing_gate_on_windows=sum(r['gate']=='True' for r in missing),
            stable_windows=summary['stable_windows'], gate_first_on_s=min(float(r['start_s']) for r in rows if r['gate']=='True'),
            first_available_prediction_s=float(rows[0]['start_s']), minimum_reward=summary['minimum_reward'],
            penalized_windows=summary['penalized_windows'], verified_decoded_frames=decoded,
            reward_equations_verified=True, interval_equations_verified=True)
    destination=ROOT/'outputs/videos/contact_video_selection/verification.json'
    destination.write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))


if __name__ == '__main__':
    main()
