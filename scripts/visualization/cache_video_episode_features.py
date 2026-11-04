"""Extract real per-anchor VLA features into a separate video-only cache."""
import argparse
import hashlib
import json
import sys

from add_ink_reward_video import ROOT, Path, np, torch, load_config
from tactile_force_envelope.vision import _LocalV3VideoReader, PI05_JAX_CAMERA_MAP

sys.path.insert(0, '/path/to/rlt_openpi_franka/third_party/openpi/packages/openpi-client/src')
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=95)
    parser.add_argument('--port', type=int, default=18041)
    parser.add_argument('--task', choices=['ink','pastry'], default='ink')
    args = parser.parse_args()
    filename = 'pi05_jax_zrl_ink_beaker.yaml' if args.task=='ink' else 'pi05_jax_zrl_pastry_A298.yaml'
    cfg = load_config(ROOT/'configs/experiments'/filename)
    work = Path(cfg['data']['output_dir'])
    rows = [json.loads(l) for l in (work/'windows.jsonl').open()]
    rows = sorted([r for r in rows if r['episode_index'] == args.episode], key=lambda r:r['start_timestamp'])
    assert rows and all(r['split'] == 'test' for r in rows)
    cache = torch.load(work/'condition_tokens.pt', map_location='cpu', weights_only=False)
    vision = cfg['vision']
    client = WebsocketClientPolicy('127.0.0.1', args.port)
    assert client.get_server_metadata()['z_dim'] == 512
    reader = _LocalV3VideoReader(Path(vision['local_v3_dataset']), rows, vision['camera_order'],
        vision['source_episode_count'], vision['removed_source_episode_ids'])
    tokens, checks = {}, []
    try:
        for row in rows:
            payload = dict(state=np.asarray(row['state'], np.float32), prompt=str(row['task']))
            for field in vision['camera_order']:
                payload[PI05_JAX_CAMERA_MAP[field]] = reader.read(row, field)
            result = client.infer(payload)
            token = torch.as_tensor(np.asarray(result['z_rl'], np.float32).copy()).reshape(1,512)
            assert torch.isfinite(token).all()
            index = row['window_index']
            tokens[index] = token
            if cache['condition_mask'][index].any():
                old = cache['condition_tokens'][index]
                checks.append(dict(window=index, max_difference=float((old-token).abs().max()),
                    cosine=float(torch.nn.functional.cosine_similarity(old.flatten(),token.flatten(),dim=0))))
            print(f"window={index} t={row['anchor_frame_index']/30:.2f} real z_rl extracted", flush=True)
    finally:
        reader.close()
    assert checks and min(c['cosine'] for c in checks) > .999, 'Model path differs from training feature cache'
    output = ROOT/'outputs/videos/contact_video_selection'
    output.mkdir(exist_ok=True)
    artifact = dict(checkpoint=vision['checkpoint'], windows_sha256=hashlib.sha256((work/'windows.jsonl').read_bytes()).hexdigest(),
        tokens=tokens, source_episode=args.episode, metadata=client.get_server_metadata(),
        existing_cache_checks=checks, selection='All built episode windows, independent of gate and future force labels')
    torch.save(artifact, output/f'{args.task}_{args.episode}_all_features.pt')
    (output/f'{args.task}_{args.episode}_feature_checks.json').write_text(json.dumps(checks, indent=2))
    print('Saved', len(tokens), 'real features; minimum cached cosine', min(c['cosine'] for c in checks))


if __name__ == '__main__':
    main()
