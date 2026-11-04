"""Export timestamp-labelled camera contact sheets from recorded USB episodes."""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/path/to/VLA_RL_dataset/franka_usb_tactile_310_20260716/vla_dataset')


def sample(meta, camera, seconds, root=ROOT):
    prefix = f'videos/observation.images.{camera}'
    path = root / prefix / f"chunk-{meta[prefix + '/chunk_index']:03d}" / f"file-{meta[prefix + '/file_index']:03d}.mp4"
    target = meta[prefix + '/from_timestamp'] + seconds
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(target / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time >= target - 1 / 120:
                return frame.to_ndarray(format='bgr24')
    raise ValueError(f'Missing frame: {path} at {target}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--fractions', type=float, nargs='+', default=[.25, .45, .65])
    parser.add_argument('--camera', default='camera_camera_2_color_image_raw')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT)
    parser.add_argument('--export-video', action='store_true')
    args = parser.parse_args()
    rows = {r['episode_index']: r for path in (args.dataset / 'meta/episodes').rglob('*.parquet')
            for r in pq.read_table(path).to_pylist()}
    tiles = []
    records = []
    for episode in args.episodes:
        meta = rows[episode]
        duration = (meta['length'] - 1) / 30
        line = []
        for fraction in args.fractions:
            time = fraction * duration
            frame = cv2.resize(sample(meta, args.camera, time, args.dataset), (480, 360))
            tile = np.full((394, 480, 3), 255, np.uint8)
            tile[34:] = frame
            cv2.putText(tile, f'EP {episode:03d} | {time:.2f}s / {duration:.2f}s',
                        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 1, cv2.LINE_AA)
            line.append(tile)
            records.append(dict(episode=episode, time_s=time, camera=args.camera, dataset=str(args.dataset)))
        tiles.append(np.hstack(line))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.vstack(tiles))
    args.output.with_suffix('.json').write_text(json.dumps(records, indent=2))
    print(args.output)
    if args.export_video:
        from render_usb_tactile_video import decode_episode

        for episode in args.episodes:
            meta = rows[episode]
            prefix = f'videos/observation.images.{args.camera}'
            path = args.dataset / prefix / f"chunk-{meta[prefix + '/chunk_index']:03d}" / f"file-{meta[prefix + '/file_index']:03d}.mp4"
            frames = decode_episode(path, meta[prefix + '/from_timestamp'], meta['length'], 30)
            destination = args.output.parent / f'{args.output.stem}_ep{episode:03d}.mp4'
            with av.open(str(destination), 'w') as output:
                stream = output.add_stream('libx264', rate=30)
                stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
                stream.pix_fmt = 'yuv420p'
                stream.options = {'crf': '18', 'preset': 'fast'}
                for i, rgb in enumerate(frames):
                    frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                    frame.pts = i
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            with av.open(str(destination)) as check:
                count = sum(1 for _ in check.decode(video=0))
            assert count == meta['length'], (count, meta['length'])
            print(f'{destination}: {count} decoded frames')


if __name__ == '__main__':
    main()
