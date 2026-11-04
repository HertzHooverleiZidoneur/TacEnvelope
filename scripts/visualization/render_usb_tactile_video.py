#!/usr/bin/env python3
"""Render recorded USB cameras and timestamp-aligned measured tactile force."""

import argparse
import csv
import json
from pathlib import Path

import av
import cv2
import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


ROOT = Path('/path/to/VLA_RL_dataset/franka_usb_tactile_310_20260716')
CAMERAS = (
    ('camera_camera_1_color_image_raw', 'Wrist view'),
    ('camera_camera_2_color_image_raw', 'Overview'),
    ('d515_color_image_raw', 'Side view'),
)
COLORS = ('#2474bd', '#df6c32', '#202c39')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decode_episode(path, start, count, fps):
    images = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(start / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time < start - 0.25 / fps:
                continue
            expected = start + len(images) / fps
            if abs(frame.time - expected) > 0.5 / fps:
                raise ValueError(f'Unexpected video PTS: {frame.time} vs {expected}')
            images.append(frame.to_ndarray(format='rgb24'))
            if len(images) == count:
                break
    if len(images) != count:
        raise ValueError(f'Incomplete video: {path}, {len(images)}/{count}')
    return images


def latest_values(times, values, query, max_age=0.15):
    indices = np.searchsorted(times, query, side='right') - 1
    safe = np.maximum(indices, 0)
    valid = (indices >= 0) & ((query - times[safe]) <= max_age)
    return np.where(valid, values[safe], np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=15)
    parser.add_argument('--output', type=Path, default=Path(
        '/path/to/tactile_force_envelope/outputs/videos/usb_episode15_tactile'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    collection = json.loads((ROOT / 'manifest.json').read_text())
    batch = next(b for b in collection['batches']
                 if b['global_episode_start'] <= args.episode <= b['global_episode_end'])
    # Match original frame data to merged data; do not assume source IDs are contiguous.
    dataset = ROOT / 'vla_dataset'
    meta = next(row for path in (dataset / 'meta/episodes').rglob('*.parquet')
                for row in pq.read_table(path).to_pylist()
                if row['episode_index'] == args.episode)
    data_path = dataset / f"data/chunk-{meta['data/chunk_index']:03d}/file-{meta['data/file_index']:03d}.parquet"
    merged = pq.read_table(data_path, filters=[('episode_index', '=', args.episode)],
                           columns=['frame_index', 'observation.state']).to_pylist()
    merged.sort(key=lambda r: r['frame_index'])
    matches = []
    for frame_path in sorted((ROOT / batch['source_frames'] / 'data').glob('episode_*/frames.jsonl')):
        with frame_path.open() as handle:
            first = json.loads(handle.readline())
        if np.allclose(first['observation.state'], merged[0]['observation.state'], atol=1e-6, rtol=0):
            frames = read_jsonl(frame_path)
            if len(frames) == len(merged):
                matches.append((frame_path, frames))
    if len(matches) != 1:
        raise ValueError(f'Ambiguous source mapping: {len(matches)} matches')
    frame_path, frames = matches[0]
    source_episode = int(frames[0]['episode_index'])
    sidecars = list((ROOT / batch['tactile_raw']).glob(f'episode_{source_episode:06d}_*'))
    if len(sidecars) != 1:
        raise ValueError('Ambiguous tactile recording')
    sidecar = sidecars[0]
    manifest = json.loads((sidecar / 'manifest.json').read_text())
    offset = int(manifest['nuc_minus_pc_offset_ns'])
    origin = float(frames[0]['timestamp'])
    frame_times = np.array([f['timestamp'] - origin for f in frames])
    fps = 30
    count = int(round(frame_times[-1] * fps)) + 1
    timeline = np.arange(count) / fps
    frame_indices = np.maximum(0, np.searchsorted(frame_times, timeline, side='right') - 1)
    duration = count / fps
    report = dict(episode=args.episode, source_episode=source_episode,
                  source_frames=str(frame_path), tactile_source=str(sidecar),
                  nuc_minus_pc_offset_ns=offset, clock_rtt_ms=manifest.get('clock_rtt_ns', 0) / 1e6,
                  fps=fps, frames=count, duration_seconds=duration,
                  force_definition='norm(force_xyz - first-second median baseline), no smoothing',
                  mean_definition='mean of most recent valid left/right force at each video frame; max age 150 ms',
                  sensors={}, cameras={})
    series = []
    for side in ('left_tip', 'right_tip'):
        raw = read_jsonl(sidecar / side / 'samples.jsonl')
        samples = []
        for row in raw:
            names = row.get('feature_names') or manifest['feature_names']
            mapping = dict(zip(names, row['values']))
            xyz = [mapping[k] for k in ('force_fx', 'force_fy', 'force_fz')]
            if mapping['sample_valid'] > 0.5 and mapping['force_valid'] > 0.5 and np.isfinite(xyz).all():
                samples.append(((int(row['pc_time_ns']) + offset) / 1e9 - origin, *xyz))
        samples = np.asarray(sorted(samples))
        base = samples[(samples[:, 0] >= 0) & (samples[:, 0] < 1)]
        if len(base) < 8:
            raise ValueError(f'Insufficient baseline samples for {side}')
        baseline = np.median(base[:, 1:], axis=0)
        force = np.linalg.norm(samples[:, 1:] - baseline, axis=1)
        times = samples[:, 0]
        selected = (times >= 0) & (times <= duration)
        baseline_p95 = float(np.quantile(np.linalg.norm(base[:, 1:] - baseline, axis=1), .95))
        report['sensors'][side] = dict(baseline_xyz=baseline.tolist(),
            baseline_p95_N=baseline_p95, valid_samples=int(selected.sum()),
            median_rate_hz=float(1 / np.median(np.diff(times))),
            max_gap_s=float(np.max(np.diff(times))), max_force_N=float(force[selected].max()))
        with (args.output / f'{side}_measured_force.csv').open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['time_s', 'force_N'])
            writer.writerows(zip(times[selected], force[selected]))
        series.append((times[selected], force[selected]))
    report['baseline_gripper_min'] = min(f['observation.state'][-1] for f in frames if f['timestamp'] - origin < 1)
    aligned = [latest_values(t, f, timeline) for t, f in series]
    average = (aligned[0] + aligned[1]) / 2
    with (args.output / 'aligned_force_30fps.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_s', 'left_N', 'right_N', 'bilateral_mean_N', 'source_frame_index'])
        writer.writerows(zip(timeline, *aligned, average, frame_indices))
    series.append((timeline, average))
    camera_frames = []
    for key, label in CAMERAS:
        prefix = f'videos/observation.images.{key}'
        path = dataset / f"{prefix}/chunk-{meta[prefix + '/chunk_index']:03d}/file-{meta[prefix + '/file_index']:03d}.mp4"
        start = meta[prefix + '/from_timestamp']
        camera_frames.append(decode_episode(path, start, len(frames), fps))
        report['cameras'][label] = dict(path=str(path), episode_start_pts=start)

    fig = Figure(figsize=(19.2, 5.4), dpi=100, facecolor='white')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([.062, .145, .916, .64])
    ax.set_xlim(0, duration)
    maximum = max(float(np.nanmax(f)) for _, f in series)
    ax.set_ylim(0, np.ceil(maximum / 10) * 10 + 5)
    ax.set_xlabel('Time (s)', fontsize=19)
    ax.set_ylabel('Contact force (N)', fontsize=19)
    ax.tick_params(labelsize=16)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dbe1e6', linewidth=1)
    labels = ('Left', 'Right', 'Bilateral mean')
    lines, dots = [], []
    for label, color in zip(labels, COLORS):
        line, = ax.plot([], [], color=color, lw=2.8 if label == 'Bilateral mean' else 2.0, label=label)
        dot, = ax.plot([], [], 'o', color=color, ms=7)
        lines.append(line)
        dots.append(dot)
    cursor = ax.axvline(0, color='#7e8a96', lw=1.5, linestyle='--')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=3, frameon=False, fontsize=18)
    readout = fig.text(.975, .88, '', ha='right', fontsize=18, color='#202c39')
    output = args.output / f'usb_episode_{args.episode:03d}_task_and_tactile.mp4'
    with av.open(str(output), 'w', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = 1920, 1080
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '19', 'preset': 'fast'}
        stream.codec_context.thread_count = 4
        for index, now in enumerate(timeline):
            image = np.full((1080, 1920, 3), 255, np.uint8)
            cv2.putText(image, f'USB insertion | Episode {args.episode:03d}', (24, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (32, 44, 57), 2, cv2.LINE_AA)
            cv2.putText(image, f'{now:05.2f} / {duration:.2f} s   |   1x', (1500, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .85, (32, 44, 57), 2, cv2.LINE_AA)
            for cam, (_, label) in enumerate(CAMERAS):
                tile = camera_frames[cam][int(frame_indices[index])].copy()
                cv2.rectangle(tile, (10, 10), (230, 48), (255, 255, 255), -1)
                cv2.putText(tile, label, (20, 37), cv2.FONT_HERSHEY_SIMPLEX, .7,
                            (32, 44, 57), 2, cv2.LINE_AA)
                image[60:540, cam * 640:(cam + 1) * 640] = tile
            for line, dot, (times, force) in zip(lines, dots, series):
                mask = times <= now
                tx, fy = times[mask], force[mask]
                # Break rather than join gaps with missing sensor samples.
                if len(tx):
                    fy = fy.copy()
                    fy[np.r_[False, np.diff(tx) > .15]] = np.nan
                line.set_data(tx, fy)
                if len(tx) and now - tx[-1] <= .15:
                    dot.set_data(tx[-1:], fy[-1:])
                else:
                    dot.set_data([], [])
            cursor.set_xdata([now, now])
            vals = [a[index] for a in aligned] + [average[index]]
            readout.set_text('   '.join(f'{label}: {value:.1f} N' if np.isfinite(value)
                                       else f'{label}: --' for label, value in zip(labels, vals)))
            canvas.draw()
            image[540:] = np.asarray(canvas.buffer_rgba())[:, :, :3]
            video_frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            for packet in stream.encode(video_frame):
                container.mux(packet)
            if index == min(count - 1, 240):
                cv2.imwrite(str(args.output / 'preview.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if index % 150 == 0:
                print(f'Rendered {index}/{count}', flush=True)
        for packet in stream.encode():
            container.mux(packet)
    report['output_video'] = str(output)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
