"""Render an ink episode with recorded overview, sensor images and force traces."""

import csv
import json
from pathlib import Path

import av
import cv2
import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = Path('/path/to/VLA_RL_dataset/franka_ink_beaker_tactile_287_20260807')
OUT = Path('/path/to/tactile_force_envelope/outputs/videos/ink_episode18_tactile')


def frames(path, start, count):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(start / stream.time_base), stream=stream)
        n = 0
        for frame in container.decode(stream):
            if frame.time < start - 1 / 120:
                continue
            if abs(frame.time - start - n / 30) > 1 / 60:
                raise ValueError('Video timestamp mismatch')
            yield frame.to_ndarray(format='rgb24')
            n += 1
            if n == count:
                return
        raise ValueError('Incomplete source video')


def tile(rgb, width, height, label):
    out = np.full((height, width, 3), 245, np.uint8)
    scale = min(width / rgb.shape[1], (height - 32) / rgb.shape[0])
    w, h = round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)
    x, y = (width - w) // 2, 32 + (height - 32 - h) // 2
    out[y:y+h, x:x+w] = cv2.resize(rgb, (w, h))
    cv2.putText(out, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (25, 35, 45), 1, cv2.LINE_AA)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = next(r for p in (ROOT / 'meta/episodes').rglob('*.parquet')
                for r in pq.read_table(p).to_pylist() if r['episode_index'] == 18)
    rows = pq.read_table(ROOT / f"data/chunk-{meta['data/chunk_index']:03d}/file-{meta['data/file_index']:03d}.parquet",
                         filters=[('episode_index', '=', 18)]).to_pylist()
    rows.sort(key=lambda r: r['frame_index'])
    t = np.array([r['timestamp'] for r in rows])
    assert np.allclose(t, np.arange(len(t)) / 30, atol=1e-5)
    report = dict(dataset=str(ROOT), episode=18, frames=len(t), fps=30,
                  duration_s=len(t)/30, force_definition='Recorded norm(Fx,Fy,Fz), no additional baseline subtraction or smoothing; not Top2Mean',
                  alignment='Dataset frame timestamps; latest recorded tactile samples, invalid or age>150ms masked. Sensor videos use matching episode frame indices.', sensors={})
    forces = []
    for side in ('left', 'right'):
        key = f'observation.tactile.xense_{side}_tip'
        a = np.asarray([r[key] for r in rows])
        ages = np.asarray([r[key + '.age'] for r in rows])
        valid = (a[:, 0] > .5) & (a[:, 1] > .5) & np.isfinite(a[:, 2:5]).all(axis=1) & (ages >= 0) & (ages <= .15)
        f = np.where(valid, np.linalg.norm(a[:, 2:5], axis=1), np.nan)
        forces.append(f)
        report['sensors'][side] = dict(valid_frames=int(valid.sum()), max_force_N=float(np.nanmax(f)),
            unique_sample_timestamps=len(set(r[key+'.stamp'] for r in rows)), age_p95_ms=float(np.quantile(ages,.95)*1000))
    values = np.column_stack([*forces, (forces[0] + forces[1]) / 2])
    peak = int(np.nanargmax(values[:, 2]))
    with (OUT / 'force_timeseries.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'left_N', 'right_N', 'bilateral_mean_N'])
        w.writerows(zip(t, *values.T))
    cameras = ['camera_camera_2_color_image_raw', 'xense_left_tip_image_raw', 'xense_right_tip_image_raw']
    iterators = []
    for camera in cameras:
        key = 'videos/observation.images.' + camera
        path = ROOT / key / f"chunk-{meta[key+'/chunk_index']:03d}/file-{meta[key+'/file_index']:03d}.mp4"
        iterators.append(frames(path, meta[key+'/from_timestamp'], len(t)))
    fig = Figure(figsize=(12.8,4.2), dpi=100, facecolor='white')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([.08,.17,.90,.60])
    ax.set(xlim=(0,len(t)/30), ylim=(0,np.ceil(np.nanmax(values)/5)*5), xlabel='Time (s)', ylabel='Measured force (N)')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.tick_params(labelsize=12)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', color='#dbe1e6')
    labels = ['Left','Right','Bilateral mean']
    colors = ['#2474bd','#df6c32','#202c39']
    lines = [ax.plot([],[],color=c,lw=2,label=l)[0] for l,c in zip(labels,colors)]
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.05),ncol=3,frameon=False,fontsize=12)
    readout = fig.text(.98,.87,'',ha='right',fontsize=12)
    cursor = ax.axvline(0,color='#9099a0',ls='--',lw=1)
    path = OUT / 'ink_episode_018_task_and_tactile.mp4'
    with av.open(str(path),'w',options={'movflags':'+faststart'}) as output:
        stream = output.add_stream('libx264',rate=30)
        stream.width, stream.height, stream.pix_fmt = 1280,1080,'yuv420p'
        stream.options = {'crf':'18','preset':'fast'}
        stream.codec_context.thread_count = 4
        for i, now in enumerate(t):
            rgb = [next(it) for it in iterators]
            image = np.full((1080,1280,3),255,np.uint8)
            cv2.putText(image,'Ink into beaker | Episode 018',(18,38),cv2.FONT_HERSHEY_SIMPLEX,.85,(25,35,45),2,cv2.LINE_AA)
            cv2.putText(image,f'{now:05.2f} / {len(t)/30:.2f} s | 1x',(955,38),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,35,45),1,cv2.LINE_AA)
            image[60:660,:800] = tile(rgb[0],800,600,'Overview')
            image[60:360,800:] = tile(rgb[1],480,300,'Left tactile image (dataset label)')
            image[360:660,800:] = tile(rgb[2],480,300,'Right tactile image (dataset label)')
            for j,line in enumerate(lines):
                line.set_data(t[:i+1],values[:i+1,j])
            cursor.set_xdata([now,now])
            readout.set_text('  '.join(f'{l}: {v:.1f} N' for l,v in zip(labels,values[i])))
            canvas.draw()
            image[660:] = np.asarray(canvas.buffer_rgba())[:,:,:3]
            for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):
                output.mux(packet)
            if i in (15,peak):
                name = 'early' if i == 15 else 'peak'
                for camera, raw in zip(cameras,rgb):
                    cv2.imwrite(str(OUT/f'{name}_{camera}.png'),cv2.cvtColor(raw,cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(OUT/f'{name}_preview.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            if i % 150 == 0:
                print(f'Rendered {i}/{len(t)}',flush=True)
        for packet in stream.encode():
            output.mux(packet)
    for it in iterators:
        it.close()
    with av.open(str(path)) as check:
        decoded = sum(1 for _ in check.decode(video=0))
    assert decoded == len(t)
    report.update(output=str(path), verified_decoded_frames=decoded, peak_time_s=float(t[peak]))
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
