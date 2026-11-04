"""Illustrative held-out ink episode: cameras, calibrated load band and local reward."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from add_ink_reward_video import (
    ROOT, av, cv2, np, torch, pq, DataLoader, Figure, FigureCanvasAgg,
    load_config, TransformerEnvelopeExpert, TransformerWindowDataset,
    _gates_for_rows, _predict,
)
from render_ink_tactile_video import frames, tile


def evaluate(source_episode, out, task='ink', feature_override=None):
    config_name = 'pi05_jax_zrl_pastry_A298.yaml' if task == 'pastry' else 'pi05_jax_zrl_ink_beaker.yaml'
    config = load_config(ROOT/'configs/experiments'/config_name)
    workspace = Path(config['data']['output_dir'])
    folder = workspace/'sustained_contact_seed_44/stable_evaluation'
    calibration_path = folder/'bilateral_sigma_calibration.json'
    cal = json.loads(calibration_path.read_text())
    checkpoint_path = folder/'stable_contact_calibrated_checkpoint.pt'
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == cal['checkpoint_sha256']
    ckpt = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    condition = torch.load(workspace/'condition_tokens.pt',map_location='cpu',weights_only=False)
    history = torch.load(workspace/'causal_tactile_history.pt',map_location='cpu',weights_only=False)
    digest = hashlib.sha256((workspace/'windows.jsonl').read_bytes()).hexdigest()
    assert condition['windows_sha256'] == history['windows_sha256'] == digest
    if feature_override:
        extra = torch.load(feature_override, map_location='cpu', weights_only=False)
        assert extra['windows_sha256'] == digest
        assert extra['checkpoint'] == config['vision']['checkpoint']
        for index, token in extra['tokens'].items():
            if not condition['condition_mask'][index].any():
                condition['condition_tokens'][index] = token
                condition['condition_mask'][index] = True
    rows = [json.loads(l) for l in (workspace/'windows.jsonl').open()]
    rows = sorted([r for r in rows if r['episode_index']==source_episode],key=lambda r:r['start_timestamp'])
    assert rows and all(r['split']=='test' for r in rows)
    removed = config['vision']['removed_source_episode_ids']
    assert source_episode not in removed
    episode = source_episode-sum(e<source_episode for e in removed)
    dataset_root = Path(config['vision']['local_v3_dataset'])
    meta = next(r for p in (dataset_root/'meta/episodes').rglob('*.parquet') for r in pq.read_table(p).to_pylist() if r['episode_index']==episode)
    data_path = dataset_root/f"data/chunk-{meta['data/chunk_index']:03d}/file-{meta['data/file_index']:03d}.parquet"
    video_rows = pq.read_table(data_path,filters=[('episode_index','=',episode)]).to_pylist()
    states = {r['frame_index']:r['observation.state'] for r in video_rows}
    assert all(np.allclose(r['state'],states[r['anchor_frame_index']],atol=1e-5) for r in rows)
    gates = _gates_for_rows(rows,history['history'],history['mask'],config).all(axis=1)
    cache_ok = [bool(condition['condition_mask'][r['window_index']].any()) for r in rows]
    selected = [r for r,valid in zip(rows,cache_ok) if valid]
    d=ckpt['dimensions']
    model=TransformerEnvelopeExpert(condition_dim=d['condition_dim'],state_dim=d['state_dim'],
        action_dim=d['action_dim'],action_steps=d['action_steps'],config=ckpt['model_config'],
        history_dim=d.get('history_dim',0),history_steps=d.get('history_steps',0))
    model.load_state_dict(ckpt['model_state_dict'])
    ds=TransformerWindowDataset(selected,condition['condition_tokens'],condition['condition_mask'],
        ckpt['normalization'],config,history['history'],history['mask'])
    pred=_predict(model,DataLoader(ds,batch_size=64),torch.device('cpu'),ckpt['normalization'])
    positions={r['window_index']:i for i,r in enumerate(selected)}
    records=[]
    for row,gate,valid in zip(rows,gates,cache_ok):
        sides=row['tactile']['per_sensor']
        y=.5*sum(sides[s]['filtered_top2_mean'] for s in ('left_tip','right_tip'))
        peak=max(sides[s]['raw_max'] for s in ('left_tip','right_tip'))
        peak_cost=float(peak>cal['raw_peak_threshold']['threshold_N'])
        mu=scale=lower=upper=low=high=float('nan')
        soft=0. if not gate else float('nan')
        if valid:
            i=positions[row['window_index']]
            mu=float(.5*(pred['left']['mu'][i]+pred['right']['mu'][i]))
            scale=max(1e-6,float(.5*(pred['left']['sigma'][i]+pred['right']['sigma'][i])))
            lower=mu+cal['lower_alpha']*scale;upper=mu+cal['upper_alpha']*scale
            low=max(0.,lower-y);high=max(0.,y-upper)
            soft=float(gate)*(np.clip(low/2.,0,1)**2+np.clip(high/2.,0,1)**2)
        reward=-float(min(1.,.6*soft+.4*peak_cost)) if np.isfinite(soft) else float('nan')
        start=row['anchor_frame_index']/30
        records.append(dict(window_index=row['window_index'],start_s=start,end_s=start+.3,
            gate=bool(gate),cached=bool(valid),observed_avg_N=y,predicted_avg_N=mu,
            lower_N=lower,upper_N=upper,scale_N=scale,low_deviation_N=low,high_deviation_N=high,
            soft_cost=float(soft),peak_cost=peak_cost,reward=reward))
    assert not any(r['gate'] and not r['cached'] for r in records), 'Active prediction missing'
    with (out/'windows.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary=dict(source_episode=source_episode,dataset_episode=episode,split='test',dataset=str(dataset_root),
        task=task, feature_override=str(feature_override) if feature_override else None,
        checkpoint=str(checkpoint_path),calibration=str(calibration_path),duration_s=meta['length']/30,
        windows=len(records),stable_windows=int(sum(gates)),
        penalized_windows=sum(r['reward']<0 for r in records),minimum_reward=min(r['reward'] for r in records),
        peak_penalty_windows=sum(r['peak_cost']>0 for r in records),
        calibration_unchanged=True,force_scale_N=2.,soft_weight=.6,peak_weight=.4,
        alpha05=cal['lower_alpha'],alpha95=cal['upper_alpha'],raw_peak_limit_N=cal['raw_peak_threshold']['threshold_N'],
        observed='Average of left/right zero-biased median-filtered Top2Mean over each 300ms window, not instantaneous force.',
        reward='-min(1, 0.6 * gate * (clip(low/2,0,1)^2 + clip(high/2,0,1)^2) + 0.4 * peak)',
        limitations=['Selected for nonzero reward illustration, not an unbiased episode-performance estimate.',
        'Offline demonstrated actions and per-anchor cached z_rl; not a recorded online rollout or deployed dual-rate replay.',
        f"Offline causal 3-of-4 per-side gate (>{config['stable_contact']['tau_on']}N in 200ms); no online hysteresis simulation.",
        'Local 300ms reward before E18 averaging and episode budget; no task reward.',
        'Predictions absent outside cached domain are not filled. Missing build windows remain gaps.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return dataset_root,meta,records,summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-episode',type=int,default=53)
    parser.add_argument('--task', choices=['ink','pastry'], default='ink')
    parser.add_argument('--feature-override', type=Path)
    parser.add_argument('--show-ungated-predictions', action='store_true', help='Display real cached predictions outside gate; reward gate remains unchanged')
    parser.add_argument('--continuous',action='store_true',help='Connect adjacent valid window centers for display only')
    parser.add_argument('--bridge-display-gaps',action='store_true',help='Presentation-only interpolation across internal gaps; preserve raw CSV and rewards')
    parser.add_argument('--stable-only',action='store_true',help='Show longest contiguous valid contact segment, explicitly cropped')
    parser.add_argument('--zero-time',action='store_true',help='Label video and plot time relative to clip start')
    parser.add_argument('--full-presentation',action='store_true',help='Whole task with explicit inactive envelope regions and dotted reward-gap connectors')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.full_presentation:
        args.bridge_display_gaps=True
        args.zero_time=True
        if args.stable_only:parser.error('Full presentation cannot crop to stable-only')
    if args.bridge_display_gaps:
        args.continuous=True
    out=ROOT/f'outputs/videos/{args.task}_source{args.source_episode}_envelope_reward'
    if args.show_ungated_predictions:
        out=out.with_name(out.name+'_early_predictions')
    if args.continuous:
        out=out.with_name(out.name+'_continuous')
    if args.bridge_display_gaps:
        out=out.with_name(out.name+'_presentation')
    if args.stable_only:
        out=out.with_name(out.name+'_stable_clip')
    if args.zero_time:
        out=out.with_name(out.name+'_t0')
    if args.full_presentation:
        out=out.with_name(out.name+'_full_task')
    out.mkdir(parents=True,exist_ok=True)
    root,meta,records,summary=evaluate(args.source_episode,out,args.task,args.feature_override)
    print(json.dumps(summary,indent=2),flush=True)
    duration=meta['length']/30
    first_frame,last_frame=0,meta['length']
    if args.stable_only:
        runs=[];run=[]
        for r in records:
            if not (r['gate'] and r['cached']):
                if run:runs.append(run)
                run=[]
                continue
            if run and r['start_s']-run[-1]['start_s']>.38:
                runs.append(run);run=[]
            run.append(r)
        if run:runs.append(run)
        records=max(runs,key=len)
        first_frame=round(records[0]['start_s']*30)
        last_frame=min(meta['length'],round(records[-1]['end_s']*30)+1)
        errors=np.array([r['predicted_avg_N']-r['observed_avg_N'] for r in records])
        summary['clip']=dict(start_s=first_frame/30,end_s=(last_frame-1)/30,
            frames=last_frame-first_frame,windows=len(records),all_gates_and_predictions_valid=True,
            mae_N=float(np.abs(errors).mean()),rmse_N=float(np.sqrt(np.mean(errors**2))),
            coverage=float(np.mean([r['lower_N']<=r['observed_avg_N']<=r['upper_N'] for r in records])),
            selection='Longest continuous valid stable-contact run; selected illustration, not full-episode evaluation')
        with (out/'displayed_windows.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    fig=Figure(figsize=(12.8,7.4),dpi=100,facecolor='white');canvas=FigureCanvasAgg(fig)
    ax=fig.add_axes([.085,.48,.89,.37]);reward_ax=fig.add_axes([.085,.10,.89,.23])
    for a in (ax,reward_ax):
        a.set_xlim(first_frame/30,last_frame/30);a.spines[['top','right']].set_visible(False)
        a.tick_params(labelsize=12);a.grid(axis='y',color='#dbe1e6')
    ax.set_ylabel('Bilateral average force (N)',fontsize=14)
    reward_ax.set_ylabel('Tactile reward',fontsize=14);reward_ax.set_xlabel('Time (s)',fontsize=14)
    if args.zero_time:
        origin=first_frame/30
        clip_duration=(last_frame-first_frame)/30
        ticks=np.arange(0,clip_duration,1. if clip_duration<=6 else 2.)
        for a in (ax,reward_ax):
            a.set_xticks(origin+ticks,labels=[f'{t:g}' for t in ticks])
        summary['display_time_origin_episode_s']=origin
        summary['display_time_definition']='Time from clip start; original episode timestamps retained in CSV and clip metadata'
    ax.set_ylim(min(0.,min(r['lower_N'] for r in records if np.isfinite(r['lower_N'])))-.5,
                max(max(r['observed_avg_N'] for r in records),
                    max(r['upper_N'] for r in records if np.isfinite(r['upper_N'])))*1.1)
    reward_extent=max(.005,abs(summary['minimum_reward'])*1.2)
    reward_ax.set_ylim(-reward_extent,reward_extent*.08)
    # Draw separate window segments: no interpolation across missing windows or predictions.
    xs=[];obs=[];means=[];lo=[];hi=[];rr=[];ends=[]
    for r in records:
        xs.extend([r['start_s'],r['end_s'],np.nan]);ends.extend([r['end_s']]*3)
        for array,key in ((obs,'observed_avg_N'),(means,'predicted_avg_N'),(lo,'lower_N'),(hi,'upper_N'),(rr,'reward')):
            value=r[key]
            if key in ('predicted_avg_N','lower_N','upper_N') and not (r['gate'] or args.show_ungated_predictions):value=np.nan
            array.extend([value,value,np.nan])
    xs,obs,means,lo,hi,rr,ends=map(np.array,(xs,obs,means,lo,hi,rr,ends))
    force_x,force_ends=xs,ends
    if args.continuous:
        # Linear connections cannot overshoot endpoint values or invert ordered bounds.
        points=[]
        for i,r in enumerate(records):
            if i and r['start_s']-records[i-1]['start_s']>.38:
                points.append([np.nan]*6)
            active=(r['gate'] or args.show_ungated_predictions) and r['cached']
            points.append([.5*(r['start_s']+r['end_s']),r['end_s'],r['observed_avg_N'],
                           r['predicted_avg_N'] if active else np.nan,
                           r['lower_N'] if active else np.nan,r['upper_N'] if active else np.nan])
        force_x,force_ends,obs,means,lo,hi=np.array(points).T
        summary['display_interpolation']='Linear between adjacent valid 300ms window centers; break at anchor gap >380ms. No prediction across gate-off windows. Reveal only completed windows. Reward unchanged.'
    if args.bridge_display_gaps:
        # Retrospective display only: bridge internal gaps without extrapolating endpoints.
        grid=np.unique(np.r_[force_x[np.isfinite(force_x)],np.arange(.15,duration,1/30)])
        arrays=[]
        for values in (obs,means,lo,hi):
            valid=np.isfinite(force_x)&np.isfinite(values)
            arrays.append(np.interp(grid,force_x[valid],values[valid],left=np.nan,right=np.nan))
        force_x=grid
        obs,means,lo,hi=arrays
        force_ends=grid
        summary['display_interpolation']='Retrospective linear interpolation over internal gaps, including gate-off prediction gaps. Uses bracketing endpoints; not causal inference. No endpoint extrapolation. Raw values and rewards unchanged.'
        summary['limitations'][-1]='Presentation fills internal force/prediction gaps, including gate-off intervals, by retrospective interpolation; these are not additional measurements or valid model predictions. Reward gaps remain unchanged.'
    band=ax.fill_between(force_x,lo,hi,color='#9acdc7',alpha=.55,label='Envelope [L05, U95]')
    mean_line,=ax.plot(force_x,means,'--',color='#00867d',lw=2,label='Predicted mean')
    obs_line,=ax.plot(force_x,obs,color='#26323c',lw=2,label='Observed mean (300 ms)')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=3,fontsize=12,frameon=False)
    reward_line,=reward_ax.plot(xs,rr,color='#af3557',lw=2.4)
    reward_fill=reward_ax.fill_between(xs,rr,0,color='#af3557',alpha=.20)
    connector=None
    if args.full_presentation:
        # Connect presentation gaps only with a visually distinct dotted stroke.
        gap_x=[];gap_y=[]
        for previous,current in zip(records,records[1:]):
            if current['start_s']>previous['end_s']:
                gap_x.extend([previous['end_s'],current['start_s'],np.nan])
                gap_y.extend([previous['reward'],current['reward'],np.nan])
        gap_x=np.asarray(gap_x);gap_y=np.asarray(gap_y)
        connector,=reward_ax.plot(gap_x,gap_y,':',color='#af3557',lw=1.4)
        active=[r for r in records if r['gate'] and r['cached']]
        for left,right in ((0,active[0]['start_s']),(active[-1]['end_s'],duration)):
            if right>left:
                ax.axvspan(left,right,color='#eef0f2',zorder=-2)
                if right-left>1.5:
                    label='Reward gate off' if args.show_ungated_predictions else 'Envelope inactive'
                    ax.text((left+right)/2,.96,label,transform=ax.get_xaxis_transform(),
                            ha='center',va='top',fontsize=10,color='#69747c')
        summary['reward_display']='Original window rewards retained as solid steps; dotted internal-gap connections are illustrative, not computed rewards.'
        summary['full_task']=True
    fig.text(.085,.96,'Measured load, prediction and calibrated envelope',fontsize=16)
    if args.bridge_display_gaps:
        note='Presentation: gaps linearly interpolated; reward unchanged'
        if args.full_presentation:note='Display interpolation only; dotted reward gaps are not evaluated windows'
        fig.text(.085,.415,note,fontsize=11,color='#606b73')
    fig.text(.085,.365,'Offline tactile reward per 300 ms window',fontsize=14)
    if args.show_ungated_predictions:
        summary['ungated_prediction_notice']='Actual Expert predictions outside stable-contact domain are exploratory, not calibrated-coverage validated. Reward gate unchanged.'
        fig.text(.085,.915,'Outside stable contact: exploratory predictions, not coverage-validated',fontsize=10,color='#865e16')
    status=fig.text(.975,.365,'',ha='right',fontsize=12)
    cursors=[a.axvline(0,color='#9099a0',ls='--',lw=1) for a in (ax,reward_ax)]
    cameras=['camera_camera_2_color_image_raw','xense_left_tip_image_raw','xense_right_tip_image_raw']
    streams=[]
    for key in cameras:
        prefix='videos/observation.images.'+key
        path=root/prefix/f"chunk-{meta[prefix+'/chunk_index']:03d}/file-{meta[prefix+'/file_index']:03d}.mp4"
        streams.append(frames(path,meta[prefix+'/from_timestamp']+first_frame/30,last_frame-first_frame))
    selected=min(records,key=lambda r:r['reward']);preview_index=min(meta['length']-1,int(np.ceil(selected['end_s']*30))+2)
    destination=out/f'{args.task}_task_envelope_reward.mp4';preview_rgb=None
    with av.open(str(destination),'w',options={'movflags':'+faststart'}) as output:
        stream=output.add_stream('libx264',rate=30);stream.width=1280;stream.height=1400;stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'fast'};stream.codec_context.thread_count=4
        for i in range(first_frame,last_frame):
            now=i/30;rgb=[next(s) for s in streams]
            image=np.full((1400,1280,3),255,np.uint8)
            title='Pastry into blue box' if args.task=='pastry' else 'Ink into beaker'
            cv2.putText(image,f"{title} | Episode {meta['episode_index']:03d}",(18,38),cv2.FONT_HERSHEY_SIMPLEX,.85,(25,35,45),2,cv2.LINE_AA)
            time_label=f'{now:05.2f} / {duration:.2f} s | 1x'
            if args.stable_only:time_label=f'Episode t={now:05.2f}s | Clip | 1x'
            if args.zero_time:time_label=f'{now-first_frame/30:05.2f} / {(last_frame-first_frame)/30:.2f} s | 1x'
            cv2.putText(image,time_label,(900,38),cv2.FONT_HERSHEY_SIMPLEX,.6,(25,35,45),1,cv2.LINE_AA)
            image[60:660,:800]=tile(rgb[0],800,600,'Overview')
            image[60:360,800:]=tile(rgb[1],480,300,'Left tactile image')
            image[360:660,800:]=tile(rgb[2],480,300,'Right tactile image')
            # Reveal completed observation/reward windows only, never future measured force.
            visible=np.where(ends<=now,xs,np.nan)
            force_visible=np.where(force_ends<=now,force_x,np.nan)
            if connector is not None:connector.set_data(np.where(gap_x<=now,gap_x,np.nan),gap_y)
            obs_line.set_data(force_visible,obs);mean_line.set_data(force_visible,means);reward_line.set_data(visible,rr)
            band.remove();band=ax.fill_between(force_visible,lo,hi,color='#9acdc7',alpha=.55)
            reward_fill.remove();reward_fill=reward_ax.fill_between(visible,rr,0,color='#af3557',alpha=.20)
            for c in cursors:c.set_xdata([now,now])
            finished=[r for r in records if r['end_s']<=now]
            status.set_text(f"Last reward: {finished[-1]['reward']:.3f}" if finished else 'Awaiting completed window')
            canvas.draw();image[660:]=np.asarray(canvas.buffer_rgba())[:,:,:3]
            for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):output.mux(packet)
            if i==preview_index:
                cv2.imwrite(str(out/'preview.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR));preview_rgb=image[:660].copy()
                for name,raw in zip(cameras,rgb):cv2.imwrite(str(out/f'{name}.png'),cv2.cvtColor(raw,cv2.COLOR_RGB2BGR))
            if i%200==0:print(f'Rendered {i}/{meta["length"]}',flush=True)
        for packet in stream.encode():output.mux(packet)
    for s in streams:s.close()
    obs_line.set_data(force_x,obs);mean_line.set_data(force_x,means);reward_line.set_data(xs,rr)
    if connector is not None:connector.set_data(gap_x,gap_y)
    band.remove();ax.fill_between(force_x,lo,hi,color='#9acdc7',alpha=.55)
    reward_fill.remove();reward_ax.fill_between(xs,rr,0,color='#af3557',alpha=.20)
    for c in cursors:c.set_visible(False)
    status.set_text('');canvas.draw()
    full_plot_rgb=np.asarray(canvas.buffer_rgba())[:,:,:3].copy()
    fig.savefig(out/'force_envelope_reward.png',dpi=180);fig.savefig(out/'force_envelope_reward.pdf')
    cv2.imwrite(str(out/'overview_full_curves.png'),cv2.cvtColor(np.vstack([preview_rgb,full_plot_rgb]),cv2.COLOR_RGB2BGR))
    with av.open(str(destination)) as check:count=sum(1 for _ in check.decode(video=0))
    assert count==last_frame-first_frame
    summary['verified_decoded_frames']=count;summary['preview_time_s']=preview_index/30
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(destination,flush=True)


if __name__=='__main__':main()
