"""Experiment 4, called by analyze_demo_latents.py --flow-geometry-experiment.

Keep expensive model execution separate from repeatable statistics/plotting.
Rows of J are outputs, columns are inputs; coordinates are time-major (4,7).
All primary geometry is PRE final clipping. Actual control outputs are saved
in parallel. Relative trajectory progress is a sampling proxy, not a label
for physical task phases. No training or online interaction is performed.
"""
from pathlib import Path
import csv
import hashlib
import json
import pickle

import numpy as np

NAMES = ('x', 'y', 'z', 'rx', 'ry', 'rz', 'g')
DIR_NAMES = ('vmax', 'vmin', 'random', 'low4_random', 'arm_vmax', 'arm_vmin')
VARIANTS = ('preclip', 'clipped')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_csv(path, records):
    if not records:
        raise ValueError('Empty table')
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n': 0, 'mean': None, 'std': None, 'min': None, 'max': None,
                'q05': None, 'q50': None, 'q95': None}
    return dict(n=int(x.size), mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), max=float(x.max()),
                q05=float(np.quantile(x, .05)), q50=float(np.median(x)),
                q95=float(np.quantile(x, .95)))


def spectrum(jacobian):
    """Float64 SVD of stored float32 AD matrices; not extra model precision."""
    u, sigma, vh = np.linalg.svd(jacobian.astype(np.float64), full_matrices=False)
    weights = (vh.reshape(*vh.shape[:-1], 4, 7) ** 2).sum(axis=-2)
    return u, sigma, vh, weights


def component_geometry(jacobian):
    j = jacobian.astype(np.float64)
    e = j.reshape(len(j), 4, 7, 4, 7) ** 2
    # e axes: state, output time, output component, input time, input component.
    out = np.sqrt(e.sum(axis=(1, 3, 4)) / (4 * 28))
    inp = np.sqrt(e.sum(axis=(1, 2, 3)) / 4)
    blocks = np.sqrt(e.sum(axis=(1, 3)))
    return out, inp, blocks


def activity_and_sample(dataset, data, args):
    import h5py
    demo_ids, lengths, sums, squares, counts_near_zero, actions = [], [], [], [], [], []
    meta = None
    with h5py.File(dataset, 'r') as f:
        meta = json.loads(f['data'].attrs['env_args'])
        for name in sorted(f['data'], key=lambda name: int(name[5:])):
            a = np.asarray(f['data'][name]['actions'], dtype=float)
            if a.shape[1] != 7 or not np.isfinite(a).all() or np.abs(a).max() > 1:
                raise ValueError('Expected finite seven-component normalized actions')
            demo_ids.append(int(name[5:])); lengths.append(len(a))
            sums.append(a.sum(0)); squares.append((a*a).sum(0))
            counts_near_zero.append((np.abs(a)<.05).sum(0))
            actions.append(a)
    ids = np.asarray(demo_ids); lengths = np.asarray(lengths)
    all_actions = np.concatenate(actions)
    id_to_length = dict(zip(ids, lengths))
    progress = np.array([t/max(id_to_length[d]-1, 1) for d,t in zip(data['demo_id'],data['timestep'])])
    phase = np.minimum((progress*args.geometry_phase_bins).astype(int), args.geometry_phase_bins-1)
    eligible = [d for d in ids if all(np.any((data['demo_id']==d)&(phase==p))
                                    for p in range(args.geometry_phase_bins))]
    rng = np.random.default_rng(args.seed)
    selected_ids = np.sort(rng.choice(eligible, args.geometry_num_trajectories, replace=False))
    rows = [int(rng.choice(np.flatnonzero((data['demo_id']==d)&(phase==p))))
            for d in selected_ids for p in range(args.geometry_phase_bins)]
    stored = dict(all_demo_ids=ids, all_demo_lengths=lengths, all_demo_action_sums=np.asarray(sums),
                  all_demo_action_squares=np.asarray(squares), all_demo_near_zero_counts=np.asarray(counts_near_zero),
                  all_action_quantiles=np.quantile(all_actions,[.01,.05,.25,.5,.75,.95,.99],axis=0),
                  all_action_min=all_actions.min(0),all_action_max=all_actions.max(0))
    return np.asarray(rows), progress, phase, stored, meta


def neighbor_control(data, selected_rows, progress, phase):
    """Cross-demo standardized-observation kNN, never an exact Var(a|s).

    Retain all neighbor identities/distances. Two controls: unrestricted and
    same progress quartile; k=10/20/50. Neither temporal quartiles nor this
    Euclidean observation metric guarantee matching physical task stages.
    Canonicalize robot quaternion sign for the distance metric only.
    Original observations fed to the model remain untouched.
    """
    from scipy.spatial.distance import cdist
    states = data['observations'].astype(np.float64).copy()
    states[:,3:7] *= np.where(states[:,6:7] < 0, -1., 1.)
    center = states.mean(0); scale = states.std(0)
    scale = np.where(scale > 1e-6, scale, 1.)
    features = (states-center)/scale
    distances = cdist(features[selected_rows], features)/np.sqrt(features.shape[1])
    chunks = data['action_chunks'].reshape(-1,4,7).astype(np.float64)
    B = len(selected_rows); ks = np.array([10,20,50])
    neighbors = np.empty((B,2,50),dtype=np.int64)
    nearest_dist = np.empty((B,2,50))
    variances = np.empty((B,2,len(ks),7))
    rms = np.empty_like(variances)
    random_distance_median = np.empty(B)
    for b,row in enumerate(selected_rows):
        cross_demo = data['demo_id'] != data['demo_id'][row]
        random_distance_median[b] = np.median(distances[b,cross_demo])
        for mode in range(2):
            allowed = cross_demo.copy()
            if mode:
                allowed &= phase == phase[row]
            dist = np.where(allowed,distances[b],np.inf)
            idx = np.argpartition(dist,49)[:50]
            idx = idx[np.argsort(dist[idx])]
            if not np.isfinite(dist[idx]).all():
                raise ValueError('Fewer than 50 cross-demo neighbors')
            neighbors[b,mode] = idx; nearest_dist[b,mode] = dist[idx]
            for ik,k in enumerate(ks):
                # Across-neighbor variance at each chunk time, then average time.
                values = chunks[idx[:k]]
                variances[b,mode,ik] = values.var(axis=0,ddof=1).mean(0)
                rms[b,mode,ik] = np.sqrt((values**2).mean(axis=(0,1)))
    return dict(knn_sizes=ks,knn_rows=neighbors,knn_distances=nearest_dist,
                knn_demo_ids=data['demo_id'][neighbors],knn_timesteps=data['timestep'][neighbors],
                knn_progress=progress[neighbors],knn_phase=phase[neighbors],
                knn_variance=variances,knn_rms=rms,
                random_cross_demo_distance_median=random_distance_median,
                observation_metric_mean=center,observation_metric_std=scale)


def run_experiment(args, run, config, paths, load_demos, make_functions):
    import jax
    import jax.numpy as jnp
    from ogpo.agents.modules.flow_inverse import FLOW_INVERSE_METHOD, FLOW_INVERSE_OUTPUT_DIR, inverse_time_grid

    if config['flow_steps']!=10 or config['horizon_length']!=4:
        raise ValueError('Experiment 4 currently requires 4x7 chunks and 10 flow steps')
    data,total = load_demos(args.dataset,4,0,args.seed)
    rows,progress,phase,activity,env_meta = activity_and_sample(args.dataset,data,args)
    states = data['observations'][rows]; expert = data['action_chunks'][rows]
    B,D = expert.shape
    if D!=28:
        raise ValueError('Expected 28-dimensional decoder')
    print(f'Experiment 4: Square, {total} valid chunks, {len(rows)} states = '
          f'{args.geometry_num_trajectories} trajectories x {args.geometry_phase_bins} progress bins',flush=True)
    print('Computing cross-demo neighborhood controls (k=10,20,50)',flush=True)
    neighbors = neighbor_control(data,rows,progress,phase)
    invert,_,decode,decode_raw = make_functions(config,D,args,return_decoder=True)

    def batches(params,ss,zz):
        values=[]
        for start in range(0,len(zz),args.batch_size):
            end=min(start+args.batch_size,len(zz)); pad=args.batch_size-(end-start)
            sx=np.pad(ss[start:end],((0,pad),(0,0)),mode='edge')
            zx=np.pad(zz[start:end],((0,pad),(0,0)),mode='edge')
            values.append(np.asarray(decode_raw(params,sx,zx))[:end-start])
        result=np.concatenate(values)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite decoder output')
        return result

    @jax.jit
    def jacobian_batch(params,ss,zz):
        def local(state,z):
            def output(latent):
                raw=decode_raw(params,state[None],latent[None])[0]
                return jnp.stack((raw,jnp.clip(raw,-1,1)))
            return jax.jacfwd(output)(z)
        return jax.vmap(local)(ss,zz)

    radii=np.array([.001,.01,.1,.25,.5,1.,2.],dtype=np.float32)
    signs=np.array([-1.,1.],dtype=np.float32)
    arm=np.flatnonzero(np.arange(D)%7!=6)
    for path in (paths if args.steps else paths[-1:]):
        step=int(path.stem.split('_')[-1])
        out=args.output/run.name/FLOW_INVERSE_OUTPUT_DIR/f'flow_geometry_step_{step}_seed_{args.seed}'
        out.mkdir(parents=True,exist_ok=True)
        with path.open('rb') as f:
            payload=pickle.load(f)
        params=payload['agent']['bc_flow_network']['params']['modules_actor_bc_flow'];del payload
        z0,baseline,errors=[np.asarray(x) for x in invert(params,states,expert)]
        raw_base=np.asarray(decode_raw(params,states,z0))
        np.testing.assert_allclose(baseline,np.clip(raw_base,-1,1),atol=3e-6,rtol=2e-5)
        print('Computing automatic-differentiation Jacobians',flush=True)
        jac=[]
        for start in range(0,B,4):
            end=min(start+4,B);pad=4-(end-start)
            ss=np.pad(states[start:end],((0,pad),(0,0)),mode='edge')
            zz=np.pad(z0[start:end],((0,pad),(0,0)),mode='edge')
            jac.append(np.asarray(jacobian_batch(params,ss,zz))[:end-start])
            if start%20==0 or end==B:print(f'  Jacobians {end}/{B}',flush=True)
        jac=np.concatenate(jac)
        if not np.isfinite(jac).all():raise ValueError('Nonfinite Jacobian')
        svds=[spectrum(jac[:,v]) for v in range(2)]
        u,sigma,vh,weights=svds[0]
        arm_jac=jac[:,0][:,arm][:,:,arm].astype(np.float64)
        arm_u,arm_sigma,arm_vh=np.linalg.svd(arm_jac)
        rng=np.random.default_rng(args.seed+400)
        random=rng.normal(size=(B,D));random/=np.linalg.norm(random,axis=1,keepdims=True)
        lowcoeff=rng.normal(size=(B,4));lowcoeff/=np.linalg.norm(lowcoeff,axis=1,keepdims=True)
        low=np.einsum('bi,bij->bj',lowcoeff,vh[:,-4:])
        arm_high=np.zeros((B,D));arm_low=np.zeros((B,D))
        arm_high[:,arm]=arm_vh[:,0];arm_low[:,arm]=arm_vh[:,-1]
        directions=np.stack((vh[:,0],vh[:,-1],random,low,arm_high,arm_low),axis=1).astype(np.float32)
        # Both signs remove the arbitrary sign of singular vectors from conclusions.
        perturbed=z0[:,None,None,None,:]+directions[:,:,None,None,:]*signs[None,None,:,None,None]*radii[None,None,None,:,None]
        ss=np.broadcast_to(states[:,None,None,None,:],(*perturbed.shape[:-1],states.shape[1])).reshape(-1,states.shape[1])
        print(f'4.4: decoding {np.prod(perturbed.shape[:-1]):,} finite interventions (including +/- and small-radius checks)',flush=True)
        raw_perturbed=batches(params,ss,perturbed.reshape(-1,D)).reshape(perturbed.shape)
        # A conventional finite-difference check of the COMPLETE J at three states.
        fd_ids=np.unique(np.linspace(0,B-1,3,dtype=int));fd_eps=np.array([.001,.01],dtype=np.float32)
        fd_z=z0[fd_ids,None,None,None,:]+fd_eps[None,:,None,None,None]*signs[None,None,:,None,None]*np.eye(D,dtype=np.float32)[None,None,None,:,:]
        fd_s=np.broadcast_to(states[fd_ids,None,None,None,:],(*fd_z.shape[:-1],states.shape[1])).reshape(-1,states.shape[1])
        fd_raw=batches(params,fd_s,fd_z.reshape(-1,D)).reshape(fd_z.shape)
        prior=rng.normal(size=(args.geometry_prior_samples,D)).astype(np.float32)
        prior_s=np.repeat(states,len(prior),axis=0)
        prior_z=np.tile(prior,(B,1))
        print(f'Conditional decoder variance: decoding {len(prior_s):,} shared N(0,I) samples at fixed states',flush=True)
        prior_raw=batches(params,prior_s,prior_z).reshape(B,len(prior),D)
        # Check against existing corrected inverse cache without rerunning all demos.
        canonical=args.output/run.name/f'all_demo_inverse_step_{step}_time_shift_1'
        cache_delta=None
        if (canonical/'summary.json').exists():
            cm=json.loads((canonical/'summary.json').read_text())
            if cm['checkpoint_sha256']==digest(path) and cm['dataset_sha256']==digest(args.dataset):
                with np.load(canonical/'all_latents.npz') as cache:
                    np.testing.assert_allclose(cache['z_demo'][rows],z0,rtol=2e-5,atol=3e-5)
                    cache_delta=float(np.max(abs(cache['z_demo'][rows]-z0)))
        arrays=dict(source_rows=rows,demo_id=data['demo_id'][rows],timestep=data['timestep'][rows],
                    progress=progress[rows],phase=phase[rows],states=states,expert_actions=expert,
                    z0=z0,baseline_actions=baseline,raw_baseline_actions=raw_base,reconstruction_max_abs=errors,
                    jacobians=jac,svd_u=np.stack([v[0] for v in svds],axis=1),
                    singular_values=np.stack([v[1] for v in svds],axis=1),
                    svd_vh=np.stack([v[2] for v in svds],axis=1),
                    singular_component_weights=np.stack([v[3] for v in svds],axis=1),
                    arm_indices=arm,arm_singular_values=arm_sigma,arm_svd_vh=arm_vh,
                    directions=directions,direction_names=np.array(DIR_NAMES),radii=radii,signs=signs,
                    perturbed_z=perturbed,raw_perturbed_actions=raw_perturbed,
                    perturbed_actions=np.clip(raw_perturbed,-1,1),prior_z=prior,prior_raw_actions=prior_raw,
                    prior_actions=np.clip(prior_raw,-1,1),fd_state_indices=fd_ids,fd_eps=fd_eps,fd_raw_outputs=fd_raw,
                    fd_perturbed_z=fd_z,**activity,**neighbors)
        np.savez_compressed(out/'geometry_samples.npz',**arrays)
        meta=dict(task='Square',checkpoint=str(path.resolve()),checkpoint_sha256=digest(path),
                  dataset=str(args.dataset.resolve()),dataset_sha256=digest(args.dataset),
                  inverse_method=FLOW_INVERSE_METHOD,reverse_query_times=list(inverse_time_grid(config['flow_steps'])),
                  forward_query_times=[i/config['flow_steps'] for i in range(config['flow_steps'])],
                  num_states=B,num_sampled_trajectories=args.geometry_num_trajectories,
                  num_progress_bins=args.geometry_phase_bins,num_dataset_demos=len(activity['all_demo_ids']),
                  num_raw_transitions=int(activity['all_demo_lengths'].sum()),num_valid_chunks=total,
                  seed=args.seed,platform=args.platform,num_prior_samples=len(prior),
                  max_difference_to_corrected_inverse_cache=cache_delta,env_metadata=env_meta,
                  sampling=f'Uniform {args.geometry_num_trajectories} trajectories without replacement, one full window uniformly in each of {args.geometry_phase_bins} relative-progress bins; no reconstruction filtering',
                  uncertainty='Percentile bootstrap resampling complete trajectories; progress states remain clustered',
                  primary_variant='preclip',direction_names=list(DIR_NAMES),
                  neighborhood_warning='Cross-demo standardized-observation neighborhoods approximate local spread, NOT exact conditional expert variance; progress quartiles are not semantic stages',
                  raw_data_axes={'jacobians':'state, variant(preclip/clipped), output, input',
                                 'directions':'state, direction, coordinate',
                                 'perturbed_actions':'state, direction, sign(-/+), radius, action',
                                 'knn_variance':'state, matching(unrestricted/same_progress_bin), k(10/20/50), component'})
        write_json(out/'metadata.json',meta)
        print(f'Saved model outputs: {out}',flush=True)
        if not args.geometry_model_only:
            postprocess(out,args.geometry_bootstrap)


def spearman_rows(x,y):
    """Average-tie rank correlations, along final axis; undefined -> NaN."""
    from scipy.stats import rankdata
    a=rankdata(np.asarray(x),axis=-1);b=rankdata(np.asarray(y),axis=-1)
    a=a-a.mean(axis=-1,keepdims=True);b=b-b.mean(axis=-1,keepdims=True)
    den=np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    return np.divide((a*b).sum(-1),den,out=np.full_like(den,np.nan,dtype=float),where=den>0)


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def partial_spread_correlations(a,sout,decoder_std,expert_rms,boot_idx):
    """Partial rank association controlling command RMS and relative progress.

    Exploratory adjustment, not causal identification. Rank each bootstrap
    sample afresh and project using a pseudoinverse (handles constant gripper
    RMS). Keep the exact same trajectory-cluster resamples as the main tables.
    """
    from scipy.stats import rankdata
    indices=np.concatenate((np.arange(len(sout))[None],boot_idx),axis=0)
    records=[]
    for k,name in enumerate(NAMES):
        control=np.stack((expert_rms[:,k],a['progress']),axis=1)[indices]
        control=rankdata(control,axis=1)
        control-=control.mean(axis=1,keepdims=True)
        design=np.concatenate((np.ones((*control.shape[:2],1)),control),axis=2)
        inverse=np.linalg.pinv(design)

        def residual(values):
            ranked=rankdata(values[indices],axis=1)
            coefficients=np.einsum('bkn,bn->bk',inverse,ranked)
            return ranked-np.einsum('bnk,bk->bn',design,coefficients)

        yr=residual(sout[:,0,k]);yr-=yr.mean(1,keepdims=True)
        predictors={'decoder_prior_std':decoder_std[:,0,k],
                    'cross_demo_knn_std_mode0_k20':np.sqrt(a['knn_variance'][:,0,1,k]),
                    'cross_demo_knn_std_mode1_k20':np.sqrt(a['knn_variance'][:,1,1,k])}
        for label,x in predictors.items():
            xr=residual(x);xr-=xr.mean(1,keepdims=True)
            denominator=np.sqrt((xr*xr).sum(1)*(yr*yr).sum(1))
            rho=np.divide((xr*yr).sum(1),denominator,out=np.full(len(indices),np.nan),where=denominator>1e-20)
            valid=rho[1:][np.isfinite(rho[1:])]
            records.append(dict(component=name,predictor=label,target='sout_preclip',
                                controls='expert_chunk_RMS_and_relative_trajectory_progress',
                                partial_rank_rho=finite_or_none(rho[0]),
                                trajectory_bootstrap_ci025=float(np.quantile(valid,.025)) if len(valid) else None,
                                trajectory_bootstrap_ci975=float(np.quantile(valid,.975)) if len(valid) else None))
    return records


def postprocess(directory,bootstrap=2000):
    """Only read saved arrays; recomputing reports never reruns the model."""
    out=Path(directory).resolve();a=np.load(out/'geometry_samples.npz')
    meta=json.loads((out/'metadata.json').read_text())
    print('Postprocessing experiment 4: '+str(out),flush=True)
    rng=np.random.default_rng(meta['seed']+404)
    B=len(a['demo_id']);ids=np.unique(a['demo_id'])
    groups=[np.flatnonzero(a['demo_id']==d) for d in ids]
    group_choices=rng.integers(0,len(ids),size=(bootstrap,len(ids)))
    boot_idx=np.array([np.concatenate([groups[i] for i in choices]) for choices in group_choices])
    assert all(len(group)==meta['num_progress_bins'] for group in groups)
    all_lengths=a['all_demo_lengths'].astype(float)
    sums=a['all_demo_action_sums'];squares=a['all_demo_action_squares']
    n=all_lengths.sum();mean=sums.sum(0)/n
    rms=np.sqrt(squares.sum(0)/n)
    std=np.sqrt(np.maximum(squares.sum(0)/n-mean**2,0))
    zero=a['all_demo_near_zero_counts'].sum(0)/n
    counts=rng.multinomial(len(all_lengths),np.full(len(all_lengths),1/len(all_lengths)),size=bootstrap)
    boot_n=counts@all_lengths
    boot_mean=(counts@sums)/boot_n[:,None]
    boot_rms=np.sqrt((counts@squares)/boot_n[:,None])
    boot_std=np.sqrt(np.maximum((counts@squares)/boot_n[:,None]-boot_mean**2,0))
    boot_zero=(counts@a['all_demo_near_zero_counts'])/boot_n[:,None]
    activity_records=[]
    for k,name in enumerate(NAMES):
        rec=dict(component=name,n_raw_transitions=int(n),mean=float(mean[k]),rms=float(rms[k]),std=float(std[k]),
                 near_zero_threshold=.05,near_zero_fraction=float(zero[k]),min=float(a['all_action_min'][k]),max=float(a['all_action_max'][k]))
        for label,values in [('rms',boot_rms),('std',boot_std),('near_zero_fraction',boot_zero)]:
            rec[label+'_ci025'],rec[label+'_ci975']=map(float,np.quantile(values[:,k],[.025,.975]))
        for qi,q in enumerate((1,5,25,50,75,95,99)):rec[f'action_q{q:02d}']=float(a['all_action_quantiles'][qi,k])
        activity_records.append(rec)
    write_csv(out/'4_1_action_activity.csv',activity_records)
    geometry=[component_geometry(a['jacobians'][:,v]) for v in range(2)]
    sout=np.stack([g[0] for g in geometry],axis=1)
    sin=np.stack([g[1] for g in geometry],axis=1)
    blocks=np.stack([g[2] for g in geometry],axis=1)
    sout_boot=np.median(sout[boot_idx],axis=1)
    sin_boot=np.median(sin[boot_idx],axis=1)
    sensitivity_records=[];block_records=[];comparison=[]
    for v,label in enumerate(VARIANTS):
        for k,name in enumerate(NAMES):
            so=stats(sout[:,v,k]);si=stats(sin[:,v,k])
            ciout=np.quantile(sout_boot[:,v,k],[.025,.975]);ciin=np.quantile(sin_boot[:,v,k],[.025,.975])
            comparison.append(dict(variant=label,component=name,activity_rms=float(rms[k]),activity_std=float(std[k]),
                                   near_zero_fraction=float(zero[k]),sout_median=so['q50'],sout_ci025=float(ciout[0]),sout_ci975=float(ciout[1]),
                                   sin_median=si['q50'],sin_ci025=float(ciin[0]),sin_ci975=float(ciin[1])))
        for b in range(B):
            for k,name in enumerate(NAMES):
                sensitivity_records.append(dict(state_index=b,demo_id=int(a['demo_id'][b]),timestep=int(a['timestep'][b]),
                                                phase=int(a['phase'][b]),variant=label,component=name,sout=float(sout[b,v,k]),sin=float(sin[b,v,k]),
                                                output_block_frobenius=float(sout[b,v,k]*np.sqrt(112)),
                                                input_block_frobenius=float(sin[b,v,k]*2)))
                for l,latent_name in enumerate(NAMES):
                    block_records.append(dict(state_index=b,variant=label,output_component=name,input_component=latent_name,
                                              block_frobenius=float(blocks[b,v,k,l])))
    write_csv(out/'4_2_activity_sensitivity_comparison.csv',comparison)
    write_csv(out/'4_2_state_sensitivities.csv',sensitivity_records)
    write_csv(out/'4_2_component_blocks.csv',block_records)
    # Joint bootstrap comparisons use SAME sampled trajectories for data and J.
    selected_demo_positions=np.searchsorted(a['all_demo_ids'],ids)
    chosen_lengths=all_lengths[selected_demo_positions]
    chosen_sum=sums[selected_demo_positions];chosen_sq=squares[selected_demo_positions]
    cn=chosen_lengths[group_choices].sum(1)
    cm=chosen_sum[group_choices].sum(1)/cn[:,None]
    crms=np.sqrt(chosen_sq[group_choices].sum(1)/cn[:,None])
    cstd=np.sqrt(np.maximum(chosen_sq[group_choices].sum(1)/cn[:,None]-cm**2,0))
    primary_correlations=[]
    for v,label in enumerate(VARIANTS):
        for sensitivity,sens,boot_sens in [('sout',sout[:,v],sout_boot[:,v]),('sin',sin[:,v],sin_boot[:,v])]:
            for activity,x,xboot in [('rms',rms,crms),('std',std,cstd)]:
                for use_gripper in (False,True):
                    use=np.arange(7 if use_gripper else 6)
                    rho=float(spearman_rows(x[use],np.median(sens,axis=0)[use]))
                    rho_boot=spearman_rows(xboot[:,use],boot_sens[:,use])
                    finite=rho_boot[np.isfinite(rho_boot)]
                    primary_correlations.append(dict(variant=label,activity=activity,sensitivity=sensitivity,
                        components='all7' if use_gripper else 'arm6',rho_full_data=finite_or_none(rho),
                        paired_subset_bootstrap_ci025=float(np.quantile(finite,.025)) if len(finite) else None,
                        paired_subset_bootstrap_ci975=float(np.quantile(finite,.975)) if len(finite) else None,
                        note='rho uses full dataset activity; CI pertains to the paired sampled-trajectory subset statistic'))
    # Explicit subset point estimate makes the CI target unambiguous.
    subset_n=chosen_lengths.sum();subset_mean=chosen_sum.sum(0)/subset_n
    subset_rms=np.sqrt(chosen_sq.sum(0)/subset_n)
    subset_std=np.sqrt(np.maximum(chosen_sq.sum(0)/subset_n-subset_mean**2,0))
    for rec in primary_correlations:
        v=VARIANTS.index(rec['variant']);sens=sout[:,v] if rec['sensitivity']=='sout' else sin[:,v]
        x=subset_rms if rec['activity']=='rms' else subset_std;K=7 if rec['components']=='all7' else 6
        rec['rho_paired_subset']=finite_or_none(spearman_rows(x[:K],np.median(sens,axis=0)[:K]))
    write_csv(out/'4_2_component_correlations.csv',primary_correlations)
    sigma=a['singular_values'];vh=a['svd_vh'];weights=a['singular_component_weights']
    low4=weights[:,:,-4:].mean(axis=2)
    low8=weights[:,:,-8:].mean(axis=2)
    energy=sigma**2
    energy_fraction=energy/np.maximum(energy.sum(-1,keepdims=True),1e-30)
    ranks95=(np.cumsum(energy_fraction,axis=-1)<.95).sum(-1)+1
    ranks99=(np.cumsum(energy_fraction,axis=-1)<.99).sum(-1)+1
    participation=1/np.maximum((energy_fraction**2).sum(-1),1e-30)
    svd_records=[];spectrum_records=[]
    for v,label in enumerate(VARIANTS):
        for b in range(B):
            rec=dict(state_index=b,demo_id=int(a['demo_id'][b]),timestep=int(a['timestep'][b]),variant=label,
                     sigma_max=float(sigma[b,v,0]),sigma_min=float(sigma[b,v,-1]),
                     condition_number=float(sigma[b,v,0]/sigma[b,v,-1]) if sigma[b,v,-1]>0 else None,
                     rank95_energy=int(ranks95[b,v]),rank99_energy=int(ranks99[b,v]),participation_rank=float(participation[b,v]),
                     low4_boundary_gap_ratio=float(sigma[b,v,-5]/sigma[b,v,-4]) if sigma[b,v,-4]>0 else None)
            for tau in (.001,.01,.05,.1):rec[f'num_sigma_gt_{tau:g}_sigma_max']=int((sigma[b,v]>tau*sigma[b,v,0]).sum())
            for k,name in enumerate(NAMES):
                rec['low4_energy_'+name]=float(low4[b,v,k]);rec['low8_energy_'+name]=float(low8[b,v,k])
            svd_records.append(rec)
            for i in range(28):
                rr=dict(state_index=b,variant=label,singular_index_descending=i,sigma=float(sigma[b,v,i]))
                rr.update({f'weight_{name}':float(weights[b,v,i,k]) for k,name in enumerate(NAMES)})
                spectrum_records.append(rr)
    write_csv(out/'4_3_svd_state_metrics.csv',svd_records)
    write_csv(out/'4_3_singular_vector_weights.csv',spectrum_records)
    # Pre-clipping SVD vectors define interventions; BOTH output variants tested.
    outputs=np.stack((a['raw_perturbed_actions'],a['perturbed_actions']),axis=1).astype(float)
    baselines=np.stack((a['raw_baseline_actions'],a['baseline_actions']),axis=1).astype(float)
    da=np.linalg.norm(outputs-baselines[:,:,None,None,None,:],axis=-1)
    dz=np.linalg.norm(a['perturbed_z'].astype(float)-a['z0'][:,None,None,None,:].astype(float),axis=-1)
    gain=da/a['radii'][None,None,None,None,:]
    pred=np.linalg.norm(np.einsum('bvoi,bdi->bvdo',a['jacobians'].astype(float),a['directions'].astype(float)),axis=-1)
    gains_by_state=gain.mean(axis=3)
    da_by_state=da.mean(axis=3)
    perturb_records=[];perturb_summary=[]
    for v,label in enumerate(VARIANTS):
        for d,name in enumerate(DIR_NAMES):
            for ri,r in enumerate(a['radii']):
                values=gains_by_state[:,v,d,ri]
                valid=pred[:,v,d]>1e-5
                rel=np.abs(values[valid]-pred[valid,v,d])/pred[valid,v,d]
                ci=np.quantile(np.median(values[boot_idx],axis=1),[.025,.975])
                summary=dict(variant=label,direction=name,radius=float(r),n_states=B,n_antipodes=2,
                             action_distance_median=float(np.median(da_by_state[:,v,d,ri])),
                             gain_median_ci025=float(ci[0]),gain_median_ci975=float(ci[1]),
                             local_prediction_gain_median=float(np.median(pred[:,v,d])),
                             num_predictions_above_1e_minus5=int(valid.sum()),
                             median_relative_gain_error_when_resolved=float(np.median(rel)) if len(rel) else None,
                             fraction_gain_within_20pct_when_resolved=float(np.mean(rel<=.2)) if len(rel) else None)
                summary.update({'gain_'+key:value for key,value in stats(values).items()})
                perturb_summary.append(summary)
                for b in range(B):
                    for si,sign in enumerate(a['signs']):
                        perturb_records.append(dict(state_index=b,demo_id=int(a['demo_id'][b]),timestep=int(a['timestep'][b]),
                            variant=label,direction=name,sign=float(sign),radius=float(r),measured_dz=float(dz[b,d,si,ri]),
                            action_distance=float(da[b,v,d,si,ri]),gain=float(gain[b,v,d,si,ri]),
                            local_prediction_gain=float(pred[b,v,d]),
                            linear_predicted_distance=float(pred[b,v,d]*r)))
    write_csv(out/'4_4_finite_perturbations.csv',perturb_records)
    write_csv(out/'4_4_finite_perturbation_summary.csv',perturb_summary)
    # Finite differences: output coordinate row, perturbed input column.
    fd=(a['fd_raw_outputs'][:,:,1].astype(float)-a['fd_raw_outputs'][:,:,0].astype(float))/(2*a['fd_eps'][None,:,None,None])
    fd=fd.transpose(0,1,3,2)
    fd_records=[]
    for bi,b in enumerate(a['fd_state_indices']):
        for ei,eps in enumerate(a['fd_eps']):
            target=a['jacobians'][b,0].astype(float)
            fd_records.append(dict(state_index=int(b),epsilon=float(eps),
                                   relative_frobenius_error=float(np.linalg.norm(fd[bi,ei]-target)/np.linalg.norm(target)),
                                   max_absolute_entry_error=float(np.max(abs(fd[bi,ei]-target)))))
    write_csv(out/'finite_difference_checks.csv',fd_records)
    prior=np.stack((a['prior_raw_actions'],a['prior_actions']),axis=1).reshape(B,2,-1,4,7).astype(float)
    decoder_var=prior.var(axis=2,ddof=1).mean(axis=2)
    decoder_std=np.sqrt(decoder_var)
    expert_rms=np.sqrt((a['expert_actions'].reshape(B,4,7).astype(float)**2).mean(1))
    conditional_records=[];conditional_correlations=[]
    for b in range(B):
        for k,name in enumerate(NAMES):
            rec=dict(state_index=b,demo_id=int(a['demo_id'][b]),timestep=int(a['timestep'][b]),component=name,
                     expert_chunk_rms=float(expert_rms[b,k]),sout_preclip=float(sout[b,0,k]),sin_preclip=float(sin[b,0,k]),
                     decoder_std_preclip=float(decoder_std[b,0,k]),decoder_std_clipped=float(decoder_std[b,1,k]),
                     nearest20_median_normalized_distance=float(np.median(a['knn_distances'][b,0,:20])),
                     nearest20_same_progress_bin_fraction=float(np.mean(a['knn_phase'][b,0,:20]==a['phase'][b])))
            for mode in range(2):
                for ki,kval in enumerate(a['knn_sizes']):
                    rec[f'knn_std_mode{mode}_k{kval}']=float(np.sqrt(a['knn_variance'][b,mode,ki,k]))
            conditional_records.append(rec)
    write_csv(out/'conditional_spread_state_components.csv',conditional_records)
    for k,name in enumerate(NAMES):
        predictors={'expert_chunk_rms':expert_rms[:,k], 'decoder_prior_std':decoder_std[:,0,k]}
        for mode in range(2):
            for ki,kval in enumerate(a['knn_sizes']):
                predictors[f'cross_demo_knn_std_mode{mode}_k{kval}']=np.sqrt(a['knn_variance'][:,mode,ki,k])
        for label,x in predictors.items():
            y=sout[:,0,k]
            rho=spearman_rows(x,y);rb=spearman_rows(x[boot_idx],y[boot_idx]);valid=rb[np.isfinite(rb)]
            conditional_correlations.append(dict(component=name,predictor=label,target='sout_preclip',
                rho=finite_or_none(rho),trajectory_bootstrap_ci025=float(np.quantile(valid,.025)) if len(valid) else None,
                trajectory_bootstrap_ci975=float(np.quantile(valid,.975)) if len(valid) else None))
    write_csv(out/'conditional_spread_correlations.csv',conditional_correlations)
    partial_correlations=partial_spread_correlations(a,sout,decoder_std,expert_rms,boot_idx)
    write_csv(out/'conditional_spread_partial_correlations.csv',partial_correlations)
    np.savez_compressed(out/'derived_metrics.npz',sout=sout,sin=sin,blocks=blocks,activity_rms=rms,activity_std=std,
                        activity_near_zero_fraction=zero,low4_component_energy=low4,low8_component_energy=low8,
                        rank95=ranks95,rank99=ranks99,participation_rank=participation,
                        finite_action_distances=da,finite_latent_distances=dz,finite_gains=gain,
                        jacobian_predicted_gains=pred,decoder_component_variance=decoder_var,
                        finite_difference_jacobians=fd,bootstrap_state_indices=boot_idx)
    summary=dict(metadata=meta,bootstrap_replicates=bootstrap,
                 activity=activity_records,component_comparison=comparison,component_correlations=primary_correlations,
                 conditional_correlations=conditional_correlations,conditional_partial_correlations=partial_correlations,
                 preclip_sigma_max=stats(sigma[:,0,0]),preclip_sigma_min=stats(sigma[:,0,-1]),
                 preclip_rank95=stats(ranks95[:,0]),preclip_rank99=stats(ranks99[:,0]),
                 preclip_participation_rank=stats(participation[:,0]),
                 preclip_bottom4_mean_component_energy=dict(zip(NAMES,map(float,low4[:,0].mean(0)))),
                 preclip_bottom8_mean_component_energy=dict(zip(NAMES,map(float,low8[:,0].mean(0)))),
                 clipped_sigma_min=stats(sigma[:,1,-1]),
                 arm_only_preclip_sigma_min=stats(a['arm_singular_values'][:,-1]),
                 finite_difference_checks=fd_records,
                 finite_perturbation_summary=perturb_summary,
                 gripper_preclip_baseline_saturation_fraction=float((abs(a['raw_baseline_actions'].reshape(B,4,7)[:,:,6])>=1).mean()),
                 knn_neighbor_distance_to_random_median_ratio=stats(np.median(a['knn_distances'][:,0,:20],axis=1)/a['random_cross_demo_distance_median']),
                 knn_same_progress_bin_fraction=stats(np.mean(a['knn_phase'][:,0,:20]==a['phase'][:,None],axis=1)))
    write_json(out/'summary.json',summary)
    make_figures(out,a,summary,sout,sin,blocks,da_by_state,gains_by_state,pred,decoder_std)
    write_report(out,a,summary,sout,sin,blocks,da_by_state,gains_by_state,pred,decoder_std)
    write_conclusions(out)
    print('All four experiment parts complete: '+str(out),flush=True)


def make_figures(out,a,summary,sout,sin,blocks,da,gain,pred,decoder_std):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    colors={'vmax':'#d44b40','vmin':'#2679b8','random':'#3a9d62','arm_vmax':'#d44b40','arm_vmin':'#2679b8'}
    x=np.arange(7)

    def save(fig,name):
        fig.tight_layout()
        for extension in ('png','pdf'):fig.savefig(out/f'{name}.{extension}',dpi=180)
        plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,metric,title in zip(axes[0],('rms','std'),('Offline action RMS (A)','Offline action standard deviation (V)')):
        values=np.array([r[metric] for r in summary['activity']])
        lo=np.array([r[metric+'_ci025'] for r in summary['activity']]);hi=np.array([r[metric+'_ci975'] for r in summary['activity']])
        ax.bar(x,values,color='#5588b7',alpha=.8)
        ax.errorbar(x,values,yerr=[values-lo,hi-values],fmt='none',color='black',capsize=3)
        for i,v in enumerate(values):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
        ax.set(title=title,ylabel='Normalized action command',xticks=x,xticklabels=NAMES)
    for ax,metric,title in zip(axes[1],('sout','sin'),('Output sensitivity: all inputs -> component','Input sensitivity: component -> all outputs')):
        for v,label in enumerate(VARIANTS):
            records=[r for r in summary['component_comparison'] if r['variant']==label]
            values=np.array([r[metric+'_median'] for r in records])
            lo=np.array([r[metric+'_ci025'] for r in records]);hi=np.array([r[metric+'_ci975'] for r in records])
            offset=-.18 if v==0 else .18
            ax.bar(x+offset,values,width=.34,label=label,alpha=.8)
            # Percentile CI need not bracket a finite-sample point estimate.
            ax.vlines(x+offset,np.maximum(lo,1e-10),np.maximum(hi,1e-10),color='black',linewidth=1)
            for i,value in enumerate(values):
                if value==0:ax.text(i+offset,1.1e-8,'0',ha='center',fontsize=8)
        ax.set(xticks=x,xticklabels=NAMES,title=title,ylabel='Median sensitivity (log scale)',yscale='log')
        ax.set_ylim(bottom=1e-8);ax.legend()
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Square: activity versus decoder sensitivity | 100 states / 25 trajectories\nError bars: trajectory-bootstrap 95% intervals; output/input normalizations differ')
    save(fig,'figure1_activity_sensitivity')

    fig,axes=plt.subplots(1,3,figsize=(18,6))
    medians=[np.median(blocks[:,v],axis=0) for v in range(2)]
    vmax=max(m.max() for m in medians);vmin=max(vmax*1e-6,1e-9)
    for ax,m,title in zip(axes[:2],medians,('Preclip median block Frobenius norm','Clipped median block Frobenius norm')):
        im=ax.imshow(np.maximum(m,vmin),norm=LogNorm(vmin=vmin,vmax=vmax),cmap='viridis')
        for k in range(7):
            for l in range(7):ax.text(l,k,f'{m[k,l]:.2g}',ha='center',va='center',fontsize=7,color='white' if m[k,l]<vmax*.1 else 'black')
        ax.set(title=title,xticks=x,xticklabels=NAMES,yticks=x,yticklabels=NAMES,xlabel='Input latent component group',ylabel='Output action component')
        fig.colorbar(im,ax=ax,fraction=.046)
    energy=np.mean(blocks[:,0]**2,axis=0);fraction=energy/np.maximum(energy.sum(0,keepdims=True),1e-30)
    im=axes[2].imshow(fraction,vmin=0,vmax=1,cmap='magma')
    for k in range(7):
        for l in range(7):axes[2].text(l,k,f'{fraction[k,l]:.0%}',ha='center',va='center',fontsize=8,color='white' if fraction[k,l]<.6 else 'black')
    axes[2].set(title='Preclip pooled energy / input column',xticks=x,xticklabels=NAMES,yticks=x,yticklabels=NAMES,xlabel='Input latent component group',ylabel='Output action component')
    fig.colorbar(im,ax=axes[2],fraction=.046)
    fig.suptitle('Square: 7 x 7 blocks (each block includes ALL 4 x 4 time couplings)')
    save(fig,'figure2_component_blocks')

    fig,axes=plt.subplots(2,2,figsize=(15,10))
    indices=np.arange(1,29)
    for v,label in enumerate(VARIANTS):
        q=np.quantile(a['singular_values'][:,v],[.05,.5,.95],axis=0)
        line,=axes[0,0].plot(indices,np.maximum(q[1],1e-12),label=label)
        axes[0,0].fill_between(indices,np.maximum(q[0],1e-12),np.maximum(q[2],1e-12),alpha=.15,color=line.get_color())
    axes[0,0].set(yscale='log',xlabel='Singular index (descending)',ylabel='Singular value',title='Median and P05/P95 across states (zeros plotted at 1e-12)');axes[0,0].legend()
    w=a['singular_component_weights'][:,0].mean(0)
    im=axes[0,1].imshow(w,aspect='auto',vmin=0,vmax=1,cmap='magma')
    axes[0,1].set(xticks=x,xticklabels=NAMES,yticks=np.arange(0,28,3),yticklabels=np.arange(1,29,3),xlabel='Latent component group',ylabel='Singular index (descending)',title='Preclip right-vector component energy (state mean)')
    fig.colorbar(im,ax=axes[0,1])
    for i,nlow in enumerate((4,8)):
        value=a['singular_component_weights'][:,0,-nlow:].mean(axis=(0,1))
        axes[1,0].bar(x+(-.18 if i==0 else .18),value,width=.34,label=f'bottom-{nlow} subspace')
    axes[1,0].axhline(1/7,color='gray',linestyle='--',label='Uniform 1/7')
    axes[1,0].set(xticks=x,xticklabels=NAMES,ylabel='Mean component energy',title='Subspace energy is invariant to rotations INSIDE the subspace');axes[1,0].legend(fontsize=8)
    q=np.quantile(a['arm_singular_values'],[.05,.5,.95],axis=0)
    axes[1,1].plot(np.arange(1,25),q[1],color='#3578a8')
    axes[1,1].fill_between(np.arange(1,25),q[0],q[2],alpha=.2)
    axes[1,1].set(yscale='log',xlabel='Singular index (descending)',ylabel='Singular value',title='Arm-only preclip 24 x 24: exclude gripper INPUTS and OUTPUTS')
    for ax in (axes[0,0],axes[1,0],axes[1,1]):ax.grid(alpha=.2)
    fig.suptitle(f'Square Jacobian SVD | preclip median energy rank95={summary["preclip_rank95"]["q50"]:g}/28 | '
                 f'participation rank={summary["preclip_participation_rank"]["q50"]:.2f}')
    save(fig,'figure3_singular_spectrum')

    fig,axes=plt.subplots(2,3,figsize=(18,10))
    selected=np.flatnonzero(a['radii']>=.099)
    rs=a['radii'][selected]
    for v,label in enumerate(VARIANTS):
        for d,name in enumerate(DIR_NAMES[:3]):
            q=np.quantile(da[:,v,d][:,selected],[.05,.5,.95],axis=0)
            axes[v,0].plot(rs,q[1],'-o',color=colors[name],label=name)
            axes[v,0].fill_between(rs,np.maximum(q[0],1e-10),np.maximum(q[2],1e-10),color=colors[name],alpha=.12)
            axes[v,0].plot(rs,rs*np.median(pred[:,v,d]),'--',color=colors[name],alpha=.6)
            qg=np.quantile(gain[:,v,d][:,selected],[.05,.5,.95],axis=0)
            axes[v,1].plot(rs,qg[1],'-o',color=colors[name],label=name)
            axes[v,1].fill_between(rs,np.maximum(qg[0],1e-10),np.maximum(qg[2],1e-10),color=colors[name],alpha=.12)
            axes[v,1].axhline(np.median(pred[:,v,d]),linestyle='--',color=colors[name],alpha=.6)
        for d,name in ((4,'arm_vmax'),(5,'arm_vmin')):
            q=np.quantile(da[:,v,d][:,selected],[.05,.5,.95],axis=0)
            axes[v,2].plot(rs,q[1],'-o',color=colors[name],label=name)
            axes[v,2].fill_between(rs,np.maximum(q[0],1e-10),np.maximum(q[2],1e-10),color=colors[name],alpha=.12)
            axes[v,2].plot(rs,rs*np.median(pred[:,v,d]),'--',color=colors[name],alpha=.6)
        for c,title in enumerate(('Action distance','Gain da/r','Arm-derived directions: action distance')):
            axes[v,c].set(yscale='log',xlabel='Perturbation radius',ylabel=title,title=f'{label}: {title}')
            axes[v,c].legend(fontsize=8);axes[v,c].grid(alpha=.2)
    fig.suptitle('Square finite perturbations | mean of +/- per state, then median and P05/P95 across states\nSolid: real decoder; dashed: local Jacobian prediction. Directions come from PRECLIP SVD.')
    save(fig,'figure4_finite_perturbations')

    fig,axes=plt.subplots(1,3,figsize=(18,5))
    axes[0].bar(x-.18,np.median(decoder_std[:,0],axis=0),width=.34,label='Preclip decoder N(0,I) std')
    axes[0].bar(x+.18,np.median(np.sqrt(a['knn_variance'][:,0,1]),axis=0),width=.34,label='Cross-demo kNN expert std (k=20)')
    axes[0].set(xticks=x,xticklabels=NAMES,ylabel='Median component standard deviation',title='Two DIFFERENT notions of conditional spread');axes[0].legend(fontsize=7)
    for predictor,offset in (('decoder_prior_std',-.18),('cross_demo_knn_std_mode0_k20',.18)):
        records=[r for r in summary['conditional_correlations'] if r['predictor']==predictor]
        for k,rec in enumerate(records):
            if rec['rho'] is None:continue
            axes[1].scatter(k+offset,rec['rho'],s=25,color='#2679b8' if offset<0 else '#db8a36')
            axes[1].plot([k+offset]*2,[rec['trajectory_bootstrap_ci025'],rec['trajectory_bootstrap_ci975']],color='#2679b8' if offset<0 else '#db8a36')
    axes[1].axhline(0,color='gray',linestyle='--');axes[1].set(xticks=x,xticklabels=NAMES,ylim=(-1.05,1.05),ylabel='Within-component Spearman with Sout',title='Blue: decoder std; orange: kNN expert std\n95% trajectory bootstrap intervals')
    ratio=np.median(a['knn_distances'][:,0,:20],axis=1)/a['random_cross_demo_distance_median']
    axes[2].hist(ratio,bins=16,color='#4c9a78');axes[2].set(xlabel='Median neighbor distance / median random cross-demo distance',ylabel='States',title='Neighborhood quality (smaller = relatively closer)')
    fig.suptitle('Conditional-spread diagnostics: kNN spread is NOT exact Var(a | s)')
    save(fig,'conditional_spread_diagnostics')


def write_report(out,a,s,sout,sin,blocks,da,gain,pred,decoder_std):
    meta=s['metadata'];B=len(a['demo_id'])
    pre=[r for r in s['component_comparison'] if r['variant']=='preclip']
    lines=['# Square 实验四：动作分布与 Flow Decoder 的 latent 敏感性','',
           '**范围：按用户确认，仅做 Square；四个分实验及条件分布补充对照均已实际运行。**','',
           '## 设置与统计约定','',
           f'- checkpoint：`{meta["checkpoint"]}`；数据：`{meta["dataset"]}`；SHA256 记录在 metadata.json。',
           f'- 全数据 {meta["num_dataset_demos"]} 条轨迹，{meta["num_raw_transitions"]:,} 条原始 transition，{meta["num_valid_chunks"]:,} 个有效完整动作块。',
           f'- 随机无放回选 {meta["num_sampled_trajectories"]} 条轨迹，每条轨迹按相对时间四等分，每区间随机取一个有效状态，共 {B} 个状态。不按反解误差筛选。相对进度不是语义任务阶段。',
           '- 沿用统一反解：固定 s，从专家 4×7 动作块开始，查询 t=0.9,…,0，每步 x←x−0.1v(s,x,t)。它是显式反向减法近似，有限步误差仍存在。',
           '- Decoder 使用原始前向 Euler t=0,…,0.9；主要结果在最终裁剪之前计算，并平行保存 [-1,1] 裁剪后的实际控制动作结果。模型冻结，无 latent 裁剪/tanh/优化。',
           f'- 状态敏感度和扰动曲线的置信区间以整条轨迹为重采样单位，{s["bootstrap_replicates"]} 次 percentile bootstrap；同轨迹四个状态一起保留。动作活动度则重采样全部 300 条轨迹。P05/P95 表示状态分布，不能当成置信区间。',
           '- 归一化 OSC_POSE 相对命令：(x,y,z,rx,ry,rz,g)。旋转是轴角增量；夹爪独立分析。不同分量量纲/控制缩放不同，统计的是命令空间，而非真实物理位移或旋转角大小。','',
           '## 4.1 全 demonstration 动作活动度','',
           '每条原始 transition 只计一次，不使用重叠动作块重复计数。A=√E[a²]，V=√Var(a)（总体方差）；近零比例为 |a|<0.05。','',
           '| 分量 | A RMS | V std | 近零比例 | A 的轨迹 bootstrap 95% CI |',
           '|---|---:|---:|---:|---|']
    for r in s['activity']:lines.append(f'| {r["component"]} | {r["rms"]:.5f} | {r["std"]:.5f} | {r["near_zero_fraction"]:.1%} | [{r["rms_ci025"]:.5f}, {r["rms_ci975"]:.5f}] |')
    lines += ['', '## 4.2 输入、输出和整块敏感度','',
              'J 的行是动作输出，列是 latent 输入，索引均为 7t+k。严格按方案定义：',
              '- Sout_k = √(Σ_t,j J[7t+k,j]² / (4×28))：所有输入对该动作分量的作用。',
              '- Sin_k = √(Σ_i,τ J[i,7τ+k]² / 4)：该组 latent 对所有输出的作用。',
              '- B[k,l] = ||J[A_k,Z_l]||F，包含完整 4×4 跨时间子块。Sout 和 Sin 归一化不同，不应直接比较绝对大小。','',
              '| 分量 | 裁剪前 Sout 中位数 | Sout 95% CI | 裁剪前 Sin 中位数 | Sin 95% CI |',
              '|---|---:|---|---:|---|']
    for r in pre:lines.append(f'| {r["component"]} | {r["sout_median"]:.6f} | [{r["sout_ci025"]:.6f}, {r["sout_ci975"]:.6f}] | {r["sin_median"]:.6f} | [{r["sin_ci025"]:.6f}, {r["sin_ci975"]:.6f}] |')
    lines += ['', '跨动作分量的相关性只有 6 个机械臂分量/7 个总分量，属于描述性证据；不以 100 个状态复制同一个全局活动度，伪造大量独立观测。',
              '4_2_component_correlations.csv 同时给全 300 条轨迹的活动度相关性和配对 25 条轨迹子集的相关性；子集的 bootstrap CI 仅对应子集点估计。','',
              '| 活动度 | 敏感度 | 分量集合 | 全数据活动度 Spearman | 配对子集 Spearman | 子集 95% CI |',
              '|---|---|---|---:|---:|---|']
    for r in s['component_correlations']:
        if r['variant']=='preclip':
            lines.append(f'| {r["activity"]} | {r["sensitivity"]} | {r["components"]} | {r["rho_full_data"]:.3f} | {r["rho_paired_subset"]:.3f} | [{r["paired_subset_bootstrap_ci025"]:.3f}, {r["paired_subset_bootstrap_ci975"]:.3f}] |')
    lines += ['', '## 4.3 SVD 和低敏感子空间','',
              f'- 裁剪前 σmax 中位数={s["preclip_sigma_max"]["q50"]:.6g}，σmin 中位数={s["preclip_sigma_min"]["q50"]:.6g}。',
              f'- 覆盖 95% / 99% Jacobian 平方奇异值能量所需维数中位数分别为 {s["preclip_rank95"]["q50"]:g} / {s["preclip_rank99"]["q50"]:g}，总维度 28。能量参与秩中位数={s["preclip_participation_rank"]["q50"]:.3f}。',
              '- 低敏感部分同时报告 bottom-4 和 bottom-8 子空间，取右奇异向量在各坐标组上的平方能量平均；子空间内旋转不改变这个统计量。单个向量可能因近重根改变，子空间在截断边界缺少谱隙时也可能不稳定；谱隙另存 CSV。',
              '- 另算去除夹爪输入和输出的 24×24 机械臂子矩阵；其 σmin 中位数='+f'{s["arm_only_preclip_sigma_min"]["q50"]:.6g}。',
              '- SVD 使用 float64 分解已保存的 float32 自动微分矩阵。这不能提高模型导数本身的精度；极小奇异值/条件数需结合有限差分和有限扰动理解。平方奇异值的能量维数不是矩阵精确秩，也不证明该流形只有这些维度。','',
              '| latent 分量组 | bottom-4 平均能量 | bottom-8 平均能量 |',
              '|---|---:|---:|']
    for name in NAMES:lines.append(f'| {name} | {s["preclip_bottom4_mean_component_energy"][name]:.1%} | {s["preclip_bottom8_mean_component_energy"][name]:.1%} |')
    lines += ['', '## 4.4 真实 Decoder 的有限扰动验证','',
              '- 对每个相同的 (s,z0)，取裁剪前 J 的 vmax、vmin、随机单位方向；另取 bottom-4 内随机方向、去夹爪子矩阵的 vmax/vmin 作为补充。',
              '- 所有方向都测试 ± 两侧，半径 0.1/0.25/0.5/1/2；额外 0.001/0.01 用于检查局部近似，共 8,400 个真实解码点。',
              '- 距离参考始终为对应输出版本的 D(s,z0)，而非专家动作。表中每个状态先平均正负两侧，再取状态中位数。预测增益为 ||Jv||，实际增益为 da/r。','',
              '| r | 方向 | 裁剪前 da 中位数 | 实际 da/r 中位数 | ||Jv|| 中位数 |',
              '|---:|---|---:|---:|---:|']
    for rr in s['finite_perturbation_summary']:
        if rr['variant']=='preclip' and rr['direction'] in DIR_NAMES[:3] and rr['radius']>=.099:
            lines.append(f'| {rr["radius"]:g} | {rr["direction"]} | {rr["action_distance_median"]:.6g} | {rr["gain_q50"]:.6g} | {rr["local_prediction_gain_median"]:.6g} |')
    lines += ['', '4_4_finite_perturbation_summary.csv 还给增益 P05/P95、轨迹 bootstrap 置信区间和对局部预测的相对误差。相对误差仅在 ||Jv||>1e-5 时汇总，以避免用几乎为零的分母放大浮点噪声；全部原始响应仍保存。','',
              '## 条件分布对照：区分假设 A 与 B','',
              '- **专家局部散布代理**：每个状态从其他轨迹找近邻，标准化全部 observation 坐标后计算欧氏距离，机器人四元数仅在距离计算中统一符号。k=10/20/50；同时给不限制进度和限制同进度四分位两种版本。所有近邻的轨迹、时刻、距离均保存。',
              '- 对每个动作时间坐标先跨近邻计算样本方差，再平均四个时间步。它受邻域半径、状态度量和阶段混合影响，**不是精确的 Var_demo(a|s)**，全局动作方差也不能代替该量。',
              '- 近邻排除了查询轨迹，但允许来自同一其他轨迹的多个相邻时刻；它们不能当成 k 个独立专家重复试验。相关性 CI 重采样查询轨迹、固定已估计的邻域，并未包含全部近邻估计误差。',
              f'- **Decoder 生成分布**：在每个固定 s 下，用同一组 {meta["num_prior_samples"]} 个 z~N(0,I) 生成动作，计算各时间坐标跨 z 的方差，再平均四个时间步。它检验整个先验生成分布，并非反解 z0 邻域，也不是专家条件分布真值。',
              '- conditional_spread_correlations.csv 按每个动作分量分别计算跨状态相关性，给轨迹 bootstrap CI，避免不同分量间的尺度差异制造合并相关。',
              '- conditional_spread_partial_correlations.csv 进一步对秩变量回归，控制当前专家动作块 RMS 和相对轨迹进度，再计算残差相关；使用相同轨迹 bootstrap。它检验散布关联能否超出这两个观测混杂量，仍不构成因果识别。',
              '- 这些证据可提示数据散布与网络敏感性相关，但无法单靠观察性分析判定训练分布导致收缩，更不能证明删掉某些 latent 方向能提高 online RL 样本效率。','',
              '## 夹爪与裁剪','',
              f'- 基线夹爪输出在裁剪范围外/边界的比例为 {s["gripper_preclip_baseline_saturation_fraction"]:.1%}。',
              '- 裁剪会制造零输出导数及零奇异值，因此主分析用裁剪前矩阵；裁剪后结果反映实际控制，但不能把其零梯度全部归因于 Flow 学到的收缩。','',
              '## 图与原始文件','',
              '1. [图1：活动度与输入/输出敏感度](figure1_activity_sensitivity.png)。上排是全数据动作强弱和变化，下排是敏感度；低值所在分量是否一致是主要观察。',
              '2. [图2：7×7 整块敏感度](figure2_component_blocks.png)。行是动作、列是 latent 组；每格包含全部跨时间耦合。右侧每列归一化，查看某组 latent 是否通过其他动作产生影响。',
              '3. [图3：奇异值谱和低敏感子空间组成](figure3_singular_spectrum.png)。先看裁剪前曲线和去夹爪曲线，再看低敏感能量集中在哪些组；裁剪后的零值仅为画图被放到 1e-12。',
              '4. [图4：有限扰动](figure4_finite_perturbations.png)。实线是真实响应、虚线是局部 Jacobian 预测，阴影是跨状态 P05/P95；左/中列对照三种方向，右列检查机械臂子矩阵方向。',
              '5. [条件散布对照](conditional_spread_diagnostics.png)。蓝色相关性针对 Decoder 生成方差，橙色针对近邻专家散布代理；二者不能混为一谈。',
              '- `geometry_samples.npz`：所有源数据、完整 J、SVD、方向、正负扰动输出、近邻、先验样本以及有限差分原始输出。',
              '- `derived_metrics.npz`：全部敏感度、有限扰动距离/增益、局部预测、生成方差及 bootstrap 索引。',
              '- `4_1_*.csv` / `4_2_*.csv` / `4_3_*.csv` / `4_4_*.csv`：对应各分实验完整统计。',
              '- `metadata.json` / `summary.json`：来源、校验和、参数与统计；`validation.txt`：独立复核。','',
              '## 复现','',
              '```bash',
              '.venv/bin/python scripts/dsrlna/analyze_demo_latents.py --flow-geometry-experiment --steps 2000000 --geometry-num-trajectories 25 --geometry-phase-bins 4 --geometry-prior-samples 128 --geometry-bootstrap 2000 --batch-size 256 --platform cpu --seed 1',
              '# 只重画图和统计，不重跑反解或 Decoder：',
              f'.venv/bin/python scripts/dsrlna/analyze_demo_latents.py --geometry-postprocess-from {out}',
              '```','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def write_conclusions(directory):
    """Readable interpretation and low-subspace uncertainty from saved outputs."""
    out=Path(directory);a=np.load(out/'geometry_samples.npz');d=np.load(out/'derived_metrics.npz')
    s=json.loads((out/'summary.json').read_text());boot=d['bootstrap_state_indices']
    low_records=[]
    for nlow in (4,8):
        value=a['singular_component_weights'][:,0,-nlow:].mean(1)
        boot_mean=value[boot].mean(1)
        for k,name in enumerate(NAMES):
            ci=np.quantile(boot_mean[:,k],[.025,.975])
            low_records.append(dict(low_subspace_dimension=nlow,component=name,mean_energy=float(value[:,k].mean()),
                                    trajectory_bootstrap_ci025=float(ci[0]),trajectory_bootstrap_ci975=float(ci[1]),
                                    state_q05=float(np.quantile(value[:,k],.05)),state_q50=float(np.median(value[:,k])),
                                    state_q95=float(np.quantile(value[:,k],.95))))
    write_csv(out/'4_3_low_subspace_uncertainty.csv',low_records)
    low4=d['low4_component_energy'][:,0]
    rotation_gripper=low4[:,3:].sum(1)
    rg_ci=np.quantile(rotation_gripper[boot].mean(1),[.025,.975])
    # Use identical within-state comparisons, not ratios of unrelated medians.
    da=d['finite_action_distances'].mean(axis=3)
    radii=a['radii'];finite_comparisons=[]
    for ri,r in enumerate(radii):
        if r<.099:continue
        rr=dict(radius=float(r),fraction_vmin_lt_random=float((da[:,0,1,ri]<da[:,0,2,ri]).mean()),
                fraction_random_lt_vmax=float((da[:,0,2,ri]<da[:,0,0,ri]).mean()),
                random_over_vmin=stats(da[:,0,2,ri]/np.maximum(da[:,0,1,ri],1e-30)),
                vmax_over_vmin=stats(da[:,0,0,ri]/np.maximum(da[:,0,1,ri],1e-30)))
        finite_comparisons.append(rr)
    gap=a['singular_values'][:,0,-5]/a['singular_values'][:,0,-4]
    supplemental=dict(low4_rotation_plus_gripper_energy=stats(rotation_gripper),
                      low4_rotation_plus_gripper_mean_ci025=float(rg_ci[0]),
                      low4_rotation_plus_gripper_mean_ci975=float(rg_ci[1]),
                      low4_subspace_boundary_spectral_gap=stats(gap),finite_within_state_comparisons=finite_comparisons)
    unique_neighbors=np.array([len(np.unique(neighbors)) for neighbors in a['knn_demo_ids'][:,0,:20]])
    supplemental['distinct_other_demos_among_20_neighbors']=stats(unique_neighbors)
    write_json(out/'supplementary_statistics.json',supplemental)
    pre=[r for r in s['component_comparison'] if r['variant']=='preclip']
    arm_rho=next(r for r in s['component_correlations'] if r['variant']=='preclip' and r['activity']=='rms' and r['sensitivity']=='sout' and r['components']=='arm6')
    all_rho=next(r for r in s['component_correlations'] if r['variant']=='preclip' and r['activity']=='rms' and r['sensitivity']=='sout' and r['components']=='all7')
    other_arm_rhos=[r['rho_full_data'] for r in s['component_correlations'] if r['variant']=='preclip' and r['components']=='arm6'
                    and (r['activity'],r['sensitivity']) in (('rms','sin'),('std','sout'),('std','sin'))]

    def response(direction,radius):
        return next(r for r in s['finite_perturbation_summary'] if r['variant']=='preclip'
                    and r['direction']==direction and np.isclose(r['radius'],radius))

    min_errors=[response('vmin',r)['median_relative_gain_error_when_resolved'] for r in (.01,.1,1,2)]
    max_fd_error=max(r['relative_frobenius_error'] for r in s['finite_difference_checks'] if r['epsilon']>.002)
    meta=s['metadata']
    energy=np.mean(d['blocks'][:,0]**2,axis=0)
    column_fraction=energy/energy.sum(0,keepdims=True)
    lines=['# Square 实验四：结果、解释和读图','',
           '**四个分实验已全部执行。结果支持 Square 中机械臂动作活动度与 latent 敏感性相关，也找到了在有限扰动下仍然低敏感的组合方向；但“动作幅度唯一决定敏感度”不成立，夹爪是明显例外。**','',
           '只使用 Square，继续使用之前的 `dsrlna_squareseed1_20261102_121751_418831UTC/params_2000000.pkl`。反解时间统一为 `0.9,…,0`。',
           f'动作活动度使用全部 {meta["num_dataset_demos"]} 条 demo、{meta["num_raw_transitions"]:,} 条原始 transition；模型分析随机选 {meta["num_sampled_trajectories"]} 条轨迹，每条相对时间 {meta["num_progress_bins"]} 等分各取一个状态，共 {meta["num_states"]} 个状态。阶段分层是时间代理，未声称有真实任务阶段标签。',
           '主结果均使用裁剪前 Decoder；实际裁剪后结果单独保留。所有有限扰动以 Decoder 基线 D(s,z0) 为参考，正负两侧都测试。','',
           '## 4.1–4.2：低活动分量是否对应低敏感性','',
           '| 分量 | 全数据 RMS A | 全数据 std V | 近零比例 | 输出敏感度 Sout 中位数 | 输入敏感度 Sin 中位数 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in pre:
        lines.append(f'| {r["component"]} | {r["activity_rms"]:.4f} | {r["activity_std"]:.4f} | {r["near_zero_fraction"]:.1%} | {r["sout_median"]:.6f} | {r["sin_median"]:.6f} |')
    lines += ['', '**机械臂六个分量：支持相关性。** rx/ry 的动作幅度低，输出敏感度与输入敏感度也低。',
              f'六分量 A 与 Sout 中位数的 Spearman 为 **{arm_rho["rho_full_data"]:.3f}**；A 与 Sin、V 与 Sout、V 与 Sin 分别为 '+', '.join(f'{rho:.3f}' for rho in other_arm_rhos)+f'。配对轨迹子集的 bootstrap 95% CI 为 [{arm_rho["paired_subset_bootstrap_ci025"]:.3f}, {arm_rho["paired_subset_bootstrap_ci975"]:.3f}]。只有六个动作分量，不能把这个相关性当成大样本因果证据。',
              '也不是严格按幅度排序：X 命令的 RMS 最大，Z 输出敏感度却最高。',
              f'**夹爪反驳统一的幅度规则。** g 的 RMS=1、全局 std≈1，但裁剪前 Sout 反而很低。加入 g 后，七分量 A–Sout Spearman 降至 **{all_rho["rho_full_data"]:.3f}**。',
              f'基线夹爪饱和比例为 {s["gripper_preclip_baseline_saturation_fraction"]:.1%}；但 g 的裁剪前敏感度已经低，因此其全部局部收缩也不能只解释为最终裁剪。',
              '**低输出敏感性不等于这组 latent 完全无用。** 图2的每一列汇总某组 latent 对所有动作的影响。',
              f'例如 rx 型 latent 的合并平方导数能量只有 {column_fraction[3,3]:.1%} 落在 rx 输出上，其余分布到其他动作；g 型 latent 也有 {1-column_fraction[6,6]:.1%} 能量影响其他输出。这些是 Jacobian 能量比例，不是任务收益或实际探索价值。','',
              '## 4.3：真正的低敏感方向是什么','',
              f'- 裁剪前 σmax 中位数 **{s["preclip_sigma_max"]["q50"]:.6f}**，σmin 中位数 **{s["preclip_sigma_min"]["q50"]:.8f}**。',
              f'- 解释 95% 平方奇异值能量所需维数的中位数为 **{s["preclip_rank95"]["q50"]:g}/28**（跨状态 P05/P95：{s["preclip_rank95"]["q05"]:g}/{s["preclip_rank95"]["q95"]:g}）；99% 能量为 **{s["preclip_rank99"]["q50"]:g}/28**。能量参与秩中位数为 **{s["preclip_participation_rank"]["q50"]:.2f}**。这是敏感性集中程度，**不是矩阵精确秩或任务真实维度**。',
              '- bottom-4 子空间的平均能量组成如下。使用整个子空间，避免给一个可能因近重奇异值旋转的向量强行命名。','',
              '| latent 组 | bottom-4 平均能量 | 轨迹 bootstrap 95% CI |',
              '|---|---:|---|']
    for r in low_records:
        if r['low_subspace_dimension']==4:
            lines.append(f'| {r["component"]} | {r["mean_energy"]:.1%} | [{r["trajectory_bootstrap_ci025"]:.1%}, {r["trajectory_bootstrap_ci975"]:.1%}] |')
    lines += ['', f'旋转 rx/ry/rz 与夹爪合计占 bottom-4 能量 **{rotation_gripper.mean():.1%}**（轨迹 bootstrap 95% CI [{rg_ci[0]:.1%}, {rg_ci[1]:.1%}]）。bottom-8 的结果同方向；详细不确定性在 `4_3_low_subspace_uncertainty.csv`。',
              f'去掉夹爪输入和输出后的 24×24 机械臂矩阵，σmin 中位数仍为 **{s["arm_only_preclip_sigma_min"]["q50"]:.8f}**，说明弱方向不全来自夹爪裁剪。',
              f'bottom-4 边界谱隙 σ24/σ25 的中位数为 {np.median(gap):.2f}；子空间内旋转不影响能量，但边界接近时，恰好截取四维也可能不稳定，不能声称存在唯一固定的四维弱子空间。','',
              '## 4.4：低敏感方向在有限扰动下是否还成立','',
              f'下面是裁剪前动作距离：每个状态先平均正负两侧，再取 {meta["num_states"]} 个状态的中位数。动作距离覆盖整个 4×7 归一化动作块。','',
              '| 半径 | 最大奇异方向 | 随机方向 | 最小奇异方向 |',
              '|---:|---:|---:|---:|']
    for ri,r in enumerate(radii):
        if r>=.099:lines.append(f'| {r:g} | {np.median(da[:,0,0,ri]):.6f} | {np.median(da[:,0,2,ri]):.6f} | {np.median(da[:,0,1,ri]):.6f} |')
    lines += ['', '低敏感结构在实际有限扰动中保留了区分作用；不是仅由一张 Jacobian 矩阵推断。',
              '**但最小奇异值的线性预测很快失准。** 对 vmin，实际 da/r 相对 ||Jv|| 的逐状态相对误差中位数：'+', '.join(f'r={r:g} 时约 {err:.1%}' for r,err in zip((.01,.1,1,2),min_errors))+'。后两者虽然仍是小动作变化，却已经不能用 σmin×r 定量预测。',
              f'r=0.001 时 vmin 的动作差接近 float32 舍入量级，增益估计明显受噪声影响；不能把该点的偏差解释为网络行为。完整 J 的中心差分在 ε=0.01 时，检查状态的相对 Frobenius 误差最大为 {max_fd_error:.4%}。',
              f'最大奇异方向和随机方向的线性近似保持得更好：r=1 时相对误差中位数分别约 {response("vmax",1)["median_relative_gain_error_when_resolved"]:.1%} 和 {response("random",1)["median_relative_gain_error_when_resolved"]:.1%}。因此“低敏感仍有效”与“线性近似仍准确”必须分别判断。','',
              '## 假设 B：局部专家散布是否还提供额外解释','',
              f'已补上两类对照：跨轨迹相似状态专家动作散布（k=10/20/50，两种进度匹配方式），以及每个固定状态 {meta["num_prior_samples"]} 个 N(0,I) latent 的 Decoder 输出方差。',
              '下表按分量比较跨状态关联；专家散布取其他轨迹最近 20 个状态。偏相关控制当前专家动作块 RMS 和相对轨迹进度，对秩变量回归后计算残差相关。','',
              '| 分量 | 专家近邻 std 与 Sout 的 rho | 控制幅度/进度后 rho | 偏相关轨迹 bootstrap 95% CI | Decoder 先验 std 与 Sout 的 rho |',
              '|---|---:|---:|---|---:|']
    for name in NAMES:
        c=next(r for r in s['conditional_correlations'] if r['component']==name and r['predictor']=='cross_demo_knn_std_mode0_k20')
        cp=next(r for r in s['conditional_partial_correlations'] if r['component']==name and r['predictor']=='cross_demo_knn_std_mode0_k20')
        dec=next(r for r in s['conditional_correlations'] if r['component']==name and r['predictor']=='decoder_prior_std')
        lines.append(f'| {name} | {c["rho"]:.3f} | {cp["partial_rank_rho"]:.3f} | [{cp["trajectory_bootstrap_ci025"]:.3f}, {cp["trajectory_bootstrap_ci975"]:.3f}] | {dec["rho"]:.3f} |')
    lines += ['', '机械臂六分量中，控制幅度和进度后仍有正关联，提示**仅靠动作幅度解释敏感性不充分**。这为 B 提供了额外的观察性证据；g 不呈现同样关联。',
              '近邻估计混合了不同的精确状态、可能的任务阶段和操作者差异，**不等于真实条件方差 Var_demo(a|s)**；Decoder 先验方差也不是专家方差真值。控制两个变量不能排除全部混杂，因此尚不能证明 B 的因果机制或其优于 A 的完整解释力。',
              f'20 个近邻只排除查询轨迹，允许同一其他轨迹的相邻时刻；不同邻居轨迹数中位数为 {np.median(unique_neighbors):g}（范围 {unique_neighbors.min():g}–{unique_neighbors.max():g}），不能当成 20 次独立专家重复。偏相关 CI 固定邻域、重采样查询轨迹，未覆盖全部近邻估计误差。','',
              '## 四张主图怎么读','',
              '1. **[图1：动作活动度与敏感度](figure1_activity_sensitivity.png)**：上排看动作本身的 RMS/std；下排看该输出分量和该组 latent 的敏感度。重点看 rx/ry 同时较低，以及 g 幅度大但局部敏感度低的例外。',
              '2. **[图2：7×7 块敏感度](figure2_component_blocks.png)**：列是输入 latent 组，行是输出动作组，每格包含全部 4×4 时间耦合。右图一列加起来为 100%，它回答某组 latent 是否通过其他动作分量起作用。',
              '3. **[图3：SVD](figure3_singular_spectrum.png)**：左上先看裁剪前蓝线；其尾部明显小于头部。左下展示低敏感子空间以旋转/夹爪坐标组合为主；右下说明去夹爪后也存在弱方向。裁剪后接近零的尾部不可直接归因于网络。',
              '4. **[图4：有限扰动验证](figure4_finite_perturbations.png)**：红/绿/蓝实线分别为最大、随机、最小奇异方向，虚线是局部预测。蓝线一直更低，但逐渐偏离虚线，说明弱方向仍弱、线性外推却不准确。纵轴是对数尺度。',
              '补充：[条件散布对照图](conditional_spread_diagnostics.png)。阴影 P05/P95 是状态分布，bootstrap CI 才是不确定性区间。','',
              '当前结果支持研究低敏感方向的探索价值，但**尚未检验删除这些方向能否提高 online RL 效率**，不能直接据此剪掉某组 latent。','',
              '[完整设计、全部指标与复现命令](REPORT.md)；[独立验证](validation.txt)。原始 28×28 Jacobian、SVD 向量、有限扰动和先验采样均在 `geometry_samples.npz`，对应各实验 CSV 保存可读统计。','']
    (out/'CONCLUSIONS.md').write_text('\n'.join(lines))

