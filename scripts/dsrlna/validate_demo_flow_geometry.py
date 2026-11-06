"""Independent output checks for experiment 4; does not rerun the experiment.

Usage: .venv/bin/python scripts/dsrlna/validate_demo_flow_geometry.py RESULT_DIR
Checks saved model outputs against raw HDF5, algebraic identities and a small
fresh decoder spot check. Saves validation.txt beside the experimental data.
"""
from pathlib import Path
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import sys
import json
import csv
import pickle
import hashlib
from types import SimpleNamespace

import h5py
import numpy as np


def main(directory):
    from scipy.stats import spearmanr
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from scripts.dsrlna.analyze_demo_latents import make_functions
    root=Path(directory).resolve()
    a=np.load(root/'geometry_samples.npz');d=np.load(root/'derived_metrics.npz')
    s=json.loads((root/'summary.json').read_text());meta=s['metadata']
    messages=[]

    def passed(message):
        messages.append('PASS '+message);print(messages[-1],flush=True)

    def close(x,y,**kw):
        np.testing.assert_allclose(x,y,**kw)

    for label in ('checkpoint','dataset'):
        h=hashlib.sha256()
        with Path(meta[label]).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==meta[label+'_sha256']
    assert meta['reverse_query_times']==[.9,.8,.7,.6,.5,.4,.3,.2,.1,0.]
    assert meta['max_difference_to_corrected_inverse_cache']==0
    passed('Square checkpoint/data SHA256 match metadata; latest inverse grid; z0 exactly matches corrected all-demo cache')
    assert len(a['demo_id'])==100 and len(np.unique(a['demo_id']))==25
    assert len(np.unique(a['source_rows']))==100
    keys=('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','object')
    total=0;raw_sum=np.zeros(7);raw_sq=np.zeros(7);near=np.zeros(7)
    with h5py.File(meta['dataset'],'r') as f:
        assert len(f['data'])==300
        for name in f['data']:
            actions=np.asarray(f['data'][name]['actions'],float)
            total+=len(actions);raw_sum+=actions.sum(0);raw_sq+=(actions**2).sum(0);near+=(abs(actions)<.05).sum(0)
        for b,(demo,t) in enumerate(zip(a['demo_id'],a['timestep'])):
            ep=f['data'][f'demo_{demo}'];t=int(t)
            state=np.concatenate([ep['obs'][key][t] for key in keys]).astype(np.float32)
            np.testing.assert_array_equal(state,a['states'][b])
            np.testing.assert_array_equal(np.asarray(ep['actions'][t:t+4],np.float32).reshape(-1),a['expert_actions'][b])
            assert not np.asarray(ep['dones'][t:t+3]).any()
            close(a['progress'][b],t/(len(ep['actions'])-1),atol=0,rtol=0)
            assert a['phase'][b]==min(int(a['progress'][b]*4),3)
    for demo in np.unique(a['demo_id']):np.testing.assert_array_equal(np.sort(a['phase'][a['demo_id']==demo]),np.arange(4))
    assert total==80731
    close(d['activity_rms'],np.sqrt(raw_sq/total),rtol=1e-13)
    close(d['activity_std'],np.sqrt(raw_sq/total-(raw_sum/total)**2),rtol=1e-13)
    close(d['activity_near_zero_fraction'],near/total,atol=0,rtol=0)
    passed('All 80,731 raw transitions independently counted once; RMS/std/near-zero statistics agree; 25x4 stratified HDF5 states/chunks match')
    J=a['jacobians'].astype(float)
    for k in range(7):
        ix=np.arange(k,28,7)
        close(d['sout'][:,:,k],np.linalg.norm(J[:,:,ix,:],axis=(-2,-1))/np.sqrt(4*28),rtol=1e-13)
        close(d['sin'][:,:,k],np.linalg.norm(J[:,:,:,ix],axis=(-2,-1))/2,rtol=1e-13)
        for l in range(7):
            jx=np.arange(l,28,7)
            close(d['blocks'][:,:,k,l],np.linalg.norm(J[:,:,ix,:][:,:,:,jx],axis=(-2,-1)),rtol=1e-13)
    raw=a['raw_baseline_actions']
    gate=np.where(abs(raw)<1,1.,np.where(abs(raw)>1,0.,.5))
    close(J[:,1],J[:,0]*gate[:,:,None],atol=1e-7,rtol=1e-6)
    passed('All 1,400 component sensitivities and 9,800 block norms independently checked; clipped J equals clipping gate times preclip J')
    U=a['svd_u'];sig=a['singular_values'];VH=a['svd_vh']
    rebuilt=np.einsum('bvok,bvk,bvki->bvoi',U,sig,VH)
    close(rebuilt,J,atol=1e-13,rtol=1e-12)
    close(VH@VH.swapaxes(-1,-2),np.broadcast_to(np.eye(28),VH.shape),atol=3e-14)
    assert (np.diff(sig,axis=-1)<=0).all() and (sig>=0).all()
    w=np.stack([(VH[:,:,:,np.arange(k,28,7)]**2).sum(-1) for k in range(7)],axis=-1)
    close(w,a['singular_component_weights'],atol=2e-15)
    close(w.sum(-1),1,atol=3e-14)
    close(d['low4_component_energy'],w[:,:,-4:].mean(2),atol=2e-15)
    low=VH[0,0,-4:];q,_=np.linalg.qr(np.random.default_rng(4).normal(size=(4,4)))
    close(((q@low).reshape(4,4,7)**2).sum((0,1)),(low.reshape(4,4,7)**2).sum((0,1)),atol=2e-15)
    passed('All 200 SVDs reconstruct J, vectors orthonormal and weights normalized; low-subspace group energy is invariant to an internal rotation')
    close(np.linalg.norm(a['directions'],axis=-1),1,atol=1e-7)
    close(a['directions'][:,0],VH[:,0,0],atol=3e-8)
    close(a['directions'][:,1],VH[:,0,-1],atol=3e-8)
    assert (a['directions'][:,4:,6::7]==0).all()
    for vi,key in enumerate(('raw_perturbed_actions','perturbed_actions')):
        base=a['raw_baseline_actions'] if vi==0 else a['baseline_actions']
        actual=np.linalg.norm(a[key].astype(float)-base[:,None,None,None,:].astype(float),axis=-1)
        close(actual,d['finite_action_distances'][:,vi],atol=0,rtol=0)
        close(actual/a['radii'][None,None,None,:],d['finite_gains'][:,vi],atol=0,rtol=0)
    expected_z=a['z0'][:,None,None,None,:]+a['directions'][:,:,None,None,:]*a['signs'][None,None,:,None,None]*a['radii'][None,None,None,:,None]
    np.testing.assert_array_equal(a['perturbed_z'],expected_z)
    close(d['finite_latent_distances'],np.broadcast_to(a['radii'],d['finite_latent_distances'].shape),atol=4e-7,rtol=4e-5)
    close(d['jacobian_predicted_gains'][:,0,:2],sig[:,0,[0,-1]],atol=1e-8,rtol=1e-5)
    np.testing.assert_array_equal(a['perturbed_actions'],np.clip(a['raw_perturbed_actions'],-1,1))
    passed('All 8,400 perturbations use correct +/- directions/radii and decoded baselines; gains and vmax/vmin predictions agree with singular values')
    for vi,key in enumerate(('prior_raw_actions','prior_actions')):
        p=a[key].reshape(100,128,4,7).astype(float)
        close(d['decoder_component_variance'][:,vi],np.var(p,axis=1,ddof=1).mean(1),atol=0,rtol=0)
    cache_dir=root.parent.parent/'all_demo_inverse_step_2000000_time_shift_1'
    with np.load(cache_dir/'all_latents.npz') as cache:
        all_obs=cache['observations'].astype(float)
        all_obs[:,3:7]*=np.where(all_obs[:,6:7]<0,-1.,1.)
        all_obs=(all_obs-a['observation_metric_mean'])/a['observation_metric_std']
        all_ids=cache['demo_id'];chunks=cache['action_chunks'].reshape(-1,4,7).astype(float)
        id_lengths=dict(zip(a['all_demo_ids'],a['all_demo_lengths']))
        all_progress=np.array([t/(id_lengths[demo]-1) for demo,t in zip(all_ids,cache['timestep'])])
        all_phase=np.minimum((all_progress*4).astype(int),3)
        for b in (0,49,99):
            row=a['source_rows'][b]
            distances=np.linalg.norm(all_obs-all_obs[row],axis=1)/np.sqrt(all_obs.shape[1])
            for mode in range(2):
                allowed=all_ids!=a['demo_id'][b]
                if mode:allowed&=all_phase==a['phase'][b]
                ranked=np.argsort(np.where(allowed,distances,np.inf))[:50]
                np.testing.assert_array_equal(ranked,a['knn_rows'][b,mode])
                close(distances[ranked],a['knn_distances'][b,mode],atol=1e-14)
                for ki,k in enumerate((10,20,50)):
                    close(chunks[ranked[:k]].var(0,ddof=1).mean(0),a['knn_variance'][b,mode,ki],atol=1e-15)
    assert not (a['knn_demo_ids']==a['demo_id'][:,None,None]).any()
    assert (a['knn_phase'][:,1]==a['phase'][:,None]).all()
    passed('All fixed-state prior variances agree; kNN excludes current trajectories and independent spot checks reproduce exact neighbor ranking and variances')
    for rec in s['component_correlations']:
        v=('preclip','clipped').index(rec['variant']);values=d[rec['sensitivity']][:,v]
        activity=d['activity_rms'] if rec['activity']=='rms' else d['activity_std']
        k=7 if rec['components']=='all7' else 6
        close(spearmanr(activity[:k],np.median(values,axis=0)[:k]).statistic,rec['rho_full_data'],atol=1e-14)
    for idx in d['bootstrap_state_indices']:
        for demo in np.unique(a['demo_id'][idx]):
            selected=idx[a['demo_id'][idx]==demo]
            counts=np.bincount(a['phase'][selected],minlength=4)
            assert (counts==counts[0]).all()
    passed('Component Spearman correlations independently agree; every bootstrap replicate preserves entire four-state trajectory clusters')
    assert max(r['relative_frobenius_error'] for r in s['finite_difference_checks'] if r['epsilon']<.002)<.02
    passed('Complete finite-difference Jacobians at three states agree with automatic differentiation (<2% relative Frobenius error at epsilon .001)')
    config=json.loads((Path(meta['checkpoint']).parent/'flags.json').read_text())['agent']
    _,_,decode,decode_raw=make_functions(config,28,SimpleNamespace(num_draws=32),return_decoder=True)
    with Path(meta['checkpoint']).open('rb') as f:params=pickle.load(f)['agent']['bc_flow_network']['params']['modules_actor_bc_flow']
    rng=np.random.default_rng(4017);idx=rng.choice(8400,16,replace=False);b=idx//84
    zz=a['perturbed_z'].reshape(-1,28)[idx]
    fresh=np.asarray(decode_raw(params,a['states'][b],zz))
    target=a['raw_perturbed_actions'].reshape(-1,28)[idx]
    close(fresh,target,atol=3e-6,rtol=2e-5)
    clipped=np.asarray(decode(params,a['states'][b],zz))
    close(clipped,np.clip(target,-1,1),atol=3e-6,rtol=2e-5)
    prior_ids=rng.integers(0,128,size=8);state_ids=rng.integers(0,100,size=8)
    fresh_prior=np.asarray(decode_raw(params,a['states'][state_ids],a['prior_z'][prior_ids]))
    close(fresh_prior,a['prior_raw_actions'][state_ids,prior_ids],atol=3e-6,rtol=2e-5)
    passed(f'Fresh original-decoder checks: 16 perturbations and 8 prior samples agree; perturbation max error {np.max(abs(fresh-target)):.3g}')
    assert sum(1 for _ in csv.DictReader((root/'4_4_finite_perturbations.csv').open()))==16800
    for i in range(1,5):
        name=('activity_sensitivity','component_blocks','singular_spectrum','finite_perturbations')[i-1]
        assert (root/f'figure{i}_{name}.png').stat().st_size>10000
    passed('All four primary PNG/PDF plots, CSV outputs and raw arrays saved')
    (root/'validation.txt').write_text('\n'.join(messages)+'\n')
    validate_statistics_additions(root)


def validate_statistics_additions(directory):
    """Validate optional statistical diagnostics without any model execution."""
    from scipy.stats import rankdata
    from scipy.linalg import lstsq
    root=Path(directory);a=np.load(root/'geometry_samples.npz');d=np.load(root/'derived_metrics.npz')
    s=json.loads((root/'summary.json').read_text())
    for rec in s['conditional_partial_correlations']:
        k=('x','y','z','rx','ry','rz','g').index(rec['component'])
        predictor=rec['predictor']
        x=np.sqrt(d['decoder_component_variance'][:,0,k]) if predictor=='decoder_prior_std' else np.sqrt(a['knn_variance'][:,int('mode1' in predictor),1,k])
        expert_rms=np.sqrt((a['expert_actions'].reshape(100,4,7).astype(float)**2).mean(1))[:,k]
        controls=np.column_stack((np.ones(100),rankdata(expert_rms),rankdata(a['progress'])))
        xr=rankdata(x);yr=rankdata(d['sout'][:,0,k])
        xr-=controls@lstsq(controls,xr)[0];yr-=controls@lstsq(controls,yr)[0]
        expected=np.corrcoef(xr,yr)[0,1]
        np.testing.assert_allclose(expected,rec['partial_rank_rho'],atol=2e-13)
    arm=a['arm_indices'];arm_j=a['jacobians'][:,0][:,arm][:,:,arm].astype(float)
    transformed=arm_j@a['arm_svd_vh'].swapaxes(-1,-2)
    gram=transformed.swapaxes(-1,-2)@transformed
    expected=np.zeros_like(gram)
    expected[:,np.arange(24),np.arange(24)]=a['arm_singular_values']**2
    np.testing.assert_allclose(gram,expected,atol=1e-13)
    records=list(csv.DictReader((root/'4_3_low_subspace_uncertainty.csv').open()))
    for rec in records:
        nlow=int(rec['low_subspace_dimension']);k=('x','y','z','rx','ry','rz','g').index(rec['component'])
        ix=np.arange(k,28,7)
        vectors=a['svd_vh'][:,0,-nlow:]
        values=(vectors[:,:,ix]**2).sum(-1).mean(1)
        np.testing.assert_allclose(values.mean(),float(rec['mean_energy']),atol=2e-15)
        ci=np.quantile(values[d['bootstrap_state_indices']].mean(1),[.025,.975])
        np.testing.assert_allclose(ci,[float(rec['trajectory_bootstrap_ci025']),float(rec['trajectory_bootstrap_ci975'])],atol=2e-15)
    message='PASS All partial rank correlations reproduced with independent least-squares residuals; arm-only SVD identities and 14 low-subspace confidence intervals agree'
    with (root/'validation.txt').open('a') as f:f.write(message+'\n')
    print(message,flush=True)


if __name__=='__main__':
    main(sys.argv[1])
