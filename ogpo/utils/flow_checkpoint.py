"""Portable offline flow weights for compatible baseline architectures.

The canonical policy is OGPO's EMA actor, used for BC SDE/BON evaluation.
Critics, SAC actors, optimizer states, and exploration networks are not shared.
"""
import os
import pickle
import tempfile

import jax
import numpy as np


SEMANTIC_KEYS = ('horizon_length', 'flow_steps', 'time_embedding',
                 'time_embedding_dim', 'actor_layer_norm', 'act_min', 'act_max')


def _flow_modules(agent):
    """Return the flow TrainState and its online/target module names."""
    if hasattr(agent, 'bc_flow_network'):
        return 'bc_flow_network', 'modules_actor_bc_flow', None
    if hasattr(agent, 'actor_network'):
        return 'actor_network', 'modules_actor', 'modules_target_actor'
    if hasattr(agent, 'network'):
        return 'network', 'modules_actor_bc_flow', None
    raise ValueError('This baseline has no supported flow network')


def save_flow_checkpoint(agent, path, *, env_name, seed, step):
    """Export only the evaluated flow policy and its compatibility metadata."""
    network_name, online_name, target_name = _flow_modules(agent)
    params = getattr(agent, network_name).params
    source_name = target_name if target_name in params else online_name
    modules = {'flow': jax.device_get(params[source_name])}
    encoder_name = source_name + '_encoder'
    if encoder_name in params:
        modules['encoder'] = jax.device_get(params[encoder_name])
    payload = {
        'format': 'ogpo_offline_flow_v1', 'env_name': env_name,
        'seed': seed, 'bc_step': step, 'source_module': source_name,
        'config': {key: agent.config.get(key) for key in SEMANTIC_KEYS},
        'hidden_dims': tuple(agent.config.get('bc_flow_hidden_dims') or
                             agent.config['actor_hidden_dims']),
        'modules': modules,
    }
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            pickle.dump(payload, stream)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f'Saved shared offline flow to {path} (BC step {step}, {source_name})')


def _checked_weights(template, source, label):
    if jax.tree_util.tree_structure(template) != jax.tree_util.tree_structure(source):
        raise ValueError(f'{label}: flow parameter structure mismatch')
    for expected, actual in zip(jax.tree_util.tree_leaves(template),
                                jax.tree_util.tree_leaves(source)):
        if expected.shape != actual.shape:
            raise ValueError(f'{label}: parameter shape {actual.shape} != {expected.shape}; '
                             'match the offline flow architecture before loading')
    return jax.tree_util.tree_map(lambda ref, value: np.asarray(value, dtype=ref.dtype),
                                  template, source)


def load_flow_checkpoint(agent, path, *, env_name):
    """Load the same evaluated flow into every online and EMA flow module."""
    with open(os.path.expanduser(path), 'rb') as stream:
        payload = pickle.load(stream)
    if payload.get('format') != 'ogpo_offline_flow_v1':
        raise ValueError('Expected an offline flow exported by ogpo/train_offline.py')
    if payload['env_name'] != env_name:
        raise ValueError(f"Flow task {payload['env_name']} != {env_name}")
    for key, source_value in payload['config'].items():
        if agent.config.get(key) != source_value:
            raise ValueError(f'Flow setting {key}: {agent.config.get(key)} != {source_value}')
    network_name, online_name, target_name = _flow_modules(agent)
    network = getattr(agent, network_name)
    params = dict(network.params)
    if online_name not in params:
        raise ValueError(f'Baseline flow module {online_name} is missing')
    for name in (online_name, target_name):
        if name is None or name not in params:
            continue
        params[name] = _checked_weights(params[name], payload['modules']['flow'], name)
        encoder_name = name + '_encoder'
        if (encoder_name in params) != ('encoder' in payload['modules']):
            raise ValueError('Flow encoder configuration mismatch')
        if encoder_name in params:
            params[encoder_name] = _checked_weights(
                params[encoder_name], payload['modules']['encoder'], encoder_name)
    return agent.replace(**{network_name: network.replace(params=params)})
