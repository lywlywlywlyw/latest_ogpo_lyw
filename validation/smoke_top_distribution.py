"""Small-network MPPCA training and ODE/SDE/BON smoke check.

Run from the repository root with PYTHONPATH=. JAX_PLATFORMS=cpu
.venv/bin/python validation/smoke_top_distribution.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from ogpo.train_invertion_offline import make_inverse_agent
from ogpo.agents.ogpo import OGPOAgent
from ogpo.utils.config_loader import load_config, config_to_agent_dict
from ogpo.agents.modules.mppca import fit_mppca


def main():
    config = config_to_agent_dict(load_config('ogpo'))
    config.update(actor_hidden_dims=(16,16), critic_hidden_dims=(16,16),
                  noise_net_hidden_dims=(16,16), horizon_length=2, batch_size=4,
                  flow_steps=2, action_chunking=True, use_denoiser=False, useFiLM=False,
                  useTF=False, actor_backbone="mlp", critic_backbone="mlp")
    obs, action = jnp.zeros(3), jnp.zeros(2)
    teacher = OGPOAgent.create(0, obs, action, config, config['adv_clip_min'])
    agent = make_inverse_agent(teacher, dict(observations=obs, actions=action), config, 0)
    model, _ = fit_mppca(np.random.default_rng(0).normal(size=(40, 4)), rank=1)
    agent = agent.replace(top_distribution=model)
    agent = agent.reset_flow_for_retraining(obs, action)
    batch = dict(observations=jnp.zeros((4,3)), actions=jnp.zeros((4,2,2)), valid=jnp.ones((4,2)))
    for ot in (False, True):
        agent = agent.replace(config=agent.config.copy(dict(use_OT_optimization=ot)))
        agent, info = agent.bc_update(batch)
        assert np.isfinite(float(info['bc_loss']))
        print('bc_update OT=', ot, 'loss=', float(info['bc_loss']))
    for name in ('compute_flow_actions', 'sample_actions', 'sample_actions_BON'):
        result = getattr(agent, name)(jnp.zeros((4, 3)), rng=jax.random.key(1))
        print(name, jax.tree.map(lambda a: a.shape, result))
        assert all(np.isfinite(a).all() for a in jax.tree.leaves(result))
    print('agent smoke passed')


if __name__ == "__main__":
    main()
