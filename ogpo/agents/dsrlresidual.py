import copy
from functools import partial
from typing import Any, Tuple

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from ogpo.networks.encoders import encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks import Actor, Value, ActorVectorField, LogParam
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.agents.modules.igp_targets_helper import get_flow_targets
from ogpo.agents.modules.maintenance import (
    restore_critic_params as _restore_critic_params,
)
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    apply_chunking_mask,
)
from ogpo.agents.modules.q_helper import (
    get_q_values as _get_q_values,
)
from ogpo.agents.modules.pg_helper import (
    sample_flow_actions_ode,
)

class ResidualMean(nn.Module):
    """Zero-initialized delta(s, mu_b); only the mean can change online."""
    hidden_dims: Any
    action_dim: int
    layer_norm: bool = False
    encoder: Any = None
    obs_kwargs: Any = None

    @nn.compact
    def __call__(self, observations, base_mean):
        from ogpo.networks.modules import MLP, TwoTierObsEncoder
        x = self.encoder(observations) if self.encoder is not None else observations
        kw = self.obs_kwargs
        if kw['obs_two_tier']:
            x = TwoTierObsEncoder(
                img_feat_dim=kw['two_tier_img_dim'],
                proprio_dim=kw['two_tier_proprio_dim'],
                fused_dim=kw['two_tier_fused_dim'] or self.hidden_dims[0])(x)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(
            jnp.concatenate([x, base_mean], axis=-1))
        return nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.zeros)(x)


class DSRLResidualAgent(flax.struct.PyTreeNode):
    """Frozen flow/base Gaussian with a KL-constrained residual mean actor."""

    rng: Any
    actor_network: TrainState
    critic_network: TrainState
    bc_flow_network: TrainState
    config: Any = nonpytree_field()
    action_critic_network: TrainState
    base_policy_network: TrainState

    def _action_td_target(self, batch, rng):
        """Bootstrap through the frozen decoder and target action critic."""
        _, sample_rng = jax.random.split(rng)
        dist = self.policy_distribution(batch['next_observations'])
        next_z, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        next_actions = self.noise_to_actions_public(batch['next_observations'], next_z)
        next_qs = self.action_critic_network.select('target_critic')(
            batch['next_observations'], actions=next_actions)
        next_q = (jnp.min(next_qs, axis=0) if self.config['q_agg'] == 'min'
                  else jnp.mean(next_qs, axis=0))
        if self.config['entropy_backup']:
            next_q -= self.actor_network.select('temp')() * jnp.clip(log_probs, -100, 100)
        # Use actual executed chunk discounts, including terminal/short chunks.
        discounts = batch.get('discounts', self.config['discount'] ** self.config['horizon_length'])
        target = jax.lax.stop_gradient(batch['rewards'] + discounts * batch['masks'] * next_q)
        return target, next_z, log_probs

    def action_critic_loss(self, batch, grad_params, rng):
        target, _, _ = self._action_td_target(batch, rng)
        q = self.action_critic_network.select('critic')(
            batch['observations'], actions=batch['true_actions'], params=grad_params)
        loss = jnp.mean(jnp.square(q - target))
        return loss, {'loss': loss, 'q_mean': q.mean(), 'target_q_mean': target.mean()}

    def _action_step(self, batch, rng):
        state, info = self.action_critic_network.apply_loss_fn(
            lambda p: self.action_critic_loss(batch, p, rng))
        self.target_update(state, 'critic')
        return self.replace(action_critic_network=state), {
            f'action_critic/{k}': v for k, v in info.items()}

    def critic_loss(self, batch, grad_params, rng):
        """Distill Qa(s, flow(s,z)) at prior or frozen-base samples.

        Only replay states are reused: replay actions and TD fields do not enter
        this loss. Base sampling excludes the online residual. Match ensemble
        heads individually, as in DSRL-NA.
        """
        observations = batch['observations']
        use_base = self.config.get('use_base_distribution_in_qdistill', False)
        if use_base:
            z = jax.lax.stop_gradient(
                self.base_distribution(observations).sample(seed=rng))
        else:
            z = jax.random.normal(
                rng, (observations.shape[0], self.config['noise_dim']),
                dtype=observations.dtype)
        actions = self.noise_to_actions_public(observations, z)
        target_q = jax.lax.stop_gradient(
            self.action_critic_network.select('critic')(observations, actions=actions))
        q = self.critic_network.select('critic')(
            observations, actions=z, params=grad_params)
        # Keep the existing mean-over-heads MSE convention for critic losses.
        loss = jnp.mean(jnp.square(q - target_q))
        return loss, {
            'critic_loss': loss, 'distillation_loss': loss,
            'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
            'target_q_mean': target_q.mean(),
            'distill_noise_mean': z.mean(), 'distill_noise_std': z.std(),
            'distill_uses_base_distribution': jnp.float32(use_base),
        }

    def base_distribution(self, observations):
        """Frozen reference distribution shared by sampling, residual inputs and KL."""
        if self.config.get('use_normal_base_encoder', False):
            batch_shape = observations.shape[:-len(self.config['ob_dims'])]
            shape = (*batch_shape, self.config['noise_dim'])
            return distrax.MultivariateNormalDiag(
                loc=jnp.zeros(shape, dtype=observations.dtype),
                scale_diag=jnp.ones(shape, dtype=observations.dtype))
        base = self.base_policy_network.select('actor')(observations)
        mean, std = jax.tree.map(jax.lax.stop_gradient, (base.mean(), base.stddev()))
        if self.config.get('use_one_variance', False):
            std = jnp.ones_like(std)
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

    def policy_distribution(self, observations, params=None):
        base = self.base_distribution(observations)
        delta = self.actor_network.select('actor')(observations, base.mean(), params=params)
        return distrax.MultivariateNormalDiag(
            loc=base.mean() + delta, scale_diag=base.stddev())

    def actor_loss(self, batch, grad_params, rng):
        dist = self.policy_distribution(batch['observations'], params=grad_params)
        z, log_probs = dist.sample_and_log_prob(seed=rng)
        qs = self.critic_network.select('critic')(batch['observations'], actions=z)
        q = jnp.min(qs, axis=0) if self.config['q_agg'] == 'min' else jnp.mean(qs, axis=0)
        base = self.base_distribution(batch['observations'])
        delta = dist.mean() - jax.lax.stop_gradient(base.mean())
        multiplier = self.actor_network.select('kl_multiplier')()
        # The frozen-covariance Gaussian has no mean-dependent entropy term.
        actor_loss = -jnp.mean(q)
        use_kl_loss = self.config.get('use_kl_loss', True)
        kl = jnp.float32(0.0)
        dual_loss = jnp.float32(0.0)
        constrained_loss = actor_loss
        if use_kl_loss:
            if self.config.get('use_only_mean_klloss', False):
                # Mean MSE ignores variance and averages over batch and latent dimensions.
                kl = (0.5 * jnp.sum(jnp.square(delta), axis=-1)).mean()
            else:
                # Floor only the denominator of this KL surrogate; keep sampling scales.
                kl_std = jnp.maximum(dist.stddev(), jnp.exp(self.config['base_log_std_min']))
                kl = (0.5 * jnp.sum(jnp.square(delta / kl_std), axis=-1)).mean()
            constrained_loss = actor_loss + multiplier * (kl - self.config['kl_epsilon'])
            # Dual descent increases lambda when KL exceeds epsilon. Stop gradients
            # on both sides so actor and dual receive their respective objectives.
            dual = self.actor_network.select('kl_multiplier')(params=grad_params)
            dual_loss = dual * jax.lax.stop_gradient(self.config['kl_epsilon'] - kl)
        total_loss = constrained_loss + dual_loss
        return total_loss, {
            'total_loss': total_loss, 'actor_loss': actor_loss,
            'kl': kl, 'kl_enabled': jnp.float32(use_kl_loss),
            'kl_multiplier': multiplier, 'dual_loss': dual_loss,
            'log_probs_mean': log_probs.mean(), 'entropy': -log_probs.mean(),
            'q_actor_mean': q.mean(), 'actor_mode': dist.mode().mean(),
            'base_std_mean': dist.stddev().mean(), 'delta_rms': jnp.sqrt(jnp.mean(delta ** 2)),
        }

    @jax.jit
    def actions_to_noise(self, observations, actions):
        """Invert demo chunks using the repository's shared reverse Euler rule."""
        from ogpo.agents.modules.flow_inverse import invert_flow_actions_ode
        return invert_flow_actions_ode(
            self.bc_flow_network.select('actor_bc_flow'), observations, actions,
            self.config['flow_steps'], is_encoded=False)

    @jax.jit
    def base_bc_update(self, batch):
        """Gaussian NLL learns both mu_b(s) and each sigma_b,i(s) offline."""
        if self.config.get('use_normal_base_encoder', False):
            return self, {'base_bc_skipped': jnp.float32(1.0)}
        def loss_fn(p):
            dist = self.base_policy_network.select('actor')(batch['observations'], params=p)
            loss = -dist.log_prob(batch['actions']).mean()
            return loss, {'base_bc_nll': loss, 'base_std_mean': dist.stddev().mean()}
        state, info = self.base_policy_network.apply_loss_fn(loss_fn)
        return self.replace(base_policy_network=state), info

    def reset_flow_for_retraining(self, ex_observations, ex_actions):
        """Start a fresh flow after inverse labels and the frozen base are fitted."""
        rng, init_rng = jax.random.split(self.rng)
        if self.config['action_chunking']:
            ex_actions = jnp.concatenate([ex_actions] * self.config['horizon_length'], axis=-1)
        network = self.bc_flow_network
        params = network.model_def.init(
            init_rng, actor_bc_flow=(ex_observations, ex_actions, ex_actions[..., :1]))['params']
        # Recreate Adam moments and its step counter as well as network weights.
        state = TrainState.create(network.model_def, params, network.tx)
        return self.replace(rng=rng, bc_flow_network=state)

    def actor_total_loss(self, batch, grad_params, rng):
        a_loss, a_info = self.actor_loss(batch, grad_params, rng)
        info = {f'actor/{k}': v for k, v in a_info.items()}
        return a_loss, info

    def critic_total_loss(self, batch, grad_params, rng):
        c_loss, c_info = self.critic_loss(batch, grad_params, rng)
        info = {f'critic/{k}': v for k, v in c_info.items()}
        return c_loss, info

    def bc_loss(self, batch, grad_params, rng, use_base_distribution=False):
        batch_actions = preprocess_actions(batch, self.config["action_chunking"])
        batch_size = batch_actions.shape[0]

        rng, targets_rng = jax.random.split(rng)
        if use_base_distribution:
            _, noise_rng, time_rng = jax.random.split(targets_rng, 3)
            x_0 = jax.lax.stop_gradient(
                self.base_distribution(batch['observations']).sample(seed=noise_rng))
            t = jax.random.uniform(time_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * batch_actions
            vel = batch_actions - x_0
        else:
            x_t, vel, t = get_flow_targets(
                batch['observations'], batch_actions, targets_rng)
        pred = self.bc_flow_network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        valid_mask = batch.get("valid", jnp.ones((batch_size, self.config["horizon_length"])))
        bc_flow_loss = apply_chunking_mask(
            jnp.square(pred - vel), valid_mask, batch_size,
            self.config["horizon_length"], self.config["action_dim"],
            self.config["action_chunking"]
        )

        return bc_flow_loss, {
            'bc_flow_loss': bc_flow_loss,
            'flow_uses_base_distribution': jnp.float32(use_base_distribution),
        }

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch: dict) -> Tuple['DSRLResidualAgent', dict]:
        new_rng, rng1, rng2 = jax.random.split(agent.rng, 3)

        def actor_loss_fn(p):
            return agent.actor_total_loss(batch, p, rng1)
        new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        agent, action_info = agent._action_step(batch, rng2)

        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, rng2)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        return agent.replace(
            actor_network=new_actor_state, critic_network=new_critic_state, rng=new_rng,
        ), {**actor_info, **critic_info, **action_info}

    @staticmethod
    def _update_offline(agent, batch: dict, use_base_distribution=False) -> Tuple['DSRLResidualAgent', dict]:
        new_rng, rng1 = jax.random.split(agent.rng, 2)
        def bc_loss_fn(p):
            return agent.bc_loss(batch, p, rng1, use_base_distribution)
        new_bc_flow_state, info = agent.bc_flow_network.apply_loss_fn(bc_loss_fn)
        return agent.replace(bc_flow_network=new_bc_flow_state, rng=new_rng), info

    @partial(jax.jit, static_argnames=('use_base_distribution',))
    def bc_update(self, batch, use_base_distribution=False):
        return self._update_offline(self, batch, use_base_distribution)

    @jax.jit
    def batch_update(self, batch, batch_success=None, success_flag=False):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @staticmethod
    def _critic_only_update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)
        agent, action_info = agent._action_step(batch, rng)
        state, critic_info = agent.critic_network.apply_loss_fn(
            lambda p: agent.critic_total_loss(batch, p, rng))
        agent.target_update(state, 'critic')
        return agent.replace(critic_network=state, rng=new_rng), {
            **critic_info, **action_info}

    @jax.jit
    def batch_critic_only_update(self, batch):
        """Extra Qa/Qz updates for utd_q > utd_pi; actor and flow stay fixed."""
        agent, infos = jax.lax.scan(self._critic_only_update, self, batch)
        return agent, jax.tree.map(lambda x: x.mean(), infos)

    @partial(jax.jit, static_argnames=('num_samples',))
    def sample_base_actions(self, observations, rng=None, num_samples=1):
        """Frozen-base Flow evaluation; Best-of-N follows OGPO's action-Q rule."""
        single = observations.ndim == len(self.config['ob_dims'])
        if single:
            observations = observations[None]
        batch_size = observations.shape[0]
        repeated = jnp.repeat(observations, num_samples, axis=0)
        _, sample_rng = jax.random.split(rng)
        z = self.base_distribution(repeated).sample(seed=sample_rng)
        actions = self.noise_to_actions_public(repeated, z)
        if num_samples > 1:
            qs = self.action_critic_network.select('target_critic')(repeated, actions=actions)
            qs = qs.reshape(qs.shape[0], batch_size, num_samples)
            if self.config.get('subsample_bon', True):
                selection_rng, _ = jax.random.split(rng)
                _, selection_rng = jax.random.split(selection_rng)
                indices = jax.random.randint(selection_rng, (2, batch_size, num_samples), 0, qs.shape[0])
                qs = jnp.take_along_axis(qs, indices, axis=0)
            best = jnp.argmax(jnp.min(qs, axis=0), axis=-1)
            actions = actions.reshape(batch_size, num_samples, -1)[jnp.arange(batch_size), best]
        return actions[0] if single else actions

    @jax.jit
    def sample_noise(self, observations, rng=None, deterministic=False):
        dist = self.policy_distribution(observations)
        if deterministic:
            return dist.mode() if hasattr(dist, 'mode') else dist.mean()
        if self.config.get('add_noise2mean_insample', False):
            mean_rng, rng = jax.random.split(rng)
            mean = dist.mean()
            dist = distrax.MultivariateNormalDiag(
                loc=mean + jax.random.normal(mean_rng, mean.shape, dtype=mean.dtype),
                scale_diag=dist.stddev())
        return dist.sample(seed=rng)

    @jax.jit
    def sample_actions(self, observations, rng=None, deterministic=False):
        actions = self.sample_noise(observations, rng=rng, deterministic=deterministic)

        if self.config["noise_chunk_length"] > 0:
            assert self.config["noise_chunk_length"] == 1
            actions = jnp.concatenate([actions] * self.config["horizon_length"], axis=-1)

        actions = self.compute_flow_actions(observations, actions)
        return actions

    @jax.jit
    def noise_to_actions_public(self, observations, actor_noise):
        if self.config["noise_chunk_length"] > 0:
            assert self.config["noise_chunk_length"] == 1
            actor_noise = jnp.concatenate([actor_noise] * self.config["horizon_length"], axis=-1)
        return self.compute_flow_actions(observations, actor_noise)

    @jax.jit
    def sample_actions_normal(self, observations, rng=None):
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'] * \
                    (self.config['horizon_length'] if self.config["action_chunking"] else 1),
            ),
        )
        actions = self.compute_flow_actions(observations, noises)
        return actions

    def get_q_values(self, observations, actions, is_encoded=False):
        return _get_q_values(
            critic_fn=self.critic_network.select('target_critic'),
            observations=observations, actions=actions,
            q_agg=self.config['q_agg'], is_encoded=is_encoded,
        )

    @jax.jit
    def compute_flow_actions(self, observations, noises=None, params=None, rng=None):
        add_batch_dim = False
        if (self.config['encoder'] is not None and observations.ndim == 3) or \
           (self.config['encoder'] is None and observations.ndim == 1):
            observations = observations[None, ...]
            if noises is not None:
                noises = noises[None, ...]
            add_batch_dim = True

        act_dim = self.config['noise_dim']

        if self.config['encoder'] is not None:
            observations = self.bc_flow_network(
                observations=observations, params=params, method='call_submodule',
                submodule='actor_bc_flow', submethod='encode')

        def actor_fn(obs, actions, t, is_encoded=True):
            if params is not None:
                return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded, params=params)
            else:
                return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded)

        actions = sample_flow_actions_ode(
            actor_fn=actor_fn, observations=observations, rng=rng,
            flow_steps=self.config['flow_steps'], act_dim=act_dim,
            act_min=-1.0, act_max=1.0, clip_intermediate=False,
            noises=noises, is_encoded=True,
        )

        if add_batch_dim:
            actions = actions[0]

        return actions

    def restore_critic_params(self, critic_path):
        _restore_critic_params(self.critic_network, critic_path)

    def restore_actor_params(self, actor_path):
        import pickle
        with open(actor_path, 'rb') as f:
            source = pickle.load(f)['agent']
        self.actor_network.params.update(source['actor_network']['params'])
        if not self.config.get('use_normal_base_encoder', False):
            self.base_policy_network.params.update(source['base_policy_network']['params'])

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        if config['noise_chunk_length'] != -1:
            raise ValueError('DSRLResidual demo inversion requires noise_chunk_length=-1')
        if config.get('kl_epsilon', 0.1) < 0 or config.get('init_kl_multiplier', 1.0) <= 0:
            raise ValueError('Require kl_epsilon >= 0 and init_kl_multiplier > 0')
        config = dict(config)
        config.setdefault('use_invertionencoder_retrainflow', False)
        config.setdefault('invertionencoder_retrainflow_steps', None)
        if config['use_invertionencoder_retrainflow']:
            if config.get('use_normal_base_encoder', False):
                raise ValueError('Flow retraining requires use_normal_base_encoder=false and a trained inverse encoder')
            if (config['invertionencoder_retrainflow_steps'] is not None
                    and config['invertionencoder_retrainflow_steps'] < 1):
                raise ValueError('invertionencoder_retrainflow_steps must be positive')
        config.setdefault('use_normal_base_encoder', False)
        config.setdefault('use_one_variance', False)
        config.setdefault('add_noise2mean_insample', False)
        config.setdefault('use_base_distribution_in_qdistill', False)
        config.setdefault('use_kl_loss', True)
        config.setdefault('use_only_mean_klloss', False)
        config.setdefault('kl_epsilon', 0.1)
        config.setdefault('init_kl_multiplier', 1.0)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        if config["noise_chunk_length"] == -1:
            noise_dim = full_action_dim
            ex_noise = full_actions
        elif config["noise_chunk_length"] == 1:
            noise_dim = action_dim * config["noise_chunk_length"]
            ex_noise = jnp.concatenate([ex_actions] * config["noise_chunk_length"], axis=-1)
        else:
            raise ValueError(f"Invalid noise chunk length: {config['noise_chunk_length']}")

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()

        # Two-tier obs encoder: splits encoded [image_features, proprio] obs into
        # separate towers so proprio isn't buried in a high-dim flat MLP.
        _two_tier_kwargs = dict(
            obs_two_tier=config.get('obs_two_tier', False),
            two_tier_img_dim=config.get('_two_tier_img_dim', 0),
            two_tier_proprio_dim=config.get('_two_tier_proprio_dim', 0),
            two_tier_fused_dim=config.get('two_tier_fused_dim', 0),
        )

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            critic_loss_type=config['critic_loss_type'],
            num_bins=config['num_bins'],
            q_min=config['q_min'],
            q_max=config['q_max'],
            **_two_tier_kwargs,
        )

        base_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=noise_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
            tanh_squash=False,
            state_dependent_std=True,
            **_two_tier_kwargs,
        )

        actor_def = ResidualMean(
            hidden_dims=config['actor_hidden_dims'], action_dim=noise_dim,
            layer_norm=config['actor_layer_norm'], encoder=encoders.get('actor'),
            obs_kwargs=flax.core.FrozenDict(_two_tier_kwargs))
        # Retain the scalar for checkpoint compatibility and optional entropy backup;
        # the online mean-actor objective does not optimize temperature.
        temp_def = LogParam(init_value=config['init_temperature'])

        time_emb_type = config.get('time_embedding', 'scalar')
        time_emb_module = None
        if time_emb_type == 'sinusoidal':
            time_emb_module = SinusoidalTimeEmbedding(embed_dim=config.get('time_embedding_dim', 32))

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config.get('bc_flow_hidden_dims') or config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            time_embedding=time_emb_module,
            **_two_tier_kwargs,
        )

        actor_nets = {
            'actor': actor_def,
            'temp': temp_def,
            'kl_multiplier': LogParam(init_value=config['init_kl_multiplier']),
        }
        actor_args = {
            'actor': (ex_observations, ex_noise),
            'temp': (),
            'kl_multiplier': (),
        }

        critic_nets = {
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def),
        }
        critic_args = {
            'critic': (ex_observations, ex_noise),
            'target_critic': (ex_observations, ex_noise),
        }

        bc_flow_nets = {
            'actor_bc_flow': actor_bc_flow_def,
        }
        bc_flow_args = {
            'actor_bc_flow': (ex_observations, full_actions, ex_times),
        }

        base_net_def = ModuleDict({"actor": base_def})
        actor_net_def = ModuleDict(actor_nets)
        critic_net_def = ModuleDict(critic_nets)
        bc_flow_net_def = ModuleDict(bc_flow_nets)

        actor_lr = float(config['lr_actor'])
        critic_lr = float(config['lr_critic'])
        bc_flow_lr = float(config['lr_actor_bc_flow'])

        actor_opt_type = config['optimizer_actor']
        critic_opt_type = config['optimizer_critic']
        bc_flow_opt_type = config['optimizer_actor_bc_flow']

        clip_norm = float(config.get('clip_grad_norm', 1000.0))

        def get_optimizer(lr, opt_type):
            if opt_type == 'adam':
                base = optax.adam(learning_rate=lr)
            elif opt_type == 'adamw':
                base = optax.adamw(learning_rate=lr)
            elif opt_type == 'sgd':
                base = optax.sgd(learning_rate=lr)
            else:
                raise ValueError(f"Unknown optimizer type: {opt_type}")
            return optax.chain(optax.clip_by_global_norm(clip_norm), base)

        actor_tx = get_optimizer(actor_lr, actor_opt_type)
        critic_tx = get_optimizer(critic_lr, critic_opt_type)
        bc_flow_tx = get_optimizer(bc_flow_lr, bc_flow_opt_type)

        rng, actor_rng, critic_rng, bc_flow_rng = jax.random.split(rng, 4)
        actor_params = actor_net_def.init(actor_rng, **actor_args)['params']
        critic_params = critic_net_def.init(critic_rng, **critic_args)['params']
        bc_flow_params = bc_flow_net_def.init(bc_flow_rng, **bc_flow_args)['params']
        action_critic_params = critic_net_def.init(
            jax.random.fold_in(critic_rng, 917),
            critic=(ex_observations, full_actions),
            target_critic=(ex_observations, full_actions),
        )['params']
        action_critic_params['modules_target_critic'] = action_critic_params['modules_critic']
        action_critic_state = TrainState.create(critic_net_def, action_critic_params, critic_tx)


        base_params = base_net_def.init(jax.random.fold_in(actor_rng, 101),
                                        actor=(ex_observations,))['params']
        base_state = TrainState.create(base_net_def, base_params, actor_tx)
        actor_state = TrainState.create(actor_net_def, actor_params, actor_tx)
        critic_state = TrainState.create(critic_net_def, critic_params, critic_tx)
        bc_flow_state = TrainState.create(bc_flow_net_def, bc_flow_params, bc_flow_tx)

        critic_state.params[f'modules_target_critic'] = critic_state.params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config["noise_dim"] = noise_dim

        if config['target_entropy'] is None:
            config['target_entropy'] = -float(noise_dim) / 2

        return cls(rng, actor_state, critic_state, bc_flow_state,
                   config=flax.core.FrozenDict(**config), action_critic_network=action_critic_state,
                   base_policy_network=base_state)
