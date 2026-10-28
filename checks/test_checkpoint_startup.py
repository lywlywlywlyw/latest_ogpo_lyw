"""Explicit startup modes, atomic online checkpoints, and timestamped runs."""
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import flax
import numpy as np

from ogpo.utils.config_loader import parse_cli_args
from ogpo.utils.log_utils import CsvLogger, setup_wandb
from ogpo.utils.main_helper import (
    get_checkpoint_dir, load_rolling_checkpoint, save_rolling_checkpoint,
    setup_experiment_logging,
)


@flax.struct.dataclass
class Agent:
    weights: object
    config: object = flax.struct.field(pytree_node=False)

    def sample_noise(self, observations, rng=None):
        return np.zeros(4, np.float32)

    def noise_to_actions_public(self, observations, noise):
        return np.zeros(4, np.float32)

    def sample_actions(self, observations, rng=None):
        return np.zeros(4, np.float32)

    def compute_flow_actions(self, observations, rng=None):
        return np.zeros(4, np.float32)

    def batch_update(self, *args):
        return self, {}

    @classmethod
    def create(cls, seed, observations, actions, config):
        return cls(np.array([seed], np.float32), config)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_directories_have_unique_timestamps(self):
        first = get_checkpoint_dir(1, 'square-mh-low_dim', 'dsrl')
        second = get_checkpoint_dir(1, 'square-mh-low_dim', 'dsrl')
        self.assertNotEqual(first, second)
        self.assertRegex(Path(first).name, r'^\d{8}_\d{6}_\d{6}UTC$')

    def test_latest_complete_checkpoint_and_corrupt_fallback(self):
        with TemporaryDirectory() as directory:
            template = Agent(np.array([0], np.float32), {})
            for step in (10, 20, 30):
                save_rolling_checkpoint(
                    template.replace(weights=np.array([step], np.float32)),
                    {'buffer_step': step}, step, step, 1, 'square-mh-low_dim',
                    algo_name='dsrl', checkpoint_dir=directory)
            self.assertFalse((Path(directory) / '10').exists())
            (Path(directory) / '40').mkdir()  # Interrupted/incomplete save.
            result = load_rolling_checkpoint(1, 'square-mh-low_dim', template,
                                             'dsrl', resume_path=directory)
            self.assertEqual(result[2:4], (30, 30))
            self.assertEqual(float(result[0].weights[0]), 30)
            (Path(directory) / '30' / 'agent.pkl').write_bytes(b'broken')
            result = load_rolling_checkpoint(1, 'square-mh-low_dim', template,
                                             'dsrl', resume_path=directory)
            self.assertEqual(result[2:4], (20, 20))
            self.assertEqual(result[1], {'buffer_step': 20})

    def test_failed_save_never_publishes_a_partial_checkpoint(self):
        with TemporaryDirectory() as directory:
            with patch('ogpo.utils.main_helper.pickle.dump', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    save_rolling_checkpoint(Agent(np.zeros(1), {}), {}, 10, 10, 1,
                                            'square-mh-low_dim', checkpoint_dir=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaises(FileNotFoundError):
                load_rolling_checkpoint(1, 'square-mh-low_dim', Agent(np.zeros(1), {}),
                                        resume_path=directory)

    def test_resume_mismatch_is_an_error(self):
        with TemporaryDirectory() as directory:
            agent = Agent(np.zeros(1), {})
            save_rolling_checkpoint(agent, {}, 10, 10, 1, 'square-mh-low_dim',
                                    algo_name='dsrl', checkpoint_dir=directory)
            with self.assertRaisesRegex(ValueError, 'seed'):
                load_rolling_checkpoint(2, 'square-mh-low_dim', agent, 'dsrl', resume_path=directory)

    def test_explicit_legacy_directory_is_readable(self):
        import pickle
        with TemporaryDirectory() as directory:
            agent = Agent(np.zeros(1), {})
            with open(Path(directory) / '5_agent.pkl', 'wb') as stream:
                pickle.dump(flax.serialization.to_state_dict(agent), stream)
            with open(Path(directory) / '5_buffer.pkl', 'wb') as stream:
                pickle.dump({'old': True}, stream)
            (Path(directory) / '5_meta.json').write_text(json.dumps({'online_loop_step': 5}))
            result = load_rolling_checkpoint(1, 'square-mh-low_dim', agent, 'dsrl', resume_path=directory)
            self.assertEqual(result[2:4], (5, 5))


class RunnerSavingTests(unittest.TestCase):
    def test_both_online_runners_save_under_the_allocated_timestamp_directory(self):
        from ogpo.runners.dsrl_runner import run_dsrl_online_rl
        from ogpo.runners.online_rl_runner import run_online_rl

        class Env:
            def reset(self, **kwargs):
                self.elapsed = 0
                return np.zeros(3, np.float32), {}

            def step(self, action):
                self.elapsed += 1
                return np.zeros(3, np.float32), -1.0, self.elapsed == 3, False, {'success': 0}

        for algorithm, runner in (('dsrl', run_dsrl_online_rl), ('ogpo', run_online_rl)):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                timestamped = str(Path(directory) / Path(get_checkpoint_dir(1, 'task', algorithm)).name)
                dataset = SimpleNamespace(sample=lambda _: dict(
                    observations=np.zeros(3, np.float32), actions=np.zeros(1, np.float32)))
                config = {'noise_chunk_length': -1, 'horizon_length': 4}
                agent = Agent(np.zeros(1, np.float32), config)
                logger = SimpleNamespace(log=lambda *args, **kwargs: None)
                module = 'ogpo.runners.' + ('dsrl_runner' if algorithm == 'dsrl' else 'online_rl_runner')
                with patch(module + '.save_agent'), patch(
                        module + '._run_evaluation', return_value={'success': 0.0}):
                    runner(agent=agent, env=Env(), eval_env=None, eval_envs=None,
                           train_dataset=dataset, config=config, logger=logger,
                           save_dir=directory, algo_name=algorithm, online_steps=7,
                           start_training=0, batch_size=1, horizon_length=4, discount=0.99,
                           buffer_size=32, n_eval_envs=1, force_jax_sync=False,
                           preemption_backup_interval=2, checkpoint_dir=timestamped,
                           log_enabled=False, step_data_log_interval=-1)
                self.assertTrue((Path(timestamped) / '7' / 'meta.json').is_file())
                _, _, _, online_step, _ = load_rolling_checkpoint(
                    0, '', agent, algorithm, resume_path=timestamped)
                self.assertEqual(online_step, 7)
                self.assertEqual(sorted(p.name for p in Path(timestamped).iterdir()), ['6', '7'])


class LoggingTests(unittest.TestCase):
    def test_every_wandb_creation_has_a_timestamp(self):
        with patch('ogpo.utils.log_utils.wandb.init') as initialize:
            setup_wandb(name='dsrl_squareseed1')
            name = initialize.call_args.kwargs['name']
            self.assertRegex(name, r'^dsrl_squareseed1_\d{8}_\d{6}_\d{6}UTC$')
            setup_wandb(name=name)
            self.assertEqual(initialize.call_args.kwargs['name'], name)
            setup_wandb(name=None, project='baseline')
            self.assertRegex(initialize.call_args.kwargs['name'], r'^baseline_\d{8}_')

    def test_log_disabled_still_gets_unique_timestamped_save_directory(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=False):
            kwargs = dict(seed=1, project='test', run_group='square', wandb_name='ogpo',
                          save_dir=directory, env_name='square-mh-low_dim', log_enabled=False)
            first, _, _ = setup_experiment_logging(**kwargs)
            second, _, _ = setup_experiment_logging(**kwargs)
            self.assertNotEqual(first, second)
            self.assertTrue(Path(first).is_dir())
            self.assertRegex(Path(first).name, r'^ogposeed1_\d{8}_\d{6}_\d{6}UTC$')

    def test_wandb_resume_id_only_used_when_explicitly_supplied(self):
        run = SimpleNamespace(id='new-id', project='test')
        with TemporaryDirectory() as directory, patch(
                'ogpo.utils.main_helper.setup_wandb', return_value=run) as initialize:
            kwargs = dict(seed=1, project='test', run_group='square', wandb_name='ogpo',
                          save_dir=directory, env_name='square-mh-low_dim')
            _, logger, _ = setup_experiment_logging(**kwargs)
            self.assertIsNone(initialize.call_args.kwargs['id'])
            self.assertIsNone(initialize.call_args.kwargs['resume'])
            logger.close()
            _, logger, _ = setup_experiment_logging(**kwargs, resume_wandb_id='old-id')
            self.assertEqual(initialize.call_args.kwargs['id'], 'old-id')
            self.assertEqual(initialize.call_args.kwargs['resume'], 'allow')
            logger.close()

    def test_resuming_csv_does_not_overwrite_previous_rows(self):
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'eval.csv')
            for step in (1, 2):
                logger = CsvLogger(path)
                logger.log({'success': step / 10}, step=step)
                logger.close()
            lines = Path(path).read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], 'success,step')


class StartupTests(unittest.TestCase):
    def run_main(self, directory, overrides):
        from ogpo import main as training
        base = {'experiment.log': False, 'eval.n_eval_envs': 1,
                'training.online_steps': 20, 'experiment.seed': 1}
        base.update(overrides)
        dataset = SimpleNamespace(sample=lambda _: dict(observations=np.zeros(3), actions=np.zeros(2)))
        logger = SimpleNamespace(log=lambda *args, **kwargs: None, close=lambda: None)
        agent = Agent(np.array([1], np.float32), {})
        with patch.object(training, 'parse_cli_args', return_value=('dsrl', base)), \
             patch.dict(training.AGENTS, {'dsrl': Agent}), \
             patch.object(training, 'make_env_and_datasets', return_value=(None, None, dataset, None)), \
             patch.object(training, 'process_train_dataset', side_effect=lambda data, *args, **kwargs: data), \
             patch.object(training, 'setup_experiment_logging', return_value=(directory, logger, None)) as logging, \
             patch.object(training, 'load_rolling_checkpoint', wraps=load_rolling_checkpoint) as resume, \
             patch.object(training, 'load_flow_checkpoint', return_value=agent) as flow, \
             patch.object(training, 'run_offline_pretraining', return_value=(agent, 5, {})) as offline, \
             patch.object(training, 'save_agent'), \
             patch.object(training, 'run_dsrl_online_rl', return_value=(agent, 20, {}, {})) as online:
            training.main()
            return resume, flow, offline, online, logging

    def test_flow_starts_fresh_even_when_old_online_checkpoint_exists(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {'OGPO_CHECKPOINT_ROOT': directory}):
            old = Path(directory) / 'dsrl' / '1_square-mh-low_dim'
            save_rolling_checkpoint(Agent(np.array([99]), {}), {}, 12, 12, 1,
                                    'square-mh-low_dim', algo_name='dsrl', checkpoint_dir=str(old))
            resume, flow, offline, online, logging = self.run_main(
                directory, {'checkpoint.flow_path': '/shared/flow.pkl'})
            resume.assert_not_called()
            flow.assert_called_once()
            offline.assert_not_called()
            self.assertFalse(online.call_args.kwargs['is_resumed'])
            self.assertEqual(online.call_args.kwargs['start_step'], 0)
            self.assertIsNone(logging.call_args.kwargs['resume_wandb_id'])

    def test_no_paths_runs_bc_and_never_searches_for_resume(self):
        with TemporaryDirectory() as directory:
            resume, flow, offline, online, _ = self.run_main(directory, {})
            resume.assert_not_called()
            flow.assert_not_called()
            offline.assert_called_once()
            self.assertFalse(online.call_args.kwargs['is_resumed'])

    def test_explicit_resume_skips_bc_and_uses_saved_training_state(self):
        with TemporaryDirectory() as directory:
            checkpoint_dir = get_checkpoint_dir(1, 'square-mh-low_dim', 'dsrl')
            checkpoint_dir = str(Path(directory) / Path(checkpoint_dir).name)
            save_rolling_checkpoint(Agent(np.array([9], np.float32), {}), {'old': True},
                                    12, 7, 1, 'square-mh-low_dim', algo_name='dsrl',
                                    checkpoint_dir=checkpoint_dir, wandb_run_id='old-run')
            resume, flow, offline, online, logging = self.run_main(
                directory, {'checkpoint.resume_path': checkpoint_dir})
            resume.assert_called_once()
            flow.assert_not_called()
            offline.assert_not_called()
            args = online.call_args.kwargs
            self.assertTrue(args['is_resumed'])
            self.assertEqual(args['resumed_online_step'], 7)
            self.assertEqual(args['start_step'], 12)
            self.assertEqual(args['checkpoint_dir'], checkpoint_dir)
            self.assertEqual(float(args['agent'].weights[0]), 9)
            self.assertEqual(logging.call_args.kwargs['resume_wandb_id'], 'old-run')

    def test_conflicting_paths_and_missing_resume_fail_before_environment_creation(self):
        from ogpo import main as training
        for overrides, error in [
                ({'checkpoint.flow_path': 'flow.pkl', 'checkpoint.resume_path': '/old'}, ValueError),
                ({'checkpoint.resume_path': '/this/directory/does/not/exist'}, FileNotFoundError)]:
            with self.subTest(overrides=overrides), patch.object(
                    training, 'parse_cli_args', return_value=('dsrl', overrides)), patch.object(
                    training, 'make_env_and_datasets') as environment:
                with self.assertRaises(error):
                    training.main()
                environment.assert_not_called()

    def test_friendly_python_cli_aliases(self):
        _, overrides = parse_cli_args(['--algo=dsrl', '--flow_checkpoint_path=/flow.pkl',
                                      '--resume_path=null'])
        self.assertEqual(overrides['checkpoint.flow_path'], '/flow.pkl')
        self.assertIsNone(overrides['checkpoint.resume_path'])


if __name__ == '__main__':
    unittest.main()
