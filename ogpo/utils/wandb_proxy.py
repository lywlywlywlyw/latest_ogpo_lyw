"""Load repository-owned W&B routing rules and pin an API-tested proxy node."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import tempfile
import threading
import time

import requests
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_active = None


def load_rules(path=None):
    path = Path(path or os.environ.get('OGPO_WANDB_PROXY_RULES', _ROOT / 'wandb_proxy_rules.yaml'))
    rules = yaml.safe_load(path.read_text())
    if not rules.get('candidates'):
        raise ValueError(f'No W&B proxy candidates in {path}')
    return path, rules



def _running_clash():
    """Discover the active system proxy, excluding our isolated child processes."""
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().decode().split('\0')
            if Path(args[0]).name not in ('mihomo', 'clash', 'clash-meta'):
                continue
            if '-f' not in args:
                continue
            source = Path(args[args.index('-f') + 1])
            if 'ogpo-wandb-proxy-' in str(source) or 'wandb-node-probe-' in str(source):
                continue
            if not source.is_absolute():
                source = (entry / 'cwd').resolve() / source
            binary = str((entry / 'exe').resolve())
            if source.is_file():
                found.append((str(source), binary))
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    return sorted(set(found))


def resolve_environment(rules):
    """Resolve machine details from environment/active clashctl, never repo constants."""
    rules = dict(rules)
    running = _running_clash()
    source = os.environ.get('OGPO_CLASH_CONFIG') or os.environ.get('CLASH_CONFIG_RUNTIME')
    source = source or rules.get('source_config')
    binary = os.environ.get('OGPO_MIHOMO_BINARY') or rules.get('mihomo_binary')
    if not source and len(running) == 1:
        source, active_binary = running[0]
        binary = binary or active_binary
    if not source:
        homes = [Path(os.environ['CLASHCTL_HOME'])] if os.environ.get('CLASHCTL_HOME') else []
        homes.append(Path.home() / 'clashctl')
        command = shutil.which('clashctl')
        if command:
            homes.append(Path(command).resolve().parent)
        candidates = [home / 'resources/runtime.yaml' for home in homes]
        source = next((str(path) for path in candidates if path.is_file()), None)
    if not source:
        raise RuntimeError('Cannot uniquely discover clashctl config; set OGPO_CLASH_CONFIG')
    source_path = Path(source).expanduser().resolve()
    config = yaml.safe_load(source_path.read_text())
    if not binary:
        binary = next((b for p, b in running if Path(p).resolve() == source_path), None)
    if not binary:
        sibling = source_path.parent.parent / 'bin/mihomo'
        binary = str(sibling) if sibling.is_file() else shutil.which('mihomo')
    if not binary:
        raise RuntimeError('Cannot discover mihomo executable; set OGPO_MIHOMO_BINARY')
    port = config.get('mixed-port') or config.get('port')
    if not port:
        raise RuntimeError('clashctl config requires an HTTP or mixed proxy listener')
    host = config.get('bind-address', '127.0.0.1')
    if host in ('*', '0.0.0.0', '::'):
        host = '127.0.0.1'
    upstream = {'host': host, 'port': int(port)}
    if config.get('authentication'):
        username, password = config['authentication'][0].split(':', 1)
        upstream.update(username=username, password=password)
    rules.update(source_config=str(source_path), mihomo_binary=str(binary), upstream_proxy=upstream)
    return rules


def _stop(process, directory):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    directory.cleanup()


class PinnedProxy:
    def __init__(self, rules, node):
        self.node = node['name']
        self.directory = tempfile.TemporaryDirectory(prefix='ogpo-wandb-proxy-')
        self.process = None
        try:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                self.port = sock.getsockname()[1]
            self.url = f'http://127.0.0.1:{self.port}'
            upstream = rules['upstream_proxy']
            config = {
                'mixed-port': self.port, 'bind-address': '127.0.0.1',
                'allow-lan': False, 'mode': 'rule', 'log-level': 'silent',
                'proxies': [dict(node, name='WANDB_PINNED'), {
                    'name': 'EXISTING_PROXY', 'type': 'http',
                    'server': upstream['host'], 'port': upstream['port'],
                }],
                'rules': [f'DOMAIN-SUFFIX,{domain},WANDB_PINNED'
                          for domain in rules['domains']] + ['MATCH,EXISTING_PROXY'],
            }
            if upstream.get('username') is not None:
                config['proxies'][1].update(username=upstream['username'], password=upstream['password'])
            path = Path(self.directory.name) / 'config.yaml'
            # Node credentials come from the existing private config, never the repo.
            path.write_text(yaml.safe_dump(config, allow_unicode=True))
            path.chmod(0o600)
            self.process = subprocess.Popen(
                [rules['mihomo_binary'], '-d', self.directory.name, '-f', str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Dedicated mihomo process exited during startup')
                try:
                    with socket.create_connection(('127.0.0.1', self.port), timeout=.1):
                        break
                except OSError:
                    time.sleep(.05)
            else:
                raise RuntimeError('Dedicated W&B proxy did not start')
        except BaseException:
            self.close()
            raise

    def session(self):
        session = requests.Session()
        session.trust_env = False
        session.proxies = {'http': self.url, 'https': self.url}
        return session

    def check(self, attempts=3, timeout=10, auth=None):
        """Probe actual W&B GraphQL, never a generic latency/health-check URL."""
        with self.session() as session:
            if auth is not None:
                session.auth = auth
            query = '{viewer {username}}' if auth else '{__typename}'
            for _ in range(attempts):
                response = session.post('https://api.wandb.ai/graphql',
                                        json={'query': query}, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                data = payload.get('data') or {}
                valid = bool(data.get('viewer')) if auth else data.get('__typename') == 'Query'
                if payload.get('errors') or not valid:
                    raise RuntimeError('W&B GraphQL preflight failed')

    def close(self):
        if self.process is not None:
            _stop(self.process, self.directory)
        else:
            self.directory.cleanup()


def _auth():
    key = os.environ.get('WANDB_API_KEY')
    if key:
        return ('api', key)
    # Same credential source as requests/W&B, without printing secrets.
    return requests.utils.get_netrc_auth('https://api.wandb.ai')


def ensure_wandb_proxy():
    """Run before wandb.init; fail startup if no configured node passes preflight."""
    global _active
    with _lock:
        if _active is not None:
            if _active.process.poll() is not None:
                raise RuntimeError('Dedicated W&B proxy exited')
            return _active
        path, rules = load_rules()
        rules = resolve_environment(rules)
        source = yaml.safe_load(Path(rules['source_config']).read_text())
        nodes = {node['name']: node for node in source['proxies']}
        failures = []
        for name in rules['candidates']:
            if name not in nodes:
                failures.append(f'{name}: missing from source config')
                continue
            proxy = None
            try:
                proxy = PinnedProxy(rules, nodes[name])
                proxy.check(attempts=rules.get('preflight_attempts', 3),
                            timeout=rules.get('request_timeout_seconds', 10), auth=_auth())
            except Exception as exc:
                if proxy is not None:
                    proxy.close()
                # Do not include exception text: it may contain private proxy URLs.
                failures.append(f'{name}: {type(exc).__name__}')
                print(f'[W&B proxy] rejected {name}: {type(exc).__name__}', flush=True)
                continue
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                        'http_proxy', 'https_proxy', 'all_proxy'):
                os.environ[key] = proxy.url
            # Loopback controller calls remain local; W&B cannot bypass its route.
            os.environ['NO_PROXY'] = os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
            _active = proxy
            atexit.register(proxy.close)
            print(f'[W&B proxy] rules={path}; pinned={name}; proxy={proxy.url}; '
                  'W&B GraphQL preflight passed', flush=True)
            return proxy
        raise RuntimeError('No configured W&B proxy passed preflight: ' + '; '.join(failures))


if __name__ == '__main__':
    proxy = ensure_wandb_proxy()
    print(json.dumps({'node': proxy.node, 'proxy': proxy.url, 'status': 'passed'},
                     ensure_ascii=False))
