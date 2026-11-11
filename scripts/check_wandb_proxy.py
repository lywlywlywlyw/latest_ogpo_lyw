"""Validate nodes, or prove metrics arrive online before wandb.finish()."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from ogpo.utils.wandb_proxy import PinnedProxy, load_rules, resolve_environment


def scan(rules):
    source = yaml.safe_load(Path(rules['source_config']).read_text())
    nodes = [n for n in source['proxies'] if n['name'].startswith(('🇯🇵', '🇺🇸', '🇩🇪', '🇸🇬', '🇭🇰', '🇹🇼', '🇫🇮'))]
    def probe(node):
        result = {'node': node['name'], 'checks': []}
        proxy = PinnedProxy(rules, node)
        try:
            for _ in range(3):
                start = time.monotonic()
                try:
                    proxy.check(attempts=1, timeout=10)
                    check = {'ok': True}
                except Exception as exc:
                    check = {'ok': False, 'error': type(exc).__name__}
                check['seconds'] = round(time.monotonic() - start, 3)
                result['checks'].append(check)
        finally:
            proxy.close()
        result['passed'] = all(c['ok'] for c in result['checks'])
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(probe, nodes))


def smoke(rules, node):
    if node and node not in rules['candidates']:
        raise ValueError('--node must be a configured candidate')
    with tempfile.TemporaryDirectory(prefix='wandb-smoke-rules-') as tmp:
        if node:
            override = dict(rules, candidates=[node])
            path = Path(tmp) / 'rules.yaml'
            path.write_text(yaml.safe_dump(override, allow_unicode=True))
            os.environ['OGPO_WANDB_PROXY_RULES'] = str(path)
        from ogpo.utils.wandb_proxy import ensure_wandb_proxy
        from ogpo.utils.log_utils import setup_wandb
        import wandb
        proxy = ensure_wandb_proxy()
        run = setup_wandb(entity=os.environ.get('WANDB_ENTITY'),
                          project=os.environ.get('WANDB_PROJECT', 'OGPO'), group='wandb-network-diagnostic',
                          name='wandb-pinned-proxy-validation', mode='online',
                          config={'purpose': 'online proxy validation', 'node': proxy.node})
        result = {'node': proxy.node, 'run_id': run.id, 'url': run.url}
        try:
            # Wait for server history WHILE the SDK run remains active.
            run.log({'network_probe/sequence': 1, 'network_probe/ok': 1})
            deadline = time.monotonic() + 60
            visible = False
            while time.monotonic() < deadline:
                api = wandb.Api(timeout=10)
                remote = api.run(f'{run.entity}/{run.project}/{run.id}')
                rows = list(remote.scan_history(keys=['network_probe/sequence']))
                if any(row.get('network_probe/sequence') == 1 for row in rows):
                    visible = True
                    break
                time.sleep(2)
            result['history_visible_before_finish'] = visible
            if not visible:
                raise RuntimeError('Metric was not visible online before finish')
            artifact = Path(run.dir) / 'network-probe.txt'
            artifact.write_text('W&B dedicated proxy online file-upload validation\n')
            run.save(str(artifact), base_path=run.dir, policy='now')
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                remote = wandb.Api(timeout=10).run(f'{run.entity}/{run.project}/{run.id}')
                if 'network-probe.txt' in [f.name for f in remote.files()]:
                    result['file_visible_before_finish'] = True
                    break
                time.sleep(2)
            else:
                raise RuntimeError('File was not visible online before finish')
            result['passed'] = True
        except Exception as exc:
            result.update(passed=False, error=type(exc).__name__)
        finally:
            run.finish()
            result['debug_internal_log'] = str(Path(run.dir).parent / 'logs/debug-internal.log')
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-nodes', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Create a real online diagnostic W&B run')
    parser.add_argument('--node')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.all_nodes == args.smoke:
        parser.error('Choose exactly one of --all-nodes / --smoke')
    _, rules = load_rules()
    rules = resolve_environment(rules)
    report = {'tested_at': datetime.now(timezone.utc).isoformat()}
    report['results'] = scan(rules) if args.all_nodes else [smoke(rules, args.node)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if args.smoke and not all(r.get('passed') for r in report['results']):
        sys.exit(1)
