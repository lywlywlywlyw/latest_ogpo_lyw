# W&B 实时上传专用代理规则

## 公开仓库中的内容

本目录的 `wandb_proxy_rules.yaml` 可提交到公开仓库，只包含公共路由域名、已验证节点名称和预检参数。它没有代理服务器 IP、域名、端口、密码、订阅链接、用户目录或本机可执行文件路径。

代码部署到另一台机器后，会从那台机器的 clashctl 环境读取实际代理配置，无需复制此机器的私有文件，也不依赖 Git 忽略的配置文件。候选节点名称用于匹配目标环境已有配置中的节点；远端地址和凭据只在运行时从该环境读取。

## 运行时发现环境

公共日志初始化函数 `setup_wandb()` 在 `wandb.init()` 前读取本目录规则，然后自动发现机器相关配置：

1. 优先使用环境变量 `OGPO_CLASH_CONFIG` 或 `CLASH_CONFIG_RUNTIME` 指定的配置。
2. 未显式指定时，读取运行中 mihomo/clash 进程的配置参数与可执行文件路径；排除本工具创建的独立代理进程。
3. 无法从进程唯一确定时，尝试 `CLASHCTL_HOME`、当前用户的 `~/clashctl` 以及 PATH 中的 clashctl 安装目录下的 `resources/runtime.yaml`。
4. 从配置中的 `mixed-port` 或 `port`、监听地址和认证信息读取现有上游代理。无需在仓库写死端口。
5. mihomo 可执行文件从活动进程、配置所在安装目录或 PATH 发现；可通过 `OGPO_MIHOMO_BINARY` 覆盖。

存在多套配置且无法唯一发现时会报错，此时用环境变量明确指定目标环境的配置路径即可，不要将具体地址或私有路径提交到仓库。

## 固定 W&B 出口

依次尝试规则中的候选节点，对 W&B GraphQL 连续预检 3 次。有现有 W&B 凭据时查询 `viewer.username`，同时验证认证。失败时关闭该候选并尝试下一候选；全部失败则报错中止启动。

通过预检后启动本机回环地址上的独立代理，固定 W&B 出口，并显式设置 SDK 的 HTTP/HTTPS 代理。不会跟随全局“自动选择”切换，不改写全局 clashctl 配置。

```text
DOMAIN-SUFFIX,wandb.ai,WANDB_PINNED
DOMAIN-SUFFIX,wandb.com,WANDB_PINNED
DOMAIN-SUFFIX,storage.googleapis.com,WANDB_PINNED
MATCH,EXISTING_PROXY
```

其他域名交给从目标环境发现的原有代理。节点运行配置只在权限受限的临时目录生成，正常退出时清理。保持实时在线上传，无自动离线降级或训练后补传流程。

新启动的训练生效；不热修改已有训练进程。同一 run 内固定节点，若运行中该节点之后失效，需要修复节点或调整候选规则并重启相应训练。

## 已验证节点

2026-11-11 测试了 40 个实际节点，每节点 3 次 W&B GraphQL 请求：17 个全通过，23 个出现 403、SSL 错误或超时。

公开规则中的 3 个候选分别为日本 BGP-02、新加坡 BGP-01、美国 BGP-07【优化】。这 3 个节点各 15 次认证 GraphQL 请求全部通过，并均完成真实指标和文件上传；在 `wandb.finish()` 前已从服务端验证可见。少量请求出现超时或重试后恢复，所以每次启动仍执行预检。

逐次结果位于 Git 忽略的 `analysis/wandb_proxy_validation/`，这些历史报告不参与运行配置加载，也不需要部署。报告包含本机诊断信息，勿强制提交。

## 使用和复验

原训练命令不变。可在仓库根目录先执行以下命令，验证自动发现和预检（不创建 W&B run）：

```bash
.venv/bin/python -m ogpo.utils.wandb_proxy
```

扫描目标环境的全部节点：

```bash
.venv/bin/python scripts/check_wandb_proxy.py --all-nodes \
  --output analysis/wandb_proxy_validation/node_recheck.json
```

真实在线上传测试（会创建诊断 run）：

```bash
.venv/bin/python scripts/check_wandb_proxy.py --smoke \
  --output analysis/wandb_proxy_validation/smoke_recheck.json
```

诊断账户使用目标环境的 W&B 登录信息，可通过 `WANDB_ENTITY` 和 `WANDB_PROJECT` 指定账户和项目。可通过 `OGPO_WANDB_PROXY_RULES` 指定另一份通用规则文件。
