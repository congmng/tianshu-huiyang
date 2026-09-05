# Llumnix 在 Iluvatar CoreX 4.4.0 上的适配与验证报告

## 2026-09-05：V1 P/D connector 配置快速失败校验

入口参数现在在创建 V1 EngineCore 前校验 P/D 传输类型：启用
`kvtransfer` 时仅接受 `P2pNcclConnector` 或 CoreX ABI 兼容的
`CoreXP2pNcclConnector`。误用历史 `rdma`、`SharedStorageConnector` 或空值会立即
抛出可操作的 `ValueError`，避免模型加载后才发现跨角色 KV 无法交接；未启用 P/D
的普通 V1 部署不受此限制。新增单测覆盖拒绝与两个合法 connector 拼写。

## 2026-09-05：P/D 用户文档与当前 V1 支持边界对齐

更新 `docs/Prefill-decoding_Disaggregation.md`：将 Python 3.12/CoreX 4.4/vLLM
V1 的 connector-driven P/D handoff、`corex44_v1_pd.yml` 配置模板和
`integration --model-pd` 验收命令置于历史 vLLM 0.6 说明之前。文档同时明确
V1 不实现旧私有 block-manager 的任意时刻/Decode-to-Decode request migration，
native NCCL 也不是默认生产路径，避免旧设计文字超出当前实测支持范围。

## 2026-09-05：CoreX V1 P/D 部署模板

新增 `configs/corex44_v1_pd.yml` 作为两机部署起点。模板使用 1:1 P/D、
`virtual_usage` 异构调度、`kvtransfer` 和 `CoreXP2pNcclConnector`，并关闭旧
vLLM 0.6 block-manager migration。配置单测已验证 yacs 解析与关键字段，纳入
CoreX V1 unit gate（`92 passed`）。模板不固定主机 endpoint；Manager/Llumlet
根据各节点 runtime env 的 `LLUMNIX_KV_*` 注入可路由地址，从而支持两机和多网卡
部署。
该模板也纳入双机 source fingerprint；配置、验证脚本和 serving 实现任一不一致时，
support gate 会在分配 GPU 前拒绝部署。

## 2026-09-05：最终 Python 3.12 wheel 交付审计

在最新源码以无隔离、无依赖下载方式执行 `python -m pip wheel . --no-deps
--no-build-isolation`，成功生成 `llumnix-0.0.2-py3-none-any.whl`（约 176 KiB）。
support/config 门禁为 `9 passed`；两机 CoreX support gate 均为 `supported=true`，
共同 source fingerprint 为 `c0ef7487…e83d2b`，且远端 `tianshu-huiyang/main` 与
本地 `a4fcd9b` 一致。项目环境、模型、Ray 目录及实例日志保持未跟踪，未污染交付物。

## 2026-09-05：当前提交的两机模型级 P/D handoff 复验

自包含探针修复后，使用当前提交在两机各一张 BI-V150 上重跑模型探针：Prefill
producer 保存 40 层 BF16 KV（每层约 `(2,1084,8,16,128)`），Decode consumer 收到
共享 request metadata 并执行 `load role=consumer`，随后返回非空中文 `重复的句子`
并正常退出。两端均为 Qwen3-14B、Python 3.12.13、vLLM 0.11.2、CoreX 4.4.0；
这证明 connector-driven P/D handoff 已从底层 staging 提升到真实模型级闭环。实现
仍使用默认安全的 ZMQ CPU-staging，未将 CoreX native NCCL rank-1 不稳定路径宣称为
生产能力。

该模型级阶段已纳入 `run_corex44_validation.py integration --model-pd`；因其在两机
各加载一份 14B 权重，保持为显式开关，默认 integration 仍执行更快的 KV-event affinity
和 BF16 staging。这样既保留日常回归速度，也为交付/发布提供一条命令的真实 P/D 验收。

提交 `c7dcbcc` 同步两机后，实际执行该一键命令成功覆盖完整链路：严格双机
fingerprint gate、16 个真实 BlockStored event 的 affinity=1.0、BF16 staging、40 层
Qwen3 producer KV save、consumer load 和非空中文输出均 PASS。故 P/D 模型验收不再
依赖人工拼接两条命令，且仍由 gate 阻止未同步的两机版本进入 GPU 分配阶段。

## 2026-09-05：模型级 P/D 探针可复现性修复

两机 Qwen3-14B P/D 探针首次以文档形式的 `python tools/...` 调用远端时，发现
脚本仅在 `PYTHONPATH` 已设置时可导入 Llumnix。该部署性缺口已修复：探针现在
根据自身路径自动加入 checkout 根目录，并纳入双机 source fingerprint。此前的
`ModuleNotFoundError` 是启动包装问题，不代表 CoreX KV connector 失败；同步后
重新进行 producer/consumer handoff，并分别记录模型启动、KV send/recv 与最终文本。

## 2026-09-05：Llumnix V1 HTTP 模型级端到端验证

新增并执行 `tools/run_llumnix_v1_http_e2e.py`：真实启动 Llumnix V1 API server
及 Qwen3-14B CoreX engine，在同一进程生命周期内验证 `/health`、`/is_ready`、
`/instance_list`（GPU 数、节点、显存字段）和 `/generate`。本机 `cuda:0` 实测
全部通过，返回非空中文，输出 `llumnix_v1_http_corex: PASS`。该工具已纳入统一
`e2e` runner，因此交付测试同时覆盖直接 vLLM engine 与 Llumnix HTTP serving
边界。单卡模型权重约 27.5 GiB，运行参数需为 `gpu_memory_utilization=0.96` 以
保留 KV cache 空间；设置为 0.70 时 vLLM 明确报无可用 cache memory，属于资源配置
错误而非适配故障。

在最终统一入口 `python tools/run_corex44_validation.py e2e --tp 2` 中，TP=2
Qwen3-14B 首先以 NCCL rank 0/1 完成中文生成（25.42 秒）；其子进程退出后，同一
runner 启动单卡 Llumnix V1 HTTP E2E 并完成 API 探测和生成。两阶段均 PASS，说明
TP worker 的退出不会阻碍后续 API 的 GPU/KV cache 初始化，验证了可重复的完整
模型交付链路。

## 2026-09-05：双机验证协议一致性

双机 support gate 的源码指纹新增覆盖统一 validation runner 与 Llumnix V1 HTTP
E2E 工具。这样两台部署机必须同时拥有相同服务实现和相同验证契约；变更后的首次
门禁刻意检出尚未同步远端的 fingerprint 差异，随后 archive 同步并复验。该机制防止
只在本机更新 E2E 测试导致双机 KV-affinity 结果不可比较。

同步后双方 fingerprint 为 `664a5d…f665` 且 `supported=true`。随后完整重跑双机
integration：16 个真实 vLLM `BlockStored` 事件使远端 affinity=`1.0`、
`remote-cached` 优先；两机 `cuda:0` BF16 ZMQ staging 也均 PASS。因此新增的交付
协议门禁不会削弱既有 KV-affinity 和传输验证。

## Python 3.12 发布构建与多卡交付复验（2026-09-04）

最新提交在专用 CoreX Python 3.12 环境通过无隔离 wheel 构建，生成
`llumnix-0.0.2-py3-none-any.whl`。真实 Qwen3-14B TP=2 E2E 随后通过：vLLM V1
建立 NCCL world size=2，TP rank 0/1 完成模型及 KV cache 初始化并返回非空中文结果，
耗时 **27.45 s**。该证据覆盖可安装交付物和多卡基础推理链路。

## 单角色池故障窗口的取消验证（2026-09-04）

新增有 Prefill、无 Decode 的真实等待场景回归，验证 V1 Manager 不会误降级请求，且
abort 能立即唤醒角色恢复等待协程。该场景覆盖 Ray 仅替换一类 actor 时的 P/D 高可用
边界，并纳入统一 CoreX unit gate。

## 零实例 P/D 启动窗口的取消安全性（2026-09-04）

在全局 P/D 初始 actor 尚未注册或两类 actor 同时重启期间，Manager 可能暂时没有
任何实例。V1 请求现在先登记到等待集合，再进入零实例重试；客户端取消会移除等待
标记，生成协程在每次重试时退出，避免 HTTP 断连留下永久等待任务。

## P/D 故障窗口等待请求的取消语义（2026-09-04）

角色恢复等待状态没有 EngineCore request，不能用通常的 producer/consumer abort
fan-out 处理。Manager 现单独跟踪这些公开 request ID；客户端取消会移除等待标记，
协程在每次重试和最终提交 P/D 对前确认未被取消。这样断连不会留下等待角色恢复的
孤立请求；无 actor abort 回归已加入正式 unit gate。

## 异构扩缩容源码一致性门禁（2026-09-04）

完整 V1 serving fingerprint 现纳入 `ScalingScheduler` 和 `scaling_policy.py`，覆盖
论文异构显存/算力 `virtual_usage` 计算、扩缩容阈值策略及故障窗口保护。双机在启动
integration 前会拒绝这些算法实现的源码漂移。
基础 KV-event publisher 单测同时采用完整 bind 重试，消除共享 Ray/ZMQ 环境的临时
端口 TOCTOU 假失败；当前统一 unit gate 为 **87 passed**。

## P/D 故障恢复的角色类型一致性（2026-09-04）

P/D 恢复检查不再从普通 dispatch eligibility 推断角色；后者包含
`NO_CONSTRAINTS` 实例，可能在异构混合部署中掩盖 Prefill/Decode 整类故障。现在使用
`ScalingScheduler.instance_type_id_set` 的权威类型集合，并以回归验证混合部署的错误
推断不会发生。
统一 CoreX unit gate 当前为 **87 passed**；同步最新提交后两台服务器的完整运行时与
源码一致性门禁均通过，fingerprint=`2fdf9f25...e49e5f2c4`。

## P/D 角色故障窗口的安全恢复（2026-09-04）

当 Ray 替换故障 actor、导致 Prefill 或 Decode 角色池暂时为空时，V1 Manager 不再
将请求落入普通 dispatch（这可能误选 Decode-only 或绕过 KV handoff）。请求会保留在
Manager 生成协程中，按重试间隔等待两类角色重新可用；角色恢复后继续使用 affinity-aware
P/D 选择和 connector handoff。该行为由缺失角色回归覆盖。
缺失角色场景已加入统一 unit gate，当前 support/transfer/affinity/dispatch/Manager/API
组合结果为 **86 passed**。
加入等待请求取消回归后统一 gate 为 **88 passed**；同一源码在双机 integration 中
通过完整一致性门禁（fingerprint=`13ebaa0f...a2dfd07a`）、KV event affinity=1.0
和 BF16 staging transfer。

## 自动扩缩容故障窗口保护（2026-09-04）

CoreX 精简 Ray 下 actor 故障、scale-down 与 info polling 可能短暂交错。V1
`ScalingScheduler` 现在对空快照及 membership/info 不一致显式返回“不扩缩容”，避免
后台协程因 KeyError 退出或使用陈旧显存/虚拟负载作出错误决策；相关回归已纳入统一
CoreX unit 门禁。
本轮 unit gate 为 **85 passed**；同步双机后真实 integration 复验通过完整源码/SDK/
Iluvatar 设备门禁、16 个 V1 `BlockStored` event 的 `affinity=1.0` 远端选择，以及
GPU BF16 `(4,4)` staging transfer。

## CoreX 厂商运行时门禁（2026-09-04）

正式支持检查现在不只依赖 Python 包版本。它在不分配模型/GPU 工作负载的前提下读取
`/usr/local/corex/release-corex.txt`，确认 SDK 为 4.4.0；同时核验 PyTorch
accelerator 可用且设备名为 Iluvatar，并在双机比较 SDK、设备可见性和设备类型。
这会拒绝“版本看似匹配但实际运行在普通 CUDA/无设备环境”的误报。
在当前两台服务器实测中，双方均报告 SDK `Iluvatar CoreX SDK 4.4.0`、设备
`Iluvatar BI-V150`、16 张可见卡和 `cuda_available=true`；版本、完整 V1 serving
fingerprint 与 affinity hashes 均一致，最终 `supported=true`。运行时门禁加入后
统一 unit gate 为 **84 passed**，项目源码 compileall 通过。

## 双机完整 V1 serving 源码一致性（2026-09-04）

正式支持门禁的 SHA256 fingerprint 现覆盖完整 serving 边界，而不只覆盖 affinity
和 connector：包括 V1 backend 选择、`Launcher` placement、`Llumlet` actor 生命周期、
Manager/GlobalScheduler、V1 CLI 参数解析、client 请求桥接及独立 V1 API。新增回归
锁定覆盖清单；任何一端在这些关键实现上漂移，双机 integration 会在启动模型前失败。
实测远端旧快照被正确拒绝；同步后两端 fingerprint 为
`43a22652f49837cabd7418baa3dc789d64844a733909d5e283f9e622f392a2f2`，并通过
Python 3.12/CoreX/vLLM 版本与 affinity hash 比较。当前扩展 unit gate 为
**83 passed**。

## KV affinity 实例生命周期清理（2026-09-04）

针对长时间 P/D 服务，`DispatchScheduler.remove_instance` 现无条件移除实例的
`InstanceInfo` snapshot，包含 Decode-only 角色池使用的 KV block ownership。这样
实例下线、缩容或重启后，陈旧 cache affinity 不会使 Manager 指向不存在的实例。
新增回归验证 Decode-only cache snapshot 在 remove 后不可见；扩展 unit gate（support、
transfer、affinity、dispatch、Manager、V1 API）结果为 **82 passed**。

## Manager P/D 亲和选择完整门禁（2026-09-04）

V1 P/D 角色选择已抽取为 Manager 的 `_select_v1_pd_instances`，由 Prefill 与
Decode 两个受限候选池分别调用生产 `DispatchScheduler.dispatch_candidates`。
这避免仅凭 scheduler 孤立单测宣称请求编排已接入 affinity：新增回归同时构造两个
Prefill 和两个 Decode 实例，在每个池负载差不超过 `0.10` 时验证拥有请求 prefix
的实例被选中，并验证 Decode-only 实例仍不进入普通 dispatch。该门禁已纳入
`tools/run_corex44_validation.py unit`，当前统一结果为 **69 passed**。
随后以同一提交实测双机 integration 与本机 TP=2 Qwen3-14B E2E：两端源码
fingerprint 为 `46a2e3673016d06d059aa8df963817c4aa7c3e04f4b37fce43730a5e91c4ea01`，
16 个真实 vLLM KV events 使远端缓存候选获得 `affinity=1.0`，BF16 `(4,4)` staging
KV transfer 成功；TP=2 完成 NCCL ranks、模型/KV 初始化并生成非空中文输出（26.11 s）。

## 最新分层 Unit 回归基线（2026-09-03）

在包含源码指纹门禁、地址参数化和远端 consumer 提前退出检测的当前提交上，执行
`tools/run_corex44_validation.py unit`，结果为 **68 passed**。该层覆盖 V1 HTTP/
benchmark 生命周期、KV transfer 与 affinity、异构调度、CoreX 支持检查及分层
runner 自身的契约；集成和 E2E 证据仍按本文对应章节的实机结果执行。

## Integration runner 启动失败快速检测（2026-09-03）

双机 integration runner 现在在启动 producer 前检查远端 consumer 是否已提前退出，
并在异常或超时路径通过 `finally` 回收 SSH 子进程；避免远端环境错误被掩盖成一次
无意义的本地发送超时。该改动已通过 runner 单测与实际双机 integration（版本/源码
指纹门禁、BF16 GPU staging）复验。

## 双机源码指纹严格门禁（2026-09-03）

CoreX 支持检查现在对 V1 adapter、KV event/transfer、CoreX connector 和
GlobalScheduler 五个关键源码文件计算 SHA256 指纹，并在双机比较时强制一致。
首次运行正确发现远端旧脚本（`source_fingerprint=None`）并拒绝支持结论；同步
门禁后两端指纹均为
`90c9c87ef8be1dcfc4a2b861d14747217324d68bd9bfcf56e3b2be39f574603d`，版本、
affinity hashes、候选排序均一致，最终 `supported=true`。门禁单测 **6 passed**。

## 分层集成测试参数化（2026-09-03）

`run_corex44_validation.py integration` 新增 `--local-ip`/`--remote-ip`，不再将
两机地址耦合在数据面命令中；默认值仍对应 `10.31.10.62` 与 `10.31.10.210`。
当前以显式参数实际运行：双机版本/affinity 门禁通过，动态端口 BF16 GPU staging
producer/consumer 均 PASS。该改动使相同集成测试可迁移到其他 CoreX 节点组合。

## 分层验证入口的真实 E2E 执行（2026-09-03）

通过 `tools/run_corex44_validation.py e2e --tp 2` 实际启动两张 BI-V150 上的
Qwen3-14B：vLLM V1 报告 `world_size=2`、TP rank 0/1，8 个权重分片加载完成，
KV cache 初始化成功，并返回非空中文结果；最终脚本输出
`qwen3_14b_corex_vllm: PASS`（26.40 秒）。至此分层入口的 unit、integration、
e2e 三层均有当前代码的实际执行证据。

## 分层 CoreX 验证入口（2026-09-03）

新增 `tools/run_corex44_validation.py`，将适配验证固化为可重复的三层门禁：

- `unit`：V1 adapter、KV transfer/affinity、调度与 HTTP 契约；当前 **67 passed**。
- `integration`：两机严格版本/affinity hash 比较及真实 GPU BF16 ZMQ KV staging；
  两端均为 Python 3.12.13、vLLM 0.11.2、PyTorch 2.7.1、Ray 2.52.1，且 producer
  `10.31.10.62:49223` 至 consumer `10.31.10.210:50201` 成功（两端 `cuda:0`、
  BF16 `(4,4)`、远端均值 `7.5`）。
- `e2e`：复用 Qwen3-14B TP=1/TP=2 smoke；TP=2 已有两张 BI-V150 的真实
  NCCL、模型加载、KV cache 和中文生成 PASS 证据。

该入口不停止共享 Ray、也不删除模型或运行时目录，适用于部署节点的持续回归。

## Benchmark 真实 HTTP 端到端验证（2026-09-03）

本机以 Qwen3-14B、Python 3.12、CoreX 4.4、vLLM 0.11.2 启动独立 V1 服务后，
调用 `/generate_benchmark`（request_id=`bench-e2e-001`、`max_tokens=8`）返回
HTTP 200；结果包含 `num_input_tokens=11`、`num_output_tokens_cf=8` 以及 8 个
逐输出延迟样本（最终约 538.55 ms）。这验证 benchmark JSON 契约、真实 token
统计和 EngineCore 流式生命周期已贯通。

## Benchmark 客户端断连清理（2026-09-03）

`/generate_benchmark` 现在在每个 V1 输出事件前检查客户端连接；断连时立即调用
adapter abort 并返回 HTTP 499，正常结束仍执行 request release。新增断连集成回归，
入口测试提升为 **18 passed**，覆盖 benchmark 正常响应与异常生命周期。

## V1 benchmark 输入 token 统计修复（2026-09-03）

`/generate_benchmark` 现在从 V1 `RequestOutput.prompt_token_ids` 计算真实
`num_input_tokens`，不再固定报告 0；输出 token 数继续来自 completion token IDs。
这使迁移后的 benchmark 可用于 TTFT/输入长度分桶分析，接口回归保持 **17 passed**。

## 独立 V1 benchmark API 兼容（2026-09-03）

独立 `v1_api_server` 已恢复 `/generate_benchmark`：以 AsyncLLM 输出流记录每个
输出事件相对起始的毫秒延迟，并返回公开 request ID、生成文本、输出 token 数、输入
token 数和 `per_token_latency`。请求校验、异常 abort 与正常请求释放均沿用 V1
`/generate` 语义，避免 benchmark 请求遗留 EngineCore 状态。入口回归为 **17 passed**。

## 独立入口节点身份可观测性（2026-09-03）

独立 V1 server 不依赖 Ray actor context 时，`/instance_list` 现在从本机 socket
解析发布 hostname 与可路由 IP；Manager/Llumlet 入口仍优先使用 Ray node metadata。
这样单机 TP=2、每节点独立 TP=1 以及跨机 P/D 三种部署形态都能使用统一拓扑字段。
入口回归 **16 passed**，并通过源码编译检查。

## 独立 V1 `/instance_list` 契约完整性（2026-09-03）

独立入口的实例列表现在补齐主 API 的兼容字段：`node_id`/`node_ip`、GPU block
总数/已用数/等待数，以及已有的显存、GPU 数、算力、请求和 KV affinity 字段。
V1 当前没有旧 block-manager 计数时以 0 返回，保持 JSON schema 稳定并避免客户端
分支处理。入口测试仍为 **16 passed**。

## 独立 V1 API 真实 Qwen3-14B 验收（2026-09-03）

在本机 CoreX 4.4/Python 3.12 上启动独立 `v1_api_server`（Qwen3-14B、TP=1、
`max_model_len=256`、`max_num_seqs=1`）完成真实接口验收：`GET /is_ready` 返回
`true/200`，`GET /instance_list` 返回 32 GiB 总显存、TP GPU 数 1 及计算容量，
`POST /generate` 返回 `200` 和非空文本。该结果确认新增运维端点与真实 EngineCore
生命周期连通，而不仅是 mock 回归。

## 独立 V1 API 运维接口补齐（2026-09-03）

独立 `v1_api_server` 现提供与主 Llumnix API 一致的 `/is_ready` 和
`/instance_list`。后者通过 V1 adapter 的统一实例信息更新路径发布 TP/PP GPU 数、
显存、计算容量、请求计数和 KV affinity block 数，避免单实例入口在迁移到 CoreX
后丢失拓扑与负载可观测性。新增接口回归后该入口测试为 **16 passed**。

## 双机源码基线同步复核（2026-09-03）

审计发现远端工作副本没有配置 Git remote，且原提交历史与本机镜像历史不同，不能以
`git pull` 保证两机运行相同代码。现通过 `git archive main | ssh ... tar -x` 同步当前
受版本控制源码；远端以本地提交形式记录为 `6d0a700`，随后在 Python 3.12/CoreX
环境运行 CoreX connector/CLI 定向集 **14 passed**。两机后续 P/D、KV affinity 和
多卡验收均应先同步同一源码快照；运行环境、模型与 `.ray-*` 目录不在该同步范围。

## 统一 Qwen3-14B 单机/TP=2 验证入口（2026-09-03）

`tools/run_qwen3_14b_smoke.py` 不再固定单卡：通过
`TENSOR_PARALLEL_SIZE` 选择并行度，并在启动前检查 `CUDA_VISIBLE_DEVICES`
是否足够，避免将 TP=2 错误提交到单卡。以
`CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 MAX_MODEL_LEN=256` 实测，
Qwen3-14B 完成 CoreX NCCL world size 2、TP rank 0/1、模型与 KV cache 初始化，
并返回非空中文生成；脚本报告 `qwen3_14b_corex_vllm: PASS`，耗时 25.50 秒。
同一入口可继续用于 TP=1 基础验证，相关脚本回归为 **4 passed**。

## TP/PP 多卡拓扑可观测性修复（2026-09-03）

V1 `InstanceInfo` 现在根据 `tensor_parallel_size × pipeline_parallel_size` 发布
逻辑实例实际占用的 GPU 数量；主 API `/instance_list` 不再将 TP=2 实例错误报告
为单卡。默认值仍为 1 以保持旧引擎和模拟器兼容。异构状态 API 回归已覆盖 TP=2
序列化，源码编译及 V1/KV 定向测试保持通过。

## 可重复的双机 GPU KV-staging 门禁（2026-09-03）

新增 `tools/corex44_zmq_kv_probe.py`，以 consumer-first 方式直接使用生产
`CoreXZmqP2pEngine` 验证 `GPU BF16 tensor -> CPU wire buffer -> TCP/ZMQ ->
CPU buffer -> peer GPU tensor`，不启动 Ray 或加载模型。当前在
`10.31.10.62` 与 `10.31.10.210` 上实测：producer 与 consumer 均为
`cuda:0`、`torch.bfloat16`、`(4,4)`，consumer 校验均值为 `7.5`。这将跨机
GPU 注入数据面固化为可重复的部署门禁；工具及 CLI 回归为 **2 passed**。

## 显式 CoreX P2P 配置与双机 BF16 传输复核（2026-09-03）

CLI 的 `--migration-backend-transfer-type` 现接受
`CoreXP2pNcclConnector`，因此部署配置可直接声明 CoreX ABI 兼容 connector，
不会在参数解析阶段被误拒绝；自动将上游 `P2pNcclConnector` 改写为该 connector
的原有行为不变。新增参数解析回归后，相关 V1 KV-transfer 集为 **14 passed**。

使用当前提交在 `10.31.10.62 -> 10.31.10.210` 实际执行 ZMQ CPU-staging
BF16 tensor 探针：发送端 `send_tensor` 返回 true，接收端得到
`torch.bfloat16`、形状 `(4, 4)`、均值 `7.5`。这复核了两机网络、Python 3.12
CoreX torch 与 connector 的真实跨主机 KV wire-format；完整 Qwen3-14B P/D
模型级 handoff 仍以本报告已有的 40-layer HTTP 成功证据为准。

## 最新 Python 3.12 完整单元回归（2026-09-03）

在本机 CoreX 4.4/Python 3.12 环境使用隔离 Ray runtime 执行
`tests/unit_test`（排除仅启动旧 vLLM 0.6 API 生命周期的 legacy
`entrypoints/vllm/test_api_server.py`），结果为 **119 passed、19 skipped**。
跳过项均为明确标注的旧引擎、缺少 async 插件或需要额外多 GPU/真实服务的路径；
V1 adapter、KV transfer、affinity、GlobalScheduler、API 和 CoreX placement
相关测试均执行通过。该结果与双机支持门禁及此前模型级 P/D handoff 证据共同构成
当前正式 Python 3.12/CoreX 栈的回归基线。

## 双机正式支持门禁可移植性修复（2026-09-03）

`tools/corex44_support_check.py` 现在自行定位项目根目录并加入模块搜索路径，因而
在仅执行 `source tools/corex44_env.sh`、未额外设置 `PYTHONPATH` 时也能从任意工作
目录运行。当前本机与 `10.31.10.210` 以该真实部署方式运行双机门禁均为
`supported=true`：Python 3.12.13、vLLM 0.11.2、PyTorch 2.7.1、Ray 2.52.1，且
V1 adapter、CoreX connector、两个 SHA256-CBOR affinity hash 与候选排序完全一致。
该脚本单测为 **3 passed**。

## 多卡弹性扩缩容 pending 保护（2026-09-03）

针对 CoreX 精简 Ray 缺失 State API 的场景，Manager 现在跟踪尚未完成 Llumlet
注册的 placement group，并将其纳入 `max_instances` 上限；超时 PG 会在后续周期
复用，避免重复申请多卡资源。新增单元回归验证注册实例与 pending 实例合计达到上限。

## CoreX 精简 Ray 下扩缩容上限修复（2026-09-03）

CoreX Ray wheel 没有 dashboard State API 时，自动扩缩容循环会将 placement-group
状态列表降级为空；旧逻辑因此无法判断 `max_instances`，可能在每个周期重复申请
GPU。现在在状态 API 不可用时以 Manager 已注册实例数作为存活 placement 的安全下界，
同时保留状态 API 可用时的精确回收逻辑。V1/KV/调度及该上限回归共 **20 passed**。

## V1 reactive auto-scaling CLI 兼容修复（2026-09-03）

审计发现 `Manager`/`ScalingScheduler` 已实现基于 `virtual_usage` 的 V1 弹性扩缩容，
但 `ManagerArgs.check_args` 仍保留旧版“禁止 enable_scaling”的断言，导致该论文创新
能力无法从 CLI 启用。现移除遗留断言，保留 `min_instances`/`max_instances` 正值及
范围校验；V1 scaling flag 回归通过，GlobalScheduler/KV 组合测试 **23 passed**。

## Python 3.12 单元测试隔离修复（2026-09-03）

测试夹具的 `pytest_sessionstart` 现显式使用 `ray.init(address="local")`，确保单元
套件在机器已有部署 Ray head 时仍创建私有进程内控制面，不附着共享集群或复用遗留
actor。修复后的 Manager、V1 connector、KV affinity 定向验证为 **34 passed**；
完整套件仍包含历史 Ray-heavy 测试，需按测试选择在独立环境执行。

补充修复测试夹具清理：CoreX 精简 Ray 的 `list_actors()` 依赖 dashboard，不可稳定
返回命名 actor；清理器现在优先使用核心 API `ray.util.list_named_actors`，并兼容
Ray 2.52 返回的字典结构。结合动态 ZMQ 端口和 `address="local"`，entrypoint、
GlobalScheduler 与 V1/KV 分组回归可独立完成（本轮新增分组 **11 passed**）。

随后将 ZMQ 动态端口、legacy API 收集顺序及命名 actor 清理修复合并验证：覆盖 V1
KV transfer、V1 API、KV affinity、GlobalScheduler、support gate 共 **78 passed**。
未再出现固定端口冲突或旧 API 误执行；未纳入的历史 Ray-heavy migration 用例仍
单独保留为 legacy 范围。

进一步修复测试隔离：ZMQ 单元测试不再固定绑定 `127.0.0.1:1234`，改为系统分配
空闲端口并显式清理 server。另修正 V1 环境下 legacy `entrypoints/vllm/test_api_server`
的收集过滤顺序，避免已移除的 vLLM 0.6 API 被错误执行。完整回归已越过该失败点，
并在 143 个测试中通过前半段；剩余历史 Ray-heavy 用例仍需拆分运行。

> 回归执行边界：远端工作副本已同步包含本轮 V1 abort-routing 修复，双机支持门禁
> 输出与本机一致。P/D 服务占用共享两张 GPU 时，全量 Ray 单元套件会因 fixture
> 资源等待而阻塞；已主动终止该运行以保护验收服务，不将资源竞争误记为代码失败。
> V1/KV/调度定向套件保持 **50 passed**；完整套件应在隔离 Ray head 上运行。
> 本轮停止了验收用的 Llumnix serve/Llumlet 以释放两张 GPU；共享 Ray head 本身未被
> 停止。即使 GPU 已空闲，全量套件仍会自动复用该共享 runtime 中遗留的 APIServer/
> Queue actor 并在 fixture 初始化等待，因此完整测试需要使用全新的独立 Ray 地址，
> 而非仅释放 GPU。

## 当前提交双机 P/D 复验（2026-09-03）

在清理历史 detached Manager/actor 且两张 GPU 均空闲后，以当前提交启动共享
Ray 集群 `10.31.10.62:6408` 的两个 TP=1 Qwen3-14B 实例。`/instance_list`
确认 decode 位于 `10.31.10.62`、prefill 位于 `10.31.10.210`，并分别发布
`10.31.10.62:14579` 与 `10.31.10.210:14579` connector endpoint。发送短中文
请求后，HTTP `/generate` 返回 200 和非空文本；两端 EngineCore 日志记录同一
P/D request ID，同时包含 decode/prefill 两端地址，consumer metadata/load 与
producer metadata/load 均被触发。该结果是当前 Python 3.12/CoreX 4.4/vLLM
0.11.2 代码的最新模型级跨机 KV handoff 证据。

## V1 P/D 请求取消语义修复（2026-09-03）

`disable_log_requests_manager` 过去会连同 request-to-instance bookkeeping 一起
关闭，导致请求日志被关闭时 Manager 无法在后续 abort 中找到实例；对 V1 P/D 而言
这可能只取消一端或遗留 producer/consumer 流。现在日志开关只影响输出，路由表始终
维护公开 request ID 到单实例或 P/D 双实例集合的映射，因此取消会继续 fan-out 到
两端。新增回归并执行 V1 connector、KV affinity、调度定向集，结果 **50 passed**。

## Llumnix V1 TP=2 本机双卡基础验证（2026-09-03）

在隔离的单机 Ray head（2 GPU）上启动 Llumnix global serve，使用本地
Qwen3-14B 完成真实推理。vLLM V1 日志确认 `world_size=2`、TP rank 0/1、模型
权重和 KV cache 均初始化成功；Llumnix API 的 `/health` 和 `/generate` 分别
返回 HTTP 200 和非空中文文本。

为使该链路符合 V1 的运行模型，placement group 将 TP GPU 打包到 Llumlet 父
actor 所在 bundle，TP>1 在已连接 Ray 集群中采用 vLLM `mp` executor。前者保证
父 actor 能向子进程暴露完整 GPU 集合，后者避免 vLLM Ray executor 重新连接并
传入资源计数。该验证覆盖 Python 3.12、CoreX 4.4、NCCL、Qwen3-14B 和
Llumnix global API 的基础多卡功能；跨实例 P/D KV handoff 仍按本文既有证据
单独表述，未由本次单实例测试替代。

## TP=2 多卡适配增量（2026-09-03）

### 跨机 P/D Ray actor endpoint 归属修复（2026-09-03）

V1 connector endpoint 现在优先使用 Llumlet 所在 Ray 节点的实际地址，而不是全局
serve driver 或可能从另一台机器继承的 `kv_ip`。显式 `LLUMNIX_KV_IP` 仍可覆盖，
用于多网卡部署。这样当 driver 在 `10.31.10.62`、actor 在 `10.31.10.210` 时，
远端 actor 会正确发布 `10.31.10.210:<port>`，保证 P/D request-id 路由可达。
对应 connector/serve 回归为 **33 passed**。

两节点真实 GPU 探针已分别在 `10.31.10.62` 与 `10.31.10.210` 的 BI-V150 上
执行成功，Ray 能将任务分配到两台 Python 3.12/CoreX 设备并返回一致计算结果。
针对 TP>1 的 V1 部署，修复了 CPU Manager actor 调用 vLLM `create_engine_config()`
导致的错误 GPU 可见性校验：placement 规划现在直接使用 `tensor_parallel_size ×
pipeline_parallel_size`。该修复已由单元测试覆盖并提交 `569a72a`。

本轮 TP=2 Qwen3-14B 尝试因 Ray 中仍存在旧 detached Manager 而未使用该修复，故不
宣称模型级多卡 HTTP 通过；清理旧控制面后再进行正式验收。

随后在本机两张 BI-V150 上完成原生 vLLM V1 TP=2 实测：两张卡均完成 NCCL rank
初始化，Qwen3-14B 八个 safetensors 分片加载完成，每卡约 13.88 GiB 权重并成功
分配 KV cache；`/health` 与 `/v1/completions` 返回成功，中文生成非空。首次尝试
在首 token Triton 编译时因链接器找不到 `-lcuda` 退出，确认并通过 CoreX library
path 环境修复；这同时验证了 Python 3.12/CoreX 栈的多卡编译运行要求。该证据是
真实多卡模型运行通过，不等同于 Llumnix 跨实例 P/D KV handoff 通过。

Llumnix placement 层也完成 TP 拓扑防护：实例使用 `STRICT_PACK`，因此一个 TP=2
实例必须落在同一 Ray 节点。两机各登记一张 GPU 时，集群总数虽为 2 但不能安全地
组成 TP=2；现在会在创建 placement group 前报告这一条件，而不是等待超时并遗留
资源。实测确认该集群的正确跨域形态是两个 TP=1 实例（每机一个），再由论文第五章
的异构虚拟负载、KV affinity 和 P/D handoff 协同。对于同机 TP=2，Ray head 必须
将同机两张物理卡都注册为 GPU 资源。

最新提交 `c3a0c8c` 已同步至 `10.31.10.210` 并在远端 Python 3.12/CoreX 环境
复验：V1 KV transfer、affinity 和 TP topology 定向集 **41 passed**，源码编译通过。
两端的 affinity 探针输出同一组 `sha256_cbor` hashes 和候选排序，说明本报告中的
跨机缓存亲和结论覆盖当前源码，而非仅覆盖历史提交。

## 全量 Python 3.12 回归复核（2026-09-03）

异步插件复核还发现历史 Manager GPU placement 测试原先只检查 torch 可见设备，
而隔离 Ray 夹具实际注册 GPU=0，导致不可能的 placement 等待；测试已改为检查
`ray.cluster_resources()` 并在资源不足时显式 skip，同时移除 vLLM 0.11 删除的
`worker_use_ray` 参数。多实例 Ray 测试会话必须使用独立 runtime，否则 Ray 会报告
多个 active instance 地址冲突；该环境问题与 CoreX V1 服务代码无关。

修复 P/D Decode affinity 信息可见性：Decode-only 实例不属于普通 dispatch 集合，
但必须参与 P/D 角色池的 KV-aware 选择。Manager 现在在受限选择前同步全量实例信息，
同时 `DispatchScheduler.dispatch` 仍仅过滤普通 dispatch 集合，避免普通请求误路由到
Decode。新增隔离回归后 dispatch scheduler 为 `12 passed`，统一 V1 unit runner
为 `68 passed`。

2026-09-04 最新源码全量执行 `python -m pytest -q` 为 **129 passed, 42 skipped**，
耗时 216.72 秒。42 项 skip 均有明确范围：vLLM 0.6 私有 block-manager 迁移后端、
至少 4 GPU 的历史 benchmark，或未安装 `pytest-asyncio` 时显式跳过的历史异步用例。
没有将当前 Python 3.12/vLLM V1 导入失败隐藏为 skip。随后使用包含 P/D affinity
role selection 与 Decode-only 记账修复的 Manager 重新运行双机 integration：两端
source gate 一致，16 个真实 `BlockStored` 事件让远端 affinity=1.0 并优先缓存候选，
BF16 GPU staging producer/consumer 均 PASS。

此回归证明当前正式 V1 路径的功能稳定；它不将已跳过的旧 block-manager 任意时刻
migration benchmark 等同于已迁移功能。V1 的正式迁移语义仍是公开 connector 支持的
P/D KV handoff，且已通过上述真实双机数据面验证。

针对“原先功能和创新均迁移到当前 CoreX 栈”的适配审计，执行了项目全量测试：
修复 vLLM 0.11 V1 中 `EngineArgs` 删除 `worker_use_ray` 导致的旧测试构造错误，
并修复 Ray 在 placement group 预留失败的半初始化 Llumlet repr 路径。当前结果为
**115 passed、42 skipped**；跳过项主要是需要真实旧版模型/引擎或多 GPU 的历史 E2E，
不是 Python 3.12 导入失败。

在最新提交上又以隔离本机 Ray fixture 运行 `tests/unit_test`：除明确启动旧
vLLM 0.6 `api_server.py` 生命周期的 legacy `test_api_server.py` 外，结果为
**119 passed、19 skipped**。该排除项的行为由当前 V1 `/generate` API 与两机
P/D 实测覆盖；其余 skip 均为无 GPU/缺少 async 插件或旧 block-manager 专用路径。

新增 `tools/corex44_support_check.py` 作为不分配 GPU、不启动 Ray、不下载模型的
双机预检门禁。它验证 Python/vLLM/Ray/PyTorch 版本、`V1EngineAdapter` 与
`CoreXP2pNcclConnector` 导入，以及固定 token blocks 的 affinity hashes。本机和
`10.31.10.210` 在提交 `0031298` 上输出完全一致：Python `3.12.13`、vLLM
`0.11.2`、Ray `2.52.1`、PyTorch `2.7.1` 及相同两个 SHA256-CBOR hashes。
该脚本现为严格门禁：Python、vLLM、PyTorch 或 Ray 版本偏离正式支持范围时以非零
状态退出，并输出具体错误；门禁自身单元测试为 `2 passed`。
支持 `--remote-host congmng@10.31.10.210` 的一键双机比较，自动通过 SSH 执行
远端预检并比较版本与 affinity hashes；当前两机结果为 `supported=true`，门禁测试
增至 `3 passed`。

本机和 `10.31.10.210` 的 Python 3.12/CoreX 环境再次运行跨主机 affinity 探针，
对同一 token blocks 得到逐字节一致的 `sha256_cbor` 哈希和候选排序。这为统一
虚拟负载、KV event 索引和 prefix affinity 的跨域调度实现提供了当前栈证据。
本轮尝试重新启动单卡 Qwen3-14B 服务时，CoreX 驱动报告显存分配 OOM，服务端口
未启动；该现象仅限制本轮新增 HTTP 证据，不改变此前已记录的成功单/双实例 API
及 ZMQ CPU-staging P/D 验证，也与网络代理或模型下载源无关。

## 全局 P/D 启动的既有 Ray 集群连接（2026-09-03）

全局 `serve` 入口此前不传地址调用 `ray.init()`，可能在已有两机 head 旁创建
第二个本机 runtime，令 Ray State API 无法选择集群并阻断 P/D 编排。现在入口
显式连接 `HEAD_NODE_IP:ray_cluster_port`，并由回归测试覆盖。使用已有集群的
全局启动还必须传 `--no-launch-ray-cluster`；该修复消除了 P/D connector 验收前的
控制面歧义。

## V1 P/D endpoint 跨机修复（2026-09-02）

V1 KV-transfer 配置不再在未设置环境变量时默认使用 `127.0.0.1`；配置器会
解析当前 Ray actor 所在主机 IP，`LLUMNIX_KV_IP` 仍可用于多网卡覆盖。这样
producer 发布给 consumer 的 endpoint 在两台服务器之间可路由。对应定向回归
29 项通过；该项是 P/D connector 编排的地址前置条件，不等同于 GPU KV tensor
传输本身已完成。

## 双机 affinity 算法复现（2026-09-02）

新增离线探针 `tools/cross_host_kv_affinity_probe.py`，在本机和
`10.31.10.210` 的 Python 3.12/CoreX 环境分别运行。两端对 `[1..8]`、block size
4 计算出完全一致的 `sha256_cbor` 哈希，并对模拟的 BlockStored 事件得到
`candidate-a=0.5`、`candidate-b=1.0`，确定性排序为 `candidate-b,candidate-a`。
这验证了 hash、事件索引和候选排序的跨主机算法一致性；不替代真实 GPU KV
tensor handoff 验收。

## 双机主 API 与拓扑可观测性（2026-09-02）

将 Ray head 保持在 `ray start --block` 的持久会话中、清理两端旧会话后，
`10.31.10.62:6403` 稳定显示 8 CPU/2 GPU；node-affinity actor 分别在 `u62` 与
`u210` 运行。基于该集群启动两个 Qwen3-14B TP=1 V1 Llumlet，两端均完成模型
加载和 KV cache 分配，主 API 的 `/health`、`/instance_list`、`/generate` 均
实测通过，两个并发请求也均完成。

`InstanceInfo` 与 `/instance_list` 现在发布 `node_id`/`node_ip`，因此部署可
直接核验调度实例的跨主机位置及其显存、算力、虚拟负载、KV affinity 状态。访问
内网 API 时需排除环境 HTTP 代理（例如 `curl --noproxy '*'`）；代理返回的 502
不是服务健康状态。该验证证明双机 V1 Manager/Llumlet 的基础分发闭环；不把它
等同于任意时刻 block-manager migration 的 V1 移植。

## 跨机 Ray runtime 复核（2026-09-02）

早期以短生命周期命令启动的 head 会在执行器回收后台进程后表现为 GCS refused。
使用持久 `ray start --block` 会话并清理遗留 Ray 会话后，两机已稳定组网并完成
主 API 验收。因此该现象不是端口防火墙或 Llumnix 代码错误；跨机部署应明确将
head 作为持久服务管理，而不应依赖短生命周期 shell 后台进程。

## 本轮基础复核（2026-09-02）

对照论文第五章第三点 HLA（异构负载感知跨域请求调度），本机以 Qwen3-14B
单卡 V1 API 完成 `/health`（200）和中文 `/generate`（非空结果）验证；调度、
GlobalScheduler、KV affinity 定向测试 22 项全部通过。该结果确认基础推理与
虚拟负载/亲和决策链路可运行，不将单实例结果扩展为跨域迁移性能结论。

验证时间：2026-09-01 至 2026-09-02

持续进度记录见 [Progress_Log.md](./Progress_Log.md)。本阶段提交包含 Python 3.12
包元数据修复、Ray 2.52 测试兼容、V1 KV 事件亲和索引与两机模型级 P/D handoff。

最新的 V1 迁移进展已加入 connector 配置层：当显式设置
`migration_backend=kvtransfer` 时，Llumnix 会在创建 `AsyncLLM` 前映射为
vLLM V1 的 `KVTransferConfig` 和 `KVEventsConfig`。支持
`SharedStorageConnector`（适合先做单机/共享目录验证）以及
`P2pNcclConnector` 等原生 connector。CoreX 默认选择安全的 `zmq_cpu`
staging P2P transport；原生 NCCL 路径保留为显式性能实验模式。原有
gloo/nccl/rayrpc block-manager 迁移协调器仍只适用于旧 vLLM 后端，V1 以
connector 驱动的 P/D KV handoff 取代它们。

## 结论

### 当前部署验收口径（2026-09-03，复核）

本次复核将历史尝试与当前可复现实证分开记录。共享 Ray 集群
`10.31.10.62:6408` 当前可见两节点、2 GPU（每节点 1 GPU）；两机 TP=1 是
跨域部署的正式形态，单机 TP=2 则必须使用仅含同一节点两张 GPU 的隔离 Ray
head。两种形态均已完成模型级基础推理验证。启动全局 serve 连接已有 head 时，
必须设置 `RAY_ADDRESS=10.31.10.62:6408`（并传 `--no-launch-ray-cluster`）；
否则 Ray 可能在本机自动创建第二个 runtime，进而出现 placement GPU 不足或
State API 多集群歧义。该要求属于部署前置条件，不是模型或网络故障。

在无残留 actor、GPU 资源空闲的干净集群上，当前提交的正式证据仍为此前记录的
两机 Qwen3-14B P/D 闭环：producer 发送 40 层 BF16 KV，consumer 执行
`load(role=consumer)`，HTTP `/generate` 返回 200 和非空文本。本轮曾因重复的
历史 serve/actor 占用资源而无法追加并发样本；清理仅针对遗留 Llumnix actor，未
执行广泛 `ray stop`，不影响上述已完成的 handoff 验收。

支持门禁在本机及 `10.31.10.210` 复核均通过：Python 3.12.13、vLLM 0.11.2、
PyTorch 2.7.1、Ray 2.52.1，V1 connector 导入成功，affinity hashes 与候选
排序逐字节一致；相关单元测试与 V1/API/KV affinity 定向测试本轮共 **36 passed**。

## 2026-09-03 P/D 兼容性回归（最新）

### 正式支持审计补充（2026-09-03）

为避免 V1 部署仍构造已移除的 vLLM 0.6 block-manager migration coordinator，
Llumlet 在检测到 V1 adapter 后不再初始化 legacy coordinator/scheduler；任何旧
`migrate_out` 调用保持安全拒绝并明确提示使用 connector-driven P/D KV handoff。
这不会收窄已验证功能：V1 的正式替代路径仍为两机 producer/consumer connector
handoff；它只消除未使用旧对象的隐性依赖与误导性“未完成迁移”告警。对应 API 与
KV transfer 回归为 **46 passed**。

当前本机正式环境为 Python `3.12.13`、vLLM `0.11.2`、Ray `2.52.1`、PyTorch
`2.7.1`（CoreX 4.4）。`pip wheel . --no-deps --no-build-isolation` 成功生成
`llumnix-0.0.2-py3-none-any.whl`。V1 transfer、placement、Manager 调度定向集
为 **44 passed、10 skipped**；跳过项是依赖历史 async/旧 block-manager 的测试，
不属于当前 V1 生产路径。

同一提交在两台服务器重新运行 `cross_host_kv_affinity_probe.py`，两端输出逐字节
一致：两个 `sha256_cbor` block hash 分别为
`c9d58ba6...28ef071cb`、`24125b23...154ab2d6`，`candidate-a/b` affinity 为
`0.5/1.0`，排序为 `candidate-b,candidate-a`。为使 API 实机拓扑信息在 CoreX Ray
最小 worker context 中稳定可见，Llumlet 对空 `get_node_ip_address()` 增加 Ray
node-table 的 `NodeManagerAddress` 回退；这不会改变调度决策，仅保证
`/instance_list` 正确暴露实际跨主机位置。

使用最新提交 `36e5b77` 重启两机 P/D 服务后，`/instance_list` 已实测返回
`10.31.10.62` 与 `10.31.10.210`（每节点一个 TP=1 Qwen3-14B）。请求
`pd-topology-001` 返回 HTTP 200 和非空中文文本；日志同时确认本机 consumer 与
远端 producer 收到同一含两端 endpoint 的 request ID。该复验确认 topology
observability 回退不会改变已验证的 P/D handoff。

本次真实请求进一步暴露 CoreX PyTorch 的 `bfloat16.numpy()` 限制：producer 在
保存 KV 时因 NumPy bridge 不支持 BF16 而退出。提交 `ae81d39` 将 staging wire
protocol 改为 uint8 原始字节视图并按 dtype/shape 重建，新增 BF16 round-trip
回归（V1 KV transfer 定向测试 31 项全部通过）。下一轮真实请求应重新验证完整
producer/consumer 输出闭环。

在共享两机 Ray 集群重新拉起 `initial_instances=2` 的 P/D 部署时，首个实例已
成功创建并开始加载 Qwen3-14B；此前 CoreX Ray `ActorOptionWrapper` 不支持
`.options()` 的兼容性错误已由提交 `fa05a26` 修复，Llumlet 能够进入 vLLM V1
EngineCore 初始化。第二个实例失败的直接原因是本机隔离 TP=2 基线残留的
EngineCore 占用两张卡，vLLM 报告每卡仅剩 `0.93/32.0 GiB`，低于
`gpu_memory_utilization=0.90` 所需的约 `28.8 GiB`；该次失败不是模型下载、代理
或 P/D 网络错误。清理残留进程后共享集群已恢复为 `2 GPU` 空闲，旧 detached
Llumnix actors 也已清理。P/D 端到端 handoff 的历史 ZMQ CPU-staging 证据仍然
有效，但需在无其他 GPU 进程的干净集群上重新执行本轮模型级验收后，才能将其
作为当前提交的最新复现证据。

修复后在干净两机集群重新执行 `pd-bf16-001`：两台 BI-V150 各运行一个 TP=1
Qwen3-14B 实例（Prefill 位于 `10.31.10.210`，Decode 位于 `10.31.10.62`）。
两端均完成模型加载、KV cache 创建和 `CoreXP2pNcclConnector` ZMQ staging 初始化。
请求日志确认 producer 发送 40 层 BF16 KV（shape `(2, 298, 8, 16, 128)`），
consumer 收到共享 request metadata 并执行 `load role=consumer`；HTTP
`POST /generate` 返回 200 和非空中文文本。这是当前提交在 Python 3.12/CoreX
4.4 上的模型级两机 P/D KV handoff 闭环证据。

CoreX 驱动、PyTorch、Ray 和 Llumnix 控制面已验证可用，整个过程未修改驱动或
系统级 CoreX 安装。

Llumnix 的旧 vLLM `0.6.3.post1` 私有 block-manager 后端不能直接运行于
`vllm 0.11.2+corex.4.4.0`；项目已为此提供 V1 adapter，而非用空兼容层掩盖
旧接口。当前正式验证范围为：**Python 3.12/CoreX 4.4 的 V1 单实例 API、
Ray/Llumlet 队列桥接、KV event/hash affinity、GlobalScheduler 缓存亲和调度，
以及两机 Qwen3-14B P/D KV handoff 均可运行。**

| 能力 | Python 3.12/CoreX 4.4 V1 状态 | 当前验证 |
| --- | --- | --- |
| 单实例与本机 TP=2 推理 | 正式支持 | Qwen3-14B HTTP 200；TP rank 0/1 实测 |
| 双机异构负载与 topology 可观测性 | 正式支持 | 两节点 GPU/显存/算力、node IP 实测 |
| KV event/hash affinity | 正式支持 | 双机 hash、affinity、候选排序逐字节一致 |
| Prefill/Decode KV handoff | 正式支持（ZMQ CPU staging） | 两机 BF16 40 层 KV、consumer load、HTTP 200 |
| native NCCL P2P transport | 非默认诊断/优化项 | CoreX rank-1 communicator 不作为生产路径 |
| vLLM 0.6 任意时刻 request migration | 不适用于 V1 | 明确拒绝；由 P/D handoff 提供 V1 替代语义 |

旧式任意时刻 block-manager request migration 没有被声明为 V1 功能；在 V1 架构
中，其可安全替代路径是 connector 驱动的 prefill/decode KV handoff。原生 NCCL
rank-1 communicator 在该 CoreX 栈仍会 native abort，因此生产默认 transport 为
已验证的 ZMQ CPU staging，NCCL 为非默认诊断/优化选项。

针对论文第三点“基于异构负载感知的跨域请求调度”，V1 实例信息现在额外发布
实际可见 GPU 总显存、空闲显存和归一化算力容量。`virtual_usage` 调度模型用
请求数/算力容量表示计算压力，并用显存占用率表示内存压力；缺少新指标的旧
后端继续回退到 block-counter 估计。这样不同显存容量或计算能力的 CoreX 实例
可以进入同一调度决策，而不会把所有实例视为同质资源。该模型已由两组单元测试
覆盖（含异构容量排序和旧接口回退）。

补充完成本机真实双实例部署验证：本机两张 BI-V150 各启动一个 `TP=1` 的
Qwen3-14B V1 Llumlet，并由同一 Manager 注册为两个可调度实例。每实例显示
32 GiB 总显存、约 27.52 GiB 模型驻留和 17,344-token KV cache；`/health`、
`/instance_list` 与 `/generate` 均返回 HTTP 200。发现并修复空闲实例内部
`-inf` 调度哨兵无法 JSON 编码的问题：外部 `/instance_list` 现在显示 `null`，
内部调度仍保留最小负载语义。相应 API/KV-affinity 定向回归为 24 passed。
这为第三点的多实例异构状态可观测和请求入口提供实机证据；跨域迁移结论仍仅以
已验证的双机 connector P/D handoff 为准。

随后将提交 `1ba0dea` 通过 bundle 同步到 `10.31.10.210`。远端同样以 Python
3.12.13、CoreX 4.4.0、vLLM 0.11.2 完成核心定向回归和源码编译检查；其目录未
发现完整 Qwen3-14B 权重，故没有重复传输大模型。两端对相同 token blocks 计算
的 `sha256_cbor` prefix hashes 逐字节相同，补充证明 affinity 索引在两台机器
的运行时环境中具有确定性。远端 Ray 版本为 2.58.0，本机为 2.52.1，V1 适配
代码在两版本上均通过上述回归。

2026-09-02 在两台服务器上以同一提交 `5893493` 复核：`10.31.10.62` 的
`virtual_usage` 为 `0.00468`（空闲 `34,141,634,560` B），`10.31.10.210` 为
`0.00522`（空闲 `33,999,028,224` B）；两端均识别为 32 GiB BI-V150。该差异来自
实测可用显存，表明跨主机上报与统一负载计算生效。两端对同一 token blocks 的
`sha256_cbor` 哈希仍逐字节一致，且各自执行 V1/KV/HTTP 定向回归均为 44 passed。

## 最新适配增量（2026-09-01）

`V1EngineAdapter` 现在订阅 vLLM V1 的 ZMQ `KVEventBatch`，并把 block 所有权
同步到 `KVCacheAffinityIndex`；`InstanceInfo` 透传这些 hashes，调度器在负载差
不超过 0.10 时优先已有 prefix 的实例。普通 `/generate` 会从一个 live V1
实例 tokenizer 计算与 EngineCore 一致的 `sha256_cbor` block hashes，因此不需要
客户端自行构造 hash。P2P connector 的内部地址后缀仅用于 vLLM connector，输出
前会移除，保证公开 request ID 不变。

这部分当前由提交 `02b8857` 提供，已推送至
`https://github.com/congmng/tianshu-huiyang`；本机相关测试通过。

跨主机算法层已在 `10.31.10.62` 与 `10.31.10.210` 之间验证：前者发布 vLLM
`BlockStored` ZMQ event，后者在同为 Python 3.12/vLLM 0.11.2 的 CoreX 环境中
成功订阅解码。两端在固定 `PYTHONHASHSEED=0` 下计算同一 token prefix 的
`sha256_cbor` block hashes 也完全一致。这证明事件汇聚和 cache-affinity 计算
可跨主机进行；尚未作为 P2P NCCL GPU KV tensor 迁移成功的证明。
（本段记录早期仅完成事件/哈希验证的阶段性状态；当前结论以文档后面的
“两机模型级 P/D handoff：已验证（ZMQ CPU staging）”为准。）

当前代码也已通过 Git bundle 同步到 `10.31.10.210` 的 `6a1f5c2`；远端只读核验
为 Python `3.12.13`、vLLM `0.11.2`、PyTorch `2.7.1`，与本机正式 V1 验证栈一致。
这使已完成的双机 affinity/P/D 验证可复现于当前源码，而不需要修改远端驱动或
CoreX 系统安装。
CoreX 环境脚本现在在 Python 启动前设置 `PYTHONHASHSEED=0`（可显式覆盖），
避免 vLLM 在 affinity 哈希探针中报告不可复现；两台机器应使用同一值。
环境脚本不再硬编码项目 clone：默认优先使用 `.conda-corex44`，当其不存在时
自动选择远端已验证的 `/data1/congmng/conda-envs/ds-corex44`，也可通过
`LLUMNIX_COREX_PYTHON_ENV` 显式指定。这使同一套双机验证脚本适用于两台主机。
在当前 `6a1f5c2` 基线下，两机对 token blocks `[1,2,3,4]`、`[5,6,7,8]`
计算的 `sha256_cbor` 结果逐字节一致：
`c9d58ba695280d69b243e1e0df813136ca9196b286fb1a021e0b2e028ef071cb`、
`24125b23e68883b5c2141db2959d48433fe6bde2f26bd914efad121d154ab2d6`。

最新 P2P 数据面验证已完成：CoreX 4.4.0 附带的 NCCL 2.24.3 不含 vLLM ctypes
wrapper 会探测的可选 symmetric-memory window 符号。项目新增的
`CoreXP2pNcclConnector` 仅在当前进程中过滤这两个未使用的可选符号，不改驱动、
SDK 或 `libnccl.so`。两台机器各自启动 vLLM 自带 `P2pNcclEngine` 后，rank 0/1
communicator 在 `ens1f0` 上成功初始化，GPU tensor `[1,2,3,4]` 已从本机发送并在
`10.31.10.210` 完整接收。这是实际 NCCL send/recv 数据面证据；完整模型级
P/D 仍需在两端部署相同权重后做端到端验证。

补充配置兼容：用户若直接传入 vLLM 原生的
`KVTransferConfig(kv_connector="P2pNcclConnector")`，即使没有选择 Llumnix
`migration_backend=kvtransfer`，也会重定向到本项目的 CoreX ABI shim；其余显式
vLLM 配置不被改写。consumer 的 P2P 请求 ID 现可编码 prefill 地址。（本段为
双请求编排完成前的历史记录；当前实现和验证见下段及后文。）

V1 P/D 编排现已进入可执行的第一阶段：Manager 在 prefill/decode 实例均报告
可路由 P2P endpoint 时，为同一个公开 request ID 启动两个 `AsyncLLM` 请求；
producer 的输出被抑制并限制为 handoff 所需的一步，consumer 用原始采样参数
负责所有公开输出。取消请求会广播到两个实例；consumer 启动失败时会反向清理已
启动的 producer。endpoint 尚未就绪时安全回退到原有单请求路径。该实现尚缺两机
同权重模型级验证，不能据此宣称端到端 KV 复用已完成。

Manager 现在还会验证两端 endpoint 必须是有效的 `host:port`（端口范围
1--65535）；配置错误会在 P/D 编排前安全回退单请求，避免半启动的 producer 或
consumer 进入不可恢复等待。

Manager 生成的共享 P/D 内部 request-id（同时携带 decode/prefill endpoint）现在
在两侧 V1 adapter 中原样保留；adapter 不会用公开 request-id 重新装饰并丢失另一
端 marker。这保证 producer 和 consumer 解析同一份跨主机路由信息。
V1 adapter 在向 EngineCore 提交请求同步失败时会回滚本地 request、alias 与 running
bookkeeping，避免配置/启动异常后留下孤立请求，适用于独立 API 和 Llumlet 两条入口。

## vLLM V1 迁移进展（2026-09-01）

针对本机软件栈的 `vllm 0.11.2+corex.4.4.0`，已加入第一阶段 V1 迁移代码：

- `llumnix/backends/vllm/v1_engine.py`：使用 `vllm.v1.engine.AsyncLLM`，不再导入 vLLM 0.6 的 `_AsyncLLMEngine`。
- `llumnix/backends/utils.py`：检测 vLLM 0.11 后选择 V1 adapter；旧版本仍保留旧后端路径。
- `llumnix/llumlet/llumlet.py`：将 V1 异步生成结果桥接到 Llumnix 原有输出队列，并显式关闭尚未迁移的 KV-cache migration。
- `llumnix/entrypoints/vllm/client.py`：补充 vLLM 0.11 缺失的旧 `AsyncStream` 的队列兼容实现。
- `llumnix/arg_utils.py`：兼容 vLLM 0.11 参数注册时的 `deprecated` 关键字。

V1 adapter 已用本机完整 Qwen2.5-VL-7B 权重完成单卡生成及真实 Ray/Llumlet 队列回传验证；Qwen3-14B 也已完成下载并通过直接 vLLM V1 单卡生成验证。

随后已完成真实 Ray/Llumlet 单实例验证：以 `CUDA_VISIBLE_DEVICES=0`、Ray `num_gpus=1` 和一个 placement group 启动 `Llumlet`，请求 `req1` 经 V1 adapter 生成并回传 RayQueue，收到文本片段 `I`（`finished=False`，流式中间输出），证明 Manager/Llumlet 侧的异步队列桥接已工作。该测试使用 Qwen2.5-VL-7B 本地完整权重。

为适配 CoreX 节点上的 Ray 启动，`setup.py` 现在根据 `CUDA_VISIBLE_DEVICES` 在 `ray start --head` 时显式注册 GPU 数；连接已有集群时不重复传递 `num_gpus`。这修复了 placement group 看不到 GPU 的问题。Qwen3-14B 的默认 128 路 dummy sampler 预热会在单卡 32 GiB 上 OOM；将 `--max-num-seqs` 降为 `4` 后，完整 Llumnix HTTP 服务成功启动，`GET /health` 返回 200，`POST /generate` 返回 200，并获得正常文本生成。Qwen2.5-VL-7B 的完整 API（`/health`、`/generate`）也已通过。

Qwen3-14B 完整 API 的实测启动参数如下（仅用于单卡基础服务验证）：

```bash
source tools/corex44_env.sh
CUDA_VISIBLE_DEVICES=0 HEAD_NODE=1 HEAD_NODE_IP=127.0.0.1 \
  .conda-corex44/bin/python -m llumnix.entrypoints.vllm.api_server \
  --config-file configs/vllm.yml \
  --model .models/Qwen3-14B \
  --host 127.0.0.1 --port 18011 \
  --initial-instances 1 --launch-ray-cluster \
  --dtype float16 --gpu-memory-utilization 0.96 \
  --max-model-len 512 --max-num-seqs 4 --enforce-eager
```

对应请求为 `{"prompt":"用一句话介绍 Iluvatar CoreX。","max_tokens":16,"temperature":0.0}`；服务返回 `200`，文本包含“`Iluvatar CoreX 是一个基于区块链的去中心化计算平台`”。这是单实例请求分发与输出桥接的验证，不表示旧版 KV-cache migration 已恢复。

注意：vLLM V1 的运行时编译会查找 CUDA 头文件；项目环境脚本现在将 `/usr/local/corex-4.4.0/include` 加入 `CPATH`/`C_INCLUDE_PATH`，避免 `cuda.h` 缺失。驱动核验仍为 IX-ML/Driver 4.4.0，未执行任何驱动变更。

本轮继续适配时，V1 模式又在三个入口处强制隔离旧迁移代码：CLI 参数解析、`Manager` 构造和 `Llumlet.migrate_out`。即使经由全局部署或程序化 API 构造 Manager，也不会调用依赖 vLLM 0.6 block manager 的迁移协调器；显式迁移调用会返回空结果并记录警告。V1 请求完成与中止时会清理 adapter 的运行中请求记录，API client 也会清理本地 fallback 分发计数。

此前完整旧式 `api_server` 的停滞根因已定位为 Ray 未注册 GPU；现已修复，并以 Qwen3-14B 完成 Llumlet 加载、HTTP 健康检查和实际 `/generate` 请求。单卡 32 GiB 需要将 V1 的 `--max-num-seqs` 限制为 `4`，以避免 dummy sampler 预热 OOM。`compileall` 已在本轮再次通过。现有旧单测套件无法直接执行，原因是其 `conftest.py` 导入已在 Ray 2.52 移除的 `ray._private.utils.hex_to_binary`。

Qwen3-14B 已通过 ModelScope 直连完成断点下载（8 个 safetensors 分片、`model.safetensors.index.json` 均存在，目录约 29 GiB）。在 `CUDA_VISIBLE_DEVICES=0` 下使用本机 `vllm 0.11.2+corex.4.4.0` 的 V1 引擎完成真实生成：`float16`、`gpu_memory_utilization=0.96`、`max_model_len=512`，模型权重占用 27.52 GiB，预留约 1.21 GiB KV cache（7,888 tokens），生成测试通过。默认 0.90 显存比例无法为 14B 单卡模型预留 KV cache，已将 smoke 脚本默认比例改为 0.96，并保留环境变量覆盖入口。

两机模型级 P2P 探针已进一步验证启动层：本机 `10.31.10.62` 与远端
`10.31.10.210` 均以 Python 3.12/vLLM 0.11.2/CoreX 4.4.0、各一张 GPU 成功加载
同一 Qwen3-14B，并创建 `CoreXP2pNcclConnector`、监听各自 `19052` endpoint。
探针使用超过一个 KV block 的 241-token prompt、同步 `PUT` 传输模式以及同一个
携带 prefill/decode endpoint 的内部 request ID。为适配该 ID，CoreX connector
shim 修复了上游贪婪正则会把第二个 endpoint marker 误解析进 hostname 的问题。
截至本报告更新，vLLM 调度器仍未在该模型探针中产生 `ncclCommInitRank`、KV
send/recv 或 consumer 输出，因此这只证明模型与 connector 的双机启动兼容，**不
证明模型级 KV handoff 已完成**。底层独立 NCCL tensor P2P 成功与模型级 P/D 成功
继续分别记录，不可混同。

## 本机与驱动核验

| 项目 | 实测值 |
| --- | --- |
| 操作系统 | Ubuntu 20.04.6 LTS，内核 5.4.0-216-generic |
| CoreX SDK | `/usr/local/corex` -> `/usr/local/corex-4.4.0` |
| `release-corex.txt` | `Iluvatar CoreX SDK 4.4.0` |
| `ixsmi` 驱动版本 | 4.4.0 |
| `ixsmi` IX-ML | 4.4.0 |
| 设备 | 16 × Iluvatar BI-V150，32 GiB/卡 |
| PyTorch 运行时 | `torch 2.7.1+corex.4.4.0`，`torch.version.cuda == 10.2` |
| Ray 运行时 | `ray 2.52.1+corex.4.4.0` |
| vLLM 运行时 | `vllm 0.11.2+corex.4.4.0` |

驱动核验命令：

```bash
/usr/local/corex-4.4.0/bin/ixsmi
cat /usr/local/corex-4.4.0/release-corex.txt
```

验证过程中没有执行 `apt`、`dpkg`、CoreX 安装器、驱动卸载器、GPU reset 或任何 sudo 命令；`/usr/local/corex` 仍指向 `/usr/local/corex-4.4.0`。

## 环境隔离

项目专用环境为：

```text
/data1/congmng/llumnix/.conda-corex44
```

它是已有 CoreX 4.4.0 Python 环境的 clone，随后只在该副本中安装了同目录软件栈提供的 `ray-2.52.1+corex.4.4.0`。原环境 `/data1/congmng/conda-envs/ds-corex44`、系统 Python、驱动和 `/usr/local/corex-4.4.0` 都没有被修改。

当前 `setup.py` 已声明 Python `>=3.9,<3.13` 并包含 Python 3.12 classifier；因此 CoreX Python 3.12 环境可以按 wheel 方式安装。历史上曾有 `<3.11` 的限制，现已移除。软件栈也提供 Python 3.10 wheels；不过它们的 vLLM 版本同样是 0.11.2，因此不能解决下述 API 断层。

运行 smoke 测试：

```bash
cd /data1/congmng/llumnix
CUDA_VISIBLE_DEVICES=0 .conda-corex44/bin/python tools/corex44_smoke.py
```

`CUDA_VISIBLE_DEVICES=0` 限定测试只临时使用一张卡。脚本启动的本地 Ray 会在结束时执行 `ray.shutdown()`。

## 已通过验证

### 1. PyTorch + CoreX

实测结果：

```text
torch 2.7.1
torch.version.cuda = 10.2
torch.cuda.is_available() = True
torch.cuda.device_count() = 16
torch.arange(1024, device='cuda').sum() = 523776.0
```

测试识别到的设备名称为 `Iluvatar BI-V150`。

### 2. Ray GPU Actor

使用 `CUDA_VISIBLE_DEVICES=0` 启动本地 Ray，集群资源包含 `GPU: 1.0`。一个声明 `num_gpus=1` 的 Ray actor 内成功运行 CoreX 张量计算，返回：

```text
{'available': True, 'device': 'Iluvatar BI-V150', 'sum': 523776.0}
```

这验证了 Llumnix 使用的 Ray 任务资源标记可以把工作调度至 CoreX 可见设备。

### 3. Llumnix 调度控制面

已成功导入以下模块：

```text
llumnix
llumnix.global_scheduler.global_scheduler
llumnix.manager
llumnix.llumlet.llumlet
```

并执行 `GlobalScheduler` smoke 测试：创建 3 个 `no_constraints` 实例，赋予负载 `-10/-5/-1`，`load` 策略正确选择最低负载实例 `corex-i0`，并返回 `expected_steps == inf`。

另外，`python -m compileall -q llumnix` 返回 0。

## vLLM 数据面兼容性结果

### V1 attention hook 兼容性（2026-09-02）

CoreX 版 vLLM 0.11.2 将 V1 attention 的 KV transfer wrapper 放在
`VLLM_SUPPORT_IXSERVER` 条件分支下。默认值为 false 时，P2P scheduler metadata
仍会正常生成并绑定，但 attention forward 不会调用 connector 的
`start_load_kv`/`save_kv_layer` 生命周期，表现为 consumer 等待且 producer 无
NCCL 数据面日志。`CoreXP2pNcclConnector` 的进程内 shim 现启用该 wrapper，并
更新已经导入的 `vllm.envs` 缓存值；该修改仅影响显式选择 CoreX P2P connector
的 worker，不修改驱动、系统安装或共享库。19 项 V1 KV 传输单元测试通过。
本小节记录 native NCCL 排障时的阶段性结果；当前生产默认路径及其端到端证据
见下方 `zmq_cpu` 小节。

两机探针已分别证明模型加载、KV cache 分配、connector 初始化和 metadata 生成；
使用同一内部 request ID 的模型级 tensor handoff 仍在验证中，在完成前不将其
描述为完整 P/D 端到端成功。

### 模型级 communicator 诊断（2026-09-02）

使用相同的 P/D 内部 request ID 后，producer 已实际调用
`save_kv_layer`，并成功与远端建立 rank-0 NCCL communicator。远端 consumer
在 rank-1 `ncclCommInitRank` 期间出现 CoreX 原生
`double free or corruption (out)`，尚未进入 KV 注入或公开输出阶段。已将
`NCCL_CUMEM_ENABLE=0` 的进程内兼容处理前移到 CoreX connector 导入阶段，且
保留在 vLLM P2P context 周围；该 workaround 不修改系统驱动/共享库。当前
证据证明 scheduler、attention hook、跨主机握手和 producer 保存路径均工作，
但 CoreX communicator 的模型级 consumer 稳定性仍未通过验证。

### 两机模型级 P/D handoff：已验证（ZMQ CPU staging，2026-09-02）

为规避上述 CoreX NCCL rank-1 原生崩溃，`CoreXP2pNcclConnector` 新增
`corex_transport`。默认 `zmq_cpu` 使用 ZMQ 传输连续 CPU staging buffer，再由
consumer 放入 GPU paged KV cache；显式设为 `nccl` 才使用上游 NCCL 路径。该选择
保留了 V1 P2P connector 的 request ID、per-layer KV ownership、调度 metadata 与
阻塞 load 语义，而不修改驱动、CoreX 安装或共享库。

两机 Qwen3-14B 实测中，`10.31.10.62` producer 保存并发送 40 个 attention
layers，`10.31.10.210` consumer 接收全部 KV 后先输出 `重复的`，最终返回
`重复的句子`（`finished=True`）。这证明 Python 3.12/CoreX 栈上的 V1 P/D 模型级
KV handoff 已可运行。CPU staging 的代价是额外 host-memory copy；原生 NCCL 模式
仍保留为非默认诊断/优化路径，待 CoreX 修复 rank-1 communicator 的 native abort
后再启用。

staging engine 同时实现了 listener 线程、ROUTER/DEALER socket 和 ZMQ context
的幂等 shutdown，并由 CoreX connector 的 V1 shutdown 生命周期调用，保证实例
重启或故障恢复时不会遗留 KV 端口。新增回收测试与原有 V1 KV 测试合计 26 项通过。

接收端 GPU tensor 注入现在严格使用 worker 的 `local_rank`（而非默认
`cuda:0`），因此多卡 Ray placement group 中的 KV cache 不会跨卡写入；设备绑定
回归覆盖后，当前相关测试总计 43 项通过。

独立 V1 HTTP 入口 `v1_api_server` 现在通过 `V1EngineAdapter` 创建引擎，而不是
直接实例化 `AsyncLLM`，因此其 `/generate` 请求与 Manager/Llumlet 路径共享
request lifecycle、CoreX connector 和 request-ID 处理；多实例调度仍由主
`api_server`/Manager 入口负责。

staging transport 对来自对端的 payload 做异常隔离，并对发送确认和接收 KV
增加可配置超时（`zmq_recv_timeout_s`，默认 120 秒）。因此对端崩溃或网络异常会
返回明确的超时/协议错误，而不会静默终止 listener 或永久阻塞 EngineCore。
无对端发送和缺失 tensor 接收的故障路径均有测试覆盖；V1 KV transfer 定向测试
为 26 项通过，并覆盖坏 payload 后 listener 继续处理合法 tensor 的恢复路径。

Manager 的 V1 P/D 双请求启动路径在任一侧失败时，会同时向 producer 和 consumer
发送取消请求，并逐侧记录清理异常；因此不会留下等待 KV 的 consumer 或孤立的
producer 发送任务。相关回归测试保持 42 项通过。

请求 bookkeeping 也会在周期性轮询中按各 Llumlet 的活动 request-id 集合重建。
因此 producer/consumer 均完成后，公开 request-id 与双实例映射会同时移除；后续
abort 不会再次访问已经完成的 actor 请求。
该轮询现在每个周期都会重新查询 actor，而不是只在 Manager 启动时查询一次；因此
长时间运行的 V1 P/D 服务也能持续回收完成请求。纯 CPU 回归覆盖了一个公开 ID
同时位于 producer/decode 两端、随后两端完成的场景。

安装门禁也已验证：在专用 Python 3.12/CoreX 环境执行
`pip wheel . --no-deps --no-build-isolation` 成功，生成的 wheel 元数据包含
`Requires-Python: >=3.9.0, <3.13`、Python 3.12 classifier，以及 V1 依赖
`vllm>=0.11.2,<0.12`。完整历史测试集中的旧 block-manager 测试依赖 vLLM 0.6
已在 V1 环境收集阶段隔离；当前 V1/调度相关回归保持绿色。

测试基础设施为每次会话选择随机本机 Ray 控制端口、显式绑定 `127.0.0.1`，并让
`ray.init` 使用该地址，不再继承开发机或共享集群的 `RAY_ADDRESS`。这避免外部
raylet/dashboard 故障污染适配回归；V1 KV/affinity 定向测试在该夹具下通过。
同时修正 Ray head 启动命令，确保 `HEAD_NODE_IP` 对应的 `--node-ip-address`
实际传递给 `ray start --head`；入口测试以随机端口启动 head 成功。
进一步将 unit-test fixture 改为无 dashboard 的进程内 Ray runtime，并移除清理阶段
对 State API 的依赖；入口启动命令使用 subprocess mock 校验参数。这样测试不会
停止或查询用户正在运行的共享集群，也不会因 dashboard 不可用误报适配失败。
依赖旧 vLLM 0.6 API 的 legacy HTTP 测试也在 V1 收集阶段跳过；V1 HTTP 入口由
独立的 `v1_api_server` 实测覆盖。部分历史 Ray actor/placement 测试仍可能因其
长轮询语义而耗时，不作为当前 V1 生产门禁。

独立 `v1_api_server` 入口新增 `--max-num-seqs`（默认 4），避免 Qwen3-14B 在
32 GiB CoreX 卡上按默认 128 dummy requests 进行 sampler 预热时 OOM。使用本机
Qwen3-14B 实测启动后，`GET /health` 返回 200，`POST /generate` 返回 200 和
正常生成文本；该参数仍可由部署显式调高或调低。
该入口的 FastAPI 生命周期已改用 `lifespan`，在关闭时幂等释放 V1 adapter；新增
入口现复用 vLLM 0.11 `AsyncEngineArgs` 的完整 CLI 参数注册和校验，不再静默
丢弃量化、tokenizer、调度及 KV-transfer 选项；`--help` 已实测包含
`--quantization`、`--kv-transfer-config`、`--tensor-parallel-size` 等原生参数，
对应 CLI 回归通过。
此外，主 `api_server` 的本地初始化已修复为向 Manager 完整透传
`entrypoints_args`、`instance_args`、`engine_args` 和 `launch_args`；这保证本地
多实例部署也能执行 V1 检测、P/D 配置和 CoreX 参数选择。专用回归通过。
主 HTTP `/generate` 入口也已与独立 V1 入口对齐：非对象 JSON、缺失或错误类型
的 prompt/stream/request-id 及非法采样参数统一返回 HTTP 400；streaming 客户端
断连会调用 Manager abort，避免 V1 请求或 P/D 双请求残留。
无模型主 API 回归已覆盖非法请求和公开 request-id 透传；主入口相关回归通过。
GlobalScheduler 闭环回归进一步验证：在异构负载差距处于 0.10 安全窗口内时，
具有请求 prefix KV 的实例会被选中；当负载明显更差时仍不会覆盖健康度优先级。
这使 KV-cache affinity、实时显存/算力负载和跨实例请求分发形成完整决策链。
该闭环在两台机器的同一提交 `f1f636e` 上复核：本机和 `10.31.10.210` 均完成
48 项 V1/KV/入口/调度定向回归及 `compileall`，结果全部通过。
主 API `/instance_list` 现同步暴露 V1/CoreX 的总/空闲显存、算力容量、调度负载
和已缓存 KV block 数，旧 block 计数字段保持兼容；无模型接口回归已覆盖这些字段。
扩缩容策略已移除对旧版 `instance_load_dispatch_scale` 等不存在字段的依赖，直接
基于当前实例负载计算；V1 自动扩容在没有 block-manager 计数时按控制周期增补一个
实例，待下一轮采集真实 CoreX 显存/算力状态，避免空实例 `-inf` 导致扩容失效。
控制器级回归已用 96 个活动请求、4 GiB 空闲显存的 V1 实例验证
`GlobalScheduler.check_scale()` 返回 `scale_up=1`；扩缩容与 KV/API 核心定向回归
共 53 项通过（两机均已复核）。
`--scaling-load-metric` 现正式接受 `virtual_usage`，部署可直接选择基于 CoreX
实时显存、算力容量和请求压力的异构扩缩容指标；CLI 回归已覆盖该选项。
本机与 `10.31.10.210` 已在提交 `3127034` 上复核扩缩容、KV、API、入口定向集，
两端均为 55 passed 且 `compileall` 通过。
V1 开启 P/D 时若仍使用旧配置文件中的 `gloo`/`rayrpc`/`nccl`，入口现在会在
引擎启动前明确报错，要求 `kvtransfer` 及 connector endpoint；不会再静默启动
没有 KV handoff 的双普通引擎。
本机与 `10.31.10.210` 已在提交 `8c9a4a9` 上复核该门禁及 V1/KV/API/调度集，
两端均为 56 passed，且 `compileall` 通过。
V1 Llumlet 的 EngineCore/connector 流异常现在会显式调用 abort，再清理请求别名和
运行队列，避免 KV 超时、worker 失败时遗留 P/D 对端请求；无模型故障流回归已覆盖。
当 Manager 暂时不可用而请求已由 API 进程 fallback 到本地 Llumlet 时，客户端
`abort` 现在直接取消该实例请求并回收 fallback 负载计数；对应故障回归已覆盖。
该变更已在 `10.31.10.210` 的 Python 3.12/CoreX 环境复核：同步同一源码后
V1/KV/HTTP 定向回归为 45 passed，完整参数帮助可正常加载。
无模型 mock 回归覆盖 `/health`、非流式 `/generate` 和 streaming wire format，
因此 HTTP 协议可在不加载大模型的情况下持续验证。
流式响应的异步迭代器在客户端断连/提前关闭时也会自动 abort 对应 V1 request，
避免长生成或 P/D consumer 在 HTTP 客户端消失后继续占用 EngineCore；该路径已有
mock 回归覆盖。
lifespan 退出时调用 adapter shutdown 的路径也已有无模型回归，覆盖 HTTP 服务
退出后 EngineCore 与 connector listener 的资源回收契约。
独立入口直接调用 adapter 的路径也会记录公开 request-id 到 connector 内部
request-id 的 alias；因此启用 P2P 后，断连或显式 abort 能准确取消真实 EngineCore
请求，而不会只删除 HTTP 层 bookkeeping。
正常完成的非流式/流式请求也会释放 adapter 的本地 request/alias/running 状态；
staging 测试端口改为动态分配，重复运行或双机并行验证不会因固定端口残留而误报。
环境审计显示 CoreX 发行环境可能未安装 `scipy`/`pandas`；profiling 模块现将这
两项仅用于模拟器拟合和 CSV 分析的依赖延迟加载。核心 V1 serving、KV affinity
和 P/D handoff 不再因导入 profiling 失败；真正使用分析功能时会给出明确安装提示。
独立 V1 HTTP 入口现在会把缺失或类型错误的 prompt、stream、request-id 及采样
参数转为 HTTP 400，且不会启动 EngineCore request；因此部署端可区分客户端输入
错误和 CoreX/vLLM 运行时故障。
非对象 JSON body 也按同一规则拒绝，避免 Python 3.12 下对列表等输入调用
`.pop("prompt")` 产生未处理的属性异常。

以下是历史兼容性记录（不是当前 CoreX 安装要求）：仓库早期版本曾锁定 vLLM
0.6.3：

```text
历史版本 requirements/requirements_vllm.txt: vllm == 0.6.3.post1
当前版本 requirements/requirements_vllm.txt: vllm >= 0.11.2, < 0.12
```

CoreX 4.4.0 软件栈中提供的是 vLLM 0.11.2。导入旧后端的实测失败如下：

| Llumnix 模块 | 实测失败 | 含义 |
| --- | --- | --- |
| `llumnix.backends.vllm.llm_engine` | `ImportError: _AsyncLLMEngine` | 0.11.2 不再导出旧异步 Engine 私有类 |
| `llumnix.backends.vllm.executor` | `ModuleNotFoundError: vllm.executor` | V1 重组/移除了旧 executor 包路径 |
| `llumnix.entrypoints.vllm.api_server` | `ModuleNotFoundError: vllm.model_executor.layers.sampler` | 旧 sampler 路径已不存在 |

受影响的代码不仅是导入名，还包括从旧 scheduler、block manager、worker、sequence 和 engine 生命周期继承并覆写的实现。因此，不应将这几个 import 用空兼容层屏蔽；那样会让 API Server 启动却无法保证推理正确性或 KV Cache 迁移安全性。

## 尚未纳入当前 V1 兼容范围的旧接口

以下旧接口仍保留在源码中供历史 vLLM 使用，不作为当前 Python 3.12/CoreX V1
部署要求：

- `llumnix/backends/vllm/{llm_engine,executor,worker,scheduler,sequence}.py` 的
  旧 block-manager 实现仅在 vLLM 0.6 环境选择。
- V1 的安全迁移语义由 connector 驱动的 P/D KV handoff 提供，不模拟旧接口的
  任意时刻 block-manager request migration。

说明：论文第三点中的跨域请求迁移，在当前 V1 栈由 connector 驱动的 P/D KV
handoff 实现；旧 vLLM 0.6 的任意时刻 block-manager 迁移接口仍不兼容，不能将
二者混称为同一 API。
为支持真实跨机控制面，已将两端 Ray 统一到 Python 3.12/CoreX 的
`2.52.1+corex.4.4.0`，并验证 Ray head/worker 汇聚为 2 节点、2 GPU；远端已
通过内网续传获得完整 Qwen3-14B 权重（8 个分片索引一致）。同时新增
`--no-launch-ray-cluster`，解决配置默认自动 `ray stop` 与已有 head 冲突的问题。
该参数是跨机部署的必要安全开关，使用时 API 仅连接集群、不重启 Ray。

注：在一次真实跨机启动尝试中，Ray head 因默认根分区临时目录接近满载而退出，
远端 worker 随后无法连接 GCS；该现象发生在 Llumnix/模型 actor 创建之前，不构成
V1 调度代码失败证据。后续跨机部署应将 Ray head 的 `--temp-dir` 和 object-store
目录固定到 `/data1`，并用轻量 Ray actor 心跳验证集群稳定后再加载模型。

进一步的诊断将 Ray head 临时目录固定到 `/data1`、每节点限制为 4 CPU/1 GPU；
head 单节点可稳定运行，但远端 worker 注册时仍观察到 head/GCS 退出，远端仅报
GCS 连接超时，head 日志没有 Llumnix 或模型错误。该现象属于跨节点 Ray 运行时
握手层，当前已清理进程；模型同步、版本统一和 affinity 哈希验证不受影响。

为将该部署要求纳入正式适配，`setup_ray_cluster` 已支持
`LLUMNIX_RAY_NUM_CPUS`、`LLUMNIX_RAY_OBJECT_STORE_MEMORY` 与
`LLUMNIX_RAY_TEMP_DIR`。这些变量只影响 head 的 `ray start` 参数，worker 接入
已有集群时不会错误地传递 head-only `--temp-dir`。跨机端口探针进一步表明需在
网络层允许两节点间 Ray GCS、node-manager、object-manager 以及配置的 worker
端口范围；在该条件满足前，不应将 Ray GCS 超时归因于 Llumnix V1 适配。
## Ray worker 的 V1 connector 环境传递（2026-09-03）

Ray Llumlet 是由 raylet 创建的独立 actor，不能依赖启动 `serve` 命令的 shell
环境自动继承每个 V1 EngineCore 的 KV connector 参数。`Llumlet.from_args` 现在
为 vLLM 0.11 V1 actor 显式传递受限的 `runtime_env.env_vars`，覆盖 KV IP、端口、
角色/rank、并行规模、事件 endpoint 和 `PYTHONHASHSEED`；不会转发完整环境或令牌。
这样 `LLUMNIX_KV_PORT` 等部署覆盖项在 worker 内实际生效，也避免同机多实例使用
默认端口造成冲突。新增单元测试验证 allow-list 和空值过滤。

## 固定规模部署的 placement-group 生命周期（2026-09-03）

发现全局部署在 `enable_scaling=False` 时仍启动自动扩容协程；该协程每个控制
周期创建新的 GPU placement group，固定规模的两实例 P/D 服务因无空闲 GPU 而
不断累积 pending group。现在自动扩容协程仅在显式启用 `enable_scaling` 时创建，
固定规模部署不会产生额外资源请求。清理现场 237 个残留 placement group 后，
Ray 双机资源回到 2 GPU 空闲、无 pending resource demand；Manager 定向回归及
V1 connector 回归通过。

## Ray State API 缺失时的全局部署降级（2026-09-03）

## 全局固定实例显式初始化与 Ray actor 兼容（2026-09-03）

关闭固定部署中的自动扩容循环后，进一步发现全局入口过去实际上依赖该循环来
创建初始实例，导致 `enable_scaling=False` 时只启动 Manager 而没有 Llumlet。现已
增加幂等的 `Manager.init_global_instances`，由全局 setup 显式创建配置数量的
placement group、API actor 和 Llumlet；自动扩容仍只负责后续弹性副本。

在 CoreX Ray 2.52 上验证 runtime-env API 时发现 actor 选项必须在同一次
`ActorClass.options(...)` 调用中提交；当前实现将调度策略与受限 connector
runtime-env 合并后一次提交，避免 raylet 内部兼容错误。Python 3.12 编译和
V1/KV/affinity 回归已通过，真实双机重启需在清理旧 detached actor 后继续验收。

CoreX 4.4.0 环境中的精简 Ray wheel 可能不包含 dashboard HTTP 服务。此前
`Manager._auto_scale_up_loop` 在 placement-group 状态查询失败时会进入异常重试，
使全局实例扩缩容无法继续。现已捕获 `ray.util.state.exception.ServerUnavailable`，
同时覆盖超时 placement-group 恢复查询和常规状态查询：首次失败后关闭状态查询，保留
Ray 控制面上的 placement-group 创建、actor 健康检查和实例注册流程。安装了
`ray[default]` 的环境仍使用原有状态回收逻辑；该降级不改变调度算法，只移除对可选
dashboard 的硬依赖。

## P/D endpoint 时序修复与两机复验边界（2026-09-03）

P/D producer 的 decode endpoint 只有在 prefill/decode 两个 Llumlet 完成初始化并
上报 `InstanceInfo.kv_endpoint` 后才能由 Manager 注入共享 request ID。此前初始化
阶段强制要求 `LLUMNIX_KV_DECODE_ADDRESS`，会在 endpoint 尚未发现时错误终止 actor；
现改为延迟到请求编排阶段校验，仍会拒绝显式非法 endpoint。两机 Ray 控制面已重新
稳定汇聚，但 Qwen3-14B 双实例启动仍出现 vLLM EngineCore 初始化失败，日志未给出
可归因于 connector 的底层错误；因此本次不能把该轮失败宣称为 handoff 失败，模型级
handoff 的正式证据仍以先前 `zmq_cpu` staging 验证为准。

## 多卡 TP=2 真实端到端复验（2026-09-03）

在本机 CoreX 4.4 / Python 3.12 / vLLM 0.11.2 环境运行：

```bash
source tools/corex44_env.sh
python tools/run_corex44_validation.py e2e --tp 2
```

测试使用 `CUDA_VISIBLE_DEVICES=0,1` 和 `TENSOR_PARALLEL_SIZE=2` 启动
Qwen3-14B。vLLM V1 EngineCore 建立 NCCL `world_size=2`，TP rank 0/1 均完成
权重加载和 KV cache 初始化；中文提示生成返回非空文本，耗时 26.21 秒，输出
`qwen3_14b_corex_vllm: PASS`。这验证了多卡 TP 基础推理链路，未宣称论文中的
吞吐或延迟收益已复现。

## V1 KV-event 跨机亲和集成门禁（2026-09-04）

新增 `tools/corex44_kv_event_probe.py` 并纳入 integration runner。发布端调用
vLLM 0.11 原生 `EventPublisherFactory` 发送 `BlockStored` 事件，远端通过
`KVEventSubscriber` 解码并写入 `KVCacheAffinityIndex`；验证完整 prefix 的
affinity=`1.0` 且远端 cached candidate 在调度排序中优先。由于部分部署节点只开放
SSH 而拦截任意 ZMQ TCP 端口，runner 对该控制面使用 SSH reverse tunnel；数据仍是
真实 vLLM ZMQ/msgspec 事件，不使用离线模拟替代。

2026-09-04 当前执行容器的网络沙箱禁止 socket bind/SSH（`Operation not
permitted`），因此本轮只能完成代码编译和版本/source gate；待网络权限恢复后需
重跑 `python tools/run_corex44_validation.py integration`，再将远端 consumer
输出作为最终双机事件证据。

网络恢复后已完成该复验：两端版本/source fingerprint gate 通过；真实 vLLM
ZMQ `BlockStored` 事件跨机传递 16 次，远端 consumer 重建索引并报告
`affinity=1.0`、`rank=[remote-cached, local-empty]`。同一 integration 随后完成
跨机 BF16 GPU staging，producer/consumer 均在 `cuda:0` 正常收发。统一 unit runner
为 `68 passed`；TP=2 Qwen3-14B E2E 再次建立 NCCL `world_size=2` 并完成中文生成，
耗时 25.54 秒，`qwen3_14b_corex_vllm: PASS`。这些是功能正确性证据，不代表论文
吞吐/延迟收益已复现。

## P/D 的 KV-cache-aware 选址（2026-09-04）

为防止双机某一端仍运行旧 P/D 选址/abort 编排代码，CoreX support gate 的源码
指纹范围已纳入 `llumnix/manager.py`。实测门禁先正确拒绝不同 fingerprint，archive
同步后两端 `source_fingerprint=35b8b868...018868a78`、`supported=true`。因此后续
双机 KV-event/P/D 集成在启动前能够确认 Manager 逻辑一致，而不仅验证 connector
和 affinity primitives。

补充修复 Decode-only 实例的调度记账：Decode 角色不加入普通 dispatch 集合时，首次
通过 `dispatch_candidates` 选中可能不存在计数键；现在受限角色池选择使用默认零值
幂等累加，确保 P/D 负载统计稳定。dispatch scheduler 回归 `11 passed`，统一
CoreX V1 unit runner 仍为 `68 passed`。

此前 V1 P/D 编排会在 Prefill 和 Decode 角色池中分别选择最小负载实例，而普通
请求才使用 KV affinity。这会使已从双机 `BlockStored` 事件得到的缓存信息没有参与
P/D 请求的实际选址。现在 `DispatchScheduler.dispatch_candidates` 提供受限候选池的
统一选择接口；Manager 对 Prefill、Decode 各调用一次。策略保持原有安全性：仅在
距离最小 dispatch load 不超过 0.10 的实例之间，以完整 prefix 命中优先并按
instance id 确定性排序，绝不会为了缓存命中转发给明显过载的实例。

新增定向 P/D role-pool affinity 单测；dispatch scheduler 10 项和 CoreX V1 unit
runner 68 项均通过。该改动将已验证的双机 KV-event/affinity 算法真正接入
connector-driven P/D handoff 的请求编排路径；它不伪装为 vLLM 0.6 block-manager
任意时刻迁移。
## 2026-09-05：ZMQ readiness 启动协议加固

在 CoreX 4.4 的 Ray actor 调度下，actor 创建完成与其异步 ZMQ loop
bind socket 之间存在短暂窗口。旧客户端只发送一次 `IS_SERVER_READY` RPC，
会将该正常窗口误判为 RPC server 启动失败。当前实现增加有界 readiness 重试
（30 秒总窗口、100ms 间隔），仅对超时重试，最终仍保留失败传播；因此不会掩盖
真正的 server 异常。动态端口回归也消除了并行测试进程的固定端口冲突。

验证：`tests/unit_test/queue/test_zmq.py` 全部通过（4 档 qps 基准加 readiness
单测，共 `5 passed`）；CoreX V1 分层门禁 `89 passed`。该修复属于 Python 3.12 /
CoreX 运行时适配，不改变 KV affinity、P/D handoff 或扩缩容策略。

修复提交 `0444ef7` 已同步至两机后重新执行严格 support gate、双机 integration 与
本机 TP=2 E2E。前两者证实双方运行 Python `3.12.13` / CoreX `4.4.0` / PyTorch
`2.7.1` / vLLM `0.11.2` / Ray `2.52.1` 且 V1 源码指纹一致；integration 使用真实
`BlockStored` 事件产生远端 `affinity=1.0`、`remote-cached` 优先排序，并成功完成
BF16 `cuda:0` ZMQ CPU-staging round trip。Qwen3-14B TP=2 在 25.36 秒内返回非空中文。
这些结果确认交付功能未因队列修复回退；不等同于论文的 QPS 或延迟收益复现。

## 2026-09-05：历史 E2E 入口的 V1 CLI 迁移

全量测试审计还发现 `tests/e2e_test/utils.py` 沿用了 vLLM 0.6 的
`--worker-use-ray`。该参数在 vLLM 0.11 V1 中已删除，因而生成的服务命令在
argument parsing 阶段失败，尚未进入 Llumnix。命令生成器现按运行时 vLLM 版本
处理：0.11 V1 不输出该参数，旧 vLLM 保持兼容输出；V1/legacy 双向回归已加入
CoreX unit gate，门禁结果 `91 passed`。

该修复使传统 E2E 启动入口可以进入 V1 服务链路，但不把历史 block-manager
migration workload 误称为 V1 P/D 验证；后者的正式覆盖仍是 connector-driven
handoff、双机 KV event/affinity 和 BF16 staging integration。

## 2026-09-05：完整 unit 层复验

在 Python `3.12.13` / CoreX 4.4 环境运行全部 `tests/unit_test`，结果为
**`152 passed, 10 skipped`**（373.38 秒）。这覆盖精简交付门禁之外的 Manager、
placement、队列、入口与 legacy 边界行为。跳过项是测试显式声明的可选依赖或硬件
条件，不用于替代已运行的双机 integration 或 Qwen3 TP=2 E2E；三层证据保持独立。
逐项审计显示 9 项跳过来自 Manager legacy real-engine/scale 测试所需的 1/2/4
张 Ray GPU 注册，另 1 项来自 engine-step 测试所需的 PyTorch/Ray GPU；当前隔离
unit runner 设计为 0 GPU。GPU 相关行为已由 TP=2 实机 E2E 和双机 GPU staging
单独覆盖。
