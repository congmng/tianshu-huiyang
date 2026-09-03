# Llumnix CoreX 4.4.0 适配进度

## 2026-09-03：Llumnix V1 TP=2 本机双卡基础推理通过

- 在不影响现有两机 `10.31.10.62:6408` P/D 集群的隔离 Ray head
  `10.31.10.62:6410` 上，以两张 BI-V150（`CUDA_VISIBLE_DEVICES=0,1`）启动
  Llumnix global serve 和 Qwen3-14B。
- 修复 V1 的 placement 语义：AsyncLLM 在 Llumlet 父 actor 内启动，TP GPU 必须
  打包在同一 bundle；同时 TP>1 使用 vLLM `mp` executor，避免已连接 Ray 集群
  内部再次申请 Ray GPU 资源。日志确认 `world_size=2`、TP rank 0/1、8 个权重
  分片加载、每卡约 13.88 GiB 权重及 193,760-token KV cache。
- 实测 `GET http://10.31.10.62:37112/health` 返回 200，`POST /generate` 返回
  200 和非空中文生成文本。该结果证明当前 Python 3.12/CoreX/V1/Llumnix 基础
  多卡服务链路可运行；不将其扩大解释为跨实例 P/D handoff 验收。
- 新增 V1 TP placement 回归测试，定向集为 **3 passed**。Ray State API 在 CoreX
  minimal dashboard 不可用时增加 control-plane 回退，避免全局监控线程误报。

## 2026-09-03：TP=2 placement 规划兼容修复与双卡实测

- 在两机 Ray 集群上完成真实双卡 CoreX 探针：任务分别落到 `u62`、`u210` 的
  BI-V150，每卡 CUDA 张量求和结果均为 `8386560.0`，证明 Python 3.12/Ray
  GPU 调度和两节点设备运行正常。
- 发现 vLLM V1 TP=2 启动在 CPU Manager actor 中提前失败：Manager 调用
  `create_engine_config()` 时自身无 GPU，vLLM 将可见 GPU 数判为 0，即使 Ray
  placement group 可提供两张卡也会拒绝 TP=2。`get_engine_world_size` 已改为仅由
  TP×PP 计算 placement 所需 world size，并新增回归测试。
- 本次重试命中了此前残留的 detached Manager（未加载最新代码），因此尚未把
  TP=2 模型级 HTTP 结果记为通过；后续实测必须先清理旧 Manager/placement group，
  确保 Ray actor 使用提交 `569a72a`。

## 2026-09-03：Qwen3-14B 本机 TP=2 V1 真实推理通过

- 在两张本机 BI-V150（`CUDA_VISIBLE_DEVICES=0,1`）上直接启动 vLLM 0.11.2 V1
  `tensor_parallel_size=2`，日志确认 `world_size=2`、TP rank 0/1、NCCL 初始化，
  8 个权重分片加载完成；每卡模型驻留约 13.88 GiB，并成功创建 193,424-token
  KV cache。
- `GET /health` 与 `/v1/completions` 均成功；提示“用一句话介绍 CoreX”返回非空
  中文文本（8 tokens）。这证明 CoreX/PyTorch/NCCL/Triton 和 Qwen3-14B 的真实
  多卡推理链路可用。首次失败原因已确定为 Triton 编译环境未找到 `-lcuda`，而非
  显存；通过环境脚本提供的 CoreX library path 后重试成功。
- 本次使用原生 vLLM API，不将其误记为 Llumnix Manager/P-D handoff 验收；Llumnix
  多卡入口仍需在清理旧 detached Manager 后使用新 world-size 逻辑复验。

## 2026-09-03：Llumnix TP 拓扑门禁

- 在干净的两机 Ray 集群对 Llumnix TP=2 入口实测，最新代码正确将 placement 请求
  计算为 `CPU=2,GPU=2`；当前集群每台主机仅向 Ray 登记 1 张 GPU，无法满足
  Llumnix 的同节点 `STRICT_PACK` 语义。该语义保证 TP worker 本地通信，不会将
  一个 TP=2 实例错误地拆到两台服务器。
- `initialize_placement_group` 现于创建前检查每个 live Ray 节点的 GPU 容量。若
  aggregate GPU 足够而单节点不足，立即报出“TP GPU 必须位于同一节点”的可操作错误，
  不再遗留 1,800 秒 pending placement group；对应回归通过。
- 清理本轮 Manager 和 placement group 后，生产两机集群恢复 `0/2 GPU` 使用、无
  pending demand。部署 TP=2 时需让一个 Ray 节点以 `--num-gpus 2` 或更多启动；
  两机各 1 卡的跨域部署应使用两个 TP=1 Llumnix 实例，由 HLA/KV affinity/P-D
  策略进行请求协同。

## 2026-09-03：远端最新代码复验

- 将当前 `c3a0c8c` 通过 Git bundle 同步到 `10.31.10.210`，远端工作副本已从
  `2176773` fast-forward 到相同提交；模型目录和系统 CoreX 安装均未改动。
- 远端 Python 3.12/CoreX 环境的 V1 KV-transfer、KV affinity、TP placement
  规划回归共 **41 passed**，`compileall` 通过。
- 两端分别运行跨机 affinity 探针，得到相同 hashes
  `c9d58ba6...ef071cb`、`24125b23...54ab2d6` 与相同排序
  `candidate-b,candidate-a`，再次证明最新代码在两台正式运行时具有确定性。

## 2026-09-03：Python 3.12 全量回归与跨主机 affinity 复核

- 最新提交以隔离 Ray fixture 重跑 `tests/unit_test`：排除唯一显式启动 vLLM 0.6
  legacy `api_server.py` 生命周期的 `test_api_server.py` 后，结果为
  **119 passed, 19 skipped**。当前 V1 `/generate` 和两机 P/D 实机链路覆盖该
  legacy 测试不再代表的生产 API；其余 skip 为无 GPU/async 插件或旧
  block-manager 专用路径。

- 全量 Python 3.12/CoreX 4.4.0 测试首轮暴露 3 个 vLLM V1 兼容点：旧测试仍向
  `EngineArgs` 传入已删除的 `worker_use_ray`，以及 Ray 在 Llumlet placement
  group 预留失败时序列化对象 repr 会再次访问尚未初始化的 `instance_id`。
- 已移除测试中的旧参数；`Llumlet.__repr__` 现在安全处理半初始化对象，并将需
  GPU 的旧引擎测试在 Ray 未登记 GPU 时明确 skip。全量回归结果为 **115 passed,
  42 skipped**（784 warnings，均为现有 pytest/Ray 弃用或异步标记提示）。
- 本机与 `10.31.10.210` 均运行 `tools/cross_host_kv_affinity_probe.py`，在各自
  Python 3.12/CoreX 环境得到相同 `sha256_cbor` hashes
  (`c9d58ba6...ef071cb`、`24125b23...54ab2d6`)，affinity 排序均为
  `candidate-b,candidate-a`。该项再次确认论文第五章第三点的异构负载/缓存亲和
  决策跨主机确定性；不替代 GPU KV tensor handoff 验收。
- 本轮真实 14B 单卡服务启动因 CoreX 驱动显存不足未完成 HTTP 验证；该失败发生
  在模型/EngineCore 显存初始化阶段，与下载源、梯子或 affinity 算法无关。此前
  已有的双实例及 ZMQ CPU-staging P/D 证据仍按报告中的范围保留。

## 2026-09-03：Ray State API 缺失的全局扩缩容降级

- 现场验证 CoreX 精简 Ray 集群没有 `127.0.0.1:8265` dashboard API；Manager 原有
  超时 placement-group 查询位于状态查询保护范围之外，会导致扩缩容线程持续异常。
- 现在 `ServerUnavailable` 会关闭可选 State API 查询，但继续创建 placement group、
  启动 Llumlet 并依赖 actor 健康回调完成注册；常规 Ray[default] 环境行为不变。
- Python 3.12 编译及调度/KV/API 定向回归已通过；下一步继续在干净两机集群复验 P/D
  实例和 KV handoff。

## 2026-09-03：P/D endpoint 延迟校验

- 干净两机 Ray 集群重新建立并验证为 2 节点、8 CPU、2 GPU；State API 缺失不再使
  Manager 扩缩容线程报错。
- 发现 producer 在 Llumlet 初始化阶段要求 `LLUMNIX_KV_DECODE_ADDRESS`，与 Manager
  运行后发现 peer endpoint 的实际时序冲突；现改为初始化允许缺省，请求编排时由
  共享 P/D request ID 携带并严格校验 host:port。
- 该修复后的双机启动仍遇 EngineCore 初始化失败，当前证据不足以归因于 KV connector；
  本轮不更新“两机 handoff 已通过”的正式结论。V1/KV/调度回归为 **47 passed**。

## 2026-09-03：全局 P/D 启动连接修复

- P/D 首次启动发现 `llumnix.entrypoints.vllm.serve` 在全局模式下未指定地址调用
  `ray.init()`，会额外创建本机 Ray runtime，使 `State API` 报 multiple active
  Ray instances，阻断实例编排。这是入口连接问题，不是 connector 数据面错误。
- `serve` 现显式使用 `HEAD_NODE_IP` 和 `--ray-cluster-port` 连接既有 head，行为
  与主 API 入口一致；新增 AST 回归测试避免退化。后续 P/D 启动必须同时使用
  `--no-launch-ray-cluster`，防止默认配置重建集群。

## 2026-09-02：V1 P/D connector 默认地址修复

- 修复 `configure_v1_kv_transfer` 在跨机部署中的默认地址：此前无
  `LLUMNIX_KV_IP` 时写入 `127.0.0.1`，producer 生成的 endpoint 对远端
  consumer 不可达；现在自动解析当前 Ray actor 所在主机 IP，显式环境变量仍
  可覆盖多网卡场景。
- 新增默认 routable IP 单测，V1 KV-transfer 定向回归为 **29 passed**。
  该修复完成 P/D connector 地址编排前置条件；真实 GPU tensor handoff 仍需
  使用 `migration_backend=kvtransfer` 的 prefill/decode 实例专项验收。

## 2026-09-02：双机 Llumnix 主服务验证

- 纠正前一轮的 Ray 生命周期判断：以持久 `ray start --block` 会话保存 head，并在
  两端先清理遗留 Ray 会话后，`10.31.10.62:6403` 成功稳定汇聚为 8 CPU/2 GPU；
  node-affinity actor 分别返回 `u62`、`u210`。此前 head 消失与命令执行器回收
  非持久后台会话有关，不能归因于 CoreX 跨节点 Ray runtime。
- 在该集群启动 `initial_instances=2` 的 Llumnix V1 主 API：两个 Qwen3-14B
  TP=1 Llumlet 分别在本机和远端完成 27.52 GiB 权重加载、2.65 GiB KV cache
  分配并注册到同一 Manager。`/health` 返回 200，`/instance_list` 返回两个
  32 GiB 实例，`/generate` 成功产生中文文本；两个并发请求均完成。
- 内网地址经环境 HTTP 代理会返回 502；使用 `curl --noproxy '*'` 可直连 API，
  此为客户端代理设置而非 Llumnix 服务错误。
- 实例状态新增 `node_id`/`node_ip`，并由 `/instance_list` 透出，提供跨域
  实例拓扑的正式可观测字段；对应 V1 API/KV affinity 回归为 **24 passed**。

## 2026-09-02：跨机 Ray head 生命周期复核

- 以两端相同的 Ray 2.52.1/CoreX Python 3.12 环境重新建立 head：本机
  `10.31.10.62:6396`、2 CPU/1 GPU、1 GiB object store、session/spill 均落在
  `/data1/congmng/llumnix/.ray-live3`。启动后 `gcs_server` 与 `raylet` 均存活，
  GCS 监听 `*:6396`，因此不能把此前问题归因于高端口防火墙。
- 远端 `10.31.10.210` 以匹配版本和显式 node IP 加入时，先报告 GCS 连接失败；
  随后本机 head 进程消失、远端 TCP 连接变为 `Connection refused`。head 的
  `gcs_server.err`、`raylet.err` 未出现 Llumnix、模型、GPU 或网络拒绝错误。
- 该复现把完整跨机 Manager/Llumlet 部署阻塞定位为 Ray/CoreX 跨节点 runtime 的
  head 生命周期。已经完成的双机 hash/event/affinity 算法验证、单机 V1 serving
  和 ZMQ CPU-staging handoff 不受影响；在 runtime 稳定前不将其表述为跨机
  Llumnix API 验收通过。

## 2026-09-02：双机 affinity 可复现探针

- 新增 `tools/cross_host_kv_affinity_probe.py`，离线复现两台 Python 3.12/CoreX
  环境共同使用 `sha256_cbor` 计算 prefix block hash、导入 BlockStored 事件并
  按 affinity 排序的完整算法链路。探针不下载模型、不启动 Ray、不修改驱动。
- 本机与 `10.31.10.210` 输出完全一致：两个 hash 为
  `c9d58ba6...ef071cb`、`24125b23...54ab2d6`，候选 affinity 分别为 0.5/1.0，
  排序均为 `candidate-b,candidate-a`。
- 该证据进一步确认论文第三点的跨域请求下发决策在两台正式运行时可复现；GPU
  tensor 迁移和完整跨机 Llumnix 多实例服务仍需独立验收。

## 2026-09-02：论文第五章第三点基础功能复核

- 对照论文第五章“异构负载感知的跨域请求调度”（计算/显存统一虚拟负载、
  多实例请求分发及后续迁移），在本机 CoreX 4.4.0 + vLLM 0.11.2 V1 环境
  执行最小服务冒烟。使用本地 Qwen3-14B、单卡 TP=1、显存比例 0.96、最大
  序列 128 和并发序列 2，V1 API 成功启动。
- `GET /health` 返回 HTTP 200；`/generate` 中文请求返回非空结果，验证模型
  加载、V1 EngineCore、请求队列和 HTTP 输出桥接基础链路。
- dispatch、GlobalScheduler、KV affinity 定向回归为 **22 passed**，覆盖虚拟
  负载排序和 prefix affinity 决策；单实例冒烟不代表跨域迁移性能已完成。
- 一次并行通信初始化 RuntimeError 在清理残留进程、释放 GPU 后重试成功；该
  现象与网络防火墙无关。跨机部署仍需单独验证 Ray head 生命周期和多实例资源隔离。

> 本文件按时间保留阶段性结论；较早条目中的“尚未验证/未完成”仅反映当时
> 状态。当前状态以 2026-09-02 的“两机模型级 P/D KV handoff 验证通过”及
> 其后续条目为准。

## 2026-09-02：论文第三点本机双实例部署与可观测性修复

- 使用本地完整 Qwen3-14B 在两张 BI-V150 上以 `TP=1` 启动两个独立 V1
  Llumlet（Ray placement group，各占一张 32 GiB 卡），主 API 的
  `initial_instances=2` 成功注册两个实例；每个实例实际加载约 27.52 GiB 权重，
  `gpu_memory_utilization=0.96` 时均保留约 2.65 GiB KV cache（17,344 tokens）。
- 主 API 的 `GET /health`、`GET /instance_list`、`POST /generate` 均返回 HTTP
  200。`/instance_list` 正确显示两实例的 32 GiB 总显存、实时空闲显存、算力容量
  和 KV-affinity block 数；空闲实例内部以 `-inf` 表示最小调度负载，现安全映射为
  JSON `null`，避免 FastAPI 因非有限浮点数返回 500，同时不改变调度比较语义。
- 添加 API 回归覆盖该 idle-sentinel 序列化场景；`test_v1_api_server.py` 与
  `test_v1_kv_affinity.py` 定向回归为 **24 passed**。这验证了论文第三点中
  “异构状态采集—统一虚拟负载—多实例分发”的实际部署入口；本机同构双卡不用于
  宣称跨域迁移性能，跨机 P/D KV handoff 仍以此前已验证的 ZMQ CPU staging 路径为准。
- 实测也确认 Qwen3-14B 单卡实例不能使用低于模型驻留需求的
  `gpu_memory_utilization=0.45`：V1 会报告负的 KV cache 可用内存并拒绝启动。
  这是显存预算约束，部署示例应为单卡实例保留至少模型权重所需预算。

## 2026-09-02：远端 CoreX 栈同步与跨机 affinity 复核

- 通过 Git bundle 将提交 `1ba0dea` 同步到 `10.31.10.210`；远端 Python
  `3.12.13`、CoreX/Driver `4.4.0`、PyTorch `2.7.1`、vLLM `0.11.2`、Ray
  `2.58.0` 环境均可导入，`compileall -q llumnix` 通过，核心 V1/API/KV 定向
  回归全部通过（远端使用其现有环境执行）。
- 远端项目目录当前没有完整 Qwen3-14B 权重，因此未重复传输约 29 GiB 模型；
  使用两端相同的 `KVCacheAffinityIndex.prefix_hashes` 输入验证
  `sha256_cbor` 结果完全一致：
  `c9d58ba6...ef071cb`、`24125b23...54ab2d6`。这证明跨主机 affinity 哈希
  算法在 Python 3.12/CoreX 两套环境中保持一致；模型级跨域请求仍复用此前已
  完成的双机 P/D handoff 证据。

## 2026-09-02：跨机 Ray 接入参数与模型准备

- 发现直接把两机加入同一 Ray 集群时，Ray 2.52.1 与远端原有 Ray 2.58.0
  会被 GCS 拒绝；远端已切换到项目提供的 Python 3.12/CoreX Ray
  `2.52.1+corex.4.4.0` wheel，随后两机成功显示为同一集群的 2 个节点、2 个
  GPU。两端版本一致性现已满足。
- 将本地完整 Qwen3-14B 权重通过内网 `rsync --partial` 同步至远端
  `.models/Qwen3-14B`（约 30.1 GiB，8 个 safetensors 分片，索引校验一致，
  远端缺失分片为 0）。该模型复制不涉及外网下载或系统安装。
- 发现配置文件默认 `LAUNCH_RAY_CLUSTER=True` 会在接入已有集群时执行
  `ray stop`。新增 `--no-launch-ray-cluster` 显式开关，允许 API 进程安全连接
  已存在的跨机 head；新增入口回归通过。跨机实测仍需以该开关启动，避免误杀 head。

- 随后进行真实跨机启动时，Ray head 在 API 接入前后因 GCS 会话不稳定退出，远端
  worker 进入等待 GCS 状态；该次未启动模型、未产生 GPU actor，也未把失败误记为
  Llumnix 调度故障。现场核验确认远端 `/tmp` 根分区仅约 5 GiB 可用（约 99%），而
  `/data1` 仍有充足空间；后续应固定 Ray head 的 `/data1` 临时目录并先完成轻量
  actor 心跳，再启动大模型。

- 本轮将 head 临时目录固定到 `/data1/congmng/llumnix/.ray-stable`，并将两端
  Ray 限制为每节点 4 CPU、1 GPU；head 单节点可稳定启动并报告 1 GPU。远端
  worker 注册阶段仍复现 head/GCS 退出，worker 日志持续为 “Failed to connect to
  GCS”，且 head 日志没有 Llumnix、模型或 connector 错误。该结果将跨机 Ray
  控制面问题隔离在运行时/网络握手层；本轮已清理所有 Ray 进程，避免遗留资源。

## 2026-09-02：CoreX Ray 启动资源参数化

- `setup_ray_cluster` 现支持进程级环境变量 `LLUMNIX_RAY_NUM_CPUS`、
  `LLUMNIX_RAY_OBJECT_STORE_MEMORY`、`LLUMNIX_RAY_TEMP_DIR`，仅在创建 head
  时转化为 Ray CLI 参数；worker 接入已有 head 时不传 `--temp-dir`。这使 CoreX
  节点可避免默认按宿主机 CPU 数预启动大量 worker，且可将 Ray session/spill 目录
  放到空间充足的 `/data1`。
- 新增资源覆盖参数入口回归；启动参数相关定向测试为 3 passed，`compileall` 和
  `diff --check` 均通过。两机实测端口探针显示跨节点高位端口仍受网络策略约束：
  该问题需由网络/防火墙层开放 Ray 所需的 GCS、node-manager、object-manager 和
  worker 端口范围后，才能继续真实跨机 Manager 部署。

## 2026-09-02：论文第三点基础部署冒烟验证

论文第三个研究点为“基于异构负载感知的跨域请求调度技术”，核心包括异构
计算/显存负载感知的请求分发，以及结合存算协同的请求迁移。使用本地
Qwen3-14B（vLLM 0.11.2 V1、CoreX 4.4.0、TP=2）启动
`llumnix.entrypoints.vllm.v1_api_server` 做最小验证：

- `GET /health` 返回 HTTP 200；
- 单个 `/generate` 请求正常返回中文结果；
- 4 个并发请求均成功返回，端到端耗时约 0.42--0.45 秒，request ID
  路径未出现错误。

启动前需 `source tools/corex44_env.sh`；脚本现同时设置 `LD_LIBRARY_PATH`
和 `LIBRARY_PATH`，解决 CoreX 驱动库位于 `/usr/local/corex-4.4.0/lib64`
时 Triton 编译 CUDA helper 找不到 `-lcuda` 的问题。本次仅验证基础服务和
并发分发入口，尚未宣称跨域多实例迁移实验完成。

## 2026-09-02：论文第三点异构负载模型补齐

V1 `InstanceInfo` 新增 GPU 总/空闲显存和归一化算力容量字段，V1 adapter 从
`torch.cuda.mem_get_info()` 发布实际设备状态。`virtual_usage` 现按请求数除以
算力容量计算计算压力，并融合显存占用率；旧引擎无新字段时仍回退到 block
counter。异构容量排序与回退逻辑新增测试，相关定向回归为 44 passed。

## 2026-09-02：V1 部署参数完整兼容

独立 `v1_api_server` 改为复用 vLLM 0.11 `AsyncEngineArgs` 的完整 CLI 参数
注册与校验。量化、tokenizer、调度和 KV-transfer 等原生选项不再被静默丢弃；
新增 CLI/HTTP 回归共 16 passed，`compileall` 通过。
远端 `10.31.10.210` 同步提交后执行扩展定向集也为 45 passed。

主 `api_server` 本地部署路径修复 Manager 上下文透传，避免 V1/P-D 配置在本地
多实例启动时丢失；专用入口回归与 V1/KV 回归合计 44 passed。

主 API `/generate` 已补齐 V1 请求校验与 streaming 断连 abort 生命周期，行为与
独立 `v1_api_server` 一致；`compileall` 通过。
无模型回归确认非法请求返回 400，正常请求保留公开 request-id。

新增 GlobalScheduler 闭环测试，验证异构负载与 KV prefix affinity 的联合决策：
负载差在安全窗口内选缓存命中实例，明显过载时保留负载优先级；本轮完整定向
回归为 48 passed。
远端同步提交 `f1f636e` 后同一 48 项回归和 `compileall` 也通过。

主 API `/instance_list` 增加 V1 异构状态和 KV affinity 可观测字段（总/空闲显存、
算力容量、调度负载、缓存 block 数），并保持旧字段兼容；无模型接口回归 12 passed。

修复 V1 自动扩缩容对旧 `instance_load_dispatch_scale` 和 block-manager 空实例
假设的依赖，改为直接计算当前实例负载并按周期安全扩容；全局调度、KV、API 和
入口定向回归共 53 passed。
补充控制器级测试：高请求压力下 `GlobalScheduler.check_scale()` 正确返回一个
扩容实例，V1 异构负载确实进入自动扩缩容决策。

将 `virtual_usage` 加入正式 `--scaling-load-metric` CLI 选项，允许部署直接启用
CoreX 异构显存/算力扩缩容；参数注册回归通过。
远端同步 `3127034` 后，扩展定向回归为 55 passed，`compileall` 通过。

V1 P/D 配置门禁已补齐：旧 block-manager 后端配置会在启动前明确拒绝，要求
`kvtransfer` 与 P2P connector endpoint，避免半配置的 P/D 服务；KV transfer 回归
为 28 passed。
远端同步 `8c9a4a9` 后，扩展 V1/KV/API/调度回归为 56 passed，`compileall` 通过。

补齐 V1 Llumlet 故障流回收：EngineCore/connector 异常时先 abort 内部请求，再清理
bookkeeping；无模型故障流测试通过，本轮核心回归为 57 passed。

补齐 Manager 故障期间的 fallback 请求取消：客户端 abort 会直接通知 fallback
Llumlet 并回收本地计数；入口故障回归扩展后本机核心集为 58 passed。

两机同步到提交 `5893493` 后分别执行同一 44 项定向回归均通过。实机采样的
`virtual_usage` 为本机 0.00468、远端 0.00522，差异由两端的实时空闲显存产生；
同一 token prefix 的 `sha256_cbor` 哈希保持逐字节一致。

## 2026-09-01：Python 3.12 与 V1/KV 亲和基础

- 将 `setup.py` 的 Python 版本范围扩展到 `>=3.9,<3.13`，并补充 Python 3.11/3.12 classifiers，匹配 CoreX 4.4.0 本机 Python 3.12 环境。
- 将 vLLM extra 依赖切换到 `>=0.11.2,<0.12`，Ray 下限调整为 `2.52.1`；旧 vLLM 0.6 后端仍保留在源码中，但仅由旧版本运行时选择。
- 修复 Ray 2.52 测试夹具对已删除 `ray._private.utils.hex_to_binary` 的依赖。
- 新增 `llumnix.backends.vllm.v1_kv.KVCacheAffinityIndex`：消费 vLLM V1 KV 事件，按实例维护 block ownership，并提供 prefix affinity 与候选实例排序。
- 新增 V1 KV 亲和单元测试；当前测试结果：12 passed（包含既有全局调度测试）。
- 提交 `4b61873` 修复 KV block hash：由 Python 内置 `hash` 改为稳定的
  BLAKE2b-64 摘要，保证跨进程/跨主机收到相同 token prefix 时亲和计算一致。
- 提交 `f1beadf` 与 `4b61873` 已推送到 `tianshu-huiyang/main`。

## 2026-09-01：V1 connector 配置层

- 新增 `llumnix.backends.vllm.v1_kv_transfer.configure_v1_kv_transfer`，将
  Llumnix 的 `migration_backend=kvtransfer` 映射到 vLLM 0.11 的
  `KVTransferConfig`，并默认打开 ZMQ KV events；支持通过
  `LLUMNIX_KV_ROLE/RANK/PARALLEL_SIZE/IP/PORT` 注入多实例连接参数。
- `migration_backend` 不是 `kvtransfer` 时不修改 vLLM 原生配置，保持既有
  单实例行为；V1 不再在 CLI 层无条件清除显式的 connector 选择。
- 扩展 `migration_backend_transfer_type` 的合法值，允许
  `SharedStorageConnector`、`P2pNcclConnector`、`NixlConnector` 等 vLLM V1
  connector 名称。connector 仅完成配置注入，尚未宣称跨机传输已验证。
- 新增 3 个配置映射单测，结果：`4 passed`（含 V1 KV 亲和测试）。
- 已确认 CoreX/IX-ML/Driver 仍为 4.4.0，未修改驱动。

## 当前边界

V1 KV events、prefix hash 和 cache-aware dispatch 已接入；vLLM 原生
`P2pNcclConnector` 负责物理 GPU block 的导出/传输/导入，但在 CoreX 两机上
尚未完成实际 worker tensor transfer 验证，因此当前不宣称跨实例迁移全部完成。

## 2026-09-02：显式 P2P 配置兼容与边界收紧

- 即使未设置 Llumnix 的 `migration_backend=kvtransfer`，只要用户显式提供
  vLLM `KVTransferConfig(kv_connector="P2pNcclConnector")`，CoreX NCCL ABI
  shim 也会被应用；非 P2P 原生配置保持不变。
- 增加 consumer 侧 `___prefill_addr_<host>:<port>___` 请求 ID 编码/解码，
  与 producer 的 `___decode_addr_...___` 对称，并在输出前恢复公开 request ID。
- 暂不宣称 V1 P/D 编排完成：当前 Manager 不会凭空创建第二个 consumer 请求，
  因此仍需实现 producer/consumer 双请求生命周期、输出抑制、abort/超时清理后，
  才能进行模型级端到端验证。
- 本轮相关单测：31 passed；未下载或提交 `.models/`、`.conda-corex44/`。

## 2026-09-02：V1 P/D 双请求编排第一阶段

- `Manager.generate` 在同时存在 prefill/decode V1 实例且两端报告
  `kv_endpoint` 时，会为一个公开 request ID 启动 producer 与 consumer 两个
  `AsyncLLM` 请求；producer 只转发 KV、不向 API 输出，consumer 负责公开输出。
- `Manager.abort` 记录同一公开请求对应的两个实例，并将取消操作广播到两端；
  普通单池部署仍保持原有单请求路径。
- `V1EngineAdapter.get_kv_endpoint` 发布 P2P 基础端口（connector 内部按 rank
  加偏移），避免跨主机连接发生双重端口偏移。
- 该阶段尚未完成模型级跨机 P/D 验证；仍需在两台机器用相同权重验证 KV
  复用、生成一致性、producer 失败/超时清理和 HTTP 端到端输出。
- 若实例尚未报告完整的两端 `kv_endpoint`，Manager 会安全回退到 prefill 单请求，
  避免部署收敛期间请求悬挂；相关回归测试当前为 32 passed。一次旧
  `test_manager.py` 运行因已连接 Ray 集群的 raylet 被 OOM/终止而在 teardown
  失败，不作为本次代码功能失败结论。
- 若 producer 已启动而 consumer actor 启动失败，Manager 会 best-effort abort
  producer，避免遗留后台 KV 发送任务。
- endpoint 发布现在优先采用显式 vLLM `KVTransferConfig.kv_ip`，再由
  `LLUMNIX_KV_IP` 覆盖；这支持不依赖 Llumnix legacy 参数的双机配置。

## 2026-09-02：两机模型级启动探针

- 通过 Git bundle 将当前代码部署到 `10.31.10.210`，避免 GitHub 拉取超时；
  远端补齐缺失的用户态 `yacs` 与 `backports.zstd` 后，Llumnix 可正常导入。
- 两端使用已有 Qwen3-14B 权重、Python 3.12、vLLM 0.11.2、CoreX 4.4.0，
  各用一张 GPU 成功加载模型和 `CoreXP2pNcclConnector`；本机与远端均报告
  17K 级 KV cache 并监听 `19052` 基础端口。
- 独立 `AsyncLLM` producer/consumer 探针尚未观察到 `ncclCommInitRank`、KV
  send/recv 或 consumer 输出。当前证据证明“模型/connector 可启动”，不证明
  “模型级 KV handoff 已完成”；探针脚本保留为 `tools/v1_p2p_model_probe.py`，
  默认使用超过一个 KV block 的 prompt，后续继续定位 vLLM P2P 调度元数据条件。
- 探针结束后已清理两端 engine 进程，GPU 恢复空闲；没有修改驱动或系统安装。
- 进一步定位到共享双 endpoint request ID 的解析问题：上游 connector 的贪婪
  正则会把第二个 marker 吞入 hostname。CoreX shim 现按 marker/suffix 精确解析
  producer 与 consumer 地址；相关测试 23 passed。下一次模型探针将验证该修正
  是否触发真实 `ncclCommInitRank` 和 consumer 输出。
- 修正后使用 241-token prompt 与同步 `PUT` 模式重测：两端共享内部 request ID
  且均正常启动模型/connector，但未出现 communicator、send/recv 或 consumer
  输出。因此模型级 handoff 仍未通过；该结果与已通过的底层 NCCL tensor P2P
  验证严格区分。
- CoreX shim 增加了低侵入诊断：每个调度 step 记录 connector metadata 请求数，
  以及 producer `save_kv_layer`/consumer `start_load_kv` 的调用。该日志只用于
  定位 V1 调度元数据触发条件，不改变上游传输实现。

## 2026-09-01：跨主机事件与 hash 一致性验证

- 本机 `10.31.10.62` 使用 vLLM `EventPublisherFactory` 在
  `tcp://*:18077` 发布 `BlockStored`；第二台 `u210`
  (`10.31.10.210`) 使用其 Python 3.12.13 / vLLM 0.11.2 环境成功订阅并解码，
  收到 `BlockStored cross-host-test`。验证通过 `ens1f0`，未改远端文件或系统配置。
- 两机以 `PYTHONHASHSEED=0`、`sha256_cbor` 对 token blocks
  `[1,2,3,4]`、`[5,6,7,8]` 计算出的 32-byte hashes 完全一致：
  `c9d58ba6...ef071cb`、`24125b23...54ab2d6`。
- 该验证证明跨机 KV ownership 事件和亲和 hash 算法链路可用；尚未证明
  `P2pNcclConnector` 的 GPU tensor 传输，后者需要两端启动实际 V1 worker 后再测。

## 2026-09-01：P2P 启动保护与普通请求自动亲和

- 普通 Manager `/generate` 请求现在会从 live V1 Llumlet tokenizer 计算
  `sha256_cbor` prefix block hashes，再交给 GlobalScheduler；调用方无需手工
  传入 token IDs。tokenizer/多模态不适用时自动回退原调度策略。
- `P2pNcclConnector` 增加早期配置校验：要求 `kv_parallel_size=2`，producer
  要求 `LLUMNIX_KV_DECODE_ADDRESS=host:port`；否则在创建 GPU engine 前返回
  明确错误。公开 request ID 与 connector 内部地址后缀继续分离。
- 本机测试结果：`20 passed`，compileall 与 diff-check 通过。

## 2026-09-02：CoreX NCCL P2P 数据面验证

- 发现 CoreX 4.4.0 的 `libnccl.so.2` 为 2.24.3，不导出 vLLM ctypes wrapper
  探测的可选 `ncclCommWindowRegister/Deregister`。新增
  `corex_p2p_connector.py` shim，仅从进程内函数描述表中移除缺失的可选符号，
  不修改 `/usr/local/corex-4.4.0` 或驱动；普通 NCCL InitRank/send/recv 保持原实现。
- `kvtransfer` 在检测到该 ABI 时自动选择 `CoreXP2pNcclConnector`，仍要求
  `kv_parallel_size=2` 和 producer 的 `LLUMNIX_KV_DECODE_ADDRESS`。
- 两机真实探针成功：`10.31.10.62` 与 `10.31.10.210` 各启动一个
  `P2pNcclEngine`，通过 `ens1f0` 初始化 rank 0/1 communicator，并传输 GPU
  tensor `[1.0, 2.0, 3.0, 4.0]`，接收端逐值一致。
- 测试夹具的失败日志路径改为可写临时目录（可由
  `LLUMNIX_TEST_ERROR_LOG_DIR` 覆盖），避免 CoreX 节点的 `/home/lhy` 权限错误
  遮蔽真实测试失败。当前本机相关测试：`23 passed`。

## 2026-09-01：真实事件订阅与亲和调度接入

- `V1EngineAdapter` 新增 ZMQ `KVEventSubscriber`，按实例消费 vLLM
  `KVEventBatch`，将 `BlockStored`/`BlockRemoved`/`AllBlocksCleared` 实时写入
  `KVCacheAffinityIndex`；shutdown 时关闭后台订阅线程。
- 修复默认 vLLM `sha256` 事件 hash 的 bytes 保真处理，并在 `kvtransfer`
  配置下启用 `sha256_cbor`、prefix caching 与固定 `PYTHONHASHSEED`，使跨主机
  prefix hash 可复现。
- `InstanceInfo` 透传实例已缓存 block hashes；GlobalScheduler 支持可选
  `llumnix_kv_block_hashes`，在负载差不超过 0.10 时优先缓存命中，负载差距较大
  时仍遵循原有调度策略。
- P2P connector 请求 ID 的地址后缀现在由 adapter 添加、输出前移除，避免破坏
  Manager/API 的公开 request ID；支持 `LLUMNIX_KV_DECODE_ADDRESS=host:port`。
- 本机相关测试结果：`19 passed`；compileall 与 diff-check 通过。
- 已只读核验第二台 `u210`：CoreX/IX-ML/Driver 4.4.0、Python 3.12.13、
  torch 2.7.1、ray 2.58.0、vLLM 0.11.2；`ens1f0=10.31.10.210` 可用于后续
  NCCL/事件端口验证。尚未启动跨机推理或修改远端文件。

## 2026-09-02：V1 P2P attention hook 定位与修复

- 两端 Qwen3-14B V1 探针均可启动，且 producer/consumer scheduler 均生成
  `requests=1` 的 KV connector metadata；之前 consumer 未启动或 request ID
  不一致，不能作为数据面失败依据。
- 在 CoreX vLLM 0.11.2 attention 实现中确认，`save_kv_layer` /
  `wait_for_layer_load` 仅由 `VLLM_SUPPORT_IXSERVER` 守卫的
  `maybe_transfer_kv_layer` 调用。CoreX 默认该标志为 false，导致 metadata
  已绑定但 producer 从未导出 KV，consumer 会持续等待。
- `CoreXP2pNcclConnector` 现在在进程内启用该 V1 attention hook，并同步更新
  已缓存的 `vllm.envs.VLLM_SUPPORT_IXSERVER`，不修改驱动、系统库或 CoreX
  安装。针对性测试 `test_v1_kv_transfer.py`：`19 passed`，compileall 通过。
- 修复提交：`0696bf1`（后续缓存配置同步补丁待推送）。模型级跨机 handoff
  仍需用同一 request ID 的干净双端实验最终确认，当前不宣称端到端已完成。

## 2026-09-02：真实模型 P2P 数据面继续定位

- 使用同一内部 request ID 的两机 Qwen3-14B 探针验证：producer 已执行
  `save_kv_layer`，并与 `10.31.10.210:19052` 成功完成
  `ncclCommInitRank`；说明 V1 attention hook 和 request-ID 路由已生效。
- consumer 在 CoreX NCCL rank-1 初始化阶段报告 native
  `double free or corruption (out)` 并退出。已在 shim 中禁用 vLLM P2P helper
  的 cuMem allocator 模式（`NCCL_CUMEM_ENABLE=0`），且将设置前移到 connector
  模块导入时；针对性单元测试仍为 `19 passed`。模型级 handoff 尚未宣称成功，
  需要继续确认 CoreX/远端运行时的 communicator ABI 与 allocator 兼容性。

## 2026-09-02：两机模型级 P/D KV handoff 验证通过

- 在两台 CoreX 4.4 / Python 3.12 / vLLM 0.11.2 主机上，以同一内部 request ID
  运行 Qwen3-14B producer/consumer。CoreX 专用 connector 默认使用
  `corex_transport=zmq_cpu`：以 ZMQ 传输 CPU-staged KV tensor，再注入 consumer
  GPU paged cache；`corex_transport=nccl` 仍保留为显式性能实验选项。
- 实测 producer 在 `10.31.10.62` 导出全部 40 个 attention layer KV；
  `10.31.10.210` consumer 完成 KV load 后输出 `重复的`，随后以
  `finished=True` 输出 `重复的句子`。没有 EngineCore 崩溃或 NCCL rank-1 初始化。
- 该结果首次证明当前 V1 P/D 内部 request ID、调度 metadata、attention hooks、
  跨主机 KV 数据面与 consumer 公开生成能够端到端协同。原生 NCCL 模式仍因
  CoreX rank-1 native double-free 保持非默认诊断路径，不能作为生产默认。
- 本地 V1 KV transfer 测试：`20 passed`；同时修复测试失败日志夹具漏导入
  `tempfile` 的 Python 3.12 路径。
- 补齐 `zmq_cpu` staging transport 的 listener/socket/context shutdown 和幂等
  回收，避免长运行服务重启时残留端口；新增回收测试后 V1 KV transfer 为
  `21 passed`。
- 新增纯 CPU 的 staging wire-protocol round-trip 测试，覆盖 tensor 的
  shape/dtype/payload 传输与双端关闭；V1 KV transfer 测试增至 `22 passed`。
- 修复 staging consumer 使用默认 CUDA 设备的问题：接收 KV 现在绑定
  EngineCore 的 `local_rank`，避免多 GPU/Ray placement group 下跨卡注入；相关
  多 GPU 设备绑定测试加入后，完整相关回归为 `40 passed`。
- 独立 `v1_api_server` 入口改为通过 `V1EngineAdapter` 创建引擎，统一采用
  Llumnix 的请求生命周期、CoreX connector 和 request-ID 处理；多实例编排仍由
  Manager 负责。
- staging transport 增加 malformed payload 防护、发送/接收超时和可诊断异常，
  防止对端故障导致 listener 线程退出或请求永久阻塞；回归保持 `40 passed`。
- 新增无对端发送、无 tensor 接收的超时故障恢复测试；V1 KV transfer 定向测试
  当前为 `25 passed`，超时默认值可通过 `zmq_recv_timeout_s` 调整。
- 新增 malformed payload 后继续接收合法 tensor 的恢复测试，确认 listener 不会
  因单个坏包退出；V1 KV transfer 定向测试当前为 `26 passed`。
- Manager 的 V1 P/D 启动异常现在会并行取消已启动的 producer 与 consumer，
  防止 consumer 阻塞等待 KV 或 producer 遗留发送任务；相关回归保持 `42 passed`。
- 独立 `v1_api_server` 实测发现 Qwen3-14B 在 32 GiB 卡上需要限制 vLLM
  sampler 预热并发；入口新增 `--max-num-seqs`（默认 4）。使用 CoreX 栈真实
  验证 `/health=200`，`/generate=200` 并返回正常文本。

## 2026-09-02：V1 请求生命周期 bookkeeping 收敛

- Manager 周期性请求轮询现在同时重建 `request_instance` 和
  `request_instances`，以各 Llumlet 的活动 request-id 集合为权威来源。
  这覆盖同一公开请求对应 producer/consumer 两个 V1 actor 的完成场景，避免
  完成后残留双实例映射导致后续 abort 访问过期请求。
- CoreX/V1 相关回归：`43 passed, 7 warnings`；Python 3.12 `compileall` 通过。
- 随后修正周期任务：每个清理间隔都从各 Llumlet 重新拉取活动 request-id，原子
  重建 `request_instance`/`request_instances`；新增 P/D 双端完成回收单测，相关
  回归为 `44 passed, 7 warnings`。

- Python 3.12 安装门禁验证：`pip wheel . --no-deps --no-build-isolation` 成功，
  wheel 元数据声明 `<3.13`、Python 3.12 classifier 与 vLLM V1 依赖范围。
  V1 测试收集会跳过仅依赖 vLLM 0.6 block-manager API 的历史测试；专用 V1/调度
  回归为 `44 passed, 7 warnings`。一次全量 unit-test 运行受已有 Ray 集群的
  raylet/dashboard 异常退出影响，未归因于代码断言失败。
- 测试夹具改为随机本机 Ray 控制端口并显式 `ray.init(address=...)`，清除外部
  `RAY_ADDRESS` 继承，避免连接共享集群；V1 KV/affinity 定向回归在隔离夹具下
  为 `30 passed, 1 warning`。
- 修正生产 Ray head 启动分支，显式传递 `--node-ip-address`，避免多网卡 CoreX
  节点选错接口；随机端口入口测试 `test_launch_ray_cluster` 通过。
- unit-test fixture 改为无 dashboard 的进程内 Ray，并使用 `ray.shutdown()` 清理；
  `test_launch_ray_cluster` 改用 subprocess mock 校验启动参数，避免测试停止或
  依赖共享 Ray 集群。缺少 `pytest-asyncio` 的 async 测试会明确 skip；当前 V1
  定向测试保持通过。
- 追加隔离 legacy `entrypoints/vllm/test_api_server.py`（其启动的是 vLLM 0.6
  HTTP 生命周期）；V1 HTTP 行为以 `v1_api_server` 实测为准。完整 unit-test
  仍可能在历史 Ray actor/placement 长轮询处停滞，已停止该次运行并保留专用
  V1 门禁结果。
- 独立 V1 API 入口改用 FastAPI `lifespan` 释放 adapter，移除弃用的
  `@app.on_event("shutdown")`；新增无模型 health/普通生成/streaming mock 测试，
  定向测试 `2 passed`。
- 流式响应 iterator 提前关闭时自动调用 V1 `abort`，覆盖客户端断连后的请求
  回收；V1 API mock 测试增至 `3 passed`。
- 修复独立 API 直连 `V1EngineAdapter.generate` 未登记 request alias 的问题，
  现在 connector 内部 ID 可被 abort 正确解析；API/KV transfer 定向回归为
  `30 passed, 1 warning`。
- 补充正常完成请求的 adapter bookkeeping 释放，并修复 `abort_request` 的 alias
  双重解析；ZMQ staging 单测改用动态空闲端口。API/KV transfer 回归为
  `31 passed, 1 warning`，compileall 通过。
- `scipy`/`pandas` 改为 profiling 分析功能的延迟依赖，缺失时返回明确 RuntimeError；
  核心 V1 serving 不受影响。新增可选依赖回归后 API/KV/profiling 测试为
  `33 passed, 1 warning`。
- V1 HTTP `/generate` 增加输入验证：非法 prompt/stream/request-id/采样参数返回
  HTTP 400，并保证不调用 EngineCore。API/KV/profiling 回归为 `34 passed, 1 warning`。
- 补充非对象 JSON body 校验，避免列表输入触发未处理异常；API/KV/profiling
  回归仍为 `34 passed, 1 warning`。
- 新增统一 P2P endpoint 校验（具体 host 与 1--65535 端口），Manager 在 P/D
  编排前拒绝 malformed endpoint 并安全回退；KV/affinity/调度/API/profiling
  回归为 `52 passed, 7 warnings`。
- 修复 V1 adapter 对 Manager 共享 P/D request-id 的二次装饰：producer/consumer
  均保留同时含 decode/prefill endpoint 的内部 ID，避免丢失对端路由 marker；相关
  KV/affinity/调度/API 回归仍为 `52 passed, 7 warnings`。
- V1 adapter 在 EngineCore 同步 `generate` 启动失败时回滚 request/alias/running
  状态，防止独立 API 或 Llumlet 遗留孤立 bookkeeping；API/KV/affinity 定向回归
  `40 passed, 1 warning`，compileall 通过。
- 新增 V1 API FastAPI lifespan shutdown 回归，确认服务退出调用 adapter shutdown；
  API/KV transfer/affinity 定向回归为 `38 passed, 1 warning`，compileall 通过。
- 使用 Git bundle 将当前 `f084a52` 同步至 `10.31.10.210`，只读核验远端栈为
  Python `3.12.13` / vLLM `0.11.2` / PyTorch `2.7.1`；与本机双机 KV-affinity
  和 P/D 验证的正式版本一致，未改动远端系统或驱动。
- `tools/corex44_env.sh` 前移设置 `PYTHONHASHSEED=0`，保证 Python 启动前即
  满足 vLLM `sha256_cbor` affinity hash 的跨主机可复现要求；远端探针曾发现
  未设置该变量的诊断提示，未修改远端系统环境。
- `tools/corex44_env.sh` 改为自动发现项目 clone 或远端共享 CoreX Python 环境，
  支持 `LLUMNIX_COREX_PYTHON_ENV` 覆盖；本机路径和远端环境路径均已验证可选中，
  V1/KV/API 回归为 `40 passed, 1 warning`。
- 通过 bundle 将 `946304a` 同步至 `10.31.10.210`，远端环境脚本自动选择
  `/data1/congmng/conda-envs/ds-corex44`，编译及 `sha256_cbor` hash 探针通过，
  `PYTHONHASHSEED=0` 生效。报告中的早期 native-NCCL/P/D 阶段结论已显式标为
  历史记录，当前正式结论仍为两机 `zmq_cpu` 模型级 handoff 已验证。
- 随后将最新报告与代码同步到 `6a1f5c2`，远端 Python 3.12.13 编译及关键
  endpoint/hash 导入探针通过；两机部署基线与 GitHub `main` 一致。
- 在本机与 `10.31.10.210` 使用当前基线、`PYTHONHASHSEED=0` 对相同 token blocks
  `[1,2,3,4]`、`[5,6,7,8]` 重新计算 `sha256_cbor`，两端 hash 逐字节一致，
  进一步确认 KV-cache affinity 算法在双机 Python 3.12/CoreX 栈上可复现。

## 2026-09-03：论文第三点基础部署验收

### 2026-09-03：P/D 重新回归与资源隔离修复

### 2026-09-03：正式支持审计与双机 affinity 复验

- V1 Llumlet 不再初始化依赖 vLLM 0.6 block-manager 的 legacy migration
  coordinator/scheduler；旧迁移 RPC 安全拒绝并引导至 connector-driven P/D handoff。
  API/KV transfer 回归 `46 passed`，避免当前 V1 栈携带不可用旧迁移对象。

- 最新提交 `36e5b77` 的两机 P/D 实机复验：`/instance_list` 正确显示
  `10.31.10.62` 和 `10.31.10.210`，`pd-topology-001` 的 `/generate` 返回 HTTP
  200 非空中文；两端 producer/consumer 日志确认共享 endpoint request ID。

- 完成当前 V1 支持边界审计：报告新增能力矩阵，明确 ZMQ CPU-staging P/D 为正式
  路径、native NCCL 为非默认诊断项、旧 vLLM 0.6 任意时刻迁移不适用于 V1；避免
  历史实现与当前正式支持范围混淆。

- 本机 Python `3.12.13` / CoreX 4.4 / vLLM `0.11.2` / Ray `2.52.1` /
  PyTorch `2.7.1` 环境成功执行 `pip wheel . --no-deps --no-build-isolation`。
- V1 transfer、placement 与 Manager 调度定向回归为 `44 passed, 10 skipped`；
  单独 API/transfer 复验为 `46 passed`。
- 本机与 `10.31.10.210` 对 KV-affinity probe 得到完全一致的 hash、affinity
  (`0.5/1.0`) 与排序 (`candidate-b,candidate-a`)。
- 修复 CoreX Ray worker context 返回空 node IP 时的 node-table 回退，确保跨机
  `/instance_list` 可观测性不依赖 dashboard State API。

- 真实 P/D 请求已到达两端 connector：producer 保存 KV、consumer 收到 metadata
  并执行 load。请求随后因 CoreX torch 的 `bfloat16.numpy()` 不支持而失败；已将
  ZMQ staging 改为 uint8 原始字节视图/声明 dtype 重建，并加入 BF16 跨端 round-trip
  测试。V1 KV-transfer 定向测试 `31 passed`，提交 `ae81d39` 已推送。

- 在同步 `4ee79e3` 的干净两机集群上重新执行 `pd-bf16-001` 成功：两台 BI-V150
  各运行一个 TP=1 Qwen3-14B（Prefill=`10.31.10.210`、Decode=`10.31.10.62`）。
  producer 日志显示 40 层 BF16 KV 已发送，consumer 显示共享 metadata 和
  `load role=consumer`，HTTP `/generate` 返回 200 非空中文结果。此前 BF16
  序列化故障已闭环解决，当前提交获得模型级两机 P/D handoff 证据。

- 现场日志定位到 CoreX Ray 特有的 `ActorOptionWrapper` 不支持 `.options()`，
  导致 Llumlet 在 actor 创建阶段退出；提交 `fa05a26` 增加兼容回退，并同步到
  `tianshu-huiyang/main`。
- P/D 状态检查现在显式使用 `RAY_ADDRESS`；State API/dashboard 不可用时仅停用
  placement-group reconciliation，不再把该异常当作 P/D 实例故障。
- 修复后部署已进入 vLLM V1 EngineCore 初始化。最新失败日志显示另一实例启动时
  被隔离 TP=2 基线残留进程耗尽显存（`0.93/32 GiB`），已释放该进程并清理共享
  集群中的 detached actors。下一轮需保持两张卡无其他模型进程，再复现模型级
  producer/consumer KV handoff。

- 对照论文第五章“基于异构负载感知的跨域请求调度技术”，在本机
  `10.31.10.62` 与远端 `10.31.10.210` 的双机 Ray/P-D 服务上完成基础验收。
- `/health` 返回 HTTP 200；`/instance_list` 返回 Prefill、Decode 两个实例，
  节点信息与 KV endpoint 已注册；普通 `/generate` 请求连续两次返回 HTTP 200 和
  正常文本。
- 请求 `pd-smoke-001` 经 Manager 分发至 Decode；Decode 日志确认收到共享请求 ID
  携带的 Prefill/Decode endpoint metadata（`requests=1`），并执行
  `load role=consumer metadata_requests=1`；Prefill 侧记录 KV block save。
  由此确认当前 CoreX 兼容路径已完成模型级 P/D KV handoff 基础闭环。
- 本次仅验证功能可用性，不宣称论文中的 QPS/延迟收益已复现；论文原生 vLLM
  0.6 block-manager 与 native NCCL GPU handoff 仍属于后续适配范围。
- 修复 Ray Llumlet worker 不继承 `LLUMNIX_KV_PORT` 等 V1 connector 环境变量的
  部署缺口：V1 actor 现在通过受限 `runtime_env.env_vars` 接收 connector IP、端口、
  rank/role、事件端点和固定 hash seed；回归测试与 Python 3.12 `compileall` 通过。
- 修复固定规模全局部署仍启动自动扩容循环的问题。`enable_scaling=False` 时不再
  创建额外 placement group，避免 GPU 不足时 pending PG 无界累积；现场清理 237 个
  残留 PG，Ray 状态恢复为无资源需求。该修复已提交并同步两机。
- 修复全局固定规模入口依赖自动扩容创建初始副本的问题：新增幂等
  `Manager.init_global_instances`，由 setup 显式创建初始 API/Llumlet 实例；并修正
  CoreX Ray actor runtime-env 的提交方式，避免 raylet 内部兼容错误。Python 3.12
  编译与 39 项 V1/KV/affinity 回归通过。
