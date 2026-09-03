# Llumnix 在 Iluvatar CoreX 4.4.0 上的适配与验证报告

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
