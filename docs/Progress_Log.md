# Llumnix CoreX 4.4.0 适配进度

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
