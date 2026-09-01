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

KV 亲和索引已经可以独立消费 V1 事件，但尚未把物理 GPU block 的导出、跨主机传输和目标实例导入接入 vLLM V1 scheduler。下一阶段将实现 V1 KV connector/transfer executor，并在本机与 `10.31.10.210` 分层验证；在此之前不宣称跨实例迁移完成。

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
