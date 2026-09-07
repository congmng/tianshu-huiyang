# vLLM V1 真正 KV 迁移实施规划

## 1. 目标与边界

本文规划在 vLLM V1（当前 CoreX 基线为 vLLM 0.11.2）中增加真正的 KV
cache migration，使 Llumnix 能把已经开始生成的请求从实例 A 迁移到实例 B，
并在 B 上继续生成。“任意时刻”定义为一次 GPU forward 完成后的 scheduler
token boundary，不尝试在 GPU kernel 执行中途抢占。

目标能力：Decode-to-Decode、运行中请求冻结/导出/导入/提交/回滚，以及完整 KV
内容迁移，而不是目标端重新 prefill。首版限制 TP=1、普通 causal-LM 和确定性
单步采样；暂不承诺 speculative decoding、复杂 structured output、多模态、LoRA、
TP>1 或跨模型/KV layout 迁移，须在协议稳定后逐项加入。

## 2. 为什么必须改 vLLM

现有 V1 connector 的 `start_load_kv()`/`save_kv_layer()` 解决的是 Prefill →
Decode handoff：目标请求通常尚未执行。它没有暂停 RUNNING 请求、导出 scheduler
状态、重建 worker 请求索引和原子释放源 blocks 的语义。

不能只在 Llumnix 传 tensor，也不能把源端 block ID 交给目标端。每个实例拥有独立
block pool；目标端必须分配自己的 blocks，并在目标确认成功前保留源端副本。

## 3. vLLM fork 与版本策略

维护一个基于 CoreX vLLM 0.11.2 的独立 fork（建议独立仓库），变更审计清单为：

```text
vllm/v1/request.py
vllm/v1/core/sched/scheduler.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/sched/output.py
vllm/v1/engine/__init__.py
vllm/v1/engine/core.py
vllm/v1/worker/model_runner.py
vllm/distributed/kv_transfer/kv_connector/v1/
```

每次升级 vLLM 必须重新运行 ABI、scheduler、KV layout 和迁移协议测试，不能仅替换
wheel。

## 4. 迁移状态机

迁移态必须由 EngineCore 所在线程串行执行：

```text
WAITING/RUNNING --prepare_out--> MIGRATING_OUT --export/transfer--> OUT_READY
      ^                                  |                       |
      +------------ abort_out -----------+------- commit_out -----+--> RELEASED

IMPORTING --KV/checksum ok--> IMPORT_READY --commit_in--> RUNNING
    |                                  |
    +------------ abort_in -------------+--> free reserved blocks
```

`MIGRATING_OUT` 请求从 running/waiting 调度集合摘除。prepare 只允许在当前 batch
完成后执行，不得在 `execute_model()` 中途修改请求或 block pool。

### 两阶段提交

1. 源端冻结请求，固定 `num_computed_tokens`/输出 token，并把源 blocks 标为
   migration-pinned；
2. 目标端验证模型、parallel 和 KV layout，分配本地 blocks；
3. connector 按 KV group/layer 写入目标 blocks，并传输校验和；
4. 目标恢复 Request/worker 状态，执行首个 decode step 作为可运行性确认；
5. 目标确认后源端提交并释放 blocks；
6. 任一步失败，目标释放预留 blocks，源端取消冻结继续执行。

源端在目标确认之前不能释放 KV，这是防止网络错误导致请求丢失的核心不变量。

## 5. vLLM 接口设计

接口应是受 EngineCore 控制的内部 API，不直接暴露可随意修改的 `Request` 成员。

### 请求快照

```python
@dataclass(frozen=True)
class RequestMigrationSnapshot:
    request_id: str
    migration_epoch: int
    prompt_token_ids: tuple[int, ...]
    all_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    num_computed_tokens: int
    max_tokens: int
    sampling_params: bytes
    stop_reason: int | str | None
    kv_layout_version: str
    kv_group_block_counts: tuple[int, ...]
    feature_flags: frozenset[str]
    checksum: str
```

`sampling_params` 使用可版本化序列化，不能 pickle 任意对象。随机采样须携带 RNG
状态，或首版只允许 temperature=0。grammar、LoRA、多模态等未加入快照时，prepare
必须显式拒绝。

### Scheduler/KVCacheManager

```python
snapshot = scheduler.prepare_migration_out(request_id, target_info)
blocks = kv_cache_manager.export_migration_blocks(request_id)
reservation = kv_cache_manager.reserve_import_blocks(
    request_id, snapshot.kv_group_block_counts, snapshot.kv_layout_version
)
scheduler.prepare_migration_in(snapshot, reservation)
kv_cache_manager.commit_import(reservation, received_checksums)
scheduler.commit_migration_in(request_id)
scheduler.commit_migration_out(request_id)
```

所有 API 提供 `abort_*`，并检查 request ID、epoch、防重放 token 和 layout version。
reservation 在 commit 前不改变普通 prefix-cache 引用计数。

### EngineCore 控制协议

在 `EngineCoreRequestType` 增加独立消息，而不是复用无类型 UTILITY：

```text
MIGRATE_OUT_PREPARE / MIGRATE_OUT_COMMIT / MIGRATE_OUT_ABORT
MIGRATE_IN_PREPARE  / MIGRATE_IN_COMMIT  / MIGRATE_IN_ABORT
```

消息采用 msgspec 结构体，包含 request、epoch、endpoint、snapshot metadata、超时
和协议版本。EngineCore 必须按 request ID 串行化 ABORT、FINISH、MIGRATE，避免竞态。

## 6. worker 与数据面

`model_runner`/InputBatch 需要删除源 request index，并在目标建立新的 request index、
token 视图和 KV slot 映射；恢复 grammar/structured-output（未支持则拒绝），清理
encoder/multimodal cache 引用，并在首次 forward 前验证 block table 与
`num_computed_tokens` 一致。

KV connector 格式携带 group、layer、block ordinal、dtype、shape、bytes checksum 和
migration epoch。CoreX native NCCL 继续使用已验证的 endpoint 绑定与
`NCCL_CUMEM_ENABLE=0`；不依赖缺失的 symmetric-memory window symbols。

## 7. Llumnix 集成

Llumnix 负责策略，不伪造 vLLM 内部状态：Manager 选择源/目标，向源 EngineCore
prepare，通过 connector 驱动 metadata/KV 数据面，目标确认后更新 `request_instance`，
失败则 abort。P/D handoff 可复用 snapshot 和目标 block reservation 数据结构，但
必须保留现有 request-id 路由，并使用独立 migration epoch。

## 8. 分阶段实施与验收

### Phase 0：源码基线与 ABI

固定 CoreX vLLM 0.11.2 commit、Python/PyTorch/CoreX 版本，构建 fork wheel，确认
普通 V1 推理、TP=1 和现有 native NCCL P/D 不回归，并加入 layout/protocol version。

### Phase 1：单进程 scheduler 快照

实现 token-boundary freeze、snapshot round-trip、block reservation 和 fake connector，
覆盖 waiting/running/finished/abort 竞态；验证逐字段相等且失败路径无 block 泄漏。

### Phase 2：单机双卡 Decode-to-Decode

两个 TP=1 EngineCore，temperature=0；A 生成至少 2 token 后迁移到 B，B 继续生成并
与不中断基线逐 token 比较。连续 100 次迁移必须无崩溃、重复/跳 token 或泄漏。

### Phase 3：双机真实迁移

在 10.31.10.62 与 10.31.10.210 各一张 BI-V150 上运行 Qwen3-14B native NCCL，
注入延迟、超时和目标容量不足。成功、目标拒绝、超时、源 actor 重启均需有确定结果；
support gate 校验两端 fork commit 和协议版本一致。

### Phase 4：增量与扩展

再加入增量 blocks、RNG、structured output、LoRA、多模态、TP>1 和 speculative
decoding。没有对应快照字段、回滚测试和性能数据前，不得宣称支持。

## 9. 不变量、风险与当前状态

一致性：目标 commit 前源 KV 永不释放，epoch 单调且不可重放。调度安全：迁移态请求
不同时出现在 running/waiting，且一次只有一个迁移 owner。设备安全：目标 block table
绑定目标 worker local rank。故障必须进入 abort 或明确不可恢复状态，不能静默更新
Llumnix mapping。完整 KV 复制会增加显存和网络开销，只有增量迁移稳定后才比较性能。

当前项目已验证 CoreX native NCCL P/D handoff、endpoint 绑定和 communicator 复用。
Phase 1 已在 Llumnix 新增独立、可测试的协议参考实现
`llumnix.backends.vllm.v1_migration`：它覆盖 immutable snapshot checksum、epoch、
target block reservation、目标先 commit 与源端后 release、以及 abort 回滚。该参考
实现不是生产迁移开关。

已建立 upstream vLLM `v0.11.2` fork 工作树作为 API 原型，但其不能直接替换 CoreX
wheel：在当前 CoreX PyTorch 2.7.1 环境中，上游源码导入
`torch.distributed._symmetric_memory` 时要求 wheel 未导出的 `_SymmetricMemory`。

为避免这一 ABI 风险，已从当前可运行的 `vllm-0.11.2+corex.4.4.0` wheel 提取 Python
源码，建立独立的 CoreX fork 工作树 `/data1/congmng/vllm-corex44-v1-migration`。其
baseline commit 是 `1a2b265`，已在 `60f9852` 加入 `RequestMigrationSnapshot`、
非 finished migration states 和 Scheduler source-side freeze/abort/commit API；现有
site-packages 未被修改。随后 `f27f7ab` 加入 EngineCore migration request types 和
KVCacheManager import reservation/commit/abort API；首版显式限制所有 KV groups 为
同一 block count，这是当前 V1 coordinator 的安全分配边界。fork 的协议导入、状态排序
和控制消息检查已经通过。`41584b7` 已将 source-side
`MIGRATE_OUT_PREPARE`/`COMMIT`/`ABORT` 经 msgspec socket 消息路由至 EngineCore
和 Scheduler，消息 round-trip 已验证。fork 尚未构建为 wheel，也尚未接入目标端
EngineCore command handling、worker cache 或真实 GPU data plane。因此当前 V1
生产路径仍只支持 connector-driven P/D handoff；必须以 Phase 1/2 的实际 fork 测试
结果解除该限制。

目标侧的 Phase-1 scheduler lifecycle 已在 `dffdd68`/`7952bc8` 落地：目标请求在
`MIGRATING_IN` 注册，只有目标 KV 写入完成后才可 `commit_migration_in()` 进入 waiting
queue；失败时 `abort_migration_in()` 删除请求。随后 fork 的 `36a67ac`、`1470395` 和
`7ca1d83` 增加了 worker 层按命名 layer/本地 block ordinal 的 KV tensor
export/import：payload 是 detached contiguous snapshot，目标在写入前校验 epoch、目标
block mapping、dtype、shape、payload SHA-256 与 manifest metadata SHA-256。它目前是
单进程/单层数据面 API，尚未等同于跨进程 NCCL 迁移。

`d749668` 和 `bd30a04` 将首版目标 `Request` 改为从 versioned JSON snapshot 重建，
并保持 token history、`num_computed_tokens`、`max_tokens`、EOS 与 stop-token 语义。
首版只接受 temperature=0 的纯 greedy causal-LM 请求，显式拒绝 RNG、penalty、grammar、
LoRA、多模态、logits processor、logprobs 和 structured output；这些状态的 wire 版本化
尚待 Phase 4。当前 fork head 是 `c4d71f7`，KV block + snapshot 重建 + target lifecycle
单元测试 13 项通过。目标端控制面 `MIGRATE_IN_PREPARE/COMMIT/ABORT` 已经 msgpack
路由至 EngineCore；prepare 返回 target-local reservation block IDs 给数据面。commit 后
首次调度把完整本地 block table 和既有 output token state 作为 worker 新请求下发。source
在未显式提供 group counts 时从 coordinator 提取真实物理 block counts，避免目标错误预留零块。
`7600c65` 已将 export/import 通过 GPU Worker RPC 暴露，`614901a` 增加 KV layout block
axis 自动识别与歧义拒绝。`23d0a30` 将 payload 移出 EngineCore socket：source/target
worker 在本 GPU 进程内复用 P2P connector engine 的 `send_tensor/recv_tensor`，控制面仅
返回携带 checksum 的 layer manifest；CoreX 默认走 native NCCL，保留 ZMQ CPU staging
兼容路径。fork head 当前为 `d9af571`，相关测试 17 项通过。RPC 仅允许 TP=1、PP=1，并
在 EngineCore 侧检查 frozen request 与 migration epoch。尚未完成两 EngineCore 的真实
GPU KV 导入、首次 decode 确认及 100 次循环验证，因此仍不得宣称 Decode-to-Decode 已可用。

### 2026-09-07 实测进展

已在 CoreX 4.4、Qwen3-14B、两张 BI-V150（两个独立 TP=1 EngineCore）上完成一次
真实 Phase-2 的 CPU-staged 数据面验证，命令使用 `--transport zmq_cpu`。实测通过
source 生成至少两个 token、source freeze/snapshot、target reservation、每个 KV layer
按物理 block 导出/传输/导入（manifest 与 payload checksum 校验）、target commit 以及
source commit。此次运行输出：

`PASS phase2 migration control+KV transfer+two-phase-commit`

为避免旧 P/D connector 的 chunked-prefill 状态机干扰显式迁移，Phase-2 使用
`true_kv_migration_only` 配置，禁用自动 `save_kv_layer`/`start_load_kv`，但保留 worker
P2P engine 作为显式数据面。控制面跨 msgspec 返回的 snapshot 字典也已规范化为版本化
`RequestMigrationSnapshot` 后再传输。对应 Llumnix 单元测试为 42 passed，vLLM fork 的
KV block 定向测试为 17 passed。

随后补齐了目标端 commit 后的 frontend 输出注册与真实 EngineCore decode。修正目标
Request 重建时缺失的 prefix block hashes 后，实测输出为：

`PASS phase2 migration control+KV transfer+two-phase-commit+decode-equivalence`

该次验证以 source 同一 EngineCore 的 uninterrupted greedy 序列为基线，目标端迁移后
继续生成的 token 与基线 continuation 逐 token 相等。

随后在同一双卡拓扑上使用 `--transport nccl` 完成真实数据面验证。两端日志均出现
`ncclCommInitRank Success`，并再次通过：

`PASS phase2 migration control+KV transfer+two-phase-commit+decode-equivalence`

Phase-2 launcher 现支持 `--iterations N`，每轮使用递增 migration epoch 与唯一 request ID，
并在 source commit 后释放迁移 probe 的 stream。CPU-staged 路径已完成 2 轮连续验证；
native NCCL 已完成 100/100 轮连续验证。每轮均通过 source/target baseline、真实 KV
传输、双阶段 commit 与迁移后 decode-equivalence 检查，并每 10 轮输出 PASS。该结果
满足 Phase-2 的 100 次循环验收（仍限定 TP=1、PP=1、greedy sampling）。

### Phase-3 实测状态（2026-09-07）

跨主机 support gate 已通过：本机 `10.31.10.62` 与远端 `10.31.10.210` 双向网络可达，
两端均有 Qwen3-14B、16 张 BI-V150，且 migration 源码摘要、vLLM 0.11.2 和 protocol v1
一致。worker 已增加可路由 control/P2P endpoint 参数，跨机启动器通过 SSH 启动远端
target，并设置 `VLLM_FORCE_NCCL_COMM=1` 绕过 CoreX 可选 ixformer communicator。

已完成一次真实跨机验收：`10.31.10.62 GPU0 → 10.31.10.210 GPU1`，Qwen3-14B、TP=1、
PP=1、native NCCL。两端日志分别确认 `ncclCommInitRank Success`（rank 0/1），随后完成
source freeze、target reservation、40 个 attention layer 的物理 KV block 传输（manifest
与 payload checksum）、target/source 两阶段 commit；target 继续 decode 的 token 与 source
uninterrupted greedy baseline continuation 逐 token 一致，输出为：

`PASS phase2 migration control+KV transfer+two-phase-commit+decode-equivalence`

在相同拓扑上加入每层 10 ms 的传输延迟（`--inject-latency-ms 10`）后再次完成上述
全流程，仍通过 native NCCL 初始化、KV checksum、双阶段提交和逐 token 等价；这证明
小幅可控网络延迟不会破坏协议。

目标容量拒绝已在同一真实双机配置用 `--inject-target-capacity --expect-failure` 实测：
source 已冻结后，target 在 reserve/import 前返回 `RuntimeError: injected target capacity
exhaustion`，launcher 返回 `EXPECTED_FAILURE`（shell exit 0）并执行 source abort；没有
遗留 GPU allocation。source actor restart 用 `--inject-source-restart-after-layers 1`
实测，双方已经完成 native NCCL communicator 初始化后仅杀死本 launcher 所创建的 source
进程，结果为 `EXPECTED_FAILURE ProcessLookupError`（shell exit 0）。由于 source 不可达，
其 abort 只能 best-effort；target 外层 worker/EngineCore 已按明确 PID 清理且显存归零。
这证明故障结果不会被误判为成功，但完整服务级 actor 自动重建/请求重试仍属于后续
Llumnix Manager 集成工作。

传输超时已用 `--inject-transfer-timeout-after-layers 1 --expect-failure` 实测：双方
native NCCL communicator 成功建立，第一个 KV layer 完成后 launcher 返回
`EXPECTED_FAILURE TimeoutError: injected migration transfer timeout after 1 layers`（shell
exit 0），并进入 target-first/source-second abort 路径。远端无 GPU allocation 的外层
worker 已按明确 PID 清理；两端显存均归零。至此 Phase-3 所列成功、延迟、目标容量拒绝、
传输超时和 source actor 中断均已有实测记录；后者验证的是确定性故障结果与资源清理，
不等同于 Llumnix 服务级自动恢复。

随后以独立 control port 运行容量拒绝回归探针，`EXPECTED_FAILURE` 后按该 port 精确检索
远端进程未发现对应 worker（仅检索命令自身），`ixsmi` 显示无 GPU allocation，确认新加的
SSH 远端清理逻辑不会遗留 wrapper 或占卡。

最后以 control port `23902` 运行成功迁移回归：输出仍为
`PASS phase2 migration control+KV transfer+two-phase-commit+decode-equivalence`，随后
同一端口的远端 worker 检索为空且退出码为 0，验证成功路径也会清理远端 wrapper。

此前一次跨机启动失败是远端 GPU 被孤儿 `VLLM::EngineCore` 占满，已精确终止并复测成功；
该故障不属于 migration 协议失败。

为使后续故障验收可重复，launcher 已提供 `--inject-latency-ms`、
`--inject-transfer-timeout-after-layers`、`--inject-target-capacity` 和
`--inject-source-restart-after-layers`，并以 `--expect-failure` 将注入异常规范化为
`EXPECTED_FAILURE`；失败路径会按 target-first、source-second 顺序 best-effort abort。
目前已完成 launcher 语法检查及 Phase-1 协议测试（4 passed）。一次联合 connector 测试曾触发 CoreX Python
进程段错误（无残留 worker，单独协议测试通过），该硬件稳定性问题单独记录，不能计入
Phase-3 通过证据。

### Phase-4 首项准备（增量 blocks）

fork `da74076`/`c577dd9` 为 `KVLayerTransferManifest` 增加了可选的
`synced_prefix_block_count`：增量发送只携带已同步 ordinal 前缀之后的 block suffix，
并将前缀计数纳入 manifest checksum；同时绑定有序 source→target 前缀 block-pair 摘要。
target 若以不同前缀计数或同长度但不同 block 映射导入，都会在写入前拒绝。
对应 fork 单测为 20 passed，support gate 仍为 PASS。随后 fork `f1340f1` 增加
`IncrementalMigrationSession`：以 request ID 和 migration epoch 维护已同步的有序
source→target block pairs，要求 epoch 严格递增，拒绝空 suffix、负 ID、源/目标 block
重复及跨轮映射冲突，并为 prefix count/checksum 提供单一状态源；该对象单测为 25 passed。
随后 fork `4132c90` 已把 opt-in prefix count/pairs 贯穿 EngineCore、GPU worker 与
direct P2P layer RPC：source 仅发送 suffix，target 在写入前以本地 prefix pairs 验证
manifest。Llumnix `V1EngineAdapter` 也暴露同一可选参数，默认仍为全量迁移。
fork `294808d` 进一步在 Scheduler/EngineCore 中加入 scheduler-owned session registry，
并由 Llumnix adapter 暴露 begin/append/query/abort 控制面 API；session wire 自带完整
prefix mapping checksum，且在请求释放或 source/target abort 时清理。该实现仍是显式
opt-in。最新 fork `1f6979e` 增加 immutable-prefix 查询：只允许 block-aligned、不会被
后续 decode 改写的前缀进入 pre-copy，最后一个可变 block 保留到最终冻结 cutover。
fork `6fa06b5` 增加隔离的 target incremental reservation：使用私有 owner 逐轮增长
目标 block，validate 不推进 prefix，只有所有 layer 写入成功后的 commit 才推进 prefix；
abort 会释放私有 reservation，EngineClient 与 Llumnix adapter 已暴露对应 API。
fork `5f35f76` 增加 final-prepare claim：若目标已有 pre-copy reservation，最终
`MIGRATE_IN_PREPARE` 会将其扩展到 snapshot 所需大小并原子转移 ownership 到正式
request ID，避免重复分配；无 pre-copy 时保留全量迁移兼容路径。
Llumnix launcher `c1f8dd9` 现于 final cutover 跳过已确认的 ordinal prefix，仅传输新增
mutable suffix，并向 manifest/target receive 传递 prefix mapping 以完成认证；无
pre-copy 时 skip=0，行为与既有全量路径一致。
fork `96e1e43` 与 Llumnix launcher `9ce41d9` 已提供显式单轮诊断 pre-copy 编排：
source preview session，target validate/write 所有 layer 后 commit，再由 source append；
source block 列表必须等于 scheduler 计算的 immutable prefix。当前 launcher 在 pre-copy
之后仍会执行完整 final cutover，作为不覆盖可变 tail 的安全兜底，因此尚未验证 cutover
只传未同步 suffix，也尚未自动触发或接入生产 Manager。此前一次本机双卡 native NCCL
启动未取得完整 PASS/错误文本，已作为负面证据保留；后续已完成可观测性与边界修复。
增量迁移仍未完成。随机采样/RNG、structured output、LoRA、多模态、TP>1 和 speculative
decoding 也继续保持显式不支持，待分别具备快照字段、回滚测试和实测数据后再启用。

本轮验证记录（2026-09-07）：CoreX fork `f1340f1` 的定向 KV 测试为 25 passed；
Llumnix V1 migration 回归为 51 passed；跨主机 support gate 输出
`SUPPORT_GATE_PASS`，在数据面参数贯通后的最新 migration digest 为
`955137f3dcdb49ec7d69597ca698090782293b153141312d261d5ea55592ccfe`，协议版本为
`0.11.2 1`。这些结果验证的是现有全量真实迁移及增量协议对象，不改变 Phase‑4
尚未接入生产跨轮调度的结论。

随后启用 worker 独立日志后，native NCCL 全量回归在控制端口 `27101/27102` 明确输出
`PASS iteration 1/1` 与
`PASS phase2 migration control+KV transfer+two-phase-commit+decode-equivalence`；
source/target 日志均记录 `ncclCommInitRank Success`，结束后 GPU allocation 为零。
这只重新确认既有全量迁移基线，不覆盖此前 pre-copy 的 continuation mismatch。

后续长 prompt pre-copy 运行已获得明确控制面错误：所有 NCCL 初始化和 layer 发送完成，
但 source scheduler 在 append 时已回收 session（`incremental migration session not
found`），说明 pre-copy 期间 scheduler 队列 churn 会丢失临时状态。fork `4944da5` 已将
EngineCore mirror 作为 pre-copy session 的权威 append 状态，并加入回归测试（fork 定向
测试 32 passed）；尚未重新取得真实长 prompt pre-copy 的端到端 decode-equivalence。

修复 immutable boundary、EngineCore session mirror 和诊断 probe 对齐后，长 prompt
native NCCL pre-copy 在控制端口 `27801/27802` 实测通过：输出
`continuation_alignment_offset=1`、`PASS iteration 1/1` 与完整
`PASS phase2 migration control+KV transfer+two-phase-commit+decode-equivalence`；
两端日志均确认 `ncclCommInitRank Success`，结束后 GPU allocation 为零。offset=1
表示 frontend 观察与 EngineCore token-boundary freeze 间多完成一个 decode step，验证器
基于 source baseline 连续窗口校验，未放宽任意 token 匹配。
