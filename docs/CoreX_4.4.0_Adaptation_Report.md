# Llumnix 在 Iluvatar CoreX 4.4.0 上的适配与验证报告

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

CoreX 驱动、PyTorch、Ray 和 Llumnix 控制面已验证可用，整个过程未修改驱动或
系统级 CoreX 安装。

Llumnix 的旧 vLLM `0.6.3.post1` 私有 block-manager 后端不能直接运行于
`vllm 0.11.2+corex.4.4.0`；项目已为此提供 V1 adapter，而非用空兼容层掩盖
旧接口。当前正式验证范围为：**Python 3.12/CoreX 4.4 的 V1 单实例 API、
Ray/Llumlet 队列桥接、KV event/hash affinity、GlobalScheduler 缓存亲和调度，
以及两机 Qwen3-14B P/D KV handoff 均可运行。**

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
