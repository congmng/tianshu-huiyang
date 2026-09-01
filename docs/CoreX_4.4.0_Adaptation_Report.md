# Llumnix 在 Iluvatar CoreX 4.4.0 上的适配与验证报告

验证时间：2026-09-01

持续进度记录见 [Progress_Log.md](./Progress_Log.md)。本阶段提交包含 Python 3.12 包元数据修复、Ray 2.52 测试兼容和 V1 KV 事件亲和索引基础。

## 结论

CoreX 驱动、PyTorch、Ray 和 Llumnix 的非 vLLM 控制面已在本机验证可用，且整个过程未修改驱动或系统级 CoreX 安装。

Llumnix 当前仓库的 vLLM 后端**不能直接运行**在所给软件栈的 `vllm 0.11.2+corex.4.4.0` 上。原因是该后端为 vLLM `0.6.3.post1` 的内部 API 实现；vLLM 0.11.2 已迁移到 V1 架构并移除了多个被直接引用的私有模块。完整启动 Llumnix API Server、vLLM Engine 和 KV Cache 迁移，需要针对 vLLM V1 的后端重构，不能仅通过改依赖版本完成。

因此当前状态是：**CoreX 4.4.0 运行时、Llumnix 调度控制面及 vLLM V1 单实例请求链路已可运行；KV-cache 迁移数据面仍未适配。**

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

由于当前 Llumnix 的 `setup.py` 限制 Python 为 `>=3.9,<3.11`，而已完整可用的本机 CoreX 环境是 Python 3.12，验证通过项目目录运行源码（不以 package metadata 形式安装）。软件栈也提供 Python 3.10 wheels；不过它们的 vLLM 版本同样是 0.11.2，因此不能解决下述 API 断层。

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

当前仓库明确锁定的是 vLLM 0.6.3：

```text
requirements/requirements_vllm.txt: vllm == 0.6.3.post1
```

CoreX 4.4.0 软件栈中提供的是 vLLM 0.11.2。导入旧后端的实测失败如下：

| Llumnix 模块 | 实测失败 | 含义 |
| --- | --- | --- |
| `llumnix.backends.vllm.llm_engine` | `ImportError: _AsyncLLMEngine` | 0.11.2 不再导出旧异步 Engine 私有类 |
| `llumnix.backends.vllm.executor` | `ModuleNotFoundError: vllm.executor` | V1 重组/移除了旧 executor 包路径 |
| `llumnix.entrypoints.vllm.api_server` | `ModuleNotFoundError: vllm.model_executor.layers.sampler` | 旧 sampler 路径已不存在 |

受影响的代码不仅是导入名，还包括从旧 scheduler、block manager、worker、sequence 和 engine 生命周期继承并覆写的实现。因此，不应将这几个 import 用空兼容层屏蔽；那样会让 API Server 启动却无法保证推理正确性或 KV Cache 迁移安全性。

## 后续完整适配范围

要使 Llumnix 在该 CoreX 栈上提供真正的多实例 vLLM 服务，需要完成：

1. 将 `llumnix/backends/vllm/{llm_engine,executor,worker,scheduler,sequence}.py` 迁移到 vLLM 0.11.2 V1 公开/当前内部接口。
2. 重新实现或验证 KV Cache block 导出、预分配和传输；这是 Llumnix 的关键功能，不能只保留请求分发。
3. 更新 `entrypoints/vllm/{client,arg_utils,api_server}.py` 的 AsyncStream、采样输出和 EngineArgs 对接。
4. 用一个本地文本生成模型完成单实例生成、两实例调度、迁移、故障恢复的端到端验证。
5. 如要正式支持现有 Python 3.12 CoreX 栈，评估并更新 `setup.py` 的 Python 上限；该修改应与 vLLM V1 迁移一起完成，而不是单独放宽声明。

在完成上述工作前，建议把本报告中的结果视为运行时与控制面适配验证，而不是“Llumnix 多实例 vLLM 服务已可用”的结论。
