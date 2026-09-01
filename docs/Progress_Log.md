# Llumnix CoreX 4.4.0 适配进度

## 2026-09-01：Python 3.12 与 V1/KV 亲和基础

- 将 `setup.py` 的 Python 版本范围扩展到 `>=3.9,<3.13`，并补充 Python 3.11/3.12 classifiers，匹配 CoreX 4.4.0 本机 Python 3.12 环境。
- 将 vLLM extra 依赖切换到 `>=0.11.2,<0.12`，Ray 下限调整为 `2.52.1`；旧 vLLM 0.6 后端仍保留在源码中，但仅由旧版本运行时选择。
- 修复 Ray 2.52 测试夹具对已删除 `ray._private.utils.hex_to_binary` 的依赖。
- 新增 `llumnix.backends.vllm.v1_kv.KVCacheAffinityIndex`：消费 vLLM V1 KV 事件，按实例维护 block ownership，并提供 prefix affinity 与候选实例排序。
- 新增 V1 KV 亲和单元测试；当前测试结果：12 passed（包含既有全局调度测试）。
- 已确认 CoreX/IX-ML/Driver 仍为 4.4.0，未修改驱动。

## 当前边界

KV 亲和索引已经可以独立消费 V1 事件，但尚未把物理 GPU block 的导出、跨主机传输和目标实例导入接入 vLLM V1 scheduler。下一阶段将实现 V1 KV connector/transfer executor，并在本机与 `10.31.10.210` 分层验证；在此之前不宣称跨实例迁移完成。

