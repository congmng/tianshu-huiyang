# CoreX 4.4/4.5 异构调度实验设计

本文档定义天垓 150（CoreX 4.4/V150）与天垓 300（CoreX 4.5/V300）纳入统一调度框架后的论文实验。当前 4.5 vLLM 0.25.1 不包含 `vllm.v1.migration`，因此跨 4.4/4.5 只验证普通请求调度；真 KV 迁移只在协议、KV layout、设备类别和能力集合完全兼容的实例之间进行。

## 1. 实验矩阵

固定模型、tokenizer、采样参数和请求 trace 后，至少比较以下系统：

| 配置 | 调度 | 真 KV 迁移 | 设备 |
| --- | --- | --- | --- |
| A | 单实例 FCFS | 否 | V150 |
| B | 两实例 round-robin | 否 | V150 |
| C | Llumnix 负载调度 | 否 | V150 |
| D | Llumnix + KV affinity | 否 | V150 |
| E | Llumnix + V1 真 KV | 是 | V150/V150 |
| F | 异构统一调度 | 否 | V150/V300 |
| G | 同栈异构实例 | 是 | 仅能力兼容组 |

配置 E 用本机两个 V150 实例，配置 G 只在真实 capability gate 通过后执行。不得把 V150/V300 的普通请求调度结果作为跨设备真 KV 迁移结果。

## 2. 自变量

- 输入长度：`128/512/2048/8192` tokens；输出长度：`32/128/512` tokens。
- 并发度：`1/4/8/16/32`，每组预热后运行固定时长或固定请求数。
- 迁移时机：prefill 完成后、生成 `25%/50%/75%` 时，以及无迁移对照。
- 负载分布：均匀、长短混合、突发、单实例过载。
- 设备组合：V150/V150、V300/V300、V150/V300。
- 传输方式：同栈 native NCCL；CPU/ZMQ 仅作为功能和故障对照，不与 NCCL 性能结果混合。

每个格子使用固定随机种子，至少重复 5 次；报告均值、标准差和 95% 置信区间。请求 trace、模型版本、CoreX release marker、vLLM/Torch/Ray 版本和 git fingerprint 必须随结果保存。

## 3. 采集指标

### 服务指标

- TTFT、ITL/TPOT、端到端延迟：P50/P95/P99。
- 请求完成率、request/s、input/output tokens/s。
- SLO 违约率、排队时间、实例选择次数和 KV-affinity 命中率。
- 每卡显存使用、KV cache 容量、利用率、功耗和温度。

### 迁移指标

- prepare/freeze、snapshot、KV export、传输、import、target commit、source release 的分段耗时。
- 迁移字节数、bytes/token、有效带宽和请求暂停时间。
- 迁移成功率、abort 次数、retry 次数、永久失败次数。
- 迁移前后 token 序列：lost、duplicate、顺序错误和 continuation 对齐偏移。
- 迁移时源/目标显存 reservation 与释放时间，确认无泄漏。

## 4. 正确性与故障实验

每次迁移都保存 request snapshot、迁移 epoch、协议版本、KV layout 和 capability 集合。验证 greedy continuation 逐 token 一致；随机采样验证 RNG witness；结构化输出、LoRA、prompt embedding 和多模态分别作为独立实验，不能用纯文本结果代替。

故障注入至少包括：

1. target 容量不足：目标拒绝，source 解冻，请求继续完成。
2. 单层传输超时：target-first/source-second abort，下一轮可重新配对。
3. source/target actor 在传输期间退出：请求不应被静默标记为已迁移，记录恢复或失败原因。
4. 连续两轮迁移：epoch 单调递增，不重复释放、不丢请求。
5. V150/V300 能力不匹配：普通请求可调度，真 KV pair 必须为空。

## 5. 验收门槛

- 所有配置先通过版本、设备、代码 fingerprint 和 migration capability gate。
- 功能验收：服务级请求完成率 100%，成功迁移无 token duplication/loss，失败迁移后请求仍可完成。
- 资源验收：每轮迁移结束后 source/target reservation 和 GPU 显存回收；无残留 Ray/EngineCore 进程。
- 性能结果只在同一模型、同一请求 trace、同一采样设置下比较；功能 PASS 不等于性能收益成立。
- 4.5 只有在真实出现协议版本 1、兼容 KV layout 和完整能力集合后，才允许加入真 KV 迁移实验；在此之前只报告异构普通调度。

## 6. 当前证据状态

- CoreX 4.4/V150：原生推理、V1 HTTP、两实例服务级真 KV 迁移和直接 Manager NCCL 迁移已通过。
- CoreX 4.4/V150：Manager 真实两实例 retry E2E 已通过，首次注入瞬态失败后第二次迁移成功，输出 `retry_attempts=2` 和 continuation 对齐偏移 0。
- CoreX 4.5/V300：vLLM 0.25.1 单实例 Manager/Llumlet HTTP 服务已通过；`vllm.v1.migration` 缺失，真 KV migration 未适配。
- 混合栈：能力门禁已识别 4.4 protocol 1 与 4.5 protocol 0；当前还存在两端源码 fingerprint 不一致，统一调度部署验收尚未完成。
