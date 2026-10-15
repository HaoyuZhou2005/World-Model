# 101601 研究方案：恢复 Proposal 的 Token 级 Dual-Horizon AdaptiveCache

依据：上传的 7 页 PPT《面向自回归 World Model 的 Dual-Horizon Adaptive Cache》（2026-09-08 修改）和 2026-10-16 对 101501 的本地反馈。

## 1. P0 纠偏：101501 的结果不等于 Proposal 的 Token Gate

原始 Idea：帧级变化粗筛，再进行 token importance 细筛，在两种时间条件下对每一个 token 或 token group 选择重新计算 / 复用旧特征。所谓 Future-safe 是复用后不应破坏后续 AR rollout 的稳定性，而不是仅当前帧像素接近。

101501 的算法是 BLOCK×DENOISING_STEP 的统一重算/复用和 CHUNK 边界 K 步全算，不能验证 token 门。K4 DiT≈1.81×，多 seed MAE≈8.16/17.49/11.01 的旧实验将仅作为整层参照，不构成停止 Token 工作的门槛。

### 101601 四条设计原则

1. 必须基于真实 Token Layout（FramePack 4x、2x、1x、target）而不是猜帧数。
2. token mask 必须产生真正的 selected-query 计算，不能全算再 blend。
3. Short condition = step/age/hidden change；Long condition = AR chunk、历史变化、后续 chunk 风险（初期 proxy，后续 learned）。
4. 跨 chunk 特征复用是另一技术难题，尚未实现，也不阻止 chunk 内 token gate 研究。

## 2. 真实 Astra 核心实现核查

上游仓库 EternalEvan/Astra。diffsynth/models/wan_video_dit_moe.py 的 WanModelMoe.process_input_hidden_states 首先处理 target，然后前插 1x、2x、4x，因此最终 hidden packed 顺序是：

    [ 4x history | 2x history | 1x history | current noisy target ]

不同 scale 经 Conv3D padding 后形状不同。layout.py 用上游 patch_embedding、clean_x_embedder 的 forward hook 读出实际 (T,H,W)，校验累计 token 数与 DiT block 输入长度相等。这是适配 FramePack，而不是任意重排 token。

DiTBlockWithMoE.forward：modulated norm1 -> MoE/Camera -> Self Attention -> Cross Attention -> MLP；原先默认对全部 token 计算。

sparse_block.py：
- Q 只投影选中的 M 行，K/V 仍对全部 S 行计算以保证 chosen query 的上下文；
- Q×K 的 attention 形状从 S×S 变为 M×S；
- text cross-attention、FFN 只计算所选行；
- 没有选中的 token 返回 previous residual + 当前 input；
- MoE 及部分 norm 仍做 full-token 计算；真实加速必须测量，M/S 不等于 FLOPs 比率。
- 目前仅 B=1、原 50 denoising steps、经过校验的 Camera CFG 0/2 branches、无 text CFG。

token_cache.py 严格按 CFG branch、block 和每个 token 的最近刷新 step 保存缓存；chunk 更换清除所有条目，故没有跨 chunk 拼接未对齐残差。

## 3. Proposal 页面对应实现

| PPT | 方法要求 | 101601 文件 | 完成程度 |
|---|---|---|---|
| 页2 | 缓存计算冗余、Future-safe 标准 | future_safety.py / counterfactual.py | 定义风险代理；真实性仍待验证 |
| 页3 | VLN/VLA token 复用及 AdaCache 相关方法 | 研究计划与对照 | 尚未移植全部文献代码 |
| 页4 | 轻量 MLP、训练软分数、推理硬选择 | train_gate.py / gate.py | 可用真实 labels 训练，但并非端到端 soft-gate rollout 训练 |
| 页5 | Frame 粗筛 + Token 细筛 | layout.py / gate.py | 帧组激活变化 proxy 与 token top-k；未实现 RGB 两帧直接对齐 |
| 页6 | 输入 token hidden → MLP → Recompute score | features.py / train_gate.py | 9 维 token 特征与 MLP |
| 页6 | fresh / stale 混合、预算防止全开 | sparse_block.py / token_cache.py / gate.py | hard token 选择可执行；soft training 不代表已完成 |

**PPT 中 SP-KV 是 next-token CE 范式；Astra 是 diffusion/flow 视频生成，不能照搬 CE。** 对门控可以训练局部 full-vs-stale residual 区别，也可以通过一次只 stale 一个 token 的反事实完整 AR rollout 测量它对未来 chunk 的影响。两者不是等价指标。

## 4. Token Gate 特征与两层筛选

输入至少包含：

- 短时：同一 chunk 上一 denoising step 的 hidden 相对变化、stale residual magnitude、上次重算 step age、当前 timestep；
- 长时：AR chunk index（目前只是 proxy）、历史尺度标签、未来风险 oracle labels（需要实际实验采样）；
- 结构：FramePack 4x/2x/1x/target token group one-hot。

粗筛：按每个 scale 的 temporal frame group 汇总 hidden activation change，大变化组强制全算；精筛：其余组按每个 stream 的 fresh ratio 对 token 重要性排序。此处粗筛是 hidden feature 的 step-level proxy，**不是已实现的 RGB current-vs-last-frame 对齐检测**。将来要接真实观察流、camera pose 和层内特征级 motion。

长时 horizon_boost 当前直接增大后续 AR chunk 的 token fresh 预算，属于 rule-based baseline，不是经过未来风险监督的学得策略。

## 5. Future-safe：实证风险而非视觉相似

先做一组官方 full/no-cache rollout。在相同 seed、scene、actions、prompt、weights 下，指定一个 (chunk, denoising step, block, token)，让它在这一次计算中使用上一步 residual，其他全部 fresh。保持该 AR rollout 剩余所有 denoising steps 和随后多个 chunk 的计算继续运行，观察被干预 token 在后续视频上的影响。

counterfactual.py 对实际 MP4 构造不同未来 chunk 的 MSE，给每个干预 token 赋 future-risk 标签；oracle_plan.py 在实际测得的 FramePack layout 内采样干预位置，不猜 token index。

这些 Future Risk Labels 是**与无缓存生成视频的长期偏差代理**，不等于真值预测性能或世界状态不变量。必须另外使用 GT 视频、真实 camera actions、motion/structure consistency、long-horizon benchmark 等指标验证是否真的“future-safe”。

train_gate.py：
- local label：来自 full-teacher 的局部 block residual MSE；
- future label：来自逐 token 反事实 rollout 的远期视频 MSE；
- MLP 输入同一 9-feature schema；在固定预算下学排序，推理 top-k 选择；
- 明确拒绝没有训练 checkpoint 就启用 learned gate；
- 当前是监督式 importance learning，**不是**跨所有 diffusion steps/chunks 反向传播的可微 future-safety loss；future_safety.py 只定义目标辅助函数。

## 6. 实验阶段与验收闸门

| Phase | 验收 |
|---|---|
| 0 | 运行 CPU tests；Astra hooks FramePack token counts/顺序可核验 |
| 1 | 官方 unhooked vs 101601 token instrumentation no-cache 97 帧逐像素 MAE=0 |
| 2 | 同一输入 x 下稀疏计算 selected rows 与 full block 数值相等（实测 RoPE+MoE+CFG） |
| 3 | 单场景 token 稀疏推理，统计 fresh query 比例、DiT CUDA time、wall、VRAM、各 chunk 输出偏差 |
| 4 | 单 token 反事实干预生成未来风险标签；核对未来 chunk 轨迹 |
| 5 | 训练 local proxy vs learned future risk 两种 MLP；检验 held-out scene 及预算；
| 6 | 公开 benchmark、多场景、多 seed、真实 action/camera，比较速度–长期稳定性 Pareto |
| 7（独立） | 如果要真正跨 chunk feature/KV reuse，先解决 FramePack/1x2x4x、RoPE、动作和目标 token 对齐 |

**Phase 3、4 不依赖于 101501 的整层调度必须击败 uniform i2+K4。**
Phase 7 不阻塞 Phase 3–6。

## 7. 消融与对照

1. official unhooked、101601 no-cache、101501 whole-block interval2 K4、101601 token heuristic、local MLP、future-risk MLP；
2. 4x/2x/1x/target 分别独立选择 fresh ratio 10%、25%、50%、75%、100%；
3. history only、target only、group only、token only、group+token；
4. first full steps、max token age、last step full；
5. 每个 block 的 token sensitivity（重点包含反馈中 block6）；
6. 长时 no chunk condition / chunk-rule condition / learned future label；
7. DiT CUDA vs 完整 wall、显存、FLOP 计数、token gate overhead；
8. future rollout 1、3、5+ chunks，seed 0/1/2，garden_1 到其他已核实的输入场景；
9. train-test scene 隔离（不能仅换 seed 就声称泛化）。

## 8. 边界与必要声明

当前 101601 是独立 Python 研究工程，已编写 CPU 测试但未在真实服务器实际运行。
FlashAttention selected-Q 不一定比 full attention 快；全量 K/V、MoE、dynamic top-k/nonzero、scatter 和额外显存可能抵消收益。
实测之前**不可宣称**速度倍数、Future-safe 达成、learned Gate 的多场景泛化，或跨 chunk feature reuse 已完成。
