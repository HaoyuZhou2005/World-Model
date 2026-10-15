# 对 2026-10-15 上传实验包的审查（代码 + 结果 + 后续规划）

审查来源：用户上传的 `26101501.zip`（顶层 `26101501/`，含 `总结.md`、源代码、实验配置、logs、86 份 metadata，**零份 MP4**、9 组 20261014 实验、权重 SHA-256 清单）。
同时核对了 GitHub 原版 `astra_inference_research/` 的 15 个文件，以及公开 Astra 的已知推理结构。
**结论分三类：已验证、可疑/不充分、尚未实现。**

## A. 已有结果：可信的范围与含义

固定展示场景为 `garden_1`、seed=0、Astra/Wan2.1-1.3B、
3 chunk × 8 latent、50 denoising steps、相机 CFG 关闭。
**以下数值来自上传实验日志/摘要，并非在本次任务中重跑 GPU、或从视频独立复算得到。压缩包实际包含 86 份已完成的 metadata.json、0 份 .mp4（总结文档称含有视频，与实际附件不符）。所有 MAE 均属于历史报告数字，不是本次复核的测量。**

| 方法 | DiT CUDA 时间 / 相对无 cache | MAE 相对无 cache 视频 | 逐 chunk MAE | 科学判断 |
| --- | --- | --- | --- | --- |
| no-cache | ≈138.6 s / 1.00× | — | — | 当前对照 |
| target interval-2, 全层 | ≈71.2 s / **1.95×** | **10.40** | 5.59, 11.20, 14.63 | DiT 真正明显减时，误差随 AR 累积 |
| interval-2 + K=2 | ≈73.9 s / **1.88×** | 8.95 | 5.89, 9.08, 12.06 | 精度–速度折中 |
| interval-2 + K=4 | ≈76.8 s / **1.80×** | 8.10 | 5.24, 7.51, 11.72 | 精度–速度折中 |
| early residual (blocks 0/4) | 约 0.97× 墙钟中位 | 约 10–14 | 取决配置 | 无稳定端到端收益 |
| history K/V stitch | 约 **1.01×** 墙钟中位（2次） | 9.43 | 未提供 | 无稳定加速证据 |
| target L1 gate (threshold 0.05–0.25) | 快但跨多步复用老 residual | 15.6–17.4 | 未提供 | 不是质量相当的优选 |

上传证据路径：
- `experiments/20261014_official_vs_residual/SUMMARY.md`
- `experiments/20261014_stage_b/SUMMARY.md`
- `experiments/20261014_kv_attn_v2/SUMMARY.md`
- `experiments/20261014_target_step/SUMMARY.md`
- `experiments/20261014_target_l1_sweep/SUMMARY.md`
- `experiments/20261014_dual_horizon_k/rows.json`

### Stage A：基线等价

官方原始推理视频与无缓存 instrumented wrapper，在一次 garden 运行中 **97 帧逐像素 MAE=0**。
这是强的集成正确性证据，但不意味着所有数据、模态、CFG 设置都覆盖。

### Stage C/D：跨 chunk 边界

`experiments/20261014_stage_c_framepack_map/overlap_report.json` 显示：
chunk 开始历史长度 1/9/17；邻段虽有帧重叠，但 FramePack 槽位和 RoPE
发生变化，记录中的 `same_stream_and_rope` 为空。
1x history spatial overwrite 能改输出、但不减少 150 次 DiT forward。
**禁止**将此方法表述为跨 chunk 加速，更不能直接跳整个 block。

## B. 发现的问题和风险（严重性分级）

### P0：需先于论文结论修正

1. **训练/质量概念混淆**：MAE/PSNR 对比对象是无缓存生成视频，不是真实未来帧；
   这些是 cache-induced *output deviation*，不是 FVD、VBench、运动保真、
   camera following 或 world-model accuracy。
2. **速度口径混淆**：DiT CUDA ≈1.95× 不等于端到端视频 1.95×。
   生成过程包含加载、VAE、IO；garden 单场景、单次/两次测量，不能用于发表表格。
3. **“双时间尺度”的命名需要限缩**：K 只在每个 chunk 初期强制全算；
   没有复用跨 chunk 的 feature，不能声称实现跨 chunk feature cache。
4. **缺少可移植的源码标识**：元数据中的 `astra_revision=UNKNOWN_GIT_REVISION`；
   上传的 `WEIGHTS.json` 有权重 SHA-256，但并不能替代上游源码 SHA。
   101501 新增 source SHA-256 manifest / weights manifest 程序。
5. **单场景=不可推断泛化**：所有现有结果都基于 garden_1、合成相机轨迹；
   `use_real_poses=false`；不能得出真实 camera-control quality 结论。

### P1：代码审查问题（必须处理或隔离）

6. 旧 `report.compare()` 以完整 `config_fingerprint` 比较；
   baseline/cache 的 cache config 不同导致自然 fingerprint 不同；
   原逻辑特判一个 mismatch 为可比，容易掩盖真实未核实的 weights/source。
   101501 用字段级规范化比较，并 fail-closed 处理未记录上游 SHA 的情况。
7. 旧 `paired_video_metrics()` 在两视频帧数不等时有截断风险；
   101501 新严格评估要求精确帧数、精确尺寸、明确初始条件帧。
8. 旧 `target_l1` 累积门控在 `max_age=7` 下会连续跳很多步，
   MAE 快速上升；本阶段**不继续**声称简单 L1 阈值是合理自适应策略。
9. 旧 `warmup_steps` 是统一标量、层间 skip 强耦合；
   101501 增加 **per-block refresh period** 和 **per-chunk warmup**，
   可做“保留敏感层”消融。属于新研究假说，不是已验证结果。
10. 旧 KV stitch/freeze 最终生成视频相同值得核实算子计算图、
    性能瓶颈与是否在不同实现分支触发；只有约 1.01× 中位墙钟，
    暂不将 KV 视作主线加速。
11. Astra 官方 infer_demo 使用 `load_state_dict(strict=False)`，
    潜在 missing/unexpected keys 需要现场完整记录。也应核实
    `moe_num_experts=3`、`moe_top_k=1` 与 checkpoint 结构实际匹配，
    不要仅凭生成了视频认为权重全部正确加载。
12. 上传的 `machine.py` 含特定服务器路径、A100-40GB 限制，
    不适合直接当通用代码。新实验只携带 placeholder 配置；
    使用机器前显式查 `nvidia-smi`，**绝不自动占用另一张卡**。

### P2：科研报告不充分

13. temporal flicker 代理是绝对帧差；数值下降也可能由画面变糊造成，
    不是“生成更稳定”的充分证据。
14. 跨 chunk 的 pooled cosine≈0.99 与 token 对齐无关；
    在 RoPE/FramePack 条件下不能直接拿来做 token skip。
15. garden 视频 97 帧由 1 个条件帧 + 3×32 个 RGB 帧组成；
    在其他模型 VAE 设置下**不能写死**这个分段公式，需要校验。
16. 测速必须考虑热身、排序效应、相邻作业的 GPU/CPU 干扰；
    可采用 ABBA 交错重复，并同时报告 DiT 和 wall 指标。
17. 旧 `101401` 原版功能简单，不能直接代表本次上传实验包中的
    `target_cache.py`、`kv_cache.py` 等后续本机开发内容。
    101401 **严格保存 GitHub 原版**，没有假装包含本机实验拓展。

## C. 101501 的方案与已编写的代码

1. `scheduled_cache.py`：仅同 chunk 跨 denoising step 的 block residual reuse；
   支持 layer-specific period；p=1 不跳层；支持 per-chunk 强制重算 K；
   CFG branch 单独存；每 chunk 清空；形状/精度/设备不符全算。
2. `instrument.py`：对接官方 DiT block hook 和 50-step 调用次序（沿用 GitHub）。
   本阶段新方法设置 `cache.method=scheduled`，不再使用危险跨 chunk overwrite。
3. `experiment_plan.py`：创建 baseline / interval-2 / K=2 / K=4 /
   protect first+last / half-block / progressive-K 的消融矩阵。
4. `evaluate.py`：逐 chunk 视频输出偏差与 flicker proxy；严格检验
   RGB 帧数量和尺寸，始终注明参考是无缓存生成视频。
5. `benchmark.py`：校验相同 seed、生成参数、源码版本、
   forward 数量、Profiler 状态；对多次配对输出中位数。
6. `identity.py`：本地生成 Astra 三份核心源码与 Wan/Astra 权重 SHA-256。
7. `tests/`：CPU synthetic cases；不能代替上游 Astra 的 GPU 集成测试。

## D. 当前不能声称完成的部分

- 101501 新代码**尚未在你的 A100/真实 checkpoint 上运行**。
- 上传实验包实际有 **86 份完成的 metadata、0 份视频**，并非 86 个视频；没有把原始大量日志提交 GitHub；
  源档 `26101501.zip` 保留于对话附件。
- 未实现跨 chunk 位置对齐后的真实计算复用；
  没有 token/frame learned gate，**无训练**。
- 没有多数据集或独立 camera GT 的 FVD/VBench/相机轨迹 benchmark。
- 没有跨多个场景/seed 的统计显著结论，也没有达成可发表验证。

请先执行 `tests` + **纯官方 → 101501 no-cache** 等价测试，
再执行消融矩阵；不要将“有代码”当成“GPU 已验证”。

## E. 实施次序

**Phase 0**：固定 Astra 源码文件 SHA、权重 SHA、输入；建立私有路径配置。
**Phase 1**：官方与 101501 no-cache 固定 seed 双次 MP4 一致性。
**Phase 2**：复现 interval-2 + K=1/2/4 的 MAE–DiT Pareto。
**Phase 3**：逐层敏感度/保留关键层，验证是否改善 K4 Pareto。
**Phase 4**：多场景、多 seed、单卡 ABBA 顺序及逐 chunk 报告。
**Phase 5**：只有明确的新机制后才开始跨 chunk token map/caching 或可学习 gate。

实验假说而不是结果：分层策略有可能保护重要层而提高长期生成保真；
其效果必须以 GPU/视频数据证实，不能从旧结果推定。
