# 101601 服务器运行与未来风险实验指南

本阶段不重新训练 Astra / Wan2.1 backbone，只可能在后期训练 MLP gate。
在共享服务器绝不能自动选择 GPU，也不能停止其他用户的作业。

## A. 安装与路径（TODO 需手动填）

    cd World-Model/astra_inference_research/101601
    python -m pip install -e ".[test]"
    python -m pytest -q tests/
    cp configs/proposal_token.json configs/local.json

需补充：astra_repo、astra_checkpoint、wan_model_dir、condition_image，以及
garden_1 实际使用的 prompt。若实际使用了 pre-encoded condition.pth、不同
camera trajectory，请严格对齐之前实验设置，不可猜。
帧数单位：generation 中 frames_per_generation 等采用 latent frames。

    python -m astra_token_research.cli run --config configs/local.json --mode baseline --dry-run
    python -m astra_token_research.cli doctor --config configs/local.json

doctor 不实际加载模型，只检验路径和 SHA256 的上游源码；
需另外记录 checkpoint 权重 SHA256、模型加载 missing/unexpected keys。

## B. Phase 1：官方 No-Cache 等价

    nvidia-smi
    export CUDA_VISIBLE_DEVICES=YOUR_IDLE_PHYSICAL_GPU
    export PYTHONPATH=$PWD/src
    python -m astra_token_research.cli run \
      --config configs/local.json --mode baseline --out runs

若输出与之前 101501/官方 garden_1 同条件的 97 帧存在任何差异：
先检查 prompt、seed、FramePack 顺序、CFG、T5、scheduler、模态专家、权重。
**不要**直接把改动后的结果标成推理加速成功。

## C. Phase 2：真实 Token 稀疏性

    python -m astra_token_research.cli run \
      --config configs/local.json --mode token --out runs

默认只示范对 block 0/4 使用 token gate；先小规模运行，
确认 actual reported fresh/stale query rows，而不是整层 skipped calls。

对每个被选择重算的 token，sparse_block.py 的 self-attn Query,
cross-attn, FFN 只计算该行。但 K/V 和 MoE 仍计算全部 token。
建议记录 PyTorch profiler 或 CUDA event；trace 不用于主要速度比。

**特别验收**：同一输入 x，计算完整 DiT block 与稀疏 Q 子行，
验证输出对应行在 BF16/FP32 的容差内一致；CPU fake tests
不能代替这一步。

## D. Phase 3：Full teacher 的局部 proxy 数据

    python -m astra_token_research.cli run \
      --config configs/local.json --mode teacher --out runs

根据 metadata，保存 teacher_local_proxy.pt；此模式对观测的 block
依然全量计算，采集 x/previous residual/损失，不是推理加速模式。

    python -m astra_token_research.train_gate \
      --data /PATH/teacher_local_proxy.pt \
      --output /PATH/local_gate.pt --label local --budget 0.5

修改 local.json 中 token_cache.policy = learned，
model_checkpoint 指向 local_gate.pt 再 --mode token 推理。
这属于局部误差学生，并非 future-safe 监督。

## E. Phase 4：一个 token 对未来 AR chunks 的影响

先从 Phase 1 完整 baseline 的 metadata 读取真实 4x/2x/1x/target
token offset；生成有限规模反事实配置（不会运行 GPU）：

    python -m astra_token_research.oracle_plan \
      --baseline-metadata /BASELINE_RUN/metadata.json \
      --out configs/oracle_local \
      --blocks 0 4 6 --chunks 0 1 --steps 10 25 \
      --tokens-per-stream 2 --max-configs 24

逐个运行 oracle_manifest.json 中列出的 JSON 配置：

    python -m astra_token_research.cli run \
      --config /PATH/ONE_ORACLE_CONFIG.json --mode token --out runs

一个干预 = 指定 chunk/step/block 的一个 token 用上一步 residual，
其他所有 token 和 block 都做 full compute，然后继续完整 AR rollout。
不能将此实验误报为能加速的方法；它很昂贵，是收集 future-risk label。

从真实生成的 MP4 建标签（97 帧假设只适用于 3×8 latent 设置）：

    python -m astra_token_research.counterfactual \
      --baseline /BASELINE_RUN/metadata.json \
      --interventions /RUN_1/metadata.json /RUN_2/metadata.json \
      --rgb-per-chunk 32 --condition-frames 1 \
      --out /PATH/future_oracle.pt

以上对比测的是未来无缓存输出的 RGB MSE 偏差，不是 GT 预测误差。
至少要有多个不同 token，最好涵盖多个场景/动作/种子。
未来训练集和测试集按 scene 分开，不能只换 seed 冒充泛化。

    python -m astra_token_research.train_gate \
      --data /PATH/future_oracle.pt \
      --output /PATH/future_gate.pt --label future --budget 0.5

新代码执行前务必验证每个 metadata 中的源 SHA、权重、输入、种子。
只有通过训练+独立 test 后才能称其为 learned future-risk gate。

## F. 实验对照与明确未完成部分

对照：官方无 hook、101601 baseline、101501 whole-block K4、
Token heuristic、local learned、future-risk learned。
在真实 CUDA 上分别测 DiT、wall、显存、token mask/查询行数、
各 chunk 输出偏差与真实任务的 action following。

未完成：跨 AR chunk 的真实 hidden/KV 复用；end-to-end 多 chunk
soft gate 反向传播；RGB current-vs-last-observation 精确粗筛；
生产级 fused sparse kernel。不得用规则 horizon_boost 代替
proposal 的 future-safe token-level learned gate 结论。
