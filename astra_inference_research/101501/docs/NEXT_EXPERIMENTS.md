# 101501 服务器实验执行指南（严格依赖审查报告）

## 0. 前提

- 环境：按上传 \`docs/要求.txt\`，conda \`adaptivecache\`，
  Python 3.10.21 / torch 2.6.0+cu124；本次不会重装、修改这些环境。
- 共享机：**每次启动前人工检查 \`nvidia-smi\`**，选当时闲置的物理 GPU；
  CUDA_VISIBLE_DEVICES 限制到**单卡**，绝不停止别人进程。
- 可选 \`taskset -c 80\` 等小 CPU 亲和性，选卡与选核不可默认替其他用户决定。
- 本阶段训练次数=0。无需从头训练 Wan/Astra。
- **TODO_PATHS**：Astra clean clone / Astra ckpt / Wan dir / condition image /
  prompt / evaluation dataset。填写本机私有配置；不要提交绝对路径和权重。
- 对于旧 \`26091001\` 没有 .git 的情况，见 identity 工具的源码哈希替代证据。

## 1. 先验收代码

进入 \`101501\`，确保优先加载其自己的模块（避免 101401 名称冲突）：

~~~bash
cd World-Model/astra_inference_research/101501
python -m pip install -e ".[dev]"
python -m pytest -q tests/
python -m astra_research.cli run --config configs/scheduled_example.json --mode baseline --dry-run
~~~

项目内同名包 \`astra_research\` 版本只能安装一个，切换 101401 前注意 editable install；
建议使用 \`PYTHONPATH=$PWD/src\` 明确优先级。

## 2. 生成配置及核对文件

~~~bash
cp configs/scheduled_example.json configs/local.json
# Edit ALL TODO_* values; use the exact garden prompt and camera trajectory.
python -m astra_research.cli doctor --config configs/local.json
# Check block count from actual checkpoint; prior garden run used 30, do NOT assume universal.
python -m astra_research.study configs \
    --template configs/local.json --out configs/sweep_local --num-blocks 30
~~~

本机 \`configs/local.json\`、\`configs/sweep_local/\` 请保密。
尤其不要 Git commit 模型路径和私人数据。

使用 identity 工具一次性计算哈希：

~~~bash
python -m astra_research.identity \
 --astra-repo /TODO/Astra \
 --astra-ckpt /TODO/Astra/checkpoints/diffusion_pytorch_model.ckpt \
 --wan-dir /TODO/Wan2.1-T2V-1.3B \
 --out configs/identity_local.json
~~~

这些权重总量接近数十 GB，首次 SHA-256 会产生大量 I/O，避免同时运行多任务。

## 3. 在共享卡上运行

~~~bash
nvidia-smi                  # MANUAL CHECK: do not select a busy GPU.
export CUDA_VISIBLE_DEVICES=TODO_IDLE_PHYSICAL_GPU
export PYTHONPATH=$PWD/src
# Example: taskset -c 80 is ONLY an example, pick an actually available CPU core.
python -m astra_research.cli run --config configs/sweep_local/baseline.json --mode baseline
python -m astra_research.cli run --config configs/sweep_local/uniform_i2_k1.json --mode cache
python -m astra_research.cli run --config configs/sweep_local/uniform_i2_k2.json --mode cache
python -m astra_research.cli run --config configs/sweep_local/uniform_i2_k4.json --mode cache
~~~

随后比较新旧结果的 DiT ms、wall s、reuse calls 和 MAE/chunk。
如果 no-cache 与官方 output 有任何像素差，停止 cache 实验定位
种子、prompt、权重、CFG、帧长、版本不一致等问题。

## 4. 逐 chunk 质量诊断

~~~bash
python -m astra_research.study videos \
 --reference /TODO/RUN_BASELINE/generated.mp4 \
 --candidate /TODO/RUN_CACHE/generated.mp4 \
 --rgb-per-chunk 32 --chunks 3 --condition-frames 1
~~~

这套配置的 97 帧假设仅适用于这份 8-latent/chunk 的官方示例输出；
其他视频必须实际确认帧长，而不能假定公式永远成立。

## 5. 速度报告规范

同一场景/seed：交错运行 baseline, cache, cache, baseline
并至少重复数个 cycle；不要并发抢卡。需报告：

- CUDA DiT time (ms) — 不是完整推理耗时；
- wall time (s) — 包括加载+编码+解码+存储；
- reused block calls / total selected calls，peak VRAM；
- per-chunk output MAE/PSNR relative to no cache（不是视频质量）；
- independent benchmark: 多样本、多 seed、camera GT / motion metrics；
- 单卡空闲证明及权重、源码完整性记录。

\`python -m astra_research.study pairs\` 要求多条互相配对的已完成 metadata，
且当上游 \`git SHA\` 不存在时，必须拥有本项目生成的源文件 SHA 哈希。
不能用 profiler trace 做速度表；不能用首个冷启动比值代替论文数字。

## 6. 判断是否继续研究新 cache

如果 protect key layers / progressive K 的质量—速度 Pareto
不优于 uniform i2+K4，就不要因为方法名字更复杂而默认它更好；
改为分析决定哪些层敏感（per-layer ablation）。
如果多个场景都稳定出现新 Pareto 点，再研究 adaptive gate。
直到完成 FramePack/rope/action 位置对齐验证，跨 chunk 仅能做
强制刷新控制，不得直接在目标 token 上复用上一 chunk residual。
