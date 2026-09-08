# Dual-Horizon Adaptive Cache for Autoregressive World Model

## Overview

This project explores adaptive inference acceleration for autoregressive (AR) World Models. The backbone is **Wan2.1-1.3B**. The goal is not only to reuse previous computation, but to learn whether reuse is safe for future rollout stability.

The core idea:

```
Observation sequence
        |
        v
Frame-level screening
        |
        +----------------+
        |                |
   Large change      Small change
        |                |
 Full recompute    Token-level Gate
                         |
              +----------+----------+
              |                     |
          Recompute              Reuse cache
```

## Motivation

Traditional cache methods usually judge whether two inputs are similar. However, in World Models, visual similarity does not always mean future prediction consistency. A small visual region can still affect long-term rollout.

Therefore, this project introduces a future-aware cache strategy:

- Frame-level coarse filtering reduces unnecessary token evaluation.
- Token-level learned gate predicts whether recomputation is needed.
- The gate is trained through future prediction loss.

## Architecture

```
Adaptive-WM-Cache
|
|-- models/
|   |-- wan/                  # Wan2.1 wrapper
|   |-- world_model/          # AR world model interface
|   |-- cache/                # cache controller and gates
|
|-- datasets/                 # video dataset loader
|
|-- training/                # gate training
|
|-- inference/                # rollout with cache
|
|-- configs/                 # experiment configs
```

## Backbone

Backbone:

- Wan2.1 1.3B
- Autoregressive video generation / world model setting

The first implementation should keep Wan2.1 unchanged and add a wrapper layer.

```
Wan Model
    |
    v
WorldModel Wrapper
    |
    v
Cache Controller
```

## Dataset

The dataset follows Astra's setting.

Dataset pipeline:

1. Download Astra repository.
2. Follow Astra dataset preparation instructions.
3. Convert videos into sequential frames.
4. Use frame sequences:

```
(frame_t-k,...,frame_t)
        |
        v
predict frame_(t+1)
```

Dataset loader will be implemented in:

```
datasets/video_dataset.py
```

## Implementation Roadmap

### Stage 1: Wan2.1 AR World Model

Goal:

Run original Wan2.1 inference.

File:

```
models/world_model/ar_world_model.py
```

Need to implement:

- frame encoder
- token extraction
- transformer forward
- future frame decoding

### Stage 2: Cache Manager

File:

```
models/cache/cache_manager.py
```

Responsibilities:

- store previous hidden states
- retrieve stale features
- update cache

### Stage 3: Frame Gate

File:

```
models/cache/frame_gate.py
```

Input:

```
current frame
previous frame
```

Output:

```
change score
```

A simple baseline:

```
mean(abs(current-previous))
```

Large score:

```
full computation
```

Small score:

```
token gate
```

### Stage 4: Token Gate

File:

```
models/cache/token_gate.py
```

Input:

```
token hidden state
```

Network:

```
Linear
ReLU
Linear
Sigmoid
```

Output:

```
recomputation score in [0,1]
```

Interpretation:

```
gate≈1 : recompute

gate≈0 : reuse cache
```

### Stage 5: Feature Fusion

For token i:

```
h = gate * h_fresh + (1-gate) * h_stale
```

This allows differentiable training.

## Training Strategy

Freeze Wan2.1 first.

Only train gate network.

For each training sample:

1. Full computation produces reference prediction.
2. Cache computation uses gate.
3. Compute future prediction loss.
4. Backpropagate into gate.

Objective:

```
min L(predicted future, ground truth)
```

Later, joint fine-tuning can be explored.

## Evaluation

Compare:

- Full Wan2.1
- Fixed cache ratio
- Random cache
- Adaptive cache

Metrics:

Efficiency:

- latency
- FPS
- FLOPs

Generation quality:

- FVD
- LPIPS
- PSNR
- SSIM

Long rollout stability:

```
1 step
5 steps
10 steps
20 steps
```

## Important TODO

The following parts require further engineering:

1. Locate exact hidden-state insertion point in Wan2.1 transformer.
2. Modify attention/block forward function to accept cached features.
3. Design cache index mapping between adjacent frames.
4. Determine threshold strategy for hard gate inference.
5. Reproduce Astra dataset preprocessing exactly.
