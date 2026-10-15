"""Recover ACTUAL packed-token lengths from Astra convolution outputs.

Verified upstream construction: target patch_embedding first; history
clean_x_embedder called with 1x, 2x, 4x, then prepended in that order.
Final hidden order is [4x | 2x | 1x | target]. Do not guess layout from
RGB/latent frame counts: 3D padding/convolution changes token numbers.

No feature reuse across AR chunks. RoPE positions may change even for images
belonging to the same underlying historical frame.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


STREAMS = ("4x", "2x", "1x", "target")


@dataclass(frozen=True)
class Stream:
    name: str
    start: int
    stop: int
    t: int
    h: int
    w: int

    @property
    def count(self) -> int:
        return self.stop - self.start

    def frame_slices(self) -> list[tuple[int, int]]:
        spatial = self.h * self.w
        return [(self.start + i * spatial, self.start + (i+1) * spatial)
                for i in range(self.t)]


@dataclass(frozen=True)
class PackedLayout:
    streams: tuple[Stream, ...]
    total: int

    @classmethod
    def from_shapes(cls, shapes: dict[str, tuple[int, int, int]],
                    total: int) -> "PackedLayout":
        unknown = set(shapes) - set(STREAMS)
        if unknown:
            raise ValueError(f"Unknown history stream(s): {unknown}")
        if "target" not in shapes:
            raise ValueError("Target patch_embedding shape never observed")
        cursor = 0
        streams = []
        for name in STREAMS:
            t, h, w = shapes.get(name, (0, 0, 0))
            if any(i < 0 for i in (t, h, w)):
                raise ValueError(f"Negative shape: {name}: {(t,h,w)}")
            count = t*h*w
            streams.append(Stream(name, cursor, cursor+count, t, h, w))
            cursor += count
        if cursor != total or shapes["target"][0] <= 0:
            raise ValueError(f"Packed token count mismatch: measured={cursor}, DiT={total}")
        return cls(tuple(streams), total)

    def get(self, name: str) -> Stream:
        return next(s for s in self.streams if s.name == name)

    def signature(self) -> tuple[tuple[str, int, int, int], ...]:
        return tuple((s.name, s.t, s.h, s.w) for s in self.streams)

    def as_dict(self) -> dict[str, Any]:
        return {s.name: {"start": s.start, "stop": s.stop,
                         "shape": [s.t, s.h, s.w], "count": s.count}
                for s in self.streams}


class LayoutRecorder:
    def __init__(self):
        self.shapes: dict[str, tuple[int, int, int]] = {}
        self.main_seen = False

    def reset(self) -> None:
        self.shapes.clear()
        self.main_seen = False

    def capture_target(self, _module, _args, output) -> None:
        if self.main_seen:
            raise RuntimeError("Target patch_embedding called twice in one DiT forward")
        self.main_seen = True
        self.shapes["target"] = tuple(int(v) for v in output.shape[-3:])

    def capture_history(self, _module, args, kwargs, output) -> None:
        scale = kwargs.get("scale", args[1] if len(args) > 1 else "1x")
        if scale not in ("1x", "2x", "4x"):
            raise ValueError(f"Unknown FramePack scale {scale}")
        if scale in self.shapes:
            raise RuntimeError(f"FramePack {scale} projected twice within one forward")
        self.shapes[scale] = tuple(int(v) for v in output.shape[-3:])

    def layout(self, num_tokens: int) -> PackedLayout:
        return PackedLayout.from_shapes(self.shapes, num_tokens)
