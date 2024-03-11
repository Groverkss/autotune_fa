from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkConfig:
    batch: int
    num_heads: int
    seq_len: int
    head_dim: int
    transpose_v: bool
    dtype: str = "f16"

@dataclass
class TunerConfig:
    chip: str
    spec_template: Path
    artifact_dir: Path
    iree_build_dir: Path
    debug: bool = False

@dataclass
class RunnerConfig:
    seed: int = 7
    iree_benchmark_reps: int = 100
    validation_tol: float = 1e-1
    vmfb_file: Path = Path("attn.vmfb")
    func_name: Path = Path("attention")
    benchmark_file_prefix: Path = Path("attention_")


@dataclass
class KernelConfig:
    block_n: int
    block_m: int
    num_warps: int
    waves_per_eu: int
    # 1 --> 16x16x16
    # 2 --> 32x32x8
    layout: int
