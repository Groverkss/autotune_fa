from dataclasses import dataclass
from pathlib import Path

@dataclass
class TunerConfig:
    batch: int
    num_heads: int
    seq_len: int
    head_dim: int
    chip: str
    spec_template: Path
    transpose_v: bool
    artifact_dir: Path
    iree_build_dir: Path
    debug: bool = False
    dtype: str = "f16"

@dataclass
class RunnerConfig:
    seed: int = 7
    iree_benchmark_reps: int = 100
    validation_tol: float = 1e-1
    vmfb_file: Path = Path('attn.vmfb')
    func_name: Path = Path('attention')
    benchmark_file_prefix: Path = Path('attention_')

@dataclass
class KernelConfig:
    attn_tile_size: int
    waves_per_eu: int
