import os
import subprocess
import re
import numpy as np
import torch
import logging
from config import TunerConfig, KernelConfig, RunnerConfig, BenchmarkConfig
from spec_gen import SpecGen
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger("[fa]")
logger.setLevel(logging.INFO)


def execute_command(command, output_file=""):
    """Executes the given command and logs the output to the given file."""
    logger.info("Executing command: " + " ".join(command))
    out = None
    err = None
    if output_file != "":
        with open(output_file, "w") as f:
            process = subprocess.Popen(command, stderr=f)
            process.wait()
    else:
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        out, err = process.communicate()
        process.wait()
    return out, err

class Runner:
    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
        tuner_config: TunerConfig,
        kernel_config: KernelConfig,
        runner_config: RunnerConfig,
    ):
        self.benchmark_config = benchmark_config
        self.tuner_config = tuner_config
        self.kernel_config = kernel_config
        self.runner_config = runner_config

        spec = SpecGen(self.tuner_config.spec_template).render(self.kernel_config, self.benchmark_config)
        query_shape = self.get_shape_data()
        self.spec_file = "spec_" + self.runner_config.benchmark_file_prefix.stem + query_shape + ".mlir"
        with open(self.spec_file, "w") as f:
            f.write(spec)

    def get_vmfb_file(self):
        return f"{self.tuner_config.artifact_dir}/{self.runner_config.vmfb_file}"

    def get_dims(self):
        config = self.benchmark_config
        return config.batch, config.num_heads, config.seq_len, config.head_dim

    def compute_tflops(self, time_in_ms) -> float:
        """Computes the TFLOPS / sec for FA (2 matmuls)"""
        batch, num_heads, seq_len, head_dim = self.get_dims()
        time_in_s = (time_in_ms) * 1e-3
        flops = (4 * (seq_len**2) * head_dim * batch * num_heads) / time_in_s
        return flops / 1e12

    def get_shape(self, transpose: bool = False):
        batch, num_heads, seq_len, head_dim = self.get_dims()
        if transpose:
            return f"{batch * num_heads}x{head_dim}x{seq_len}"
        return f"{batch * num_heads}x{seq_len}x{head_dim}"

    def get_shape_data(self, transpose: bool = False):
        # TODO: Do not hard code f16.
        return self.get_shape() + "x" + self.benchmark_config.dtype

    def create_mlir(self) -> Path:
        ir = ""
        query_shape = self.get_shape_data()
        key_shape = self.get_shape_data()
        value_shape = self.get_shape_data(self.benchmark_config.transpose_v)
        transpose_v = "false"
        if self.benchmark_config.transpose_v:
            transpose_v = "true"

        ir += f"func.func @{self.runner_config.func_name}(%query: tensor<{query_shape}>, %key: tensor<{key_shape}>, %value: tensor<{value_shape}>) -> tensor<{query_shape}> {{\n"
        ir += f"  %0 = tensor.empty() : tensor<{query_shape}>\n"
        ir += f"  %scale = arith.constant 1.0 : f16\n"
        ir += f"  %1 = iree_linalg_ext.attention {{transpose_v = {transpose_v}}} ins(%query, %key, %value, %scale: tensor<{query_shape}>, tensor<{key_shape}>, tensor<{value_shape}>, f16) outs(%0 : tensor<{query_shape}>) -> tensor<{query_shape}>\n"
        ir += f"  return %1 : tensor<{query_shape}>\n"
        ir += "}\n"
        filename = self.runner_config.benchmark_file_prefix.stem + query_shape + ".mlir"
        with open(filename, "w") as f:
            f.write(ir)
        return Path(filename)

    def get_rocm_flags(self):
        return [
            f"--iree-hal-target-backends=rocm",
            f"--iree-rocm-target-chip={self.tuner_config.chip}",
            "--iree-rocm-waves-per-eu=2",
            "--iree-codegen-gpu-native-math-precision=true",
        ]

    def get_td_flags(self):
        spec = self.spec_file
        return [
            f"--iree-codegen-transform-dialect-library={spec}",
        ]

    def get_debug_flags(self):
        return [
            "--mlir-disable-threading",
            "--mlir-print-ir-after-all",
            f"--iree-hal-dump-executable-binaries-to={self.tuner_config.artifact_dir}",
            f"--iree-hal-dump-executable-intermediates-to={self.tuner_config.artifact_dir}",
        ]

    def compile(self, input_file: Path):
        flags = [
            f"{self.tuner_config.iree_build_dir}/tools/iree-compile",
            "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
        ]
        # TD specific flags
        flags += self.get_td_flags()
        # Backend specific flags
        flags += self.get_rocm_flags()
        if self.tuner_config.debug:
            flags += self.get_debug_flags()
        flags += [
            f"-iree-hal-benchmark-dispatch-repeat-count={self.runner_config.iree_benchmark_reps}"
        ]
        flags += [f"{str(input_file)}", "-o", self.get_vmfb_file()]
        execute_command(flags, "fa_dump.txt")
        if not os.path.exists(self.get_vmfb_file()):
            logger.warning("Compilation failed!")

    def compute_reference_inputs_and_outputs(self):
        # TODO: Load binary numpy file when available
        batch, num_heads, seq_len, head_dim = self.get_dims()
        torch.manual_seed(self.runner_config.seed)

        def compute_attention_reference(q, k, v):
            kT = torch.permute(k, (0, 2, 1))
            s = torch.matmul(q, kT)
            p = torch.nn.Softmax(dim=2)(s)
            return torch.matmul(p, v)

        def construct_inputs(B, N, d):
            q = torch.rand((B, N, d), dtype=torch.float16).cuda() / 5.0
            k = torch.rand((B, N, d), dtype=torch.float16).cuda() / 5.0
            v = torch.rand((B, N, d), dtype=torch.float16).cuda() / 5.0
            return q, k, v

        q, k, v = construct_inputs(batch * num_heads, seq_len, head_dim)
        output = compute_attention_reference(q, k, v)

        # Write matrices
        with open(f"query_{self.get_shape_data()}.npy", "wb") as f:
            np.save(f, q.detach().cpu().numpy())
        with open(f"key_{self.get_shape_data()}.npy", "wb") as f:
            np.save(f, k.detach().cpu().numpy())
        if self.benchmark_config.transpose_v:
            v = torch.permute(v, (0, 2, 1))
        with open(
            f"value_{self.get_shape_data(self.benchmark_config.transpose_v)}.npy", "wb"
        ) as f:
            np.save(f, v.detach().cpu().numpy())
        with open(f"output_{self.get_shape_data()}.npy", "wb") as f:
            np.save(f, output.detach().cpu().numpy())

    def check_result(self):
        golden = np.load(f"output_{self.get_shape_data()}.npy")
        computed = np.load(f"computed_{self.get_shape_data()}.npy")
        error = np.max(np.abs(golden - computed))
        # TODO: This tolerance might be too high
        validation_tol = self.runner_config.validation_tol
        if error < validation_tol:
            logger.info(f"[Success] With error = {error} < {validation_tol}")
        else:
            logger.info(f"[Failure] Got {error} > {validation_tol}")

    def validate(self):
        self.compute_reference_inputs_and_outputs()
        flags = [
            f"{self.tuner_config.iree_build_dir}/tools/iree-run-module",
            "--module=" + self.get_vmfb_file(),
            f"--function={self.runner_config.func_name}",
            f"--input=@query_{self.get_shape_data()}.npy",
            f"--input=@key_{self.get_shape_data()}.npy",
            f"--input=@value_{self.get_shape_data(self.benchmark_config.transpose_v)}.npy",
            f"--device=rocm://5",
            f"--output=@computed_{self.get_shape_data()}.npy",
        ]
        execute_command(flags)
        self.check_result()

    def extract_time(self, out):
        output = out.decode("utf-8")
        logger.info("\n" + output)
        time_in_ms = float(re.findall(r"[-+]?(?:\d*\.*\d+)", output.split("\n")[3])[0])
        return time_in_ms

    def benchmark(self):
        command = [
            f"{self.tuner_config.iree_build_dir}/tools/iree-benchmark-module",
            "--module=" + self.get_vmfb_file(),
            f"--function={self.runner_config.func_name}",
            f"--device=rocm://5",
            f"--batch_size={self.runner_config.iree_benchmark_reps}",
        ]

        query_shape = self.get_shape_data()
        key_shape = self.get_shape_data()
        value_shape = self.get_shape_data(self.benchmark_config.transpose_v)
        command += [
            f'--input="{query_shape}"',
            f'--input="{key_shape}"',
            f'--input="{value_shape}"',
        ]
        out, _ = execute_command(command)
        if out is None:
            logger.warning("Failed to extract metrics!")
            return
        time_in_ms = self.extract_time(out)
        tflops = self.compute_tflops(time_in_ms)
        logger.info("Throughput (TFLOPS/s) = " + str(tflops))

    def run(self):
        COMPILE = 0
        VALIDATE = 1
        BENCHMARK = 2

        state = COMPILE
        input_file = self.create_mlir()

        def evaluate():
            nonlocal self
            if state == COMPILE:
                self.compile(input_file)
            if state == VALIDATE:
                self.validate()
            if state == BENCHMARK:
                self.benchmark()

        def log():
            if state == COMPILE:
                logger.info("Compiling ...")
            if state == VALIDATE:
                logger.info("Validating ...")
            if state == BENCHMARK:
                logger.info("Benchmarking ...")

        log()
        evaluate()
        state = VALIDATE
        log()
        evaluate()
        state = BENCHMARK
        log()
        evaluate()


if __name__ == "__main__":
    benchmark_config = BenchmarkConfig(1, 20, 4096, 64, False)
    tuner_config = TunerConfig(
        "gfx942",
        Path("spec2.mlir"),
        Path("./tmp"),
        Path("../iree/build"),
    )
    kernel_config = KernelConfig(64, 128, 4, 2, 2)
    runner_config = RunnerConfig()
    runner = Runner(benchmark_config, tuner_config, kernel_config, runner_config)
    runner.run()
