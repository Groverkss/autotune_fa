from jinja2 import Environment, FileSystemLoader
from config import KernelConfig, BenchmarkConfig 
from pathlib import Path

class SpecGen:
    def __init__(self, spec_template: Path):
        template_loader = FileSystemLoader(searchpath="./")
        env = Environment(loader=template_loader)
        self.template = env.get_template(str(spec_template))

    def render(self, kernel_config: KernelConfig, benchmark_config: BenchmarkConfig):
        spec_config = {
            **kernel_config.__dict__,
            **benchmark_config.__dict__
        }

        output_spec = self.template.render(spec_config)
        return output_spec
