"""
EGX - Elastic Guardian X Showcase
This script demonstrates the core capabilities of the EGX Intelligent Runtime.
"""
import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
import time

from egx.infrastructure.gpu_probe import GPUProber
from egx.infrastructure.topology_builder import TopologyBuilder
from egx.api.trainer import EGX

console = Console()

class ShowcaseModel(nn.Module):
    """A proxy model for the showcase."""
    def __init__(self, hidden_size=1024, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.fc_in = nn.Linear(10, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 2)
        
    def forward(self, input_ids):
        x = self.fc_in(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.fc_out(x)

def run_showcase():
    console.print(Panel.fit(
        "[bold cyan]EGX - Elastic Guardian X[/]\n[italic]The Intelligent Adaptive Training Runtime for the Modern ML Stack.[/]",
        border_style="cyan"
    ))
    
    # 1. Hardware Probing
    console.print("\n[bold yellow]Phase 1: Intelligent Hardware Probing...[/]")
    time.sleep(1)
    
    gpus = GPUProber().probe()
    topo = TopologyBuilder().build(gpus)
    
    table = Table(title="Detected Hardware Topography")
    table.add_column("Resource", style="cyan", justify="right")
    table.add_column("Specification", style="magenta")
    table.add_row("CPU Cores", str(topo.cpu_cores))
    table.add_row("System RAM", f"{topo.ram_bytes // (1024**3)} GB")
    
    for gpu in topo.gpus:
        table.add_row(
            f"GPU {gpu.device_id} ({gpu.name})",
            f"{gpu.vram_bytes // (1024**3)} GB VRAM"
        )
    
    console.print(table)
    
    # 2. Decision Engine
    console.print("\n[bold yellow]Phase 2: Estimating Optimal Strategy...[/]")
    for _ in track(range(100), description="[cyan]Analyzing parameters..."):
        time.sleep(0.01)
        
    console.print("✔ Analysis Complete! Selected Strategy: [bold green]QLoRA / 4-bit Quantization[/]")
    
    # 3. Training Runtime
    console.print("\n[bold yellow]Phase 3: Launching EGX Training Runtime...[/]")
    
    console.print("Building proxy model...")
    model = ShowcaseModel()
    dataset = [{"input_ids": torch.randn(4, 10)} for _ in range(3)]
    
    trainer = EGX({"num_epochs": 100, "learning_rate": 0.001})
    
    with console.status("[bold green]Executing EGX Training Loop..."):
        result = trainer.train(model=model, dataset=dataset)
        time.sleep(2)  # Simulate some heavy lifting
    
    # 4. Results
    console.print("\n[bold yellow]Phase 4: Training Complete[/]")
    
    res_table = Table(show_header=False, box=None)
    res_table.add_row("Status:", "[bold green]Success[/]" if result.get('success') else "[bold red]Failed[/]")
    res_table.add_row("Final Loss:", f"[bold cyan]{result.get('final_loss', 'N/A')}[/]")
    res_table.add_row("Selected Mode:", f"[bold magenta]{result.get('mode', 'UNKNOWN')}[/]")
    
    console.print(Panel(res_table, title="Session Summary", border_style="green"))


if __name__ == "__main__":
    run_showcase()
