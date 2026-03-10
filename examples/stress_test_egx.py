"""
EGX - Elastic Guardian X Stress Tester
This script iteratively tests how large of a model the system can handle training
before crashing due to OOM (Out of Memory).
"""
import torch
import torch.nn as nn
import psutil
from rich.console import Console
from rich.table import Table

from egx.api.trainer import EGX

console = Console()

class ScalableModel(nn.Module):
    """A model whose layer dimensions can scale to simulate arbitrary parameter counts."""
    def __init__(self, hidden_size, num_layers):
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

def estimate_parameters(hidden_size, num_layers):
    """Calculates approximate number of parameters in the model."""
    layer_params = (hidden_size * hidden_size + hidden_size) # weight + bias
    in_params = (10 * hidden_size + hidden_size)
    out_params = (hidden_size * 2 + 2)
    return layer_params * num_layers + in_params + out_params

def run_stress_test():
    console.print("[bold red]EGX Maximum Capacity Stress Test Initiated[/]")
    
    # Configurations to test in escalating order
    configs = [
        {"hidden_size": 256, "num_layers": 4, "name": "Tiny (Base)"},
        {"hidden_size": 512, "num_layers": 8, "name": "Small"},
        {"hidden_size": 1024, "num_layers": 12, "name": "Medium"},
        {"hidden_size": 2048, "num_layers": 16, "name": "Large"},
        {"hidden_size": 4096, "num_layers": 24, "name": "XL"},
        {"hidden_size": 8192, "num_layers": 32, "name": "XXL (~2B)"},
        {"hidden_size": 12000, "num_layers": 50, "name": "Massive (~7B)"},
        {"hidden_size": 16000, "num_layers": 80, "name": "Colossal (~20B)"}
    ]
    
    results = []
    
    mem_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    console.print(f"System RAM Available: [cyan]{mem_avail_gb:.2f} GB[/]")
    
    max_params_reached = 0
    passed_configs = 0
    
    for cfg in configs:
        hs = cfg["hidden_size"]
        nl = cfg["num_layers"]
        name = cfg["name"]
        
        params = estimate_parameters(hs, nl)
        params_mil = params / 1_000_000
        
        console.print(f"\n[bold yellow]Testing Config:[/] {name} (Hidden: {hs}, Layers: {nl})")
        console.print(f"Est. Parameters: {params_mil:.1f}M")
        
        # Determine strict memory requirements (approx 4 bytes per param for standard fp32)
        # Training overhead typically adds ~2-3x memory
        est_memory_gb = (params * 4 * 3) / (1024**3)
        current_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        if est_memory_gb > current_ram_gb:
            console.print(f"[bold red]❌ Memory Check Failed![/] Estimated need: {est_memory_gb:.2f} GB. Available: {current_ram_gb:.2f} GB.")
            results.append((name, f"{params_mil:.1f}M", "[bold red]OOM Prevented[/]"))
            break
            
        try:
            console.print("  [cyan]Instantiating model...[/]")
            model = ScalableModel(hidden_size=hs, num_layers=nl)
            
            console.print("  [cyan]Launching EGX...[/]")
            dataset = [{"input_ids": torch.randn(4, 10)} for _ in range(2)]
            trainer = EGX({"num_epochs": 2, "learning_rate": 0.001})
            
            # Disable logger outputs to keep terminal clean 
            import logging
            logging.getLogger("egx").setLevel(logging.CRITICAL)
            
            res = trainer.train(model=model, dataset=dataset)
            
            if res.get("success"):
                console.print(f"  [bold green]✔ Passed via Mode: {res.get('mode')}[/]")
                results.append((name, f"{params_mil:.1f}M", f"[bold green]Success ({res.get('mode')})[/]"))
                max_params_reached = params
                passed_configs += 1
            else:
                results.append((name, f"{params_mil:.1f}M", "[bold red]Framework Failed[/]"))
                break
                
        except Exception as e:
            console.print(f"  [bold red]❌ Failed Execution:[/] {e}")
            results.append((name, f"{params_mil:.1f}M", "[bold red]Crash[/]"))
            break
            
    # Final Report
    console.print("\n[bold cyan]=== Stress Test Summary ==-[/]")
    table = Table()
    table.add_column("Tier", style="magenta")
    table.add_column("Parameters", justify="right")
    table.add_column("Result")
    
    for r in results:
        table.add_row(r[0], r[1], r[2])
        
    console.print(table)
    console.print(f"\n[bold green]MAXIMUM CAPACITY REACHED:[/] {max_params_reached / 1_000_000:.1f}M Parameters")


if __name__ == "__main__":
    run_stress_test()
