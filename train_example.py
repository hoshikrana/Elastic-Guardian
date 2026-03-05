import torch
import torch.nn as nn
from egx.api.trainer import EGX

class LargeModel(nn.Module):
    # To hit ~7B parameters: 8192 * 8192 = 67.1M params per layer. 
    # 67.1M * 104 layers = ~6.97B parameters.
    def __init__(self, hidden_size=7000, num_layers=104):
        super().__init__()
        # Creating a large sequence of linear layers to simulate a big model
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

def main():
    print("Initializing EGX Framework...")
    
    # 1. Instantiate a MASSIVE model (Warning: This will use ~28GB of CPU RAM just to instantiate!)
    print("Building a ~7B parameter model...")
    try:
        model = LargeModel()
    except MemoryError:
        print("\n[Simulation Warning] Not enough system RAM to literally instantiate a 7B parameter model in memory. Falling back to a smaller proxy model to demonstrate the EGX pipeline.")
        # Fall back to a smaller proxy
        model = LargeModel(hidden_size=1024, num_layers=4)
        
    print("\nModel Built. Generating dummy dataset...")
    # 2. Dummy dataset with input_ids matching the model's signature
    dataset = [
        {"input_ids": torch.randn(4, 10)} for _ in range(5)
    ]
    
    # 3. Initialize EGX Trainer with minimal config
    trainer = EGX({"num_epochs": 10000, "learning_rate": 0.01})
    
    # 4. Run Training
    print("Starting EGX Training session...")
    result = trainer.train(model=model, dataset=dataset)
    
    print("\nTraining Complete!")
    print(f"Result Status: {result.get('success')}")
    print(f"Final Loss: {result.get('final_loss')}")
    print(f"Training Mode Selected: {result.get('mode')}")
    
if __name__ == "__main__":
    main()
