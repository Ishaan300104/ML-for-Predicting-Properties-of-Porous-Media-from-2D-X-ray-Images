import torch.nn as nn

class PorosityCNN(nn.Module):
    def __init__(self):
        super(PorosityCNN, self).__init__()
        
        # Reduced number of features in each layer
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool3d((4, 4, 4))
        
        self.regressor = nn.Sequential(
            nn.Linear(32 * 4 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

model = PorosityCNN()
total_params = sum(p.numel()for p in model.parameters())
print(f"Total number of parameters: {total_params}")