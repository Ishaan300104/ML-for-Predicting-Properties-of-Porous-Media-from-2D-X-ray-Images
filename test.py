import torch
import torch.nn as nn
import numpy as np

class PorosityCNN(nn.Module):
    def __init__(self):
        super(PorosityCNN, self).__init__()
        
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
            nn.Dropout(0.5),
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

def predict(raw_file_path, model_path='best_model.pth'):
    # Load and prepare the data
    with open(raw_file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16)
        volume = data.reshape((500, 1000, 1000))
    
    # Downsample
    volume = volume[::4, ::4, ::4]
    
    # Normalize
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    # Convert to tensor
    volume = torch.FloatTensor(volume).unsqueeze(0).unsqueeze(0)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PorosityCNN()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Predict
    with torch.no_grad():
        volume = volume.to(device)
        prediction = model(volume)
        return prediction.item()

# Example usage:
test_paths = [r"Binary RAW images\BanderaBrown_2d25um_binary.raw", 
               r"Binary RAW images\BanderaGray_2d25um_binary.raw", 
               r"Binary RAW images\BB_2d25um_binary.raw", 
               r"Binary RAW images\Bentheimer_2d25um_binary.raw", 
               r"Binary RAW images\Berea_2d25um_binary.raw", 
               r"Binary RAW images\BSG_2d25um_binary.raw", 
               r"Binary RAW images\BUG_2d25um_binary.raw", 
               r"Binary RAW images\CastleGate_2d25um_binary.raw", 
               r"Binary RAW images\Kirby_2d25um_binary.raw", 
               r"Binary RAW images\Leopard_2d25um_binary.raw", 
               r"Binary RAW images\Parker_2d25um_binary.raw"]

true_labels = [24.11, 18.10, 24.02, 22.64, 18.96, 19.07, 18.56, 26.54, 19.95, 20.22, 14.77]     # List of corresponding porosity labels

for i in range(len(test_paths)):
    porosity = predict(test_paths[i])
    print(f"Predicted porosity: {porosity}, actual porosity = {true_labels[i]}")
    