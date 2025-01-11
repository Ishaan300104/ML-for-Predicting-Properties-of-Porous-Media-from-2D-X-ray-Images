import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from CNN_model_1 import PorosityCNN
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset import MicroCTDataset
from sklearn.model_selection import train_test_split
model = PorosityCNN()

raw_files = [r"Binary RAW images\BanderaBrown_2d25um_binary.raw", 
               r"Binary RAW images\BanderaGray_2d25um_binary.raw", 
               r"Binary RAW images\BB_2d25um_binary.raw", 
               r"Binary RAW images\Bentheimer_2d25um_binary.raw", 
               r"Binary RAW images\Berea_2d25um_binary.raw", 
               r"Binary RAW images\BSG_2d25um_binary.raw", 
               r"Binary RAW images\BUG_2d25um_binary.raw", 
               r"Binary RAW images\CastleGate_2d25um_binary.raw", 
               r"Binary RAW images\Kirby_2d25um_binary.raw", 
               r"Binary RAW images\Leopard_2d25um_binary.raw", 
               r"Binary RAW images\Parker_2d25um_binary.raw"]  # List of paths to your .raw files
labels = [24.11, 18.10, 24.02, 22.64, 18.96, 19.07, 18.56, 26.54, 19.95, 20.22, 14.77]     # List of corresponding porosity labels
    
    # Split the data
train_files, val_files, train_labels, val_labels = train_test_split(raw_files, labels, test_size=0.2, random_state=42)


# Create datasets
train_dataset = MicroCTDataset(train_files, train_labels)
val_dataset = MicroCTDataset(val_files, val_labels)
    
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enable memory efficient features
    torch.backends.cudnn.benchmark = True
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for volumes, labels in train_loader:
            # Clear cache before each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Move data back to CPU to free GPU memory
            volumes = volumes.cpu()
            labels = labels.cpu()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for volumes, labels in val_loader:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                volumes = volumes.to(device)
                labels = labels.to(device)
                outputs = model(volumes)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                volumes = volumes.cpu()
                labels = labels.cpu()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')


if __name__ == "__main__":
    train_model(model, train_loader, val_loader)