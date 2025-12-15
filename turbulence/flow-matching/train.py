import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os

# Import model architectures
from model.mixer import MLPMixer
# from model.unet import UNet  # Optional UNet

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_id = 0
cuda = True
DEVICE = torch.device(f"cuda:{gpu_id}" if cuda else "cpu")

# Set random seeds
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

# Hyperparameters
sigma_min = 0
n_epochs = 10000
lr = 4e-3
train_batch_size = 1024
dim = 12
image_size = 4

# DataLoader configuration
kwargs = {'num_workers': 1, 'pin_memory': True}

# Load and preprocess data
train_data = torch.load("../latent/train_latents.pt").reshape(-1, 1, 4, 4)
print(train_data.shape)
train_data = torch.tensor(train_data, dtype=torch.float).detach().cpu().view(-1, 4, 4, 4)
train_data = train_data.view(-1, 4, 4, 4)

# Create DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True, **kwargs)


class Model(nn.Module):
    def __init__(self, layer, sigma_min=0.):
        super(Model, self).__init__()
        self.layer = layer
        self.sigma_min = sigma_min

    def get_velocity(self, x_0, x_1):
        """Compute target velocity field: v = x_1 - (1 - sigma_min) * x_0"""
        return x_1 - (1 - self.sigma_min) * x_0
    
    def interpolate(self, x_0, x_1, t):
        """Linear interpolation: x_t = (1 - (1-sigma_min)*t)*x_0 + t*x_1"""
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1
    
    def forward(self, x, t):
        """Estimate velocity field at position x and time t"""
        y = self.layer(x, t)
        return y

    @torch.no_grad()
    def sample(self, t_steps, shape, DEVICE):
        """Generate samples by integrating the learned velocity field"""
        B, C, W, H = shape
        x_0 = torch.randn(size=shape, device=DEVICE)
        t_vals = torch.linspace(0, 1, t_steps, device=DEVICE)
        delta = 1.0 / (t_steps - 1)
        x_1_hat = x_0
        
        for i in range(t_steps - 1):
            t_cur = t_vals[i].view(-1, 1, 1, 1)
            velocity_pred = self(x_1_hat, t_cur)
            x_1_hat = x_1_hat + velocity_pred * delta
            
        return x_1_hat

# Create backbone model
mixer_model = MLPMixer(
    in_channels=4, 
    dim=dim, 
    patch_size=1, 
    num_patches=16, 
    depth=5, 
    token_dim=dim, 
    channel_dim=dim
)

# unet_model = UNet(in_channels=1, kernel_size=3, n_feat=dim, lattice_shape=(4, 4))

# Initialize CFM model
model = Model(layer=mixer_model, sigma_min=sigma_min).to(DEVICE)
print(model)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Load pretrained weights (optional)
# state_dict = torch.load("1.pt")
# model.load_state_dict(state_dict)

# Setup optimizer
optimizer = Adam(model.parameters(), lr=lr)

# Training loop
print("Start training CFM...")
model.train()

for epoch in range(n_epochs):
    total_loss = 0
    for batch_idx, x_1 in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        # Move data to device
        x_1 = x_1.to(DEVICE)
        
        # Sample noise and random time
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], 1, 1, 1, device=DEVICE)
        
        # Interpolate and compute velocities
        x_t = model.interpolate(x_0, x_1, t)
        velocity_target = model.get_velocity(x_0, x_1)
        velocity_pred = model(x_t, t)
        
        # Compute loss
        loss = ((velocity_pred - velocity_target) ** 2).mean()
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    # Save checkpoint
    if epoch % 50 == 0:
        torch.save(model.state_dict(), "state_dict.pt")
    
    # Print epoch statistics
    print(f"Epoch {epoch + 1} complete! CFM Loss: {total_loss / len(train_loader)}")

print("Finish!!")