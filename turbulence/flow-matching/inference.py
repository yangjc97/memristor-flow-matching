import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.optim import Adam
import os

# Import model architecture
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
n_epochs = 1000
lr = 1e-4
train_batch_size = 2048
dim = 12

# Load and preprocess data
train_data = torch.load("../latent/train_latents.pt").reshape(-1, 1, 4, 4)
train_data = train_data.detach().clone().cpu().float().view(-1, 4, 4, 4)

class Model(nn.Module):
    def __init__(self, layer, sigma_min=0.):
        super(Model, self).__init__()
        self.layer = layer
        self.sigma_min = sigma_min

    def get_velocity(self, x_0, x_1):
        """Compute target velocity field"""
        return x_1 - (1 - self.sigma_min) * x_0
    
    def interpolate(self, x_0, x_1, t):
        """Linear interpolation between noise and data"""
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1
    
    def forward(self, x, t):
        """Estimate velocity field"""
        y = self.layer(x, t)
        return y

    @torch.no_grad()
    def sample(self, t_steps, shape, DEVICE):
        """Generate samples by integrating learned velocity field"""
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

# Initialize CFM model
model = Model(layer=mixer_model, sigma_min=sigma_min).to(DEVICE)
# print(model)

# Load pretrained weights
state_dict = torch.load("state_dict.pt")
model.load_state_dict(state_dict)
model.eval()

# Generate samples
B = 1024
N = 4
inference_n_flows = 20

cfgs2 = []
for i in range(16):
    x_hats = model.sample(inference_n_flows, shape=[B, 4, N, N], DEVICE=DEVICE)
    cfgs2.append(x_hats.detach().cpu().numpy())

# Post-process generated data
cfgs2 = np.array(cfgs2).reshape(-1, 1, 4, 4)
gen_cfgs = torch.tensor(cfgs2).reshape(-1, 1, 16)
gen_cfgs = gen_cfgs.reshape(-1, 16)
torch.save(gen_cfgs, "../latent/gen_feature.pt")

# Compare with original data
cfgs = train_data.reshape(-1, 16)
print(f"Original data range: [{cfgs.min():.6f}, {cfgs.max():.6f}]")
print(f"Generated data range: [{cfgs2.min():.6f}, {cfgs2.max():.6f}]")

def plot_distributions(target, prediction, num_samples=50000, filename="distribution_comparison.jpg"):
    """Plot comparison between target and prediction distributions"""
    flattened_target = target.flatten()[:num_samples].numpy()
    flattened_prediction = prediction.flatten()[:num_samples]
    
    xmin = min(flattened_target.min(), flattened_prediction.min())
    xmax = max(flattened_target.max(), flattened_prediction.max())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(
        flattened_target, 
        bins=40, 
        kde=True, 
        color='blue', 
        label='Target', 
        alpha=0.5, 
        binrange=(xmin, xmax)
    )
    sns.histplot(
        flattened_prediction, 
        bins=40, 
        kde=True, 
        color='red', 
        label='Prediction', 
        alpha=0.5, 
        binrange=(xmin, xmax)
    )
    
    plt.title('Distribution of Target and Prediction')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(filename)
    plt.close()

plot_distributions(cfgs, cfgs2)