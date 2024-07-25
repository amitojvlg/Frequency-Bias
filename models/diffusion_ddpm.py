import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['DDPM']

# # Define the noise schedule
# class Diffusion:
#     def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02):
#         self.num_timesteps = num_timesteps
#         self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
#         self.alpha = 1.0 - self.beta
#         self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
#     def q_sample(self, x_0, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_0)
#         sqrt_alpha_hat_t = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
#         sqrt_one_minus_alpha_hat_t = torch.sqrt(1.0 - self.alpha_hat[t]).view(-1, 1, 1, 1)
#         return sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise

# # Define the UNet model
# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, num_features=64):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.middle = nn.Sequential(
#             nn.Conv2d(num_features, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features * 2, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, out_channels, 3, 1, 1),
#         )
    
#     def forward(self, x):
#         x1 = self.encoder(x)
#         x2 = self.middle(x1)
#         x3 = self.decoder(x2)
#         return x3

# # Define the DDPM class
# class DDPM:
#     def __init__(self, in_channels, out_channels, num_features=64, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, resolution=64):
#         self.diffusion = Diffusion(num_timesteps, beta_start, beta_end)
#         self.model = UNet(in_channels, out_channels, num_features)
#         self.resolution = resolution

#     def train(self, dataloader, num_epochs, learning_rate=1e-4):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
#         loss_fn = nn.MSELoss()

#         for epoch in range(num_epochs):
#             for batch in dataloader:
#                 x_0 = batch[0]  # Assuming batch is a tuple (data, label)
#                 x_0 = F.interpolate(x_0, size=self.resolution)  # Resize images to the given resolution
#                 t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],))
#                 noise = torch.randn_like(x_0)
#                 x_t = self.diffusion.q_sample(x_0, t, noise)

#                 predicted_noise = self.model(x_t)
#                 loss = loss_fn(predicted_noise, noise)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Example usage (not included in the module, for reference only)
# if __name__ == "__main__":
#     class CustomDataset(torch.utils.data.Dataset):
#         def __getitem__(self, index):
#             return torch.randn(3, 64, 64), 0
#         def __len__(self):
#             return 100
#     
#     dataset = CustomDataset()
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
#     
#     ddpm = DDPM(in_channels=3, out_channels=3, resolution=128)
#     ddpm.train(dataloader, num_epochs=10)



# Define the noise schedule
 




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # Define the noise schedule
# class Diffusion:
#     def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02):
#         self.num_timesteps = num_timesteps
#         self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
#         self.alpha = 1.0 - self.beta
#         self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
#     def q_sample(self, x_0, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_0)
#         sqrt_alpha_hat_t = torch.sqrt(self.alpha_hat[t]).view(-1, 1)
#         sqrt_one_minus_alpha_hat_t = torch.sqrt(1.0 - self.alpha_hat[t]).view(-1, 1)
#         return sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise

# # Define the UNet model for 1D input
# class UNet1D(nn.Module):
#     def __init__(self, in_channels, out_channels, num_features=64):
#         super(UNet1D, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(in_channels, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(num_features, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.middle = nn.Sequential(
#             nn.Conv1d(num_features, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(num_features * 2, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv1d(num_features * 2, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(num_features, out_channels, 3, 1, 1),
#         )
    
#     def forward(self, x):
#         x1 = self.encoder(x)
#         x2 = self.middle(x1)
#         x3 = self.decoder(x2)
#         return x3

# # Define the DDPM class for 1D input
# class DDPM:
#     def __init__(self, in_channels, out_channels, num_features=64, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, resolution=64):
#         self.diffusion = Diffusion(num_timesteps, beta_start, beta_end)
#         self.model = UNet1D(in_channels, out_channels, num_features)
#         self.resolution = resolution

#     def train(self, dataloader, num_epochs, learning_rate=1e-4):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
#         loss_fn = nn.MSELoss()

#         for epoch in range(num_epochs):
#             for batch in dataloader:
#                 x_0 = batch[0].float().to(next(self.model.parameters()).device)  # Ensure x_0 is on the same device as the model
#                 x_0 = F.interpolate(x_0.unsqueeze(1), size=self.resolution).squeeze(1)  # Resize sequences to the given resolution
#                 t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],)).to(x_0.device)
#                 noise = torch.randn_like(x_0)
#                 x_t = self.diffusion.q_sample(x_0, t, noise)

#                 predicted_noise = self.model(x_t)
#                 loss = loss_fn(predicted_noise, noise)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

#         return self

#     def requires_grad_(self, requires_grad=True):
#         for param in self.model.parameters():
#             param.requires_grad = requires_grad
#         return self

#     def to(self, device):
#         self.model.to(device)
#         return self
    
#     def parameters(self):
#         return self.model.parameters()
    
#     def eval(self):
#         self.model.eval()
#         return self
    
#     def __call__(self, x):
#         device = next(self.model.parameters()).device
#         if x.dtype != torch.float32:
#             x = x.float()
#         x = x.to(device)  # Ensure input tensor is on the same device as the model
#         if x.dim() == 2:
#             x = x.unsqueeze(1)  # Add channel dimension if it's a single sequence
#         elif x.dim() != 3:
#             raise ValueError(f"Expected input to be 2D or 3D tensor, but got {x.dim()}D tensor.")
#         return self.model(x)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Diffusion:
#     def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02):
#         self.num_timesteps = num_timesteps
#         self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
#         self.alpha = 1.0 - self.beta
#         self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
#     def q_sample(self, x_0, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_0)
#         sqrt_alpha_hat_t = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1).to(device= torch.device('cuda:0'))
#         sqrt_one_minus_alpha_hat_t = torch.sqrt(1.0 - self.alpha_hat[t]).view(-1, 1, 1, 1).to(device= torch.device('cuda:0'))
#         return sqrt_alpha_hat_t * x_0.to(device= torch.device('cuda:0')) + sqrt_one_minus_alpha_hat_t * noise

# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, num_features=64):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.middle = nn.Sequential(
#             nn.Conv2d(num_features, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features * 2, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, out_channels, 3, 1, 1),
#         )
    
#     def forward(self, x):
#         x1 = self.encoder(x)
#         x2 = self.middle(x1)
#         x3 = self.decoder(x2)
#         return x3

# class DDPM(nn.Module):  # Inherit from nn.Module
#     def __init__(self, in_channels, out_channels, num_features=64, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, resolution=64):
#         super(DDPM, self).__init__()  # Call the superclass constructor
#         self.diffusion = Diffusion(num_timesteps, beta_start, beta_end)
#         self.model = UNet(in_channels, out_channels, num_features)
#         self.resolution = resolution
        

#     def train(self, dataloader, num_epochs, learning_rate=1e-4):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
#         loss_fn = nn.MSELoss()

#         for epoch in range(num_epochs):
#             for batch in dataloader:
#                 x_0 = batch[0]  # Assuming batch is a tuple (data, label)
#                 x_0 = F.interpolate(x_0, size=self.resolution)  # Resize images to the given resolution
#                 t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],))
#                 noise = torch.randn_like(x_0).to(device= torch.device('cuda:0'))
#                 x_t = self.diffusion.q_sample(x_0, t, noise).to(device = torch.device('cuda:0'))
#                 predicted_noise = self.model(x_t).to(device= torch.device('cuda:0'))
#                 loss = loss_fn(predicted_noise, noise)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

#         return self

#     def requires_grad_(self, requires_grad=True):
#         for param in self.model.parameters():
#             param.requires_grad = requires_grad
#         return self

#     def to(self, device):
#         self.model.to(device)
#         return self
    
#     def parameters(self):
#         return self.model.parameters()

#     def eval(self):
#         self.model.eval()
#         return self

    #def __call__(self, x):
     #   if x.dtype != torch.float32:
     #       x = x.float()
     #   if x.dim() == 3:
     #       x = x.unsqueeze(0)  # Add batch dimension if it's a single image
      #  elif x.dim() != 4:
       #     raise ValueError(f"Expected input to be 3D or 4D tensor, but got {x.dim()}D tensor.")
        #return self.model(x)
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Diffusion:
#     def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02):
#         self.num_timesteps = num_timesteps
#         self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
#         self.alpha = 1.0 - self.beta
#         self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
#     def q_sample(self, x_0, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_0)
#         # Ensure t is a tensor and broadcast correctly
#         t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
#         sqrt_alpha_hat_t = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
#         sqrt_one_minus_alpha_hat_t = torch.sqrt(1.0 - self.alpha_hat[t]).view(-1, 1, 1, 1)
#         return sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise

# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, num_features=64):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.middle = nn.Sequential(
#             nn.Conv2d(num_features, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features * 2, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, out_channels, 3, 1, 1),
#         )
    
#     def forward(self, x):
#         x1 = self.encoder(x)
#         x2 = self.middle(x1)
#         x3 = self.decoder(x2)
#         return x3

# class DDPM(nn.Module):
#     def __init__(self, in_channels, out_channels, num_features=64, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, resolution=64):
#         super(DDPM, self).__init__()
#         self.diffusion = Diffusion(num_timesteps, beta_start, beta_end)
#         self.model = UNet(in_channels, out_channels, num_features)
#         self.resolution = resolution

#     def forward(self, x_t):
#         return self.model(x_t)
        
#     def train_model(self, dataloader, num_epochs, learning_rate=1e-4, device=torch.device('cuda:0')):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
#         loss_fn = nn.MSELoss()
#         self.to(device)  # Move the model to the correct device

#         for epoch in range(num_epochs):
#             epoch_loss = 0
#             for batch in dataloader:
#                 x_0 = batch[0].to(device)  # Move batch to the correct device
#                 x_0 = F.interpolate(x_0, size=self.resolution)
#                 t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],), device=device)
#                 noise = torch.randn_like(x_0).to(device)
#                 x_t = self.diffusion.q_sample(x_0, t, noise).to(device)
#                 predicted_noise = self(x_t).to(device)
#                 loss = loss_fn(predicted_noise, noise)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 epoch_loss += loss.item()

#             avg_epoch_loss = epoch_loss / len(dataloader)
#             print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss}')

#         return self

#     def requires_grad_(self, requires_grad=True):
#         for param in self.model.parameters():
#             param.requires_grad = requires_grad
#         return self

#     def to(self, device):
#         self.model.to(device)
#         self.device = device
#         return self
    
#     def parameters(self):
#         return self.model.parameters()

#     def eval_model(self):
#         self.model.eval()
#         return self

# Example usage:
# ddpm = DDPM(in_channels=3, out_channels=3, num_features=64, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, resolution=64)
# ddpm.train_model(dataloader, num_epochs=10, learning_rate=1e-4)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Diffusion:
#     def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02):
#         self.num_timesteps = num_timesteps
#         self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
#         self.alpha = 1.0 - self.beta
#         self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
#     def q_sample(self, x_0, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_0)
#         # Convert t to long for indexing and to float for computation
#         t = t.long().to(x_0.device)
#         sqrt_alpha_hat_t = torch.sqrt(self.alpha_hat[t].float()).view(-1, 1, 1, 1).to(x_0.device)
#         sqrt_one_minus_alpha_hat_t = torch.sqrt(1.0 - self.alpha_hat[t].float()).view(-1, 1, 1, 1).to(x_0.device)
#         return sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise

# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, num_features=64):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.middle = nn.Sequential(
#             nn.Conv2d(num_features, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features * 2, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, out_channels, 3, 1, 1),
#         )
    
#     def forward(self, x):
#         x1 = self.encoder(x)
#         x2 = self.middle(x1)
#         x3 = self.decoder(x2)
#         return x3

# class DDPM(nn.Module):
#     def __init__(self, in_channels, out_channels, num_features=64, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, resolution=64):
#         super(DDPM, self).__init__()
#         self.diffusion = Diffusion(num_timesteps, beta_start, beta_end)
#         self.model = UNet(in_channels, out_channels, num_features)
#         self.resolution = resolution

#     def forward(self, x_t):
#         return self.model(x_t)
        
#     def train_model(self, dataloader, num_epochs, learning_rate=1e-4, device=torch.device('cuda:0')):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
#         loss_fn = nn.MSELoss()
#         self.to(device)  # Move the model to the correct device

#         for epoch in range(num_epochs):
#             epoch_loss = 0
#             for batch in dataloader:
#                 x_0 = batch[0].to(device)  # Move batch to the correct device
#                 x_0 = F.interpolate(x_0, size=self.resolution)
#                 t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],), device=device)
#                 noise = torch.randn_like(x_0).to(device)
#                 x_t = self.diffusion.q_sample(x_0, t, noise)
#                 predicted_noise = self(x_t)
#                 loss = loss_fn(predicted_noise, noise)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 epoch_loss += loss.item()

#             avg_epoch_loss = epoch_loss / len(dataloader)
#             print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss}')

#         return self

#     def requires_grad_(self, requires_grad=True):
#         for param in self.model.parameters():
#             param.requires_grad = requires_grad
#         return self

#     def to(self, device):
#         self.model.to(device)
#         self.device = device
#         return self
    
#     def parameters(self):
#         return self.model.parameters()

#     def eval_model(self):
#         self.model.eval()
#         return self
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
__all__ = ['DDPM']

class Diffusion:
    def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        # Ensure t is a tensor and of the correct type
        t = torch.tensor(t, dtype=torch.long, device=x_0.device)
        alpha_hat_t = self.alpha_hat[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        sqrt_alpha_hat_t = torch.sqrt(alpha_hat_t)
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1.0 - alpha_hat_t)
        return sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_features=64):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(num_features, num_features * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features * 2, num_features * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 3, 1, 1),
        )
    
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

class DDPM(nn.Module):
    def __init__(self, in_channels, out_channels, num_features=64, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, resolution=64):
        super(DDPM, self).__init__()
        self.diffusion = Diffusion(num_timesteps, beta_start, beta_end)
        self.model = UNet(in_channels, out_channels, num_features)
        self.resolution = resolution

    def forward(self, x_t):
         return self.model(x_t)
        
    def train_model(self, dataloader, num_epochs, learning_rate=1e-4, device=torch.device('cuda:0')):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        self.to(device)  # Move the model to the correct device

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in dataloader:
                x_0 = batch[0].to(device, dtype=torch.float32)  # Ensure batch is float and on the correct device
                x_0 = F.interpolate(x_0, size=self.resolution)
                t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],), device=device, dtype=torch.long)
                noise = torch.randn_like(x_0)
                x_t = self.diffusion.q_sample(x_0, t, noise)
                predicted_noise = self(x_t)
                loss = loss_fn(predicted_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss}')

        return self

    def requires_grad_(self, requires_grad=True):
        for param in self.model.parameters():
            param.requires_grad = requires_grad
        return self

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self
    
    def parameters(self):
        return self.model.parameters()

    def eval_model(self):
        self.model.eval()
        return self                                                                                                                       

