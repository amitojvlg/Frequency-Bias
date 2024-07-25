import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
__all__ = ['LDM']

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, num_features=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(num_features * 8 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(num_features * 8 * 4 * 4, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, num_features * 8 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features, in_channels, 4, 2, 1),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), -1, 4, 4)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class Diffusion:
    def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
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
# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, num_features=64):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, num_features, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, num_features * 2, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.middle = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features * 4, 3, 1, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(num_features * 2, out_channels, 4, 2, 1),
#         )
    
#     def forward(self, x):
#         x1 = self.encoder(x)
#         x2 = self.middle(x1)
#         x3 = self.decoder(x2)
#         return x3


class LDM(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, num_features=64, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, resolution=64):
        super(LDM, self).__init__()
        self.vae = VAE(in_channels, latent_dim, num_features)
        self.diffusion = Diffusion(num_timesteps, beta_start, beta_end)
        self.model = UNet(latent_dim, latent_dim, num_features)
        self.resolution = resolution

    def forward(self, z_t):
        return self.model(z_t)

    def train_model(self, dataloader, num_epochs, learning_rate=1e-4, device=torch.device('cuda:0')):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        self.to(device)

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in dataloader:
                x_0 = batch[0].to(device, dtype=torch.float32)
                x_0 = F.interpolate(x_0, size=self.resolution)
                x_recon, mu, logvar = self.vae(x_0)
                
                t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],), device=device, dtype=torch.long)
                noise = torch.randn_like(mu)
                z_t = self.diffusion.q_sample(mu, t, noise)
                predicted_noise = self(z_t)
                loss = loss_fn(predicted_noise, noise) + self.kl_divergence(mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss}')

        return self

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def requires_grad_(self, requires_grad=True):
        for param in self.vae.parameters():
            param.requires_grad = requires_grad
        for param in self.model.parameters():
            param.requires_grad = requires_grad
        return self

    def to(self, device):
        self.vae.to(device)
        self.model.to(device)
        self.device = device
        return self

    def parameters(self):
        return list(self.vae.parameters()) + list(self.model.parameters())

    def eval_model(self):
        self.vae.eval()
        self.model.eval()
        return self




