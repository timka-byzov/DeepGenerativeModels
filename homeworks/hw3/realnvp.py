import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CouplingLayer(nn.Module):
    def __init__(self, mask, nets, nett):
        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.scale_net = nets()
        self.trans_net = nett()

    def forward(self, x):
        x_masked = x * self.mask
        s = self.scale_net(x_masked) * (1 - self.mask)
        t = self.trans_net(x_masked) * (1 - self.mask)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_J = s.sum(dim=(1, 2, 3))
        return z, log_det_J

    def inverse(self, z):
        z_masked = z * self.mask
        s = self.scale_net(z_masked) * (1 - self.mask)
        t = self.trans_net(z_masked) * (1 - self.mask)
        x = (1 - self.mask) * (z - t) * torch.exp(-s) + z_masked
        log_det_J = -s.sum(dim=(1, 2, 3))
        return x, log_det_J

class RealNVP(nn.Module):
    def __init__(self, nets, nett, masks, prior):
        super().__init__()
        self.prior = prior
        self.coupling_layers = nn.ModuleList([CouplingLayer(mask, nets, nett) for mask in masks])

    def f(self, x):
        log_det_J = x.new_zeros(x.shape[0])
        z = x
        for layer in self.coupling_layers:
            z, ldj = layer.inverse(z)
            log_det_J += ldj
        return z, log_det_J

    def g(self, z):
        x = z
        for layer in reversed(self.coupling_layers):
            x, _ = layer.forward(x)
        return x

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        return self.prior.log_prob(z).sum(dim=(1, 2, 3)) + log_det_J

    def sample(self, batch_size):
        z = self.prior.sample((batch_size, 3, 64, 64))
        return self.g(z)

# Define CNN-based s and t networks
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# Generate checkerboard masks
def create_masks():
    masks = []
    for i in range(4):  # More masks improve expressivity
        mask = np.random.randint(0, 2, size=(3, 64, 64)).astype(np.float32)
        masks.append(torch.tensor(mask))
    return masks

# Define prior
prior = torch.distributions.Normal(torch.zeros((3, 64, 64)), torch.ones((3, 64, 64)))

# Initialize model
masks = create_masks()
model = RealNVP(SmallCNN, SmallCNN, masks, prior)
