import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.models.resnet import resnet50
import typing as tp
import torchmetrics


def cov(x: torch.Tensor, x_mean: torch.Tensor = None) -> torch.Tensor:
    dim = x.shape[-1]
    if x_mean is None:
        x_mean = x.mean(-1)
    x = x - x_mean
    return x @ x.T / (dim - 1)


@torch.no_grad()
def calculate_activation_statistics(
    real_activations: torch.Tensor,
    fake_activations: torch.Tensor,
    # classifier: nn.Module
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    real_activations_mean = torch.mean(real_activations, dim=0)
    fake_activations_mean = torch.mean(fake_activations, dim=0)

    real_activations_cov = cov(real_activations, real_activations_mean)
    fake_activations_cov = cov(fake_activations, fake_activations_mean)
    return (
        real_activations_mean,
        real_activations_cov,
        fake_activations_mean,
        fake_activations_cov,
    )


def calculate_frechet_distance(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:

    diff = mu1 - mu2

    offset = eps * torch.eye(*sigma1.shape, device=sigma1.device)
    eigenvals, _ = torch.linalg.eig((sigma1 + offset) @ (sigma2 + offset))
    tr_covmean = torch.sum(torch.sqrt(torch.abs(eigenvals)))

    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


class FidScore(torchmetrics.Metric):
    def __init__(self, classifier: tp.Optional[nn.Module] = None):
        super().__init__()

        self.classifier = classifier
        if self.classifier is None:
            import types

            @torch.no_grad()
            def get_activations(self_, x):
                x = self_.conv1(x)
                x = self_.bn1(x)
                x = self_.relu(x)
                x = self_.maxpool(x)

                x = self_.layer1(x)
                x = self_.layer2(x)
                x = self_.layer3(x)
                x = self_.layer4(x)

                x = self_.avgpool(x)
                x = torch.flatten(x, 1)

                return x

            model = resnet50(pretrained=True, progress=False)
            model.requires_grad_(False)
            model.get_activations = types.MethodType(get_activations, model)
            self.classifier = model

        self.add_state("real_activations", default=[], dist_reduce_fx=None)
        self.add_state("fake_activations", default=[], dist_reduce_fx=None)

    @torch.no_grad()
    def update(self, real_images, fake_images) -> None:
        self.real_activations.append(self.classifier.get_activations(real_images))
        self.fake_activations.append(self.classifier.get_activations(fake_images))

    def compute(self):
        m1, s1, m2, s2 = calculate_activation_statistics(
            torch.cat(self.real_activations, dim=0),
            torch.cat(self.fake_activations, dim=0),
        )
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value
