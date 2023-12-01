# -*- coding: utf-8 -*-

import typing as t

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from matplotlib import pyplot as plt


class Discriminator(nn.Module):

  def __init__(self, img_channels: int, feature_dim: int) -> None:
    super().__init__()
    self.backbone = nn.Sequential(
        nn.Conv2d(
            in_channels=img_channels,
            out_channels=feature_dim,
            kernel_size=4,
            stride=1,
            padding=1
        ),
        nn.LeakyReLU(0.2),
        self.building_block(
            in_channels=feature_dim, out_channels=feature_dim * 2,
            kernel_size=4, stride=1, padding=1
        ),
        self.building_block(
            in_channels=feature_dim * 2, out_channels=feature_dim * 4,
            kernel_size=4, stride=2, padding=1
        ),
        self.building_block(
            in_channels=feature_dim * 4, out_channels=feature_dim * 8,
            kernel_size=4, stride=2, padding=1
        ),
        self.building_block(
            in_channels=feature_dim * 8, out_channels=feature_dim * 16,
            kernel_size=4, stride=2, padding=1
        )
    )
    self.detector = nn.Sequential(
        nn.Conv2d(
            in_channels=feature_dim * 16, out_channels=1,
            kernel_size=3, stride=1, padding=0
        ),
        nn.Flatten()
    )

  def building_block(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int,
      padding: int
  ) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        ),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.2)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.backbone(x)
    return self.detector(x)

class Generator(nn.Module):

  def __init__(self, noize_channels: int, feature_dim: int, img_channels) -> None:
    super().__init__()
    self.backbone = nn.Sequential(
        self.building_block(
            in_channels=noize_channels, out_channels=feature_dim * 64,
            kernel_size=4, stride=1, padding=0
        ),
        self.building_block(
            in_channels=feature_dim * 64, out_channels=feature_dim * 32,
            kernel_size=4, stride=2, padding=0
        ),
        self.building_block(
            in_channels=feature_dim * 32, out_channels=feature_dim * 16,
            kernel_size=4, stride=2, padding=0
        ),
        self.building_block(
            in_channels=feature_dim * 16, out_channels=feature_dim * 8,
            kernel_size=4, stride=1, padding=0
        )
    )
    self.generator = nn.ConvTranspose2d(
        in_channels=feature_dim * 8,
        out_channels=img_channels,
        kernel_size=4,
        stride=1,
        padding=0
    )

  def building_block(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int,
      padding: int
  ) -> nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        ),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU()
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.backbone(x)
    return self.generator(x)

def weight_init(module: nn.Module) -> None:
  if isinstance(module, nn.Conv2d):
    nn.init.normal_(module.weight.data, mean=0, std=0.02)
  if isinstance(module, nn.BatchNorm2d):
    nn.init.normal_(module.weight.data, mean=0, std=0.02)
    nn.init.constant_(module.weight.data, val=0)

def gradient_penalty(
    d_model: nn.Module,
    fake_images: torch.Tensor,
    real_images: torch.Tensor,
    device: str
) -> torch.Tensor:

  batch_size, c, w, h = real_images.shape
  alphas = torch.rand((batch_size, 1, 1, 1)).repeat((1, c, w, h)).to(device)
  interpolate_images = alphas * real_images + (1 - alphas) * fake_images
  scores = d_model(interpolate_images)

  gradient = torch.autograd.grad(
      outputs=scores,
      inputs=interpolate_images,
      grad_outputs=torch.ones_like(scores),
      create_graph=True,
      retain_graph=True
  )[0]

  gradient = gradient.view(gradient.shape[0], -1)
  g_norm = gradient.norm(p=2, dim=1)
  penalty = torch.mean((g_norm - 1) ** 2)
  return penalty

def add_label_to_image(image: torch.Tensor, n_labels: int, label: int) -> torch.Tensor:
  _, w, h = image.shape
  label_channels = torch.zeros((n_labels, w, h))
  label_channels[label, :, :] = 1
  return torch.cat((image, label_channels), dim=0)

channels_dataset = [
    tuple([
        add_label_to_image(
            image=dataset[i][0],
            n_labels=len(idx_to_class),
            label=dataset[i][1]
        ),
        dataset[i][1]
    ])
    for i in range(len(dataset))
]

def plot_generator(
    g_model: nn.Module,
    gen_dim: int,
    img_channels: int,
    idx_to_class: t.Dict[int, str],
    n_labels: int,
    device: str,
    n_cols: int = 10
) -> None:
  noize = torch.randn((n_labels, gen_dim, 1, 1))
  labels = torch.zeros((n_labels, n_labels, 1, 1))
  for i in range(n_labels):
    labels[i, i, :, :] = 1
  noize_and_label = torch.cat((noize, labels), dim=1).to(device)

  with torch.no_grad():
    gen_images = g_model(noize_and_label).detach().cpu()

  n_rows = n_labels // n_cols + 1
  fig, axes = plt.subplots(
      n_rows, n_cols,
      figsize=(int(n_cols * 1.5), int(n_rows * 2))
  )

  for i in range(n_cols * n_rows):
    row_id = i // n_cols
    col_id = i % n_cols

    if i < n_labels:
      axes[row_id][col_id].set_title(idx_to_class[i])
      axes[row_id][col_id].imshow(
          gen_images[i, 0:img_channels, :, :].permute(1, 2, 0),
          cmap='gray_r'
      )
    axes[row_id][col_id].set_axis_off()

  plt.show()

def train_batch(
    d_model: nn.Module,
    g_model: nn.Module,
    d_optimizer: torch.optim.Optimizer,
    g_optimizer: torch.optim.Optimizer,
    real_batch: t.Tuple,
    gen_dim: int,
    n_labels: int,
    device: str
) -> t.Tuple[float, float]:
  batch_size = real_batch[0].shape[0]

  # D_model step
  noize = torch.randn((batch_size, gen_dim, 1, 1))
  labels = torch.zeros((batch_size, n_labels))
  real_labels = real_batch[1]
  labels[torch.arange(batch_size), real_labels] = 1
  labels = labels.unsqueeze(-1).unsqueeze(-1)

  noize_and_label = torch.cat((noize, labels), dim=1).to(device)
  gen_images = g_model(noize_and_label)
  labels = labels.repeat(1, 1, gen_images.shape[2], gen_images.shape[3])
  labels = labels.to(device)
  gen_images = torch.cat((gen_images, labels), dim=1)
  real_images = real_batch[0].to(device)

  d_score_real = d_model(real_images)
  d_score_gen = d_model(gen_images)
  grad_penalty = gradient_penalty(d_model, gen_images, real_images, device=device)

  d_loss = torch.mean(d_score_gen) - torch.mean(d_score_real) + 10 * grad_penalty
  d_optimizer.zero_grad()
  d_loss.backward(retain_graph=True)
  d_optimizer.step()

  # G_model step
  gen_score = d_model(gen_images).reshape(-1)
  g_loss = -torch.mean(gen_score)
  g_optimizer.zero_grad()
  g_loss.backward()
  g_optimizer.step()

  return d_loss.detach().cpu().item(), g_loss.detach().cpu().item()


if __name__ == '__main__':
	torch.manual_seed(0)

	transform = T.Compose([
	    T.ToTensor(),
	    T.Normalize([0.5], [0.5])
	])

	dataset = torchvision.datasets.FashionMNIST(
	    root='./data',
	    transform=transform,
	    download=True,
	    train=True
	)

	idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	n_epochs = 100
	n_labels = 10
	noize_dim = 100
	img_channels = 1
	learning_rate = 1e-4
	train_loader = torch.utils.data.DataLoader(channels_dataset, batch_size=32, shuffle=True)

	d_model = Discriminator(img_channels=1 + n_labels, feature_dim=4).to(device)
	g_model = Generator(noize_channels=noize_dim + n_labels, img_channels=1, feature_dim=4).to(device)

	d_optimizer = torch.optim.Adam(d_model.parameters(), lr=learning_rate, betas=(0., 0.9))
	g_optimizer = torch.optim.Adam(g_model.parameters(), lr=learning_rate, betas=(0., 0.9))

	d_losses = []
	g_losses = []

	for e in range(n_epochs):
	  d_loss_epoch = 0.
	  g_loss_epoch = 0.

	  for batch in train_loader:
	    d_loss, g_loss = train_batch(
	        d_model, g_model,
	        d_optimizer, g_optimizer,
	        batch,
	        gen_dim=100, n_labels=10,
	        device=device
	    )
	    d_loss_epoch += d_loss
	    g_loss_epoch += g_loss

	  d_loss_epoch /= len(train_loader)
	  g_loss_epoch /= len(train_loader)
	  print(f'Epoch: {e}, Losses: {d_loss_epoch}, {g_loss_epoch}')

	  d_losses.append(d_loss_epoch)
	  g_losses.append(g_loss_epoch)

	  plot_generator(
	      g_model,
	      gen_dim=100, img_channels=1,
	      idx_to_class=idx_to_class, n_labels=10,
	      device=device, n_cols=5
	  )

	plt.plot(range(n_epochs), d_losses, label='d loss')
	plt.plot(range(n_epochs), g_losses, label='g loss')
	plt.legend()
	plt.show()
