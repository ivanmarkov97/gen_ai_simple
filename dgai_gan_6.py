# -*- coding: utf-8 -*-

import os
import warnings
import typing as t

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from PIL import Image

warnings.filterwarnings('ignore')


class Horse2ZebraDataset(Dataset):

  def __init__(
      self, path: str,
      codec: str = '.jpg',
      transforms: t.Optional[T.Compose] = None
  ) -> None:
    self.images_paths = []
    self.transforms = transforms

    for file in os.listdir(path):
      if file.endswith(codec):
        self.images_paths.append(f'{path}/{file}')

  def __getitem__(self, index: int) -> torch.Tensor:
    image_path = self.images_paths[index]
    image = Image.open(image_path)
    if self.transforms:
      if len(image.getbands()) != 3:
        image = image.convert(mode='RGB')
      image = self.transforms(image)
    else:
      image = torch.tensor(image)
    return image

  def __len__(self) -> int:
    return len(self.images_paths)


class ComposeHorse2ZebraDataset(Dataset):

  def __init__(self, horse_dataset: Dataset, zebra_dataset: Dataset) -> None:
    self.horse_dataset = horse_dataset
    self.zebra_dataset = zebra_dataset

  def __getitem__(self, index: int) -> t.Tuple[torch.Tensor]:
    return horse_dataset[index], self.zebra_dataset[index]

  def __len__(self) -> int:
    return min(len(horse_dataset), len(zebra_dataset))


class DBlock(nn.Module):

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      stride: int = 2,
      kernel: int = 4,
      padding: int = 0
    ) -> None:
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False
        ),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(x)


class Discriminator(nn.Module):

  def __init__(
      self,
      in_channels: int = 3,
      features: t.List[int] = [32, 64, 128, 256]
    ) -> None:
    super().__init__()

    layers = []
    prev_feature = in_channels

    for feature in features:
      layers.append(DBlock(prev_feature, feature, stride=2, kernel=5, padding=0))
      prev_feature = feature

    layers.append(
        nn.Conv2d(
            in_channels=features[-1],
            out_channels=1,
            kernel_size=5,
            stride=2,
            padding=0,
            bias=False
        ))
    layers.append(nn.Flatten())
    self.backbone = nn.Sequential(*layers)
    self.clf = nn.Sigmoid()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.clf(self.backbone(x))

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect"
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True)
          )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, features: t.List[int] = [64,128,256,512]) -> None:
        super().__init__()

        layers = []
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        in_channels = features[0]

        for feature in features[1:]:
            layers.append(
                Block(
                    in_channels,
                    feature,
                    stride=1 if feature == features[-1] else 2
                )
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect"
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(self.initial(x))
        return torch.sigmoid(out)

class GBlock(nn.Module):

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int = 1,
      padding: int = 0,
      down: bool = True,
      activate: bool = True
    ) -> None:
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        ) if down else nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        ),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True) if activate else nn.Identity()
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(x)


class ResidualGBlock(nn.Module):

  def __init__(self, channels: int) -> None:
    super().__init__()
    self.net = nn.Sequential(
        GBlock(channels, channels, kernel_size=3, stride=1, padding=1, down=False, activate=True),
        GBlock(channels, channels, kernel_size=3, stride=1, padding=1, down=False, activate=False)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.net(x)


class Generator(nn.Module):

  def __init__(self, img_channels: int, features: int, n_resid: int) -> None:
    super().__init__()
    self.down_net = nn.ModuleList([
        GBlock(img_channels, features, kernel_size=4, stride=2, down=True, activate=True),
        GBlock(features, 2 * features, kernel_size=4, stride=1, down=True, activate=True),
        GBlock(2 * features, 4 * features, kernel_size=4, stride=1, down=True, activate=True),
    ])
    self.resid = nn.Sequential(
        *[ResidualGBlock(4 * features) for _ in range(n_resid)]
    )
    self.up_net = nn.ModuleList([
        GBlock(4 * features, 2 * features, kernel_size=4, stride=1, down=False, activate=True),
        GBlock(2 * features, features, kernel_size=4, stride=1, down=False, activate=True),
        GBlock(features, img_channels, kernel_size=4, stride=2, down=False, activate=False),
    ])
    self.output = nn.Tanh()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # print('Gen init', x.shape)
    for layer in self.down_net:
      x = layer(x)
      # print('Gen down', x.shape)
    x = self.resid(x)
    # print('Gen resid', x.shape)
    for layer in self.up_net:
      x = layer(x)
      # print('Gen up', x.shape)
    return self.output(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels,
                                    out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity())
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels,channels,kernel_size=3,padding=1),
            ConvBlock(channels,channels,
                      use_act=False, kernel_size=3, padding=1))
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64,
                 num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect"
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList([
            ConvBlock(
                num_features,
                num_features*2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ConvBlock(
                num_features*2,
                num_features*4,
                kernel_size=3,
                stride=2,
                padding=1
            )
        ])

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_features * 4)
             for _ in range(num_residuals)
        ])

        self.up_blocks = nn.ModuleList([
            ConvBlock(
                num_features * 4,
                num_features * 2,
                down=False,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            ConvBlock(
                num_features * 2,
                num_features * 1,
                down=False,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
        ])
        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect"
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

def train_epoch(
    horse_d_model: nn.Module,
    zebra_d_model: nn.Module,
    horse_g_model: nn.Module,
    zebra_g_model: nn.Module,
    loader: DataLoader,
    d_optimizer: torch.optim.Optimizer,
    g_optimizer: torch.optim.Optimizer,
    l1_criteria: t.Callable,
    mse_criteria: t.Callable,
    d_scaler: GradScaler,
    g_scaler: GradScaler,
    cycle_lambda: float,
    device: str
) -> t.Tuple[float, float]:

  d_epoch_loss = 0.
  g_epoch_loss = 0.

  for h_batch, z_batch in loader:
    h_batch = h_batch.to(device)
    z_batch = z_batch.to(device)

    # Train Discriminator
    with torch.cuda.amp.autocast():

      # Discminator horse train
      # gen horse from zebra
      gen_horse = horse_g_model(z_batch)
      # get real horse scores
      real_horse_scores = horse_d_model(h_batch)
      # get gen horse scores
      gen_horse_scores = horse_d_model(gen_horse.detach())

      d_h_loss_real = mse_criteria(real_horse_scores, torch.ones_like(real_horse_scores))
      d_h_loss_gen = mse_criteria(gen_horse_scores, torch.zeros_like(gen_horse_scores))
      d_h_loss = d_h_loss_real + d_h_loss_gen

      # Discriminator zebra train
      # gen zebra from horse
      gen_zebra = zebra_g_model(h_batch)
      # get real zebra scores
      real_zebra_scores = zebra_d_model(z_batch)
      # get gen zebra scores
      gen_zebra_score = zebra_d_model(gen_zebra.detach())

      d_z_loss_real = mse_criteria(real_zebra_scores, torch.ones_like(real_zebra_scores))
      d_z_loss_gen = mse_criteria(gen_zebra_score, torch.zeros_like(gen_zebra_score))
      d_z_loss = d_z_loss_real + d_z_loss_gen

      d_loss = (d_h_loss + d_z_loss) / 4

    d_optimizer.zero_grad()
    d_scaler.scale(d_loss).backward()
    d_scaler.step(d_optimizer)
    d_scaler.update()

    # Train Generator
    with torch.cuda.amp.autocast():
      # Traditional loss
      # get gen scores
      gen_horse_scores = horse_d_model(gen_horse)
      gen_zebra_scores = zebra_d_model(gen_zebra)
      # get disc loss
      g_h_loss = mse_criteria(gen_horse_scores, torch.ones_like(gen_horse_scores))
      g_z_loss = mse_criteria(gen_zebra_scores, torch.ones_like(gen_zebra_scores))

      # Cycle loss
      # gen zebra from generated horse
      cycle_zebra = zebra_g_model(gen_horse)
      # gen horse from generated zebra
      cycle_horse = horse_g_model(gen_zebra)

      cycle_h_loss = l1_criteria(h_batch, cycle_horse)
      cycle_z_loss = l1_criteria(z_batch, cycle_zebra)
      g_loss = (g_h_loss + cycle_lambda * cycle_h_loss) + (g_z_loss + cycle_lambda * cycle_z_loss)

    g_optimizer.zero_grad()
    g_scaler.scale(g_loss).backward()
    g_scaler.step(g_optimizer)
    g_scaler.update()

    d_epoch_loss += d_loss.detach().cpu().item()
    g_epoch_loss += g_loss.detach().cpu().item()

  d_epoch_loss /= len(loader)
  g_epoch_loss /= len(loader)

  return d_epoch_loss, g_epoch_loss

from tqdm import tqdm


def train_epoch(disc_A, disc_B, gen_A, gen_B, loader, opt_disc,
        opt_gen, l1, mse, d_scaler, g_scaler,cycle_lambda,device):
    loop = tqdm(loader, leave=True)

    for i, (A,B) in enumerate(loop):
        A=A.to(device)
        B=B.to(device)
        # Train Discriminators A and B
        with torch.cuda.amp.autocast():
            fake_A = gen_A(B)
            D_A_real = disc_A(A)
            D_A_fake = disc_A(fake_A.detach())
            D_A_real_loss = mse(D_A_real,
                                torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake,
                                torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss
            fake_B = gen_B(A)
            D_B_real = disc_B(B)
            D_B_fake = disc_B(fake_B.detach())
            D_B_real_loss = mse(D_B_real,
                                torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake,
                                torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss
            # Average loss of the two discriminators
            D_loss = (D_A_loss + D_B_loss) / 2
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        # Train the two generators
        with torch.cuda.amp.autocast():
            D_A_fake = disc_A(fake_A)
            D_B_fake = disc_B(fake_B)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))
            # NEW in Cycle GANs: cycle loss
            cycle_B = gen_B(fake_A)
            cycle_A = gen_A(fake_B)
            cycle_B_loss = l1(B, cycle_B)
            cycle_A_loss = l1(A, cycle_A)
            # Total generator loss
            G_loss=(loss_G_A + loss_G_B + cycle_A_loss * cycle_lambda + cycle_B_loss * cycle_lambda)
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

    return 0., 0.

def gen_zebra_from_horse(model: nn.Module, device: str) -> None:

  fig, axes = plt.subplots(2, 5, figsize=(int(n_cols * 2.5), int(n_rows * 4)))

  random_indexes = np.random.randint(0, 500, size=10)
  image_paths = os.listdir('trainA')
  random_images = [f'trainA/{image_paths[i]}' for i in random_indexes]

  with torch.inference_mode():
    for i in range(10):
      image = Image.open(random_images[i])
      if len(image.getbands()) != 3:
        image = image.convert(mode='RGB')
      image = transforms(image).unsqueeze(0).to(device)

      gen_zebra = zebra_g_model(image)[0]
      gen_zebra = gen_zebra.detach().cpu().permute(1, 2, 0) * 0.5 + 0.5

      row_id = i // 5
      col_id = i % 5
      axes[row_id][col_id].set_xticklabels([])
      axes[row_id][col_id].set_yticklabels([])
      axes[row_id][col_id].imshow(gen_zebra)
      axes[row_id][col_id].set_axis_off()

  fig.subplots_adjust(wspace=0, hspace=0)
  plt.show()


if __name__ == '__main__':
	torch.manual_seed(0)

	n_epochs = 10
	batch_size = 1
	learning_rate = 1e-5
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	transforms = T.Compose([
	    T.Resize((256, 256)),
	    T.ToTensor(),
	    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])

	horse_dataset = Horse2ZebraDataset(path='trainA', codec='.jpg', transforms=transforms)
	zebra_dataset = Horse2ZebraDataset(path='trainB', codec='.jpg', transforms=transforms)
	dataset = ComposeHorse2ZebraDataset(horse_dataset, zebra_dataset)
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

	horse_g_model = Generator(img_channels=3, num_features=64, num_residuals=9).to(device)
	zebra_g_model = Generator(img_channels=3, num_features=64, num_residuals=9).to(device)

	horse_d_model = Discriminator(in_channels=3, features=[32, 64, 128, 256, 512]).to(device)
	zebra_d_model = Discriminator(in_channels=3, features=[32, 64, 128, 256, 512]).to(device)

	l1_criteria = nn.L1Loss()
	mse_criteria = nn.MSELoss()

	d_optimizer = torch.optim.Adam(
	    list(horse_d_model.parameters()) + list(zebra_d_model.parameters()),
	    lr=learning_rate,
	    betas=(0.5, 0.999)
	)
	g_optimizer = torch.optim.Adam(
	    list(horse_g_model.parameters()) + list(zebra_g_model.parameters()),
	    lr=learning_rate,
	    betas=(0.5, 0.999)
	)

	g_scaler = GradScaler()
	d_scaler = GradScaler()

	d_losses = []
	g_losses = []

	for e in range(n_epochs):
	  d_loss, g_loss = train_epoch(
	      horse_d_model=horse_d_model,
	      zebra_d_model=zebra_d_model,
	      horse_g_model=horse_g_model,
	      zebra_g_model=zebra_g_model,
	      loader=train_loader,
	      d_optimizer=d_optimizer,
	      g_optimizer=g_optimizer,
	      l1_criteria=l1_criteria,
	      mse_criteria=mse_criteria,
	      d_scaler=d_scaler,
	      g_scaler=g_scaler,
	      cycle_lambda=10,
	      device=device
	    )

	  gen_zebra_from_horse(zebra_g_model, device)
	  print(f'epoch: {e}. D loss: {d_loss}, G loss: {g_loss}')

	  d_losses.append(d_loss)
	  g_losses.append(g_loss)

	plt.plot(range(n_epochs), d_losses, label='D loss')
	plt.plot(range(n_epochs), g_losses, label='G loss')
	plt.legend()
	plt.show()

