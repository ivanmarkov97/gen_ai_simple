import typing as t

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt


torch.manual_seed(0)


def create_discriminator() -> nn.Module:
  # creates discriminator network
  return nn.Sequential(
      nn.Linear(2, 256),
      nn.ReLU(),
      nn.Dropout(0.2),

      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(0.2),

      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Dropout(0.2),

      nn.Linear(64, 1),
      nn.Sigmoid()
  )

def create_generator() -> nn.Module:
  # creates generator network
  return nn.Sequential(
      nn.Linear(2, 16),
      nn.ReLU(),

      nn.Linear(16, 32),
      nn.ReLU(),

      nn.Linear(32, 2)
  )


def select_device() -> str:
  return 'cuda' if torch.cuda.is_available() else 'cpu'


# TRAIN DISCRIMINATOR
def train_dis_on_real_batch(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: t.Callable,
    real_batch: torch.tensor,
    device: str
  ) -> float:

  batch_size = real_batch.shape[0]
  real_samples = real_batch.to(device)
  opt.zero_grad()

  output = model(real_samples)
  true_labels = torch.ones((batch_size, 1), device=device)
  loss = loss_fn(output, true_labels)
  loss.backward()

  loss_value = loss.detach().cpu().item()

  opt.step()
  return loss_value


def train_dis_on_fake_batch(
    model: nn.Module,
    fake_creator: nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: t.Callable,
    batch_size: int,
    device: str
) -> float:

  noize = torch.rand((batch_size, 2))
  noize = noize.to(device)
  fake_output = fake_creator(noize)

  opt.zero_grad()

  output = model(fake_output)
  true_labels = torch.zeros((batch_size, 1), device=device)
  loss = loss_fn(output, true_labels)
  loss.backward()

  loss_value = loss.detach().cpu().item()

  opt.step()
  return loss_value


# TRAIN GENERATOR
def train_gen(
    model: nn.Module,
    dis: nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: t.Callable,
    batch_size: int,
    device: str
) -> float:

  noize = torch.rand((batch_size, 2))
  noize = noize.to(device)
  opt.zero_grad()

  fake_output = model(noize)
  dis_detections = dis(fake_output)
  true_labels = torch.ones((batch_size, 1), device=device)
  loss = loss_fn(dis_detections, true_labels)
  loss.backward()

  loss_value = loss.detach().cpu().item()

  opt.step()
  return loss_value


# USING GENERATOR
def generate(model: nn.Module, batch_size: int, device: str) -> torch.tensor:
  with torch.no_grad():
    noize = torch.rand((batch_size, 2))
    noize = noize.to(device)
    gen_output = model(noize)
    gen_output = gen_output.detach().cpu()
  return gen_output


def train_epoch(
    dis: nn.Module,
    gen: nn.Module,
    loader: DataLoader,
    loss_fn: t.Callable,
    dis_opt: torch.optim.Optimizer,
    gen_opt: torch.optim.Optimizer,
    device: str
) -> t.Tuple[float, float]:

  dis_epoch_loss = 0
  gen_epoch_loss = 0

  for batch in loader:
    dis_true_loss = train_dis_on_real_batch(
        model=dis,
        opt=dis_opt,
        loss_fn=loss_fn,
        real_batch=batch,
        device=device
    )
    dis_fake_loss = train_dis_on_fake_batch(
        model=dis,
        fake_creator=gen,
        opt=dis_opt,
        loss_fn=loss_fn,
        batch_size=batch.shape[0],
        device=device
    )
    gen_loss = train_gen(
        model=gen,
        dis=dis,
        opt=gen_opt,
        loss_fn=loss_fn,
        batch_size=batch.shape[0],
        device=device
    )

    dis_epoch_loss += (dis_true_loss + gen_epoch_loss) / 2
    gen_epoch_loss += gen_loss

  dis_epoch_loss = dis_epoch_loss / len(loader)
  gen_epoch_loss = gen_epoch_loss / len(loader)

  return dis_epoch_loss, gen_epoch_loss


def get_observed_data(n_dots: int, start: float, end: float) -> torch.tesnor:
	observed_data = torch.zeros((n_dots, 2))
	observed_data[:, 0] = torch.arange(start, end, step=(end - start) / n_dots)
	observed_data[:, 1] = torch.sin(observed_data[:, 0])
	return observed_data


if __name__ == '__main__':

	N_DOTS = 2000
	START_V, END_V = 0., 10.

	observed_data = get_observed_data(N_DOTS, START_V, END_V)

	loader = DataLoader(observed_data, batch_size=128, shuffle=True)
	device = select_device()
	loss_function = nn.BCELoss().to(device)
	gen = create_generator().to(device)
	dis = create_discriminator().to(device)

	gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.001)
	dis_optimizer = torch.optim.Adam(dis.parameters(), lr=0.001)

	# train all
	dis_losses = []
	gen_losses = []

	for e in range(2_000):

	  dis_epoch_loss, gen_epoch_loss = train_epoch(
	      dis=dis,
	      gen=gen,
	      loader=loader,
	      loss_fn=loss_function,
	      dis_opt=dis_optimizer,
	      gen_opt=gen_optimizer,
	      device=device
	  )

	  dis_losses.append(dis_epoch_loss)
	  gen_losses.append(gen_epoch_loss)

	  if e % 100 == 0:
	    print('Dis loss', dis_epoch_loss, '\t', 'Gen loss', gen_epoch_loss)
	    generated_data = generate(model=gen, batch_size=100, device=device)
	    generated_data = generated_data.numpy()

	    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
	    axes[0].scatter(observed_data[:, 0], observed_data[:, 1], label='Obs')
	    axes[0].scatter(generated_data[:, 0], generated_data[:, 1], label='Gen')
	    axes[0].grid(True)
	    axes[0].legend()

	    plot_x = list(range(e + 1))
	    axes[1].plot(plot_x, dis_losses, label='Dis losses')
	    axes[1].plot(plot_x, gen_losses, label='Gen losses')
	    axes[1].grid(True)
	    axes[1].legend()

	    plt.show()

	dis_traced = torch.jit.trace(func=dis, example_inputs=torch.rand((2, 2)))
	gen_traced = torch.jit.trace(func=gen, example_inputs=torch.rand((2, 2)))

	dis_traced.save('discriminator_traced.pt')
	gen_traced.save('generator_traced.pt')

	gen_traced = torch.jit.load('generator_traced.pt', map_location=device)
	generated_data = generate(model=gen_traced, batch_size=1000, device=device)
	generated_data = generated_data.numpy()
