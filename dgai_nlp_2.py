# -*- coding: utf-8 -*-

import string
import math
import typing as t
from time import perf_counter

import torch
import torch.nn as nn


def build_char_vocab(filename: str) -> t.Dict[str, int]:
  vocab = {
      '<start>': 0,
      '<end>': 1,
      '<pad>': 2,
      '<unk>': 3
  }

  with open(filename, 'r') as f:
    for line in f.readlines():
      line = line.translate(str.maketrans('', '', string.punctuation))
      line = line.replace('»', '')
      line = line.replace('«', '')
      line = line.replace('—', '')
      line = line.replace('…', '')

      for char in line:
        # char = char.lower()
        if char not in vocab:
          vocab[char] = len(vocab)
  return vocab


def make_train_pair(token_ids: t.List[int], seq_length: int, pad_token: int) -> t.Tuple[str, str]:
  if len(token_ids) > seq_length:
    _token_ids = token_ids[:seq_length]
  elif len(token_ids) < seq_length:
    n_tokens_to_pad = (seq_length - len(token_ids))
    _token_ids = token_ids + [pad_token] * n_tokens_to_pad
  else:
    _token_ids = token_ids

  source_tokens = _token_ids[:-1]
  target_tokens = _token_ids[1:]
  return source_tokens, target_tokens


def tokenize_string(vocab: t.Dict[str, int], text: str) -> t.List[int]:
  tokens = [vocab.get(char, vocab['<pad>']) for char in text]
  if vocab['<pad>'] in tokens:
    print('ALERT PADDING!!!')
    print(text)
    print(tokens)
    print('pad index', tokens.index(vocab['<pad>']))
    print('pad value', text[tokens.index(vocab['<pad>'])])
  return tokens


def create_batches(
    vocab: t.Dict[str, int],
    data: str,
    context_length: int,
    batch_size: int,
    n_batches: int
) -> t.List[t.List[int]]:

  total_examples = 0

  while total_examples < n_batches:

    source_batch = []
    target_batch = []
    sample_indexes = torch.randint(0, len(data) - (context_length + 1), size=(batch_size,))

    for index in sample_indexes.tolist():
      text = data[index: index + context_length + 1]
      source_ids, target_ids = make_train_pair(
          tokenize_string(vocab, text),
          seq_length=context_length + 1,
          pad_token=vocab['<pad>']
      )
      source_batch.append(source_ids)
      target_batch.append(target_ids)

    yield torch.LongTensor(source_batch), torch.LongTensor(target_batch)

    total_examples += 1



class Encoder(nn.Module):

  def __init__(self, vocab: t.Dict[str, int], n_pos: int, emb_dim: int) -> None:
    super().__init__()
    self.vocab_embed = nn.Embedding(len(vocab), emb_dim)
    self.pos_embed = nn.Embedding(n_pos, emb_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, t = x.shape
    x_embed = self.vocab_embed(x)
    pos_x = self.pos_embed(torch.arange(t)).unsqueeze(0)
    return x_embed + pos_x


class MultiHeadAttentionLayer(nn.Module):

  def __init__(self, emb_dim: int, n_heads: int, context_length: int) -> None:
    super().__init__()
    self.n_heads = n_heads
    self.head_dim = emb_dim // n_heads
    self.q_proj = nn.Linear(self.head_dim, self.head_dim)
    self.k_proj = nn.Linear(self.head_dim, self.head_dim)
    self.v_proj = nn.Linear(self.head_dim, self.head_dim)
    self.output = nn.Linear(emb_dim, emb_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch, seq_len, emb = x.shape
    x = x.view((batch, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
    # x: [batch, n_heads, seq_len, head_dim]
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    pos_mask = torch.tril(torch.ones(seq_len, seq_len))

    attention = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
    attention = attention.masked_fill(pos_mask == 0, -1e9)
    attention = nn.Softmax(dim=3)(attention)
    attention_values = attention @ v

    attention_values = attention_values.transpose(1, 2).contiguous()
    attention_values = attention_values.view(batch, seq_len, emb)
    x = self.output(attention_values)

    return x


class TransformerBlock(nn.Module):

  def __init__(self, emb_dim: int, n_heads: int, context_length: int) -> None:
    super().__init__()
    self.ln_1 = nn.LayerNorm(emb_dim)
    self.ln_2 = nn.LayerNorm(emb_dim)
    self.mh_at = MultiHeadAttentionLayer(emb_dim, n_heads, context_length)
    self.mlp = nn.Sequential(
        nn.Linear(emb_dim, 4 * emb_dim),
        nn.GELU(),
        nn.Linear(4 * emb_dim, emb_dim)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.mh_at(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


class CustomGPT(nn.Module):

  def __init__(
      self,
      vocab: t.Dict[str, int],
      n_pos: int,
      emb_dim: int,
      n_layers: int,
      n_heads: int,
      context_length: int,
      drop: float = 0.3
  ) -> None:
    super().__init__()
    self.encoder = Encoder(vocab, n_pos, emb_dim)
    self.drop = nn.Dropout(drop)
    self.layers = nn.ModuleList([
        TransformerBlock(emb_dim=emb_dim, n_heads=n_heads, context_length=context_length)
        for _ in range(n_layers)
    ])
    self.output_head = nn.Linear(emb_dim, len(vocab))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    x = self.drop(x)
    for layer in self.layers:
      x = layer(x)
    x = self.drop(x)
    return self.output_head(x)


def train_epoch(
    model: nn.Module,
    loader: t.Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criteria: t.Callable
) -> float:

  epoch_loss = 0.
  n_iter = 0

  for i, (source_batch, target_batch) in enumerate(loader):
    outputs = model(source_batch)
    # outputs: [batch, seq_len, n_chars]
    outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
    target_batch = target_batch.ravel()

    loss = criteria(outputs, target_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    epoch_loss += loss.detach().item()
    n_iter += 1

  return epoch_loss


@torch.inference_mode
def generate(
    vocab: t.Dict[str, int],
    model: nn.Module,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 20,
    top_k: int = 5
) -> str:
  model.eval()
  prompt_ids = tokenize_string(vocab, prompt)
  prompt_ids_pt = torch.LongTensor([prompt_ids])
  id2char = {pos: char for char, pos in vocab.items()}
  result = prompt

  for i in range(max_tokens):
    output = model(prompt_ids_pt)
    topk_v, topk_idx = torch.topk(output[:, -1, :], k=top_k, dim=1)

    if temperature > 0.:
      topk_v = topk_v / temperature
      probs = nn.Softmax(dim=1)(topk_v)

      next_index = torch.multinomial(probs, num_samples=1)[0].item()
      next_char = topk_idx[0, next_index].item()
    else:
      next_char = torch.argmax(output[:, -1, :], dim=1).item()

    result = result + id2char[next_char]

    if prompt_ids_pt.shape[-1] < model.encoder.pos_embed.weight.shape[0]:
      prompt_ids_pt = torch.cat([prompt_ids_pt, torch.LongTensor([[next_char]])], dim=1)
    else:
      prompt_ids_pt = torch.cat([prompt_ids_pt[:, 1:], torch.LongTensor([[next_char]])], dim=1)

    if next_char == vocab['<end>']:
      print('break')
      break

  return result


def process_line(line: str) -> str:
  line = line.translate(str.maketrans('', '', string.punctuation))
  line = line.replace('»', '')
  line = line.replace('«', '')
  line = line.replace('—', '')
  line = line.replace('…', '')
  return line


if __name__ == '__main__':
with open('data.txt', 'r') as f:
  text_data = ''.join([process_line(line) for line in f.readlines()])

vocab = build_char_vocab('data.txt')
id2char = {pos: char for char, pos in vocab.items()}

n_epochs = 200
context_length = 64
criteria = nn.CrossEntropyLoss()
gpt = CustomGPT(vocab, n_pos=context_length, context_length=context_length, emb_dim=128, n_layers=4, n_heads=2, drop=0.3)
optimizer = torch.optim.Adam(gpt.parameters(), lr=1e-3, betas=(0.7, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100 * 100, 100 * 125, 100 * 150], gamma=0.1)

gpt.train()
for e in range(n_epochs):
  loader = create_batches(vocab, data=text_data, batch_size=4, context_length=context_length, n_batches=100)
  epoch_loss = train_epoch(gpt, loader, optimizer, scheduler, criteria)
  print(f'Epoch: {e + 1}. Loss: {epoch_loss}. LR: {optimizer.param_groups[0]["lr"]}')



t1 = perf_counter()

prompt="""Три богатыря"""
output_text = generate(vocab, gpt, prompt, temperature=0.5, max_tokens=500)
print(output_text)
print('*' * 89)

t2 = perf_counter()

print('\ntime took', t2 - t1, 'sec')
