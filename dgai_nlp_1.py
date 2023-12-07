# -*- coding: utf-8 -*-

import typing as t
from time import perf_counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset


def build_char_vocab(filename: str) -> t.Dict[str, int]:
  vocab = {
      '<start>': 0,
      '<end>': 1,
      '<pad>': 2,
      '<unk>': 3
  }

  with open(filename, 'r') as f:
    for line in f.readlines():
      for char in line:
        if char == '\n':
          continue
        if char not in vocab:
          vocab[char] = len(vocab)
  return vocab


def text_to_ids(
    vocab: t.Dict[str, int],
    texts: t.List[t.List[str]],
    max_len: int
) -> t.List[t.List[int]]:

  output = []
  for text in texts:
    text = text.replace('\n', '')
    if len(text) > max_len and max_len > 2:
      cut_off_index = max_len - 2 # <start> + <end> tokens
      text = text[:cut_off_index]

    raw_token_ids = [vocab.get(char, vocab['<unk>']) for char in text]

    n_paddings = max_len - (len(text) + 2)
    raw_token_ids = [vocab['<start>']] + raw_token_ids + [vocab['<end>']]
    padded_token_ids = raw_token_ids + [vocab['<pad>']] * n_paddings

    assert len(padded_token_ids) == max_len, f'{len(padded_token_ids)} {max_len}'

    output.append(padded_token_ids)
  return output


def ids_to_text(
    vocab: t.Dict[str, int],
    token_ids: t.List[int]
) -> t.List[t.List[str]]:

  output = []
  id2char = {pos: char for char, pos in vocab.items()}

  for ids in token_ids:
    text_tokens = [id2char[_id] for _id in ids]
    output.append(text_tokens)
  return output


class NamesDataset(Dataset):

  def __init__(self, filename: str) -> None:
    self.all_texts: t.List[str] = []
    self.max_len: int = 0
    self._vocab: t.Dict[str, int] = build_char_vocab(filename)

    with open(filename, 'r') as f:
      for line in f.readlines():
        line = line.replace('\n', '')
        self.all_texts.append(line)
        if len(line) > self.max_len:
          self.max_len = len(line)

  @property
  def vocab(self) -> t.Dict[str, int]:
    return self._vocab

  def __getitem__(self, index: int) -> t.Dict[str, torch.LongTensor]:
    source_text = self.all_texts[index]
    # print('source_text', source_text)
    source_token_ids = text_to_ids(self._vocab, [source_text], max_len=self.max_len + 2)[0]
    # print('source_token_ids', source_token_ids)

    target_token_ids = source_token_ids[1:] + [self.vocab['<pad>']]
    # print('target_token_ids', target_token_ids)
    target_text = ids_to_text(self._vocab, [target_token_ids])
    # print('target_text', target_text)

    source_token_ids_pt = torch.LongTensor(source_token_ids)
    target_token_ids_pt = torch.LongTensor(target_token_ids)

    return {
        'source_ids': source_token_ids_pt,
        'target_ids': target_token_ids_pt
    }

  def __len__(self) -> int:
    return len(self.all_texts)


class CharLanguageModel(nn.Module):

  def __init__(
      self,
      vocab: t.Dict[int, int],
      embedding_dim: int = 256,
      hidden_size: int = 128,
      num_layers: int = 2,
      dropout: float = 0.2
  ) -> None:
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.char_embedding = nn.Embedding(len(vocab), embedding_dim)
    self.rnn = nn.LSTM(
        input_size=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout,
        bidirectional=False
    )
    self.clf = nn.Linear(hidden_size, len(vocab))

    pad_token_idx = vocab.get('<pad>', -1)
    if pad_token_idx != -1:
      self.char_embedding.weight.data[pad_token_idx].fill_(0)

  def init_hidden_and_cell(self, batch_size: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.zeros((self.num_layers, batch_size, self.hidden_size))
    cell = torch.zeros((self.num_layers, batch_size, self.hidden_size))
    return hidden, cell

  def forward(
      self,
      x: torch.Tensor,
      hidden_and_cell: torch.Tensor
  ) -> t.Tuple[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:
    # Forward pass
    embeddings = self.char_embedding(x)
    output, (hidden, cell) = self.rnn(embeddings)
    # output = output.reshape(-1, self.hidden_size)
    # print(output.shape, hidden.shape, cell.shape)
    return self.clf(output), (hidden, cell)


def create_train_batch(
    vocab: t.Dict[str, int],
    text: t.List[str]
) -> t.Tuple[torch.Tensor, torch.Tensor]:

  source_text = text_to_ids(vocab, [text], max_len=len(text) + 2)[0]
  target_text = source_text[1:] + [vocab['<pad>']]

  for t in range(len(source_text)):
    iter_source_ids = source_text[:t + 1]
    iter_target_text = target_text[:t + 1:]
    yield torch.LongTensor([iter_source_ids]), torch.LongTensor([iter_target_text])


def create_train_batch_v2(
    vocab: t.Dict[str, int],
    text: t.List[str],
    context_length: int
) -> t.Tuple[torch.Tensor, torch.Tensor]:

  source_text = text_to_ids(vocab, [text], max_len=len(text) + 2)[0]
  target_text = source_text + [vocab['<pad>']]
  for t in range(1, len(source_text)):
    start = max(0, t - context_length)
    iter_source_ids = source_text[start: t]
    next_char = target_text[t]
    yield torch.LongTensor([iter_source_ids]), torch.LongTensor([next_char])


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
  hidden_and_cell = model.init_hidden_and_cell(batch_size=1)
  prompt_ids = text_to_ids(vocab, [prompt], max_len=len(prompt) + 2)[0][:-1]
  prompt_ids_pt = torch.LongTensor([prompt_ids])

  for _ in range(max_tokens):
    output, hidden_and_cell = model(prompt_ids_pt, hidden_and_cell)
    topk_v, topk_idx = torch.topk(output[:, -1, :], k=top_k, dim=1)

    if temperature > 0.:
      topk_v = topk_v / temperature
      probs = nn.Softmax(dim=1)(topk_v)

      next_index = torch.multinomial(probs, num_samples=1)[0].item()
      next_char = topk_idx[0, next_index].item()
    else:
      next_char = torch.argmax(output[:, -1, :], dim=1).item()

    prompt_ids_pt = torch.cat([prompt_ids_pt, torch.LongTensor([[next_char]])], dim=1)

    if next_char == vocab['<end>']:
      break

  generated_ids = prompt_ids_pt.detach().tolist()
  generated_tokens = ids_to_text(vocab, generated_ids)[0]
  return ''.join(generated_tokens)


# v1
if __name__ == '__main__':
  dataset = NamesDataset('men_names.txt')

  lm = CharLanguageModel(
      dataset.vocab,
      embedding_dim=256,
      hidden_size=128,
      num_layers=2,
      dropout=0.2
  )
  optimizer = torch.optim.Adam(lm.parameters(), lr=1e-3)
  critetia = nn.CrossEntropyLoss()

  for i in range(75):
    epoch_loss = 0
    hidden_and_cell = lm.init_hidden_and_cell(batch_size=1)

    for text in dataset.all_texts:
      # print(text)
      for source, target in create_train_batch(vocab, text=text):
        output, hidden_and_cell = lm(source, hidden_and_cell)
        output = output.reshape(-1, output.shape[-1])
        target = target.squeeze(0)
        # print(output.shape, hidden_and_cell[0].shape, hidden_and_cell[1].shape, target.shape)

        loss = critetia(output, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(lm.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss += loss.item()

    print(i, epoch_loss)


  t1 = perf_counter()

  for _ in range(10):
    generated_text = generate(vocab, lm, prompt='N', temperature=0.9)
    print(generated_text)

  t2 = perf_counter()

  print('\ntime took', t2 - t1, 'sec')
