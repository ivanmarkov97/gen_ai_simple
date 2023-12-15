# -*- coding: utf-8 -*-

import os
import typing as t

import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer


class MyDataset(Dataset):

  def __init__(
      self,
      file_path: str,
      tokenizer: GPT2TokenizerFast,
      context_length: int,
  ) -> None:

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
      all_text = f.read()

    self.file_path = file_path
    self.tokenizer = tokenizer
    self.context_length = context_length
    self.n_samples = len(all_text) - context_length
    self.all_text = all_text

  def __getitem__(self, index: int) -> t.Dict[str, torch.Tensor]:

    text_data = self.all_text[index: index + self.context_length]

    encoded = self.tokenizer(
        text_data,
        padding='max_length',
        truncation=True,
        max_length=self.context_length,
        return_tensors='pt'
    )

    return encoded


  def __len__(self) -> int:
    return self.n_samples


if __name__ == '__main__':
	vocab_size = 500
	context_length = 128

	gpt2_config = GPT2Config(
	    vocab_size=vocab_size + 1,
	    n_positions=context_length,
	    n_embd=256,
	    n_layer=4,
	    n_head=2
	)

	tokenizer = ByteLevelBPETokenizer()

	tokenizer.train(
	    files=['data.txt'],
	    vocab_size=vocab_size,
	    min_frequency=2,
	    show_progress=True
	)

	if not os.path.exists('my_tokenizer'):
	  os.mkdir('my_tokenizer')

	tokenizer.save_model('my_tokenizer')

	gpt_model = GPT2LMHeadModel(gpt2_config)
	gpt_tokenizer = GPT2TokenizerFast.from_pretrained('./my_tokenizer')
	gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

	dataset = MyDataset(
	    tokenizer=gpt_tokenizer,
	    file_path="./data.txt",
	    context_length=context_length,
	)

	data_collator = DataCollatorForLanguageModeling(tokenizer=gpt_tokenizer, mlm=False)

	training_args = TrainingArguments(
	    output_dir="./saltan_model",
	    evaluation_strategy='epoch',
	    learning_rate=2e-4,
	    weight_decay=0.01,
	    num_train_epochs=10,
	    prediction_loss_only=True,
	    overwrite_output_dir=True,
	    logging_steps=100
	)

	trainer = Trainer(
	    model=gpt_model,
	    args=training_args,
	    data_collator=data_collator,
	    train_dataset=dataset,
	    eval_dataset=dataset
	)

	trainer.train()

	gpt_model = gpt_model.cpu()

	model_inputs = gpt_tokenizer('Белка', return_tensors='pt')
	greedy_output = gpt_model.generate(**model_inputs, max_new_tokens=100)
	print("Output:\n" + 100 * '-')
	print(gpt_tokenizer.decode(greedy_output[0], skip_special_tokens=True))

