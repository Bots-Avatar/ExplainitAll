import numpy as np
import torch
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset


class StringDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, texts: list, block_size=256):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        self.examples = []

        for text in texts:
            if len(text)==0: 
                continue
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            if len(tokenized_text) >= block_size:
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                    self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size]))
            else:
              self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text))
    
    def __len__(self):
            return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

class GPTProjectionTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def set_variety(self, bias_mask, variety=0., min_prob=3e-3):
        bias = np.zeros((self.model.lm_head.out_features,))
        coef_mask = np.log2(variety + min_prob) / np.log2(np.e)
        bias += coef_mask

        for token in bias_mask:
            bias[token] = 0

        b_tensor = torch.tensor(bias, dtype=torch.float32)
        out_gpt_layer = torch.nn.Linear(in_features=self.model.lm_head.in_features,
                                        out_features=self.model.lm_head.out_features, bias=True)
        out_gpt_layer.weight = self.model.lm_head.weight
        out_gpt_layer.bias.data.copy_(b_tensor)
        self.model.lm_head = out_gpt_layer

    def load_dataset(self, texts, block_size=256):
        dataset = StringDataset(
            tokenizer=self.tokenizer,
            texts=texts,
            block_size=block_size,
        )
        return dataset

    def create_data_collator(self, mlm=False):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=mlm,
        )
        return data_collator

    def train(self, train_texts, bias_mask, variety=0.0, output_dir="new_gpt", last_k=10,
              per_device_train_batch_size=2, num_train_epochs=3, save_steps=1000, device=None):

        self.set_variety(bias_mask, variety=variety)
        self.model.to(device)

        params = []

        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if "c_proj.weight" in name and "mlp" in name:
                params.append(param)

        for param in params[-last_k:]:
            param.requires_grad = True

        train_dataset = self.load_dataset(train_texts)
        if len(train_dataset) == 0:
            raise ValueError("Dataset is empty. Ensure that the input texts are not empty and of sufficient length.")

        data_collator = self.create_data_collator()

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()
