import numpy as np
import torch
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


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

    def load_dataset(self, file_path, block_size=256):
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=file_path,
            block_size=block_size,
        )
        return dataset

    def create_data_collator(self, mlm=False):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=mlm,
        )
        return data_collator

    def train(self, train_file_path, bias_mask, variety=0.0, output_dir="new_gpt", last_k=10,
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

        train_dataset = self.load_dataset(train_file_path)
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
