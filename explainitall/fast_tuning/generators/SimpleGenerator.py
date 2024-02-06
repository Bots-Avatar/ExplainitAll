from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import numpy as np
import torch

class TextGenerator:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)

        self.y_set = np.load(f'{path}/set.data.npy')
        self.set_variety_of_answers(0.0)

        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)

    def set_variety_of_answers(self, variety=0, min_prob=3e-3):
        bias = np.zeros((self.model.lm_head.out_features,))
        coef_mask = np.log2(variety + min_prob) / np.log2(np.e)
        bias += coef_mask

        for token in self.y_set:
            bias[token] = 0

        b_tensor = torch.tensor(bias, dtype=torch.float32)
        self.model.lm_head.bias.data.copy_(b_tensor)

    def generate(self, start_text, args=None):
        if args is None:
            args = {
                "temperature": 0.7,
                "no_repeat_ngram_size": 2,
                "num_beams": 12,
                "top_k": 30,
            }
        return self.pipeline(start_text, **args)["generated_text"]
