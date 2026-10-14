import threading
from typing import List, Optional, Union

from .openrouter_client import MAX_TOP_LOGPROBS, ChatResult, TokenLogprob
from .perturbations import normalize_messages
from .surrogate import chat_prompt_text


class LocalChatClient:
    """
    Локальная модель Hugging Face с интерфейсом OpenRouterClient.chat: генерация, logprobs и prefill.
    Нужна для бесплатной отладки и для проверки методов на модели, у которой есть и «API», и веса:
    расхождения из-за квантования у провайдера здесь исключены.
    """

    def __init__(self, model, tokenizer, name: Optional[str] = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.name = name or getattr(model.config, 'name_or_path', 'local')
        self._lock = threading.Lock()
        self.n_requests = 0
        self.total_cost = 0.0

    @classmethod
    def from_pretrained(cls, name_or_path: str, device: Optional[str] = None, torch_dtype=None) -> 'LocalChatClient':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch_dtype)
        if device:
            model = model.to(device)
        return cls(model, tokenizer, name=name_or_path)

    def chat(self,
             messages: Union[str, List[dict]],
             model: Optional[str] = None,
             temperature: Optional[float] = 0.0,
             max_tokens: Optional[int] = None,
             seed: Optional[int] = None,
             logprobs: bool = False,
             top_logprobs: Optional[int] = None,
             provider=None,
             extra: Optional[dict] = None,
             sample_id: int = 0,
             use_cache: bool = True) -> ChatResult:
        import torch

        messages = normalize_messages(messages)
        prefill = ''
        if messages and messages[-1].get('role') == 'assistant':
            prefill = messages[-1]['content']
            messages = messages[:-1]
        text = chat_prompt_text(self.tokenizer, messages) + prefill

        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        do_sample = bool(temperature)
        generation_kwargs = dict(max_new_tokens=max_tokens or 256, do_sample=do_sample,
                                 # обработчики из generation_config (штраф за повторы, top-k/top-p) не применяются
                                 repetition_penalty=1.0, top_k=None, top_p=None,
                                 output_logits=True, return_dict_in_generate=True,
                                 pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if do_sample:
            generation_kwargs['temperature'] = temperature

        with self._lock, torch.no_grad():
            if do_sample:
                torch.manual_seed((seed or 0) + sample_id)
            out = self.model.generate(**inputs, **generation_kwargs)
            self.n_requests += 1

        new_ids = out.sequences[0, inputs['input_ids'].shape[1]:].tolist()
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        finish_reason = 'length'
        for i, token_id in enumerate(new_ids):
            if token_id in eos_ids:
                new_ids, finish_reason = new_ids[:i], 'stop'
                break

        steps = None
        if logprobs or top_logprobs is not None:
            k = min(top_logprobs or 0, MAX_TOP_LOGPROBS)
            steps = []
            for i, token_id in enumerate(new_ids):
                log_probs = torch.log_softmax(out.logits[i][0].float(), dim=-1)
                top = torch.topk(log_probs, k) if k else None
                steps.append(TokenLogprob(
                    token=self.tokenizer.decode([token_id]),
                    logprob=float(log_probs[token_id]),
                    top_logprobs=[(self.tokenizer.decode([int(t)]), float(lp))
                                  for lp, t in zip(top.values, top.indices)] if top is not None else []))

        return ChatResult(text=self.tokenizer.decode(new_ids, skip_special_tokens=True), finish_reason=finish_reason,
                          model=self.name, provider='local', usage={'cost': 0.0}, cost=0.0, logprobs=steps,
                          reasoning=None, cached=False, raw={})
