from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self._tokenizer = None

    def load(self):
        self._model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
        self._model.to("cuda")

    def predict(self, model_input: Any, max_length=100, do_sample=True, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=1) -> str:
        x = self._tokenizer.encode(model_input, return_tensors='pt')
        y = self._model.generate(x.cuda(), max_length=max_length, do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature, num_return_sequences=num_return_sequences, eos_token_id=self._tokenizer.eos_token_id)
        generated_code = self._tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return generated_code
