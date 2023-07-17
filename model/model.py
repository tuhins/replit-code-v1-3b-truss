from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

class Model:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
        
        config = AutoConfig.from_pretrained(
            "replit/replit-code-v1-3b",
            trust_remote_code=True
        )
        config.init_device="cuda:0"
        config.attn_config['attn_impl'] = 'triton'
        self._model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True, config=config, )
        self._model.to(device='cuda:0', dtype=torch.bfloat16)

    def predict(self, model_input: Dict) -> str:
        prompt = model_input.pop("prompt")
        defaults = {
         "max_length":100, "do_sample":True, "top_p":0.95, "top_k":4, "temperature":0.2
        }
        for k,v in defaults.items():
            model_input.setdefault(k, v)
        x = self._tokenizer.encode(prompt, return_tensors='pt')
    
        y = self._model(x.cuda())#),  **model_input, num_return_sequences=1)
        generated_code = self._tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return generated_code
