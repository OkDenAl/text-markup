import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel


class Embedder:
    def __init__(self, model, device='cuda'):
        self.model = AutoModel.from_pretrained(model)
        self.device = device
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.to(device)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, add_special_tokens=True)
        with torch.no_grad:
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
