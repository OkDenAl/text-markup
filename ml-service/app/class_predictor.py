import torch
import joblib

from transformers import AutoModel, AutoTokenizer

device = torch.device('cpu')


class Classificator:
    def __init__(self):
        self.bert = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
        self.tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
        self.bert.to(device)

        self.kmeans = joblib.load('model_data/kmeans_model2.pkl')

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding='longest', return_tensors="pt", add_special_tokens=True, max_length=50, truncation=True
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = self.bert(**encoded_input)
        return model_output.last_hidden_state[:, 0]

    def predict(self, embedding):
        cluster_labels = self.kmeans.predict(list(embedding.cpu().detach().numpy()))
        return cluster_labels[0]
