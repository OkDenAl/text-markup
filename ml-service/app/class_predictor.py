import torch
import joblib
import numpy as np

from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from scipy.spatial.distance import cdist

device = torch.device('cpu')


class Classificator:
    def __init__(self):
        self.bert = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
        self.tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
        self.bert.to(device)

        self.kmeans = joblib.load('model_data/kmeans_15_clusterss_2005.pkl')

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


class Classificator2:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v1')
        self.model = AutoModel.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v1')
        self.model.to(device)

        self.kmeans = joblib.load('model_data/kmeans_15_clusterss_2005.pkl')

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def predict(self, embedding):
        cluster_labels = self.kmeans.predict(list(embedding.cpu().detach().numpy()))
        return cluster_labels[0]

    def predict_cluster_proba(self, text):
        text_embedding = self.get_embeddings(text)

        text_embedding = text_embedding.cpu().numpy().astype('float64')

        cluster = self.kmeans.predict(text_embedding.reshape(1, -1))[0]

        distances = cdist(text_embedding.reshape(1, -1), self.kmeans.cluster_centers_, 'minkowski')[0]

        # Преобразуем расстояния в "вероятности", используя softmax
        probabilities = np.exp(-distances) / np.sum(np.exp(-distances))

        return cluster, probabilities

