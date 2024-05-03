from docutils.nodes import classifier
from fastapi import FastAPI
import torch
from pydantic import BaseModel
from own_lsg_converter import MYLSGConverter
from natasha import (
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc, Segmenter,
    NewsNERTagger,
    norm
)
from class_predictor import Classificator, Classificator2


class Item(BaseModel):
    text: str


app = FastAPI()

converter = MYLSGConverter(max_sequence_length=4096)
model, tokenizer = converter.convert_from_pretrained(
    'KodKio/ruBert-base-finetuned-ner',
    architecture="BertForTokenClassification"
)

classificator = Classificator()
classificator2 = Classificator2()


class TagTransformer:
    def __init__(self):
        self.morph_vocab = MorphVocab()
        self.segmenter = Segmenter()
        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)
        self.syntax_parser = NewsSyntaxParser(emb)
        self.ner_tagger = NewsNERTagger(emb)

    def transform_tag(self, tag):
        return tag.replace(" ##ии", "ии") \
            .replace(" ##и", "й") \
            .replace("нии", "ний") \
            .replace("вои", "вой") \
            .replace(" ##", "") \
            .replace(" , ", ", ") \
            .replace(" . ", ".") \
            .replace(" ( ", "(") \
            .replace(" )", ")") \
            .replace(" ) ", ")") \
            .strip().title()

    def normalize_tag(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.parse_syntax(self.syntax_parser)
        if len(doc.tokens) == 1:
            doc.tokens[0].lemmatize(self.morph_vocab)
            return doc.tokens[0].lemma
        return norm.syntax_normalize(self.morph_vocab, doc.tokens)

    def __call__(self, text):
        return self.normalize_tag(self.transform_tag(text))


def transform_model_output(token_list, token_labels):
    tag = ""
    tag_label = ""
    tags = []
    tag_labels = []

    normalizer = TagTransformer()

    for i in range(1, len(token_list)):
        if token_labels[i] == "O":
            if tag != "":
                normalized = normalizer(tag)
                if tag_label != "O":
                    tags.append(normalizer.transform_tag(normalized))
                    tag_labels.append(tag_label)
                tag = ""
            tag_label = "O"
            continue
        if token_labels[i].startswith("B"):
            if tag != "" and token_labels[i][2:] != tag_label:
                normalized = normalizer(tag)
                if tag_label != "O":
                    tags.append(normalizer.transform_tag(normalized))
                    tag_labels.append(tag_label)
                tag = token_list[i]
                tag_label = token_labels[i][2:]
                continue
            tag += (" " + token_list[i])
            tag_label = token_labels[i][2:]
        if token_labels[i].startswith("I"):
            tag += (" " + token_list[i])

    return tags, tag_labels


@app.get("/api/v1/tokens")
async def get_tokens(item: Item):
    try:
        text = item.text
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

        token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

        print(token_list)

        token_labels = [model.config.id2label[label_id] for label_id in predicted_labels]

        token_list, token_labels = transform_model_output(token_list, token_labels)

        print(token_list, token_labels)

        return {"tags": token_list, "labels": token_labels}
    except Exception as e:
        print(e)


@app.get("/api/v1/class")
async def get_class(item: Item):
    # classes = ["Home", "Health", "Celebrities", "Films and Shows", "Incidents", "Researches"] # kmeans_model
    # classes = ["Celebrities", "Incidents", 'Weather',
    #           "Family", "Sport", "Health", "Realty", "Home", "Films & Shows"] # kmeans_model2
    # classes = ["Celebrities", "Films & Shows", "Incidents", "Family", "Weather",
    #            "Sports", "Money", "Health", "Interior", "Social Security"] # kmeans_model_10_clusters
    # classes = ["Crimes", "Food", "Social Security", "Celebrities", "Films & Shows", "Regional news", "Family",
    #            "Incidents", "Weather", "Sports", "Finances", "Health"]  # kmeans_model_12_clusters
    classes = ["Design", "Foreign Films", "Sports", "Incidents", "Celebrities", "Shows", "Researches", "Health", "Food",
               "Regional news", "Children", "Russian Films", "Doctors", "Home", "Weather"] # kmeans_15_clusters_new_model.pkl

    try:
        text = item.text
        # embedding = classificator.get_embeddings([text])
        # class_label = classificator.predict(embedding)
        embedding = classificator2.get_embeddings(text)
        class_label = classificator2.predict(embedding)
        return {"class": classes[class_label]}
    except Exception as e:
        print(e)
