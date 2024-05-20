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
from transformers import pipeline
from keybert import KeyBERT

from collections import OrderedDict
from class_predictor import Classificator, Classificator2


class Item(BaseModel):
    text: str


class KeywordsReq(BaseModel):
    text: str
    keyword_count: int


app = FastAPI()

converter = MYLSGConverter(max_sequence_length=4096)
model, tokenizer = converter.convert_from_pretrained(
    'KodKio/ruBert-base-finetuned-ner',
    architecture="BertForTokenClassification"
)

classificator = Classificator()
classificator2 = Classificator2()

keywords_model = pipeline("feature-extraction", model="KodKio/rubert-finetuned-keywords")

kw_model = KeyBERT(model=keywords_model)


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
            .replace("нои", "ной") \
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


normalizer = TagTransformer()


def transform_model_output(token_list, token_labels):
    tag = ""
    tag_label = ""
    tags = []
    tag_labels = []

    for i in range(1, len(token_list)):
        if token_labels[i] == "O":
            if tag != "":
                normalized = normalizer(tag)
                if tag_label not in ["O", "AGE", "DATE", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "TIME"]:
                    tags.append(normalizer.transform_tag(normalized))
                    tag_labels.append(tag_label)
                tag = ""
            tag_label = "O"
            continue
        if token_labels[i].startswith("B"):
            if tag != "" and token_labels[i][2:] != tag_label:
                normalized = normalizer(tag)
                if tag_label not in ["O", "AGE", "DATE", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "TIME"]:
                    tags.append(normalizer.transform_tag(normalized))
                    tag_labels.append(tag_label)
                tag = token_list[i]
                tag_label = token_labels[i][2:]
                continue
            tag += (" " + token_list[i])
            tag_label = token_labels[i][2:]
        if token_labels[i].startswith("I"):
            tag += (" " + token_list[i])

    tmp = list(OrderedDict.fromkeys(zip(tags, tag_labels)))

    tags = [i[0] for i in tmp]
    tag_labels = [i[1] for i in tmp]

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


def process_list(input_list):
    input_list = list(input_list)
    max_value = max(input_list)
    max_index = input_list.index(max_value)

    indexes = set()

    result = [(max_index, max_value)]
    indexes.add(max_index)

    for i, value in enumerate(input_list):
        if ((max_value > 0.071 and 0.07 < value and value >= max_value - 0.0008) or
            value >= max_value - 0.0008) and i not in indexes:
            result.append((i, value))
            indexes.add(i)

    for i, value in enumerate(input_list):
        if value > 0.073 and i not in indexes:
            result.append((i, value))
            indexes.add(i)

    return result


@app.get("/api/v1/class")
async def get_class(item: Item):
    # classes = ["Home", "Health", "Celebrities", "Films and Shows", "Incidents", "Researches"] # kmeans_model
    # classes = ["Celebrities", "Incidents", 'Weather',
    #           "Family", "Sport", "Health", "Realty", "Home", "Films & Shows"] # kmeans_model2
    # classes = ["Celebrities", "Films & Shows", "Incidents", "Family", "Weather",
    #            "Sports", "Money", "Health", "Interior", "Social Security"] # kmeans_model_10_clusters
    # classes = ["Crimes", "Food", "Social Security", "Celebrities", "Films & Shows", "Regional news", "Family",
    #            "Incidents", "Weather", "Sports", "Finances", "Health"]  # kmeans_model_12_clusters
    # classes = ["Design", "Foreign Films", "Sports", "Incidents", "Celebrities", "Shows", "Researches", "Health", "Food",
    #           "Regional news", "Children", "Russian Films", "Doctors", "Home",
    #           "Weather"]  # kmeans_15_clusters_new_model.pkl
    classes = ["Regional News", "Kids & Parents", "Sports celebrities", "Design", "Health", "Films & Shows",
               "Incidents", "Diseases",
               "Celebrities", "Money", "Food", "Home", "Politics", 'Weather', "Sports"]  # kmeans_15_clusterss_2005.pkl

    try:
        text = item.text
        # embedding = classificator.get_embeddings([text])
        # class_label = classificator.predict(embedding)

        # embedding = classificator2.get_embeddings(text)
        # class_label = classificator2.predict(embedding)

        cluster, probs = classificator2.predict_cluster_proba(text)
        clusters_with_probs = process_list(probs)

        res = list()
        for cluster in clusters_with_probs:
            res.append(classes[cluster[0]])

        result_string = ", ".join(res)
        return {"class": result_string}

    except Exception as e:
        print(e)


@app.get("/api/v1/keywords")
async def get_keywords(item: KeywordsReq):
    try:
        text = item.text
        kw_count = item.keyword_count

        # keywords_1 = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1))
        # keywords_2 = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 2))
        keywords_3 = kw_model.extract_keywords(text, keyphrase_ngram_range=(3, 3))

        keywords = keywords_3
        # keywords.sort(key=lambda x: x[1], reverse=True)

        words = [x[0] for x in keywords]
        scores = [x[1] for x in keywords]

        for i in range(len(words)):
            print(words[i], scores[i])

        if len(words) < kw_count:
            return {"keywords": words, "scores": scores}

        return {"keywords": words[:kw_count], "scores": scores[:kw_count]}
    except Exception as e:
        print(e)
