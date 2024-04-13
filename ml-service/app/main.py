from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pydantic import BaseModel
from own_lsg_converter import MYLSGConverter
import uvicorn
from pymorphy2 import MorphAnalyzer


class Item(BaseModel):
    text: str


app = FastAPI()

converter = MYLSGConverter(max_sequence_length=4096)
model, tokenizer = converter.convert_from_pretrained(
    'KodKio/ruBert-base-finetuned-ner',
    architecture="BertForTokenClassification"
)


def transform_tag(tag):
    return tag.replace(" ##ии", "ии").replace(" ##и", "й").replace(" ##", "").replace(" . ", ".") \
        .replace(" ( ", "(").replace(" )", ")").replace(" ) ", ")").strip().title()


def normalize_tag(text, morph):
    is_prep = False
    words = text.split()
    normalized_words = []
    for word in words:
        parsed_word = morph.parse(word)[0]
        if is_prep:
            is_prep = False
            normalized_word = word
        else:
            if 'PREP' in parsed_word.tag:
                is_prep = True
            if 'NOUN' in parsed_word.tag:
                if 'sing' in parsed_word.tag:
                    normalized_word = parsed_word.inflect({'nomn'}).word
                elif 'plur' in parsed_word.tag:
                    normalized_word = parsed_word.inflect({'nomn', 'plur'}).word
                else:
                    normalized_word = parsed_word.normal_form
            else:
                if 'PREP' not in parsed_word.tag and parsed_word.tag.case:
                    normalized_word = parsed_word.inflect({'nomn'}).word
                else:
                    normalized_word = parsed_word.normal_form
        normalized_words.append(normalized_word)
    return ' '.join(normalized_words)


def transform_model_output(token_list, token_labels):
    tag = ""
    tag_label = ""
    tags = []
    tag_labels = []
    morph = MorphAnalyzer()

    for i in range(1, len(token_list)):
        if token_labels[i] == "O":
            if tag != "":
                norm = normalize_tag(transform_tag(tag), morph)
                tags.append(transform_tag(norm))
                tag_labels.append(tag_label)
                tag = ""
            tag_label = "O"
            continue
        if token_labels[i].startswith("B"):
            if tag != "" and token_labels[i][2:] != tag_label:
                norm = normalize_tag(transform_tag(tag), morph)
                tags.append(transform_tag(norm))
                tag_labels.append(tag_label)
                tag = token_list[i]
                tag_label = token_labels[i][2:]
                continue
            tag += (" " + token_list[i])
            tag_label = token_labels[i][2:]
        if token_labels[i].startswith("I"):
            tag += (" " + token_list[i])

    return tags, tag_labels


@app.get("/api/v1/prediction")
async def get_prediction(item: Item):
    try:
        text = item.text
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

        token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

        token_labels = [model.config.id2label[label_id] for label_id in predicted_labels]

        token_list, token_labels = transform_model_output(token_list, token_labels)

        print(token_list, token_labels)

        return {"tokens": token_list, "labels": token_labels}
    except Exception as e:
        print(e)
