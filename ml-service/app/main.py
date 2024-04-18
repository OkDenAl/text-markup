from fastapi import FastAPI
import torch
from pydantic import BaseModel
from own_lsg_converter import MYLSGConverter
from class_predictor import Classificator


class Item(BaseModel):
    text: str


app = FastAPI()

converter = MYLSGConverter(max_sequence_length=4096)
model, tokenizer = converter.convert_from_pretrained(
    'KodKio/ruBert-base-finetuned-ner',
    architecture="BertForTokenClassification"
)

classificator = Classificator()


def transform_tag(tag):
    return tag.replace(" ##ии", "ии").replace(" ##и", "й").replace(" ##", "").replace(" . ", ".") \
        .replace(" ( ", "(").replace(" )", ")").replace(" ) ", ")").strip().title()


def transform_model_output(token_list, token_labels):
    tag = ""
    tag_label = ""
    tags = []
    tag_labels = []

    for i in range(1, len(token_list)):
        if token_labels[i] == "O":
            if tag != "":
                tags.append(transform_tag(tag))
                tag_labels.append(tag_label)
                tag = ""
            tag_label = "O"
            continue
        if token_labels[i].startswith("B"):
            if tag != "" and token_labels[i][2:] != tag_label:
                tags.append(transform_tag(tag))
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

        token_labels = [model.config.id2label[label_id] for label_id in predicted_labels]

        token_list, token_labels = transform_model_output(token_list, token_labels)

        print(token_list, token_labels)

        return {"tags": token_list, "labels": token_labels}
    except Exception as e:
        print(e)




@app.get("/api/v1/class")
async def get_class(item: Item):
    classes = ["Home", "Health", "Celebrities", "Films and Shows", "Incidents", "Researches"]
    try:
        text = item.text
        embedding = classificator.get_embeddings([text])
        class_label = classificator.predict(embedding)
        return {"class": classes[class_label]}
    except Exception as e:
        print(e)
