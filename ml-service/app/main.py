from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pydantic import BaseModel
import uvicorn


class Item(BaseModel):
    text: str


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("viktoroo/sberbank-rubert-base-collection3")
model = AutoModelForTokenClassification.from_pretrained("viktoroo/sberbank-rubert-base-collection3")


def transform_tag(tag):
    return tag.replace(" ##и", "й").replace(" ##", "").replace(" . ", ".")\
        .replace(" ( ", "(").replace(" )", ")").replace(" ) ", ")").strip().title()


def transform_model_output(token_list, token_labels):
    tag = ""
    tag_label = ""
    tags = []
    tag_labels = []

    for token, label in zip(token_list, token_labels):
        if label == "O":
            if tag != "":
                tags.append(transform_tag(tag))
                tag_labels.append(tag_label)
                tag = ""
                tag_label = ""
            continue
        if label.startswith("B"):
            if tag != "":
                tags.append(transform_tag(tag))
                tag_labels.append(tag_label)
                tag = ""
            tag += (" " + token)
            tag_label = label[-3:]
        if label.startswith("I"):
            tag += (" " + token)

    return tags, tag_labels

@app.get("api/v1/prediction")
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
