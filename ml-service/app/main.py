from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pydantic import BaseModel
from own_lsg_converter import MYLSGConverter
import uvicorn

from protos import predictions_pb2_grpc as pb2_grpc
from protos import predictions_pb2 as pb2


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


class MLService(pb2_grpc.MLServiceServicer):
    def GetPredictions(self, request, context):
        try:
            text = request.text
            inputs = tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

            token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

            token_labels = [model.config.id2label[label_id] for label_id in predicted_labels]

            print(token_list, token_labels)

            return pb2.PredictionResponse(token_list, token_labels)
        except Exception as e:
            print(e)


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
