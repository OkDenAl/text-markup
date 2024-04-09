from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pydantic import BaseModel

from protos import predictions_pb2_grpc as pb2_grpc
from protos import predictions_pb2 as pb2


class Item(BaseModel):
    text: str


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("viktoroo/sberbank-rubert-base-collection3")
model = AutoModelForTokenClassification.from_pretrained("viktoroo/sberbank-rubert-base-collection3")


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

        print(token_list, token_labels)

        return {"tokens": token_list, "labels": token_labels}
    except Exception as e:
        print(e)
