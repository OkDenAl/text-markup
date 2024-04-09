from concurrent import futures
import logging

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pydantic import BaseModel

import grpc
from protos import predictions_pb2_grpc as pb2_grpc
from protos import predictions_pb2 as pb2

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


def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MLServiceServicer_to_server(MLService(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
