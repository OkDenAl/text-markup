from __future__ import print_function

import logging

import grpc
from protos import predictions_pb2_grpc as pb2_grpc
from protos import predictions_pb2 as pb2


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Will try to greet world ...")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = pb2_grpc.MLServiceStub(channel)
        response = stub.GetPredictions(pb2.PredictionRequest(text="Андрей Кремль"))
    print("Greeter client received: " + response.message)


if __name__ == "__main__":
    logging.basicConfig()
    run()
