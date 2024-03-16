from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from sentence_transformers import SentenceTransformer


def connect(host="localhost", port="19530"):
    connections.connect("default", host=host, port=port)


fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="text", dtype=DataType.STRING, is_primary=False, auto_id=False),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, is_primary=True, auto_id=False),
]

connect()
schema = CollectionSchema(fields=fields)
collection = Collection(name="texts", schema=schema, using="default")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
model.max_seq_length = 512

texts = ["My name is RAM. I like to play cricket".lower(), "My name is Kunal. My hobby is singing.".lower()]
embeds = [list(embed) for embed in model.encode(texts)]


