from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")


if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection") #clear collection
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)

from pymilvus import model

embedding_fn = model.DefaultEmbeddingFunction()

# with open("text.txt","r") as file:
#     docs= file.readlines()

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = embedding_fn.encode_documents(docs)
print(vectors)
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))
