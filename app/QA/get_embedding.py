from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("dunzhang/stella_en_400M_v5")

# print(doc_embedding.shape)

def embedding(doc):
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda() #https://huggingface.co/dunzhang/stella_en_400M_v5
    # doc = ["troll vn troll vn troll vn"]
    doc_embedding = model.encode(doc)
    return doc_embedding

