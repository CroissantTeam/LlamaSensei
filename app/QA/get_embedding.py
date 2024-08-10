from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("dunzhang/stella_en_400M_v5")
model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
doc = ["troll vn troll vn troll vn"]
doc_embedding = model.encode(doc)
print(doc_embedding.shape)


