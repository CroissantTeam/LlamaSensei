from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="dunzhang/stella_en_400M_v5"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True).cuda()

    def embed(self, doc):
        return self.model.encode(doc)
    
    def embed_chunks(self,chunks:List[tuple],top_chunks:int = None)->List[tuple]:
        if top_chunks==None:top_chunks=len(chunks)
        return [(chunk[0],self.embed(chunk[0]),chunk[1],chunk[2]) for chunk in chunks[:top_chunks]]

# Usage example:
if __name__ == "__main__":
    embedder = Embedder()
    doc = ["troll vn troll vn troll vn"]
    doc_embedding = embedder.embed(doc)
    print(doc_embedding.shape)