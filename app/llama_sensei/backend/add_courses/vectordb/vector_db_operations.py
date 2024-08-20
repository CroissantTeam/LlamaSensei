import chromadb


class VectorDBOperations:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/chroma_db")

    def create_collection(self, collection_name):
        try:
            self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            print(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            print(f"Failed to create collection: {str(e)}")

    def add_embedding(self, collection_name, document, embedding, metadata, id):
        try:
            collection = self.client.get_collection(collection_name)
            # either update if ids exist, or add new
            collection.upsert(
                documents=[document],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[id],
            )
            print("Embedding added successfully.")
        except Exception as e:
            print(f"Failed to add embedding: {str(e)}")

    def search_embeddings(self, collection_name, query_embedding, top_k=3):
        try:
            collection = self.client.get_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding.tolist()], 
                n_results=top_k,
                include=['documents', 'embeddings', 'metadatas']
            )
            print("Search successfully")
            return results
        except Exception as e:
            print(f"Search failed: {str(e)}")
