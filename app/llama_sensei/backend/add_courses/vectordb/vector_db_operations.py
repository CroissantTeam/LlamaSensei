import chromadb


class VectorDBOperations:
    def __init__(self):
        self.client = chromadb.Client()

    def create_collection(self, collection_name):
        try:
            self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            return f"Collection '{collection_name}' created successfully."
        except Exception as e:
            return f"Failed to create collection: {str(e)}"

    def add_embedding(self, collection_name, embedding, metadata, id):
        try:
            collection = self.client.get_collection(collection_name)
            collection.add(embeddings=[embedding], metadatas=[metadata], ids=[id])
            return "Embedding added successfully."
        except Exception as e:
            return f"Failed to add embedding: {str(e)}"

    def search_embeddings(self, collection_name, query_embedding, top_k=5):
        try:
            collection = self.client.get_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding], n_results=top_k
            )
            return results
        except Exception as e:
            return f"Search failed: {str(e)}"
