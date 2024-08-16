import chromadb
from chromadb.utils import embedding_functions


class VectorDBOperations:
    def __init__(self, persist_directory=None):
        self.persist_directory = persist_directory
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        )

    def validate_collection_name(self, collection_name):
        if (
            not collection_name
            or collection_name.strip() == ""
            or "/" in collection_name
        ):
            raise ValueError(
                f"Invalid collection name: '{collection_name}'."
                f"Collection name cannot be empty, contain only spaces, or include '/'."
            )

    def create_collection(self, collection_name):
        self.validate_collection_name(collection_name)
        try:
            self.client.create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
            return f"Collection '{collection_name}' created successfully."
        except Exception as e:
            return f"Failed to create collection: {str(e)}"

    def add_embedding(self, collection_name, text, metadata, id):
        try:
            collection = self.client.get_collection(collection_name)
            collection.add(documents=[text], metadatas=[metadata], ids=[id])
            return "Embedding added successfully."
        except Exception as e:
            return f"Failed to add embedding: {str(e)}"

    def search_embeddings(self, collection_name, query_text, top_k=5):
        try:
            collection = self.client.get_collection(collection_name)
            results = collection.query(query_texts=[query_text], n_results=top_k)
            return results
        except Exception as e:
            return f"Search failed: {str(e)}"


# Test case for running the operations manually
if __name__ == "__main__":
    vector_db = VectorDBOperations()

    collection_name = "test_collection"
    print(vector_db.create_collection(collection_name))

    text = "This is a test document."
    metadata = {"source": "test"}
    id = "test_id"
    print(vector_db.add_embedding(collection_name, text, metadata, id))

    query = "test document"
    results = vector_db.search_embeddings(collection_name, query, top_k=1)
    print(f"Search results: {results}")

    empty_collection_name = "empty_collection"
    print(vector_db.create_collection(empty_collection_name))
    empty_results = vector_db.search_embeddings(
        empty_collection_name, "nonexistent document", top_k=1
    )
    print(f"Search results in empty collection: {empty_results}")
