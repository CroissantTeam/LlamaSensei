from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility,MilvusClient
# from llama_index import Document
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
import numpy as np

class VectorDatabase:
    def __init__(self, client_name = "documentdb.db",collection_name="text_collection", dim=768):
        self.client_name = client_name
        self.collection_name = collection_name
        self.dim = dim
        self.client = MilvusClient(self.client_name)

        
        # Connect to Milvus
        # connections.connect("default", host="localhost", port="19530")
        
        # Create collection if it doesn't exist
        if not utility.has_collection(self.collection_name):
            self._create_collection()
        
        self.collection = Collection(self.collection_name)
    
    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, "Text collection with vectors")
        self.collection = Collection(self.collection_name, schema)
        
        # Create an IVF_FLAT index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index("vector", index_params)
    
    def clear_database(self):
        """Clear all data in the collection"""
        self.collection.drop()
        self._create_collection()
    
    def add_text_chunk(self, text, vector):
        """Add a new chunk of text with its vector representation to the database"""
        data = [
            [text],
            [vector]
        ]
        self.collection.insert(data)
    
    def chunk_text(self, text):
        """Chunk the input text using LlamaIndex"""
        document = Document(text=text)
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents([document])
        return [node.text for node in nodes]

def generate_random_vector(dim=768):
    """Generate a random vector (placeholder for actual embedding function)"""
    return np.random.rand(dim).tolist()

# Usage example
if __name__ == "__main__":
    db = VectorDatabase()
    
    # Clear the database
    db.clear_database()
    
    # Example text
    long_text = """
    This is a long piece of text that will be chunked into smaller parts.
    It contains multiple sentences and paragraphs.
    LlamaIndex will be used to split this text into manageable chunks.
    Each chunk will then be added to our Milvus vector database.
    """
    
    # Chunk the text
    chunks = db.chunk_text(long_text)
    
    # Add chunks to the database
    for chunk in chunks:
        vector = generate_random_vector()  # Replace with actual embedding function
        db.add_text_chunk(chunk, vector)
    
    print(f"Added {len(chunks)} chunks to the database.")

    
# import chromadb
# from chromadb.api import ChromaAPI
# from chromadb.embeddings import SimpleEmbedding

# # Initialize ChromaDB API with an in-memory database for simplicity
# chroma = ChromaAPI(SimpleEmbedding(), storage='memory')

# def clear_database():
#     """
#     Clears the entire database.
#     """
#     chroma.clear()

# def split_text_into_chunks(text, chunk_size):
#     """
#     Splits a long string into smaller chunks of specified size.
    
#     Args:
#     - text: The long string to be split.
#     - chunk_size: The size of each chunk.
    
#     Returns:
#     A list of text chunks.
#     """
#     # Check if chunk_size is valid
#     if chunk_size <= 0:
#         raise ValueError("Chunk size must be greater than 0")
    
#     # Split text into chunks
#     chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
#     return chunks

# def add_chunk_to_database(chunk_id, chunk_text):
#     """
#     Adds a new chunk of text to the database.
    
#     Args:
#     - chunk_id: Unique identifier for the text chunk.
#     - chunk_text: The actual text chunk to be added.
#     """
#     chroma.add_documents([
#         {'id': chunk_id, 'content': chunk_text}
#     ])

# # Example usage
# if __name__ == "__main__":
#     # Clear the database
#     clear_database()
    
#     # Add new chunks of text
#     add_chunk_to_database('chunk1', 'This is the first chunk of text.')
#     add_chunk_to_database('chunk2', 'This is the second chunk of text.')

#     # Fetch all documents to verify
#     docs = chroma.fetch_documents()
#     for doc in docs:
#         print(f"ID: {doc['id']}, Content: {doc['content']}")


