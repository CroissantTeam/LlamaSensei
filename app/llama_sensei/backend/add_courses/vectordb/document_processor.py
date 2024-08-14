from .text_processing import TextProcessor
from .vector_db_operations import VectorDBOperations


class DocumentProcessor:
    def __init__(self, collection_name):
        self.text_processor = TextProcessor()
        self.vector_db = VectorDBOperations()
        self.collection_name = collection_name

    def process_document(self, text, metadata):
        # Preprocess
        preprocessed_text = ' '.join(self.text_processor.preprocess(text))

        # Chunk
        chunks = self.text_processor.chunk_text(preprocessed_text)

        # Embed and store each chunk
        for i, chunk in enumerate(chunks):
            embedding = self.text_processor.get_embedding(chunk)
            chunk_metadata = {**metadata, "chunk_id": i}
            self.vector_db.add_embedding(
                self.collection_name,
                embedding,
                chunk_metadata,
                f"{metadata['video_id']}_{i}",
            )

    def search(self, query, top_k=5):
        query_embedding = self.text_processor.get_embedding(query)
        return self.vector_db.search_embeddings(
            self.collection_name, query_embedding, top_k
        )
