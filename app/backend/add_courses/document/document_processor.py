from .text_processing import TextProcessor
from .vector_db_operations import VectorDBOperations


class DocumentProcessor:
    def __init__(self, collection_name, persist_directory=None):
        self.text_processor = TextProcessor()
        self.vector_db = VectorDBOperations(persist_directory)
        self.collection_name = collection_name
        self.vector_db.create_collection(collection_name)
        self.chunk_counter = 0  # To ensure unique IDs across multiple documents

    def process_document(self, sentences):
        if not sentences:  # Handle empty input case
            return

        # Preprocess
        preprocessed_sentences = [
            {
                'text': self.text_processor.preprocess(s['text']),
                'start': s['start'],
                'end': s['end'],
            }
            for s in sentences
        ]

        # Chunk
        chunks = self.text_processor.chunk_text(preprocessed_sentences)

        # Add chunks to ChromaDB
        for chunk in chunks:
            chunk_id = f"chunk_{self.chunk_counter}"
            self.vector_db.add_embedding(
                self.collection_name,
                chunk['text'],
                {"start": chunk['start'], "end": chunk['end']},
                chunk_id,
            )
            self.chunk_counter += 1  # Ensure unique IDs

    def search(self, query, top_k=5):
        return self.vector_db.search_embeddings(self.collection_name, query, top_k)
