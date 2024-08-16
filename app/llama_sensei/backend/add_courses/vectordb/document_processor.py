from .vector_db_operations import VectorDBOperations
from llama_sensei.backend.add_courses.embedding.get_embedding import Embedder
from llama_sensei.backend.add_courses.embedding.preprocessing_text import TextPreprocessor
from llama_sensei.backend.add_courses.embedding.load_text import TranscriptLoader

class DocumentProcessor:
    def __init__(self, collection_name, search_only: bool):
        self.text_processor = TextPreprocessor()
        self.vector_db = VectorDBOperations()
        self.embedder = Embedder()
        if (search_only == False):
            self.vector_db.create_collection(collection_name)
        self.collection_name = collection_name

    def process_document(self, path, metadata):
        # load data
        transcriptLoader = TranscriptLoader(path)
        
        # Preprocess
        chunks = transcriptLoader.load_data()
        preprocessed_chunks = self.text_processor.preprocess_text(chunks)

        # embed
        chunks_with_embed = self.embedder.embed_chunks(preprocessed_chunks)

        # store each chunk
        for i, chunk in enumerate(chunks_with_embed):
            chunk_metadata = {**metadata, 'start': chunk[2], 'end': chunk[3]}
            self.vector_db.add_embedding(
                self.collection_name,
                chunks[i][0],  # raw text
                chunk[1],
                chunk_metadata,
                f"{metadata['video_id']}_{i}",
            )

    def search(self, query, top_k=5):
        query_embedding = self.embedder.embed(self.text_processor._preprocess(query))
        return self.vector_db.search_embeddings(
            self.collection_name, query_embedding, top_k
        )