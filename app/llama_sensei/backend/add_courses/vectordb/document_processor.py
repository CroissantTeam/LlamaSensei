from ..embedding.get_embedding import Embedder
from ..embedding.load_text import TranscriptLoader
from ..embedding.preprocessing_text import TextPreprocessor

from .vector_db_operations import VectorDBOperations


class DocumentProcessor:
    def __init__(self, collection_name, search_only: bool):
        self.text_processor = TextPreprocessor()
        self.vector_db = VectorDBOperations()
        self.embedder = Embedder()
        if search_only is False:
            self.vector_db.create_collection(collection_name)
        self.collection_name = collection_name

    def process_document(self, path, metadata, num_st_each_chunk=3):
        # load data
        transcriptLoader = TranscriptLoader(path)

        # Preprocess
        sentences = transcriptLoader.load_data()
        chunks = [
            self.text_processor.merge_text(
                sentences[i : min(i + num_st_each_chunk, len(sentences))]
            )
            for i in range(0, len(sentences), num_st_each_chunk)
        ]
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
