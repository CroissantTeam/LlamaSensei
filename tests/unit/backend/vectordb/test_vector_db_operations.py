import pytest

from llama_sensei.backend.add_courses.vectordb.get_embedding import Embedder
from llama_sensei.backend.add_courses.vectordb.preprocessing_text import TextPreprocessor
from llama_sensei.backend.add_courses.vectordb.vector_db_operations import VectorDBOperations

text_processor = TextPreprocessor()
embedder = Embedder()

@pytest.fixture
def vector_db():
    return VectorDBOperations("data/unittest")

def test_create_collection_success(vector_db):
    collection_name = "test_collection"
    vector_db.create_collection(collection_name)
    collection = vector_db.client.get_collection(collection_name)
    assert collection is not None

def test_search_embeddings(vector_db):
    collection_name = "test_collection"
    vector_db.create_collection(collection_name)
    
    chunk = ("This is a test document.", 123., 456.)
    metadata = {"source": "test"}
    id = "test_id"
    preprocessed_chunks = text_processor.preprocess_text([chunk])
    chunks_with_embed = embedder.embed_chunks(preprocessed_chunks)
    vector_db.add_embedding(
        collection_name, 
        chunk[0], 
        chunks_with_embed[0][1], 
        metadata, 
        id
    )

    query = embedder.embed("test document")
    results = vector_db.search_embeddings(collection_name, query, top_k=1)
    
    assert isinstance(results, dict)
    assert 'ids' in results
    assert 'distances' in results
    assert 'metadatas' in results
    assert 'documents' in results
    assert len(results['ids'][0]) > 0

def test_search_no_results(vector_db):
    collection_name = "empty_collection"
    vector_db.create_collection(collection_name)
    
    query = embedder.embed("nonexistent document")
    results = vector_db.search_embeddings(collection_name, query, top_k=1)
    
    assert len(results['ids'][0]) == 0

def test_add_multiple_embeddings(vector_db):
    collection_name = "multi_doc_collection"
    vector_db.create_collection(collection_name)
    
    chunks = [
        ("First document", 123., 234.), 
        ("Second document", 345., 456.), 
        ("Third document", 567., 678.),
    ]
    metadata = {"source": "test"}
    preprocessed_chunks = text_processor.preprocess_text(chunks)
    chunks_with_embed = embedder.embed_chunks(preprocessed_chunks)
    for i, chunk in enumerate(chunks_with_embed):
        chunk_metadata = {**metadata, 'start': chunk[2], 'end': chunk[3]}
        vector_db.add_embedding(
            collection_name,
            chunks[i][0],  # raw text
            chunk[1],
            {"index": i},
            f"id_{i}",
        )
    
    query = embedder.embed("document")
    results = vector_db.search_embeddings(collection_name, query, top_k=3)
    assert len(results['ids'][0]) == 3