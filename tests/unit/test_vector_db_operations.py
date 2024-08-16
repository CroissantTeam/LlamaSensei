import pytest

def test_create_collection_success(vector_db):
    collection_name = "test_collection"
    result = vector_db.create_collection(collection_name)
    assert "created successfully" in result
    # Validate collection exists in client
    collection = vector_db.client.get_collection(collection_name)
    assert collection is not None

def test_add_embedding_success(vector_db):
    collection_name = "test_collection"
    vector_db.create_collection(collection_name)
    
    text = "This is a test document."
    metadata = {"source": "test"}
    id = "test_id"
    
    result = vector_db.add_embedding(collection_name, text, metadata, id)
    assert "Embedding added successfully." in result

    # Validate that embedding exists
    results = vector_db.search_embeddings(collection_name, "test document", top_k=1)
    assert len(results['ids'][0]) > 0

def test_search_embeddings(vector_db):
    collection_name = "test_collection"
    vector_db.create_collection(collection_name)
    
    text = "This is a test document."
    metadata = {"source": "test"}
    id = "test_id"
    vector_db.add_embedding(collection_name, text, metadata, id)

    query = "test document"
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
    
    query = "nonexistent document"
    results = vector_db.search_embeddings(collection_name, query, top_k=1)
    
    assert len(results['ids'][0]) == 0

def test_add_multiple_embeddings(vector_db):
    collection_name = "multi_doc_collection"
    vector_db.create_collection(collection_name)
    
    documents = ["First document", "Second document", "Third document"]
    for i, doc in enumerate(documents):
        vector_db.add_embedding(collection_name, doc, {"index": i}, f"id_{i}")
    
    results = vector_db.search_embeddings(collection_name, "document", top_k=3)
    assert len(results['ids'][0]) == 3


@pytest.mark.parametrize("invalid_name", ["", " ", "invalid/name"])
def test_invalid_collection_name(vector_db, invalid_name):
    with pytest.raises(ValueError):  # Change to ValueError based on the custom exception we raise
        vector_db.create_collection(invalid_name)


