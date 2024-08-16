import pytest
from llama_sensei.backend.add_courses.document.document_processor import DocumentProcessor

@pytest.fixture
def document_processor():
    # Ensure a unique collection name for each test to avoid interference
    import uuid
    unique_collection_name = f"test_collection_{uuid.uuid4()}"
    return DocumentProcessor(collection_name=unique_collection_name)

def test_document_processor(document_processor):
    sentences = [
        {'text': 'This is the first sentence.', 'start': 0, 'end': 5},
        {'text': 'This is the second sentence.', 'start': 6, 'end': 11},
        {'text': 'This is the third sentence.', 'start': 12, 'end': 17}
    ]
    document_processor.process_document(sentences)

    query = "second sentence"
    results = document_processor.search(query, top_k=1)
    assert isinstance(results, dict)
    assert 'ids' in results
    assert 'distances' in results
    assert 'metadatas' in results
    assert 'documents' in results
    assert any("second sentence" in doc.lower() for doc in results['documents'][0])


def test_document_processor_empty_input(document_processor):
    document_processor.process_document([])
    results = document_processor.search("any query", top_k=1)
    assert len(results['ids'][0]) == 0


def test_document_processor_multiple_documents(document_processor):
    sentences1 = [
        {'text': 'This is the first sentence of document 1.', 'start': 0, 'end': 5},
        {'text': 'This is the second sentence of document 1.', 'start': 6, 'end': 11},
    ]
    sentences2 = [
        {'text': 'This is the first sentence of document 2.', 'start': 0, 'end': 5},
        {'text': 'This is the second sentence of document 2.', 'start': 6, 'end': 11},
    ]
    document_processor.process_document(sentences1)
    document_processor.process_document(sentences2)

    results = document_processor.search("document 2", top_k=1)
    assert len(results['ids'][0]) == 1
    assert "document 2" in results['documents'][0][0]

