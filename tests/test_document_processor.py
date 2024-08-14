import pytest
from backend.add_courses.document.document_processor import DocumentProcessor

@pytest.fixture
def doc_processor():
    return DocumentProcessor("test_collection")

def test_process_document(doc_processor, mocker):
    # Mock the dependencies
    mocker.patch.object(doc_processor.text_processor, 'preprocess', return_value=['test', 'document'])
    mocker.patch.object(doc_processor.text_processor, 'chunk_text', return_value=['test document'])
    mocker.patch.object(doc_processor.text_processor, 'get_embedding', return_value=[0.1, 0.2, 0.3])
    mocker.patch.object(doc_processor.vector_db, 'add_embedding', return_value="Embedding added successfully.")

    # Test data
    text = "This is a test document."
    metadata = {"video_id": "test_video", "title": "Test Video"}

    # Process the document
    doc_processor.process_document(text, metadata)

    # Assertions
    doc_processor.text_processor.preprocess.assert_called_once_with(text)
    doc_processor.text_processor.chunk_text.assert_called_once_with('test document')
    doc_processor.text_processor.get_embedding.assert_called_once_with('test document')
    doc_processor.vector_db.add_embedding.assert_called_once_with(
        "test_collection",
        [0.1, 0.2, 0.3],
        {"video_id": "test_video", "title": "Test Video", "chunk_id": 0},
        "test_video_0"
    )

def test_search(doc_processor, mocker):
    # Mock the dependencies
    mocker.patch.object(doc_processor.text_processor, 'get_embedding', return_value=[0.1, 0.2, 0.3])
    mocker.patch.object(doc_processor.vector_db, 'search_embeddings', return_value=["Result 1", "Result 2"])

    # Perform search
    results = doc_processor.search("test query")

    # Assertions
    doc_processor.text_processor.get_embedding.assert_called_once_with("test query")
    doc_processor.vector_db.search_embeddings.assert_called_once_with("test_collection", [0.1, 0.2, 0.3], 5)
    assert results == ["Result 1", "Result 2"]