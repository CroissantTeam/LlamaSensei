import pytest
from unittest.mock import Mock, patch
from llama_sensei.backend.add_courses.vectordb.document_processor import DocumentProcessor

@pytest.fixture
def mock_text_processor():
    return Mock()

@pytest.fixture
def mock_vector_db():
    return Mock()

@pytest.fixture
def mock_embedder():
    return Mock()

@pytest.fixture
def document_processor(mock_text_processor, mock_vector_db, mock_embedder):
    with patch('llama_sensei.backend.add_courses.vectordb.document_processor.TextPreprocessor', return_value=mock_text_processor), \
         patch('llama_sensei.backend.add_courses.vectordb.document_processor.VectorDBOperations', return_value=mock_vector_db), \
         patch('llama_sensei.backend.add_courses.vectordb.document_processor.Embedder', return_value=mock_embedder):
        return DocumentProcessor("test_collection", search_only=False)

def test_init(document_processor, mock_vector_db):
    """
    Test the initialization of DocumentProcessor when search_only is False.
    
    Ensures that:
    1. The collection_name is correctly set.
    2. The create_collection method is called on the vector database.
    """
    assert document_processor.collection_name == "test_collection"
    mock_vector_db.create_collection.assert_called_once_with("test_collection")

def test_init_search_only(mock_text_processor, mock_vector_db, mock_embedder):
    """
    Test the initialization of DocumentProcessor when search_only is True.
    
    Ensures that:
    1. The create_collection method is not called on the vector database.
    """
    with patch('llama_sensei.backend.add_courses.vectordb.document_processor.TextPreprocessor', return_value=mock_text_processor), \
         patch('llama_sensei.backend.add_courses.vectordb.document_processor.VectorDBOperations', return_value=mock_vector_db), \
         patch('llama_sensei.backend.add_courses.vectordb.document_processor.Embedder', return_value=mock_embedder):
        dp = DocumentProcessor("test_collection", search_only=True)
    mock_vector_db.create_collection.assert_not_called()

def test_process_document(document_processor, mock_text_processor, mock_vector_db, mock_embedder):
    """
    Test the process_document method of DocumentProcessor.

    Ensures that:
    1. The TranscriptLoader is called with the correct path.
    2. The text processing methods are called the correct number of times.
    3. The embedding method is called.
    4. The vector database add_embedding method is called the correct number of times.
    """
    mock_transcript_loader = Mock()
    mock_transcript_loader.load_data.return_value = ["Sentence 1.", "Sentence 2.", "Sentence 3.", "Sentence 4."]

    # Mock the return value of merge_text
    mock_text_processor.merge_text.side_effect = lambda x: "".join(x)

    # Mock the return value of preprocess_text
    mock_text_processor.preprocess_text.return_value = ["Preprocessed chunk 1", "Preprocessed chunk 2"]

    # Mock the return value of embed_chunks
    mock_embedder.embed_chunks.return_value = [
        ("Preprocessed chunk 1", [0.1, 0.2, 0.3], 0, 2),
        ("Preprocessed chunk 2", [0.4, 0.5, 0.6], 2, 4)
    ]

    with patch('llama_sensei.backend.add_courses.vectordb.document_processor.TranscriptLoader', return_value=mock_transcript_loader):
        document_processor.process_document("test_path", {"video_id": "123"}, num_st_each_chunk=2)

    # Verify that methods were called with correct arguments
    mock_transcript_loader.load_data.assert_called_once()
    assert mock_text_processor.merge_text.call_count == 2
    mock_text_processor.preprocess_text.assert_called_once()
    mock_embedder.embed_chunks.assert_called_once()
    assert mock_vector_db.add_embedding.call_count == 2

    # Verify the arguments of add_embedding calls
    call_args_list = mock_vector_db.add_embedding.call_args_list
    assert call_args_list[0][0][1] == "S"  # raw text for first chunk (first character only)
    assert call_args_list[1][0][1] == "S"  # raw text for second chunk (first character only)

    # Verify other arguments of add_embedding calls
    for i, call_args in enumerate(call_args_list):
        assert call_args[0][0] == "test_collection"  # collection name
        assert call_args[0][2] == [0.1, 0.2, 0.3] if i == 0 else [0.4, 0.5, 0.6]  # embedding
        assert call_args[0][3] == {"video_id": "123", "start": 0 if i == 0 else 2, "end": 2 if i == 0 else 4}  # metadata
        assert call_args[0][4] == f"123_{i}"  # document ID

def test_search(document_processor, mock_text_processor, mock_vector_db, mock_embedder):
    """
    Test the search method of DocumentProcessor.
    
    Ensures that:
    1. The query is preprocessed correctly.
    2. The query is embedded.
    3. The vector database search_embeddings method is called with correct parameters.
    """
    query = "test query"
    mock_text_processor._preprocess.return_value = "preprocessed query"
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    
    document_processor.search(query, top_k=3)
    
    mock_text_processor._preprocess.assert_called_once_with(query)
    mock_embedder.embed.assert_called_once_with("preprocessed query")
    mock_vector_db.search_embeddings.assert_called_once_with("test_collection", [0.1, 0.2, 0.3], 3)

def test_erase_all_data(document_processor, mock_vector_db):
    """
    Test the erase_all_data method of DocumentProcessor.
    
    Ensures that:
    1. The delete_collection method is called on the vector database.
    2. The create_collection method is called to recreate the collection.
    """
    document_processor.erase_all_data()
    
    mock_vector_db.delete_collection.assert_called_once_with("test_collection")
    mock_vector_db.create_collection.assert_called_with("test_collection")