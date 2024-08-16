import pytest
from llama_sensei.backend.add_courses.document.text_processing import TextProcessor
from llama_sensei.backend.add_courses.document.vector_db_operations import VectorDBOperations
from llama_sensei.backend.add_courses.document.document_processor import DocumentProcessor

@pytest.fixture
def text_processor():
    return TextProcessor()

@pytest.fixture
def vector_db():
    return VectorDBOperations()

@pytest.fixture
def document_processor():
    return DocumentProcessor("test_collection")