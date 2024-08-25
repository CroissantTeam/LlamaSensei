import pytest
from llama_sensei.backend.add_courses.vectordb.vector_db_operations import VectorDBOperations
# from llama_sensei.backend.add_courses.vectordb.document_processor import DocumentProcessor

# # @pytest.fixture
# # def text_processor():
# #     return TextProcessor()

@pytest.fixture
def vector_db():
    return VectorDBOperations("data/unittest")

# @pytest.fixture
# def document_processor():
#     return DocumentProcessor("test_collection")