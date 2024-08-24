import pytest
from typing import List
from llama_sensei.backend.add_courses.embedding.preprocessing_text import TextPreprocessor

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

@pytest.fixture
def sample_chunks():
    return [
        ("This is a sample sentence.", 0.0, 1.0),
        ("Another example with multiple words.", 1.1, 2.0),
        ("Stemming and lemmatization are important in NLP.", 2.1, 3.0)
    ]

def test_preprocessor_initialization():
    preprocessor = TextPreprocessor()
    assert preprocessor.lemmatizer is not None
    assert preprocessor.stemmer is not None
    assert preprocessor.stop_words is not None

def test_preprocess_text_default(preprocessor, sample_chunks):
    result = preprocessor.preprocess_text(sample_chunks)
    assert len(result) == 3
    assert all(isinstance(item, tuple) and len(item) == 3 for item in result)
    assert "sampl sentenc" in result[0][0]
    assert "exampl multipl word" in result[1][0]
    assert "stem lemmat import nlp" in result[2][0]

def test_preprocess_text_no_lemmatize(preprocessor, sample_chunks):
    result = preprocessor.preprocess_text(sample_chunks, apply_lemmatize=False)
    assert "stem lemmat import nlp" in result[2][0]

def test_preprocess_text_no_stem(preprocessor, sample_chunks):
    result = preprocessor.preprocess_text(sample_chunks, apply_stem=False)
    assert "example multiple word" in result[1][0]

def test_preprocess_text_no_lemmatize_no_stem(preprocessor, sample_chunks):
    result = preprocessor.preprocess_text(sample_chunks, apply_lemmatize=False, apply_stem=False)
    assert "sample sentence" in result[0][0]

def test_preprocess_text_with_punctuation(preprocessor):
    chunks = [("Hello, world! How are you?", 0.0, 1.0)]
    result = preprocessor.preprocess_text(chunks)
    assert "hello , world ! ?" in result[0][0]
    assert "how" not in result[0][0].lower()

def test_chunk(preprocessor):
    words = "This is a long sentence that needs to be chunked into smaller pieces".split()
    result = preprocessor.chunk(words, chunk_size=5)
    assert len(result) == 3
    assert result[0] == ['This', 'is', 'a', 'long', 'sentence']
    assert result[1] == ['that', 'needs', 'to', 'be', 'chunked']
    assert result[2] == ['into', 'smaller', 'pieces']

def test_merge_text(preprocessor):
    sentences = [
        ("First part", 0.0, 1.0),
        ("Second part", 1.1, 2.0),
        ("Third part", 2.1, 3.0)
    ]
    result = preprocessor.merge_text(sentences)
    assert result == ("First part Second part Third part", 0.0, 3.0)

def test_merge_text_empty(preprocessor):
    result = preprocessor.merge_text([])
    assert result is None

def test_preprocess_text_empty_input(preprocessor):
    result = preprocessor.preprocess_text([])
    assert result == []

def test_preprocess_text_with_numbers(preprocessor):
    chunks = [("There are 123 apples and 456 oranges.", 0.0, 1.0)]
    result = preprocessor.preprocess_text(chunks)
    assert "123 appl 456 orang" in result[0][0]