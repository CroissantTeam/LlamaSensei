import pytest
import numpy as np
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from llama_sensei.backend.add_courses.embedding.get_embedding import Embedder

@pytest.fixture
def embedder():
    return Embedder()

def test_embedder_initialization():
    embedder = Embedder()
    assert isinstance(embedder.model, SentenceTransformer)
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert embedder.model.device.type == expected_device.type

def test_embed_single_document(embedder):
    doc = "This is a test document."
    embedding = embedder.embed(doc)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1  # Should be a 1D array

def test_embed_chunks(embedder):
    chunks = [
        ("Chunk 1", 0, 10),
        ("Chunk 2", 11, 20),
        ("Chunk 3", 21, 30),
    ]
    embedded_chunks = embedder.embed_chunks(chunks)
    assert len(embedded_chunks) == 3
    for chunk in embedded_chunks:
        assert len(chunk) == 4
        assert isinstance(chunk[1], np.ndarray)
        assert chunk[1].ndim == 1

def test_embed_chunks_with_top_chunks(embedder):
    chunks = [
        ("Chunk 1", 0, 10),
        ("Chunk 2", 11, 20),
        ("Chunk 3", 21, 30),
    ]
    embedded_chunks = embedder.embed_chunks(chunks, top_chunks=2)
    assert len(embedded_chunks) == 2
    for chunk in embedded_chunks:
        assert len(chunk) == 4
        assert isinstance(chunk[1], np.ndarray)
        assert chunk[1].ndim == 1

def test_embed_empty_document(embedder):
    doc = ""
    embedding = embedder.embed(doc)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1

def test_embed_chunks_empty_list(embedder):
    chunks = []
    embedded_chunks = embedder.embed_chunks(chunks)
    assert len(embedded_chunks) == 0

@pytest.mark.parametrize("model_name", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"])
def test_embedder_with_different_models(model_name):
    embedder = Embedder(model_name=model_name)
    assert embedder.model.get_sentence_embedding_dimension() > 0

def test_embed_chunks_consistency(embedder):
    chunks = [
        ("Chunk 1", 0, 10),
        ("Chunk 2", 11, 20),
        ("Chunk 3", 21, 30),
    ]
    embedded_chunks1 = embedder.embed_chunks(chunks)
    embedded_chunks2 = embedder.embed_chunks(chunks)
    for chunk1, chunk2 in zip(embedded_chunks1, embedded_chunks2):
        assert np.allclose(chunk1[1], chunk2[1])