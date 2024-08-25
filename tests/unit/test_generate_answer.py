import pytest
import numpy as np
from llama_sensei.backend.add_courses.vectordb.document_processor import (
    DocumentProcessor,
)
from llama_sensei.backend.qa.generate_answer import (
    GenerateRAGAnswer
)
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_groq import ChatGroq
import json
from groq.resources.chat.completions import Completions

# Fixture to create a mocked instance of GenerateRAGAnswer

import os
os.environ["GROQ_API_KEY"] = "valid_key"

# @pytest.fixture

# Test for the external_search method
def test_external_search(mocker):
    mock_generateanswer = GenerateRAGAnswer(course="test_course", 
                                            model="valid_model")
    
    mock_search_api = mocker.patch('langchain_community.utilities.DuckDuckGoSearchAPIWrapper',
                                   return_value = [
        {'snippet': 'snippet1', 'link': 'link1'},
        {'snippet': 'snippet2', 'link': 'link2'},
        {'snippet': 'snippet3', 'link': 'link3'},
        {'snippet': 'snippet4', 'link': 'link4'},
        {'snippet': 'snippet5', 'link': 'link5'}
    ])
    mock_generateanswer.query = "test_query"
    results = mock_generateanswer.external_search()
    assert len(results) == len(mock_search_api.return_value)
    assert results[0]['snippet'] is not None
    assert results[0]['link'] is not None
    
def test_calculate_context_relevancy(mocker):
    mock_generateanswer = GenerateRAGAnswer(course="test_course", 
                                            model="valid_model")
    # Mock the embedder to return a non-trivial embedding for the query
    mocker.patch.object(mock_generateanswer.embedder, 'encode', return_value=np.array([0.5, 0.5]))

    # Prepare some mock contexts
    mock_generateanswer.contexts = [
        {'text': 'Context 1', 'embedding': np.array([0.5, 0.5])},
        {'text': 'Context 2', 'embedding': np.array([0.1, 0.9])}
    ]

    # Invoke the calculate_context_relevancy method
    relevancy_score = mock_generateanswer.calculate_context_relevancy()

    # Assert that the function does not return None
    assert relevancy_score is not None

def test_rank_and_select_top_contexts(mocker):
    mock_generateanswer = GenerateRAGAnswer(course="test_course", 
                                            model="valid_model")
    # Mock the embedder to return a specific query embedding
    mocker.patch.object(mock_generateanswer.embedder, 'encode', return_value=np.array([0.5, 0.5]))

    # Prepare test data with predefined embeddings
    mock_generateanswer.contexts = [
        {'text': 'Context A', 'embedding': np.array([1, 0])},
        {'text': 'Context B', 'embedding': np.array([0, 1])},
        {'text': 'Context C', 'embedding': np.array([0.5, 0.5])}
    ]
    
    mock_generateanswer.query = "Query"

    # Invoke the method under test with the query embedding
    top_contexts = mock_generateanswer.rank_and_select_top_contexts(top_n=2)

    # Check if the top contexts are as expected
    expected_order = ['Context C', 'Context B']
    actual_order = [context['text'] for context in top_contexts]
    assert actual_order == expected_order, f"Expected order {expected_order}, but got {actual_order}"

    # Optional: Validate the content of top_contexts if necessary
    assert top_contexts[0]['text'] == 'Context C'
    assert top_contexts[1]['text'] == 'Context B'
    
def test_generate_llm_answer(mocker):
    
    mock_generateanswer = GenerateRAGAnswer(course="test_course", 
                                            model="valid_model")
    
    groq_chat_completion_response = (
        {
            "id": "chatcmpl-7qyuw6Q1CFCpcKsMdFkmUPUa7JP2x",
            "object": "chat.completion",
            "created": 1692338378,
            "model": "",
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            [
                                {
                                    "question": "What is the document about?",
                                    "answer": "The document is about a sample topic.",
                                }
                            ],
                        ),
                    },
                    "logprobs": None,
                }
            ],
            "usage": {"completion_tokens": 9, "prompt_tokens": 10, "total_tokens": 19},
        }
    )
    
    groq_mocker = mocker.patch("langchain_groq.chat_models.ChatGroq.stream",
        return_value = groq_chat_completion_response,
        )
    
    result = mock_generateanswer.generate_llm_answer()
    assert result is not None