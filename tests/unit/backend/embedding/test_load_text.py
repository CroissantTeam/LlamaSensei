import pytest
import json
import tempfile
import os
from llama_sensei.backend.add_courses.embedding.load_text import TranscriptLoader

@pytest.fixture
def sample_data():
    return {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "paragraphs": {
                                "paragraphs": [
                                    {
                                        "sentences": [
                                            {"text": "This is a test sentence."},
                                            {"text": "This is another sentence."}
                                        ],
                                        "start": 0.0,
                                        "end": 5.0
                                    },
                                    {
                                        "sentences": [
                                            {"text": "This is a second paragraph."}
                                        ],
                                        "start": 5.1,
                                        "end": 8.0
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        },
        "metadata": {"key": "value"}
    }

@pytest.fixture
def temp_json_file(sample_data):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        json.dump(sample_data, tmp)
    yield tmp.name
    os.unlink(tmp.name)

def test_transcript_loader_initialization(temp_json_file):
    loader = TranscriptLoader(temp_json_file)
    assert loader.file_path == temp_json_file
    assert loader.data is None

def test_load_data_simple_output(temp_json_file, sample_data):
    loader = TranscriptLoader(temp_json_file)
    result = loader.load_data(simple_output=True)
    assert result == sample_data

def test_load_data_processed(temp_json_file):
    loader = TranscriptLoader(temp_json_file)
    result = loader.load_data()
    assert len(result) == 2
    assert result[0] == ("This is a test sentence. This is another sentence.", 0.0, 5.0)
    assert result[1] == ("This is a second paragraph.", 5.1, 8.0)

def test_load_data_file_not_found():
    loader = TranscriptLoader("non_existent_file.json")
    result = loader.load_data()
    assert result is None

def test_load_data_invalid_json(temp_json_file):
    with open(temp_json_file, 'w') as f:
        f.write("This is not valid JSON")
    loader = TranscriptLoader(temp_json_file)