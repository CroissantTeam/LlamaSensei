import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from llama_sensei.backend.add_courses.speech_to_text.transcript import DeepgramSTTClient

@pytest.fixture
def deepgram_client():
    return DeepgramSTTClient(output_path="/tmp/test_output")

@pytest.mark.asyncio
async def test_get_transcripts_no_files(deepgram_client):
    with patch('builtins.print') as mock_print:
        await deepgram_client.get_transcripts([])
    mock_print.assert_called_once_with("There is no file to transcribe")

@pytest.mark.asyncio
async def test_get_transcripts_existing_file(deepgram_client, tmp_path):
    existing_file = tmp_path / "existing.json"
    existing_file.touch()
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.print') as mock_print:
        await deepgram_client.get_transcripts([str(existing_file)])
    
    mock_print.assert_called_once_with("file existing.json existed")

import datetime as dt
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_transcribe_success(deepgram_client):
    mock_deepgram_client = MagicMock()
    mock_deepgram_client.listen.asyncrest.v.return_value.transcribe_file = AsyncMock()
    
    # Use the current year instead of a fixed year
    current_year = dt.datetime.now().year
    mock_datetime = MagicMock()
    mock_datetime.now.side_effect = [
        dt.datetime(current_year, 1, 1),
        dt.datetime(current_year, 1, 1, 0, 0, 10)
    ]

    mock_file = AsyncMock()
    mock_file.__aenter__.return_value.read = AsyncMock(return_value=b"mock audio data")

    with patch('llama_sensei.backend.add_courses.document.transcript.DeepgramClient', return_value=mock_deepgram_client), \
         patch('aiofiles.open', return_value=mock_file), \
         patch('builtins.open', MagicMock()), \
         patch('llama_sensei.backend.add_courses.document.transcript.datetime', mock_datetime), \
         patch('builtins.print') as mock_print:

        await deepgram_client.transcribe("test.wav", "test.json")
    
    # Check that each expected print statement was called
    expected_prints = [
        "Connecting to Deepgram...",
        "Connect successful!",
        "Sending request to Deepgram...",
        "Received transcript results from Deepgram...",
        "Transcript time: 10 seconds"
    ]
    for expected in expected_prints:
        mock_print.assert_any_call(expected)

    assert mock_print.call_count == len(expected_prints)
    mock_deepgram_client.listen.asyncrest.v.return_value.transcribe_file.assert_called_once()

@pytest.mark.asyncio
async def test_transcribe_exception(deepgram_client):
    with patch('llama_sensei.backend.add_courses.document.transcript.DeepgramClient', side_effect=Exception("Test error")), \
         patch('builtins.print') as mock_print:
        
        await deepgram_client.transcribe("test.wav", "test.json")
    
    mock_print.assert_called_with("Exception: Test error")