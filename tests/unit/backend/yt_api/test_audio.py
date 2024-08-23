import pytest
import os
import json
from unittest.mock import patch, MagicMock
from llama_sensei.backend.add_courses.yt_api.audio import YouTubeAudioDownloader, METADATA_FILENAME

@pytest.fixture
def downloader(tmp_path):
    return YouTubeAudioDownloader(str(tmp_path), "test_course")

@pytest.fixture
def sample_urls():
    return ["https://www.youtube.com/watch?v=sample1", "https://www.youtube.com/watch?v=sample2"]

@pytest.fixture
def sample_video_info():
    return {
        'id': 'sample1',
        'title': 'Sample Video',
        'channel': 'Sample Channel',
        'url': 'https://www.youtube.com/watch?v=sample1',
        'description': 'This is a sample video',
        'chapters': [],
        'duration': 300,
    }

def test_initialization(tmp_path):
    downloader = YouTubeAudioDownloader(str(tmp_path), "test_course")
    assert os.path.exists(os.path.join(str(tmp_path), "test_course"))
    assert downloader.ydl_opts['format'] == 'bestaudio/best'
    assert downloader.ydl_opts['postprocessors'][0]['preferredcodec'] == 'wav'

@patch('yt_dlp.YoutubeDL')
def test_download_audio(mock_ydl, downloader, sample_urls, sample_video_info):
    mock_ydl_instance = MagicMock()
    mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.extract_info.return_value = sample_video_info

    result = downloader.download_audio(sample_urls)

    assert result == sample_urls
    assert mock_ydl_instance.extract_info.call_count == len(sample_urls)
    assert mock_ydl_instance.download.called_once_with(sample_urls)

    metadata_file = os.path.join(downloader.output_course_path, METADATA_FILENAME)
    assert os.path.exists(metadata_file)

    with open(metadata_file, 'r') as f:
        saved_metadata = json.load(f)
    
    assert len(saved_metadata) == len(sample_urls)
    assert saved_metadata[0]['id'] == sample_video_info['id']
    assert saved_metadata[0]['title'] == sample_video_info['title']

def test_output_path_creation(tmp_path):
    course_name = "new_course"
    downloader = YouTubeAudioDownloader(str(tmp_path), course_name)
    expected_path = os.path.join(str(tmp_path), course_name)
    assert os.path.exists(expected_path)

def test_ydl_opts_configuration(downloader):
    assert 'format' in downloader.ydl_opts
    assert 'postprocessors' in downloader.ydl_opts
    assert downloader.ydl_opts['postprocessors'][0]['key'] == 'FFmpegExtractAudio'
    assert downloader.ydl_opts['postprocessors'][0]['preferredcodec'] == 'wav'

@patch('yt_dlp.YoutubeDL')
def test_metadata_file_content(mock_ydl, downloader, sample_urls, sample_video_info):
    mock_ydl_instance = MagicMock()
    mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.extract_info.return_value = sample_video_info

    downloader.download_audio(sample_urls)

    metadata_file = os.path.join(downloader.output_course_path, METADATA_FILENAME)
    with open(metadata_file, 'r') as f:
        saved_metadata = json.load(f)
    
    assert isinstance(saved_metadata, list)
    assert len(saved_metadata) == len(sample_urls)
    for video_info in saved_metadata:
        assert all(key in video_info for key in ['id', 'title', 'channel', 'url', 'description', 'chapters', 'duration'])