# Filename: tests/yt_api/test_playlist.py

import pytest
from unittest.mock import patch, MagicMock
from llama_sensei.backend.add_courses.yt_api.playlist import PlaylistVideosFetcher

@pytest.fixture
def playlist_fetcher():
    """Fixture for initializing PlaylistVideosFetcher."""
    return PlaylistVideosFetcher()

@patch('llama_sensei.backend.add_courses.yt_api.playlist.yt_dlp.YoutubeDL')
def test_get_playlist_videos_success(mock_yt_dlp, playlist_fetcher):
    """Test successful video extraction from a playlist URL."""
    # Mocking yt_dlp's extract_info method to return a fake playlist
    mock_ydl_instance = MagicMock()
    mock_ydl_instance.extract_info.return_value = {
        'entries': [{'url': 'http://video1.url'}, {'url': 'http://video2.url'}]
    }
    mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance

    playlist_url = "http://fakeplaylist.url"
    video_urls = playlist_fetcher.get_playlist_videos(playlist_url)

    assert video_urls == ['http://video1.url', 'http://video2.url']
    mock_ydl_instance.extract_info.assert_called_once_with(playlist_url, download=False)

@patch('llama_sensei.backend.add_courses.yt_api.playlist.yt_dlp.YoutubeDL')
def test_get_playlist_videos_no_entries(mock_yt_dlp, playlist_fetcher):
    """Test handling of playlist with no 'entries' key."""
    # Mocking yt_dlp's extract_info method to return a playlist with no entries
    mock_ydl_instance = MagicMock()
    mock_ydl_instance.extract_info.return_value = {}
    mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance

    playlist_url = "http://fakeplaylist.url"
    video_urls = playlist_fetcher.get_playlist_videos(playlist_url)

    assert video_urls == []
    mock_ydl_instance.extract_info.assert_called_once_with(playlist_url, download=False)

@patch('llama_sensei.backend.add_courses.yt_api.playlist.yt_dlp.YoutubeDL')
def test_get_playlist_videos_empty_playlist(mock_yt_dlp, playlist_fetcher):
    """Test handling of an empty playlist."""
    # Mocking yt_dlp's extract_info method to return an empty 'entries' list
    mock_ydl_instance = MagicMock()
    mock_ydl_instance.extract_info.return_value = {'entries': []}
    mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance

    playlist_url = "http://fakeplaylist.url"
    video_urls = playlist_fetcher.get_playlist_videos(playlist_url)

    assert video_urls == []
    mock_ydl_instance.extract_info.assert_called_once_with(playlist_url, download=False)

@patch('llama_sensei.backend.add_courses.yt_api.playlist.yt_dlp.YoutubeDL')
def test_get_playlist_videos_exception_handling(mock_yt_dlp, playlist_fetcher):
    """Test that exceptions during extraction are handled gracefully."""
    # Mocking yt_dlp's extract_info method to raise an exception
    mock_ydl_instance = MagicMock()
    mock_ydl_instance.extract_info.side_effect = Exception("Some error occurred")
    mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance

    playlist_url = "http://fakeplaylist.url"
    video_urls = playlist_fetcher.get_playlist_videos(playlist_url)

    assert video_urls == []
    mock_ydl_instance.extract_info.assert_called_once_with(playlist_url, download=False)
