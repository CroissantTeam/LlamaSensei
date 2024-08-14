from download_audio_moduled import YouTubeAudioDownloader
from get_videos_in_playlist_moduled import PlaylistVideosFetcher

if __name__=="__main__":
    # Example usage:
    playlist_url = "https://www.youtube.com/watch?v=Mad_J8s97OM&list=PL6uC-XGZC7X58oTIxIgC07QTXCJl0CBSX"
    fetcher = PlaylistVideosFetcher()
    video_urls = fetcher.get_playlist_videos(playlist_url)
    print(video_urls)
    
    downloader = YouTubeAudioDownloader("./data")
    
    # Get video info
    downloader.download_audio(video_urls, course_name="test")
    print('Download success')