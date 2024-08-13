import yt_dlp
import sys

class PlaylistVideosFetcher:
    def __init__(self):
        self.ydl_opts = {
            'extract_flat': 'in_playlist',
            'skip_download': True,
            'quiet': False,  # Set to False for more verbose output
        }

    def get_playlist_videos(self, playlist_url):
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                print(f"Extracting info from: {playlist_url}")
                playlist_info = ydl.extract_info(playlist_url, download=False)
                
                if not playlist_info:
                    print("Failed to extract playlist info.")
                    return []
                
                if 'entries' not in playlist_info:
                    print("No 'entries' found in playlist info.")
                    return []
                
                videos = self._process_entries(playlist_info['entries'])
                print(f"Total videos found: {len(videos)}")
                return videos
        
        except Exception as e:
            print(f"An error occurred: {str(e)}", file=sys.stderr)
            return []

    def _process_entries(self, entries):
        videos = []
        for entry in entries:
            if entry:
                video = {
                    'title': entry.get('title', 'No title'),
                    'url': entry.get('url', 'No URL'),
                    'duration': entry.get('duration', 'Unknown'),
                    'uploader': entry.get('uploader', 'Unknown uploader')
                }
                videos.append(video)
                print(f"Added video: {video['title']}")
        return videos

    def print_videos(self, videos):
        if videos:
            for video in videos:
                print(f"Title: {video['title']}")
                print(f"URL: {video['url']}")
                print(f"Duration: {video['duration']} seconds")
                print(f"Uploader: {video['uploader']}")
                print("---")
        else:
            print("No videos were retrieved from the playlist.")

if __name__=="__main__":
    # Example usage:
    playlist_url = "https://www.youtube.com/watch?v=Mad_J8s97OM&list=PL6uC-XGZC7X58oTIxIgC07QTXCJl0CBSX"
    fetcher = PlaylistVideosFetcher()
    playlist_videos = fetcher.get_playlist_videos(playlist_url)
    fetcher.print_videos(playlist_videos)