import yt_dlp
import sys

def get_playlist_videos(playlist_url):
    ydl_opts = {
        'extract_flat': 'in_playlist',
        'skip_download': True,
        'quiet': False,  # Set to False for more verbose output
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Extracting info from: {playlist_url}")
            playlist_info = ydl.extract_info(playlist_url, download=False)
            
            if not playlist_info:
                print("Failed to extract playlist info.")
                return []
            
            if 'entries' not in playlist_info:
                print("No 'entries' found in playlist info.")
                return []
            
            videos = []
            for entry in playlist_info['entries']:
                if entry:
                    videos.append({
                        'title': entry.get('title', 'No title'),
                        'url': entry.get('url', 'No URL'),
                        'duration': entry.get('duration', 'Unknown'),
                        'uploader': entry.get('uploader', 'Unknown uploader')
                    })
                    print(f"Added video: {entry.get('title', 'No title')}")
        
        print(f"Total videos found: {len(videos)}")
        return videos
    
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        return []

if __name__=="__main__":
    # Example usage:
    playlist_url = "https://www.youtube.com/watch?v=Mad_J8s97OM&list=PL6uC-XGZC7X58oTIxIgC07QTXCJl0CBSX"
    playlist_videos = get_playlist_videos(playlist_url)

    if playlist_videos:
        for video in playlist_videos:
            print(f"Title: {video['title']}")
            print(f"URL: {video['url']}")
            print(f"Duration: {video['duration']} seconds")
            print(f"Uploader: {video['uploader']}")
            print("---")
    else:
        print("No videos were retrieved from the playlist.")