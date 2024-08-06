import requests
import re
import json
import urllib.parse

def get_relevant_videos(url, max_results=10):
    try:
        # Extract video ID from URL
        video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if not video_id:
            print("Invalid YouTube URL")
            return []
        video_id = video_id.group(1)

        # Make a request to get the watch page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}", headers=headers)
        html = response.text

        # Extract the initial data JSON
        initial_data_match = re.search(r"var ytInitialData = ({.*?});</script>", html)
        if not initial_data_match:
            print("Could not find initial data.")
            return []

        initial_data = json.loads(initial_data_match.group(1))

        # Extract relevant videos
        relevant_videos = []
        secondary_results = initial_data['contents']['twoColumnWatchNextResults']['secondaryResults']['secondaryResults']['results']

        for item in secondary_results:
            if 'compactVideoRenderer' in item:
                video = item['compactVideoRenderer']
                video_data = {
                    'title': video['title']['simpleText'],
                    'url': f"https://www.youtube.com/watch?v={video['videoId']}",
                    'channel': video['longBylineText']['runs'][0]['text'],
                    'views': video['viewCountText']['simpleText'] if 'viewCountText' in video else 'N/A',
                    'duration': video['lengthText']['simpleText'] if 'lengthText' in video else 'N/A'
                }
                relevant_videos.append(video_data)

                if len(relevant_videos) >= max_results:
                    break

        return relevant_videos

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

# Example usage
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your video URL
    relevant_videos = get_relevant_videos(video_url)

    if relevant_videos:
        print(f"Relevant videos for {video_url}:")
        for i, video in enumerate(relevant_videos, 1):
            print(f"{i}. {video['title']}")
            print(f"   Channel: {video['channel']}")
            print(f"   Views: {video['views']}")
            print(f"   Duration: {video['duration']}")
            print(f"   URL: {video['url']}")
            print()
    else:
        print("No relevant videos found or an error occurred.")

# import yt_dlp

# def get_related_videos(video_url):
#     ydl_opts = {
#         'quiet': True,
#         'extract_flat': 'in_playlist',
#         'skip_download': True,
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info_dict = ydl.extract_info(video_url, download=False)
#         related_videos = info_dict.get('related_videos', [])

#     video_urls = []
#     for video in related_videos:
#         video_id = video.get('id')
#         if video_id:
#             video_urls.append(f"https://www.youtube.com/watch?v={video_id}")
    
#     return video_urls

# # Example usage
# if __name__ == "__main__":
#     video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your video URL
#     related_videos = get_related_videos(video_url)
#     print("Related video URLs:")
#     for url in related_videos:
#         print(url)
