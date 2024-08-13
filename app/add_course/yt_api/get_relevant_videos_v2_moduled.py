import requests
import re
import json
import urllib.parse

class YouTubeRelevantVideosFinder:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_relevant_videos(self, url, max_results=10):
        try:
            video_id = self._extract_video_id(url)
            if not video_id:
                print("Invalid YouTube URL")
                return []

            html = self._fetch_watch_page(video_id)
            initial_data = self._extract_initial_data(html)
            if not initial_data:
                return []

            return self._extract_relevant_videos(initial_data, max_results)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return []

    def _extract_video_id(self, url):
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        return video_id_match.group(1) if video_id_match else None

    def _fetch_watch_page(self, video_id):
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}", headers=self.headers)
        return response.text

    def _extract_initial_data(self, html):
        initial_data_match = re.search(r"var ytInitialData = ({.*?});</script>", html)
        if not initial_data_match:
            print("Could not find initial data.")
            return None
        return json.loads(initial_data_match.group(1))

    def _extract_relevant_videos(self, initial_data, max_results):
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

# Example usage
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your video URL
    finder = YouTubeRelevantVideosFinder()
    relevant_videos = finder.get_relevant_videos(video_url)

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