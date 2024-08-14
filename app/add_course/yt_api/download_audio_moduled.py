from typing import List
import os
import json

import yt_dlp


class YouTubeAudioDownloader:
    def __init__(self, output_path='.'):
        self.output_path = output_path
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.output_path, "audio", '%(title)s.%(ext)s'),
            'postprocessors': [],
            'no_warnings': True,
            'quiet': True
        }

    def download_audio(self, urls: List[str], course_name: str):
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                video_info_list = []
                for url in urls:
                    info = ydl.extract_info(url, download=False)
                    video_info = {
                        'id':info.get('id', 'No id'),
                        'title': info.get('title', 'No title'),
                        'channel': info.get('channel', 'No channel'),
                        'url': info.get('url', 'No URL'),
                        'description': info.get('description', 'No description'),
                        'chapters': info.get('chapters', 'No chapters'),
                        'duration': info.get('duration', 'Unknown'),
                    }
                    video_info_list.append(video_info)
                    
                with open(f"{course_name}.json", "w") as f:
                    json.dump(video_info_list, f, indent=4)
                
                # Download playlist videos
                ydl.download(urls)

                return

        except Exception as e:
            raise f"An error occurred: {str(e)}"


    def _find_downloaded_file(self, title):
        for file in os.listdir(os.path.join(self.output_path, "audio")):
            if file.startswith(title):
                output_filepath = os.path.join(self.output_path, "audio", file)
                print(f"Download completed: {output_filepath}")
                return output_filepath

        print("Could not find the downloaded file.")
        return None

    def get_video_info(self, url):
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'id':info.get('id', 'No id'),
                    'title': info.get('title', 'No title'),
                    'channel': info.get('channel', 'No channel'),
                    'url': info.get('url', 'No URL'),
                    'description': info.get('description', 'No description'),
                    'chapters': info.get('chapters', 'No chapters'),
                    'duration': info.get('duration', 'Unknown'),
                }
        except Exception as e:
            print(f"An error occurred while fetching video info: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=Mad_J8s97OM"  # Replace with your video URL
    output_path = "downloads"  # Replace with your desired output directory
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    downloader = YouTubeAudioDownloader(output_path)
    
    # Get video info
    video_info = downloader.get_video_info(video_url)
    if video_info:
        print("Video Information:")
        for key, value in video_info.items():
            print(f"{key}: {value}")
    
    # Download audio and get description
    downloaded_file, description = downloader.download_audio(video_url)
    
    if downloaded_file:
        print(f"Audio saved to: {downloaded_file}")
        print(f"Video Description:\n{description}")
    else:
        print("Failed to download the audio.")
