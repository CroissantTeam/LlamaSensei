import json
import os
from typing import List

import yt_dlp

METADATA_FILENAME = "playlist_metadata.json"


class YouTubeAudioDownloader:
    def __init__(self, output_path, course_name):
        self.course_name = course_name
        self.output_course_path = os.path.join(output_path, course_name)
        os.makedirs(self.output_course_path, exist_ok=True)
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.output_course_path, "audio", '%(id)s.%(ext)s'),
            'postprocessors': [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '10',
                }
            ],
            'no_warnings': True,
            'quiet': True,
        }

    def download_audio(self, urls: List[str]):
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                video_info_list = []
                for url in urls:
                    info = ydl.extract_info(url, download=False)
                    video_info = {
                        'id': info.get('id', 'No id'),
                        'title': info.get('title', 'No title'),
                        'channel': info.get('channel', 'No channel'),
                        'url': info.get('url', 'No URL'),
                        'description': info.get('description', 'No description'),
                        'chapters': info.get('chapters', 'No chapters'),
                        'duration': info.get('duration', 'Unknown'),
                    }
                    video_info_list.append(video_info)

                save_metadata_file = os.path.join(
                    self.output_course_path, METADATA_FILENAME
                )
                with open(save_metadata_file, "w") as f:
                    json.dump(video_info_list, f, indent=4)

                # Download playlist videos
                ydl.download(urls)

            return urls

        except Exception as e:
            print(f"An error occurred: {str(e)}")
