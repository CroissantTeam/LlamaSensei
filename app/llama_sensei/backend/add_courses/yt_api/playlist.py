import sys
from typing import List

import yt_dlp


class PlaylistVideosFetcher:
    def __init__(self):
        self.ydl_opts = {
            "extract_flat": "in_playlist",
            "skip_download": True,
            "quiet": False,  # Set to False for more verbose output
        }

    def get_playlist_videos(self, playlist_url: str) -> List[str]:
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                print(f"Extracting info from: {playlist_url}")
                playlist_info = ydl.extract_info(playlist_url, download=False)

                if not playlist_info:
                    print("Failed to extract playlist info.")
                    return []

                if "entries" not in playlist_info:
                    print("No 'entries' found in playlist info.")
                    return []

                video_urls = [
                    video["url"]
                    for video in playlist_info["entries"]
                    if video.get("url")
                ]
                print(f"Total videos found: {len(video_urls)}")
                return video_urls

        except Exception as e:
            print(f"An error occurred: {str(e)}", file=sys.stderr)
            return []  # Ensure an empty list is returned in case of an exception
