import asyncio
import glob
import os

from llama_sensei.backend.add_courses.speech_to_text.transcript import DeepgramSTTClient
from llama_sensei.backend.add_courses.yt_api.audio import YouTubeAudioDownloader
from llama_sensei.backend.add_courses.yt_api.playlist import PlaylistVideosFetcher

if __name__ == "__main__":
    playlist_url = "https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z"
    course_name = "cs224n_stanford"
    out_dir = "/shared/final"
    fetcher = PlaylistVideosFetcher()
    video_urls = fetcher.get_playlist_videos(playlist_url)
    print(video_urls)

    downloader = YouTubeAudioDownloader(out_dir, course_name=course_name)
    downloader.download_audio(video_urls)
    print('Download success')

    audio_list = glob.glob(os.path.join(out_dir, course_name, "audio/*.wav"))
    deepgram_client = DeepgramSTTClient(
        os.path.join(out_dir, course_name, "transcript")
    )
    asyncio.run(deepgram_client.get_transcripts(audio_list))
