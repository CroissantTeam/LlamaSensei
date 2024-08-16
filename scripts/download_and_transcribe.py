import asyncio
import glob
import os

from llama_sensei.backend.add_courses.document.transcript import DeepgramSTTClient

if __name__ == "__main__":
    playlist_url = "https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU"
    course_name = "cs229_stanford"
    out_dir = "./data"
    # fetcher = PlaylistVideosFetcher()
    # video_urls = fetcher.get_playlist_videos(playlist_url)
    # print(video_urls)

    # downloader = YouTubeAudioDownloader(out_dir, course_name=course_name)
    # downloader.download_audio(video_urls)
    # print('Download success')

    audio_list = glob.glob(os.path.join(out_dir, course_name, "audio/*.wav"))
    deepgram_client = DeepgramSTTClient(
        os.path.join(out_dir, course_name, "transcript")
    )
    asyncio.run(deepgram_client.get_transcripts(audio_list))
