import glob

from llama_sensei.backend.add_courses.document.transcript import DeepgramSTTClient
from llama_sensei.backend.add_courses.yt_api.audio import YouTubeAudioDownloader
from llama_sensei.backend.add_courses.yt_api.playlist import PlaylistVideosFetcher

if __name__ == "__main__":
    playlist_url = "https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU"
    course_name = "cs229_stanford"
    fetcher = PlaylistVideosFetcher()
    video_urls = fetcher.get_playlist_videos(playlist_url)
    print(video_urls)

    downloader = YouTubeAudioDownloader("./data", course_name=course_name)
    downloader.download_audio(video_urls)
    print('Download success')

    audio_list = glob.glob("data/test/audio/*.wav")
    deepgram_client = DeepgramSTTClient(f"data/{course_name}/transcript")
    deepgram_client.get_transcripts(audio_list)
