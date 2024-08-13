import yt_dlp
import os

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

    def download_audio(self, url):
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if 'title' not in info:
                    print("Could not find title in video info.")
                    return None

                title = info['title']
                print(f"Downloading audio: {title}")
                ydl.download([url])

                return self._find_downloaded_file(title)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _find_downloaded_file(self, title):
        for file in os.listdir(self.output_path):
            if file.startswith(title):
                output_filepath = os.path.join(self.output_path, file)
                print(f"Download completed: {output_filepath}")
                return output_filepath

        print("Could not find the downloaded file.")
        return None

# Example usage
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your video URL
    output_path = "downloads"  # Replace with your desired output directory
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    downloader = YouTubeAudioDownloader(output_path)
    downloaded_file = downloader.download_audio(video_url)
    
    if downloaded_file:
        print(f"Audio saved to: {downloaded_file}")
    else:
        print("Failed to download the audio.")