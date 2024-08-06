import yt_dlp
import os

def download_youtube_audio(url, output_path='.'):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_path,"audio", '%(title)s.%(ext)s'),
            'postprocessors': [],
            'no_warnings': True,
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if 'title' not in info:
                print("Could not find title in video info.")
                return None

            title = info['title']
            # sanitized_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()

            print(f"Downloading audio: {title}")
            print(title)
            # print(sanitized_title)
            ydl.download([url])

            # Find the downloaded file
            for file in os.listdir(output_path):
                if file.startswith(title):
                    output_filepath = os.path.join(output_path, file)
                    print(f"Download completed: {output_filepath}")
                    return output_filepath

            print("Could not find the downloaded file.")
            return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your video URL
    output_path = "downloads"  # Replace with your desired output directory
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    downloaded_file = download_youtube_audio(video_url, output_path)
    if downloaded_file:
        print(f"Audio saved to: {downloaded_file}")
    else:
        print("Failed to download the audio.")