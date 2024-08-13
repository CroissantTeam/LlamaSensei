import yt_dlp
import os

class YouTubeVideoDownloader:
    def __init__(self, output_path='.'):
        self.output_path = output_path
        self.ydl_opts = {
            'format': 'best[ext=mp4]/best',  # This will select the best pre-merged format
            'outtmpl': os.path.join(self.output_path, "video", '%(title)s.%(ext)s'),
            'no_warnings': True,
            'ignoreerrors': True,
        }

    def download_video(self, url):
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    print("Error: Unable to extract video information")
                    return None
                filename = ydl.prepare_filename(info)
            
            print(f"Download completed: {filename}")
            return filename
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your video URL
    output_path = "downloads"  # Replace with your desired output directory
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    downloader = YouTubeVideoDownloader(output_path)
    downloaded_file = downloader.download_video(video_url)
    
    if downloaded_file:
        print(f"Video saved to: {downloaded_file}")
    else:
        print("Failed to download the video.")