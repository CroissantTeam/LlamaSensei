import yt_dlp

def download_transcript(video_url, languages):
    ydl_opts = {
        'skip_download': True,
        'writeautomaticsub': True,
        'subtitleslangs': languages,
        'outtmpl': '%(title)s.%(ext)s'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# Usage
video_url = "https://www.youtube.com/watch?v=K6rSVoAakQ8"  # Replace with your video URL
languages = ['en', 'vi']  # English and Vietnamese

download_transcript(video_url, languages)