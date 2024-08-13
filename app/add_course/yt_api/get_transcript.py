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

if __name__ =="__main__":
    # Usage
    video_url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"  # Replace with your video URL
    languages = ['en', 'vi']  # English and Vietnamese

    download_transcript(video_url, languages)