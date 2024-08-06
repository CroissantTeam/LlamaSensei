from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import urllib.parse
import time
import requests
import json

def get_youtube_videos(query, max_results=5):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (optional)

    # Disable image loading
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)

    # Set up the WebDriver with ChromeDriverManager
    try:
        # For newer versions of Selenium
        from selenium.webdriver.chrome.service import Service
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except TypeError:
        # For older versions of Selenium
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    videos = []

    try:
        # Encode the query and navigate to the search page
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
        driver.get(search_url)

        # Wait for the video elements to be present
        wait = WebDriverWait(driver,1)  # Increase timeout to 10 seconds
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'ytd-video-renderer')))

        # Parse the video elements
        video_divs = driver.find_elements(By.CSS_SELECTOR, 'ytd-video-renderer')
        for video_div in video_divs[:max_results]:
            title_element = video_div.find_element(By.CSS_SELECTOR, '#video-title')
            title = title_element.text
            url = title_element.get_attribute('href')
            
            # try:
            #     description_element = video_div.find_element(By.CSS_SELECTOR, '#description-text')
            #     description = description_element.text
            # except NoSuchElementException:
            #     description = ''
            
            # try:
            #     thumbnail_element = video_div.find_element(By.CSS_SELECTOR, '#thumbnail img')
            #     thumbnail_url = thumbnail_element.get_attribute('src')
            # except NoSuchElementException:
            #     thumbnail_url = ''
            
            video_data = {
                'title': title,
                # 'description': description,
                'url': url,
                # 'thumbnail_url': thumbnail_url
            }
            videos.append(video_data)

    except TimeoutException:
        print("Timed out waiting for page to load")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        driver.quit()

    return videos

def get_thumbnail(url):
    search_url = f"https://noembed.com/embed?url={url}"
    response = requests.get(search_url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to load page {response.status_code}")
    
    data = json.loads(response.text)
    thumbnail_url = data.get('thumbnail_url', '')
    return thumbnail_url

def attach_thumbnail(videos:dict):
    for i in range(len(videos)):
        videos[i]['thumbnail_url'] = get_thumbnail(videos[i]['url'])


if __name__ == "__main__":
    query = "tử linh pháp sư"
    tin = time.perf_counter()
    videos = get_youtube_videos(query)
    attach_thumbnail(videos=videos)
    print(f"method: {time.perf_counter()-tin}s")
    if videos:
        for video in videos:
            print(f"Title: {video['title']}")
            # print(f"Description: {video['description']}")
            print(f"URL: {video['url']}")
            print(f"Thumbnail URL: {video['thumbnail_url']}")
            print()
    else:
        print("No videos found or an error occurred.")


