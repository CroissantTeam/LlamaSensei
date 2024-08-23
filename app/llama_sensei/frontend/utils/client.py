import os

import requests
from dotenv import load_dotenv

load_dotenv()
ADD_COURSE_API_URL = f'{os.getenv("COURSE_API_URL")}/add_course'
GET_COURSE_API_URL = f'{os.getenv("COURSE_API_URL")}/courses'
CHAT_API_URL = f'{os.getenv("CHAT_API_URL")}/generate_answer'


def get_courses():
    r = requests.get(GET_COURSE_API_URL)
    return r.json()


def add_course(playlist_url: str, course_name: str):
    course_info = {"playlist_url": playlist_url, "course_name": course_name}
    r = requests.post(ADD_COURSE_API_URL, json=course_info)
    return r.json()


def response_generator(input: str, course_name):
    chat_query = {"question": input, "course": course_name}
    r = requests.post(CHAT_API_URL, json=chat_query)
    response = r.json()
    return response["answer"], response["evidence"]
