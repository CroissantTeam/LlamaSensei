import json
import os
import time

import httpx
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
ADD_COURSE_API_URL = f'{os.getenv("COURSE_API_URL")}/add_course'
GET_COURSE_API_URL = f'{os.getenv("COURSE_API_URL")}/courses'
CHAT_API_URL = f'{os.getenv("CHAT_API_URL")}/generate_answer'
EVALUATE_API_URL = f'{os.getenv("CHAT_API_URL")}/evaluate'


def get_courses():
    r = requests.get(GET_COURSE_API_URL)
    return r.json()


def add_course(playlist_url: str, course_name: str):
    course_info = {"playlist_url": playlist_url, "course_name": course_name}
    r = requests.post(ADD_COURSE_API_URL, json=course_info)
    return r.json()


def response_generator(input: str, course_name: str, indb: bool, internet: bool):
    chat_query = {
        "question": input,
        "course": course_name,
        "indb": indb,
        "internet": internet,
    }

    with httpx.stream('POST', CHAT_API_URL, json=chat_query, timeout=None) as r:
        get_context = True
        for line in r.iter_lines():  # or, for line in r.iter_lines():
            json_object = json.loads(line)
            if get_context:
                st.session_state.contexts = json_object["context"]
                get_context = False
            yield json_object["token"]
            time.sleep(0.05)


def evaluate_evidence(query, answer, contexts, course):
    evaluate_query = {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "course_name": course,
    }
    r = requests.post(EVALUATE_API_URL, json=evaluate_query)
    return r.json()
