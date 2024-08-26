# Llama Sensei: An AI-Powered Learning Assistant

## Introduction

![llama_sensei_logo](./assets/llama_sensei_logo.png)

**Llama Sensei** is a Chatbot application designed to enhance the learning experience by providing instantaneous answers to users' questions about their lecture content on online learning platform (e.g. Youtube). It leverages Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) to ensure accurate and relevant information is delivered. Additionally, it provides precise links to the reference source that RAG uses, enhancing transparency and trust in the provided answers.

## Table of Contents

- [Introduction](#introduction)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [Credits](#credits)

## Overview

### User interface

Chat interface  |  Add new course interface
:-------------------------:|:-------------------------:
![chat_ui](./assets/chat_ui.png) | ![add_course_ui](./assets/add_course_ui.png)

### Features

- **Speech Processing**: Uses VAD to segment speech, removing silent and non-speech parts, and Deepgram for speech-to-text conversion.
- **Text Processing**: Utilizes NLTK for efficient text preprocessing and handling.
- **Dynamic Query Handling**: Retrieves answers using a RAG pipeline that combines the strengths of vector databases and LLMs.
- **Interactive User Interface**: Streamlit-based UI allowing users to add new courses and ask questions in a user-friendly manner.
- **Real-Time Answer Streaming**: Answers are generated and displayed in real-time, enhancing user experience.
- **Search Integration**: Allows users to retrieve context from the vector database or search the internet for additional information.
- **Course Management**: FastAPI backend for managing course content, enabling easy addition and updating of course materials.

### Technologies Used

- **Youtube lecture video crawling**: download youtube video playlist that contain lecture videos using [yt-dlp](https://github.com/yt-dlp/yt-dlp).
- **Speech-to-Text**: transcribe lecture videos into transcripts with specific timestamp using [Deepgram API](https://deepgram.com/).
- **Text Processing**: preprocess and handle text data using [NLTK](https://www.nltk.org/api/nltk.html).
- **Text Embedding**: embed text data using [sentence_transformers]([https://www.nltk.org/api/nltk.html](https://huggingface.co/sentence-transformers).
- **Vector Database**: store and retrieve encoded lecture transcripts for RAG using [ChromaDB](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/).
- **Internet context search**: retrieve the external context from internet for RAG using [Duckduckgo](https://duckduckgo.com/).
- **LLM API**: use [Groq](https://groq.com/) for fast LLM inference with various models (such as Llama, Mistral, etc.) and real-time answer streaming.
- **User Interface**: implement interactive front-end using [Streamlit](https://streamlit.io/).
- **Backend APIs**: implement back-end APIs using [FastAPI](https://fastapi.tiangolo.com/).
- **Containerization**: use [Docker](https://www.docker.com/) for easy deployment.

### Overall architecture design

![architecture_design](./assets/architecture_design.png)

## Installation

To run the application locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/llama-sensei.git
   cd llama-sensei
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory and add necessary environment variables (e.g., API keys for Deepgram).

5. **Run the application**:
   ```bash
   streamlit run app/llama_sensei/frontend/Chat_Interface.py
   ```

## Usage

### Adding a Course

1. Navigate to the "Add Course" page in the application.
2. Upload course materials or provide links to the resources.
3. The application will preprocess and index the course content for future queries.

### Asking Questions

1. Go to the "QA" section.
2. Select a course from the database.
3. Enter your question and submit.
4. View the AI-generated answer along with references from the knowledge base.

## API Endpoints

![api_design](./assets/api_endpoints.png)

## Project Structure

The project structure is organized as follows:

```
📦 llama-sensei
├─ .github # For Github Actions configuration
│  └─ ...
├─ app
│  └─ llama_sensei
│     ├─ backend
│     │  ├─ add_courses
│     │  │  ├─ document
│     │  │  └─ vectordb
│     │  └─ chat
│     │     ├─ gen_prompt.py
│     │     └─ output.py
│     └─ frontend
│        ├─ pages
│        │  └─ Add_Courses_Interface.py
│        └─ Chat_Interface.py
├─ assets
│  └─ ...
├─ scripts
│  └─ ...
├─ tests
│  └─ unit
├─ .env.example # example environment variables
├─ .pre-commit-config.yaml
├─ README.md
├─ docker-compose.yaml
├─ requirements-dev.txt
└─ setup.py
```

## Evaluation

To ensure the effectiveness of our RAG system, we conduct evaluations based on:

1. **Retriever Performance**: Using datasets like Qasper for benchmarking.
2. **Generator Performance**: Testing with synthetic datasets and different LLMs.
3. **Overall System Performance**: Metrics such as F1-score, relevance, and accuracy are used to evaluate the pipeline comprehensively.
4. **User Feedback**: Collecting user ratings to continuously improve the model.

## Future works

## Contributing

If you find any issues or have suggestions for improvements, please feel free to open an issue.

## Credits

This project was developed by @Croissant team.
