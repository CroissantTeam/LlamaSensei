# Llama Sensei: An AI-Powered Learning Assistant

<p align="center">
  <img src="./assets/llama_sensei_logo.png" width="300" height="300" />
</p>

## Introduction

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

**Llama Sensei** offers two main features: **Chat** and **Add New Course**.

#### Chat

- **Context Retrieval**: Retrieve context from either the selected course or the internet, including web search, to provide relevant answers.
- **Reference Linking**: Display reference links retrieved from the knowledge base and web search results, including precise timestamps for video content (e.g., specific segments in YouTube videos).
- **Web Search Integration**: Incorporate information from web searches to supplement course content and provide comprehensive answers.
- **Automated Answer Grading**: Answers are automatically graded by our algorithm to assess their relevance and accuracy.

#### Add New Course

- **Create New Course**: Easily create a new course within the application.
- **Add Videos**: Add new videos to an existing course, allowing for continuous updates and expansion of course content.

### Technologies Used

- **Youtube lecture video crawling**: download youtube video playlist that contain lecture videos using [yt-dlp](https://github.com/yt-dlp/yt-dlp).
- **Speech-to-Text**: transcribe lecture videos into transcripts with specific timestamp using [Deepgram API](https://deepgram.com/).
- **Text Processing**: preprocess and handle text data using [NLTK](https://www.nltk.org/api/nltk.html).
- **Text Embedding**: embed text data using [sentence_transformers](https://huggingface.co/sentence-transformers).
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

![api_design](./assets/api_design.png)

## Project Structure

The project structure is organized as follows:

```
📦 llama-sensei
├─ .github/ # For Github Actions configuration
│  └─ ...
├─ app/
│  └─ llama_sensei/
│     ├─ backend/
│     │  ├─ add_courses/
│     │  └─ qa/
│     └─ frontend/
│        ├─ pages/
│        └─ main.py
├─ assets/
│  └─ ...
├─ scripts/
│  └─ ...
├─ tests/
│  └─ unit/
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

## Contributing

If you find any issues or have suggestions for improvements, please feel free to open an issue.

### Future roadmap

### Code Quality

We maintain high code quality standards through:
- Continuous Integration (CI)
- Coding conventions
- Docstrings
- Pre-commit hooks
- Unit testing

## Credits

This project was developed by @Croissant team.
