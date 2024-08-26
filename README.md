# Llama Sensei: An AI-Powered Learning Assistant

**Llama Sensei** is a web application designed to enhance the learning experience by providing instantaneous answers to users' questions about their courses. It leverages Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) to ensure accurate and relevant information is delivered. Additionally, it provides precise links to the knowledge base that RAG uses, enhancing transparency and trust in the provided answers.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Speech Processing**: Uses VAD to segment speech, removing silent and non-speech parts, and Deepgram for speech-to-text conversion.
- **Text Processing**: Utilizes NLTK for efficient text preprocessing and handling.
- **Dynamic Query Handling**: Retrieves answers using a RAG pipeline that combines the strengths of vector databases and LLMs.
- **Interactive User Interface**: Streamlit-based UI allowing users to add new courses and ask questions in a user-friendly manner.
- **Real-Time Answer Streaming**: Answers are generated and displayed in real-time, enhancing user experience.
- **Search Integration**: Allows users to retrieve context from the vector database or search the internet for additional information.
- **Course Management**: FastAPI backend for managing course content, enabling easy addition and updating of course materials.
- **Feedback System**: Users can provide feedback on answers to help improve the system.

## Technologies Used

- **VAD (Voice Activity Detection)**: [Silero VAD](https://github.com/snakers4/silero-vad) for speech segmentation.
- **Speech-to-Text API**: [Deepgram](https://deepgram.com/) with [Deepgram Python SDK](https://github.com/deepgram/deepgram-python-sdk).
- **Text Processing**: [NLTK](https://www.nltk.org/api/nltk.html) for preprocessing and handling text data.
- **Vector Database**: [ChromaDB](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/) for storing and retrieving encoded documents.
- **User Interface**: [Streamlit](https://streamlit.io/) for creating an interactive and user-friendly front end.
- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) for building API endpoints.
- **LLM and RAG**: Various LLMs (such as Llama, Mistral, ChatGPT) and retrieval algorithms (like IVFPQ, HNSW) are used to optimize response accuracy and relevance.

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

### Evaluating Answers

Users can rate the quality of the answers, providing feedback that helps improve the model's performance over time.

## Project Structure

The project structure is organized as follows:

```
📦 llama-sensei
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
├─ scripts
│  └─ main.py
├─ tests
│  └─ ...
├─ .github
│  └─ ...
├─ requirements.txt
├─ setup.py
└─ ...
```

## Evaluation

To ensure the effectiveness of our RAG system, we conduct evaluations based on:

1. **Retriever Performance**: Using datasets like Qasper for benchmarking.
2. **Generator Performance**: Testing with synthetic datasets and different LLMs.
3. **Overall System Performance**: Metrics such as F1-score, relevance, and accuracy are used to evaluate the pipeline comprehensively.
4. **User Feedback**: Collecting user ratings to continuously improve the model.

## Contributing

We welcome contributions from the community. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
