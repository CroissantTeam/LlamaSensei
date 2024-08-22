import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from generate_answer import GenerateRAGAnswer
from schemas import ChatResponse, Question

load_dotenv()
CONTEXT_SEARCH_API_URL = f'{os.getenv("COURSE_API_URL")}/search'
app = FastAPI(title="LlamaSensei: Chat API")


@app.post("/generate_answer", response_model=ChatResponse)
async def api_generate_answer(question: Question):
    rag_chain = GenerateRAGAnswer(
        query=question.question,
        course=question.course,
        context_search_url=CONTEXT_SEARCH_API_URL,
    )
    answer, evidence = rag_chain.generate_answer()
    return ChatResponse(answer=answer, evidence=evidence)


@app.get("/")
async def root():
    return {"message": "Welcome to LlamaSensei: Chat API"}


def main():
    # Run web server with uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("CHAT_FASTAPI_HOST", "127.0.0.1"),
        port=int(os.getenv("CHAT_FASTAPI_PORT", 8001)),
        # reload=True,  # Uncomment this for debug
        # workers=2,
    )


if __name__ == "__main__":
    main()
# Add more endpoints as needed
