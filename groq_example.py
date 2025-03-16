import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Groq LLM Example")

# Check for Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Initialize Groq LLM
groq_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

# Create a simple prompt template
template = """
You are a helpful assistant. Answer the following question:

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Create a simple chain
chain = prompt | groq_llm | StrOutputParser()

# Define request model
class QuestionRequest(BaseModel):
    question: str

# Define response model
class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # Get the answer from Groq
        answer = chain.invoke({"question": request.question})
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Groq LLM API. Use the /ask endpoint to ask questions."}

# Run with: uvicorn groq_example:app --reload
