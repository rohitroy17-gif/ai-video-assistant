import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from vector_store import build_vector_store, load_vector_store, get_retriever
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript:str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k = 4)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(

        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ]
    )

    #full LCEL Rag pipeline 

    rag_chain = (

        {"context" : retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
         }
         |prompt|llm|StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context":  retriever| RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question:str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer


if __name__ == "__main__":

    test_transcript = """
    Today we discussed the development of our AI Video Assistant.

    The team decided to use Whisper for English speech transcription.
    Gemini will be used for summarization and title generation.

    John will implement the RAG system by Friday.
    Sarah will prepare the project documentation by Monday.

    The team decided to use Chroma as the vector database.
    We will use Hugging Face embeddings for semantic search.

    The next meeting will be held next Wednesday at 10 AM.
    """

    print("\n--- BUILDING RAG CHAIN ---")

    rag_chain = build_rag_chain(test_transcript)

    print("\n--- TEST QUESTIONS ---")

    ask_question(
        rag_chain,
        "Which speech transcription model did the team choose?"
    )

    ask_question(
        rag_chain,
        "Who will implement the RAG system?"
    )

    ask_question(
        rag_chain,
        "What vector database will be used?"
    )

    ask_question(
        rag_chain,
        "When is the next meeting?"
    )

    ask_question(
        rag_chain,
        "What did they decide about the frontend?"
    )