import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# Load .env
load_dotenv()


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:

    llm = get_llm()

    # Prompt for each transcript chunk
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of a meeting transcript concisely."
            ),
            (
                "human",
                "{text}"
            ),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    # Split transcript into chunks
    chunks = split_transcript(transcript)

    # Summarize each chunk
    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    # Combine all partial summaries
    combined = "\n\n".join(chunk_summaries)

    # Final summary prompt
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. "
                "Combine these partial summaries into one final "
                "professional meeting summary in bullet points."
            ),
            (
                "human",
                "{text}"
            ),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcipt : str) -> str:
    llm = get_llm()

    

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    return title_chain.invoke(transcipt[:2000])


if __name__ == "__main__":

    test_transcript = """
    Today we discussed the development of our AI Video Assistant.
    The team decided to use Whisper for English transcription.
    We will use Gemini for summarization and title generation.
    The next task is to implement the RAG system.
    """

    print("\n--- SUMMARY ---")
    summary = summarize(test_transcript)
    print(summary)

    print("\n--- TITLE ---")
    title = generate_title(test_transcript)
    print(title)