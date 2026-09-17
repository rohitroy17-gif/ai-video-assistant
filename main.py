from dotenv import load_dotenv

from audio_processor import process_input
from transcriber import transcribe_all
from summarizer import summarize, generate_title
from meeting_extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str) -> dict:
    print("Starting AI Video Assistant")

    # 1. Process video/audio input
    chunks = process_input(source)

    # 2. Transcribe audio in English
    transcript = transcribe_all(chunks, "english")
    print(f"Raw transcription (first 300 characters): {transcript[:300]}")

    # 3. Generate title
    title = generate_title(transcript)

    # 4. Generate summary
    summary = summarize(transcript)

    # 5. Extract action items
    action_items = extract_action_items(transcript)

    # 6. Extract key decisions
    decisions = extract_key_decisions(transcript)

    # 7. Extract open questions
    questions = extract_questions(transcript)

    # 8. Build RAG chain
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()

    result = run_pipeline(source)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")

    rag_chain = result["rag_chain"]

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)

        print(f"\n🤖 Assistant: {answer}\n")