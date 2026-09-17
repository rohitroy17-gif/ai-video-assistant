from core.vector_store import (
    build_vector_store,
    get_retriever
)

transcript = """
The team discussed the development of the AI meeting assistant.
Rohit will complete the transcription module by Friday.
The backend team will work on the API integration.
The deployment is planned for next Monday.
The team also decided to use Chroma as the vector database.
"""

# 1. Build vector database
vector_store = build_vector_store(transcript)

print("\nVector store created successfully!\n")

# 2. Create retriever
retriever = get_retriever(vector_store, k=2)

# 3. Test a question
question = "What database did the team decide to use?"

# 4. Retrieve relevant chunks
docs = retriever.invoke(question)

print("QUESTION:")
print(question)

print("\nRETRIEVED DOCUMENTS:")

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)