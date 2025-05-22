import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the vector store
vectorstore = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define user query
query = "Where is Dracula's castle located?"

# Retrieve relevant documents
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2}
)
relevant_docs = retriever.invoke(query)

# Display retrieved documents
print("\n--- Relevant Documents ---")
if relevant_docs:
    for i, doc in enumerate(relevant_docs, 1):
        print(f"\nDocument {i} (source: {doc.metadata.get('source', 'N/A')}):\n")
        print(doc.page_content.strip())
else:
    print("No relevant documents found with the given threshold.")

# Optional: Prepare to ask the assistant for a summarized answer
if relevant_docs:
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = (
        f"Based only on the following documents, answer the question:\n'{query}'\n\n"
        f"If the answer is not in the text, say: 'I'm not sure.'\n\nDocuments:\n{context}"
    )

    chat = ChatOpenAI(model="gpt-4o")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]
    result = chat.invoke(messages)

    print("\n--- Assistant's Answer ---")
    print(result.content)
