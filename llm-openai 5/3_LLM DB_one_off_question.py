import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load API key and other env variables
load_dotenv()

# Define directories
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Set up embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define the query
user_query = "What does Dracula fear the most?"

# Retrieve top k relevant documents
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrieved_docs = retriever.invoke(user_query)

# Display relevant documents
print("\n--- Top Matching Documents ---")
for i, doc in enumerate(retrieved_docs, 1):
    preview = doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else "")
    print(f"\nDocument {i} (source: {doc.metadata.get('source', 'N/A')}):\n{preview}\n")

# Prepare context for the assistant
context = "\n\n".join([doc.page_content for doc in retrieved_docs])
prompt = (
    f"You are a helpful assistant.\n\n"
    f"Based only on the documents below, answer the question:\n'{user_query}'\n\n"
    f"If the answer is not explicitly found in the documents, reply with: 'I'm not sure.'\n\n"
    f"Documents:\n{context}"
)

# Set up the OpenAI chat model
llm = ChatOpenAI(model="gpt-4o")

# Invoke the model
response = llm.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=prompt),
])

# Output the response
print("\n--- Assistant Response ---")
print(response.content)
