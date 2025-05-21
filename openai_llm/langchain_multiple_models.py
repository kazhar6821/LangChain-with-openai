from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface_hub import HuggingFaceHub
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# LangChain's Chat Models Docs
# https://python.langchain.com/docs/integrations/chat/
load_dotenv()

# New messages
messages = [
    SystemMessage(content="You are a helpful assistant for general knowledge questions."),
    HumanMessage(content="Can you explain the theory of relativity in simple terms?"),
]
model = ChatOpenAI(model="gpt-4o")
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")
# ---- Hugging Face Chat Model Example ----

hf_model = HuggingFaceHub(model=meta-llama/Llama-3.2-3B-Instruct")

result = hf_model.invoke(messages)
print(f"Answer from Hugging Face: {result.content}")
