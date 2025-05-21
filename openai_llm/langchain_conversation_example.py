from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant specialized in personal productivity."),
    HumanMessage(content="Can you give me one simple habit to boost my daily focus?"),
]

# Get the model's response to the message sequence
result = llm.invoke(messages)

# Print the model's response content
print(result.content)
