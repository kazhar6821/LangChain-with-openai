from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# Load environment variables (e.g., OpenAI API key) from a .env file
load_dotenv()

# Initialize the OpenAI chat model (GPT-4 in this case)
llm = ChatOpenAI(model="gpt-4")

# Send a prompt to the model and get the response
result = llm.invoke("What is the way to make money?")

# Print the model's response
print(result)
