from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatOpenAI(model="gpt-4")

messages = [
    ("system", "You are a witty storyteller who shares fun facts about {topic}."),
    ("human", "Please tell me {fact_count} interesting facts."),
]

# Create the prompt template from messages
prompt_template = ChatPromptTemplate.from_messages(messages)

# Fill in the template variables
prompt = prompt_template.invoke({"topic": "space exploration", "fact_count": 5})

# Call the model with the generated prompt
result = llm.invoke(prompt)

print(result.content)
