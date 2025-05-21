from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a science teacher who explains topics in a fun and simple way."),
        ("human", "Can you explain {topic} in {detail_level} detail?"),
    ]
)

chain = prompt_template | model | StrOutputParser()


result = chain.invoke({"topic": "photosynthesis", "detail_level": "basic"})

# Output
print(result)
