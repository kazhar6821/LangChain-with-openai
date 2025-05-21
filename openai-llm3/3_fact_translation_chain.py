from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a fun zoologist who enjoys sharing interesting facts about {animal}."),
        ("human", "Can you give me {count} surprising facts?"),
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a professional translator specializing in {language} language."),
        ("human", "Please translate this into {language}: {text}"),
    ]
)

count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})

chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser() 

result = chain.invoke({"animal": "cat", "count": 2})

# Output
print(result)
