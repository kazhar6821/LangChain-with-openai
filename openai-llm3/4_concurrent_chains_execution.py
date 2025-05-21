from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4")

summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a seasoned film expert known for insightful critiques."),
        ("human", "Could you summarize the film {movie_name} in a few sentences?"),
    ]
)

def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You're a professional screenwriter with a critical eye."),
            ("human", "Here is the plot: {plot}. Can you break down its main strengths and weaknesses?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)

def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You specialize in character development analysis."),
            ("human", "Here are the characters: {characters}. Please evaluate their depth and role in the story."),
        ]
    )
    return character_template.format_prompt(characters=characters)

def combine_verdicts(plot_analysis, character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"plot": plot_branch_chain, "characters": character_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
)

result = chain.invoke({"movie_name": "Inception"})

print(result)
