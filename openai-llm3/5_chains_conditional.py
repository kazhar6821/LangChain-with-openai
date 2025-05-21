from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

model = ChatOpenAI(model="gpt-4o")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a professional and friendly virtual assistant working for a product quality team."),
        ("human", "Please write a warm and sincere thank-you message in response to this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a polite and empathetic support agent trained to handle complaints constructively."),
        ("human", "Kindly write a respectful and proactive reply to this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a customer support assistant helping to clarify vague feedback."),
        ("human", "Please generate a polite message asking for more context based on this feedback: {feedback}."),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a virtual assistant trained to identify when human intervention is needed."),
        ("human", "Compose a professional message to escalate this feedback to a human team member: {feedback}."),
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant that classifies customer feedback based on sentiment."),
        ("human", "Analyze the following feedback and classify it as positive, negative, neutral, or escalate: {feedback}."),
    ]
)

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches
review = "The product is terrible. It broke after just one use and the quality is very poor."
result = chain.invoke({"feedback": review})

# Output the result
print(result)
