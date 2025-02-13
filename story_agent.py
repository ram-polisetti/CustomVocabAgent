from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import json

load_dotenv()

def load_vocab(file_path='vocab.json'):
    with open(file_path, 'r') as file:
        return json.load(file)

def create_story_agent():
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = PromptTemplate(
        input_variables=["vocab"],
        template="""You have access to a custom vocabulary of invented words. Each word represents a unique concept, emotion, or action that doesn't exist in standard language. Your task is to create a short, engaging story that incorporates as many of these words as possible, using their definitions to drive the narrative.

Here's the vocabulary:

{vocab}

Create a story that showcases these unique concepts. Be creative and try to use the words in ways that highlight their special meanings."""
    )

    return LLMChain(llm=llm, prompt=prompt)

def generate_story(agent, vocab):
    vocab_text = "\n".join([f"{word}: {definition}" for word, definition in vocab.items()])
    return agent.run(vocab_text)
