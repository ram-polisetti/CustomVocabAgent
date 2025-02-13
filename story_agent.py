from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import json
import random

load_dotenv()

def load_vocab(file_path='vocab.json'):
    with open(file_path, 'r') as file:
        return json.load(file)

def select_random_words(vocab, num_words=5):
    return dict(random.sample(list(vocab.items()), num_words))

def create_story_agent():
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = PromptTemplate(
        input_variables=["words"],
        template="Create a short story using the following custom words and their definitions:\n\n{words}\n\nStory:"
    )

    return LLMChain(llm=llm, prompt=prompt)

def generate_story(agent, words):
    words_text = "\n".join([f"{word}: {definition}" for word, definition in words.items()])
    return agent.run(words_text)
