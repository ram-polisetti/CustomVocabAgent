# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from dotenv import load_dotenv
# import os
# import json

# load_dotenv()


# def load_vocab(file_path='vocab.json'):
#     with open(file_path, 'r') as file:
#         return json.load(file)

# def create_story_agent():
#     llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"))

#     prompt = PromptTemplate(
#         input_variables=["vocab"],
#         template="""You have access to a custom vocabulary of invented words. Each word represents a unique concept, emotion, or action that doesn't exist in standard language. Your task is to create a short, engaging story that incorporates as many of these words as possible, using their definitions to drive the narrative.

# Here's the vocabulary:

# {vocab}

# Create a story that showcases these unique concepts. Be creative and try to use the words in ways that highlight their special meanings."""
#     )

#     return LLMChain(llm=llm, prompt=prompt)

# def generate_story(agent, vocab):
#     vocab_text = "\n".join([f"{word}: {definition}" for word, definition in vocab.items()])
#     return agent.run(vocab_text)


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

def select_subset_vocab(vocab, num_words=10):
    return dict(random.sample(list(vocab.items()), min(num_words, len(vocab))))

def create_story_agent():
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = PromptTemplate(
        input_variables=["vocab"],
        template="""You have access to a custom vocabulary of invented words. Each word represents a unique concept, emotion, or action that doesn't exist in standard language. These words are NOT nouns and should not be used as names for people, places, or objects.

Custom Vocabulary:
{vocab}

Create a short, engaging story that incorporates these words as descriptions of emotions, sensations, or actions. Use them to describe how characters feel or experience things in unique ways. Do not use these words as names or objects in the story.

For example, if a word is defined as "a feeling of excitement when making a mistake," you might write: "As she dropped the vase, she felt a surge of [custom word], an unexpected thrill at her error."

Write your story now, creatively using these custom words to describe unique experiences and feelings:"""
    )

    return LLMChain(llm=llm, prompt=prompt)

def generate_story(agent, vocab, num_words=10):
    subset_vocab = select_subset_vocab(vocab, num_words)
    vocab_text = "\n".join([f"{word}: {definition}" for word, definition in subset_vocab.items()])
    return agent.run(vocab_text)
