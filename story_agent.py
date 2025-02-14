from dotenv import load_dotenv
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(f"API Key: {api_key[:5]}...{api_key[-5:]}")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")


def load_vocab(file_path='vocab.json'):
    with open(file_path, 'r') as file:
        vocab = json.load(file)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vocab_texts = [f"{word}: {definition}" for word, definition in vocab.items()]
        vectorstore = Chroma.from_texts(vocab_texts, embeddings)
        return vectorstore

def create_story_agent():
    llm = ChatGroq(api_key=api_key)
    prompt = PromptTemplate(
        input_variables=["vocab"],
        template="""You have access to a custom vocabulary of invented words. Each word represents a unique concept, emotion, or action that doesn't exist in standard language. These words are NOT nouns and should not be used as names for people, places, or objects.

Custom Vocabulary:
{vocab}

Create a short, engaging story that incorporates these words as descriptions of emotions, sensations, or actions. Use them to describe how characters feel or experience things in unique ways. Do not use these words as names or objects in the story.

For example, if a word is defined as "a feeling of excitement when making a mistake," you might write: "As she dropped the vase, she felt a surge of [custom word], an unexpected thrill at her error."

Write your story now, creatively using these custom words to describe unique experiences and feelings:"""
    )
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"))
    chain = prompt | llm
    return chain
    # return LLMChain(llm=llm, prompt=prompt)

# def generate_story(agent, vectorstore, query="Create a short story", num_words=10):
#     relevant_vocab = vectorstore.similarity_search(query, k=num_words)
#     vocab_text = "\n".join([doc.page_content for doc in relevant_vocab])

#     return agent.run(vocab_text)
def generate_story(agent, vectorstore, query="Create a short story", num_words=10):
    relevant_vocab = vectorstore.similarity_search(query, k=num_words)
    vocab_text = "\n".join([doc.page_content for doc in relevant_vocab])
    return agent.invoke(vocab_text)