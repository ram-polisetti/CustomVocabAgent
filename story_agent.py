import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

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
    themed_prompt = PromptTemplate(
        input_variables=["vocab", "theme"],
        template="""You have access to a custom vocabulary of invented words. Each word represents a unique concept, emotion, or action that doesn't exist in standard language. These words are NOT nouns and should not be used as names for people, places, or objects.

Custom Vocabulary:
{vocab}

STRICT THEME REQUIREMENTS:
{theme}

IMPORTANT RULES:
1. The story MUST take place exactly where specified in the theme
2. The story MUST include ONLY the characters mentioned in the theme
3. The story MUST focus on the exact actions mentioned in the theme
4. DO NOT add any additional locations or characters
5. DO NOT create a different story than what the theme specifies
6. Use the custom vocabulary words ONLY to describe emotions, sensations, and actions

Example format:
If theme is "a man vacuuming his house with a dog and brown tornado", the story must:
- Be about a man vacuuming
- Take place in his house
- Include his dog
- Include the brown tornado
- Use custom words to describe feelings and sensations during these exact events

Write your story now, strictly following the theme and using custom words to describe emotions and sensations:"""
    )
    chain = LLMChain(llm=llm, prompt=themed_prompt)
    return chain


def generate_story(agent, vectorstore, query="Create a short story", num_words=20):
    # Create a more structured theme analysis
    enhanced_query = f"""
    Story Requirements:
    Main Theme: {query}

    Must Include:
    - All characters mentioned in: {query}
    - All actions and events from: {query}
    - Exact setting from: {query}
    - All key elements mentioned in: {query}

    Story Elements to Consider:
    - Emotions and sensations during the events
    - Interactions between all mentioned characters
    - Detailed description of the specific actions
    - Atmosphere and environment of the setting
    """

    # Get relevant vocabulary based on the theme
    relevant_vocab = vectorstore.similarity_search(enhanced_query, k=num_words)
    vocab_text = "\n".join([doc.page_content for doc in relevant_vocab])

    # Generate the story with the themed prompt
    response = agent.invoke({
        "vocab": vocab_text,
        "theme": f"Write a story that MUST include all elements from: {query}. Do not deviate from the theme's characters, setting, or events."
    })

    # Create a response object with a content attribute
    class StoryResponse:
        def __init__(self, content):
            self.content = content

    return StoryResponse(response)
