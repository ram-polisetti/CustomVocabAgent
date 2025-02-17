# image_processor.py

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def get_initial_description(image_url):
    """
    Get initial description of the image
    """
    client = Groq(api_key=api_key)
    try:
        initial_description = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",  # Using smaller model for initial description
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you see in this image in detail, focusing on emotions, actions, and sensations."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_completion_tokens=512
        )
        return initial_description.choices[0].message.content
    except Exception as e:
        return f"Error getting initial description: {str(e)}"

def analyze_image_with_vocab(image_url, vocab_text):
    """
    Analyze an image using Groq's vision model and custom vocabulary
    """
    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You have access to a custom vocabulary of invented words. Each word represents a unique concept, emotion, or action that doesn't exist in standard language. These words are NOT nouns and should not be used as names for people, places, or objects.

Custom Vocabulary:
{vocab_text}

Analyze this image using these custom vocabulary words. Describe what you see in the image, incorporating these words as descriptions of emotions, sensations, or actions. Use them to describe how the scene, people, or elements in the image might feel or experience things in unique ways. Do not use these words as names or objects.

For example, if a word is defined as "a feeling of excitement when making a mistake," you might write: "The person in the image seems to radiate [custom word], as if finding unexpected joy in their imperfect moment."

Provide your image analysis now, creatively using these custom words to describe the visual elements and their emotional qualities:"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_completion_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def process_image_with_vectorstore(image_url, vectorstore):
    """
    Process image using semantically relevant vocabulary from vectorstore
    """
    try:
        # First get initial description
        initial_description = get_initial_description(image_url)

        # Use description to find relevant vocabulary
        relevant_vocab = vectorstore.similarity_search(initial_description, k=20)  # Reduced from 1000 to 20
        vocab_text = "\n".join([doc.page_content for doc in relevant_vocab])

        # Get final analysis with relevant vocabulary
        return analyze_image_with_vocab(image_url, vocab_text)
    except Exception as e:
        return f"Error processing image: {str(e)}"
