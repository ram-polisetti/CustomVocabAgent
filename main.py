from vocab_processor import process_vocab
from story_agent import load_vocab, create_story_agent, generate_story

def main():
    process_vocab('new_words.txt')
    vectorstore = load_vocab()
    # vectorstore = create_vectorstore(vocab)  # Implement this function
    story_agent = create_story_agent()

    # Generate story using a subset of 10 words
    story = generate_story(story_agent, vectorstore)

    print("Generated Story:")
    print(story.content)  # Access the content attribute

    with open('generated_story.txt', 'w', encoding='utf-8') as file:
        file.write(story.content)  # Write the content attribute

    print("\nStory has been saved to 'generated_story.txt'")

if __name__ == "__main__":
    main()