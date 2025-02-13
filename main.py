from vocab_processor import process_vocab
from story_agent import load_vocab, create_story_agent, generate_story

def main():
    # Process vocabulary if it hasn't been processed yet
    process_vocab('new_words.txt')

    # Load processed vocabulary
    vocab = load_vocab()

    # Create story agent
    story_agent = create_story_agent()

    # Generate story using the entire vocabulary
    story = generate_story(story_agent, vocab)

    # Print the story
    print("Generated Story:")
    print(story)

    # Save the story
    with open('generated_story.txt', 'w', encoding='utf-8') as file:
        file.write(story)

    print("\nStory has been saved to 'generated_story.txt'")

    # Print used vocabulary words (optional)
    used_words = [word for word in vocab.keys() if word.lower() in story.lower()]
    print(f"\nNumber of custom vocabulary words used: {len(used_words)}")
    print("Used words:", ", ".join(used_words))

if __name__ == "__main__":
    main()
