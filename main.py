from vocab_processor import process_vocab
from story_agent import load_vocab, select_random_words, create_story_agent, generate_story

def main():
    # Process vocabulary
    process_vocab('new_words.txt')

    # Load processed vocabulary
    vocab = load_vocab()

    # Select random words
    selected_words = select_random_words(vocab)

    # Create story agent
    story_agent = create_story_agent()

    # Generate story
    story = generate_story(story_agent, selected_words)

    # Print and save the story
    print(story)
    with open('generated_story.txt', 'w') as file:
        file.write(story)

if __name__ == "__main__":
    main()
