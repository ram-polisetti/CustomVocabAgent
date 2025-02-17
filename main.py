from vocab_processor import process_vocab
from story_agent import load_vocab, create_story_agent, generate_story
from image_processor import process_image_with_vectorstore

def main():
    # Process vocabulary first
    process_vocab('new_words.txt')
    vectorstore = load_vocab()
    story_agent = create_story_agent()

    while True:
        print("\n=== Custom Vocabulary Creative System ===")
        print("1. Generate a story")
        print("2. Analyze an image")
        print("3. Exit")

        choice = input("\nEnter your choice (1/2/3): ")

        if choice == "1":
            # Story Generation
            query = input("Enter a theme for the story (or press Enter for default): ")
            if not query.strip():
                query = "Create a short story"

            print("\nGenerating story...")
            story = generate_story(story_agent, vectorstore, query)

            print("\nGenerated Story:")
            print(story.content)

            # Save the story to a file
            with open('generated_story.txt', 'w', encoding='utf-8') as file:
                file.write(story.content)
            print("\nStory has been saved to 'generated_story.txt'")

        elif choice == "2":
            # Image Analysis
            image_url = input("Enter the image URL: ")
            print("\nAnalyzing image...")

            analysis = process_image_with_vectorstore(image_url, vectorstore)

            print("\nImage Analysis:")
            print(analysis)

            # Save the analysis to a file
            with open('image_analysis.txt', 'w', encoding='utf-8') as file:
                file.write(analysis)
            print("\nAnalysis has been saved to 'image_analysis.txt'")

        elif choice == "3":
            print("Thank you for using the system. Goodbye!")
            break

        else:
            print("Invalid choice! Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
