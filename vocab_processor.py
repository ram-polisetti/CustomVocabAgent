import json

def process_vocab(file_path, output_path='vocab.json'):
    vocab = {}
    with open(file_path, 'r') as file:
        for line in file:
            word, definition = line.strip().split(': ', 1)
            vocab[word] = definition

    with open(output_path, 'w') as json_file:
        json.dump(vocab, json_file)

if __name__ == "__main__":
    process_vocab('new_words.txt')
