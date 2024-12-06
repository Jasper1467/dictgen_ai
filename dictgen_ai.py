import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Make sure the model is in evaluation mode
model.eval()

# Function to generate related words based on a keyword
def generate_wordlist(keyword, num_words):
    # Encode the keyword and generate text based on it
    inputs = tokenizer.encode(keyword, return_tensors='pt')

    # Generate outputs using GPT-2
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50, num_return_sequences=num_words, no_repeat_ngram_size=2, top_p=0.95, top_k=60)

    wordlist = []
    for i in range(num_words):
        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        words = generated_text.split()
        wordlist.append(words[1])  # The first word is the keyword, so we skip it

    return wordlist

# Function to write the generated wordlist to a file
def write_wordlist_to_file(wordlist, output_file):
    with open(output_file, 'w') as file:
        for word in wordlist:
            file.write(word + '\n')
    print(f"Wordlist has been written to {output_file}")

# Main function to parse arguments and run the wordlist generator
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate a wordlist based on a keyword using AI.')
    parser.add_argument('keyword', type=str, help='The keyword to generate related words for.')
    parser.add_argument('-o', '--output', type=str, required=True, help='The output file path to save the wordlist.')
    parser.add_argument('-n', '--num_words', type=int, default=10, help='Number of words to generate (default is 10).')

    args = parser.parse_args()

    # Generate wordlist based on the keyword
    wordlist = generate_wordlist(args.keyword, args.num_words)

    # Write the wordlist to the specified file
    write_wordlist_to_file(wordlist, args.output)

if __name__ == '__main__':
    main()
