# Step 1: Tokenization
text = "Hello how are you hello"

# Split the text into individual words (tokens)
tokens = text.split()

# Create vocabulary: assign a unique number to each unique token
vocab = {}
for token in tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

# Encode the text: convert each token to its corresponding number
encoded = [vocab[token] for token in tokens]

print("Vocabulary:", vocab)
print("Encoded:", encoded)