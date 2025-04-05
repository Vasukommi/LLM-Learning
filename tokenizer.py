text = "Hello how are you hello"

tokens = text.split()

vocab = {}
for token in tokens:
    if token not in vocab:
        vocab[token] = len(vocab)
        
encoded = [vocab[token] for token in tokens]

print("Vocabulary:", vocab)
print("Encoded:", encoded)