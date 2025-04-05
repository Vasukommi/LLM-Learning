import numpy as np

# Step 2: Embedding
# We reuse the same vocabulary and encoded tokens
vocab = {'Hello': 0, 'how': 1, 'are': 2, 'you': 3, 'hello': 4}
encoded = [0, 1, 2, 3, 4]

# Define the embedding dimension (vector size for each word)
embedding_dim = 5

# Randomly initialize embedding vectors for each token in vocabulary
embedding_matrix = np.random.rand(len(vocab), embedding_dim)

print("Embedding matrix shape:", embedding_matrix.shape)
print("Embedding matrix:\n", embedding_matrix)

# Convert encoded tokens into their embedding vectors
embedded_input = [embedding_matrix[token_id] for token_id in encoded]

print("\nEmbedded input:\n", embedded_input)