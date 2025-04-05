import numpy as np

vocab = {'Hello': 0, 'how': 1, 'are': 2, 'you': 3, 'hello': 4}
encoded = [0, 1, 2, 3, 4]

embedding_dim = 5

embedding_matrix = np.random.rand(len(vocab), embedding_dim)

print("Embedding matrix shape:", embedding_matrix.shape)
print("Embedding matrix:\n", embedding_matrix)

embedded_input = [embedding_matrix[token_id] for token_id in encoded]

print("\nEmbedded input:\n", embedded_input)