import numpy as np

# Step 3: Prepare for Attention Mechanism
# Embedded input converted to numpy array for matrix operations
embedded_input = np.array([
    [0.0616467, 0.23457053, 0.81897078, 0.64483964, 0.6291443],
    [0.4556379, 0.34064539, 0.78426019, 0.13942869, 0.35841025],
    [0.83919264, 0.92968068, 0.17596456, 0.45292273, 0.64595507],
    [0.28503859, 0.64368302, 0.70233828, 0.6449667, 0.85159985],
    [0.97214249, 0.7383931, 0.91528112, 0.44705381, 0.77576897]
])

# Get the dimension size (number of columns)
embedding_dim = embedded_input.shape[1]

# Step 4: Initialize random weight matrices for Q, K, V
W_q = np.random.rand(embedding_dim, embedding_dim)  # Query weights
W_k = np.random.rand(embedding_dim, embedding_dim)  # Key weights
W_v = np.random.rand(embedding_dim, embedding_dim)  # Value weights

# Step 5: Create Query, Key, and Value matrices by multiplying embedded input with weights
Q = embedded_input @ W_q  # Queries
K = embedded_input @ W_k  # Keys
V = embedded_input @ W_v  # Values

# Step 6: Calculate raw attention scores (dot product of Q and K transpose)
attention_scores = Q @ K.T

# Scale the scores to prevent very large numbers
scale = np.sqrt(embedding_dim)
attention_scores = attention_scores / scale

print("Attention scores:\n", attention_scores)

# Step 7: Define softmax function to normalize attention scores to probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))  # For numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Apply softmax to attention scores to get attention weights (focus percentages)
attention_weights = softmax(attention_scores)

print("\nAttention weights:\n", attention_weights)

# Step 8: Multiply attention weights by Value vectors to get the final context-aware representations
attention_output = attention_weights @ V

print("\nAttention output (context vectors):\n", attention_output)