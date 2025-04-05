import numpy as np

# Step 1: Reuse feedforward output vectors from feedforward.py
feedforward_output = np.array([
    [20.29015972, 31.81501428, 19.43536084, 21.93026848, 16.10854684],
    [20.08197235, 31.49058867, 19.23773975, 21.7083257,  15.94803391],
    [20.86698018, 32.71556777, 19.98418289, 22.54654418, 16.55472229],
    [20.86924796, 32.71888089, 19.98623393, 22.54876064, 16.55626439],
    [21.22075079, 33.26827124, 20.32116318, 22.92480993, 16.82871165]
])

# Step 2: Define the vocabulary (same as tokenizer)
vocab = ['Hello', 'how', 'are', 'you', 'hello']  # our vocab list

# Step 3: Simulate output layer (scores for each vocab word)
# Initialize random weights to map final vectors to vocab size
output_weights = np.random.rand(feedforward_output.shape[1], len(vocab))

# Step 4: Calculate raw prediction scores
logits = feedforward_output @ output_weights

print("Logits (raw prediction scores):\n", logits)

# Step 5: Apply softmax to get prediction probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

probabilities = softmax(logits)

print("\nProbabilities:\n", probabilities)

# Step 6: Pick predicted tokens (highest probability)
predicted_token_indices = np.argmax(probabilities, axis=1)

# Step 7: Convert token indices to words
predicted_words = [vocab[idx] for idx in predicted_token_indices]

print("\nPredicted words:\n", predicted_words)
