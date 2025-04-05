import numpy as np
import pickle

# Load trained weights
embedding_matrix = np.load('models/embedding_matrix.npy')
W_q = np.load('models/W_q.npy')
W_k = np.load('models/W_k.npy')
W_v = np.load('models/W_v.npy')
W1 = np.load('models/W1.npy')
b1 = np.load('models/b1.npy')
W2 = np.load('models/W2.npy')
b2 = np.load('models/b2.npy')
output_weights = np.load('models/output_weights.npy')

# Load vocab
import pickle
with open('models/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

reverse_vocab = {idx: token for token, idx in vocab.items()}

embedding_dim = embedding_matrix.shape[1]
hidden_dim = W1.shape[1]

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Generate function
def generate_text(start_token, num_words):
    current_token = start_token
    generated = [current_token]

    for _ in range(num_words):
        # Prepare input
        token_id = vocab.get(current_token, 0)  # default to 'hello' if unknown
        embedded_input = embedding_matrix[np.array([token_id])]

        # Self-attention
        Q = embedded_input @ W_q
        K = embedded_input @ W_k
        V = embedded_input @ W_v

        attention_scores = Q @ K.T
        scale = np.sqrt(embedding_matrix.shape[1])
        attention_scores = attention_scores / scale
        attention_weights = softmax(attention_scores)

        attention_output = attention_weights @ V

        # Feedforward
        hidden_output = relu(attention_output @ W1 + b1)
        feedforward_output = hidden_output @ W2 + b2

        # Output layer
        logits = feedforward_output @ output_weights

        # 🔥 Temperature scaling (keep!)
        temperature = 0.8
        logits = logits / temperature

        probabilities = softmax(logits)

        # 🔥 Random sampling based on probability
        next_token_id = np.random.choice(len(probabilities[0]), p=probabilities[0])
        next_token = reverse_vocab[next_token_id]

        # Add next token to generated sentence
        generated.append(next_token)

        # Important! Feed predicted token back as input
        current_token = next_token

    return ' '.join(generated)

# Test generation
start_word = 'hello'
generated_sentence = generate_text(start_word, num_words=20)
print("\nGenerated sentence:\n", generated_sentence)
