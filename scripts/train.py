import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# ==== Step 1: Load training data ====

# Read training data from external file
with open('data/texts.txt', 'r') as f:
    texts = [line.strip() for line in f if line.strip()]

tokens = []
for line in texts:
    tokens.extend(line.split())

# Build vocabulary
vocab = {}
for token in tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

# Encode tokens to numbers
encoded = [vocab[token] for token in tokens]

# Prepare input and target output
input_ids = np.array(encoded[:-1])   # Input tokens
target_ids = np.array(encoded[1:])   # Target tokens

# ==== Step 2: Hyperparameters ====
embedding_dim = 5
hidden_dim = 8
learning_rate = 0.01
epochs = 1000

# ==== Step 3: Auto-detect latest checkpoint and load if exists ====

def find_latest_checkpoint():
    checkpoint_dir = 'models'
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [folder for folder in os.listdir(checkpoint_dir) if folder.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1]))
    return os.path.join(checkpoint_dir, latest)

resume_checkpoint = find_latest_checkpoint()

if resume_checkpoint:
    print(f"🔄 Resuming training from checkpoint: {resume_checkpoint}")

    embedding_matrix = np.load(f'{resume_checkpoint}/embedding_matrix.npy')
    W_q = np.load(f'{resume_checkpoint}/W_q.npy')
    W_k = np.load(f'{resume_checkpoint}/W_k.npy')
    W_v = np.load(f'{resume_checkpoint}/W_v.npy')
    W1 = np.load(f'{resume_checkpoint}/W1.npy')
    b1 = np.load(f'{resume_checkpoint}/b1.npy')
    W2 = np.load(f'{resume_checkpoint}/W2.npy')
    b2 = np.load(f'{resume_checkpoint}/b2.npy')
    output_weights = np.load(f'{resume_checkpoint}/output_weights.npy')

    with open(f'{resume_checkpoint}/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    start_epoch = int(resume_checkpoint.split('_')[-1]) + 1
else:
    # Fresh training
    embedding_matrix = np.random.rand(len(vocab), embedding_dim)
    W_q = np.random.rand(embedding_dim, embedding_dim)
    W_k = np.random.rand(embedding_dim, embedding_dim)
    W_v = np.random.rand(embedding_dim, embedding_dim)
    W1 = np.random.rand(embedding_dim, hidden_dim)
    b1 = np.random.rand(hidden_dim)
    W2 = np.random.rand(hidden_dim, embedding_dim)
    b2 = np.random.rand(embedding_dim)
    output_weights = np.random.rand(embedding_dim, len(vocab))
    start_epoch = 0

# ==== Step 4: Helper functions ====

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def cross_entropy(predictions, targets):
    predictions = np.clip(predictions, 1e-12, 1.0)
    return -np.sum(np.log(predictions[np.arange(len(targets)), targets])) / len(targets)

def save_checkpoint(epoch, embedding_matrix, W_q, W_k, W_v, W1, b1, W2, b2, output_weights, vocab):
    checkpoint_dir = f'models/checkpoint_epoch_{epoch}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    np.save(f'{checkpoint_dir}/embedding_matrix.npy', embedding_matrix)
    np.save(f'{checkpoint_dir}/W_q.npy', W_q)
    np.save(f'{checkpoint_dir}/W_k.npy', W_k)
    np.save(f'{checkpoint_dir}/W_v.npy', W_v)
    np.save(f'{checkpoint_dir}/W1.npy', W1)
    np.save(f'{checkpoint_dir}/b1.npy', b1)
    np.save(f'{checkpoint_dir}/W2.npy', W2)
    np.save(f'{checkpoint_dir}/b2.npy', b2)
    np.save(f'{checkpoint_dir}/output_weights.npy', output_weights)

    with open(f'{checkpoint_dir}/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    print(f"✅ Checkpoint saved at epoch {epoch}")

# ==== Step 5: Training loop ====

loss_history = []  # Track loss for plotting

for epoch in range(start_epoch, epochs):
    # === Forward pass ===
    embedded_input = embedding_matrix[input_ids]  # Tokens to vectors

    # Self-attention
    Q = embedded_input @ W_q
    K = embedded_input @ W_k
    V = embedded_input @ W_v

    attention_scores = Q @ K.T
    scale = np.sqrt(embedding_dim)
    attention_scores = attention_scores / scale
    attention_weights = softmax(attention_scores)

    attention_output = attention_weights @ V

    # Feedforward
    hidden_output = relu(attention_output @ W1 + b1)
    feedforward_output = hidden_output @ W2 + b2

    # Output layer
    logits = feedforward_output @ output_weights
    probabilities = softmax(logits)

    # === Loss calculation ===
    loss = cross_entropy(probabilities, target_ids)
    loss_history.append(loss)

    # === Backward pass ===
    d_logits = probabilities
    d_logits[np.arange(len(target_ids)), target_ids] -= 1
    d_logits /= len(target_ids)

    d_output_weights = feedforward_output.T @ d_logits
    d_feedforward_output = d_logits @ output_weights.T

    d_W2 = hidden_output.T @ d_feedforward_output
    d_b2 = d_feedforward_output.sum(axis=0)

    d_hidden_output = d_feedforward_output @ W2.T * relu_derivative(hidden_output)

    d_W1 = attention_output.T @ d_hidden_output
    d_b1 = d_hidden_output.sum(axis=0)

    d_attention_output = d_hidden_output @ W1.T

    d_attention_weights = d_attention_output @ V.T
    d_V = attention_weights.T @ d_attention_output

    d_attention_scores = d_attention_weights * attention_weights * (1 - attention_weights)

    d_Q = d_attention_scores @ K
    d_K = d_attention_scores.T @ Q

    d_W_q = embedded_input.T @ d_Q
    d_W_k = embedded_input.T @ d_K
    d_W_v = embedded_input.T @ d_V

    d_embedded_input = d_Q @ W_q.T + d_K @ W_k.T + d_V @ W_v.T
    d_embedding_matrix = np.zeros_like(embedding_matrix)
    for idx, token_id in enumerate(input_ids):
        d_embedding_matrix[token_id] += d_embedded_input[idx]

    # === Update weights ===
    output_weights -= learning_rate * d_output_weights
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W_q -= learning_rate * d_W_q
    W_k -= learning_rate * d_W_k
    W_v -= learning_rate * d_W_v
    embedding_matrix -= learning_rate * d_embedding_matrix

    # Print loss and save checkpoints
    if epoch != 0 and epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        save_checkpoint(epoch, embedding_matrix, W_q, W_k, W_v, W1, b1, W2, b2, output_weights, vocab)

# ==== Step 6: After training ====

predicted_indices = np.argmax(probabilities, axis=1)
reverse_vocab = {idx: token for token, idx in vocab.items()}
predicted_tokens = [reverse_vocab[idx] for idx in predicted_indices]

print("\nPredicted tokens after full training:\n", predicted_tokens)

# Save final model
np.save('models/embedding_matrix.npy', embedding_matrix)
np.save('models/W_q.npy', W_q)
np.save('models/W_k.npy', W_k)
np.save('models/W_v.npy', W_v)
np.save('models/W1.npy', W1)
np.save('models/b1.npy', b1)
np.save('models/W2.npy', W2)
np.save('models/b2.npy', b2)
np.save('models/output_weights.npy', output_weights)

with open('models/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print("\n✅ Model saved successfully!")

# Plot and save loss curve
plt.plot(loss_history)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('models/loss_curve.png')
plt.show()

print("\n📈 Loss curve saved to models/loss_curve.png")