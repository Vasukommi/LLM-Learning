import numpy as np

# Step 1: Let's reuse the attention output (context vectors) from attention.py
# These are the "thinking" outputs from attention
attention_output = np.array([
    [2.13274447, 2.88274464, 1.89638607, 1.2422839, 1.73849902],
    [2.10373205, 2.8466184, 1.87751479, 1.22955748, 1.72027421],
    [2.21435989, 2.98469597, 1.94936383, 1.2762959, 1.78745687],
    [2.21490305, 2.98502429, 1.94918587, 1.27649765, 1.78774714],
    [2.26514021, 3.04795374, 1.98169899, 1.29674903, 1.81690733]
])

# Step 2: Define feedforward network dimensions
embedding_dim = attention_output.shape[1]  # This is still 5 in our case
hidden_dim = 8  # We can choose any size; let's pick 8 for this example

# Step 3: Initialize weights and biases for feedforward layers
W1 = np.random.rand(embedding_dim, hidden_dim)  # First layer weights
b1 = np.random.rand(hidden_dim)                 # First layer bias

W2 = np.random.rand(hidden_dim, embedding_dim)  # Second layer weights
b2 = np.random.rand(embedding_dim)              # Second layer bias

# Step 4: Define activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Step 5: First linear layer + activation
hidden_output = relu(attention_output @ W1 + b1)

# Step 6: Second linear layer (no activation, output layer)
feedforward_output = hidden_output @ W2 + b2

print("Feedforward output:\n", feedforward_output)
