import numpy as np
import matplotlib.pyplot as plt


# Step 1: Letters A, B, C as binary patterns (5x6 = 30 pixels)
A = [0,1,1,1,0,
     1,0,0,0,1,
     1,1,1,1,1,
     1,0,0,0,1,
     1,0,0,0,1,
     1,0,0,0,1]

B = [1,1,1,1,0,
     1,0,0,0,1,
     1,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,1,
     1,1,1,1,0]

C = [0,1,1,1,1,
     1,0,0,0,0,
     1,0,0,0,0,
     1,0,0,0,0,
     1,0,0,0,0,
     0,1,1,1,1]

X = np.array([A, B, C])  # Input data
Y = np.array([[1, 0, 0],   # A
              [0, 1, 0],   # B
              [0, 0, 1]])  # C

# Step 2: Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Step 3: Initialize weights
np.random.seed(1)
input_size = 30
hidden_size = 8
output_size = 3

W1 = np.random.rand(input_size, hidden_size)
W2 = np.random.rand(hidden_size, output_size)

# Step 4: Training the network
lr = 0.5
epochs = 1000
losses = []

for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    # Loss
    loss = np.mean((Y - A2) ** 2)
    losses.append(loss)

    # Backward pass
    dA2 = (Y - A2) * sigmoid_deriv(A2)
    dW2 = np.dot(A1.T, dA2)

    dA1 = np.dot(dA2, W2.T) * sigmoid_deriv(A1)
    dW1 = np.dot(X.T, dA1)

    # Update weights
    W2 += lr * dW2
    W1 += lr * dW1

    # Print loss every 200 epochs
    if epoch % 200 == 0:
        print(f"Epoch {epoch} Loss: {loss:.4f}")

# Step 5: Test and show result
def predict(img):
    h = sigmoid(np.dot(img, W1))
    o = sigmoid(np.dot(h, W2))
    return np.argmax(o)

letters = ['A', 'B', 'C']
for i in range(3):
    result = predict(X[i])
    print(f"True: {letters[i]}, Predicted: {letters[result]}")
    plt.imshow(np.array(X[i]).reshape(6, 5), cmap='gray')
    plt.title(f"Prediction: {letters[result]}")
    plt.axis('off')
    plt.show()

# Step 6: Plot loss
plt.plot(losses)
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()
