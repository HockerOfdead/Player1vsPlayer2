import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("player-1.csv")
dc = pd.read_csv("player-2.csv")
y_real = pd.read_csv("Result.csv")
lr = 0.01
epochs = 20000
epsilon = 1e-9
w1_1, w2_1, w3_1, w4_1, w5_1 = np.random.randn(5)  # for player 1
w1_2, w2_2, w3_2, w4_2, w5_2 = np.random.randn(5)  # for player 2
bias_1, bias_2 = 0.0, 0.0
scaler = MinMaxScaler()
cols = ['age', 'experience', 'won', 'lost', 'draw']
df[cols] = scaler.fit_transform(df[cols])
dc[cols] = scaler.transform(dc[cols])
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
def y_pred(w, bias, data):
    x = data[['age', 'experience', 'won', 'lost', 'draw']].values
    return sigmoid(np.dot(x, w) + bias)
def match_softmax(y1, y2):
    logits = np.vstack((y1, y2)).T
    return softmax(logits)
for epoch in range(epochs):
    y1 = y_pred(np.array([w1_1, w2_1, w3_1, w4_1, w5_1]), bias_1, df)
    y2 = y_pred(np.array([w1_2, w2_2, w3_2, w4_2, w5_2]), bias_2, dc)
    probs = match_softmax(y1, y2)
    p1_win = probs[:, 0]  
    p2_win = probs[:, 1]
    y_true = y_real.values.flatten()
    loss = -np.mean(y_true * np.log(p1_win + epsilon) + (1 - y_true) * np.log(p2_win + epsilon))
    dZ = probs.copy()
    dZ[:, 0] -= y_true
    dZ[:, 1] -= (1 - y_true)
    dy1 = dZ[:, 0] * y1 * (1 - y1)
    dy2 = dZ[:, 1] * y2 * (1 - y2)
    X1 = df[cols].values
    X2 = dc[cols].values
    dw1 = np.mean(X1 * dy1[:, None], axis=0)
    dw2 = np.mean(X2 * dy2[:, None], axis=0)
    db1 = np.mean(dy1)
    db2 = np.mean(dy2)
    # Gradient descent step
    w1_1, w2_1, w3_1, w4_1, w5_1 = np.array([w1_1, w2_1, w3_1, w4_1, w5_1]) - lr * dw1
    w1_2, w2_2, w3_2, w4_2, w5_2 = np.array([w1_2, w2_2, w3_2, w4_2, w5_2]) - lr * dw2
    bias_1 -= lr * db1
    bias_2 -= lr * db2
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss = {loss:.6f}")
final_probs = match_softmax(
    y_pred(np.array([w1_1, w2_1, w3_1, w4_1, w5_1]), bias_1, df),
    y_pred(np.array([w1_2, w2_2, w3_2, w4_2, w5_2]), bias_2, dc)
)
print("\nFinal probabilities (player1 win, player2 win):")
print(final_probs)
