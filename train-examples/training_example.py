# -*- coding: utf-8 -*-

from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff('/content/NEFS_SAC/Earthquakes_TRAIN.arff')
df = pd.DataFrame(data)


for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode('utf-8')

print(df.head())

from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff('/content/NEFS_SAC/Earthquakes_TEST.arff')
test = pd.DataFrame(data)


for col in test.select_dtypes([object]):
    test[col] = test[col].str.decode('utf-8')

print(test.head())

"""# SMOTE OR DATA BALANCING AND MIN-MAX SCALING AND DISCRETE WAVELET TRANSFORM RECOMMENDED (LOOK THE PAPER)

In this notebook there is no SMOTE applying if you want you can do it self.
"""

import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(columns=["target"])), columns=df.columns[:-1])
df_scaled.fillna(df_scaled.mean(), inplace=True)
y = df["target"].astype(float).values

def wavelet_transform(row, wavelet='db4', level=3):
    coeffs = pywt.wavedec(row, wavelet, level=level)
    return np.concatenate([c.flatten() for c in coeffs])

def sliding_window_features(row, window=24):
    feats = []
    for i in range(0, len(row)-window+1):
        w = row[i:i+window]
        feats.extend([w.mean(), w.std(), w.min(), w.max(), np.polyfit(range(window), w, 1)[0]])
    return np.array(feats)

X = np.array([np.concatenate([wavelet_transform(r), sliding_window_features(r)]) for r in df_scaled.values])

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        return y.transpose(1, 2)

def logistic_map(x, r=3.99):
    return r * x * (1 - x)

def henon_map_vec(x, y, a=1.4, b=0.3):
    return 1 - a * x**2 + y, b * x

class ChaosLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x, hx, cx):
        h, c = self.lstm_cell(x, (hx, cx))
        h = logistic_map(h)
        x_h, y_h = h[:, 0], h[:, 1]
        x_new, y_new = henon_map_vec(x_h, y_h)
        h = torch.cat([x_new.unsqueeze(1), y_new.unsqueeze(1), h[:, 2:]], dim=1)
        return h, c

n_qubits = 6
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RX(inputs[i % len(inputs)], wires=i)
        qml.RY(inputs[(i + 1) % len(inputs)], wires=i)
        qml.RZ(inputs[(i + 2) % len(inputs)], wires=i)
    for l in range(weights.shape[0]):
        for q in range(n_qubits):
            qml.RY(weights[l, q], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

qnode = qml.QNode(quantum_circuit, dev, interface="torch")
qlayer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits)})

def batched_quantum_layer(x):
    return torch.cat([qlayer(x[i, :n_qubits]).unsqueeze(0) for i in range(x.size(0))], dim=0)

class QuantumRNNCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(n_qubits, hidden_size)

    def forward(self, x, hx):
        return hx + self.fc(batched_quantum_layer(x))

class QuChater(nn.Module):
    def __init__(self, input_size, tcn_hidden=32, lstm_hidden=32):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, tcn_hidden)
        self.chaos = ChaosLSTMCell(tcn_hidden, lstm_hidden)
        self.qrnn = QuantumRNNCell(lstm_hidden)
        self.out = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        b = x.size(0)
        t = self.tcn(x)
        hx = torch.zeros(b, t.size(2), device=x.device)
        cx = torch.zeros(b, t.size(2), device=x.device)
        for i in range(t.size(1)):
            hx, cx = self.chaos(t[:, i, :], hx, cx)
            hx = self.qrnn(hx, hx)
        return torch.sigmoid(self.out(hx))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

model = QuChater(X_train.size(2)).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for _ in range(50):
    optimizer.zero_grad()
    loss = criterion(model(X_train).squeeze(), y_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    p = model(X_test).squeeze().cpu().numpy()
    yb = np.round(p)

accuracy = accuracy_score(y_test.cpu().numpy(), yb)
precision = precision_score(y_test.cpu().numpy(), yb, zero_division=0)
recall = recall_score(y_test.cpu().numpy(), yb, zero_division=0)
f1 = f1_score(y_test.cpu().numpy(), yb, zero_division=0)
roc_auc = roc_auc_score(y_test.cpu().numpy(), p) if len(np.unique(y_test.cpu().numpy())) > 1 else 0.0
tn, fp, fn, tp = confusion_matrix(y_test.cpu().numpy(), yb).ravel()

print("Test Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}")
