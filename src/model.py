import torch
import torch.nn as nn
import pennylane as qml

# -----------------------------
# Temporal Convolution Network
# -----------------------------
class TemporalConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=input_size if i == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU())
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        return y.transpose(1, 2)

# -----------------------------
# Chaos Functions
# -----------------------------
def logistic_map(x, r=3.99):
    return r * x * (1 - x)

def henon_map_vec(x, y, a=1.4, b=0.3):
    x_new = 1 - a * x**2 + y
    y_new = b * x
    return x_new, y_new

# -----------------------------
# Chaos LSTM Cell
# -----------------------------
class ChaosLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x, hx, cx):
        h, c = self.lstm_cell(x, (hx, cx))
        h_chaos = logistic_map(h)

        x_h = h_chaos[:, 0]
        y_h = h_chaos[:, 1]
        x_new, y_new = henon_map_vec(x_h, y_h)

        h_chaos = torch.cat(
            [x_new.unsqueeze(1), y_new.unsqueeze(1), h_chaos[:, 2:]],
            dim=1
        )
        return h_chaos, c

# -----------------------------
# Quantum Layer
# -----------------------------
n_qubits = 6 # Best qubit number according the tests
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
weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

def batched_quantum_layer(x_batch):
    outputs = []
    for i in range(x_batch.size(0)):
        q_out = qlayer(x_batch[i, :n_qubits])
        outputs.append(q_out.unsqueeze(0))
    return torch.cat(outputs, dim=0)

# -----------------------------
# Quantum RNN Cell
# -----------------------------
class QuantumRNNCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc_q = nn.Linear(n_qubits, hidden_size)

    def forward(self, x, hx):
        q_out = batched_quantum_layer(x)
        q_mapped = self.fc_q(q_out)
        return hx + q_mapped

# -----------------------------
# QuChater Model
# -----------------------------
class QuChater(nn.Module):
    def __init__(self, input_size, tcn_hidden=32, lstm_hidden=32):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, tcn_hidden)
        self.chaos_lstm = ChaosLSTMCell(tcn_hidden, lstm_hidden)
        self.quantum_rnn = QuantumRNNCell(lstm_hidden)
        self.fc_out = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        batch_size = x.size(0)
        tcn_out = self.tcn(x)

        hx = torch.zeros(batch_size, tcn_out.size(2), device=x.device)
        cx = torch.zeros(batch_size, tcn_out.size(2), device=x.device)

        for t in range(tcn_out.size(1)):
            hx, cx = self.chaos_lstm(tcn_out[:, t, :], hx, cx)
            hx = self.quantum_rnn(hx, hx)

        return torch.sigmoid(self.fc_out(hx))
        
