import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import time


# Örnek
tokens = [
    'HH', 'AH', 'L', 'OW', 'DH', 'IH', 'S', 'IH', 'Z', 'AH', 'T', 'EH', 'S', 'T'
]
n_qubits = len(tokens)
token_indices = {t: i for i, t in enumerate(tokens)}


oracle_sequence = tokens.copy()
oracle_index = sum([token_indices[t] << i for i, t in enumerate(oracle_sequence)])
num_iterations = 2
dev = qml.device("default.qubit", wires=n_qubits)


def oracle():
    for i in range(n_qubits):
        if (oracle_index >> i) & 1 == 0:
            qml.PauliX(wires=i)
    controls = list(range(n_qubits-1))
    target = n_qubits-1
    qml.MultiControlledX(wires=controls + [target], control_values=[1]*(n_qubits-1))
    for i in range(n_qubits):
        if (oracle_index >> i) & 1 == 0:
            qml.PauliX(wires=i)

def diffusion():
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.PauliX(wires=i)
    controls = list(range(n_qubits-1))
    target = n_qubits-1
    qml.MultiControlledX(wires=controls + [target], control_values=[1]*(n_qubits-1))
    for i in range(n_qubits):
        qml.PauliX(wires=i)
        qml.Hadamard(wires=i)

@qml.qnode(dev)
def grover_circuit():
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for _ in range(num_iterations):
        oracle()
        diffusion()
    return qml.probs(wires=range(n_qubits))

start_q = time.time()
quantum_probs = grover_circuit()
quantum_time = time.time() - start_q
quantum_best_index = np.argmax(quantum_probs)

def decode_quantum(index):
    sequence = []
    for i in range(n_qubits):
        sequence.append(tokens[(index >> i) & 1])
    return sequence


token_probs = {t: np.random.rand() for t in tokens}
token_probs = {t: p/sum(token_probs.values()) for t, p in token_probs.items()}  # normalize
beam_width = 3
sequences = [([],1.0)]

start_c = time.time()
for _ in range(n_qubits):
    all_candidates = []
    for seq, score in sequences:
        for token, p in token_probs.items():
            candidate = (seq + [token], score * p)
            all_candidates.append(candidate)
    sequences = sorted(all_candidates, key=lambda x:x[1], reverse=True)[:beam_width]
classic_time = time.time() - start_c
classic_best_sequence = sequences[0][0]


print(f"Quantum Beam Search: {decode_quantum(quantum_best_index)} (süre: {quantum_time:.6f} s)")
print(f"Klasik Beam Search: {classic_best_sequence} (süre: {classic_time:.6f} s)")



