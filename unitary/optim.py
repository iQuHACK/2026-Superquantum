import numpy as np
import math
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Operator
from utils import Ry

# Constants
THETA = math.pi / 7
EXPECTED_U = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0.974928, -0.222521],
    [0, 0, 0.222521, 0.974928]
], dtype=complex)

def get_t_count(qc):
    """Counts T and Tdg gates after flattening."""
    flat_qc = transpile(qc, basis_gates=['h', 's', 't', 'tdg', 'cx'], optimization_level=1)
    ops = flat_qc.count_ops()
    return ops.get('t', 0) + ops.get('tdg', 0)

def get_distance(qc, expected):
    """Calculates the best-case distance considering global phase."""
    actual = Operator(qc).data
    # Aligning the 4x4 matrices
    min_dist = float('inf')
    for phase in np.linspace(0, 2*np.pi, 100):
        dist = np.linalg.norm((np.exp(1j * phase) * actual) - expected)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def run_optimization():
    results = []
    # Test epsilon from 0.1 down to 1e-7
    epsilons = [10**(-i/2) for i in range(2, 15)] 

    print(f"{'Epsilon':<10} | {'T-Count':<8} | {'Distance':<12} | {'Score'}")
    print("-" * 50)

    for eps in epsilons:
        qc = QuantumCircuit(2)
        # Using your Ry implementation
        qc.append(Ry(THETA/2, eps).to_gate(), [1])
        qc.cx(0, 1)
        qc.append(Ry(-THETA/2, eps).to_gate(), [1])
        qc.cx(0, 1)

        t_count = get_t_count(qc)
        dist = get_distance(qc, EXPECTED_U)
        
        # Metric: Lower is better. 
        # We weight Distance heavily because passing the test is binary (pass/fail)
        score = t_count * (dist + 1e-10) 
        
        results.append((eps, t_count, dist, score))
        print(f"{eps:.1e}    | {t_count:<8} | {dist:.2e}   | {score:.4f}")

    # Find the best trade-off (min score)
    best = min(results, key=lambda x: x[3])
    print("-" * 50)
    print(f"BEST CONFIG: Epsilon {best[0]:.1e} with {best[1]} T-gates (Dist: {best[2]:.2e})")

if __name__ == "__main__":
    run_optimization()