import numpy as np
import math
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3

# Import from your existing files
from utils import Ry
from test import count_t_gates_manual, distance_global_phase, expected as EXPECTED_DICT

def run_optimization(unitary_id, theta):
    # Use the EXACT matrix from test.py's dictionary
    if unitary_id not in EXPECTED_DICT:
        print(f"Error: Unitary {unitary_id} not found in test.py expected dict.")
        return
    
    expected_u = EXPECTED_DICT[unitary_id]
    epsilons = [10**(-i/2) for i in range(2, 15)] 
    
    print(f"Optimizing Unitary {unitary_id}, theta={theta:.3f}")
    print(f"{'Epsilon':<10} | {'T-Count':<8} | {'Distance':<10}")
    print("-" * 45)
    
    best_eps, best_t, best_dist = None, float('inf'), float('inf')
    
    for eps in epsilons:
        qc = QuantumCircuit(2)
        
        # STANDARD CONTROLLED-RY DECOMPOSITION
        # To match the bottom-right block rotation:
        # Control: q[0], Target: q[1]
        qc.append(Ry(theta/2, eps).to_gate(), [1])
        qc.cx(0, 1)
        qc.append(Ry(-theta/2, eps).to_gate(), [1])
        qc.cx(0, 1)
        
        # 1. Convert to QASM string (so count_t_gates_manual can read it)
        qasm_str = dumps3(qc)
        
        # 2. Count T-gates using your working string parser
        t_count = count_t_gates_manual(qasm_str)
        
        # 3. Calculate distance using your working phase-alignment
        actual = Operator(qc).data
        aligned = distance_global_phase(actual, expected_u)
        dist = np.linalg.norm(aligned - expected_u)
        
        # 4. Track best (Minimize T-count, then minimize distance)
        if t_count > 0 and (t_count < best_t or (t_count == best_t and dist < best_dist)):
            best_eps, best_t, best_dist = eps, t_count, dist
        
        print(f"{eps:.1e}    | {t_count:<8} | {dist:.2e}")
    
    print("-" * 45)
    if best_eps:
        print(f"Best Result: eps={best_eps:.1e}, T-count={best_t}, dist={best_dist:.2e}")
    else:
        print("No valid circuits generated. Check your Ry function.")

if __name__ == "__main__":
    # Ensure theta matches your unitary2.py goal
    run_optimization(3, math.pi/7)