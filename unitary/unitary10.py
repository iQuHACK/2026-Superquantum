"""
unitary10_multicore.py  --  multiprocessing across target distances

This keeps the same optimization logic as unitary10.py, but evaluates each
TARGET_DISTANCE in a separate process to improve CPU utilization. Results are
printed in ascending target order to match the sequential script's output order.
"""

import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import os
import argparse
import multiprocessing as mp
import numpy as np

from qiskit import QuantumCircuit, quantum_info, transpile
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3
from qiskit.circuit.library import UnitaryGate

from optim import _synthesize, normalize_angle, build_circuit, total_t_count

ANGLE_TOL = 1e-1

TARGET_DISTANCES = [
    1e-1, 7e-2, 5e-2, 3e-2, 2e-2, 1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3
]

# Ultra-granular epsilon sweep: 100 points from 1e-1 to 1e-9
EPS_COARSE_SWEEP = np.logspace(-1, -9, 100)

RELAXATION_FACTORS = [
    100, 50, 30, 20, 15, 10, 7, 5, 3, 2, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02
]


def operator_distance(actual, reference):
    d = actual.shape[0]
    inner = np.trace(actual.conj().T @ reference)
    return float(np.sqrt(max(2 * d - 2 * np.abs(inner), 0.0)))


def extract_ops(target_matrix):
    template_qc = QuantumCircuit(2)
    template_qc.append(UnitaryGate(target_matrix), [0, 1])
    template = transpile(
        template_qc,
        basis_gates=["u3", "cx"],
        optimization_level=2,
        seed_transpiler=0,
    )

    ops = []
    for inst in template.data:
        name = inst.operation.name
        qubits = [template.find_bit(q).index for q in inst.qubits]
        if name == "cx":
            ops.append(("cx", qubits[0], qubits[1]))
        elif name in ("u3", "u"):
            theta, phi, lam = [float(p) for p in inst.operation.params]
            q = qubits[0]
            for axis, angle in [("rz", lam), ("ry", theta), ("rz", phi)]:
                a = normalize_angle(angle)
                if abs(a) > ANGLE_TOL:
                    ops.append((axis, q, a))
        elif name in ("rz", "ry", "rx"):
            a = normalize_angle(float(inst.operation.params[0]))
            if abs(a) > ANGLE_TOL:
                ops.append((name, qubits[0], a))
        elif name == "p":
            a = normalize_angle(float(inst.operation.params[0]))
            if abs(a) > ANGLE_TOL:
                ops.append(("rz", qubits[0], a))

    return ops


def optimize_for_target(args):
    target_dist, ops, rotation_indices, target = args
    n_rotations = len(rotation_indices)
    cache = {}

    # Phase 1: Uniform sweep
    best_uniform_eps = None
    for eps in EPS_COARSE_SWEEP:
        eps_list = [float(eps)] * n_rotations
        qc = build_circuit(ops, eps_list, cache)
        dist = operator_distance(Operator(qc).data, target)
        if dist < target_dist:
            best_uniform_eps = float(eps)
            break

    if best_uniform_eps is None:
        return (target_dist, False, None, None, None)

    # Phase 2: Per-rotation relaxation
    current_eps = [best_uniform_eps] * n_rotations
    current_t = total_t_count(ops, rotation_indices, current_eps, cache)
    current_dist = operator_distance(Operator(build_circuit(ops, current_eps, cache)).data, target)

    while True:
        any_improved = False
        for j in range(n_rotations):
            idx = rotation_indices[j]
            axis, angle, orig_eps = ops[idx][0], ops[idx][2], current_eps[j]
            _, orig_tc_j = _synthesize(axis, angle, orig_eps, cache)

            for factor in RELAXATION_FACTORS:
                trial_eps = orig_eps * factor
                if trial_eps > 0.5:
                    continue

                _, trial_tc_j = _synthesize(axis, angle, trial_eps, cache)
                if trial_tc_j >= orig_tc_j:
                    continue

                trial_eps_list = current_eps.copy()
                trial_eps_list[j] = trial_eps
                trial_dist = operator_distance(
                    Operator(build_circuit(ops, trial_eps_list, cache)).data, target
                )

                if trial_dist < target_dist:
                    current_t = current_t - orig_tc_j + trial_tc_j
                    current_eps[j] = trial_eps
                    current_dist = trial_dist
                    any_improved = True
                    break
        if not any_improved:
            break

    final_qc = build_circuit(ops, current_eps, cache)
    final_t = total_t_count(ops, rotation_indices, current_eps, cache)
    final_dist = operator_distance(Operator(final_qc).data, target)

    qasm_str = dumps3(final_qc)
    return (target_dist, True, final_t, final_dist, qasm_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--mp-start",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method (default: spawn).",
    )
    args = parser.parse_args()

    if not os.path.exists("qasm"):
        os.makedirs("qasm")

    unitary = quantum_info.random_unitary(4, seed=42)
    target = unitary.data

    ops = extract_ops(target)
    rotation_indices = [i for i, op in enumerate(ops) if op[0] in ("rx", "ry", "rz")]
    n_rotations = len(rotation_indices)

    print(f"Rotations to optimize: {n_rotations}")

    tasks = [(td, ops, rotation_indices, target) for td in TARGET_DISTANCES]

    ctx = mp.get_context(args.mp_start)
    with ctx.Pool(processes=args.workers) as pool:
        results = list(pool.imap_unordered(optimize_for_target, tasks, chunksize=1))

    for target_dist, success, final_t, final_dist, qasm_str in sorted(results, key=lambda r: r[0]):
        print("\n" + "=" * 60)
        print(f"RUNNING OPTIMIZATION FOR TARGET DISTANCE: {target_dist}")
        print("=" * 60)
        if not success:
            print(f"Warning: Target distance {target_dist} not reachable in sweep.")
            continue
        print(f"DONE -> Target: {target_dist} | Final T: {final_t} | Final Dist: {final_dist:.6e}")

    # Pick a single "balanced" result: normalize T-count and distance, then
    # minimize a 50/50 composite score.
    candidates = [
        (target_dist, final_t, final_dist, qasm_str)
        for target_dist, success, final_t, final_dist, qasm_str in results
        if success and qasm_str
    ]

    if not candidates:
        print("\nNo successful candidates; no QASM file written.")
        return

    t_values = [c[1] for c in candidates]
    d_values = [c[2] for c in candidates]
    t_min, t_max = min(t_values), max(t_values)
    d_min, d_max = min(d_values), max(d_values)

    def _norm(val, vmin, vmax):
        if vmax == vmin:
            return 0.0
        return (val - vmin) / (vmax - vmin)

    best = None
    for target_dist, final_t, final_dist, qasm_str in candidates:
        t_norm = _norm(final_t, t_min, t_max)
        d_norm = _norm(final_dist, d_min, d_max)
        score = 0.5 * t_norm + 0.5 * d_norm
        key = (score, final_t, final_dist)
        if best is None or key < best[0]:
            best = (key, target_dist, final_t, final_dist, qasm_str)

    _, best_target, best_t, best_dist, best_qasm = best

    qasm_path = os.path.join("qasm", "unitary10.qasm")
    with open(qasm_path, "w") as f:
        f.write(best_qasm)
    print("\n" + "=" * 60)
    print("SAVED SINGLE BALANCED RESULT")
    print("=" * 60)
    print(f"Selected target: {best_target} | T: {best_t} | Dist: {best_dist:.6e}")
    print(f"Saved to {qasm_path}")

    print("\nAll target optimizations complete.")


if __name__ == "__main__":
    main()
