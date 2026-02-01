import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator, state_fidelity
from qiskit.qasm3 import dumps as dumps3
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    CommutativeCancellation,
    InverseCancellation,
    Optimize1qGatesDecomposition,
    OptimizeCliffordT,
)

from utils import Rz, Ry

psi = np.array([
    0.1061479384 - 0.6796414670j,   # |00>
   -0.3622775887 - 0.4536131360j,   # |01>
    0.2614190429 + 0.0445330969j,   # |10>
    0.3276449279 - 0.1101628411j    # |11>
], dtype=complex)

psi = psi / np.linalg.norm(psi)    # normalize (safe)
M = psi.reshape(2, 2)
U, s, Vh = np.linalg.svd(M)
V = Vh.conj().T
s0, s1 = s
theta = 2 * np.arctan2(s1, s0)
V_star = V.conj()


def decompose_zyz(U_in: np.ndarray) -> tuple[float, float, float, float]:
    det = np.linalg.det(U_in)
    phase = np.angle(det) / 2
    U_local = U_in * np.exp(-1j * phase)

    a = U_local[0, 0]
    b = U_local[0, 1]
    c = U_local[1, 0]

    theta_local = 2 * np.arctan2(np.abs(b), np.abs(a))

    if np.isclose(np.abs(b), 0.0, atol=1e-12):
        phi_local = 0.0
        lam_local = -2 * np.angle(a)
    elif np.isclose(np.abs(a), 0.0, atol=1e-12):
        phi_local = 2 * np.angle(c)
        lam_local = 0.0
    else:
        phi_minus_lambda = 2 * np.angle(c)
        phi_plus_lambda = -2 * np.angle(a)
        phi_local = 0.5 * (phi_minus_lambda + phi_plus_lambda)
        lam_local = 0.5 * (phi_plus_lambda - phi_minus_lambda)

    return phase, phi_local, theta_local, lam_local


def normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def quantize_angle(angle: float, quantum: float | None) -> float:
    if quantum is None:
        return angle
    return float(round(angle / quantum) * quantum)


def choose_zyz_variant(phi: float, theta: float, lam: float, tol: float) -> tuple[float, float, float]:
    variants = [
        (phi, theta, lam),
        (phi + np.pi, -theta, lam + np.pi),
    ]

    best = None
    best_cost = None
    for v_phi, v_theta, v_lam in variants:
        v_phi = normalize_angle(v_phi)
        v_theta = normalize_angle(v_theta)
        v_lam = normalize_angle(v_lam)
        cost = sum(abs(a) for a in (v_phi, v_theta, v_lam) if abs(a) >= tol)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = (v_phi, v_theta, v_lam)

    return best


def append_rot(
    ops: list[tuple[str, float]],
    axis: str,
    angle: float,
    tol: float,
    quantum: float | None,
) -> None:
    angle = normalize_angle(angle)
    angle = normalize_angle(quantize_angle(angle, quantum))
    if abs(angle) < tol:
        return
    if ops and ops[-1][0] == axis:
        merged = normalize_angle(ops[-1][1] + angle)
        if abs(merged) < tol:
            ops.pop()
        else:
            ops[-1] = (axis, merged)
    else:
        ops.append((axis, angle))


RZ_GATE_CACHE: dict[tuple[float, float], object] = {}
RY_GATE_CACHE: dict[tuple[float, float], object] = {}


def cached_rz(angle: float, eps: float):
    key = (angle, eps)
    gate = RZ_GATE_CACHE.get(key)
    if gate is None:
        gate = Rz(angle, eps).to_gate()
        RZ_GATE_CACHE[key] = gate
    return gate


def cached_ry(angle: float, eps: float):
    key = (angle, eps)
    gate = RY_GATE_CACHE.get(key)
    if gate is None:
        gate = Ry(angle, eps).to_gate()
        RY_GATE_CACHE[key] = gate
    return gate


def apply_rotations(qc: QuantumCircuit, qubit: int, ops: list[tuple[str, float]], eps: float) -> None:
    for axis, angle in ops:
        if axis == "z":
            qc.append(cached_rz(angle, eps), [qubit])
        elif axis == "y":
            qc.append(cached_ry(angle, eps), [qubit])
        else:
            raise ValueError(f"Unsupported axis {axis}")


def append_zyz(qc: QuantumCircuit, qubit: int, phi: float, theta: float, lam: float, eps: float) -> None:
    if not np.isclose(phi, 0.0, atol=1e-12):
        qc.append(Rz(phi, eps).to_gate(), [qubit])
    if not np.isclose(theta, 0.0, atol=1e-12):
        qc.append(Ry(theta, eps).to_gate(), [qubit])
    if not np.isclose(lam, 0.0, atol=1e-12):
        qc.append(Rz(lam, eps).to_gate(), [qubit])


def build_approx_circuit(
    eps: float,
    theta_in: float,
    u_angles: tuple[float, float, float],
    v_angles: tuple[float, float, float],
    angle_tol: float,
    angle_quantum: float | None,
) -> QuantumCircuit:
    qc_local = QuantumCircuit(2)

    pre_q1: list[tuple[str, float]] = []
    post_q1: list[tuple[str, float]] = []
    post_q0: list[tuple[str, float]] = []

    append_rot(pre_q1, "y", theta_in, angle_tol, angle_quantum)
    append_rot(post_q1, "z", u_angles[0], angle_tol, angle_quantum)
    append_rot(post_q1, "y", u_angles[1], angle_tol, angle_quantum)
    append_rot(post_q1, "z", u_angles[2], angle_tol, angle_quantum)

    append_rot(post_q0, "z", v_angles[0], angle_tol, angle_quantum)
    append_rot(post_q0, "y", v_angles[1], angle_tol, angle_quantum)
    append_rot(post_q0, "z", v_angles[2], angle_tol, angle_quantum)

    apply_rotations(qc_local, 1, pre_q1, eps)
    qc_local.cx(1, 0)
    apply_rotations(qc_local, 1, post_q1, eps)
    apply_rotations(qc_local, 0, post_q0, eps)

    return qc_local


# ============================================================
# 3. Build a ZYZ-based approximation (no UnitaryGate)
# ============================================================

_, u_phi, u_theta, u_lam = decompose_zyz(U)
_, v_phi, v_theta, v_lam = decompose_zyz(V_star)


# ============================================================
# 4. Search epsilons for fidelity >= target
# ============================================================

basis_gates = ["h", "cx", "s", "sdg", "t", "tdg"]
max_total_gates = 10_000
target_fidelity = 0.97
fast_approx = True
angle_prune_tol = 1e-3 if fast_approx else 1e-4
angle_quantum = 1e-3 if fast_approx else None
epsilon_coarse = [
    1e-1, 5e-2, 2e-2, 1e-2,
    5e-3, 2e-3, 1e-3,
    5e-4, 2e-4, 1e-4,
    5e-5, 2e-5, 1e-5,
    5e-6, 2e-6, 1e-6,
    5e-7, 2e-7, 1e-7,
]
refine_steps = 9
num_seeds = 100
use_post_passes = True
post_passes = PassManager(
    [
        Optimize1qGatesDecomposition(["h", "s", "sdg", "t", "tdg"]),
        OptimizeCliffordT(),
        CommutativeCancellation(),
        InverseCancellation(),
    ]
)
early_gatecap_seeds = 5

best = None
best_score = None
best_fid = None
selected_eps = None
min_total_gates = None
max_fid_overall = None

def evaluate_epsilon(eps: float, u_angles: tuple[float, float, float], v_angles: tuple[float, float, float]) -> dict:
    candidate_best = None
    candidate_score = None
    candidate_fid = None
    candidate_min_total = None
    max_fid_eps = None
    gatecap_checks = 0
    gatecap_hits = 0

    base_qc = build_approx_circuit(eps, theta, u_angles, v_angles, angle_prune_tol, angle_quantum)
    print(f"\nEpsilon {eps:.1e}")

    for seed in range(num_seeds):
        tqc = transpile(
            base_qc,
            basis_gates=basis_gates,
            optimization_level=2,
            seed_transpiler=seed,
        )
        if use_post_passes:
            tqc = post_passes.run(tqc)

        ops = tqc.count_ops()
        total_gates = int(sum(ops.values()))
        if candidate_min_total is None or total_gates < candidate_min_total:
            candidate_min_total = total_gates

        t_count = ops.get("t", 0) + ops.get("tdg", 0)
        depth = tqc.depth()

        if gatecap_checks < early_gatecap_seeds:
            gatecap_checks += 1

        if total_gates > max_total_gates:
            if gatecap_checks <= early_gatecap_seeds:
                gatecap_hits += 1
                if gatecap_checks == early_gatecap_seeds and gatecap_hits == early_gatecap_seeds:
                    print(
                        f"  seed={seed:02d} total={total_gates:5d} t={t_count:4d} "
                        f"depth={depth:4d} fid=skipped -> first {early_gatecap_seeds} over cap, "
                        "skip epsilon"
                    )
                    break
            print(
                f"  seed={seed:02d} total={total_gates:5d} t={t_count:4d} "
                f"depth={depth:4d} fid=skipped -> skip (gate cap)"
            )
            continue

        score = (total_gates, t_count, depth)

        if candidate_score is not None and score >= candidate_score:
            print(
                f"  seed={seed:02d} total={total_gates:5d} t={t_count:4d} "
                f"depth={depth:4d} fid=skipped -> skip (worse score)"
            )
            continue

        fid = state_fidelity(Statevector.from_instruction(tqc), psi)
        if max_fid_eps is None or fid > max_fid_eps:
            max_fid_eps = fid

        if fid < target_fidelity:
            print(
                f"  seed={seed:02d} total={total_gates:5d} t={t_count:4d} "
                f"depth={depth:4d} fid={fid:.6f} -> skip (below target)"
            )
            continue

        candidate_best = tqc
        candidate_score = score
        candidate_fid = fid
        print(
            f"  seed={seed:02d} total={total_gates:5d} t={t_count:4d} "
            f"depth={depth:4d} fid={fid:.6f} -> best for epsilon"
        )

    return {
        "best": candidate_best,
        "score": candidate_score,
        "fid": candidate_fid,
        "min_total": candidate_min_total,
        "max_fid": max_fid_eps,
    }


u_angles = choose_zyz_variant(u_phi, u_theta, u_lam, angle_prune_tol)
v_angles = choose_zyz_variant(v_phi, v_theta, v_lam, angle_prune_tol)

print("\nCoarse epsilon sweep")
prev_eps = None
selected = None

for eps in epsilon_coarse:
    result = evaluate_epsilon(eps, u_angles, v_angles)
    if result["min_total"] is not None:
        if min_total_gates is None or result["min_total"] < min_total_gates:
            min_total_gates = result["min_total"]
    if result["max_fid"] is not None:
        if max_fid_overall is None or result["max_fid"] > max_fid_overall:
            max_fid_overall = result["max_fid"]

    if result["best"] is not None:
        selected = (eps, result)
        break
    prev_eps = eps

if selected is not None and prev_eps is not None:
    print(f"\nRefining between {prev_eps:.1e} and {selected[0]:.1e}")
    refine_eps = np.logspace(np.log10(prev_eps), np.log10(selected[0]), num=refine_steps)
    for eps in refine_eps[1:-1]:
        result = evaluate_epsilon(float(eps), u_angles, v_angles)
        if result["min_total"] is not None:
            if min_total_gates is None or result["min_total"] < min_total_gates:
                min_total_gates = result["min_total"]
        if result["max_fid"] is not None:
            if max_fid_overall is None or result["max_fid"] > max_fid_overall:
                max_fid_overall = result["max_fid"]

        if result["best"] is not None:
            selected = (float(eps), result)
            break

if selected is not None:
    selected_eps, selected_result = selected
    best = selected_result["best"]
    best_score = selected_result["score"]
    best_fid = selected_result["fid"]
else:
    selected_eps = None


if best is None:
    raise RuntimeError(
        "No candidate circuits met the fidelity target. "
        f"Target fidelity={target_fidelity:.3f}, "
        f"best observed fidelity={max_fid_overall}. "
        f"Smallest observed total_gates={min_total_gates}. "
        f"Increase max_total_gates (currently {max_total_gates}), "
        "decrease epsilon, or lower the target."
    )

print(f"\nSelected epsilon: {selected_eps:.1e}")
print("\nBest circuit (total gates, T-count, depth):", best_score)
print(best)
print("\nBest circuit fidelity:", best_fid)

qasm3_str = dumps3(best)
with open("qasm/unitary7.qasm", "w") as file:
    file.write(qasm3_str)

U4 = Operator(best).data

print("\nFinal 4x4 unitary matrix U:\n")
np.set_printoptions(precision=6, suppress=True)
print(U4)

e00 = np.array([1, 0, 0, 0], dtype=complex)
out = U4 @ e00

print("\nMax |U|00> - psi| =", np.max(np.abs(out - psi)))
