from typing import List

from qulacs import Observable, QuantumCircuit
from qulacs.gate import CZ, RX, RY, RZ, Identity, Y, merge
import numpy as np
from .time_evolution_gate import create_time_evo_unitary


def he_ansatz_circuit(n_qubit, depth, theta_list):
    """he_ansatz_circuit
    Returns hardware efficient ansatz circuit.

    Args:
        n_qubit (:class:`int`):
            the number of qubit used (equivalent to the number of fermionic modes)
        depth (:class:`int`):
            depth of the circuit.
        theta_list (:class:`numpy.ndarray`):
            rotation angles.
    Returns:
        :class:`qulacs.QuantumCircuit`
    """
    circuit = QuantumCircuit(n_qubit)
    for d in range(depth):
        for i in range(n_qubit):
            circuit.add_gate(
                merge(
                    RY(i, theta_list[2 * i + 2 * n_qubit * d]),
                    RZ(i, theta_list[2 * i + 1 + 2 * n_qubit * d]),
                )
            )
        for i in range(n_qubit // 2):
            circuit.add_gate(CZ(2 * i, 2 * i + 1))
        for i in range(n_qubit // 2 - 1):
            circuit.add_gate(CZ(2 * i + 1, 2 * i + 2))
    for i in range(n_qubit):
        circuit.add_gate(
            merge(
                RY(i, theta_list[2 * i + 2 * n_qubit * depth]),
                RZ(i, theta_list[2 * i + 1 + 2 * n_qubit * depth]),
            )
        )

    return circuit


def noiseless_ansatz(
    nqubits: int,
    layers: int,
    gateset: int,
    ugateH: Observable,
    param: list[float],
) -> QuantumCircuit:

    chunks = []
    circuit = QuantumCircuit(nqubits)

    flag = layers  # theta params start
    for layer in range(layers):

        # ---- Rotation gates ----
        for i in range(gateset):
            circuit.add_gate(RX(0, param[flag + 2*i]))
            circuit.add_gate(RY(0, param[flag + 2*i + 1]))

        if layer == 0:
            ti = 0.0
            tf = param[0]
        else:
            ti = param[layer-1]
            tf = param[layer]


        time_evo_gate = create_time_evo_unitary(ugateH, ti, tf)
        circuit.add_gate(time_evo_gate)

        flag += 2 * gateset
        chunks.append(circuit.copy())


    return {
        "chunks": chunks,
        "circuit": circuit,
    }

def create_noisy_ansatz(
        nqubits: int,
        layers: int,
        ugateH: Observable,
        delta_t: float,
        param: list[float],
        C: float
) -> dict:
    chunks = []
    trotter_details = []
    circuit = QuantumCircuit(nqubits)
    
    # flag = layers matches your original indexing logic
    flag = layers  
    
    for layer in range(layers):
        # 1. Apply Rotation Gates
        circuit.add_gate(RX(0, param[flag]))
        circuit.add_gate(RY(0, param[flag + 1]))
        
        # 2. Determine time boundaries
        if layer == 0:
            ti = 0.0
            tf = param[0]
        else:
            ti = param[layer-1]
            tf = param[layer]


        # 3. Trotterize and get the nested dictionary
        circuit, trotter_dict = lie_trotter_evo(
            nqubits=nqubits, 
            circuit=circuit, 
            tf=tf, 
            ti=ti, 
            delta_t=delta_t, 
            ugateH=ugateH, 
            C=C
        )
        
        # 4. Format into your requested structure
        layer_entry = {
            "parent_layer": layer,
            "time_interval": [ti, tf],
            "trotter_details": trotter_dict
        }
        trotter_details.append(layer_entry)
        
        chunks.append(circuit.copy())
        flag += 2 
        #print(f"layer entry trotter details: {layer_entry}\n")
        
    return {
        "chunks": chunks,
        "circuit": circuit,
        "trotter_details": trotter_details
    }

def lie_trotter_evo(nqubits, circuit, tf, ti, delta_t, ugateH, C):

    n_i = int(np.ceil(abs(tf-ti) / delta_t)) if delta_t > 0 else 1
    if n_i == 0: n_i = 1
    
    depolnoise_prb = C * (tf-ti / n_i)
    # ti = ti/n_i
    # tf = tf/n_i
    time_evo_gate = create_time_evo_unitary(hamiltonian=ugateH, ti=ti/n_i, tf=tf/n_i)
    
    # Initialize the structure for this parent layer
    trotter_dict = {
        "total_steps": n_i,
        "noise_prob": depolnoise_prb,
        "steps": [] # This will hold the "deep_trotter" blocks
    }
    
    for i in range(n_i):
        circuit.add_gate(time_evo_gate)
        for q in range(nqubits):
            circuit.add_noise_gate(Identity(q), "Depolarizing", depolnoise_prb)
        
        # Append in the "deep_trotter" format requested
        trotter_dict["steps"].append({
            "deep_trotter": {
                "step_loc": i,
                "Ni": n_i
            }
        })
        
    return circuit, trotter_dict





