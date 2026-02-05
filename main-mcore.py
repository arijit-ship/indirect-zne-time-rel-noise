import json
import os
import sys
import time
import shutil
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

from src.modules import get_eigen_min
from src.observable import constructObservable
from src.vqe import IndirectVQE

# Global symbol count for console output
SYMBOL_COUNT = 25

def convert_numpy(obj):
    """Recursively convert NumPy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"CRITICAL ERROR: Config file '{config_path}' not found.")
        return None
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def vqe_worker_task(iteration_id, nqubits, state, obs_def, obs_coeffs, vqe_profile, 
                    ansatz_profile, noise_profile, initialparam, run_type, 
                    C, del_t, output_subfolder, exact_cost, full_config):
    """
    Worker task that produces a 'full' JSON per iteration, matching the original format.
    """
    try:
        start_time = time.time()
        
        # Reconstruct the observable locally
        local_observable = constructObservable(
            nqubits=nqubits, 
            definition=obs_def, 
            coefficient=obs_coeffs
        )

        vqe_instance = IndirectVQE(
            nqubits=nqubits,
            state=state,
            observable=local_observable,
            vqe_profile=vqe_profile,
            ansatz_profile=ansatz_profile,
            noise_profile=noise_profile,
            identity_factors=[0, 0, 0, 0],
            init_param=initialparam,
            run=run_type,
            C=C,
            del_t=del_t,
        )
        
        vqe_output = vqe_instance.run_vqe()
        total_run_time = time.time() - start_time
        
        # Replicating your original output structure exactly
        output_data = {
            "config": full_config,
            "output": {
                "iteration_index": iteration_id,
                "exact_sol": exact_cost,
                "initial_cost_history": [vqe_output["initial_cost"]],
                "initial_randm_param_values": [vqe_output["initial_param_dict"]],
                "optimized_minimum_cost": [vqe_output["min_cost"]],
                "optimized_parameters": [vqe_output["optimized_param"]],
                "noise_details": vqe_instance.get_noise_level(),
                "run_time_sec": total_run_time,
                "lie_trotter_details_all": [vqe_output.get("lie_trotter_details")]
            },
            "others": {
                "observable_string": str(local_observable),
                "time_evolution_gate_hamiltonian_string": [str(vqe_instance.get_ugate_hamiltonain())],
            },
        }

        # Atomic file write
        file_path = os.path.join(output_subfolder, f"iter_{iteration_id:04d}.json")
        with open(file_path, "w") as f:
            json.dump(convert_numpy(output_data), f, indent=4)
        
        return iteration_id, True
    except Exception as e:
        return iteration_id, str(e)

def initialize_vqe_bigT_noisy(u_gate_final_time, C, del_t, vars_dict, original_config_path):
    config = vars_dict['config']
    ansatz = vars_dict['ansatz']
    vqe_iteration = config["vqe"]["iteration"]
    
    # Setup Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{vars_dict['file_name_prefix']}_bigT_{u_gate_final_time}_{timestamp}"
    output_subfolder = os.path.join(os.getcwd(), "output", folder_name)
    os.makedirs(output_subfolder, exist_ok=True)

    # Save original config copy
    shutil.copy2(original_config_path, os.path.join(output_subfolder, "original_config.yml"))

    print(f"\nRunning {vqe_iteration} Parallel Iterations (bigT={u_gate_final_time})...")
    
    # Update ansatz time for the specific bigT run
    ansatz["ugate"]["time"]["max"] = u_gate_final_time

    # Use i9 potential (Total cores - 2)
    num_workers = max(1, multiprocessing.cpu_count() - 2)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                vqe_worker_task, i, vars_dict['nqubits'], vars_dict['state'], 
                vars_dict['obs_def'], vars_dict['obs_coeffs'], vars_dict['vqe_profile'], 
                ansatz, vars_dict['noise_profile'], vars_dict['initialparam'],
                "vqe-bigT-simulation-noisy", C, del_t, output_subfolder, 
                vars_dict['exact_cost'], config
            ) for i in range(vqe_iteration)
        ]

        for future in as_completed(futures):
            iter_id, status = future.result()
            if status is True:
                print(f"  [COMPLETED] Iteration {iter_id + 1}")
            else:
                print(f"  [ERROR] Iteration {iter_id + 1}: {status}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <config_file>")
        sys.exit(1)

    config_path = os.path.abspath(sys.argv[1])
    config = load_config(config_path)

    if config:
        # Pre-calculation and packaging
        obs_def = config["observable"]["def"]
        obs_coeffs = config["observable"]["coefficients"]
        main_obs = constructObservable(config["nqubits"], obs_def, obs_coeffs)
        
        vars_dict = {
            'config': config, 'nqubits': config["nqubits"], 'state': config["state"],
            'obs_def': obs_def, 'obs_coeffs': obs_coeffs,
            'vqe_profile': config["vqe"], 'ansatz': config["ansatz"], 
            'noise_profile': config["noise_profile"],
            'initialparam': config["init_param"]["value"], 
            'exact_cost': float(get_eigen_min(hamiltonian=main_obs)), 
            'file_name_prefix': config["output"]["file_name_prefix"]
        }

        if config["run"].lower() == "vqe-bigt-simulation-noisy":
            for t_val in sorted(config["bigT"]): # Processing in increasing order of time
                initialize_vqe_bigT_noisy(t_val, config["C"], config["del_t"], vars_dict, config_path)