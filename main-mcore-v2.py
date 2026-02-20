"""
Multicore version of running indirect-cntrl VQE
"""

import json
import os
import sys
import time
import shutil
import multiprocessing
import copy
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

# --- PREVENT THREAD OVERLAP ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.modules import get_eigen_min
from src.observable import constructObservable
from src.vqe import IndirectVQE

def convert_numpy(obj):
    if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, np.generic): return obj.item()
    else: return obj

def vqe_worker_task(iteration_id, t_val, nqubits, state, obs_def, obs_coeffs, vqe_profile, 
                    ansatz_profile, noise_profile, initialparam, run_type, 
                    C, del_t, output_subfolder, exact_cost, full_config):
    try:
        start_time = time.time()
        
        # MISSION CRITICAL: Sync metadata with actual execution parameters
        local_run_config = copy.deepcopy(full_config)
        local_run_config["ansatz"]["ugate"]["time"]["max"] = t_val
        
        # Use the specific ansatz modified for this process
        local_ansatz = local_run_config["ansatz"]

        local_observable = constructObservable(nqubits=nqubits, definition=obs_def, coefficient=obs_coeffs)

        vqe_instance = IndirectVQE(
            nqubits=nqubits, state=state, observable=local_observable,
            vqe_profile=vqe_profile, ansatz_profile=local_ansatz,
            noise_profile=noise_profile, identity_factors=[0, 0, 0, 0],
            init_param=initialparam, run=run_type, C=C, del_t=del_t,
        )
        
        vqe_output = vqe_instance.run_vqe()
        total_run_time = time.time() - start_time
        
        output_data = {
            "config": local_run_config, # This now contains the actual T-max used
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

        file_path = os.path.join(output_subfolder, f"iter_{iteration_id:04d}.json")
        with open(file_path, "w") as f:
            json.dump(convert_numpy(output_data), f, indent=4)
        
        return (t_val, iteration_id), True
    except Exception as e:
        return (t_val, iteration_id), str(e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    config_path = os.path.abspath(sys.argv[1])
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
            # Name variables
        nqubit = config["nqubits"]
        c_val = config["C"]
        del_t = config["del_t"]
        layer_val = config["ansatz"]["layer"]
    if config and config["run"].lower() == "vqe-bigt-simulation-noisy":
        # 1. Parent Directory Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_folder_name = f"VQE_Experiment_q{nqubit}_l{layer_val}_c{c_val}_delt{del_t}_{timestamp}"
        parent_dir = os.path.join(os.getcwd(), "output", parent_folder_name)
        os.makedirs(parent_dir, exist_ok=True)
        
        # Save ONE copy of original config in parent dir
        shutil.copy2(config_path, os.path.join(parent_dir, "original_config.yml"))

        # 2. Pre-setup subdirectories and prepare Task List
        obs_def, obs_coeffs = config["observable"]["def"], config["observable"]["coefficients"]
        main_obs = constructObservable(config["nqubits"], obs_def, obs_coeffs)
        exact_cost = float(get_eigen_min(hamiltonian=main_obs))
        
        t_folder_map = {}
        all_tasks = []

        for t_val in sorted(config["bigT"]):
            # Create a subfolder for each BigT inside the parent
            subfolder_name = f"{config['output']['file_name_prefix']}_q{nqubit}_l{layer_val}_c{c_val}_delt{del_t}_bigT_{t_val}"
            subfolder_path = os.path.join(parent_dir, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            t_folder_map[t_val] = subfolder_path
            
            for i in range(config["vqe"]["iteration"]):
                all_tasks.append((t_val, i))

        # 3. Process Pool Execution
        num_workers = max(1, multiprocessing.cpu_count() - 2)
        print(f"Saturating {num_workers} cores for {len(all_tasks)} total tasks.")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    vqe_worker_task, iter_id, t_val, config['nqubits'], config['state'], 
                    obs_def, obs_coeffs, config['vqe'], config['ansatz'], 
                    config['noise_profile'], config['init_param']['value'],
                    "vqe-bigT-simulation-noisy", config['C'], config['del_t'], 
                    t_folder_map[t_val], exact_cost, config
                ) for t_val, iter_id in all_tasks
            ]

            for future in as_completed(futures):
                (t, idx), status = future.result()
                if status is not True:
                    print(f"CRITICAL ERROR at bigT={t}, Iter={idx}: {status}")