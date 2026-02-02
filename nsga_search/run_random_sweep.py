from nsga2 import Population
from typing import List, Dict, Any, Tuple
from search_space import Individual
from search_space import HeteroSearchSpace
import yaml
from remote_trainer import RemoteTrainer  
import logging
import time
import os
import argparse
import random
import json

# Configure logging to only show INFO:root messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
# Disable all other loggers except root
for name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(name).disabled = True

def load_hosts_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hosts file not found: {path}")
    hosts: List[str] = []
    _, ext = os.path.splitext(path)
    try:
        if ext.lower() not in (".yaml", ".yml"):
            raise ValueError("Hosts file must be a YAML file (.yaml or .yml) with a top-level list of IPs")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            raise ValueError("Hosts YAML must be a top-level list, e.g.\n- 1.2.3.4\n- 5.6.7.8")

        hosts = [str(x).strip() for x in data if isinstance(x, (str, int, float)) and str(x).strip()]
    except Exception as e:
        raise RuntimeError(f"Failed to parse hosts file '{path}': {e}")

    if not hosts:
        raise ValueError(f"No hosts parsed from file: {path}")
    return hosts


def load_search_space_from_yaml(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Search space file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Search space YAML must define a mapping with 'global_spec' and 'layer_spec'.")

    global_spec = data.get("global_spec")
    layer_spec = data.get("layer_spec")

    if not isinstance(global_spec, dict) or not isinstance(layer_spec, dict):
        raise ValueError("Search space YAML missing 'global_spec' or 'layer_spec' dictionaries.")

    return global_spec, layer_spec


def load_initial_individuals(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Initial population file not found: {path}")

    _, ext = os.path.splitext(path)
    with open(path, "r", encoding="utf-8") as f:
        if ext.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif ext.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError("Initial population file must be .json, .yaml, or .yml")

    # Support either a list of individuals or a dict with an 'individuals' key
    if isinstance(data, dict) and "individuals" in data and isinstance(data["individuals"], list):
        data = data["individuals"]
    elif isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("Initial population must be a list of individual dicts or a single dict")

    individuals: List[Dict[str, Any]] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Initial individual at index {idx} is not a dict")
        individuals.append(entry)

    if not individuals:
        raise ValueError("No individuals found in the initial population file")

    return individuals


def parse_constraint_arg(entry: str) -> Tuple[str, float]:
    if "=" not in entry:
        raise argparse.ArgumentTypeError("Constraints must be formatted as key=value")
    key, value = entry.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Constraint key cannot be empty")
    try:
        return key, float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Constraint value for '{key}' must be numeric")


def main():
    parser = argparse.ArgumentParser(description="Run NSGA-II search with remote evaluation")
    parser.add_argument(
        "--hosts-file",
        type=str,
        default="../host_configs/hosts.yaml",
        help="Path to a YAML hosts file containing a top-level list of IPs",
    )
    parser.add_argument("--user", type=str, default="xinting", help="SSH username")
    parser.add_argument("--key", type=str, default="/home/xinting/.ssh/id_rsa", help="Path to SSH private key")
    parser.add_argument("--max_layers", type=int, default=10, help="Max number of layers (L_max)")
    parser.add_argument("--min_layers", type=int, default=1, help="Min number of layers (L_min)")
    parser.add_argument("--total_sample_size", type=int, default=8, help="Total number of offspring to sample per generation")
    parser.add_argument("--per_round_sample_size", type=int, default=8, help="Number of offspring to sample per round")
    parser.add_argument("--exp_name", type=str, default="3layer_random_minipile_iter10k", help="Experiment name for checkpoint directory")
    parser.add_argument("--conda_env", type=str, default="reallmforge", help="Conda environment name on remote hosts")
    parser.add_argument("--max_iters", type=int, default=10000, help="Max training iterations per evaluation")
    parser.add_argument(
        "--search_space_config",
        type=str,
        default="search_space_def/default_search_space.yaml",
        help="Path to YAML file defining 'global_spec' and 'layer_spec' (relative paths resolve from this script)",
    )
    args = parser.parse_args()

    # set random seed for reproducibility
    random.seed(45)

    hosts = load_hosts_from_file(args.hosts_file)
    logging.info(f"Loaded {len(hosts)} hosts from {args.hosts_file}")
    user = args.user
    key_filename = args.key

    total_sample_size = args.total_sample_size
    per_round_sample_size = args.per_round_sample_size
    n_rounds = total_sample_size // per_round_sample_size
    max_n_layer = args.max_layers
    min_n_layer = args.min_layers
    config_path = args.search_space_config
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)

    global_spec, layer_spec = load_search_space_from_yaml(config_path)
    search_space = HeteroSearchSpace.from_dicts(global_spec, layer_spec, L_max=max_n_layer, L_min=min_n_layer)
    
    print("Using search space:")
    print(search_space.print_search_space())

    exp_name = args.exp_name
    init_population_size = args.per_round_sample_size

    sw_only = True

    # update the working directory on remote hosts
    trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
    trainer.perform_git_pull(remote_work_dir=f"/home/{user}/Evo_GPT")

    entire_population = Population([], search_space=search_space)

    for i in range(0, n_rounds):
        individuals = [search_space.sample() for _ in range(init_population_size)]
        population = Population(individuals, search_space=search_space)
        population.n_population = init_population_size
        print(f"\n\n================ Round {i} ================\n")
        population.sw_eval(hosts=hosts, user=user, key_filename=key_filename, run_dir_name=exp_name, conda_env=args.conda_env, max_iters=args.max_iters, sw_only=sw_only)
        population.print_summary()
        entire_population.append_population(added_individuals=population.individuals, added_evaluations=population.evaluations)
    
    timestamp = int(time.time())
    entire_population.save_to_csv(f"csv/{exp_name}_random_sweep_{timestamp}.csv")
        

if __name__ == "__main__":
    main()




