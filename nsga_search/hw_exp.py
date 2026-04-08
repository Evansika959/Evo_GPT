# hardware exploration by TimeLoop

from search_space import Individual
import yaml
import os
import time
import timeloopfe.v4 as tl
from utils.parse_timeloop_stats import parse_timeloop_stats, parse_dram_dataspace_stats
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Define relative paths
ARCH_PATH = f"{os.curdir}/hw_eval/arch/system_gemmini.yaml"
COMPONENTS_PATH = f"{os.curdir}/hw_eval/arch/components/*.yaml"
PROBLEM_PATH = f"{os.curdir}/hw_eval/prob/generic_GEMM.yaml"
MAPPER_PATH = f"{os.curdir}/hw_eval/mapper/mapper.yaml"
CONSTRAINTS_PATH = f"{os.curdir}/hw_eval/constraints/constraints.yaml"
VARIABLES_PATH = f"{os.curdir}/hw_eval/mapper/variables.yaml"

# DRAM read/write bandwidth from architecture spec (bytes/cycle)
DRAM_READ_BW = 4
DRAM_WRITE_BW = 4


def run_GEMM_evaluation(in_channel: int, out_channel: int, seq_length: int, work_dir: str, log_path: str = "/tmp/timeloop.log") -> dict:

    # create working directory if not exists
    os.makedirs(work_dir, exist_ok=True)
    # Prepare problem file with specific dimensions
    out_dir = os.path.join(work_dir, f"gemm_{in_channel}i_{out_channel}o_{seq_length}l")
    os.makedirs(out_dir, exist_ok=True)
    problem_file = os.path.join(out_dir, "generic_GEMM.yaml")
    with open(PROBLEM_PATH, 'r') as f:
        problem_data = f.read()
        problem_data = problem_data.replace("$IN_CHANNELS", str(in_channel))
        problem_data = problem_data.replace("$OUT_CHANNELS", str(out_channel))
        problem_data = problem_data.replace("$OUT_HEIGHT", str(seq_length))
    with open(problem_file, 'w') as f:
        f.write(problem_data)

    spec = tl.Specification.from_yaml_files(
        ARCH_PATH,
        COMPONENTS_PATH,
        MAPPER_PATH,
        problem_file,
        CONSTRAINTS_PATH,
        VARIABLES_PATH
    )

    spec.mapspace.template = 'uber' #'ruby'
    constrained_factors = ["D=1"]
    constrained_factors.append("E=1")
    tl.constraints.Factors(constrained_factors)
    if spec.constraints['targets'] is None:
        spec.constraints['targets'] = tl.constraints.ConstraintsList()

    output_file = os.path.join(out_dir, f"timeloop-mapper.stats.txt")
    if not os.path.exists(output_file):
        # Run the Timeloop mapper
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write("")  # create the log file
        tl.call_mapper(spec, output_dir=out_dir, log_to=log_path)

    return parse_timeloop_stats(output_file)


def run_GEMM_evaluation_detailed(in_channel: int, out_channel: int, seq_length: int,
                                  work_dir: str, log_path: str = "/tmp/timeloop.log") -> Tuple[dict, dict]:
    """Run GEMM evaluation and return both summary stats and per-dataspace DRAM stats.

    Returns:
        (summary_stats, dram_stats) where dram_stats is keyed by dataspace name
        ('Weights', 'Inputs', 'Outputs') with per-dataspace access counts and energy.
    """
    os.makedirs(work_dir, exist_ok=True)
    out_dir = os.path.join(work_dir, f"gemm_{in_channel}i_{out_channel}o_{seq_length}l")
    os.makedirs(out_dir, exist_ok=True)
    problem_file = os.path.join(out_dir, "generic_GEMM.yaml")
    with open(PROBLEM_PATH, 'r') as f:
        problem_data = f.read()
        problem_data = problem_data.replace("$IN_CHANNELS", str(in_channel))
        problem_data = problem_data.replace("$OUT_CHANNELS", str(out_channel))
        problem_data = problem_data.replace("$OUT_HEIGHT", str(seq_length))
    with open(problem_file, 'w') as f:
        f.write(problem_data)

    spec = tl.Specification.from_yaml_files(
        ARCH_PATH, COMPONENTS_PATH, MAPPER_PATH,
        problem_file, CONSTRAINTS_PATH, VARIABLES_PATH
    )
    spec.mapspace.template = 'uber'
    constrained_factors = ["D=1", "E=1"]
    tl.constraints.Factors(constrained_factors)
    if spec.constraints['targets'] is None:
        spec.constraints['targets'] = tl.constraints.ConstraintsList()

    output_file = os.path.join(out_dir, "timeloop-mapper.stats.txt")
    if not os.path.exists(output_file):
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write("")
        tl.call_mapper(spec, output_dir=out_dir, log_to=log_path)

    summary = parse_timeloop_stats(output_file)
    dram = parse_dram_dataspace_stats(output_file)
    return summary, dram


# ---------------------------------------------------------------------------
# Fusion savings calculation
# ---------------------------------------------------------------------------

def _dram_output_energy(dram_stats: dict) -> float:
    """Total DRAM energy (pJ) for the Outputs dataspace of a GEMM.

    This is the energy for writing partial sums + reading back for reduction.
    In a fused chain, the producer's output stays on-chip, so this is saved.
    """
    out = dram_stats.get('Outputs', {})
    return out.get('energy_pJ', 0) or 0


def _dram_input_energy(dram_stats: dict) -> float:
    """Total DRAM energy (pJ) for the Inputs dataspace of a GEMM.

    In a fused chain, the consumer reads its inputs from on-chip instead of DRAM.
    """
    inp = dram_stats.get('Inputs', {})
    return inp.get('energy_pJ', 0) or 0


def _dram_output_accesses(dram_stats: dict) -> float:
    """Total DRAM scalar accesses for Outputs (reads + updates)."""
    out = dram_stats.get('Outputs', {})
    reads = out.get('scalar_reads', 0) or 0
    updates = out.get('scalar_updates', 0) or 0
    return reads + updates


def _dram_input_accesses(dram_stats: dict) -> float:
    """Total DRAM scalar reads for Inputs."""
    inp = dram_stats.get('Inputs', {})
    return inp.get('scalar_reads', 0) or 0


def _estimate_saved_cycles(producer_dram: dict, consumer_dram: dict) -> float:
    """Estimate cycle savings from avoiding DRAM round-trip for intermediate data.

    The saved cycles come from not needing DRAM bandwidth for:
    - Producer writing outputs to DRAM
    - Consumer reading inputs from DRAM

    Uses the DRAM bandwidth from architecture spec (4 bytes/cycle read, 4 bytes/cycle write).
    The datawidth is 8 bits = 1 byte per scalar access.
    """
    out = producer_dram.get('Outputs', {})
    inp = consumer_dram.get('Inputs', {})

    # Scalar accesses that would be saved
    out_writes = (out.get('scalar_updates', 0) or 0)
    out_reads = (out.get('scalar_reads', 0) or 0)
    inp_reads = (inp.get('scalar_reads', 0) or 0)

    # Each scalar is 1 byte (datawidth=8 bits). DRAM bandwidth is in bytes/cycle.
    # But cycles are determined by max(compute_bound, memory_bound).
    # The saved memory traffic reduces the memory-bound component.
    # Conservative estimate: saved_cycles = saved_bytes / bandwidth
    saved_write_cycles = out_writes / DRAM_WRITE_BW
    saved_read_cycles = (out_reads + inp_reads) / DRAM_READ_BW

    # The producer and consumer run sequentially, so savings are additive
    # but bounded: can't save more cycles than the DRAM-bound portion
    return saved_write_cycles + saved_read_cycles


def compute_fusion_savings(
    op_stats: List[Tuple[dict, dict]],
    fusion_edges: List[Tuple[int, int]],
    scale_factors: Optional[Dict[int, float]] = None,
) -> Tuple[float, float]:
    """Compute energy and cycle savings from fusing consecutive operations.

    Args:
        op_stats: List of (summary_stats, dram_stats) per operation.
        fusion_edges: List of (producer_idx, consumer_idx) pairs defining
            which operations share intermediate data on-chip.
        scale_factors: Optional dict mapping op index to a scaling factor
            (e.g., for n_kv_groups scaling on QK_attn/PV_attn).

    Returns:
        (saved_energy_uJ, saved_cycles): Total savings from fusion.
    """
    if scale_factors is None:
        scale_factors = {}

    total_saved_energy_pJ = 0.0
    total_saved_cycles = 0.0

    for prod_idx, cons_idx in fusion_edges:
        _, prod_dram = op_stats[prod_idx]
        _, cons_dram = op_stats[cons_idx]

        # Energy savings: producer doesn't write outputs to DRAM,
        # consumer doesn't read inputs from DRAM
        saved_energy = _dram_output_energy(prod_dram) + _dram_input_energy(cons_dram)

        # Cycle savings from avoided DRAM traffic
        saved_cycles = _estimate_saved_cycles(prod_dram, cons_dram)

        # Apply scaling factors (e.g., n_kv_groups for attention ops)
        prod_scale = scale_factors.get(prod_idx, 1.0)
        cons_scale = scale_factors.get(cons_idx, 1.0)
        # Use the max scale since both endpoints contribute
        edge_scale = max(prod_scale, cons_scale)

        total_saved_energy_pJ += saved_energy * edge_scale
        total_saved_cycles += saved_cycles * edge_scale

    total_saved_energy_uJ = total_saved_energy_pJ / 1e6
    return total_saved_energy_uJ, total_saved_cycles


# ---------------------------------------------------------------------------
# Layer evaluation with fusion
# ---------------------------------------------------------------------------

def evaluate_layer(layer: dict, n_embd: int, seq_length: int, work_dir: str,
                   fused: bool = True) -> dict:
    """Evaluate a single layer's hardware metrics.

    When fused=True (default), computes fusion savings for the producer->consumer
    data flow graph within the layer and subtracts saved DRAM energy/cycles.
    """
    try:
        n_head = layer['n_head']
        n_kv_groups = layer['n_kv_group']
        n_qk_head_dim = layer['n_qk_head_dim']
        n_v_head_dim = layer['n_v_head_dim']
        n_cproj = layer['n_cproj']
        attn_variant = layer['attention_variant']
        mlp_size = layer['mlp_size']
    except KeyError as e:
        raise KeyError(f"Missing key in layer definition: {e}")

    if attn_variant == 'infinite':
        # Run all 7 GEMMs with detailed DRAM stats
        # Op 0: QK_gen  [embd -> qk*(h+kv), seq]
        qk_gen = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=n_qk_head_dim * (n_head + n_kv_groups),
            seq_length=seq_length, work_dir=work_dir)
        # Op 1: V_gen   [embd -> v*kv, seq]
        v_gen = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=n_v_head_dim * n_kv_groups,
            seq_length=seq_length, work_dir=work_dir)
        # Op 2: QK_attn [qk -> seq, h//kv]  (scaled by n_kv_groups)
        qk_attn = run_GEMM_evaluation_detailed(
            in_channel=n_qk_head_dim, out_channel=seq_length,
            seq_length=n_head // n_kv_groups, work_dir=work_dir)
        # Op 3: PV_attn [seq -> v, h//kv]    (scaled by n_kv_groups)
        pv_attn = run_GEMM_evaluation_detailed(
            in_channel=seq_length, out_channel=n_v_head_dim,
            seq_length=n_head // n_kv_groups, work_dir=work_dir)
        # Op 4: ATTN_proj [v*h -> embd, seq]
        attn_proj = run_GEMM_evaluation_detailed(
            in_channel=n_v_head_dim * n_head, out_channel=n_embd,
            seq_length=seq_length, work_dir=work_dir)
        # Op 5: MLP_FC1 [embd -> mlp, seq]
        mlp_fc1 = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=mlp_size,
            seq_length=seq_length, work_dir=work_dir)
        # Op 6: MLP_FC2 [mlp -> embd, seq]
        mlp_fc2 = run_GEMM_evaluation_detailed(
            in_channel=mlp_size, out_channel=n_embd,
            seq_length=seq_length, work_dir=work_dir)

        all_ops = [qk_gen, v_gen, qk_attn, pv_attn, attn_proj, mlp_fc1, mlp_fc2]

        # Apply n_kv_groups scaling to QK_attn (idx 2) and PV_attn (idx 3)
        for idx in [2, 3]:
            summary = all_ops[idx][0]
            for key in ['cycles', 'energy_uJ', 'total_ops', 'total_memory_accesses']:
                if summary[key] is not None:
                    summary[key] *= n_kv_groups

        # Extract summary stats for aggregation
        all_summaries = [op[0] for op in all_ops]

        if fused:
            # Define the producer->consumer fusion edges:
            #
            # Data flow graph:
            #   hidden -> QK_gen(0) -> Q,K --> QK_attn(2) -> scores --> PV_attn(3)
            #   hidden -> V_gen(1)  -> V   --------------------------> PV_attn(3)
            #   PV_attn(3) -> attended --> ATTN_proj(4)
            #   ATTN_proj(4) -> hidden' --> MLP_FC1(5)
            #   MLP_FC1(5) -> expanded --> MLP_FC2(6)
            #
            # Fusible edges (producer output = consumer input, stays on-chip):
            fusion_edges = [
                (0, 2),  # QK_gen outputs -> QK_attn inputs (Q,K projections)
                (1, 3),  # V_gen outputs -> PV_attn inputs (V values)
                (2, 3),  # QK_attn outputs -> PV_attn inputs (attention scores)
                (3, 4),  # PV_attn outputs -> ATTN_proj inputs (attended values)
                (4, 5),  # ATTN_proj outputs -> MLP_FC1 inputs (hidden states)
                (5, 6),  # MLP_FC1 outputs -> MLP_FC2 inputs (expanded activations)
            ]

            # Scale factors for ops that were scaled by n_kv_groups
            scale_factors = {2: n_kv_groups, 3: n_kv_groups}

            saved_energy_uJ, saved_cycles = compute_fusion_savings(
                all_ops, fusion_edges, scale_factors)

            layer_stats = aggregate_stats(all_summaries)

            # Subtract fusion savings
            if layer_stats['energy_uJ'] is not None:
                layer_stats['energy_uJ'] = max(0, layer_stats['energy_uJ'] - saved_energy_uJ)
            if layer_stats['cycles'] is not None:
                layer_stats['cycles'] = max(0, layer_stats['cycles'] - saved_cycles)
            # Store savings for debugging
            layer_stats['fusion_saved_energy_uJ'] = saved_energy_uJ
            layer_stats['fusion_saved_cycles'] = saved_cycles
        else:
            layer_stats = aggregate_stats(all_summaries)
            layer_stats['fusion_saved_energy_uJ'] = 0.0
            layer_stats['fusion_saved_cycles'] = 0.0

    else:
        # Identity or causal: only MLP
        mlp_fc1 = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=mlp_size,
            seq_length=seq_length, work_dir=work_dir)
        mlp_fc2 = run_GEMM_evaluation_detailed(
            in_channel=mlp_size, out_channel=n_embd,
            seq_length=seq_length, work_dir=work_dir)

        all_summaries = [mlp_fc1[0], mlp_fc2[0]]

        if fused:
            fusion_edges = [(0, 1)]  # MLP_FC1 -> MLP_FC2
            saved_energy_uJ, saved_cycles = compute_fusion_savings(
                [mlp_fc1, mlp_fc2], fusion_edges)
            layer_stats = aggregate_stats(all_summaries)
            if layer_stats['energy_uJ'] is not None:
                layer_stats['energy_uJ'] = max(0, layer_stats['energy_uJ'] - saved_energy_uJ)
            if layer_stats['cycles'] is not None:
                layer_stats['cycles'] = max(0, layer_stats['cycles'] - saved_cycles)
            layer_stats['fusion_saved_energy_uJ'] = saved_energy_uJ
            layer_stats['fusion_saved_cycles'] = saved_cycles
        else:
            layer_stats = aggregate_stats(all_summaries)
            layer_stats['fusion_saved_energy_uJ'] = 0.0
            layer_stats['fusion_saved_cycles'] = 0.0

    return layer_stats


def eval_individual(individual: Individual, work_dir: str, fused: bool = True) -> dict:
    global_spec = individual["globals"]
    layer_spec = individual["layers"]
    n_embd = global_spec["n_embd"]
    seq_length = global_spec["block_size"]
    layer_mask = global_spec.get("layer_mask", None)
    if layer_mask is None:
        raise ValueError("layer_mask is not defined in global_spec")

    hw_eval_list = []
    for i, layer in enumerate(layer_spec):
        if layer_mask[i] == 1:
            layer_stats = evaluate_layer(layer, n_embd, seq_length, work_dir, fused=fused)
            hw_eval_list.append(layer_stats)

    aggregated_stats = aggregate_stats(hw_eval_list)

    # average over sequence length
    aggregated_stats['cycles_per_token'] = aggregated_stats['cycles'] / seq_length if aggregated_stats['cycles'] is not None else None
    aggregated_stats['token_delay'] = aggregated_stats['cycles_per_token'] / 1e9  # assuming 1GHz clock
    aggregated_stats['energy_per_token_uJ'] = aggregated_stats['energy_uJ'] / seq_length if aggregated_stats['energy_uJ'] is not None else None
    aggregated_stats['edp_per_token'] = aggregated_stats['edp'] / seq_length if aggregated_stats['edp'] is not None else None
    return aggregated_stats


def evaluate_population(population: list, base_work_dir: str, fused: bool = True) -> list:
    results = []
    for i, individual in enumerate(population):
        print(f"Evaluating Individual {i}...")
        individual_stats = eval_individual(individual, work_dir=base_work_dir, fused=fused)
        results.append(individual_stats)

    return results


def aggregate_stats(stats_list: list) -> dict:
    aggregated_stats = {}
    for key in stats_list[0].keys():
        aggregated_stats[key] = sum(
            stats[key] for stats in stats_list if stats.get(key) is not None
        )

    # recalculate derived metrics
    if aggregated_stats['total_ops'] is not None and aggregated_stats['total_memory_accesses'] is not None and aggregated_stats['total_memory_accesses'] != 0:
        aggregated_stats['algorithmic_intensity_ops_per_access'] = aggregated_stats['total_ops'] / aggregated_stats['total_memory_accesses']
    else:
        aggregated_stats['algorithmic_intensity_ops_per_access'] = None
    aggregated_stats['algorithmic_intensity_ops_per_byte'] = aggregated_stats['algorithmic_intensity_ops_per_access']
    aggregated_stats['edp'] = aggregated_stats['energy_uJ'] * aggregated_stats['cycles'] / 10e6 if aggregated_stats['energy_uJ'] is not None and aggregated_stats['cycles'] is not None else None  # J*ns

    total_cycle = aggregated_stats['cycles']
    aggregated_stats['utilization_pct'] = 0
    aggregated_stats['gflops'] = 0
    for stats in stats_list:
        aggregated_stats['utilization_pct'] += (stats['utilization_pct'] * stats['cycles'] / total_cycle) if stats['utilization_pct'] is not None and stats['cycles'] is not None and total_cycle != 0 else 0
        aggregated_stats['gflops'] += (stats['gflops'] * stats['cycles'] / total_cycle) if stats['gflops'] is not None and stats['cycles'] is not None and total_cycle != 0 else 0

    return aggregated_stats
