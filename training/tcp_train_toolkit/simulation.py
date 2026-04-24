import os
import random
import shutil

#"tcp-rl-env"
TCP_SIM_LIST = ["tcp-rl-env-variable-latency", "tcp-rl-env-simple-p2p", "tcp-rl-env-high-loss", "tcp-rl-env-bottleneck"]


def enter_sim(ns3_dir: str, sim_list: list = None) -> tuple:
    """Randomly pick a simulation, create its temp directory, and chdir into it.

    Args:
        ns3_dir: Absolute path to the ns-3.40 directory.
        sim_list: List of simulation names to choose from. Falls back to TCP_SIM_LIST.

    Returns:
        (sim_name, sim_dir) — the chosen simulation name and its full path.

    Raises:
        RuntimeError: If sim_dir already exists and is non-empty (leftover from a crash).
    """
    choices = sim_list if sim_list else TCP_SIM_LIST
    sim_name = random.choice(choices)
    sim_dir = os.path.join(ns3_dir, sim_name)

    if os.path.exists(sim_dir):
        if os.listdir(sim_dir):
            raise RuntimeError(
                f"Simulation directory '{sim_dir}' already exists and is not empty. "
                "Remove it manually before running."
            )
        # Exists but empty — safe to use as-is
    else:
        os.makedirs(sim_dir)

    os.chdir(sim_dir)
    return sim_name, sim_dir


def exit_sim(ns3_dir: str, sim_dir: str) -> None:
    """Return to ns3_dir and delete the temporary simulation directory.

    Args:
        ns3_dir: Absolute path to the ns-3.40 directory to chdir back to.
        sim_dir: Absolute path to the simulation directory to remove.
    """
    os.chdir(ns3_dir)
    shutil.rmtree(sim_dir)
