import gymnasium as gym
import sys
sys.modules["gym"] = gym

import copy
import json
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from ns3gym import ns3env

from . import simulation


def _extract_metrics(extra_info_obj):
    if extra_info_obj is None:
        return None

    if isinstance(extra_info_obj, dict):
        candidate = extra_info_obj.get("extra_info", extra_info_obj.get("extraInfo", extra_info_obj))
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, str) and candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
        return None

    if isinstance(extra_info_obj, str) and extra_info_obj:
        try:
            return json.loads(extra_info_obj)
        except json.JSONDecodeError:
            return None

    return None


def _sanitize_state(state, device):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    return torch.nan_to_num(state_tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _epsilon_by_step(step, start, end, decay_steps):
    if decay_steps <= 0:
        return end
    frac = min(1.0, step / float(decay_steps))
    return start + frac * (end - start)


def _discrete_to_continuous_action(action_idx, cwnd_mss):
    c = max(1.0, float(cwnd_mss))
    if action_idx == 0:   # cwnd * 2
        return 1.0
    if action_idx == 1:   # cwnd - MSS
        target = max(1.0, c - 1.0)
        return float(np.clip(math.log2(target / c), -1.0, 1.0))
    if action_idx == 2:   # cwnd
        return 0.0
    if action_idx == 3:   # cwnd + MSS
        target = c + 1.0
        return float(np.clip(math.log2(target / c), -1.0, 1.0))
    if action_idx == 4:   # cwnd / 2
        return -1.0
    if action_idx == 5:   # MSS (1 MSS)
        target = 1.0
        return float(np.clip(math.log2(target / c), -1.0, 1.0))
    raise ValueError(f"Invalid discrete action index: {action_idx}")


def _first_episode_reaching_target(returns, target_ratio=0.95):
    if not returns:
        return None
    n = len(returns)
    w = max(1, n // 10)
    final_reward = float(np.mean(returns[-w:]))
    target = target_ratio * final_reward
    for i in range(n):
        lo = max(0, i - w + 1)
        rolling = float(np.mean(returns[lo : i + 1]))
        if rolling >= target:
            return i + 1
    return None


def _evaluate_seed(model, ns3_dir, device, port, seed, sim_list=None, eval_episodes=1):
    eval_returns = []
    for _ in range(eval_episodes):
        sim_name, sim_dir = simulation.enter_sim(ns3_dir, sim_list=sim_list)
        env = ns3env.Ns3Env(port=port, startSim=True, simSeed=seed)
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        done = False
        total_reward = 0.0

        while not done:
            state_t = _sanitize_state(state, device).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_t)
                action_idx = int(torch.argmax(q_values, dim=1).item())
            cwnd_mss = float(np.nan_to_num(state[-1], nan=1.0, posinf=1.0, neginf=1.0))
            continuous_action = _discrete_to_continuous_action(action_idx, cwnd_mss)
            step_results = env.step(np.array([continuous_action], dtype=np.float32))
            if len(step_results) == 5:
                next_state, reward, terminated, truncated, _ = step_results
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_results
            total_reward += float(reward)
            state = next_state

        env.close()
        simulation.exit_sim(ns3_dir, sim_dir)
        eval_returns.append(total_reward)

    return float(np.mean(eval_returns)) if eval_returns else 0.0


def run(model, optimizer, ns3_dir: str, device, params: dict, sim_list: list = None) -> None:
    num_episodes = params["num_episodes"]
    gamma = params["gamma"]
    port = params["port"]
    sim_seed = params["sim_seed"]

    batch_size = params["batch_size"]
    replay_capacity = params["replay_capacity"]
    warmup_steps = params["warmup_steps"]
    train_freq = params["train_freq"]
    target_update_freq = params["target_update_freq"]
    epsilon_start = params["epsilon_start"]
    epsilon_end = params["epsilon_end"]
    epsilon_decay_steps = params["epsilon_decay_steps"]
    eval_seeds = params.get("eval_seeds", [])
    eval_episodes_per_seed = int(params.get("eval_episodes_per_seed", 1))

    target_model = copy.deepcopy(model).to(device)
    target_model.eval()
    replay = deque(maxlen=replay_capacity)
    loss_fn = nn.MSELoss()

    global_step = 0
    print(f"Starting DQN Training on {device}...")
    all_returns, all_avg_tput, all_avg_rtt, all_avg_loss = [], [], [], []

    for episode in range(num_episodes):
        sim_name, sim_dir = simulation.enter_sim(ns3_dir, sim_list=sim_list)
        env = ns3env.Ns3Env(port=port, startSim=True, simSeed=sim_seed)

        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        done = False

        episode_loss_vals = []
        episode_rewards = []
        episode_throughput, episode_rtt, episode_loss = [], [], []

        while not done:
            state_t = _sanitize_state(state, device).unsqueeze(0)

            epsilon = _epsilon_by_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)
            if random.random() < epsilon:
                action_idx = random.randint(0, 5)
            else:
                with torch.no_grad():
                    q_values = model(state_t)
                    action_idx = int(torch.argmax(q_values, dim=1).item())

            cwnd_mss = float(np.nan_to_num(state[-1], nan=1.0, posinf=1.0, neginf=1.0))
            continuous_action = _discrete_to_continuous_action(action_idx, cwnd_mss)
            step_results = env.step(np.array([continuous_action], dtype=np.float32))

            if len(step_results) == 5:
                next_state, reward, terminated, truncated, info = step_results
                done = terminated or truncated
                extra_info_obj = info
            else:
                next_state, reward, done, extra_info_obj = step_results

            replay.append((state, action_idx, float(reward), next_state, done))
            episode_rewards.append(float(reward))

            metrics = _extract_metrics(extra_info_obj)
            if metrics:
                episode_throughput.append(float(metrics.get("throughput_mbps", 0.0)))
                episode_rtt.append(float(metrics.get("avg_rtt_ms", 0.0)))
                episode_loss.append(float(metrics.get("packet_loss_rate", 0.0)))

            global_step += 1

            if len(replay) >= max(batch_size, warmup_steps) and (global_step % train_freq == 0):
                batch = random.sample(replay, batch_size)
                b_state, b_action, b_reward, b_next_state, b_done = zip(*batch)

                s = torch.stack([_sanitize_state(x, device) for x in b_state], dim=0)
                a = torch.tensor(b_action, dtype=torch.long, device=device).unsqueeze(1)
                r = torch.tensor(b_reward, dtype=torch.float32, device=device).unsqueeze(1)
                ns = torch.stack([_sanitize_state(x, device) for x in b_next_state], dim=0)
                d = torch.tensor(b_done, dtype=torch.float32, device=device).unsqueeze(1)

                q = model(s).gather(1, a)
                with torch.no_grad():
                    next_q_max = target_model(ns).max(dim=1, keepdim=True)[0]
                    target = r + (1.0 - d) * gamma * next_q_max

                loss = loss_fn(q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss_vals.append(float(loss.item()))

            if global_step % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

            state = next_state

        env.close()
        simulation.exit_sim(ns3_dir, sim_dir)

        avg_t = sum(episode_throughput) / len(episode_throughput) if episode_throughput else 0
        avg_r = sum(episode_rtt) / len(episode_rtt) if episode_rtt else 0
        avg_l = sum(episode_loss) / len(episode_loss) if episode_loss else 0
        total_reward = sum(episode_rewards)
        avg_train_loss = sum(episode_loss_vals) / len(episode_loss_vals) if episode_loss_vals else 0.0
        loss_pct = avg_l * 100.0
        all_returns.append(float(total_reward))
        all_avg_tput.append(float(avg_t))
        all_avg_rtt.append(float(avg_r))
        all_avg_loss.append(float(loss_pct))

        print(
            f"Ep {episode+1}/{num_episodes} | Sim: {sim_name} | Reward: {total_reward:.2f} | "
            f"T-put: {avg_t:.2f} Mbps | RTT: {avg_r:.2f} ms | Loss: {loss_pct:.4f}% | "
            f"DQN-Loss: {avg_train_loss:.6f} | Eps: {epsilon:.4f}"
        )

    conv_ep = _first_episode_reaching_target(all_returns, target_ratio=0.95)
    stability_var = None
    if eval_seeds:
        seed_returns = []
        for s in eval_seeds:
            r = _evaluate_seed(model, ns3_dir, device, port, int(s), sim_list, eval_episodes_per_seed)
            seed_returns.append(r)
        if len(seed_returns) >= 2:
            stability_var = float(np.var(seed_returns))
        else:
            stability_var = 0.0

    print("\n=== Evaluation Summary (DQN) ===")
    print(f"Average Throughput (Mbps): {float(np.mean(all_avg_tput)) if all_avg_tput else 0.0:.4f}")
    print(f"Average RTT / Latency (ms): {float(np.mean(all_avg_rtt)) if all_avg_rtt else 0.0:.4f}")
    print(f"Packet Loss Rate (%): {float(np.mean(all_avg_loss)) if all_avg_loss else 0.0:.6f}")
    print(f"Convergence Speed (episodes to 95% final reward): {conv_ep if conv_ep is not None else 'N/A'}")
    if stability_var is None:
        print("Training Stability (variance across seeds): N/A (set eval_seeds in model_params.json)")
    else:
        print(f"Training Stability (variance of episodic return across seeds): {stability_var:.6f}")
