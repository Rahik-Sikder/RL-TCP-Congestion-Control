import gymnasium as gym
import sys
sys.modules["gym"] = gym

import copy
import json
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


def _soft_update(target_net, source_net, tau):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)


def _noise_std_by_step(step, start, minimum, decay_steps):
    if decay_steps <= 0:
        return minimum
    frac = min(1.0, step / float(decay_steps))
    return start + frac * (minimum - start)


def _evaluate_seed(model, ns3_dir, device, port, seed, sim_list=None, eval_episodes=1):
    eval_returns = []
    for _ in range(eval_episodes):
        _, sim_dir = simulation.enter_sim(ns3_dir, sim_list=sim_list)
        env = ns3env.Ns3Env(port=port, startSim=True, simSeed=seed)
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        done = False
        total_reward = 0.0

        while not done:
            state_t = _sanitize_state(state, device).unsqueeze(0)
            with torch.no_grad():
                action = model.actor(state_t)
                action = torch.clamp(action, -1.0, 1.0)

            step_results = env.step(action.detach().cpu().numpy().flatten())
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

    batch_size = int(params["batch_size"])
    replay_capacity = int(params["replay_capacity"])
    warmup_steps = int(params["warmup_steps"])
    train_freq = int(params["train_freq"])
    tau = float(params["tau"])
    exploration_noise = float(params["exploration_noise"])
    exploration_noise_min = float(params["exploration_noise_min"])
    exploration_decay_steps = int(params["exploration_decay_steps"])

    eval_seeds = params.get("eval_seeds", [])
    eval_episodes_per_seed = int(params.get("eval_episodes_per_seed", 1))

    if isinstance(optimizer, dict):
        actor_optimizer = optimizer.get("actor")
        critic_optimizer = optimizer.get("critic")
    elif isinstance(optimizer, (tuple, list)) and len(optimizer) == 2:
        actor_optimizer, critic_optimizer = optimizer
    else:
        # Fallback for external callers that pass a single optimizer object.
        actor_optimizer = torch.optim.Adam(model.actor_parameters(), lr=float(params.get("lr_actor", params.get("lr", 1e-4))))
        critic_optimizer = torch.optim.Adam(model.critic_parameters(), lr=float(params.get("lr_critic", params.get("lr", 1e-3))))

    if actor_optimizer is None or critic_optimizer is None:
        raise ValueError("DDPG requires both actor and critic optimizers.")

    target_model = copy.deepcopy(model).to(device)
    target_model.eval()

    replay = deque(maxlen=replay_capacity)
    critic_loss_fn = nn.MSELoss()

    episodes_per_sim = 1
    global_step = 0
    print(f"Starting DDPG Training on {device}...")
    all_returns, all_avg_tput, all_avg_rtt, all_avg_loss = [], [], [], []
    current_sim_name = None

    for episode in range(num_episodes):
        if episode % episodes_per_sim == 0 or current_sim_name is None:
            choices = sim_list if sim_list else simulation.TCP_SIM_LIST
            current_sim_name = random.choice(choices)

        sim_name, sim_dir = simulation.enter_sim(ns3_dir, sim_list=[current_sim_name])
        env = ns3env.Ns3Env(port=port, startSim=True, simSeed=sim_seed)

        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        done = False

        episode_critic_losses = []
        episode_actor_losses = []
        episode_rewards = []
        episode_throughput, episode_rtt, episode_loss = [], [], []
        noise_std = exploration_noise

        while not done:
            state_t = _sanitize_state(state, device).unsqueeze(0)

            with torch.no_grad():
                action = model.actor(state_t)
                noise_std = _noise_std_by_step(
                    global_step,
                    exploration_noise,
                    exploration_noise_min,
                    exploration_decay_steps,
                )
                noise = torch.normal(mean=0.0, std=noise_std, size=action.shape, device=device)
                action = torch.clamp(action + noise, -1.0, 1.0)

            numpy_action = action.detach().cpu().numpy().flatten()
            step_results = env.step(numpy_action)

            if len(step_results) == 5:
                next_state, reward, terminated, truncated, info = step_results
                done = terminated or truncated
                extra_info_obj = info
            else:
                next_state, reward, done, extra_info_obj = step_results

            replay.append((state, float(numpy_action[0]), float(reward), next_state, done))
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
                a = torch.tensor(b_action, dtype=torch.float32, device=device).unsqueeze(1)
                r = torch.tensor(b_reward, dtype=torch.float32, device=device).unsqueeze(1)
                ns = torch.stack([_sanitize_state(x, device) for x in b_next_state], dim=0)
                d = torch.tensor(b_done, dtype=torch.float32, device=device).unsqueeze(1)

                with torch.no_grad():
                    next_action = target_model.actor(ns)
                    target_q = target_model.critic(ns, next_action)
                    bellman_target = r + (1.0 - d) * gamma * target_q

                current_q = model.critic(s, a)
                critic_loss = critic_loss_fn(current_q, bellman_target)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                episode_critic_losses.append(float(critic_loss.item()))

                actor_action = model.actor(s)
                actor_loss = -model.critic(s, actor_action).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                episode_actor_losses.append(float(actor_loss.item()))

                _soft_update(target_model, model, tau)

            state = next_state

        env.close()
        simulation.exit_sim(ns3_dir, sim_dir)

        avg_t = sum(episode_throughput) / len(episode_throughput) if episode_throughput else 0
        avg_r = sum(episode_rtt) / len(episode_rtt) if episode_rtt else 0
        avg_l = sum(episode_loss) / len(episode_loss) if episode_loss else 0
        total_reward = sum(episode_rewards)
        avg_critic_loss = sum(episode_critic_losses) / len(episode_critic_losses) if episode_critic_losses else 0.0
        avg_actor_loss = sum(episode_actor_losses) / len(episode_actor_losses) if episode_actor_losses else 0.0
        loss_pct = avg_l * 100.0

        all_returns.append(float(total_reward))
        all_avg_tput.append(float(avg_t))
        all_avg_rtt.append(float(avg_r))
        all_avg_loss.append(float(loss_pct))

        print(
            f"Ep {episode+1}/{num_episodes} | Sim: {sim_name} | Reward: {total_reward:.2f} | "
            f"T-put: {avg_t:.2f} Mbps | RTT: {avg_r:.2f} ms | Loss: {loss_pct:.4f}% | "
            f"Critic-Loss: {avg_critic_loss:.6f} | Actor-Loss: {avg_actor_loss:.6f} | "
            f"Noise-Std: {noise_std:.4f}"
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

    print("\n=== Evaluation Summary (DDPG) ===")
    print(f"Average Throughput (Mbps): {float(np.mean(all_avg_tput)) if all_avg_tput else 0.0:.4f}")
    print(f"Average RTT / Latency (ms): {float(np.mean(all_avg_rtt)) if all_avg_rtt else 0.0:.4f}")
    print(f"Packet Loss Rate (%): {float(np.mean(all_avg_loss)) if all_avg_loss else 0.0:.6f}")
    print(f"Convergence Speed (episodes to 95% final reward): {conv_ep if conv_ep is not None else 'N/A'}")
    if stability_var is None:
        print("Training Stability (variance across seeds): N/A (set eval_seeds in model_params.json)")
    else:
        print(f"Training Stability (variance of episodic return across seeds): {stability_var:.6f}")
