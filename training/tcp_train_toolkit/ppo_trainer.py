import gymnasium as gym
import sys
sys.modules["gym"] = gym

import json
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ns3gym import ns3env
from . import simulation


def run(model, optimizer, ns3_dir: str, device, params: dict, sim_list: list = None) -> None:
    """Run the full PPO training loop.

    Args:
        model: PpoActorCritic1DCNN (or compatible) model, already on device.
        optimizer: Torch optimizer bound to model parameters.
        ns3_dir: Absolute path to the ns-3.40 directory.
        device: torch.device to run inference/updates on.
        params: Hyperparameter dict. Expected keys:
                  num_episodes, gamma, port, sim_seed
        sim_list: Optional list of simulation names to override simulation.TCP_SIM_LIST.
                  If None or empty, the toolkit default is used.
    """
    num_episodes = params["num_episodes"]
    gamma        = params["gamma"]
    port         = params["port"]
    sim_seed     = params["sim_seed"]

    print(f"Starting Training on {device}...")

    for episode in range(num_episodes):
        # -------------------------------------------------------
        # Set up simulation working directory (ns3-gym uses cwd
        # name as the script to run)
        # -------------------------------------------------------
        sim_name, sim_dir = simulation.enter_sim(ns3_dir, sim_list=sim_list)

        env = ns3env.Ns3Env(port=port, startSim=True, simSeed=sim_seed)

        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        done = False

        if episode == 0:
            s = np.array(state, dtype=float)
            nan_count = int(np.isnan(s).sum())
            inf_count = int(np.isinf(s).sum())
            if nan_count or inf_count:
                print(f"[DEBUG] Initial state has {nan_count} NaN and {inf_count} inf values — will be zeroed")

        log_probs, values, rewards = [], [], []
        episode_throughput, episode_rtt, episode_loss = [], [], []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            # Sanitize: NS-3 may send NaN/inf for uninitialized metrics at t=0
            state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=0.0, neginf=0.0)

            action_mean, action_std, value = model(state_tensor)
            dist = Normal(action_mean, action_std)

            action = dist.sample()
            action_clamped = torch.clamp(action, -1.0, 1.0)

            numpy_action = action_clamped.detach().cpu().numpy().flatten()
            step_results = env.step(numpy_action)

            # Gymnasium returns: obs, reward, terminated, truncated, info
            if len(step_results) == 5:
                next_state, reward, terminated, truncated, info = step_results
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_results

            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))

            # Parse metrics from C++ GetExtraInfo()
            if info and 'extra_info' in info and info['extra_info']:
                try:
                    metrics = json.loads(info['extra_info'])
                    episode_throughput.append(metrics['throughput_mbps'])
                    episode_rtt.append(metrics['avg_rtt_ms'])
                    episode_loss.append(metrics['packet_loss_rate'])
                except json.JSONDecodeError:
                    pass

            state = next_state

        # -------------------------------------------------------
        # Teardown: close env, restore cwd, remove temp dir
        # -------------------------------------------------------
        env.close()
        simulation.exit_sim(ns3_dir, sim_dir)

        # -------------------------------------------------------
        # PPO Update (end of episode)
        # -------------------------------------------------------
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.cat(returns).detach()

        log_probs_t = torch.cat(log_probs)
        values_t = torch.cat(values).squeeze()

        advantages = returns - values_t.detach()

        # Simplified Vanilla Policy Gradient / basic PPO loss.
        # For full PPO: loop over rollout multiple epochs with clipping.
        actor_loss = -(log_probs_t.squeeze() * advantages).mean()
        critic_loss = nn.MSELoss()(values_t, returns)
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------------------------------------------------------
        # Logging
        # -------------------------------------------------------
        avg_t = sum(episode_throughput) / len(episode_throughput) if episode_throughput else 0
        avg_r = sum(episode_rtt) / len(episode_rtt) if episode_rtt else 0
        avg_l = sum(episode_loss) / len(episode_loss) if episode_loss else 0
        total_reward = sum([r.item() for r in rewards])

        print(f"Ep {episode+1}/{num_episodes} | Sim: {sim_name} | Reward: {total_reward:.2f} | "
              f"T-put: {avg_t:.2f} Mbps | RTT: {avg_r:.2f} ms | Loss: {avg_l:.4f}")
