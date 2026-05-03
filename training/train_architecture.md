• End-to-end flow

  1. train_agent.py loads hyperparameters (DEFAULTS + model_params.json), picks ppo|dqn|ddpg, builds the matching CNN model +
     optimizer, and calls that trainer’s run(...) with ns3_dir and optional tcp_sim_list override from JSON (training/
     train_agent.py).
  2. Each trainer repeatedly:
      1. Chooses a simulation name.
      2. Calls simulation.enter_sim(...), which creates/chdirs into training/ns-3.40/<sim_name> temporary working dir.
      3. Starts ns3env.Ns3Env(startSim=True, ...) so ns3-gym launches the matching scratch simulation.
      4. Runs episode loop: state -> action -> env.step(...) -> reward/logging/learning.
      5. Calls simulation.exit_sim(...) to chdir back and delete temp dir (training/tcp_train_toolkit/simulation.py).
  3. After training, Python exports ONNX + {model}.info.json (contains k) for deployment (training/tcp_train_toolkit/
     export.py).

  tcp_train_toolkit internals

  - Models:
      - PPO: actor-critic with 1D CNN over 3*k history + cwnd scalar, outputs mean/std/value (training/tcp_train_toolkit/
        models/ppo_cnn.py).
      - DQN: same encoder, outputs 6 Q-values (training/tcp_train_toolkit/models/dqn_cnn.py).
      - DDPG: separate actor/critic CNNs; actor outputs continuous action, critic outputs Q(s,a) (training/tcp_train_toolkit/
        models/ddpg_cnn.py).
  - Trainers:
      - PPO: on-policy episode returns/advantages, simple actor+critic loss (training/tcp_train_toolkit/ppo_trainer.py).
      - DQN: replay buffer + target net + epsilon-greedy; maps 6 discrete choices into continuous cwnd log-scale action
        expected by NS-3 (training/tcp_train_toolkit/dqn_trainer.py).
      - DDPG: replay + target actor/critic + soft updates + decaying Gaussian exploration noise (training/tcp_train_toolkit/
        ddpg_trainer.py).
  - __init__.py files are empty package markers:
      - training/tcp_train_toolkit/__init__.py
      - training/tcp_train_toolkit/models/__init__.py

  How scratch simulations work (NS-3 side)

  - All RL tcp-rl-env*.cc files implement an OpenGymEnv:
      - Observation: k RTT + k dupACK + k timeout + current cwnd (3*k+1, usually 31).
      - Action: one continuous scalar in [-1,1].
      - Action effect: cwnd <- cwnd * 2^a, clamped.
      - Reward: throughput term - RTT penalty - loss penalty.
      - Extra info JSON returns throughput/RTT/loss to Python for logging.
      - Every 5 ACKs (predictionInterval) it calls Notify() to trigger Python step.
  - Main files:
      - Base env with custom TcpRlCongestionOps class that directly enforces target cwnd in TCP internals: training/ns-3.40/
        scratch/tcp-rl-env.cc
      - Scenario variants used for domain randomization:
          - simple p2p 10Mbps/20ms: training/ns-3.40/scratch/tcp-rl-env-simple-p2p.cc
          - bottleneck 3-node topology with 1Mbps bottleneck link: training/ns-3.40/scratch/tcp-rl-env-bottleneck.cc
          - high loss 5% loss via RateErrorModel: training/ns-3.40/scratch/tcp-rl-env-high-loss.cc
          - variable latency with scheduled channel delay cycling: training/ns-3.40/scratch/tcp-rl-env-variable-latency.cc

  All other files in scratch

  - Build wiring:
      - Auto-build single and subdir scratch programs: training/ns-3.40/scratch/CMakeLists.txt
      - Nested subdir custom library/executable example: training/ns-3.40/scratch/nested-subdir/CMakeLists.txt
  - Example/demo programs not part of RL training loop:
      - training/ns-3.40/scratch/scratch-simulator.cc
      - training/ns-3.40/scratch/subdir/scratch-subdir.cc
      - training/ns-3.40/scratch/subdir/scratch-subdir-additional-header.h
      - training/ns-3.40/scratch/subdir/scratch-subdir-additional-header.cc
      - training/ns-3.40/scratch/nested-subdir/scratch-nested-subdir-executable.cc
      - training/ns-3.40/scratch/nested-subdir/lib/scratch-nested-subdir-library-header.h
      - training/ns-3.40/scratch/nested-subdir/lib/scratch-nested-subdir-library-source.cc