# TCP Congestion Control RL Agent

Trains a PPO agent to control TCP congestion window via NS-3 simulation, then exports the policy as an ONNX model for deployment in a Kathará C environment.

---

## Setup

```bash
cd training
python -m venv ns3-venv
source ns3-venv/bin/activate
pip install -r requirements.txt
```

### Install NS-3 OpenGym contrib

```bash
cd training/ns-3.40/contrib
git clone https://github.com/tkn-tub/ns3-gym.git ./opengym
cd opengym
git checkout app-ns-3.36+
```

### Build NS-3

```bash
cd training/ns-3.40
./ns3 configure --enable-examples
./ns3 build
```

### Patch ns3gym (required)

After installing the venv, open `ns3-venv/lib/python3.12/site-packages/ns3gym/ns3env.py` and replace `np.float` with `float` on lines 114–121 (`np.float` was removed in NumPy 1.24):

```python
if mtype == pb.INT:
    mtype = int
elif mtype == pb.UINT:
    mtype = int
elif mtype == pb.DOUBLE:
    mtype = float
else:
    mtype = float
```

---

## Training

```bash
cd training
python train_agent.py [--model ppo]
```

Hyperparameters are read from `training/model_params.json`. Supported keys under `"ppo"`:

| Key | Default | Description |
|---|---|---|
| `k` | 10 | Time-series window length |
| `hidden_dim` | 64 | FC layer width |
| `lr` | 3e-4 | Adam learning rate |
| `num_episodes` | 500 | Training episodes |
| `gamma` | 0.99 | Discount factor |
| `port` | 5555 | NS-3 ZMQ port |
| `sim_seed` | 42 | Simulation seed |

`tcp_sim_list` controls which NS-3 scratch scripts are used as simulation environments.

After training, the model is exported to `training/outputs/{model}_{timestamp}/{model}.onnx`.

---

## Contributing

- New model architectures go in `training/tcp_train_toolkit/models/`
- New trainers go in `training/tcp_train_toolkit/` and must be registered in `train_agent.py`'s `MODEL_TRAINERS` dict
- NS-3 simulation scripts live in `training/ns-3.40/scratch/`
