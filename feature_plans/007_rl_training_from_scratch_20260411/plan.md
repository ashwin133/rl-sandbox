# Feature 007: RL Training from Scratch — REINFORCE & DQN

**Branch:** TBD
**Date:** 2026-04-11

---

## Goal

Implement two foundational RL training algorithms from scratch using PyTorch: **REINFORCE** (policy gradient) and **DQN** (Deep Q-Network). Both should integrate with the existing `BaseActionPolicy` abstraction so trained policies can be used interchangeably with the existing `RandomActionPolicy` for simulation and replay.

---

## Constraints

- PyTorch only (already a dependency) — no external RL libraries
- Must integrate with existing `BaseActionPolicy` interface (`choose_action(action_space, obs_space) -> int`)
- CartPole-v1 specific initially, but architecture should not hardcode this
- Training should be decoupled from rendering — train headless, then use trained policy with existing viewer/replay infrastructure
- Each algorithm should be trainable via CLI: `python -m app train --algorithm reinforce` / `--algorithm dqn`
- Trained models should be saveable/loadable (PyTorch state_dict)

---

## Background: Reinforcement Learning Fundamentals

Before diving into the specific algorithms, it's worth grounding the terminology and math they both build on.

### The RL Framework

Reinforcement learning formalises an **agent** interacting with an **environment** over discrete time steps. At each step `t`:

1. The agent observes the current **state** `s_t` (for CartPole: `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`, a 4-element vector)
2. The agent picks an **action** `a_t` from the action space (CartPole: `{0 = push left, 1 = push right}`)
3. The environment transitions to a new state `s_{t+1}` and emits a **reward** `r_{t+1}` (CartPole: +1 for every step the pole stays upright)
4. The loop repeats until the episode ends (pole falls, cart leaves bounds, or 500-step limit)

This is modelled as a **Markov Decision Process (MDP)**, defined by the tuple `(S, A, P, R, γ)`:
- `S` = set of possible states
- `A` = set of possible actions
- `P(s' | s, a)` = transition probability (stochastic in general, deterministic in CartPole)
- `R(s, a)` = reward function
- `γ` = discount factor ∈ [0, 1)

The "Markov" property means: **the next state depends only on the current state and action, not on history**. This lets us reason about each step in isolation.

### The Objective: Expected Return

The agent's goal is to maximise the **expected return** — the total discounted reward it collects over an episode:

```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ... + γ^(T-t-1)·r_T
```

The discount factor `γ` (typically 0.99) does two things:
1. **Mathematical convenience**: Keeps returns finite for infinite-horizon problems (geometric series converges)
2. **Shapes behaviour**: Makes the agent prefer rewards *now* over rewards later. A `γ = 0.99` means a reward 100 steps away is worth `0.99^100 ≈ 0.37` of its face value

### Policy vs Value Function — Two Ways to Solve the Same Problem

There are two fundamental objects you can learn:

**Policy `π(a | s)`** — a function mapping states to a probability distribution over actions. Directly answers "what should I do in this state?"
- REINFORCE learns this directly. The output is π, no intermediate step.

**Value functions** — answer "how good is a state (or state-action pair)?"
- **State value**: `V^π(s) = E[G_t | s_t = s]` — expected return starting from state s and following policy π
- **Action value (Q-function)**: `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` — expected return from taking action a in state s, then following π
- DQN learns `Q*(s, a)` — the Q-function of the optimal policy. Once you have `Q*`, the optimal policy is simply `π*(s) = argmax_a Q*(s, a)`.

These are two sides of the same coin. Both ultimately yield a policy, but the learning process differs significantly.

### The Bellman Equation — The Key Identity Both Algorithms Exploit

This is the most important equation in RL. It says the value of a state equals the immediate reward plus the discounted value of the next state:

```
Q*(s, a) = E[r + γ · max_a' Q*(s', a')]
```

In words: "the best I can get from (s, a) is the reward I receive now, plus the best I can get from wherever I land next."

This **recursive structure** is what makes RL tractable — instead of trying to estimate an infinite sum of future rewards, we just need to estimate one step ahead. DQN's entire training loop is about making the Q-network satisfy this equation.

### Exploration vs Exploitation

An agent faces a fundamental tension:
- **Exploit**: Take the action you currently think is best
- **Explore**: Try different actions to find out if something better exists

If the agent only exploits, it gets stuck with its initial (probably wrong) beliefs. If it only explores, it never actually uses what it learned.

Both algorithms handle this differently:
- **REINFORCE**: Exploration is *baked in* because it samples actions from the policy's probability distribution. Early in training the policy is near-uniform, so behaviour is exploratory; as training progresses and the policy sharpens, exploration naturally decreases.
- **DQN**: Uses **ε-greedy** — with probability ε, pick a random action; otherwise pick `argmax_a Q(s, a)`. ε starts high (e.g., 1.0) and decays to a small value (e.g., 0.01) over training.

---

## REINFORCE in Depth

### The Policy Gradient Theorem

REINFORCE is derived from the **policy gradient theorem**, which tells us how to improve a parameterised policy `π_θ(a | s)` (θ = neural network weights) using gradient ascent on expected return.

The key result: the gradient of expected return with respect to θ is

```
∇_θ J(θ) = E[ ∇_θ log π_θ(a_t | s_t) · G_t ]
```

**What this means intuitively**:
- `∇_θ log π_θ(a_t | s_t)` is the direction in parameter space that *increases* the probability of action `a_t` at state `s_t`
- `G_t` is the return — how well things went after taking that action
- Multiplying them says: **move in the direction that makes good actions more likely** (positive G), and **in the opposite direction for bad actions** (negative G, or simply less-than-average returns)

This gives us a learning rule: sample an episode, compute returns, and take a gradient step. No value function, no model of the environment — just trial and error with gradient reinforcement.

### Why It's Called "Monte Carlo"

"Monte Carlo" in RL means: estimate expected values by *sampling full episodes*. REINFORCE doesn't bootstrap (it doesn't use estimates of future value during learning) — it waits until an episode ends and uses the actual observed returns.

This is the tradeoff:
- **Unbiased**: The return `G_t` is an unbiased estimate of `Q^π(s_t, a_t)` — on average it equals the true value
- **High variance**: A single episode's return is noisy. Sometimes the pole falls early due to bad luck even with good actions.

### The Update Step in Detail

For each episode with steps `(s_0, a_0, r_1), (s_1, a_1, r_2), ..., (s_{T-1}, a_{T-1}, r_T)`:

1. **Compute returns backwards** (more efficient than forwards):
   ```
   G_{T-1} = r_T
   G_{T-2} = r_{T-1} + γ·G_{T-1}
   ...
   G_t = r_{t+1} + γ·G_{t+1}
   ```

2. **Normalise returns** (optional but stabilising): subtract the mean and divide by std of the returns across the episode. This is a **variance reduction technique** — it's equivalent to subtracting a constant baseline, which doesn't change the expected gradient but reduces noise.

3. **Compute loss**: we want to *maximise* expected return, but PyTorch optimisers minimise, so we negate:
   ```
   loss = -Σ_t log π_θ(a_t | s_t) · G_t
   ```

4. **Backprop and step**: standard PyTorch `loss.backward()`, `optimizer.step()`.

### Why Normalise Returns?

Imagine an episode where all returns are `[100, 101, 102, 103]`. Without normalisation, every action gets a large positive update — we make *all* of them more likely, which doesn't actually distinguish good from bad. After normalisation, the returns become something like `[-1.3, -0.4, 0.4, 1.3]`, and now the algorithm correctly reinforces the *relatively* better actions.

This is a form of **baseline subtraction**. The policy gradient theorem says we can subtract any state-dependent (or constant) baseline `b` without biasing the gradient:

```
∇_θ J(θ) = E[ ∇_θ log π_θ(a_t | s_t) · (G_t - b) ]
```

Normalising returns uses the mean as a rough baseline. More advanced methods (like Actor-Critic) learn a value function `V(s)` to use as the baseline — but we're keeping this simple.

### REINFORCE Characteristics

- **On-policy**: We must collect new data with the *current* policy after every update — old episodes are trained with stale weights, so their log-probs no longer reflect what the current network would do
- **Episodic**: Learning happens *after* an episode ends (we need the full return)
- **Simple**: No replay buffer, no target network, no value function (in basic form)
- **High variance**: Training curves are noisy. CartPole REINFORCE typically solves in 500-2000 episodes but with wide run-to-run variation

---

## DQN in Depth

### Value-Based RL: Learning Q*

DQN takes a different approach: instead of learning the policy directly, it learns `Q*(s, a)` — the optimal action-value function. The optimal policy then falls out "for free": `π*(s) = argmax_a Q*(s, a)`.

Why go through Q? Because of the Bellman equation:

```
Q*(s, a) = E[r + γ · max_a' Q*(s', a')]
```

This gives us a concrete learning target. If we have a current estimate `Q_θ(s, a)` and we observe a transition `(s, a, r, s')`, we can compute the **Bellman target**:

```
y = r + γ · max_a' Q_θ(s', a')
```

And update `Q_θ(s, a)` toward this target:

```
loss = (Q_θ(s, a) - y)²
```

This is **temporal difference (TD) learning**: we bootstrap our own future predictions into our current update, rather than waiting for full returns.

### Why Does Raw Q-Learning + Neural Nets Not Work?

The naive version — "run the environment, update Q(s,a) toward r + γ·max Q(s',a') at every step" — **fails spectacularly** with neural networks for two reasons:

**Problem 1: Correlated samples**
Consecutive environment steps are highly correlated (adjacent states are similar). Gradient descent assumes i.i.d. samples; violating this causes catastrophic instability — the network overfits to the current trajectory and forgets everything else.

**Problem 2: Moving targets**
The target `y = r + γ · max Q_θ(s', a')` is computed using the *same* network we're updating. So when we take a step to reduce the loss, the target also shifts. It's like trying to hit a dartboard that moves every time you throw.

DQN's two key innovations — the **replay buffer** and the **target network** — solve these two problems directly.

### The Replay Buffer

The replay buffer is a fixed-capacity ring buffer (typically 10k-1M transitions) storing `(s, a, r, s', done)` tuples. Instead of training on the most recent transition, we sample a **random mini-batch** from the buffer.

**Why this works**:
- **Breaks correlation**: Random samples are (approximately) i.i.d., satisfying gradient descent's assumption
- **Data efficiency**: Each transition can be reused many times in different batches
- **Enables off-policy learning**: Old transitions were collected with an older version of the policy, but since we're learning `Q*` (not the policy directly), those samples are still valid training data

### The Target Network

The target network is a **frozen copy of the Q-network**, used only to compute targets:

```
y = r + γ · max_a' Q_target(s', a')     # computed with frozen Q_target
loss = (Q_online(s, a) - y)²             # only Q_online gets gradient updates
```

Every N steps (e.g., every 100 environment steps), we copy the online network's weights into the target network:

```
Q_target.load_state_dict(Q_online.state_dict())
```

**Why this works**: Targets change *slowly* (only when we sync). The online network can make several updates against a stationary target before the target shifts. This turns a moving-target problem into a sequence of stationary optimisation problems.

There are two variants:
- **Hard update**: Copy every N steps (what we'll use — simpler)
- **Soft update**: Blend at every step: `θ_target ← τ·θ_online + (1-τ)·θ_target` with small τ like 0.005. Smoother but more hyperparameters.

### ε-Greedy Exploration

DQN's Q-function gives a deterministic greedy policy (`argmax`), which doesn't explore. So we add randomness explicitly:

```python
if random() < epsilon:
    action = random_action()
else:
    action = argmax(Q(s))
```

ε decays over time:
```
epsilon = max(epsilon_end, epsilon_start * (decay_rate ** episode))
```

Typical schedule: start at 1.0, decay to 0.01 over a few hundred episodes. This way the agent explores aggressively early, then gradually trusts its Q-values.

### The DQN Training Loop, Step by Step

At every environment step:

1. Observe state `s`
2. Pick action `a` via ε-greedy using `Q_online(s)`
3. Execute action, receive `r, s', done`
4. Push transition `(s, a, r, s', done)` into the replay buffer
5. **If buffer has enough samples**: sample a random mini-batch and update:
   - For each transition in the batch:
     - `y = r` if done else `r + γ · max_a' Q_target(s', a')`
   - `loss = mean((Q_online(s, a) - y)²)` over the batch
   - `loss.backward()`, `optimizer.step()`
6. **If step count % target_update_freq == 0**: sync target ← online
7. Decay epsilon

Note that steps 4-6 happen *inside* the episode, unlike REINFORCE where updates only happen at episode end.

### The Terminal State Trick

When `done` is true, there is no `s'` — the episode ended. The Bellman target becomes just the reward:

```
y = r                                    # if done
y = r + γ · max_a' Q_target(s', a')      # otherwise
```

This is usually implemented as `y = r + γ · (1 - done) · max Q_target(s')`, letting `done` mask out the future-value term.

### DQN Characteristics

- **Off-policy**: Can reuse experiences collected by older (different) policies
- **Sample efficient**: Replay buffer amortises data collection
- **More stable**: Target network + buffer tame the moving-target and correlation problems
- **More hyperparameters**: Buffer size, batch size, target sync frequency, ε schedule, learning rate, warmup period
- **Can be finicky**: Solves CartPole reliably in 200-500 episodes once tuned, but hyperparameter choices matter more than in REINFORCE

---

## REINFORCE vs DQN at a Glance

| Aspect | REINFORCE | DQN |
|--------|-----------|-----|
| Learns | Policy `π(a\|s)` directly | Q-function `Q*(s,a)` |
| Update trigger | End of episode | Every environment step (once buffer warm) |
| Uses full return? | Yes (Monte Carlo) | No (1-step TD) |
| On/off policy | On-policy | Off-policy |
| Replay buffer | No | Yes |
| Target network | No | Yes |
| Exploration | Sampling from stochastic policy | ε-greedy |
| Main failure mode | High variance | Q-value overestimation, instability if hyperparameters off |
| Extends to | Continuous actions easily | Discrete actions only (without modification) |

Both are the starting points for most modern RL algorithms:
- **PPO / A2C / SAC** extend REINFORCE with value-function baselines, clipped updates, and entropy bonuses
- **Double DQN / Rainbow / Dueling DQN** extend DQN with tricks to reduce overestimation and improve sample efficiency

Understanding these two gives you the foundation to understand any modern algorithm.

---

## Module Structure

```
app/
├── training/
│   ├── __init__.py
│   ├── base.py              # BaseTrainer ABC
│   ├── reinforce.py         # ReinforceTrainer
│   ├── dqn.py               # DqnTrainer
│   ├── replay_buffer.py     # ReplayBuffer (used by DQN)
│   └── networks.py          # PolicyNetwork, QNetwork (PyTorch nn.Modules)
├── policy/
│   ├── base.py              # (existing)
│   ├── random.py            # (existing)
│   ├── reinforce.py         # ReinforcePolicy (wraps trained PolicyNetwork)
│   └── dqn.py               # DqnPolicy (wraps trained QNetwork)
├── models/
│   └── training.py          # TrainingConfig, TrainingResult, EpisodeMetrics
```

---

## Classes and Functions

### Neural Networks (`app/training/networks.py`)

**`PolicyNetwork(nn.Module)`** — Used by REINFORCE
- `__init__(self, obs_size: int, action_size: int, hidden_size: int = 128)`
- Two hidden layers: `obs_size → hidden_size → hidden_size → action_size`
- Forward pass: `obs → Linear → ReLU → Linear → ReLU → Linear → Softmax → action probabilities`

**`QNetwork(nn.Module)`** — Used by DQN
- `__init__(self, obs_size: int, action_size: int, hidden_size: int = 128)`
- Same architecture but no softmax: outputs raw Q-values
- Forward pass: `obs → Linear → ReLU → Linear → ReLU → Linear → Q-values`

### Replay Buffer (`app/training/replay_buffer.py`)

**`Transition`** — NamedTuple: `(state, action, reward, next_state, done)`

**`ReplayBuffer`**
- `__init__(self, capacity: int)`
- `push(self, transition: Transition) -> None`
- `sample(self, batch_size: int) -> list[Transition]`
- `__len__(self) -> int`
- Circular buffer using `collections.deque`

### Trainers (`app/training/base.py`, `reinforce.py`, `dqn.py`)

**`BaseTrainer(ABC)`**
- `__init__(self, env: gym.Env, config: TrainingConfig)`
- `train(self) -> TrainingResult` — run the training loop
- `_run_episode(self) -> EpisodeMetrics` — run one episode, collect data
- `save_model(self, path: Path) -> None`
- `load_model(self, path: Path) -> None`

**`ReinforceTrainer(BaseTrainer)`**
- Owns a `PolicyNetwork` and `Adam` optimizer
- `train()`:
  - For each episode: run policy, collect (log_prob, reward) at each step
  - Compute discounted returns G_t for each step
  - Normalise returns (subtract mean, divide by std) for stability
  - Loss = `-Σ log_prob * G_t`
  - Backprop and update
- `_compute_returns(self, rewards: list[float], gamma: float) -> list[float]`

**`DqnTrainer(BaseTrainer)`**
- Owns a `QNetwork` (online), `QNetwork` (target), `Adam` optimizer, `ReplayBuffer`
- `train()`:
  - For each step: ε-greedy action selection, store transition in buffer
  - Every N steps: sample batch from buffer, compute targets, update online network
  - Every M steps: sync target network ← online network
- `_select_action(self, state: np.ndarray, epsilon: float) -> int`
- `_update(self, batch: list[Transition]) -> float` — returns loss

### Policies (`app/policy/reinforce.py`, `app/policy/dqn.py`)

**`ReinforcePolicy(BaseActionPolicy)`**
- `__init__(self, network: PolicyNetwork)`
- `choose_action()`: forward pass through network, sample from output distribution
- `load(cls, path: Path) -> ReinforcePolicy` — class method to load from saved model

**`DqnPolicy(BaseActionPolicy)`**
- `__init__(self, network: QNetwork)`
- `choose_action()`: forward pass, return argmax of Q-values (greedy, no exploration)
- `load(cls, path: Path) -> DqnPolicy` — class method to load from saved model

### Models (`app/models/training.py`)

**`TrainingConfig(BaseModel)`**
- `algorithm: str` — "reinforce" or "dqn"
- `num_episodes: int = 1000`
- `gamma: float = 0.99` — discount factor
- `learning_rate: float = 1e-3`
- `hidden_size: int = 128`
- DQN-specific: `batch_size: int = 64`, `buffer_capacity: int = 10000`, `target_update_freq: int = 100`, `epsilon_start: float = 1.0`, `epsilon_end: float = 0.01`, `epsilon_decay: float = 0.995`

**`EpisodeMetrics(BaseModel)`**
- `episode: int`
- `total_reward: float`
- `steps: int`
- `loss: float | None`

**`TrainingResult(BaseModel)`**
- `episodes: list[EpisodeMetrics]`
- `config: TrainingConfig`
- `final_avg_reward: float` — average reward over last 100 episodes
- `solved: bool` — whether CartPole was "solved" (avg reward ≥ 475 over 100 episodes)

---

## CLI Integration

New subcommand in `__main__.py`:

```
python -m app train --algorithm reinforce --episodes 1000 --lr 0.001
python -m app train --algorithm dqn --episodes 500 --lr 0.001 --batch-size 64
python -m app run-sim --policy reinforce --model-path models/reinforce_best.pt
python -m app run-sim --policy dqn --model-path models/dqn_best.pt
```

Training saves the best model (by rolling average reward) to `models/` directory.

---

## Key Tests

### Unit Tests
- `test_networks.py`: Forward pass shapes, gradient flow, deterministic output with seed
- `test_replay_buffer.py`: Push/sample, capacity overflow, sample size validation
- `test_reinforce_trainer.py`: Return computation correctness, single training step doesn't crash, loss decreases over a few episodes
- `test_dqn_trainer.py`: Epsilon-greedy action selection, target network sync, single update step
- `test_training_policies.py`: `ReinforcePolicy` and `DqnPolicy` conform to `BaseActionPolicy`, produce valid actions, load/save roundtrip

### Integration Tests
- `test_training_integration.py`: Train REINFORCE for 50 episodes (smoke test — verify it runs without error and reward is tracked)
- `test_training_integration.py`: Train DQN for 50 episodes (same)
- `test_trained_policy_in_sim.py`: Load a trained model, run it through `EpisodeRunner`, verify it produces `EpisodeResult`

---

## Sequence Diagrams

### REINFORCE Training Loop

```
User -> CLI: python -m app train --algorithm reinforce
CLI -> ReinforceTrainer: train()
loop for each episode
    ReinforceTrainer -> gym.Env: reset()
    loop for each step
        ReinforceTrainer -> PolicyNetwork: forward(obs)
        PolicyNetwork --> ReinforceTrainer: action_probs
        ReinforceTrainer -> ReinforceTrainer: sample action, store log_prob
        ReinforceTrainer -> gym.Env: step(action)
        gym.Env --> ReinforceTrainer: obs, reward, done
    end
    ReinforceTrainer -> ReinforceTrainer: compute_returns(rewards, gamma)
    ReinforceTrainer -> ReinforceTrainer: loss = -Σ(log_prob * return)
    ReinforceTrainer -> Adam: optimizer.step()
    ReinforceTrainer -> ReinforceTrainer: log EpisodeMetrics
end
ReinforceTrainer -> disk: save best model
ReinforceTrainer --> CLI: TrainingResult
```

### DQN Training Loop

```
User -> CLI: python -m app train --algorithm dqn
CLI -> DqnTrainer: train()
loop for each episode
    DqnTrainer -> gym.Env: reset()
    loop for each step
        DqnTrainer -> QNetwork: forward(obs)
        QNetwork --> DqnTrainer: q_values
        DqnTrainer -> DqnTrainer: ε-greedy action selection
        DqnTrainer -> gym.Env: step(action)
        gym.Env --> DqnTrainer: obs', reward, done
        DqnTrainer -> ReplayBuffer: push(s, a, r, s', done)
        alt buffer has enough samples
            DqnTrainer -> ReplayBuffer: sample(batch_size)
            DqnTrainer -> QNetwork(target): forward(next_states)
            DqnTrainer -> DqnTrainer: targets = r + γ * max Q_target(s')
            DqnTrainer -> QNetwork(online): compute loss, backprop
            DqnTrainer -> Adam: optimizer.step()
        end
        alt every target_update_freq steps
            DqnTrainer -> QNetwork(target): load_state_dict(online)
        end
    end
    DqnTrainer -> DqnTrainer: decay epsilon
    DqnTrainer -> DqnTrainer: log EpisodeMetrics
end
DqnTrainer -> disk: save best model
DqnTrainer --> CLI: TrainingResult
```

---

## Implementation Stages

### Stage 1: Models and Neural Networks
- `TrainingConfig`, `EpisodeMetrics`, `TrainingResult` pydantic models
- `PolicyNetwork` and `QNetwork` nn.Modules
- Tests for forward pass shapes and config validation
- ~150 lines

### Stage 2: Replay Buffer
- `Transition` namedtuple and `ReplayBuffer` class
- Tests for push, sample, overflow, edge cases
- ~80 lines

### Stage 3: REINFORCE
- `BaseTrainer` ABC
- `ReinforceTrainer` with full training loop
- `ReinforcePolicy` (BaseActionPolicy subclass)
- Tests: return computation, training smoke test, policy conformance
- ~200 lines

### Stage 4: DQN
- `DqnTrainer` with replay buffer, target network, ε-greedy
- `DqnPolicy` (BaseActionPolicy subclass)
- Tests: action selection, target sync, training smoke test, policy conformance
- ~250 lines

### Stage 5: CLI Integration and Save/Load
- `train` subcommand in `__main__.py`
- `--policy` / `--model-path` options on `run-sim`
- Model save/load (PyTorch state_dict + config JSON)
- Integration tests: train → save → load → run in sim
- ~150 lines

---

## Progress

| Stage | Status |
|-------|--------|
| 1. Models and Neural Networks | Done |
| 2. Replay Buffer | Done |
| 3. REINFORCE | Done |
| 4. DQN | Done |
| 5. CLI Integration and Save/Load | Done |
