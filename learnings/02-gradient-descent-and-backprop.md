# How Gradient Descent and Backpropagation Actually Work in REINFORCE

You've seen the loss equation:
```
L(θ) = − Σ_t log π_θ(a_t | s_t) · Ĝ_t
```

…and the three lines that train the network in `app/training/reinforce.py`:

```python
self._optimizer.zero_grad()
loss.backward()
self._optimizer.step()
```

This document explains what those three lines actually do.

---

## The big picture

Gradient descent answers one question: **"What direction should I nudge the weights θ to make `L(θ)` smaller?"**

It has three ingredients:

1. **Compute the loss** by running an episode forward (the *forward pass*).
2. **Compute the gradient** `∇_θ L(θ)` — for every weight in the network, "how much does L change if I nudge this weight a tiny bit?" (the *backward pass*).
3. **Take a step** against the gradient: `θ ← θ − lr · ∇_θ L`.

**Backpropagation** is the algorithm used in step 2. PyTorch implements it for you — you call `loss.backward()` and out come the gradients.

---

## Refresher: the chain rule

Backprop is just the chain rule applied to a graph of operations.

If `y = f(g(x))`, then:

```
dy/dx = (df/dg) · (dg/dx)
```

Each function in the chain contributes one factor. For deep networks the chain has hundreds of links, but the rule is the same: multiply local derivatives along the path from output back to input.

---

## The computational graph

When you run the forward pass, PyTorch builds a **graph** behind the scenes. Each tensor knows:
- What operation produced it
- Which input tensors it depends on
- How to compute its local derivative w.r.t. each input
- **The numerical values it will need to compute that derivative** (called *saved tensors* — see below)

For one timestep of REINFORCE, the graph looks like:

```
obs ──→ Linear ──→ ReLU ──→ Linear ──→ ReLU ──→ Linear ──→ Softmax ──→ probs
                                                                          │
                                                       Categorical(probs).log_prob(a_t)
                                                                          │
                                                                       log_prob_t
                                                                          │
                                                              log_prob_t · Ĝ_t  (then summed and negated across t)
                                                                          │
                                                                        loss
```

Each arrow is an op. Each op knows enough to compute its local derivative. The leaves of this graph are the network's weights (`θ`); the root is the scalar `loss`.

---

## Forward pass: building the graph

Look at the relevant lines in the trainer:

```python
probs = self._network(obs_tensor)              # obs → ... → probs (graph builds)
dist = torch.distributions.Categorical(probs)
action = dist.sample()                          # SAMPLE — a non-differentiable integer
log_prob = dist.log_prob(action)                # differentiable in `probs`
...
log_probs.append(log_prob)
rewards.append(float(reward))
```

Two important details:

- **`dist.sample()` returns an integer; it is NOT differentiated through.** We do not propagate gradients through the random choice itself. The chosen action is treated as a constant for gradient purposes.
- **`dist.log_prob(action)` IS differentiable w.r.t. the network weights** — it computes `log π_θ(a_t | s_t)` where the action `a_t` is fixed. This is the score function from the policy gradient theorem, and it's what carries the gradient back through the network.

After the episode loop finishes:

```python
loss = torch.zeros(1)
for log_prob, g in zip(log_probs, returns_tensor):
    loss = loss - log_prob * g
```

`loss` is now a **scalar tensor connected, through the graph, all the way back to every weight in the network**. The graph contains every forward pass from every step in the episode.

---

## Saved tensors: why backward needs the inputs

The chain rule's per-layer factor depends on the **activations** that flowed through that layer during the forward pass. Concretely, for a linear layer `y = W·x + b`:

```
∂L/∂W = ∂L/∂y · ∂y/∂W
              └───┬───┘
              this is `x`
```

To compute the gradient w.r.t. `W`, we need the actual numerical value of `x` that passed through the layer. The same is true for every op:

| Op | What backward needs cached |
|----|----------------------------|
| `y = W·x + b` (Linear) | `x` (to get `∂L/∂W`); `W` (to get `∂L/∂x`) |
| `y = ReLU(x)` | `x` (to know which elements were ≥ 0 — they keep their gradient; the rest get zeroed) |
| `y = Softmax(x)` | the *output* `y` (the Softmax Jacobian is most easily expressed in terms of outputs) |
| `loss = −log_prob · Ĝ` | both `log_prob` and `Ĝ` (product rule) |

So when you run a forward pass with `requires_grad=True` somewhere in the chain, PyTorch silently **caches the tensors needed for backward** — these are called **"saved tensors."** The computational graph isn't just a list of ops; each node also holds the input/output values it will later need.

This has three practical consequences:

- **Training takes much more memory than inference.** Every forward pass through every layer keeps its activations alive in memory until `loss.backward()` runs and frees them.
- **`torch.no_grad()` exists.** Wrapping a forward pass in `with torch.no_grad():` tells PyTorch "don't build a graph, don't save activations." We use this in `ReinforcePolicy.choose_action()` (`app/policy/reinforce.py`) — at inference time we don't need gradients, so we free that memory immediately.
- **REINFORCE's episode-end backward has a memory cost.** Because we don't call `loss.backward()` until the episode ends, every step's activations stay in memory until then. For a 500-step CartPole episode this is fine; for long-horizon RL problems, it scales with episode length and can become significant.

---

## Backward pass: `loss.backward()`

```python
loss.backward()
```

This single call walks the graph in reverse — from `loss` back to every leaf tensor that has `requires_grad=True` (i.e., every parameter of the network).

At each node, PyTorch:
1. Knows the local derivative of the operation.
2. Multiplies it by the upstream gradient (the chain-rule factor coming from the right).
3. Passes the result down to the inputs.

The result: every parameter `p` in `self._network.parameters()` ends up with a `.grad` attribute containing `∂L/∂p` — the partial derivative of the loss w.r.t. that specific weight.

For our 4 → 128 → 128 → 2 network that's about **17,000 weights**. PyTorch computes 17,000 partial derivatives by walking the graph backwards exactly *once*. (Doing it the dumb way — re-running the forward pass 17,000 times, each time with one weight nudged — would be infeasible. This efficiency is why deep learning works at all.)

---

## What `optimizer.step()` actually does

```python
self._optimizer.step()
```

This applies the gradients to the weights. For plain SGD it would be:

```
for each parameter p:
    p ← p − lr · p.grad
```

We use **Adam**, which is a smarter version of SGD: it keeps running averages of the gradient and the squared gradient, and uses them to adapt the per-parameter learning rate. Roughly:

```
m ← β₁·m + (1−β₁)·grad             # 1st moment (running mean of grad)
v ← β₂·v + (1−β₂)·grad²            # 2nd moment (running mean of grad²)
p ← p − lr · m / (√v + ε)          # update
```

Intuition: parameters whose gradient is consistently large get smaller effective steps (because √v is large); parameters with small or noisy gradients get larger steps. It's a robustness trick that makes training less sensitive to learning-rate choice.

Either way, **`step()` is the line that actually changes the weights**. Everything before it just computed *what* change to make.

---

## Why `zero_grad()`?

```python
self._optimizer.zero_grad()
```

Called *before* `loss.backward()`. Why?

PyTorch **accumulates** gradients into `.grad` rather than overwriting them. That's useful in some scenarios (e.g., when you split a batch across multiple forward passes for memory reasons), but it means you have to manually zero the gradients before each fresh `backward()`. Otherwise this update would use the *sum* of this episode's gradient and last episode's stale gradient.

So the canonical three-line train step is:

```python
optimizer.zero_grad()      # clear stale gradients from last update
loss.backward()            # compute new gradients via backprop
optimizer.step()           # apply gradients to weights
```

You'll see this pattern in basically every PyTorch training loop, RL or otherwise.

---

## Putting it all together: one REINFORCE episode

1. **Reset env**, get initial observation `s_0`.
2. **Loop for each step**:
   - Forward pass: `s_t → network → probs`. PyTorch builds the graph.
   - Sample action `a_t` from `Categorical(probs)`.
   - Capture `log_prob = dist.log_prob(a_t)` — connected to weights through the graph.
   - Step env: `(s_{t+1}, r_{t+1}, done) = env.step(a_t)`.
   - Stash `log_prob` (with its graph) and `reward` (a plain float).
3. **Compute returns** `G_t` from rewards, normalise to get `Ĝ_t`.
4. **Build loss**: `loss = − Σ log_prob_t · Ĝ_t`. Each step's graph plugs into this scalar through the shared network weights.
5. **`zero_grad()`** — wipe last episode's gradients off `.grad`.
6. **`loss.backward()`** — walk the entire episode's graph backwards. At each network weight, `p.grad = ∂L/∂p`.
7. **`optimizer.step()`** — Adam uses each `p.grad` (plus its own state) to update each `p`.
8. The graph is discarded; next episode builds a fresh one.

---

## What gets a gradient and what doesn't

- **Gets a gradient**: every parameter in the policy network — weights and biases of the Linear layers. (`requires_grad=True`, set automatically by `nn.Module`.)
- **Does NOT get a gradient**: the observation, the sampled action (an integer), the reward (a float from the env), the returns (computed from rewards, no graph). PyTorch treats these as constants.
- **Randomness is not differentiated through.** REINFORCE finesses this with the **score-function trick** — we differentiate `log π_θ(a | s)`, which is a function of *weights* and *states only*, not of the random action draws. The randomness of action sampling and environment transitions is handled by the *expectation* in the policy gradient theorem; concretely, it's averaged out by training over many episodes.

---

## Why this whole machinery is the magic of deep learning

Backprop reduces "compute gradients for millions of parameters" to "do one extra graph traversal at the cost of about 2× the forward pass." Without it, every parameter update would require re-running the forward pass once per parameter — completely infeasible.

PyTorch, TensorFlow, JAX — every modern deep-learning framework is fundamentally a tool for **building computational graphs and differentiating them**. REINFORCE just plugs into that machinery: it provides a *clever choice of loss function* (`−Σ log π · Ĝ`) such that gradient descent on it does the right thing for RL.

---

## TL;DR

> The forward pass builds a graph as it computes the loss. `loss.backward()` walks that graph backwards via the chain rule, depositing `∂L/∂p` on every weight. `optimizer.step()` uses those gradients (plus Adam's internal momentum state) to update the weights. `zero_grad()` clears the previous step's gradients beforehand. Together, this is gradient descent: "compute a direction, take a step."
