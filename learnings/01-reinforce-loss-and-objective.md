# REINFORCE: The Loss Equation and Why We Minimise It

## The objective

We have one real goal: **the agent should accumulate as much reward as possible**. Concretely, we want the *expected return*:

```
J(θ) = E[ G_0 ]
```

…where:
- `θ` are the policy network's weights
- `G_0 = r_1 + γ·r_2 + γ²·r_3 + ...` is the total discounted reward of an episode
- `E[...]` averages over the randomness of action sampling and the environment

We want `J(θ)` to be **as large as possible**. This is the *only* thing we actually care about. Everything else is mechanics.

---

## The loss equation

In code, REINFORCE minimises:

```
L(θ) = − Σ_t  log π_θ(a_t | s_t) · Ĝ_t
```

### Each symbol

| Symbol | Meaning |
|--------|---------|
| `θ` | Policy network's weights — the things we are learning |
| `L(θ)` | Loss as a function of those weights |
| `π_θ(a \| s)` | Policy: probability the network assigns to action `a` given state `s` |
| `s_t`, `a_t` | State observed and action taken at step `t` |
| `log π_θ(a_t \| s_t)` | Log-probability the network assigned to the action it actually took |
| `Ĝ_t` | Normalised return from step `t` onward (mean-zero, unit-variance across the episode) |
| `Σ_t` | Sum over every step in the episode |
| `−` | Sign flip — converts "maximise return" into "minimise loss" (see below) |

### The key term: `log π_θ(a_t | s_t) · Ĝ_t`

This is the heart of credit assignment. The product behaves like:

| `Ĝ_t` (outcome) | `log π_θ` (always ≤ 0) | Product | Effect of gradient ascent |
|-----------------|------------------------|---------|---------------------------|
| Positive (above-average outcome) | Negative | Negative | Make this action *more* likely |
| Negative (below-average outcome) | Negative | Positive | Make this action *less* likely |

Summed over the episode and gradient-ascended, this is: "for each action I took, scale its log-probability by how well things went after, and push the policy in that direction."

### Concrete example

CartPole, 3-step episode. Current policy outputs `[0.6, 0.4]` regardless of state. Agent picks actions `[0, 1, 0]`, receives rewards `[1, 1, 1]`, with γ = 0.99:

```
Returns:        G_0 = 2.97   G_1 = 1.99   G_2 = 1.00
Normalised:     Ĝ_0 = +1.22  Ĝ_1 = 0.00   Ĝ_2 = −1.22
log π:          log(0.6)     log(0.4)     log(0.6)
              = −0.51       = −0.92      = −0.51

Loss = −[ (−0.51)(+1.22) + (−0.92)(0) + (−0.51)(−1.22) ]
     = −[  −0.62          +  0        +  +0.62          ]
     = 0
```

This is the **"loss = 0 even when learning happens"** case: when `log π` is constant across the episode and normalised returns sum to zero, the scalar loss collapses to 0. But the *gradient* w.r.t. θ is not zero, because `log π_θ(a|s)` actually varies with the *state* `s` once the network is non-trivial. The variation drives learning, not the value of the loss.

**Practical implication**: in REINFORCE, watch the rolling reward, not the loss. The loss is a surrogate; its absolute magnitude isn't meaningful.

---

## Why we negate to minimise

We want to **maximise** `J(θ)`. But neural networks learn by **gradient descent** — an algorithm that goes *downhill* on whatever function you give it.

The bridge is to define:

```
L(θ) ≡ −J(θ)
```

Now minimising `L` is the same as maximising `J`, because flipping the sign flips peaks into valleys. The math:

```
∇L(θ) = −∇J(θ)
θ ← θ − lr · ∇L(θ)
    = θ − lr · (−∇J(θ))
    = θ + lr · ∇J(θ)        ← gradient ascent on J
```

Same destination as gradient ascent on the original objective, just with opposite signage.

Two reasons we phrase everything as minimisation rather than maximisation:

1. **Convention**: ML grew up on supervised learning where you minimise prediction *error*. The vocabulary (`loss.backward()`, "loss curve", "decreasing loss is good") came from there.
2. **Tooling**: PyTorch optimisers, schedulers, and APIs all assume you give them a thing to minimise. Negating the objective lets you reuse all the existing infrastructure unchanged.

---

## Where this comes from: the policy gradient theorem

The REINFORCE loss isn't an arbitrary choice — it falls out of the **policy gradient theorem**:

```
∇_θ E[ return ] = E[ ∇_θ log π_θ(a | s) · G ]
```

Two pieces of magic:

1. **The log-derivative trick.** We can't directly compute `∇_θ E[return]`, because the *distribution we are sampling from* depends on θ. The identity `∇_θ π_θ = π_θ · ∇_θ log π_θ` lets us move the gradient inside the expectation, giving something we *can* estimate from samples.

2. **Why log probability rather than raw probability**:
   - It falls out naturally from the log-derivative trick.
   - Numerical stability: sums of logs don't underflow, products of probabilities do.
   - The gradient `∇_θ log π_θ(a|s)` is called the **score function** — it points in the direction in parameter space that increases the probability of action `a`. Multiplying by `Ĝ_t` says "step in that direction, scaled by how well things went."

Replacing the expectation with a single-trajectory sample (one episode) gives the loss we use:

```
L(θ) = − Σ_t log π_θ(a_t | s_t) · Ĝ_t
```

---

## TL;DR

> The agent's job is to maximise expected return. We turn that into a minimisation problem by negating, because PyTorch optimisers minimise. The summand `log π · Ĝ` is the credit-assignment term: positive returns push the corresponding actions up; negative returns push them down. The whole formulation comes from the policy gradient theorem; the log is there for both mathematical and numerical reasons.
