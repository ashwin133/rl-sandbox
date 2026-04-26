# The Baseline Bias in CartPole REINFORCE

## The intuition that surfaces it

In CartPole every step gives reward `+1` while the pole stays up. So for an episode of length `T`, the return at step `t` depends *only* on how many steps remain:

```
G_t = 1 + γ·1 + γ²·1 + ... + γ^(T-t-1)
```

This decreases monotonically through the episode. With `γ = 0.99` and `T = 500` (a successful episode):

```
G_0   ≈ 99.3      ← lots of future ahead
G_100 ≈ 98.2
G_250 ≈ 91.7
G_400 ≈ 63.4
G_499 ≈  1.0      ← almost nothing left
```

The drop from ~99 to ~1 is **purely a function of remaining steps**. It says nothing about whether the action taken at step `t` was good or bad.

---

## What the normalisation does

The trainer normalises returns within each episode (`app/training/reinforce.py`):

```python
returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
```

For the example above, mean ≈ 85, std large. The normalised returns ramp roughly linearly from positive to negative across the episode:

```
Ĝ_0   ≈ +1.55      ← strongly positive
Ĝ_100 ≈ +1.51
Ĝ_250 ≈ +1.18
Ĝ_400 ≈ −0.74
Ĝ_499 ≈ −2.92      ← strongly negative
```

The loss term for step 499 is `−log π_θ(a_499 | s_499) · (−2.92)`. Combined with the leading minus on `loss`, the gradient pushes the policy to make `a_499` **less** likely.

But `a_499` is, *by definition* in a 500-step episode, an action that **kept the pole upright**. The algorithm penalises it anyway. **This is the bias.**

---

## Why this happens: the baseline is too crude

The policy gradient lets us subtract any baseline `b_t` without biasing the gradient (provided `b_t` doesn't depend on the action `a_t`):

```
∇J = E[ ∇log π(a_t | s_t) · (G_t − b_t) ]
```

The variance-reducing role of the baseline is to be a good *prediction* of `G_t` from the current state. If `b_t ≈ E[G_t | s_t]`, then `G_t − b_t` is the **advantage** — "did things go better or worse than expected from this state?"

The current normalisation uses `b_t = mean(G_0, G_1, ..., G_{T-1})` — a single scalar applied to every timestep. In CartPole that scalar is around 85, but the *expected* return at step 0 is ~99 and at step 499 is ~1.

Consequence:

| Step | True expected `G_t` | Episode-mean baseline | Bias direction |
|------|---------------------|----------------------|----------------|
| 0    | ~99                 | ~85                  | Artificially boosted (good actions get +14 boost) |
| 250  | ~92                 | ~85                  | Roughly correct |
| 499  | ~1                  | ~85                  | Artificially crushed (good actions get −84 penalty) |

The algorithm isn't *correctly judging* late actions as bad. It's *failing to account for the fact that small returns near the end of an episode are expected* — they don't carry information about action quality.

---

## Why training still works despite the bias

If the bias is so systematic, why does REINFORCE solve CartPole at all? Three reasons:

**1. The network is time-blind.** It sees only the 4-d state `[cart_pos, cart_vel, pole_angle, pole_ang_vel]`, never the timestep. So when a similar state appears at step 5 in one episode and step 495 in another, the gradient updates from both episodes collide on the *same* weights — once with strongly positive `Ĝ`, once with strongly negative `Ĝ`. Over many episodes the systematic part cancels and the *common* signal remains: "this state-action pair is good for keeping the pole up" or "it isn't."

**2. Episode-length variance dilutes the bias.** Early in training, episodes terminate at very different lengths (5, 17, 200, ...). The "late steps" of a 17-step episode are at completely different states than the "late steps" of a 200-step episode. The bias only fully locks in once the policy reliably hits the 500-step cap.

**3. The gradient is noisy enough that the bias is dwarfed by per-episode variance.** It slows learning, but doesn't kill it.

That said, this is a real reason CartPole REINFORCE has notoriously **plateaus near the cap** — once you're consistently hitting 500-step episodes, the systematic penalisation of late-episode actions starts pushing the policy in unhelpful directions, and progress stalls.

---

## The principled fix: a state-dependent baseline

Learn a second network `V_φ(s)` that estimates "expected return from state `s` under the current policy." Then:

```
advantage_t = G_t − V_φ(s_t)
loss = − Σ_t  log π_θ(a_t | s_t) · advantage_t
```

`V_φ(s)` learns that states that look like "early in episode, pole upright" have high expected return, and states that look like "late in episode, pole upright" *also* have high expected return. So the advantage is appropriately small for both — no time-bias.

This is exactly **Actor-Critic**: the network producing `V` is the "critic," the policy network is the "actor." It's the next natural step up from REINFORCE.

---

## Related weaknesses worth knowing

- **The policy can't represent time.** No timestep input → it must base everything on physical state. For CartPole that's actually fine (the dynamics don't care about absolute time), but for tasks where time matters — deadlines, fuel limits, stamina budgets — you'd need to either feed `t` as input or use a recurrent network.
- **Action sampling at evaluation time.** `ReinforcePolicy.choose_action` *samples* from the distribution rather than taking argmax. Even a perfect policy will occasionally make suboptimal choices in deployment.
- **Single-trajectory gradient.** One gradient step uses one episode. Modern algorithms batch multiple parallel envs to reduce variance.

---

## TL;DR

> In CartPole, raw returns drop monotonically through the episode purely because reward is `+1` per step and time is finite. Normalising within an episode subtracts the same scalar baseline at every timestep, but the *expected* return is high early and low late — so early actions are boosted and late actions are crushed regardless of their quality. The algorithm finesses this through state aliasing across episodes, but the bias is a real reason REINFORCE plateaus near the 500-step cap. The fix is a state-dependent baseline `V_φ(s)` — the move from REINFORCE to Actor-Critic.
