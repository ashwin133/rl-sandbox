# rl-sandbox

`rl-sandbox` is a toolbox for implementing, training, evaluating, and comparing reinforcement learning algorithms against registered [Gymnasium](https://gymnasium.farama.org/) environments.

The project provides reusable boundaries for trainers, action policies, environments, checkpoints, episode execution, metrics, replay, and visualisation. REINFORCE is the first implemented training algorithm, while `CartPole-v1` and `Acrobot-v1` are the initial environments used to exercise the toolbox.

## Current capabilities

- Train REINFORCE policies on registered discrete-action environments.
- Run random or trained policies through a shared episode runner.
- Evaluate training with reward, loss, entropy, and deterministic evaluation metrics.
- Save self-describing, algorithm-aware checkpoints.
- Capture episode frames and replay them independently of a live environment.
- Extend the trainer and policy abstractions to evaluate additional algorithms.

## Start here

See the [project README](https://github.com/ashwin133/demo-rl-1#readme) for installation and CLI workflows. Use the API reference to explore the extension points for new environments, policies, and training algorithms.

- [Environments](api/envs.md)
- [Policies](api/policy.md)
- [Training](api/training.md)
- [Episode execution](api/episode.md)
