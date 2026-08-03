# Learnings — .ai_agents/learnings.md

## Repository positioning

- Issue: Active project documentation described the repository as a CartPole or REINFORCE demo.
- What misunderstanding happened: Concrete first implementations were presented as the repository's identity instead of examples built on shared extension points.
- Why it was wrong: The intended product is a toolbox for evaluating new reinforcement learning algorithms across registered environments. CartPole, Acrobot, and REINFORCE describe current coverage, not the architectural boundary.
- What to do next time: Frame trainers and environments as implementations of the toolbox contracts, and audit README, CLI, package metadata, generated docs, and agent context whenever project positioning changes.
