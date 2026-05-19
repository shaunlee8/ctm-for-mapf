# CTM for MAPF Recovery

This repository studies **decentralized recovery policies for Multi-Agent Path Finding (MAPF)** under partial observability using a Continuous Thought Machine (CTM).

The project is intentionally structured as a standalone codebase:

- **Our code** lives under `src/ctm_for_mapf/` and contains the MAPF environment, recovery task, planners, datasets, models, training loops, and evaluation utilities.
- The official CTM implementation from SakanaAI is included as an **external git submodule** under `third_party/continuous-thought-machines/`.
