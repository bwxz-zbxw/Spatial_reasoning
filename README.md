# Spatial Reasoning for Service Robot Navigation

This repository is for an undergraduate thesis on spatial reasoning for hotel delivery robots.

## Thesis Focus

The project studies how a robot can understand local geometric relations in indoor scenes and use that understanding for yielding and avoidance decisions.

The core idea is:

1. Build a local semantic-geometric scene representation.
2. Convert task intent into structured spatial constraints.
3. Use geometry tools to verify and solve those constraints.
4. Map the result to robot actions such as slow down, stop, yield, or local replan.

## Target Scenario

The initial target scenario is a hotel corridor environment with three typical interaction cases:

- head-on encounter with a pedestrian
- partial blockage by a cart or cleaning trolley
- crossing motion near a corner or elevator hall

## Proposed Stack

- Development: local PC + GitHub + ModelScope cloud workspace
- Simulation: ROS 2 Jazzy + Gazebo
- Navigation: Nav2 + collision monitor
- Perception: RGB-D or LiDAR based local object and free-space extraction
- Reasoning: GCA-style constrained spatial reasoning
- Evaluation: success rate, collision count, minimum clearance, time cost, path overhead

## Repository Layout

- `docs/`: thesis-oriented documents and plans
- `src/`: project source code
- `configs/`: scenario and pipeline configuration
- `scripts/`: experiment entrypoints
- `data/`: sample inputs and scenario definitions
- `results/`: metrics and logs

## Immediate Goal

The first milestone is not full robot deployment. It is to build a reproducible baseline pipeline:

1. define the corridor scenarios
2. define the scene graph and constraint protocol
3. implement geometry utilities
4. implement a reasoning stub that outputs structured actions
5. evaluate against a simple baseline policy
