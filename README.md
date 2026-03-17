# Spatial Reasoning for Service Robot Scene Understanding

This repository is for an undergraduate thesis on spatial reasoning for hotel delivery robots.

## Thesis Focus

The project studies how a robot can understand local geometric relations in indoor scenes from visual observations.

The core idea is:

1. Use conventional perception tools to extract scene observations from an image.
2. Build a local semantic-geometric scene representation in the robot frame.
3. Use geometry tools to compute distances and relative directions.
4. Answer spatial questions such as where a wall is and how far away it is.

## Target Scenario

The current milestone is single-image spatial understanding in a hotel corridor environment.

Example questions:

- which side is the wall on
- how far is the nearest wall
- is the door in front of the robot

## Proposed Stack

- Development: local PC + GitHub + ModelScope cloud workspace
- Perception output: object detections with geometry in robot coordinates
- Geometry: explicit distance and direction computation
- Reasoning: GCA-style tool-backed spatial question answering
- Cloud usage: GPU models later on ModelScope, local development stays lightweight

## Repository Layout

- `docs/`: thesis-oriented documents and plans
- `src/`: project source code
- `configs/`: scenario and pipeline configuration
- `scripts/`: experiment entrypoints
- `data/`: sample inputs and scenario definitions
- `results/`: metrics and logs

## Immediate Goal

The first milestone is to build a reproducible single-image spatial understanding pipeline:

1. define an image observation protocol
2. load perception outputs into a scene representation
3. compute spatial facts such as side and distance
4. answer basic natural-language spatial questions
5. later replace the mock perception output with cloud-based vision models
