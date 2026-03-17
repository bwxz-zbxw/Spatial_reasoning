# Single-Image Spatial Understanding

## Goal

The current project milestone is not robot control. It is image-grounded spatial understanding.

The first capability target is:

- answer where a wall is relative to the robot
- answer how far that wall is

## GCA-style Decomposition

The pipeline is split into two stages.

### Stage 1: Visual tools extract geometry

This stage is expected to run on the cloud later when GPU is needed.

Output examples:

- wall detections
- door detections
- estimated depth or relative position
- confidence score

### Stage 2: Geometry tools answer spatial questions

This stage is lightweight and can run locally.

Inputs:

- object category
- position in robot coordinates
- size
- confidence

Outputs:

- relative side such as left, right, front-left
- metric distance
- concise explanation

## Coordinate Convention

All object positions are expressed in the robot frame:

- `x`: forward
- `y`: left
- `z`: upward

This choice makes questions such as "which side is the wall on" easy to answer.

## Immediate Deliverable

The minimum deliverable is a script that reads one image observation file and answers:

- `墙在我的哪边，离我有多远？`

The first version can use a mock perception output file instead of a real vision model.
