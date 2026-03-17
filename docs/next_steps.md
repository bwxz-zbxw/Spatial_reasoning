# Immediate Next Steps

## Current Priority

The current priority is single-image spatial understanding, not robot action decision.

The first question the system must answer reliably is:

- `墙在我的哪边，离我有多远？`

## What Has To Be True

Before using a large model, the project needs:

- a stable image observation protocol
- a clear robot-centered coordinate convention
- geometry functions for distance and side estimation
- a spatial question answering interface

## Development Order

1. keep the perception output format stable
2. support more spatial question types
3. replace mock observation files with cloud-based visual outputs
4. evaluate the correctness of spatial answers
5. only then move on to robot decision logic

## This Week's Target

- run the single-image question answering demo
- inspect whether the returned side and distance are reasonable
- add more sample observations
- prepare the cloud-side perception module interface
