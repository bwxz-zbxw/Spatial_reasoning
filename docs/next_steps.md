# Immediate Next Steps

## What To Do First

### Step 1: Freeze the scope

Do not start with full-building navigation.

The first research scope is:

- hotel corridor local interaction
- three scenarios
- high-level yielding decisions
- simulation-first validation

### Step 2: Build the minimum protocol

Before writing model code, define:

- scene object schema
- relation schema
- constraint JSON format
- action JSON format

### Step 3: Build the baseline

Implement a simple baseline policy with fixed rules:

- slow down when a human is in front within threshold
- stop when free width is too small
- replan when blockage persists

### Step 4: Build the constrained reasoning pipeline

Add:

- task parser
- geometry tools
- action selector
- experiment runner

## This Week's Deliverables

- create a GitHub repository if not already created
- commit the project skeleton
- define the three target scenarios
- define the scene and action protocol
- implement the first geometry utility functions

## Suggested Order

1. finish the protocol
2. implement the geometry layer
3. add a rule baseline
4. add the GCA-style reasoning layer
5. run the first simulated experiments
