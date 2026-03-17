# Thesis Project Plan

## Working Title

Geometrically Constrained Spatial Reasoning for Hotel Delivery Robot Yielding and Avoidance

## 1. Technical Route

The project adopts a layered architecture rather than end-to-end model control.

### 1.1 Perception Layer

Input sources:

- robot pose
- local occupancy or depth observations
- detected dynamic objects such as pedestrians and carts

Main output:

- local semantic-geometric scene graph

Each object is represented with:

- category
- position
- orientation
- size
- velocity
- confidence

### 1.2 Spatial Representation Layer

The local environment is converted into a structured scene graph.

Node types:

- robot
- human
- cart
- wall
- door
- corridor segment
- goal waypoint

Key relation types:

- in_front_of
- on_left_of
- on_right_of
- approaching
- crossing
- blocking
- traversable_width
- distance_to
- time_to_collision

### 1.3 GCA-style Reasoning Layer

This layer follows the GCA idea: formalize the task first, then solve under geometric constraints.

Pipeline:

1. parse the current navigation task and scene state
2. generate structured spatial constraints
3. call geometry tools to compute factual values
4. choose a high-level action with explicit justification

Expected action set:

- proceed
- slow_down
- stop_and_wait
- yield_right
- yield_left
- local_replan

### 1.4 Planning and Execution Layer

The high-level action is mapped to the navigation stack:

- `proceed`: keep current behavior
- `slow_down`: reduce velocity limit
- `stop_and_wait`: stop until clearance recovers
- `yield_right` or `yield_left`: shift local target or bias controller
- `local_replan`: trigger a local path update

The low-level safety loop remains in Nav2 and collision monitoring.

## 2. Research Content

### 2.1 Scene Graph Construction for Hotel Corridor Interaction

Build a local semantic-geometric representation that supports dynamic indoor interaction reasoning.

### 2.2 Spatial Constraint Formalization for Yielding Tasks

Translate robot task intent and safety requirements into machine-checkable spatial constraints.

Example constraints:

- minimum human clearance must be at least `0.8 m`
- stop when estimated time-to-collision is below threshold
- prefer right-side yielding in narrow corridors
- trigger replan when free width is below robot width plus margin

### 2.3 Geometry-backed Decision Module

Use deterministic geometry computation instead of free-form text reasoning for:

- clearance estimation
- collision risk estimation
- corridor passability checking
- candidate action scoring

### 2.4 System Integration and Evaluation

Integrate the reasoning module into a simulated hotel delivery robot pipeline and compare it with baseline methods.

## 3. Innovation Points

### 3.1 GCA Transfer to Robot Navigation Interaction

Apply geometrically constrained reasoning to a robot yielding task rather than only static benchmark reasoning.

### 3.2 Hybrid Architecture of Semantics and Geometry

Combine semantic task parsing with explicit geometry verification, reducing hallucinated spatial decisions.

### 3.3 Explainable Avoidance Decisions

Each action is backed by measurable spatial evidence such as free width, relative direction, and time-to-collision.

### 3.4 Task-oriented Spatial Constraint Schema

Design a constraint schema specialized for indoor service robot interaction scenarios.

## 4. Feasibility Analysis

The project is feasible because:

- the main innovation is in reasoning and system integration, not training a new foundation model
- a local corridor setting keeps the environment bounded and measurable
- existing ROS 2 and Gazebo tools can provide the navigation baseline
- cloud resources can be used for heavier experiments while local development stays lightweight

Main risks:

- ROS 2 and simulation integration may consume time early
- dynamic object perception can become a distraction if made too complex
- reasoning must stay structured and tool-backed to avoid becoming an untestable demo

Control strategy:

- start from simulation first
- use simple object abstractions before advanced perception models
- make all reasoning outputs structured and measurable

## 5. Implementation Schedule

### Phase 1: Project Setup and Baseline

Time window: March to early April

Tasks:

- build repository structure
- finalize scenario definitions
- set up cloud workflow with GitHub and ModelScope
- build a baseline navigation and yielding policy

### Phase 2: Scene Graph and Geometry Module

Time window: April

Tasks:

- define object schema and relation schema
- implement local scene graph builder
- implement geometry metrics such as distance, free width, and time-to-collision

### Phase 3: GCA-style Reasoning Module

Time window: May

Tasks:

- define constraint JSON protocol
- implement task-to-constraint parser
- implement action selector with explicit reasoning outputs

### Phase 4: Integration and Experiment

Time window: late May to June

Tasks:

- integrate with navigation stack
- run corridor experiments
- compare baseline, rule-based, and constrained-reasoning methods

### Phase 5: Thesis Writing and Demo

Time window: June

Tasks:

- organize experiment tables and figures
- write thesis chapters
- prepare final demo and defense materials

## 6. Evaluation Metrics

- task success rate
- collision count
- minimum human-robot clearance
- average completion time
- path length overhead
- decision latency
- yielding correctness rate
