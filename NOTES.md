# Design notes

## Sebulba / multiple inference devices for GymnasiumLoop

One global queue
 deque[tuple(initial step state, final step state, trajectory)]
Question: what should maxlen be? 1? or 2 to allow non-blocking pop and push?


One global queue: deque[Networks], which has maxlen 1 (can do 2 to avoid blocking at the cost of more memory). Basically the staging area for the networks to be used on the next inference cycle.

Inferencer:
* copy latest networks to inference device.
* run inference cycle.
* calls Agent.shard_step_state(step_state, sharding), copies trajectory and step state to update devices. append tuple(initial StepState, final StepStat, trajectory) to queue.

Later optimization: double buffering so we have two threads per inferencer, one that is stepping agent one that is stepping environment. Essentially doubles the number of inferencers. Doesn't actually complicate things too much.

Main thread, does updates, everthing with pmap:
pops tuple(initial step state, final step state, trajectory)
calls update_experience
calls GymnasiumLoop._update().
    1. Note: need to change update_experience to take initial StepState.
    2. Note: need to have some API for controlling how many times _update can run without blocking.
       Or is num_off_policy_optims_per_cycle() sufficent?

Changes:
- [ ] Change Agent.step() to only take in networks and step state (i.e. remove opt state and experience).
- [ ] Change update_experience to take initial StepState.
- [ ] Maybe: Change loss() to not take step state? Or it could be optional, and present iff the agent says it needs the final step state then it will be queued with the experience.
- [ ]

### Key question: managing recurrent state

Resolved. Plan is to have the agent implement a method to shard.

Earlier thoughts:

The podracers paper has each actor shard its trajectory to the number of learners.
But to support the stored state thing from R2D2, would need to shard the recurrent state.
So maybe: add AgentState.per_env_step or AgentState.recurrent. To hold the per-env step state
      so that it can be sharded?
Would probably want to give Agents the option of saving either the state at the start of the inference cycle and / or after the inference cycle.
Or we could delegage to the agent and have an Agent.shard_step_state(step_state, sharding).

Simpler option:
Could have a fixed correspondence between each actor -> some learner. This would probably be much
worse for perf though because it would synchronize all the learners and actors on the slowest actor.

### Good test models

- R2D2
- IMPALA

 TODO:
 try getting rid of actor learner split and just implement double buffering during acting

## Peformance

### JIT + buffer donation for inference cycle bookkeeping


With the following benchmark, this results in a 1.37 increase in
steps / second:

agent = SimplePolicyGradient
networks = 5 hidden layers of size 200
env = CartPole
num_envs = 16
inference = True
num_cycles = 20
steps_per_cycle = 100
