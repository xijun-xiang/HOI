# Method

## Problem

The release studies a single SAC policy across six HumanoidBench
whole-body-object-interaction tasks: push, kitchen, door, package, cabinet,
and truck. The central difficulty is that their state vectors have different
lengths and their return scales are not comparable.

## Shared-policy interface

`TaskAlignedObservation` pads every vector state to the largest task dimension.
The two task-aware controls append a one-hot task identity:

1. **Vanilla MLP-SAC**: aligned state only.
2. **Task-id MLP-SAC**: aligned state plus a task one-hot vector.
3. **Task-token Transformer-SAC**: the one-hot vector is projected into a
   learned prefix token and attended jointly with projected state tokens.

The implementation deliberately keeps the interface explicit: each task is
wrapped independently, all actions must share the same shape, and the final
policy input is always a flat `float32` vector.

## Why task-scale-aware evaluation

Raw mean return is a necessary benchmark score, but one task with a much larger
return magnitude can dominate it. `hoi.analyze` therefore also reports each
task delta divided by the absolute baseline return for that task (with a floor
of one). This diagnostic is reported alongside—not instead of—raw returns.
