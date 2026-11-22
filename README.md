# Sokoban solver — make.fun "Build the best Sokoban solver"

An entry for make.fun's [Build the best Sokoban solver](https://make.fun/bounty/sokoban-solver-01-run-20261121) bounty (sokoban-solver-01-run-20261121).

`main.ts` reads every level in `/input/<id>/level.txt` (XSB text) and writes the player's moves to `/output/<id>/solution.txt`. Scoring rewards solving each level in as few **pushes** as possible: a push-optimal solve is worth 1,000 points per level, and even one push over the optimum earns under half — so the solver is built around proving push-optimal solutions, not just finding them.

## How it works

- **Search over pushes, not steps.** A state is the sorted box squares plus the player's reachable region (normalised to its smallest square), so every walking variation between two pushes collapses into one state. Walking steps are free; only pushes are counted.
- **Admissible heuristic.** For every goal, a reverse "pull" BFS gives the exact number of pushes a lone box needs from any square to that goal. The heuristic is the minimum-cost assignment of boxes to distinct goals under those distances (Hungarian algorithm) — it never overestimates.
- **Deadlock pruning.** Static dead squares (a box pushed there can never reach a goal) are computed once per level; freeze deadlocks and solid 2×2 blocks are checked incrementally around the square a push changed.
- **A portfolio of searches, cheapest first.** Plain A* settles easy levels optimally; IDA* with a transposition table settles medium ones; beam search (two-stage ranking: cheap bound first, assignment bound for the head of the ranking) finds a strong incumbent on the big multi-room levels; a short bounded IDA* then improves that incumbent or proves it optimal. All budgets are counted in expanded states, never in wall-clock time, so two runs on the same input write byte-identical answers.
- On the kit's practice set the solver scores **1,000,000** (every level at its push optimum P*: 13 / 16 / 29 / 47 / 60 pushes), against the house baseline plain-astar's 200,000.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<practice>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22+, no dependencies. The program itself is `main.ts`; the design notes and the test suite that was written before it live with the build record.
