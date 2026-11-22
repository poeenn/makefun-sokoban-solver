// Sokoban solver for make.fun "Build the best Sokoban solver" (sokoban-solver-01).
//
// How the competition scores (from the kit's checker, program/scorer/checker/check.mjs):
//   - The answer for each level is solution.txt: the player's moves as u/d/l/r letters.
//   - The checker replays the moves. Any illegal step (walking into a wall, pushing a
//     box into a wall or another box) or a replay that ends with a box off its goal
//     scores 0 for that level.
//   - A solved level scores 1000 points when its push count is at most P* (the level's
//     push-optimal count, in the reference manifest), and 500 * P* / pushes otherwise.
//     One push over the optimum already halves the score, so push-optimality is
//     worth a great deal more than merely solving.
//   So the whole game is: solve every level, in as few PUSHES as possible. The number
//   of walking steps does not matter at all.
//
// How this solver works, and how it differs from the house baseline (plain-astar):
//   - The baseline searches over single steps (player position + boxes), minimising
//     moves, with a Manhattan heuristic, no deadlock detection, and a hard cap of
//     200,000 expansions; past the cap it writes an empty (scoring 0) answer.
//   - This solver searches over PUSHES instead. A state is the sorted box squares
//     plus the player's reachable region (normalised to its smallest square), which
//     collapses every walking variation between two pushes into one state. The push
//     graph is far smaller than the move graph, so an A* over pushes with an
//     admissible heuristic proves push-optimal solutions the baseline cannot reach.
//   - Heuristic: for every goal, a reverse "pull" BFS gives the exact number of
//     pushes a lone box needs from any square to that goal. The heuristic is the
//     minimum-cost assignment of boxes to goals under those distances (Hungarian
//     algorithm), which never overestimates: every box must travel to a distinct
//     goal, and other boxes can only slow it down.
//   - Deadlock pruning: static dead squares (a box pushed there can never reach a
//     goal, corners included) are computed once per level; freeze deadlocks (a box
//     immobilised by walls and other boxes while off its goal) and solid 2x2
//     blocks are checked incrementally, only around the square a push changed.
//   - Find, then prove: a weighted A* first dives for an incumbent solution; an
//     ordinary A* bounded to strictly fewer pushes then either finds a better
//     solution (which is optimal) or exhausts the bounded space (which proves the
//     incumbent optimal). Easy levels skip straight through the plain A*.
//   - Budgets are counted in expanded states, never in wall-clock time, so two runs
//     on the same input always write byte-identical answers (the grader runs the
//     entry twice and compares). If no search finishes inside its budget, the best
//     solution found still beats the empty answer a 0 would come from.
//
// Entry protocol (program/v1): for every directory <id> in MAKEFUN_INPUT_DIR
// (default /input) holding a level.txt, write MAKEFUN_OUTPUT_DIR/<id>/solution.txt
// (default /output). Levels are XSB text: '#' wall, '$' box, '.' goal, '@' player,
// '*' box on goal, '+' player on goal, ' ' floor; a cell past the end of a short
// row is a wall.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

export interface Level {
  w: number;
  h: number;
  /** wall[i] = 1 when square i is a wall. Square index i = y * w + x. */
  wall: Uint8Array;
  /** goal[i] = 1 when square i is a goal. */
  goal: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

const DIRS: readonly { dx: number; dy: number; letter: string }[] = [
  { dx: 0, dy: -1, letter: 'u' },
  { dx: 0, dy: 1, letter: 'd' },
  { dx: -1, dy: 0, letter: 'l' },
  { dx: 1, dy: 0, letter: 'r' },
];
const LETTERS = ['u', 'd', 'l', 'r'];

/** Parse XSB level text into walls, goals, boxes and the player square. */
export function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') { goal[i] = 1; goals.push(i); }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goal, goals, boxes, player };
}

/**
 * For one goal, the exact number of pushes a lone box needs from every square to
 * reach that goal, by reverse pull BFS: a box on square t could have been pushed
 * there from t - d (direction d) when t - d is floor and the pusher's square
 * t - 2d is floor too. Squares left at -1 can never reach the goal.
 */
function pullDistances(lv: Level, goalSq: number): Int32Array {
  const { w, wall } = lv;
  const dist = new Int32Array(w * lv.h).fill(-1);
  dist[goalSq] = 0;
  const queue = [goalSq];
  for (let qi = 0; qi < queue.length; qi++) {
    const t = queue[qi]!;
    for (const { dx, dy } of DIRS) {
      const d = dy * w + dx;
      const from = t - d;
      const stand = t - 2 * d;
      if (from < 0 || stand < 0 || from >= dist.length || stand >= dist.length) continue;
      if (wall[from] || wall[stand] || dist[from] !== -1) continue;
      dist[from] = dist[t]! + 1;
      queue.push(from);
    }
  }
  return dist;
}

/**
 * Static dead squares: floor squares from which a box can reach no goal at all
 * (every non-goal corner, and every square along a goal-less wall, is one).
 * dead[i] = 1 marks a square a box must never be pushed onto.
 */
export function deadSquares(lv: Level): Uint8Array {
  const dead = new Uint8Array(lv.w * lv.h).fill(1);
  for (const g of lv.goals) {
    const dist = pullDistances(lv, g);
    for (let i = 0; i < dist.length; i++) if (dist[i] !== -1) dead[i] = 0;
  }
  for (let i = 0; i < dead.length; i++) if (lv.wall[i]) dead[i] = 0;
  return dead;
}

const INF_COST = 1_000_000;

/**
 * Minimum-cost assignment of boxes to distinct goals (Hungarian / JV algorithm,
 * O(n^3)) over a flat cost matrix, with scratch arrays reused across calls.
 * Returns INF_COST when some box can reach no goal.
 */
class Assignment {
  private u = new Float64Array(64);
  private v = new Float64Array(64);
  private p = new Int32Array(64);
  private way = new Int32Array(64);
  private minv = new Float64Array(64);
  private used = new Uint8Array(64);

  solve(cost: Int32Array, n: number, m: number): number {
    if (n > m) return INF_COST;
    const { u, v, p, way, minv, used } = this;
    u.fill(0, 0, n + 1); v.fill(0, 0, m + 1); p.fill(0, 0, m + 1);
    for (let i = 1; i <= n; i++) {
      p[0] = i;
      let j0 = 0;
      minv.fill(Infinity, 0, m + 1);
      used.fill(0, 0, m + 1);
      do {
        used[j0] = 1;
        const i0 = p[j0]!;
        const base = (i0 - 1) * m;
        let delta = Infinity;
        let j1 = 0;
        for (let j = 1; j <= m; j++) {
          if (used[j]) continue;
          const cur = cost[base + j - 1]! - u[i0]! - v[j]!;
          if (cur < minv[j]!) { minv[j] = cur; way[j] = j0; }
          if (minv[j]! < delta) { delta = minv[j]!; j1 = j; }
        }
        for (let j = 0; j <= m; j++) {
          if (used[j]) { u[p[j]!] += delta; v[j] -= delta; }
          else minv[j] -= delta;
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do {
        const j1 = way[j0]!;
        p[j0] = p[j1]!;
        j0 = j1;
      } while (j0 !== 0);
    }
    let total = 0;
    for (let j = 1; j <= m; j++) {
      if (p[j] !== 0) {
        const c = cost[(p[j]! - 1) * m + j - 1]!;
        if (c >= INF_COST) return INF_COST;
        total += c;
      }
    }
    return total;
  }
}

interface Prepared {
  lv: Level;
  dead: Uint8Array;
  /** Admissible heuristic over a sorted box list, memoised per box placement. */
  heuristic: (boxes: Int32Array, n: number) => number;
  /** The cheaper admissible bound: sum of each box's nearest-goal pull distance. */
  heuristicCheap: (boxes: Int32Array, n: number) => number;
}

/** 'assign' = Hungarian box-to-goal assignment (stronger, costlier per state);
 * 'summin' = sum of each box's nearest-goal pull distance (weaker, far cheaper). */
export type HeuristicMode = 'assign' | 'summin';

function prepare(lv: Level, mode: HeuristicMode = 'assign'): Prepared {
  const dead = deadSquares(lv);
  const pullDist = lv.goals.map((g) => pullDistances(lv, g));
  const nGoals = lv.goals.length;
  const assignment = new Assignment();
  const cost = new Int32Array(24 * 24);
  const cache = new Map<string, number>();
  const sumMin = (boxes: Int32Array, n: number): number => {
    let total = 0;
    for (let bi = 0; bi < n; bi++) {
      let bestD = Infinity;
      for (let gi = 0; gi < nGoals; gi++) {
        const d = pullDist[gi]![boxes[bi]!]!;
        if (d !== -1 && d < bestD) bestD = d;
      }
      if (!Number.isFinite(bestD)) return Infinity;
      total += bestD;
    }
    return total;
  };
  const heuristic = (boxes: Int32Array, n: number): number => {
    if (mode === 'summin') return sumMin(boxes, n);
    if (n > 24 || nGoals > 24) {
      // Far larger than any level this competition draws: fall back to the
      // sum of each box's nearest-goal distance (still admissible).
      let total = 0;
      for (let bi = 0; bi < n; bi++) {
        let bestD = Infinity;
        for (let gi = 0; gi < nGoals; gi++) {
          const d = pullDist[gi]![boxes[bi]!]!;
          if (d !== -1 && d < bestD) bestD = d;
        }
        if (!Number.isFinite(bestD)) return Infinity;
        total += bestD;
      }
      return total;
    }
    const key = String.fromCharCode(...boxes.subarray(0, n));
    const hit = cache.get(key);
    if (hit !== undefined) return hit;
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      for (let gi = 0; gi < nGoals; gi++) {
        const d = pullDist[gi]![b]!;
        cost[bi * nGoals + gi] = d === -1 ? INF_COST : d;
      }
    }
    const total = assignment.solve(cost, n, nGoals);
    const value = total >= INF_COST ? Infinity : total;
    cache.set(key, value);
    return value;
  };
  return { lv, dead, heuristic, heuristicCheap: sumMin };
}

/** Static facts about a level, without searching. */
export function levelStats(text: string): { h0: number; deadCount: number; boxes: number; goals: number } {
  const lv = parseLevel(text);
  const prep = prepare(lv);
  const arr = Int32Array.from(lv.boxes);
  return {
    h0: prep.heuristic(arr, arr.length),
    deadCount: prep.dead.reduce((a, d) => a + d, 0),
    boxes: lv.boxes.length,
    goals: lv.goals.length,
  };
}

/**
 * Scratch board state shared by the search's flood fills and deadlock checks.
 * Everything is stamped or rebuilt per use, so no allocation happens per node.
 */
class Board {
  occ = new Uint8Array(0);        // occ[i] = 1 when a box sits on square i
  private parent = new Int32Array(0);
  private letter = new Int8Array(0);
  private stamp = new Uint32Array(0);
  private queue = new Int32Array(0);
  private curStamp = 0;
  private w = 0;
  private wall: Uint8Array = new Uint8Array(0);
  private goal: Uint8Array = new Uint8Array(0);

  init(lv: Level): void {
    const n = lv.w * lv.h;
    this.occ = new Uint8Array(n);
    this.parent = new Int32Array(n);
    this.letter = new Int8Array(n);
    this.stamp = new Uint32Array(n);
    this.queue = new Int32Array(n);
    this.w = lv.w;
    this.wall = lv.wall;
    this.goal = lv.goal;
  }

  setBoxes(boxes: Int32Array, n: number): void {
    this.occ.fill(0);
    for (let i = 0; i < n; i++) this.occ[boxes[i]!] = 1;
  }

  /** Flood fill of the player's walking region; returns its smallest square.
   * Parents stay valid until the next reachable() call. */
  reachable(from: number): number {
    const { wall, occ, parent, letter, stamp, queue, w } = this;
    this.curStamp++;
    const s = this.curStamp;
    let qh = 0, qt = 0, min = from;
    queue[qt++] = from;
    stamp[from] = s;
    parent[from] = -1;
    while (qh < qt) {
      const cur = queue[qh++]!;
      if (cur < min) min = cur;
      for (let di = 0; di < 4; di++) {
        const nxt = cur + (di === 0 ? -w : di === 1 ? w : di === 2 ? -1 : 1);
        if (nxt < 0 || nxt >= wall.length || wall[nxt] || occ[nxt] || stamp[nxt] === s) continue;
        // Stepping left/right must not wrap around a row edge.
        if ((di === 2 || di === 3) && Math.floor(nxt / w) !== Math.floor(cur / w)) continue;
        stamp[nxt] = s;
        parent[nxt] = cur;
        letter[nxt] = di;
        queue[qt++] = nxt;
      }
    }
    return min;
  }

  has(sq: number): boolean {
    return this.stamp[sq] === this.curStamp;
  }

  /** The walking moves from the flood's start to `to`; call before reachable() again. */
  walkTo(to: number): string {
    let out = '';
    for (let cur = to; this.parent[cur] !== -1; cur = this.parent[cur]!) {
      out = LETTERS[this.letter[cur]!]! + out;
    }
    return out;
  }

  /** Is the box on square b frozen by walls and (recursively) frozen boxes? */
  private frozen(b: number, stack: Int32Array, depth: number): boolean {
    if (this.goal[b]) return false;
    for (let i = 0; i < depth; i++) if (stack[i] === b) return true; // a leaning cycle
    stack[depth] = b;
    const w = this.w;
    const solid = (sq: number): boolean =>
      this.wall[sq] === 1 || (this.occ[sq] === 1 && this.frozen(sq, stack, depth + 1));
    const xBlocked = solid(b - 1) || solid(b + 1);
    if (!xBlocked) return false;
    const yBlocked = solid(b - w) || solid(b + w);
    return yBlocked;
  }

  /**
   * Incremental deadlock check after one push moved a box from `from` to `dest`
   * (occ already describes the new placement): only the pushed box and the boxes
   * next to the two changed squares can newly freeze, and only a 2x2 block that
   * contains dest can newly form.
   */
  deadlockAfterPush(from: number, dest: number): boolean {
    const { wall, occ, goal, w } = this;
    const stack = new Int32Array(32);
    const candidates = [dest];
    for (const c of [from, dest]) {
      for (const nb of [c - 1, c + 1, c - w, c + w]) {
        if (nb >= 0 && nb < occ.length && occ[nb] && nb !== dest) candidates.push(nb);
      }
    }
    for (const b of candidates) {
      if (!goal[b] && this.frozen(b, stack, 0)) return true;
    }
    const dx = dest % w, dy = Math.floor(dest / w);
    for (const [ox, oy] of [[0, 0], [-1, 0], [0, -1], [-1, -1]] as const) {
      const x = dx + ox, y = dy + oy;
      if (x < 0 || y < 0 || x + 1 >= w) continue;
      const cells = [y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1];
      if (cells.every((c) => wall[c] === 1 || occ[c] === 1) &&
          cells.some((c) => occ[c] === 1 && !goal[c])) return true;
    }
    return false;
  }

  /** Full deadlock check for the initial placement. */
  deadlockInitial(boxes: Int32Array, n: number, dead: Uint8Array): boolean {
    for (let i = 0; i < n; i++) {
      const b = boxes[i]!;
      if (dead[b] && !this.goal[b]) return true;
    }
    const stack = new Int32Array(32);
    for (let i = 0; i < n; i++) {
      const b = boxes[i]!;
      if (!this.goal[b] && this.frozen(b, stack, 0)) return true;
    }
    return false;
  }
}

/** A binary heap on (priority, tie-break), both numeric. */
class Heap {
  private prio: number[] = [];
  private tie: number[] = [];
  private val: number[] = [];
  get size(): number { return this.val.length; }
  push(prio: number, tie: number, val: number): void {
    this.prio.push(prio); this.tie.push(tie); this.val.push(val);
    let i = this.val.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      this.swap(i, p); i = p;
    }
  }
  pop(): number {
    const top = this.val[0]!;
    const lp = this.prio.pop()!, lt = this.tie.pop()!, lv2 = this.val.pop()!;
    if (this.val.length > 0) {
      this.prio[0] = lp; this.tie[0] = lt; this.val[0] = lv2;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= this.val.length) break;
        if (c + 1 < this.val.length && this.less(c + 1, c)) c++;
        if (!this.less(c, i)) break;
        this.swap(i, c); i = c;
      }
    }
    return top;
  }
  private less(a: number, b: number): boolean {
    return this.prio[a]! < this.prio[b]! || (this.prio[a] === this.prio[b] && this.tie[a]! < this.tie[b]!);
  }
  private swap(a: number, b: number): void {
    [this.prio[a], this.prio[b]] = [this.prio[b]!, this.prio[a]!];
    [this.tie[a], this.tie[b]] = [this.tie[b]!, this.tie[a]!];
    [this.val[a], this.val[b]] = [this.val[b]!, this.val[a]!];
  }
}

interface SearchResult {
  moves: string;
  pushes: number;
  expanded: number;
}

interface SearchOutcome {
  result: SearchResult | null;
  /** True when the search emptied its open list (space exhausted), as opposed
   * to stopping at the expansion budget. */
  exhausted: boolean;
  expanded: number;
  /** Smallest heuristic value and largest depth seen on popped nodes (diagnostics). */
  minH: number;
  maxG: number;
}

/**
 * A* over pushes (weight = 1) or weighted A* (weight > 1). When `bound` is set,
 * only solutions with strictly fewer pushes than `bound` are sought, and nodes
 * that cannot beat it are pruned: exhausting such a search proves no better
 * solution exists. The budget counts expanded states, never the clock.
 */
function search(prep: Prepared, board: Board, weight: number, budget: number, bound: number): SearchOutcome {
  const { lv } = prep;
  const n = lv.boxes.length;
  const start = Int32Array.from(lv.boxes);
  board.setBoxes(start, n);
  if (start.every((b) => lv.goal[b])) return { result: { moves: '', pushes: 0, expanded: 0 }, exhausted: true, expanded: 0, minH: 0, maxG: 0 };
  if (board.deadlockInitial(start, n, prep.dead)) return { result: null, exhausted: true, expanded: 0, minH: 0, maxG: 0 };
  const h0 = prep.heuristic(start, n);
  if (!Number.isFinite(h0) || h0 >= bound) return { result: null, exhausted: true, expanded: 0, minH: h0, maxG: 0 };

  const boxesStr: string[] = [String.fromCharCode(...start)];
  const playerPos: number[] = [lv.player];
  const gArr: number[] = [0];
  const parentArr: number[] = [-1];
  const segArr: string[] = [''];
  const startCanon = board.reachable(lv.player);
  const best = new Map<string, number>([[String.fromCharCode(startCanon) + boxesStr[0]!, 0]]);
  const heap = new Heap();
  heap.push(weight * h0, 0, 0);
  const boxes = new Int32Array(24);
  const child = new Int32Array(24);
  let expanded = 0;
  let minH = h0;
  let maxG = 0;

  while (heap.size > 0 && expanded < budget) {
    const ni = heap.pop();
    const g = gArr[ni]!;
    const bs = boxesStr[ni]!;
    for (let i = 0; i < n; i++) boxes[i] = bs.charCodeAt(i);
    board.setBoxes(boxes, n);
    const canon = board.reachable(playerPos[ni]!);
    if (best.get(String.fromCharCode(canon) + bs) !== g) continue; // stale heap entry
    { const hh = prep.heuristic(boxes, n); if (hh < minH) minH = hh; if (g > maxG) maxG = g; }
    let solved = true;
    for (let i = 0; i < n; i++) if (!lv.goal[boxes[i]!]) { solved = false; break; }
    if (solved) {
      const parts: string[] = [];
      for (let t = ni; t > 0; t = parentArr[t]!) parts.push(segArr[t]!);
      return { result: { moves: parts.reverse().join(''), pushes: g, expanded }, exhausted: false, expanded, minH, maxG };
    }
    expanded++;
    // Collect every legal push first: the walking strings must be read from
    // this flood fill before a child's own flood fill overwrites the parents.
    const pushes: { bi: number; from: number; dest: number; segment: string }[] = [];
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const d = di === 0 ? -lv.w : di === 1 ? lv.w : di === 2 ? -1 : 1;
        const stand = b - d;
        const dest = b + d;
        if (stand < 0 || dest < 0 || stand >= lv.wall.length || dest >= lv.wall.length) continue;
        if ((di === 2 || di === 3) && Math.floor(stand / lv.w) !== Math.floor(b / lv.w)) continue;
        if (!board.has(stand)) continue;
        if (lv.wall[dest] || board.occ[dest] || prep.dead[dest]) continue;
        pushes.push({ bi, from: b, dest, segment: board.walkTo(stand) + LETTERS[di]!.toUpperCase() });
      }
    }
    for (const push of pushes) {
      child.set(boxes.subarray(0, n));
      child[push.bi] = push.dest;
      // Insertion sort of the one moved box keeps the list sorted.
      child.subarray(0, n).sort();
      const ng = g + 1;
      const h = prep.heuristic(child, n);
      if (!Number.isFinite(h) || ng + h >= bound) continue;
      board.occ[push.from] = 0;
      board.occ[push.dest] = 1;
      const deadlocked = board.deadlockAfterPush(push.from, push.dest);
      // The player's new square is the pushed box's old square.
      const ncanon = deadlocked ? 0 : board.reachable(push.from);
      board.occ[push.from] = 1;
      board.occ[push.dest] = 0;
      if (deadlocked) continue;
      const cbs = String.fromCharCode(...child.subarray(0, n));
      const key = String.fromCharCode(ncanon) + cbs;
      const seen = best.get(key);
      if (seen !== undefined && seen <= ng) continue;
      best.set(key, ng);
      const idx = boxesStr.length;
      boxesStr.push(cbs);
      playerPos.push(push.from);
      gArr.push(ng);
      parentArr.push(ni);
      segArr.push(push.segment);
      heap.push(ng + weight * h, -ng, idx);
    }
  }
  return { result: null, exhausted: heap.size === 0, expanded, minH, maxG };
}

/**
 * IDA* over pushes: depth-first iterations bounded by f = g + h, with a
 * transposition table that remembers the shallowest visit to each state
 * across all iterations. A* drowns in the frontier of a many-box level (every
 * interleaving of independent rooms sits on the same f plateau), while IDA*
 * walks one interleaving at a time and backtracks locally, at a fraction of
 * the per-node cost: no heap, no node store, the path is the call stack.
 * With an admissible heuristic the first solution IDA* returns is optimal.
 * `boundCap` stops the deepening at that f bound (used to prove an incumbent:
 * exhausting every iteration up to boundCap - 1 shows nothing beats it).
 * The budget counts node visits, never the clock.
 */
function ida(
  prep: Prepared, board: Board, budget: number, boundCap: number,
): SearchOutcome {
  const { lv } = prep;
  const n = lv.boxes.length;
  const start = Int32Array.from(lv.boxes);
  board.setBoxes(start, n);
  if (start.every((b) => lv.goal[b])) {
    return { result: { moves: '', pushes: 0, expanded: 0 }, exhausted: true, expanded: 0, minH: 0, maxG: 0 };
  }
  if (board.deadlockInitial(start, n, prep.dead)) {
    return { result: null, exhausted: true, expanded: 0, minH: 0, maxG: 0 };
  }
  const h0 = prep.heuristic(start, n);
  if (!Number.isFinite(h0)) return { result: null, exhausted: true, expanded: 0, minH: h0, maxG: 0 };

  const tt = new Map<string, number>();
  const boxes = new Int32Array(24);
  const child = new Int32Array(24);
  const segments: string[] = [];
  let visits = 0;
  let minH = h0;
  let maxG = 0;
  let solution: SearchResult | null = null;

  // One depth-first pass with the given f bound; returns the smallest f that
  // exceeded the bound, or -1 when a solution was found / budget ran out.
  const dfs = (player: number, g: number, bound: number): number => {
    if (solution !== null || visits >= budget) return -1;
    visits++;
    if (g > maxG) maxG = g;
    board.setBoxes(boxes, n);
    const canon = board.reachable(player);
    const bs = String.fromCharCode(...boxes.subarray(0, n));
    const key = String.fromCharCode(canon) + bs;
    const seenG = tt.get(key);
    if (seenG !== undefined && seenG <= g) return Infinity; // already searched from here, no deeper
    tt.set(key, g);
    let solved = true;
    for (let i = 0; i < n; i++) if (!lv.goal[boxes[i]!]) { solved = false; break; }
    if (solved) {
      solution = { moves: segments.join(''), pushes: g, expanded: visits };
      return -1;
    }
    // Generate pushes, cheapest heuristic first (read all walking strings
    // before any child flood fill overwrites the parents).
    const cand: { bi: number; from: number; dest: number; segment: string; h: number }[] = [];
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const d = di === 0 ? -lv.w : di === 1 ? lv.w : di === 2 ? -1 : 1;
        const stand = b - d;
        const dest = b + d;
        if (stand < 0 || dest < 0 || stand >= lv.wall.length || dest >= lv.wall.length) continue;
        if ((di === 2 || di === 3) && Math.floor(stand / lv.w) !== Math.floor(b / lv.w)) continue;
        if (!board.has(stand)) continue;
        if (lv.wall[dest] || board.occ[dest] || prep.dead[dest]) continue;
        cand.push({ bi, from: b, dest, segment: board.walkTo(stand) + LETTERS[di]!.toUpperCase(), h: 0 });
      }
    }
    let nextBound = Infinity;
    // Score children (heuristic + deadlock), then visit in ascending h.
    const scored: typeof cand = [];
    for (const c of cand) {
      child.set(boxes.subarray(0, n));
      child[c.bi] = c.dest;
      child.subarray(0, n).sort();
      const h = prep.heuristic(child, n);
      if (!Number.isFinite(h)) continue;
      if (h < minH) minH = h;
      const f = g + 1 + h;
      if (f > bound) { if (f < nextBound) nextBound = f; continue; }
      board.occ[c.from] = 0;
      board.occ[c.dest] = 1;
      const deadlocked = board.deadlockAfterPush(c.from, c.dest);
      board.occ[c.from] = 1;
      board.occ[c.dest] = 0;
      if (deadlocked) continue;
      c.h = h;
      scored.push(c);
    }
    scored.sort((a, b2) => a.h - b2.h || a.bi - b2.bi);
    const saved = Int32Array.from(boxes.subarray(0, n));
    for (const c of scored) {
      boxes.set(saved);
      boxes[c.bi] = c.dest;
      boxes.subarray(0, n).sort();
      // The sort moved the pushed box; the child state is `boxes` now, and the
      // player's square in it is the pushed box's old square c.from.
      segments.push(c.segment);
      const t = dfs(c.from, g + 1, bound);
      segments.pop();
      if (solution !== null || visits >= budget) return -1;
      if (t !== -1 && t < nextBound) nextBound = t;
      boxes.set(saved);
    }
    return nextBound;
  };

  let bound = h0;
  while (solution === null && visits < budget && bound <= boundCap) {
    boxes.set(start);
    tt.clear();
    const t = dfs(lv.player, 0, bound);
    if (solution !== null) break;
    if (t === -1 || !Number.isFinite(t)) break; // budget out, or space exhausted
    bound = t;
  }
  const exhausted = solution === null && visits < budget;
  return { result: solution, exhausted, expanded: visits, minH, maxG };
}

/**
 * Beam search over pushes: at every depth keep only the `width` states with
 * the smallest heuristic (ties: more boxes on goals, then the state key, so
 * the choice is deterministic). A global visited table drops any state
 * already reached in fewer pushes. Beam search proves nothing, but on levels
 * whose rooms make the exact searches thrash through interleavings, it finds
 * a good incumbent fast — and an incumbent turns the exact searches into
 * bounded proofs. Returns null when the beam empties or maxDepth is hit.
 */
function beam(prep: Prepared, board: Board, width: number, maxDepth: number, maxExpansions: number = Infinity, sliceFactor: number = 2): SearchOutcome {
  const { lv } = prep;
  const n = lv.boxes.length;
  const start = Int32Array.from(lv.boxes);
  board.setBoxes(start, n);
  if (start.every((b) => lv.goal[b])) {
    return { result: { moves: '', pushes: 0, expanded: 0 }, exhausted: true, expanded: 0, minH: 0, maxG: 0 };
  }
  if (board.deadlockInitial(start, n, prep.dead)) {
    return { result: null, exhausted: true, expanded: 0, minH: 0, maxG: 0 };
  }
  interface BState { boxes: string; player: number; parent: number; segment: string; h: number; onGoal: number; key: string }
  const startCanon = board.reachable(lv.player);
  const startKey = String.fromCharCode(startCanon) + String.fromCharCode(...start);
  const visited = new Map<string, number>([[startKey, 0]]);
  let frontier: BState[] = [{
    boxes: String.fromCharCode(...start), player: lv.player, parent: -1, segment: '',
    h: prep.heuristic(start, n), onGoal: start.reduce((a, b) => a + lv.goal[b]!, 0), key: startKey,
  }];
  const layers: BState[][] = [frontier];
  const boxes = new Int32Array(24);
  const child = new Int32Array(24);
  let expanded = 0;
  let minH = frontier[0]!.h;

  for (let depth = 0; depth < maxDepth && frontier.length > 0 && expanded < maxExpansions; depth++) {
    // Stage 1: generate every child of the frontier and rank them by the
    // CHEAP heuristic (sum of nearest-goal distances). Only the best few
    // thousand earn the expensive assignment heuristic in stage 2 — the two
    // bounds are strongly correlated, so the true frontier is almost always
    // inside the cheap ranking's head. Only states actually kept enter the
    // visited table, so a state the beam drops can still be found later.
    const pool: (BState & { cheap: number })[] = [];
    for (let si = 0; si < frontier.length; si++) {
      const st = frontier[si]!;
      for (let i = 0; i < n; i++) boxes[i] = st.boxes.charCodeAt(i);
      board.setBoxes(boxes, n);
      board.reachable(st.player);
      expanded++;
      const cand: { bi: number; from: number; dest: number; segment: string }[] = [];
      for (let bi = 0; bi < n; bi++) {
        const b = boxes[bi]!;
        for (let di = 0; di < 4; di++) {
          const d = di === 0 ? -lv.w : di === 1 ? lv.w : di === 2 ? -1 : 1;
          const stand = b - d;
          const dest = b + d;
          if (stand < 0 || dest < 0 || stand >= lv.wall.length || dest >= lv.wall.length) continue;
          if ((di === 2 || di === 3) && Math.floor(stand / lv.w) !== Math.floor(b / lv.w)) continue;
          if (!board.has(stand)) continue;
          if (lv.wall[dest] || board.occ[dest] || prep.dead[dest]) continue;
          cand.push({ bi, from: b, dest, segment: board.walkTo(stand) + LETTERS[di]!.toUpperCase() });
        }
      }
      for (const c of cand) {
        child.set(boxes.subarray(0, n));
        child[c.bi] = c.dest;
        child.subarray(0, n).sort();
        const cheap = prep.heuristicCheap(child, n);
        if (!Number.isFinite(cheap)) continue;
        board.occ[c.from] = 0;
        board.occ[c.dest] = 1;
        const deadlocked = board.deadlockAfterPush(c.from, c.dest);
        const ncanon = deadlocked ? 0 : board.reachable(c.from);
        board.occ[c.from] = 1;
        board.occ[c.dest] = 0;
        if (deadlocked) continue;
        const cbs = String.fromCharCode(...child.subarray(0, n));
        const key = String.fromCharCode(ncanon) + cbs;
        const seenG = visited.get(key);
        if (seenG !== undefined && seenG <= depth + 1) continue;
        pool.push({
          boxes: cbs, player: c.from, parent: si, segment: c.segment, h: cheap, cheap, key,
          onGoal: child.subarray(0, n).reduce((a, b) => a + lv.goal[b]!, 0),
        });
      }
    }
    pool.sort((a, b2) => a.cheap - b2.cheap || b2.onGoal - a.onGoal || (a.boxes < b2.boxes ? -1 : 1));
    const next: BState[] = [];
    const seenKeys = new Set<string>();
    for (const st of pool) {
      if (next.length >= sliceFactor * width) break;
      if (seenKeys.has(st.key)) continue;
      seenKeys.add(st.key);
      for (let i = 0; i < n; i++) child[i] = st.boxes.charCodeAt(i);
      const h = prep.heuristic(child, n);
      if (!Number.isFinite(h)) continue;
      st.h = h;
      if (h < minH) minH = h;
      next.push(st);
    }
    // A solved state in this layer is the beam's answer: rebuild its moves.
    for (let si = 0; si < next.length; si++) {
      const st = next[si]!;
      let allGoal = true;
      for (let i = 0; i < n; i++) if (!lv.goal[st.boxes.charCodeAt(i)]) { allGoal = false; break; }
      if (allGoal) {
        const parts: string[] = [];
        let layer = depth + 1;
        let cur: BState | undefined = st;
        let idx = si;
        while (cur !== undefined && layer > 0) {
          parts.push(cur.segment);
          idx = cur.parent;
          layer--;
          cur = layers[layer]?.[idx];
        }
        return {
          result: { moves: parts.reverse().join(''), pushes: depth + 1, expanded },
          exhausted: false, expanded, minH, maxG: depth + 1,
        };
      }
    }
    next.sort((a, b2) => a.h - b2.h || b2.onGoal - a.onGoal || (a.boxes < b2.boxes ? -1 : 1));
    frontier = next.slice(0, width);
    for (const st of frontier) visited.set(st.key, depth + 1);
    layers.push(frontier);
  }
  return { result: null, exhausted: frontier.length === 0, expanded, minH, maxG: 0 };
}

/** Deterministic per-level budgets, in expanded push states. They are sized
 * so the whole five-level run stays well inside the grader's 60-second wall:
 * the cheap phases settle easy levels in milliseconds, and the expensive
 * beam phases only run on levels the exact searches cannot settle. */
const IDA_BUDGET = 25_000;
const QUICK_BUDGET = 3_000;
const DIVE_BUDGET = 120_000;
const PROVE_BUDGET = 40_000;

/**
 * Solve one level, with diagnostics. Strategy, cheapest first: a plain A*
 * settles easy levels optimally at once; IDA* settles medium ones; beam
 * search finds a strong incumbent on the big multi-room levels (with one
 * wider retry when its push count looks beatable), and a short bounded IDA*
 * then either improves that incumbent or proves it optimal. A weighted dive
 * is the last resort when nothing else produced a solution. 'how' reports
 * 'optimal' only when the solution is proved push-optimal.
 */
export function solveLevelWithStats(text: string, budgets?: { quick?: number; dive?: number; prove?: number; diveW?: number; ida?: number; beamW?: number; beamSlice?: number }, hMode: HeuristicMode = 'assign'): {
  moves: string; how: string; h0: number; deadCount: number; expanded: number; minH: number; maxG: number;
} {
  const quickBudget = budgets?.quick ?? QUICK_BUDGET;
  const diveBudget = budgets?.dive ?? DIVE_BUDGET;
  const proveBudget = budgets?.prove ?? PROVE_BUDGET;
  const lv = parseLevel(text);
  if (lv.boxes.length === 0 || lv.player < 0) {
    return { moves: '', how: 'none', h0: 0, deadCount: 0, expanded: 0, minH: 0, maxG: 0 };
  }
  const prep = prepare(lv, hMode);
  const board = new Board();
  board.init(lv);
  const start = Int32Array.from(lv.boxes);
  const h0 = prep.heuristic(start, start.length);
  const deadCount = prep.dead.reduce((a, d) => a + d, 0);
  let expanded = 0;

  const quick = search(prep, board, 1, quickBudget, Infinity);
  if (quick.result !== null) {
    return { moves: quick.result.moves, how: 'optimal', h0, deadCount, expanded: quick.expanded, minH: quick.minH, maxG: quick.maxG };
  }
  expanded += quick.expanded;
  const idaBudget = budgets?.ida ?? IDA_BUDGET;
  const idaRun = ida(prep, board, idaBudget, Infinity);
  expanded += idaRun.expanded;
  if (idaRun.result !== null) {
    return { moves: idaRun.result.moves, how: 'optimal', h0, deadCount, expanded, minH: idaRun.minH, maxG: idaRun.maxG };
  }
  if (idaRun.exhausted) {
    return { moves: '', how: 'none', h0, deadCount, expanded, minH: idaRun.minH, maxG: idaRun.maxG };
  }
  // Beam search is the workhorse for the big multi-room levels: it finds a
  // strong incumbent where the exact searches thrash. A wider second beam
  // only runs when the first one's push count looks beatable against the
  // heuristic's lower bound, and a short bounded IDA* gets the last word:
  // it either improves the incumbent (optimally) or proves it optimal.
  let incumbent: SearchResult | null = null;
  const beamRun = beam(prep, board, budgets?.beamW ?? (lv.boxes.length >= 15 ? 1100 : 900), 200, 170_000, budgets?.beamSlice ?? 2);
  expanded += beamRun.expanded;
  if (beamRun.result !== null) incumbent = beamRun.result;
  if (incumbent !== null && incumbent.pushes > h0 + 2) {
    const wide = beam(prep, board, 2400, 200, 130_000, budgets?.beamSlice ?? 2);
    expanded += wide.expanded;
    if (wide.result !== null && (incumbent === null || wide.result.pushes < incumbent.pushes)) {
      incumbent = wide.result;
    }
  }
  if (incumbent !== null) {
    const proveB = ida(prep, board, proveBudget, incumbent.pushes - 1);
    expanded += proveB.expanded;
    if (proveB.result !== null) {
      return { moves: proveB.result.moves, how: 'optimal', h0, deadCount, expanded, minH: proveB.minH, maxG: proveB.maxG };
    }
    if (proveB.exhausted) {
      return { moves: incumbent.moves, how: 'optimal', h0, deadCount, expanded, minH: proveB.minH, maxG: proveB.maxG };
    }
    return { moves: incumbent.moves, how: 'greedy', h0, deadCount, expanded, minH: beamRun.minH, maxG: beamRun.maxG };
  }
  const dive = search(prep, board, budgets?.diveW ?? 2, diveBudget, Infinity);
  if (dive.result === null) {
    // No incumbent at all: give the plain A* one long run before giving up.
    const last = search(prep, board, 1, proveBudget, Infinity);
    if (last.result !== null) {
      return { moves: last.result.moves, how: 'optimal', h0, deadCount, expanded: expanded + last.expanded, minH: last.minH, maxG: last.maxG };
    }
    return { moves: '', how: 'none', h0, deadCount, expanded, minH: dive.minH, maxG: dive.maxG };
  }
  expanded += dive.expanded;
  const proveIda = ida(prep, board, proveBudget, dive.result.pushes - 1);
  expanded += proveIda.expanded;
  if (proveIda.result !== null) {
    return { moves: proveIda.result.moves, how: 'optimal', h0, deadCount, expanded, minH: proveIda.minH, maxG: proveIda.maxG };
  }
  if (proveIda.exhausted) {
    return { moves: dive.result.moves, how: 'optimal', h0, deadCount, expanded, minH: proveIda.minH, maxG: proveIda.maxG };
  }
  return { moves: dive.result.moves, how: 'greedy', h0, deadCount, expanded, minH: proveIda.minH, maxG: proveIda.maxG };
}

/** Solve one level: a push-optimal solution when the searches prove one, else
 * the best solution found, else '' (a legal, scoreless answer). */
export function solveLevel(text: string): string {
  return solveLevelWithStats(text).moves;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  for (const id of readdirSync(inputDir).sort()) {
    let moves = '';
    try {
      moves = solveLevel(readFileSync(join(inputDir, id, 'level.txt'), 'utf8'));
    } catch {
      moves = ''; // one bad level must not sink the whole run; '' is a valid answer
    }
    mkdirSync(join(outputDir, id), { recursive: true });
    writeFileSync(join(outputDir, id, 'solution.txt'), moves + '\n');
  }
}

if (process.argv[1] !== undefined && fileURLToPath(import.meta.url) === process.argv[1]) {
  main();
}
