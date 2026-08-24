# musoe

A 3D rigid-body physics engine in Rust, built web-first.

`musoe` (무쇠, Korean for *cast iron*) targets the browser as a first-class platform: the
WebAssembly build is the reference target, not an afterthought. It ships a TGS-soft solver
with SIMD constraint batching, a deterministic broad phase, continuous collision detection,
and a canonical snapshot format you can replay bit-for-bit.

> **Status: working engine, pre-1.0.** All milestones M0–M5 are done (collision, solver, CCD,
> mesh/heightfield, threading, determinism level L1, character controller). The API is not
> stable yet and nothing is published to crates.io.

---

## Why another engine

Most engines treat WebAssembly as a port. That shows up in the details that matter on the web:
allocations inside the step loop, a marshalling layer between JS and the physics core, and
math kernels laid out for AoS scalar code rather than 4-wide SIMD.

`musoe` makes those the design constraints instead:

- **Zero heap allocation during a step.** Enforced by a ratchet in CI, not by convention.
  All per-step scratch is reserved up front from a 128 MiB budget, and every reservation is
  fallible — running out of memory returns an error, it does not panic.
- **Flat-buffer C ABI for the browser**, with no `wasm-bindgen`. The host reads body transforms
  straight out of linear memory; the hot path never marshals per-body objects.
- **SoA math as the canonical path.** `Vec3x4` / `Aabbx4` / `Mat3x4` are the primary types in
  the solver and broad phase; the scalar path exists to prove the SIMD path is bit-identical.
- **Failure is an error, not a trap.** Every fallible entry point has a `try_*` form. In a
  browser a panic is a trap that kills the whole module, so the engine never uses one for
  input it can reject.

## Features

| Area | What's there |
|---|---|
| **Shapes** | Sphere, capsule, cylinder, box, convex hull, triangle, compound (≤ 8 children), triangle mesh, heightfield. All convex shapes support a rounding radius. |
| **Solver** | TGS-soft (soft-step, Box2D v3 lineage), 4 substeps by default, SIMD constraint blocks with graph colouring, PCM warm start keyed by canonical feature IDs. |
| **Joints** | Point, hinge, fixed, distance, prismatic — each with limits and motors. World-axis rotation locks. |
| **CCD** | Linear conservative advancement plus rotational sweeps, gated so that non-rotating and rotation-invariant shapes pay nothing. |
| **Queries** | Ray casts (nearest / all), AABB and point queries, shape sweeps, closest-point projection — each with filters. |
| **Events** | Contact started/stopped, contact-force events with per-body thresholds, sensors, collision groups. |
| **Simulation** | Islands and sleeping, damping, restitution, friction, kinematic bodies, a character controller. |
| **State** | Canonical snapshot / restore with a replay hash. Restore validates structure and rejects states the live engine could never reach. |

## Quick start

```rust
use musoe_collide::{Pose, Shape};
use musoe_dynamics::{Body, World, WorldParams};
use musoe_math::{Quat, Vec3};

let mut world = World::new(WorldParams::default());

// A static floor. Static bodies carry no mass.
world.add_body(Body::new_static(
    Shape::Box { half: Vec3::new(10.0, 0.5, 10.0), radius: 0.0 },
    Pose { pos: Vec3::new(0.0, -0.5, 0.0), rot: Quat::IDENTITY },
));

// A falling box. The third argument is **density**, not mass.
let cube = world.add_body(Body::new_dynamic(
    Shape::Box { half: Vec3::new(0.5, 0.5, 0.5), radius: 0.0 },
    Pose { pos: Vec3::new(0.0, 4.0, 0.0), rot: Quat::IDENTITY },
    1.0,
));

for _ in 0..180 {
    world.step(1.0 / 60.0);
}

let b = world.body(cube);
println!("y = {:.4}, |v| = {:.4}, asleep = {}",
    b.pose().pos.y, b.linvel().length(), world.is_asleep(cube));
```

This is `crates/musoe-dynamics/examples/hello.rs`. Run it with:

```sh
cargo run -p musoe-dynamics --example hello
# y = 0.4999, |v| = 0.0000, asleep = true
```

The box rests at `y = 0.4999` (half-extent `0.5` minus the solver's allowed penetration) and
the island has gone to sleep.

## Layout

| Crate | Contents |
|---|---|
| `musoe-math` | Scalar `Vec3` / `Quat` / `Mat3` plus the SoA 4-wide types that the hot path uses. |
| `musoe-collide` | Shapes, dual 4-ary tree broad phase, GJK/EPA, contact manifold generation, spatial queries. |
| `musoe-dynamics` | Solver, PCM cache, islands and sleeping, joints, CCD, events, snapshots — the `World` API. |
| `musoe-bench` | Micro-benchmarks, stack scenes, and the diagnostic instruments used to justify optimisations. |
| `musoe-wasm` | Browser bindings: flat buffers over a C ABI. Demos live in `web/`. |
| `compare/` | A separate workspace that benchmarks `musoe` against rapier3d and Jolt on identical scenes. |

## Building and testing

```sh
cargo test --workspace          # native
scripts/check.sh                # the real gate: fmt + clippy(-D warnings) + tests + benches
                                # on wasm32-wasip1 and native, plus the allocation ratchet
```

`scripts/check.sh` is what has to be green. It runs the suite on **both** the wasm and native
legs, because they have caught different bugs: a clippy lint that only fires on wasm, and a
timer call that panicked only under `wasm32-unknown-unknown`.

For the browser demos:

```sh
cargo build --release -p musoe-wasm --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/musoe_wasm.wasm web/
# then serve web/ over HTTP and open city.html
```

## Performance

Measured against **rapier3d 0.34** and **Jolt 5.0** on identical scenes, min-of-3, single
thread. Two things are worth saying before the numbers.

**Sleeping and non-sleeping are separate measurements.** If one engine falls asleep and the
other does not, wall-clock time measures *settling success*, not throughput. So the two
regimes are reported separately, with sleep thresholds matched between engines — measuring
each engine's defaults compares tuning, not engines.

**Absolute milliseconds do not survive across sessions.** The same rapier build has drifted 7%
between sessions on the same machine. Only ratios within one session, and cross-executed A/B
runs of two binaries, are trusted here.

### Throughput — sleeping off on both sides

| Scene | musoe | rapier | ratio |
|---|---|---|---|
| Pyramid 10 / 20 / 30 / 40 tiers | 0.355 / 1.318 / 2.971 / 5.389 ms | 0.184 / 0.787 / 1.924 / 4.051 | 1.93 / 1.67 / 1.54 / **1.33×** slower |
| Sphere pile 500 / 1500 / 3000 | 1.037 / 3.488 / 7.907 | 0.592 / 2.052 / 4.855 | 1.75 / 1.70 / **1.63×** slower |
| Mixed (3000 settled + 5/25/100 falling) | 0.330 / 0.359 / 0.409 | 0.186 / 0.181 / 0.218 | 1.77 / 1.98 / 1.88× slower |

rapier's default `num_solver_iterations` is 4 and musoe's default substep count is also 4, so
the gap is not "we do more work per step" — it is the efficiency of one iteration. Closing it
needs both narrow phase and solve to roughly halve; micro-optimisation will not get there.

### Settled — sleeping on, thresholds matched

Most game worlds hold far more sleeping bodies than active ones, so the fully dormant case is
worth its own table. It is a distinct regime, not a summary of the engine: nothing is awake in
the scenes below. Mixed scenes — a settled pile plus a handful of falling bodies — are in the
throughput table above, and musoe is slower there.

| Scene | musoe | rapier | ratio |
|---|---|---|---|
| Pyramid 10–30 tiers | 0.000–0.001 ms | 0.005–0.039 | **5–39× faster** |
| Sphere pile 500 / 1500 / 3000 | 0.002 / 0.004 / 0.007 | 0.023 / 0.071 / 0.317 | **11–45× faster** |
| Mixed (3000 settled + falling) | 0.407 / 0.424 / 0.465 | 0.188 / 0.171 / 0.195 | 2.2–2.5× slower |

The settled numbers come from an all-dormant fast path: if nothing moved last step, the whole
step is provably a canonical identity, and the engine returns after one scan. That is locked
by a test asserting the post-step snapshot is byte-identical.

### Jolt 5.0 — sphere pile 3000, single thread

| Engine | ms/step | bodies asleep |
|---|---|---|
| **musoe** | **7.956** | 3000 / 3000 |
| rapier | 5.001 | 3000 / 3000 |
| Jolt (0 workers) | 10.433 | 0 / 3000 |

musoe is 1.31× faster than Jolt here, but 0 workers is not Jolt's home ground — its strength
is multi-threaded scaling. Jolt's 0/3000 sleep count reflects its default threshold, not a
quality difference.

Browser overhead over native is **1.4–1.75×**, and physics results match to three decimal
places. `spheres3000` runs at 2.77 ms/step in a headless browser.

## Determinism

Three levels, stated as a contract rather than a hope:

- **L0 — same binary, same platform.** Identical results regardless of insertion order,
  thread count, or broad-phase tree topology. The pair set is *defined* independently of the
  tree, and a test locks that; the tree is a performance structure only.
- **L1 — same results across platforms** for the same binary semantics. State-affecting code
  paths must not call `libm`: `atan2`, trig, and friends differ in their last ulp between
  native and wasm. Where an angle is needed the code uses sign tests and cross products.
- **Bit-identical wasm ↔ native is an explicit non-goal.**

Snapshots are canonical: the same world always encodes to the same bytes, and a replay hash
lets you detect divergence.

## Known limitations

Kept explicit rather than hidden, with measurements:

- **Tall stacks are metastable, not stable.** With sleeping disabled, a 40-tier pyramid
  diverges at step 763 with 5 substeps and 821 with 6. Adding solver iterations only delays
  it. The mechanism is residual jitter random-walking the top of a marginally stable stack —
  sleeping is what makes these scenes usable, in this engine and in every other one.
  Those two step counts double as a bit-identity check: any change that alters box-box
  contact must leave them at exactly 763 and 821.
- **Rounded convex hulls** (`Hull { radius > 0 }`) fall back to a single witness point instead
  of a face manifold. The exact inward offset of a general polygon is not implemented yet.
- **Rim contacts on cylinders** — poses where neither the cap nor the side dominates — also
  fall back to a single point. That one is by design: there is no stable feature lineage to
  attach an ID to.
- **Friction on anisotropic contacts** injects a small amount of energy (measured `dE = +0.0213`)
  because the friction cone is projected as a disc.
- The full log of what has been measured, rejected, and fixed is in
  [`docs/engineering-log.md`](docs/engineering-log.md) — including optimisations that were
  tried and reverted, which is usually the more useful half.

## How this codebase is built

Two habits shaped most of the code and are worth knowing before contributing:

**Nothing lands on reasoning alone.** Every performance claim comes from cross-executed A/B
runs of two binaries in one session, with a control group that was not touched. Six hypotheses
were rejected by measurement and six confirmed; the rejected ones are written down so nobody
retries them.

**Adversarial review, then mutation testing.** Changes go through cross-model review, and the
tests are then mutated to check they actually fire. That has repeatedly found tests that
passed for the wrong reason — a threshold test whose fixture landed on exact coordinates, a
rejection test whose counterexample never reached the code being tested. When a fix cannot be
observed by any test, that is recorded in the commit message rather than papered over.

## License

Licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([`LICENSE-MIT`](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
