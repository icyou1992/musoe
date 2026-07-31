//! `maru-dynamics` 테스트 — 주제별 모듈.
//!
//! 전에는 `lib.rs` 하나에 10,322줄이 평평하게 들어 있었다(크레이트 본체는 24줄).
//! 테스트가 크레이트 내부(`pub(crate)` 필드·`#[cfg(test)]` 접근자)를 보므로 통합 테스트
//! 디렉터리로는 못 옮긴다 — 크레이트 안에서 주제별로 나눈다. 공용 헬퍼는 여기 두고
//! 각 주제 파일이 `use super::*`로 가져간다.

use super::*;
use maru_collide::{Pose, Shape};
use maru_math::{Quat, Vec3};

mod body;
mod broadphase;
mod ccd;
mod character;
mod command;
mod determinism;
mod events;
mod friction;
mod island;
mod joints;
mod queries;
mod resources;
mod shapes;
mod sleeping;
mod snapshot;
mod solver;
mod validation;

fn unit_sphere_world() -> (World, BodyHandle) {
    let mut w = World::new(WorldParams {
        gravity: maru_math::Vec3::ZERO,
        ..WorldParams::default()
    });
    let h = w.add_body(Body::new_dynamic(
        maru_collide::Shape::Sphere { radius: 0.5 },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
        2.0,
    ));
    (w, h)
}

/// 앵커 하나로 매단 두 바디를 만든다 (A는 static, B는 dynamic).
fn hung_pair(w: &mut World, offset: maru_math::Vec3) -> (BodyHandle, BodyHandle) {
    let a = w.add_body(Body::new_static(
        maru_collide::Shape::Sphere { radius: 0.2 },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
    ));
    let b = w.add_body(Body::new_dynamic(
        maru_collide::Shape::Box {
            half: maru_math::Vec3::new(0.3, 0.3, 0.3),
            radius: 0.0,
        },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 10.0, 0.0) + offset,
            rot: maru_math::Quat::IDENTITY,
        },
        1.0,
    ));
    (a, b)
}

const DT: f32 = 1.0 / 60.0;

fn pose(x: f32, y: f32, z: f32) -> Pose {
    Pose {
        pos: Vec3::new(x, y, z),
        rot: Quat::IDENTITY,
    }
}

fn ground(world: &mut World) -> BodyHandle {
    world.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(50.0, 1.0, 50.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0), // 윗면 y=0
    ))
}

fn no_sleep_params() -> WorldParams {
    WorldParams {
        // 수면이 켜져 있으면 "멈췄다"가 마찰 때문인지 잠들어서인지 갈리지 않는다.
        time_to_sleep: maru_collide::MAX_INPUT,
        ..WorldParams::default()
    }
}

fn pyramid(w: &mut World, tiers: usize) -> Vec<BodyHandle> {
    let mut hs = Vec::new();
    let half = 0.5f32;
    for tier in 0..tiers {
        let count = tiers - tier;
        let y = half + tier as f32 * 2.0 * half;
        let x0 = -(count as f32 - 1.0) * half;
        for i in 0..count {
            hs.push(w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(half, half, half),
                    radius: 0.0,
                },
                pose(x0 + i as f32 * 2.0 * half, y, 0.0),
                1.0,
            )));
        }
    }
    hs
}

/// **M2 수용 기준 (2026-07-26 서브스텝 재보정)**: 피라미드 40단(820박스) 50초 안정.
///
/// # 창 길이가 기준의 일부다
/// 이전 기준은 40단 × 600스텝(10초)이었고 통과했다. 같은 장면을 3000스텝(50초) 돌리면
/// **1200스텝 부근에서 붕괴한다**(top_y 39.3 → 5.5). "안정"이 아니라 "아직 안 무너짐"을
/// 재고 있었다 — 설계의 무근거 목표(100단)를 실측으로 40단으로 내렸던 것과 **같은 계열의
/// 실수를 창 길이에서 반복**했다.
///
/// 실측 장기 봉투(4 서브스텝·30Hz 접촉, 50초):
/// 실측 장기 봉투는 **서브스텝에 묶여 있었다**(hertz가 아니라):
/// substeps=4에서 29단이 한계였고 30단부터 붕괴했다. 기본값을 8로 올리자 **60단**까지
/// 안정이다(80단 붕괴). 경계 근처(60~63)는 **비단조**다 — 61 붕괴·62 안정 —
/// 임계 근처의 전형적 거동이므로 기준은 여유 있는 40단에 둔다.
/// 각 측정은 재현적이다(반복 간 비트 동일 — L0 결정론 유지).
///
/// # 안정의 실질 판정은 '잠들었는가'다
/// 이 엔진에서 장기 안정과 수면은 같은 사건이다: 잔류 속도가 임계 아래로 떨어지면
/// 아일랜드가 얼어 영구 안정이 되고, 떨어지지 않으면 지터가 수십 초에 걸쳐 자라 붕괴한다.
/// 그래서 속도·형상뿐 아니라 **수면**을 함께 요구한다.
///
/// **L0 결정론 게이트 (설계 §8): 같은 빌드·같은 초기상태·같은 호출 순서면 결과는
/// 스레드 수와 무관하게 같아야 한다.**
///
/// 상태 해시만 보지 않고 **스냅샷 바이트 전체**를 대조한다 — 해시는 캐노니컬 상태의
/// 축약이라 충돌 가능성이 남고, 이 계약은 "값이 같다"가 아니라 "상태가 같다"이다.
/// 중간 스텝마다 대조해야 갈림이 생긴 **시점**을 알 수 있다(끝에서만 보면 어디서
/// 갈렸는지 모른다).
///
/// 씬은 병렬 경로를 실제로 타야 의미가 있다 — 페어가 문턱(512)을 넘는 피라미드
/// 20단(210박스)을 쓰고, 실제로 참여자 >1로 돌았는지 진단으로 **증명**한다.
/// (debug 빌드는 debug_assert가 많아 스텝이 비싸다 — 매 커밋 게이트는 짧게 잡고,
/// 설계 §8의 1,000스텝 판정은 아래 release 전용 테스트가 맡는다.)
/// (이 증명이 없으면 "직렬 두 번 돌려 같았다"를 결정론이라 부르게 된다.)
/// 결정론 게이트의 본체 — 스레드 수·솔버 병렬 스위치를 바꿔도 **상태가 같아야** 한다.
fn assert_thread_determinism(tiers: usize, steps: usize, thread_counts: &[usize]) {
    fn run(
        tiers: usize,
        threads: usize,
        par_solver: bool,
        steps: usize,
    ) -> (Vec<Vec<u8>>, usize, usize) {
        let mut w = World::new(WorldParams::default());
        w.set_threads(threads);
        w.set_parallel_solver(par_solver);
        ground(&mut w);
        let _ = pyramid(&mut w, tiers);
        let mut snaps = Vec::new();
        let (mut max_np, mut max_sp) = (0usize, 0usize);
        let every = (steps / 8).max(1);
        for i in 0..steps {
            w.step(DT);
            max_np = max_np.max(w.diagnostics().narrow_threads);
            max_sp = max_sp.max(w.diagnostics().solve_threads);
            if i % every == 0 || i + 1 == steps {
                snaps.push(w.snapshot());
            }
        }
        (snaps, max_np, max_sp)
    }
    let (base, np1, sp1) = run(tiers, 1, false, steps);
    assert_eq!(
        np1, 1,
        "스레드 1개 실행이 병렬 경로를 탔다 — 대조군이 오염됐다"
    );
    assert_eq!(
        sp1, 1,
        "스레드 1개 실행에서 솔버가 병렬로 돌았다 — 대조군 오염"
    );
    let can_thread = World::new(WorldParams::default()).threads() > 1;
    // 내로우만 병렬(기본) · 솔버까지 병렬(노브 on) 두 구성 모두 같은 상태를 내야 한다.
    for par_solver in [false, true] {
        for &t in thread_counts {
            let (got, np, sp) = run(tiers, t, par_solver, steps);
            if can_thread {
                // 스레드를 띄울 수 있는 환경에서만 요구한다(wasm은 항상 1).
                assert!(
                    np > 1,
                    "threads={t}인데 내로우페이즈가 병렬로 안 돌았다 — 게이트가 공회전이다"
                );
                assert_eq!(
                    sp > 1,
                    par_solver,
                    "threads={t} par_solver={par_solver}: 솔버 병렬 여부가 스위치와 다르다"
                );
            }
            assert_eq!(got.len(), base.len());
            for (i, (a, b)) in base.iter().zip(got.iter()).enumerate() {
                assert!(
                    a == b,
                    "threads={t} par_solver={par_solver}: 스냅샷 #{i}에서 상태가 갈렸다 (L0 결정론 위반)"
                );
            }
        }
    }
}

// ===== M4: 하이트필드 (설계 §4.2) =====

fn hf_pose() -> maru_collide::Pose {
    pose(0.0, 0.0, 0.0)
}

// ===== M5: CharacterVirtual (설계 §9) =====

fn char_world() -> World {
    let mut w = World::new(WorldParams::default());
    // 넓은 바닥
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(20.0, 0.5, 20.0),
            radius: 0.0,
        },
        pose(0.0, -0.5, 0.0),
    ));
    w
}

fn capsule_char(x: f32, y: f32, z: f32) -> CharacterVirtual {
    CharacterVirtual::new(
        Shape::Capsule {
            half_height: 0.5,
            radius: 0.3,
        },
        Vec3::new(x, y, z),
    )
}

// ─────────── 리뷰 rev_fd76395d 회귀 (BLOCKER 3 + MAJOR) ───────────

fn dyn_box(w: &mut World, x: f32, y: f32) -> BodyHandle {
    w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(x, y, 0.0),
        1.0,
    ))
}

// ==========================================================================
// 적대 리뷰(2026-07-27)에서 추가한 재현 테스트. 아래 5건은 **현 구현의 결함을
// 드러내기 위한 것**이며, 통과하도록 고치기 전까지 실패한다.
// ==========================================================================
