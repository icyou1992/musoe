//! Maru vs Rapier3D — **같은 장면·같은 하드웨어·같은 릴리스 프로파일** 대조.
//!
//! # 왜 성능만 재면 안 되는가
//! 스택이 무너지는 엔진은 접촉이 사라져 **더 빨라진다**. 그래서 매 시나리오에서
//! 안정성 판정(최종 높이·최대 속도·유한성)과 ms/step을 **함께** 낸다.
//! 안정성 실패는 성능 수치를 무효로 만든다.
//!
//! # 공정성에 대한 정직한 진술
//! - 두 엔진 모두 **각자의 기본 설정**으로 돌린다. 서로의 파라미터를 맞추면 어느 한쪽의
//!   튜닝을 다른 쪽에 강제하는 셈이라 오히려 왜곡된다. 설정은 출력에 명시한다.
//! - Maru는 M2(단일 스레드·CCD 없음·셰이프 4종)다. Rapier는 CCD·다양한 조인트·쿼리
//!   파이프라인을 **함께 돌린다** — 기능 차이가 있으므로 "같은 일을 하는" 비교가 아니다.
//! - 측정은 워밍업 후 창(window)의 평균이다. 첫 스텝은 캐시·할당이 섞여 제외한다.

use std::time::Instant;

const DT: f32 = 1.0 / 60.0;

/// **정착 모드의 수면 파라미터는 양쪽을 맞춘다.**
/// 각 엔진 기본값으로 재면 엔진이 아니라 튜닝을 재게 된다 — 실제로 Maru 기본은
/// 0.03 m/s인데 rapier 0.34 기본은 0.4 m/s로 **13배** 차이였고, 그 상태로 잰
/// 정착 수치는 like-for-like가 아니었다(2026-07-27 발견).
/// 여기서는 rapier 기본(0.4/0.5/2초)보다 엄격하고 Maru 기본보다 느슨한 공통값을 쓴다.
const SLEEP_LIN: f32 = 0.1;
const SLEEP_ANG: f32 = 0.15;
const SLEEP_TIME: f32 = 0.5;

/// 측정 모드.
///
/// **하나의 수치로 둘을 재면 안 된다**(리뷰 BLOCKER): 한쪽이 잠들고 다른 쪽이 깨어 있으면
/// wall-clock은 알고리즘 처리량이 아니라 **정착 성공 여부**를 잰다.
/// - `Throughput`: 양쪽 **수면을 끄고** 순수 처리량만 잰다(같은 일을 하는 비교).
/// - `Settle`: 수면을 켜고 정착 품질 + 총비용을 잰다(실사용 근사).
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Throughput,
    Settle,
}

struct Result_ {
    ms_per_step: f64,
    stable: bool,
    detail: String,
}

// ─────────────────────────── Maru ───────────────────────────
mod maru_side {
    use super::*;
    use maru_collide::{Pose, Shape};
    use maru_dynamics::{Body, World, WorldParams};
    use maru_math::{Quat, Vec3};

    fn pose(x: f32, y: f32, z: f32) -> Pose {
        Pose {
            pos: Vec3::new(x, y, z),
            rot: Quat::IDENTITY,
        }
    }

    pub fn pyramid(mode: Mode, tiers: usize, steps: usize, warmup: usize) -> Result_ {
        let d = WorldParams::default();
        let mut w = World::new(WorldParams {
            // 수면을 끄려면 임계를 0으로 둔다(어떤 속도도 임계 아래가 되지 않는다).
            sleep_linear_threshold: if mode == Mode::Throughput {
                0.0
            } else {
                SLEEP_LIN
            },
            sleep_angular_threshold: if mode == Mode::Throughput {
                0.0
            } else {
                SLEEP_ANG
            },
            time_to_sleep: SLEEP_TIME,
            ..d
        });
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(200.0, 1.0, 40.0),
                radius: 0.0,
            },
            pose(0.0, -1.0, 0.0),
        ));
        let mut handles = Vec::new();
        for t in 0..tiers {
            let count = tiers - t;
            let y = 0.5 + t as f32;
            for i in 0..count {
                let x = (i as f32 - (count as f32 - 1.0) * 0.5) * 1.02;
                handles.push(w.add_body(Body::new_dynamic(
                    Shape::Box {
                        half: Vec3::new(0.5, 0.5, 0.5),
                        radius: 0.0,
                    },
                    pose(x, y, 0.0),
                    1.0,
                )));
            }
        }
        for _ in 0..warmup {
            let _ = w.try_step(DT);
        }
        let t0 = Instant::now();
        let mut acc = [0u64; 5];
        let mut pairs_acc = 0usize;
        let mut reused = 0usize;
        for _ in 0..steps {
            let _ = w.try_step(DT);
            let d = w.diagnostics();
            for k in 0..5 {
                acc[k] += d.phase_ns[k];
            }
            pairs_acc += d.pairs;
            reused += d.bp_reused as usize;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        eprintln!(
            "    [maru 평균 us] aabb={:.0} broad={:.0} narrow={:.0} solve={:.0} post={:.0}  pairs={}",
            acc[0] as f64 / 1000.0 / steps as f64,
            acc[1] as f64 / 1000.0 / steps as f64,
            acc[2] as f64 / 1000.0 / steps as f64,
            acc[3] as f64 / 1000.0 / steps as f64,
            acc[4] as f64 / 1000.0 / steps as f64,
            pairs_acc / steps
        );
        eprintln!("    [maru bp 재사용] {reused}/{steps}");

        let (mut top, mut vmax, mut finite, mut asleep) = (f32::MIN, 0.0f32, true, 0usize);
        for h in &handles {
            let b = w.body(*h);
            top = top.max(b.pose().pos.y);
            vmax = vmax.max(b.linvel().length());
            finite &= b.pose().pos.y.is_finite();
            if w.is_asleep(*h) {
                asleep += 1;
            }
        }
        eprintln!("    [maru sleep] {asleep}/{} 잠듦", handles.len());
        let expect = tiers as f32 - 0.5;
        let stable = finite && (top - expect).abs() < 0.02 * tiers as f32 && vmax < 0.5;
        Result_ {
            ms_per_step: ms,
            stable,
            detail: format!("top_y={top:.2} (expect {expect:.1})  vmax={vmax:.3}  bodies={}", handles.len()),
        }
    }

    pub fn pile(mode: Mode, n: usize, steps: usize, warmup: usize) -> Result_ {
        let sub: u32 = std::env::var("MARU_SUB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(WorldParams::default().substeps);
        let _ = mode;
        let thr: f32 = std::env::var("MARU_SLEEP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let d = WorldParams::default();
        let mut w = World::new(WorldParams {
            substeps: sub,
            sleep_linear_threshold: if mode == Mode::Throughput {
                0.0
            } else if thr > 0.0 {
                thr
            } else {
                SLEEP_LIN
            },
            sleep_angular_threshold: if mode == Mode::Throughput {
                0.0
            } else if thr > 0.0 {
                thr * 1.25
            } else {
                SLEEP_ANG
            },
            time_to_sleep: SLEEP_TIME,
            ..d
        });
        for (dx, dz, hx, hz) in [
            (0.0, 0.0, 30.0, 30.0),
        ] {
            w.add_body(Body::new_static(
                Shape::Box {
                    half: Vec3::new(hx, 1.0, hz),
                    radius: 0.0,
                },
                pose(dx, -1.0, dz),
            ));
        }
        let side = (n as f32).cbrt().ceil() as usize;
        let mut handles = Vec::new();
        'outer: for iy in 0..side * 3 {
            for ix in 0..side {
                for iz in 0..side {
                    if handles.len() >= n {
                        break 'outer;
                    }
                    let x = ix as f32 * 1.1 - side as f32 * 0.55;
                    let z = iz as f32 * 1.1 - side as f32 * 0.55;
                    let y = 0.55 + iy as f32 * 1.1;
                    handles.push(w.add_body(Body::new_dynamic(
                        Shape::Sphere { radius: 0.5 },
                        pose(x, y, z),
                        1.0,
                    )));
                }
            }
        }
        for _ in 0..warmup {
            let _ = w.try_step(DT);
        }
        // **에너지 궤적**: 감소하면 아직 정착 중, 증가·유지면 주입이다(리뷰가 요구한 판별).
        let energy = |w: &World, hs: &[maru_dynamics::BodyHandle]| -> (f64, f64) {
            let (mut ke, mut pe) = (0.0f64, 0.0f64);
            for h in hs {
                let b = w.body(*h);
                let m = 1.0 / b.inv_mass() as f64;
                ke += 0.5 * m * (b.linvel().length_sq() as f64);
                let ii = b.inv_inertia_local();
                ke += 0.5 * (1.0 / ii.x as f64) * (b.angvel().length_sq() as f64);
                pe += m * 9.81 * b.pose().pos.y as f64;
            }
            (ke, pe)
        };
        let t0 = Instant::now();
        let mut acc = [0u64; 5];
        let mut reused = 0usize;
        let mut pairs_acc = 0usize;
        let mut pts_acc = 0usize;
        let mut sub = [0u64; 4];
        let mut acc_late = [0u64; 5];
        let mut sub_late = [0u64; 4];
        let mut late_steps = 0usize;
        let mut t_late = std::time::Duration::ZERO;
        for si in 0..steps {
            let t_step = std::time::Instant::now();
            let _ = w.try_step(DT);
            if si >= steps / 2 {
                t_late += t_step.elapsed();
            }
            let d = w.diagnostics();
            for k in 0..5 {
                acc[k] += d.phase_ns[k];
            }
            for k in 0..4 {
                sub[k] += d.sub_ns[k];
            }
            // **정착 구간만 따로 잰다.** 전체 평균은 앞쪽 몇 스텝의 진짜 정착 과정(수천 접점,
            // 트리 재빌드)이 지배해서 "전원 수면 상태의 스텝 비용"을 가린다 — 이 하네스로
            // 정착 레짐을 최적화하려면 후반부만 봐야 한다.
            if si >= steps / 2 {
                for k in 0..5 {
                    acc_late[k] += d.phase_ns[k];
                }
                for k in 0..4 {
                    sub_late[k] += d.sub_ns[k];
                }
                late_steps += 1;
            }
            reused += d.bp_reused as usize;
            pairs_acc += d.pairs;
            pts_acc += d.warm_total;
            if si % (steps / 8).max(1) == 0 {
                let (ke, pe) = energy(&w, &handles);
                eprintln!("      t={si:4}  KE={ke:11.1}  PE={pe:12.1}  E={:12.1}", ke + pe);
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        let (mut lowest, mut finite, mut asleep) = (f32::MAX, true, 0usize);
        let (mut vlin, mut vang) = (0.0f32, 0.0f32);
        for h in &handles {
            let b = w.body(*h);
            lowest = lowest.min(b.pose().pos.y);
            finite &= b.pose().pos.y.is_finite();
            asleep += w.is_asleep(*h) as usize;
            vlin = vlin.max(b.linvel().length());
            vang = vang.max(b.angvel().length());
        }
        eprintln!("    [maru pile] 잔류 vlin={vlin:.4} vang={vang:.4} (임계 {:.2}/{:.2})",
            WorldParams::default().sleep_linear_threshold,
            WorldParams::default().sleep_angular_threshold);
        eprintln!(
            "    [maru pile] aabb={:.0} broad={:.0} narrow={:.0} solve={:.0} post={:.0} us  수면 {asleep}/{}  bp재사용 {reused}/{steps}",
            acc[0] as f64 / 1000.0 / steps as f64,
            acc[1] as f64 / 1000.0 / steps as f64,
            acc[2] as f64 / 1000.0 / steps as f64,
            acc[3] as f64 / 1000.0 / steps as f64,
            acc[4] as f64 / 1000.0 / steps as f64,
            handles.len()
        );
        let ls = late_steps.max(1) as f64;
        eprintln!(
            "    [maru pile] 정착구간({late_steps}스텝): {:.3} ms/step  aabb={:.0} broad={:.0} narrow={:.0} solve={:.0} post={:.0} us",
            t_late.as_secs_f64() * 1000.0 / ls,
            acc_late[0] as f64 / 1000.0 / ls,
            acc_late[1] as f64 / 1000.0 / ls,
            acc_late[2] as f64 / 1000.0 / ls,
            acc_late[3] as f64 / 1000.0 / ls,
            acc_late[4] as f64 / 1000.0 / ls,
        );
        eprintln!(
            "    [maru pile] 정착구간 하위: 스크래치예약={:.0} wake순회={:.0} narrow순회={:.0} 아일랜드={:.0} us",
            sub_late[0] as f64 / 1000.0 / ls,
            sub_late[1] as f64 / 1000.0 / ls,
            sub_late[2] as f64 / 1000.0 / ls,
            sub_late[3] as f64 / 1000.0 / ls,
        );
        eprintln!(
            "    [maru pile] 전체평균 하위: 스크래치예약={:.0} wake순회={:.0} narrow순회={:.0} 아일랜드={:.0} us",
            sub[0] as f64 / 1000.0 / steps as f64,
            sub[1] as f64 / 1000.0 / steps as f64,
            sub[2] as f64 / 1000.0 / steps as f64,
            sub[3] as f64 / 1000.0 / steps as f64,
        );
        eprintln!(
            "    [maru pile] 페어 {}  접점 {}  → 페어당 narrow {:.2} us",
            pairs_acc / steps,
            pts_acc / steps,
            acc[1] as f64 / 1000.0 / steps as f64 / (pairs_acc / steps).max(1) as f64
        );
        let stable = finite && lowest > -0.5;
        Result_ {
            ms_per_step: ms,
            stable,
            detail: format!("lowest_y={lowest:.2} (관통 없으면 > 0)  bodies={}", handles.len()),
        }
    }
}

// ────────────────────────── Rapier ──────────────────────────
mod rapier_side {
    use super::*;
    use rapier3d::prelude::*;

    struct Sim {
        pipeline: PhysicsPipeline,
        islands: IslandManager,
        bp: DefaultBroadPhase,
        np: NarrowPhase,
        bodies: RigidBodySet,
        colliders: ColliderSet,
        joints: ImpulseJointSet,
        mjoints: MultibodyJointSet,
        ccd: CCDSolver,
        params: IntegrationParameters,
        gravity: rapier3d::math::Vec3,
    }

    impl Sim {
        fn new() -> Self {
            let mut params = IntegrationParameters::default();
            params.dt = DT;
            Self {
                pipeline: PhysicsPipeline::new(),
                islands: IslandManager::new(),
                bp: DefaultBroadPhase::new(),
                np: NarrowPhase::new(),
                bodies: RigidBodySet::new(),
                colliders: ColliderSet::new(),
                joints: ImpulseJointSet::new(),
                mjoints: MultibodyJointSet::new(),
                ccd: CCDSolver::new(),
                params,
                gravity: rapier3d::math::Vec3::new(0.0, -9.81, 0.0),
            }
        }
        fn step(&mut self) {
            self.pipeline.step(
                self.gravity,
                &self.params,
                &mut self.islands,
                &mut self.bp,
                &mut self.np,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.joints,
                &mut self.mjoints,
                &mut self.ccd,
                &(),
                &(),
            );
        }
    }

    /// 빌더가 수면 임계를 노출하지 않으므로 삽입 뒤 직접 맞춘다.
    ///
    /// **`can_sleep`이 참일 때만 만진다.** rapier의 `can_sleep(false)`는 임계를 −1로
    /// 두는 방식이라, 무조건 덮어쓰면 처리량 모드(수면 OFF)에서 rapier를 도로 재운다 —
    /// 공정성을 맞추려던 코드가 반대 방향으로 불공정을 만든다(2026-07-27 실측으로 발견:
    /// 40단 rapier가 3.81 → 0.27 ms로 튀었다).
    fn match_sleep(s: &mut Sim, h: RigidBodyHandle, can_sleep: bool) {
        if !can_sleep {
            return;
        }
        let a = s.bodies[h].activation_mut();
        a.normalized_linear_threshold = SLEEP_LIN;
        a.angular_threshold = SLEEP_ANG;
        a.time_until_sleep = SLEEP_TIME;
    }

    pub fn pyramid(mode: Mode, tiers: usize, steps: usize, warmup: usize) -> Result_ {
        let mut s = Sim::new();
        let can_sleep = mode == Mode::Settle;
        let ground = s.bodies.insert(RigidBodyBuilder::fixed().translation(rapier3d::math::Vec3::new(0.0, -1.0, 0.0)));
        s.colliders.insert_with_parent(
            ColliderBuilder::cuboid(200.0, 1.0, 40.0).density(1.0),
            ground,
            &mut s.bodies,
        );
        let mut handles = Vec::new();
        for t in 0..tiers {
            let count = tiers - t;
            let y = 0.5 + t as f32;
            for i in 0..count {
                let x = (i as f32 - (count as f32 - 1.0) * 0.5) * 1.02;
                let h = s.bodies.insert(
                    RigidBodyBuilder::dynamic()
                        .translation(rapier3d::math::Vec3::new(x, y, 0.0))
                        .can_sleep(can_sleep),
                );
                match_sleep(&mut s, h, can_sleep);
                s.colliders.insert_with_parent(
                    ColliderBuilder::cuboid(0.5, 0.5, 0.5).density(1.0),
                    h,
                    &mut s.bodies,
                );
                handles.push(h);
            }
        }
        for _ in 0..warmup {
            s.step();
        }
        let t0 = Instant::now();
        for _ in 0..steps {
            s.step();
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        let (mut top, mut vmax, mut finite, mut asleep) = (f32::MIN, 0.0f32, true, 0usize);
        for h in &handles {
            let b = &s.bodies[*h];
            top = top.max(b.translation().y);
            vmax = vmax.max(b.linvel().length());
            finite &= b.translation().y.is_finite();
            if b.is_sleeping() {
                asleep += 1;
            }
        }
        eprintln!("    [rapier sleep] {asleep}/{} 잠듦", handles.len());
        let expect = tiers as f32 - 0.5;
        let stable = finite && (top - expect).abs() < 0.02 * tiers as f32 && vmax < 0.5;
        Result_ {
            ms_per_step: ms,
            stable,
            detail: format!("top_y={top:.2} (expect {expect:.1})  vmax={vmax:.3}  bodies={}", handles.len()),
        }
    }

    pub fn pile(mode: Mode, n: usize, steps: usize, warmup: usize) -> Result_ {
        let mut s = Sim::new();
        let can_sleep = mode == Mode::Settle;
        let ground = s.bodies.insert(RigidBodyBuilder::fixed().translation(rapier3d::math::Vec3::new(0.0, -1.0, 0.0)));
        s.colliders.insert_with_parent(
            ColliderBuilder::cuboid(30.0, 1.0, 30.0).density(1.0),
            ground,
            &mut s.bodies,
        );
        let side = (n as f32).cbrt().ceil() as usize;
        let mut handles = Vec::new();
        'outer: for iy in 0..side * 3 {
            for ix in 0..side {
                for iz in 0..side {
                    if handles.len() >= n {
                        break 'outer;
                    }
                    let x = ix as f32 * 1.1 - side as f32 * 0.55;
                    let z = iz as f32 * 1.1 - side as f32 * 0.55;
                    let y = 0.55 + iy as f32 * 1.1;
                    let h = s.bodies.insert(
                        RigidBodyBuilder::dynamic()
                            .translation(rapier3d::math::Vec3::new(x, y, z))
                            .can_sleep(can_sleep),
                    );
                    match_sleep(&mut s, h, can_sleep);
                    s.colliders.insert_with_parent(
                        ColliderBuilder::ball(0.5).density(1.0),
                        h,
                        &mut s.bodies,
                    );
                    handles.push(h);
                }
            }
        }
        for _ in 0..warmup {
            s.step();
        }
        let t0 = Instant::now();
        // maru 쪽과 **같은 창**으로 정착 구간을 따로 잰다(전체 평균은 앞쪽 정착 과정이 지배한다).
        let mut t_late = std::time::Duration::ZERO;
        let mut late_steps = 0usize;
        for si in 0..steps {
            let t_step = Instant::now();
            s.step();
            if si >= steps / 2 {
                t_late += t_step.elapsed();
                late_steps += 1;
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        eprintln!(
            "    [rapier pile] 정착구간({late_steps}스텝): {:.3} ms/step",
            t_late.as_secs_f64() * 1000.0 / late_steps.max(1) as f64
        );
        let (mut lowest, mut finite, mut asleep) = (f32::MAX, true, 0usize);
        for h in &handles {
            let b = &s.bodies[*h];
            lowest = lowest.min(b.translation().y);
            finite &= b.translation().y.is_finite();
            asleep += b.is_sleeping() as usize;
        }
        eprintln!("    [rapier pile] 수면 {asleep}/{}", handles.len());
        let stable = finite && lowest > -0.5;
        Result_ {
            ms_per_step: ms,
            stable,
            detail: format!("lowest_y={lowest:.2} (관통 없으면 > 0)  bodies={}", handles.len()),
        }
    }
}

/// 같은 시나리오를 여러 번 돌려 **최소값**을 취한다. 벽시계는 위로만 오염되므로
/// (스케줄러·터보·캐시 경합) 최소값이 처리량의 최소분산 추정이다. 평균은 노이즈를 삼킨다.
fn best(reps: usize, mut f: impl FnMut() -> Result_) -> Result_ {
    let mut best: Option<Result_> = None;
    for _ in 0..reps {
        let r = f();
        if best.as_ref().is_none_or(|b| r.ms_per_step < b.ms_per_step) {
            best = Some(r);
        }
    }
    best.unwrap()
}

fn row(name: &str, r: &Result_) {
    println!(
        "  {:<8} {:>9.3} ms/step   {}   {}",
        name,
        r.ms_per_step,
        if r.stable { "STABLE " } else { "UNSTABLE" },
        r.detail
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scenario = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(240);
    let warmup: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(60);
    let reps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("Maru(M2) vs Rapier3D 0.34 — dt=1/60, release(lto=thin, cgu=1), 단일 스레드");
    println!(
        "설정: Maru=기본({} 서브스텝·{}Hz 접촉) / Rapier=IntegrationParameters::default()",
        maru_dynamics::WorldParams::default().substeps,
        maru_dynamics::WorldParams::default().solver.contact_hertz
    );
    // **정정**: 이 하네스의 Rapier 설정은 CCD를 켜지 않고 QueryPipeline도 만들지 않으며
    // event handler도 `()`다. 이전 문구는 Rapier가 더 많은 일을 하는 것처럼 과장했다.
    println!("주의: 양쪽 모두 CCD·쿼리 파이프라인 **없이** 돈다. Maru는 M2(셰이프 4종·단일 스레드).");
    println!("주의: 이 벤치는 '정착 품질 + 처리량'을 **함께** 잰다 — 한쪽이 잠들고 다른 쪽이");
    println!("      깨어 있으면 알고리즘 처리량 격차가 아니라 정착 성공 여부를 재는 것이다.");
    println!("측정: 워밍업 {warmup} + {steps} 스텝, {reps}회 반복의 **최소값**. UNSTABLE이면 수치는 무의미하다.\n");

    for (mode, label) in [
        (Mode::Throughput, "처리량 (양쪽 수면 OFF — 같은 일을 하는 비교)"),
        (Mode::Settle, "정착 (수면 ON — 품질 + 총비용)"),
    ] {
        println!("\n══════ {label} ══════");
        if scenario == "all" || scenario == "pyramid" {
            for tiers in [10usize, 20, 30, 40] {
                println!("[피라미드 {tiers}단]");
                row("maru", &best(reps, || maru_side::pyramid(mode, tiers, steps, warmup)));
                row(
                    "rapier",
                    &best(reps, || rapier_side::pyramid(mode, tiers, steps, warmup)),
                );
            }
        }
        if scenario == "all" || scenario == "pile" {
            for n in [500usize, 1500, 3000] {
                println!("[구 더미 {n}개]");
                row("maru", &best(reps, || maru_side::pile(mode, n, steps, warmup)));
                row("rapier", &best(reps, || rapier_side::pile(mode, n, steps, warmup)));
            }
        }
    }
}
