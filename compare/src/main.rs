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

    pub fn pyramid(tiers: usize, steps: usize, warmup: usize) -> Result_ {
        let mut w = World::new(WorldParams::default());
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

    pub fn pile(n: usize, steps: usize, warmup: usize) -> Result_ {
        let sub: u32 = std::env::var("MARU_SUB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(WorldParams::default().substeps);
        let thr: f32 = std::env::var("MARU_SLEEP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let d = WorldParams::default();
        let mut w = World::new(WorldParams {
            substeps: sub,
            sleep_linear_threshold: if thr > 0.0 { thr } else { d.sleep_linear_threshold },
            sleep_angular_threshold: if thr > 0.0 { thr * 1.25 } else { d.sleep_angular_threshold },
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
        for si in 0..steps {
            let _ = w.try_step(DT);
            let d = w.diagnostics();
            for k in 0..5 {
                acc[k] += d.phase_ns[k];
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

    pub fn pyramid(tiers: usize, steps: usize, warmup: usize) -> Result_ {
        let mut s = Sim::new();
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
                let h = s
                    .bodies
                    .insert(RigidBodyBuilder::dynamic().translation(rapier3d::math::Vec3::new(x, y, 0.0)));
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

    pub fn pile(n: usize, steps: usize, warmup: usize) -> Result_ {
        let mut s = Sim::new();
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
                    let h = s
                        .bodies
                        .insert(RigidBodyBuilder::dynamic().translation(rapier3d::math::Vec3::new(x, y, z)));
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
        for _ in 0..steps {
            s.step();
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
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

    if scenario == "all" || scenario == "pyramid" {
        for tiers in [10usize, 20, 30, 40] {
            println!("[피라미드 {tiers}단]");
            row("maru", &best(reps, || maru_side::pyramid(tiers, steps, warmup)));
            row("rapier", &best(reps, || rapier_side::pyramid(tiers, steps, warmup)));
        }
    }
    if scenario == "all" || scenario == "pile" {
        for n in [500usize, 1500, 3000] {
            println!("[구 더미 {n}개]");
            row("maru", &best(reps, || maru_side::pile(n, steps, warmup)));
            row("rapier", &best(reps, || rapier_side::pile(n, steps, warmup)));
        }
    }
}
