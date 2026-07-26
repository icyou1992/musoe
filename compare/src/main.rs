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
        let mut acc = [0u64; 4];
        let mut pairs_acc = 0usize;
        for _ in 0..steps {
            let _ = w.try_step(DT);
            let d = w.diagnostics();
            for k in 0..4 {
                acc[k] += d.phase_ns[k];
            }
            pairs_acc += d.pairs;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        eprintln!(
            "    [maru 평균 us] broad={:.0} narrow={:.0} solve={:.0} book={:.0}  pairs={}",
            acc[0] as f64 / 1000.0 / steps as f64,
            acc[1] as f64 / 1000.0 / steps as f64,
            acc[2] as f64 / 1000.0 / steps as f64,
            acc[3] as f64 / 1000.0 / steps as f64,
            pairs_acc / steps
        );

        let (mut top, mut vmax, mut finite) = (f32::MIN, 0.0f32, true);
        for h in &handles {
            let b = w.body(*h);
            top = top.max(b.pose().pos.y);
            vmax = vmax.max(b.linvel().length());
            finite &= b.pose().pos.y.is_finite();
        }
        let expect = tiers as f32 - 0.5;
        let stable = finite && (top - expect).abs() < 0.02 * tiers as f32 && vmax < 0.5;
        Result_ {
            ms_per_step: ms,
            stable,
            detail: format!("top_y={top:.2} (expect {expect:.1})  vmax={vmax:.3}  bodies={}", handles.len()),
        }
    }

    pub fn pile(n: usize, steps: usize, warmup: usize) -> Result_ {
        let mut w = World::new(WorldParams::default());
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
        let t0 = Instant::now();
        for _ in 0..steps {
            let _ = w.try_step(DT);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        let (mut lowest, mut finite) = (f32::MAX, true);
        for h in &handles {
            let b = w.body(*h);
            lowest = lowest.min(b.pose().pos.y);
            finite &= b.pose().pos.y.is_finite();
        }
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
        let (mut top, mut vmax, mut finite) = (f32::MIN, 0.0f32, true);
        for h in &handles {
            let b = &s.bodies[*h];
            top = top.max(b.translation().y);
            vmax = vmax.max(b.linvel().length());
            finite &= b.translation().y.is_finite();
        }
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
        let (mut lowest, mut finite) = (f32::MAX, true);
        for h in &handles {
            let b = &s.bodies[*h];
            lowest = lowest.min(b.translation().y);
            finite &= b.translation().y.is_finite();
        }
        let stable = finite && lowest > -0.5;
        Result_ {
            ms_per_step: ms,
            stable,
            detail: format!("lowest_y={lowest:.2} (관통 없으면 > 0)  bodies={}", handles.len()),
        }
    }
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

    println!("Maru(M2) vs Rapier3D 0.34 — dt=1/60, release(lto=thin, cgu=1), 단일 스레드");
    println!("설정: Maru=기본(4 서브스텝·30Hz 접촉) / Rapier=IntegrationParameters::default()");
    println!("주의: Maru는 CCD 없음·셰이프 4종(M2). Rapier는 CCD·쿼리 파이프라인을 함께 돌린다.");
    println!("측정: 워밍업 {warmup} 스텝 후 {steps} 스텝 평균. UNSTABLE이면 성능 수치는 무의미하다.\n");

    if scenario == "all" || scenario == "pyramid" {
        for tiers in [10usize, 20, 30, 40] {
            println!("[피라미드 {tiers}단]");
            row("maru", &maru_side::pyramid(tiers, steps, warmup));
            row("rapier", &rapier_side::pyramid(tiers, steps, warmup));
        }
    }
    if scenario == "all" || scenario == "pile" {
        for n in [500usize, 1500, 3000] {
            println!("[구 더미 {n}개]");
            row("maru", &maru_side::pile(n, steps, warmup));
            row("rapier", &rapier_side::pile(n, steps, warmup));
        }
    }
}
