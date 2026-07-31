//! **마찰 품질 계측기** — 마찰 솔버를 바꾸기 전에 "지금 무엇을 지키고 있나"를 재는 자다.
//!
//! 마찰 2x2 블록화(두 접선 축을 동시에 푸는 것)는 **물리를 바꾼다**. 그래서 속도를 재기
//! 전에 품질 기준선이 있어야 한다. 이 벤치가 그 기준선을 숫자로 뽑고, 같은 항목이
//! `maru-dynamics`의 테스트로 잠긴다.
//!
//! 재는 것:
//! 1. **경사면 정지 각도** — 이론은 `tan(theta) <= mu`면 정지. mu=0.6이면 30.96도.
//!    임계 아래/위를 **양 끝** 관측해야 구간 불변식이 성립한다(한쪽만 보면 "항상 정지"
//!    하는 버그도 통과한다).
//! 2. **미끄러질 때의 가속도** — `a = g(sin θ − mu cos θ)`. 마찰의 **크기**를 잠근다.
//!    "멈추더라" 만으로는 mu를 10배로 키워도 통과한다.
//! 3. **접선면 등방성** — 지금 마찰은 축 0 → 축 1 **순차** Gauss-Seidel이라 원리적으로
//!    비등방이다(접선 기저를 어떻게 잡느냐가 결과를 바꾼다). 같은 속력을 여러 방향으로
//!    쏴서 이동거리가 방향에 따라 얼마나 갈리는지 잰다 — 2x2 블록화가 개선할 바로 그 축이다.
//! 4. **스핀 감쇠** — 제자리 회전이 마찰로 죽는가.
//!
//! 실행: cargo run --release -p maru-bench --bin friction

use maru_collide::{Pose, Shape};
use maru_dynamics::{Body, BodyHandle, World, WorldParams};
use maru_math::{Quat, Vec3};

const DT: f32 = 1.0 / 60.0;
const MU: f32 = 0.6; // Body 기본 마찰

fn params() -> WorldParams {
    WorldParams {
        // 수면이 켜져 있으면 "멈췄다"가 마찰 때문인지 잠들어서인지 갈리지 않는다.
        time_to_sleep: maru_collide::MAX_INPUT,
        ..WorldParams::default()
    }
}

/// 각도 `theta`(rad)로 Z축 둘레로 기울인 바닥 + 그 위에 같은 자세로 놓인 상자.
/// 상자의 초기 자세를 바닥과 맞춰야 "면 접촉"으로 시작한다(모서리로 서면 다른 실험이 된다).
fn slope_world(theta: f32) -> (World, BodyHandle) {
    let mut w = World::new(params());
    let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), theta);
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(50.0, 1.0, 50.0),
            radius: 0.0,
        },
        Pose {
            pos: Vec3::ZERO,
            rot,
        },
    ));
    // 바닥 윗면은 로컬 +y=1 이므로, 경사면 위 1 + 0.5 만큼 법선 방향으로 띄운다.
    let n = rot.rotate(Vec3::new(0.0, 1.0, 0.0));
    let h = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        Pose { pos: n * 1.5, rot },
        1.0,
    ));
    (w, h)
}

/// 경사 방향(내리막) 단위벡터 — 이동거리를 이 축에 투영해 잰다.
fn downhill(theta: f32) -> Vec3 {
    let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), theta);
    rot.rotate(Vec3::new(-1.0, 0.0, 0.0))
}

fn slope_run(theta_deg: f32, steps: usize) -> (f32, f32) {
    let theta = theta_deg.to_radians();
    let (mut w, h) = slope_world(theta);
    let p0 = w.body(h).pose().pos;
    for _ in 0..steps {
        w.step(DT);
    }
    let b = w.body(h);
    let d = (b.pose().pos - p0).dot(downhill(theta));
    (d, b.linvel().dot(downhill(theta)))
}

fn main() {
    let crit = MU.atan().to_degrees();
    println!("마찰 mu={MU} · 이론 임계각 atan(mu) = {crit:.2}도 · 수면 OFF");
    println!("\n[1] 경사면 정지/미끄럼 (3초=180스텝, 내리막 변위 m)");
    println!(
        "{:>7} {:>10} {:>12} {:>10}",
        "각도", "변위", "내리막속도", "판정"
    );
    for d in [5.0f32, 15.0, 25.0, 29.0, 30.5, 31.5, 33.0, 37.0, 45.0] {
        let (disp, v) = slope_run(d, 180);
        let verdict = if disp.abs() < 0.01 {
            "정지"
        } else {
            "미끄럼"
        };
        println!("{d:>6.1}도 {disp:>10.4} {v:>12.4} {verdict:>10}");
    }

    println!("\n[2] 미끄러질 때 가속도 (이론 a = g(sinθ − mu·cosθ), 1초 뒤 속도로 역산)");
    println!(
        "{:>7} {:>12} {:>12} {:>9}",
        "각도", "이론 a", "실측 a", "비율"
    );
    for d in [37.0f32, 40.0, 45.0, 55.0] {
        let th = d.to_radians();
        let a_th = 9.81 * (th.sin() - MU * th.cos());
        // 1초(60스텝) 뒤 내리막 속도 / 1초 = 평균 가속도. 초기 접촉 안정화 20스텝은 버린다.
        let (_, v0) = slope_run(d, 20);
        let (_, v1) = slope_run(d, 80);
        let a_meas = (v1 - v0) / (60.0 * DT);
        println!(
            "{d:>6.1}도 {a_th:>12.4} {a_meas:>12.4} {:>9.3}",
            a_meas / a_th
        );
    }

    println!("\n[3] 접선면 등방성 — 같은 속력을 여러 방향으로 쏘고 이동거리 비교");
    let mut dists = Vec::new();
    for k in 0..8 {
        let ang = k as f32 * std::f32::consts::TAU / 8.0;
        let mut w = World::new(params());
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(50.0, 1.0, 50.0),
                radius: 0.0,
            },
            Pose {
                pos: Vec3::new(0.0, -1.0, 0.0),
                rot: Quat::IDENTITY,
            },
        ));
        let h = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            Pose {
                pos: Vec3::new(0.0, 0.5, 0.0),
                rot: Quat::IDENTITY,
            },
            1.0,
        ));
        w.set_linear_velocity(h, Vec3::new(ang.cos() * 3.0, 0.0, ang.sin() * 3.0))
            .expect("속도 설정");
        for _ in 0..300 {
            w.step(DT);
        }
        let d = (w.body(h).pose().pos - Vec3::new(0.0, 0.5, 0.0)).length();
        println!("  방향 {:>5.1}도 → 이동 {d:.4} m", ang.to_degrees());
        dists.push(d);
    }
    let (dmin, dmax) = (
        dists.iter().cloned().fold(f32::MAX, f32::min),
        dists.iter().cloned().fold(0.0f32, f32::max),
    );
    println!(
        "  → 최소 {dmin:.4} / 최대 {dmax:.4} · **비등방 폭 {:.2}%** (0%가 이상적)",
        100.0 * (dmax / dmin - 1.0)
    );

    println!("\n[4] 스핀 감쇠 — 제자리에서 y축으로 5 rad/s 회전 시작");
    let mut w = World::new(params());
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(50.0, 1.0, 50.0),
            radius: 0.0,
        },
        Pose {
            pos: Vec3::new(0.0, -1.0, 0.0),
            rot: Quat::IDENTITY,
        },
    ));
    let h = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        Pose {
            pos: Vec3::new(0.0, 0.5, 0.0),
            rot: Quat::IDENTITY,
        },
        1.0,
    ));
    w.set_angular_velocity(h, Vec3::new(0.0, 5.0, 0.0))
        .expect("각속도 설정");
    let mut marks = Vec::new();
    for i in 0..300 {
        w.step(DT);
        if [29, 59, 119, 299].contains(&i) {
            marks.push((i + 1, w.body(h).angvel().y));
        }
    }
    for (i, wy) in marks {
        println!("  {i:>3}스텝 후 angvel.y = {wy:.4} rad/s");
    }
}
