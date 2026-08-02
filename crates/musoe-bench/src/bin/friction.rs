//! **마찰 품질 계측기** — 마찰 솔버를 바꾸기 전에 "지금 무엇을 지키고 있나"를 재는 자다.
//!
//! 마찰 2x2 블록화(두 접선 축을 동시에 푸는 것)는 **물리를 바꾼다**. 그래서 속도를 재기
//! 전에 품질 기준선이 있어야 한다. 이 벤치가 그 기준선을 숫자로 뽑고, 같은 항목이
//! `musoe-dynamics`의 테스트로 잠긴다.
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
//! 실행: cargo run --release -p musoe-bench --bin friction

use musoe_collide::{Pose, Shape};
use musoe_dynamics::{Body, BodyHandle, World, WorldParams};
use musoe_math::{Quat, Vec3};

const DT: f32 = 1.0 / 60.0;
const MU: f32 = 0.6; // Body 기본 마찰

fn params() -> WorldParams {
    WorldParams {
        // 수면이 켜져 있으면 "멈췄다"가 마찰 때문인지 잠들어서인지 갈리지 않는다.
        time_to_sleep: musoe_collide::MAX_INPUT,
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
    probe_extra();
    probe_energy();
}

// ── 검수 rev_ac00206f 보강분 계측 ───────────────────────────────────────────────
// 아래 셋은 "2x2 블록화에서 가장 위험한 축"인데 기존 4종이 못 잡던 것들이다:
// [5] 스핀 감쇠 **곡선**(상한만 보면 토크 2배 오류가 통과한다 — 양 끝이 필요)
// [6] 비등방 관성 + 한 점/모서리 접촉(교차항이 크고 조건수가 높은 영역)
// [7] 작은 스택의 **수면 도달**과 옆흐름(접선 잔차가 남으면 영영 안 잔다)

fn ground_flat(w: &mut World) {
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
}

fn probe_extra() {
    println!("\n[5] 스핀 감쇠 곡선 (5 rad/s 시작, 스텝별 angvel.y)");
    let mut w = World::new(params());
    ground_flat(&mut w);
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
    w.set_angular_velocity(h, Vec3::new(0.0, 5.0, 0.0)).unwrap();
    let mut prev = 5.0f32;
    let mut max_seen = 5.0f32;
    for i in 0..40 {
        w.step(DT);
        let wy = w.body(h).angvel().y;
        max_seen = max_seen.max(wy.abs());
        if i < 12 || i == 19 || i == 29 || i == 39 {
            println!(
                "  {:>2}스텝: {wy:>9.5} rad/s  (부호 {})",
                i + 1,
                if wy < -1e-4 { "역전!" } else { "유지" }
            );
        }
        prev = wy;
    }
    let _ = prev;
    println!("  최대 각속력(초기 포함) = {max_seen:.5} — 초기 5.0을 넘으면 에너지 주입이다");

    println!("\n[6] 비등방 관성 + 모서리 접촉 (길쭉한 막대 0.1x0.1x2.0, 15도 기울여 떨어뜨림)");
    let mut w = World::new(params());
    ground_flat(&mut w);
    let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 15f32.to_radians());
    let h = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.1, 0.1, 2.0),
            radius: 0.0,
        },
        Pose {
            pos: Vec3::new(0.0, 1.2, 0.0),
            rot,
        },
        1.0,
    ));
    w.set_linear_velocity(h, Vec3::new(2.5, 0.0, 1.5)).unwrap();
    let mut max_pen = 0.0f32;
    let mut max_v = 0.0f32;
    for _ in 0..400 {
        w.step(DT);
        max_pen = max_pen.max(w.diagnostics().max_penetration);
        max_v = max_v.max(w.body(h).linvel().length());
    }
    let b = w.body(h);
    println!(
        "  최종 pos={:?}\n  최종 |v|={:.5} |w|={:.5} · 최대 침투 {max_pen:.5} m · 최대 |v| {max_v:.4} (시작 2.92)",
        b.pose().pos, b.linvel().length(), b.angvel().length()
    );
    println!(
        "  finite = {}",
        b.pose().pos.y.is_finite() && b.linvel().length().is_finite()
    );

    println!("\n[7] 작은 스택(6단 피라미드)의 수면 도달 + 옆흐름");
    let mut w = World::new(WorldParams::default()); // 수면 ON — 이 항목은 수면이 본체다
    ground_flat(&mut w);
    let mut hs = Vec::new();
    let mut com0 = Vec3::ZERO;
    for tier in 0..6usize {
        let count = 6 - tier;
        for i in 0..count {
            let x = (i as f32 - (count as f32 - 1.0) * 0.5) * 1.02;
            hs.push(w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.5, 0.5, 0.5),
                    radius: 0.0,
                },
                Pose {
                    pos: Vec3::new(x, 0.5 + tier as f32, 0.0),
                    rot: Quat::IDENTITY,
                },
                1.0,
            )));
            com0 += Vec3::new(x, 0.5 + tier as f32, 0.0);
        }
    }
    com0 = com0 * (1.0 / hs.len() as f32);
    let mut all_asleep_at = None;
    let mut max_pen = 0.0f32;
    for i in 0..600 {
        w.step(DT);
        max_pen = max_pen.max(w.diagnostics().max_penetration);
        if all_asleep_at.is_none() && hs.iter().all(|&h| w.is_asleep(h)) {
            all_asleep_at = Some(i + 1);
        }
    }
    let com: Vec3 =
        hs.iter().fold(Vec3::ZERO, |a, &h| a + w.body(h).pose().pos) * (1.0 / hs.len() as f32);
    println!(
        "  바디 {} · 전원 수면 도달 = {:?} 스텝 (600 안) · 최대 침투 {max_pen:.5} m",
        hs.len(),
        all_asleep_at
    );
    println!(
        "  COM 옆흐름 dx={:.5} dz={:.5} · dy={:.5}",
        com.x - com0.x,
        com.z - com0.z,
        com.y - com0.y
    );
}

// ── 검수 rev_87748fe6: 마찰이 에너지를 주입하는가 ────────────────────────────────
// 반례(수학적으로 확인): 대각 질량 스텝 후 유클리드 원판 투영은 K가 등방일 때만 제약 해와
// 같다. 비등방 접촉에서는 원판 안의 이전 임펄스에서 **에너지를 늘리는** 갱신이 가능하다.
// 여기서는 그게 실제 장면에서 나는지, 그리고 **순차 GS 시절보다 나빠졌는지**를 잰다.
// 역학적 에너지 = 운동 + 위치. 중력은 보존력이라 둘의 합은 마찰·접촉이 없으면 보존된다.
fn probe_energy() {
    println!("\n[8] 마찰 에너지 주입 — 비등방 막대, 스텝당 역학적 에너지 증가분");
    let g = 9.81f32;
    for (label, half, spin) in [
        ("등방 정육면체", Vec3::new(0.5, 0.5, 0.5), 0.0f32),
        ("비등방 막대", Vec3::new(0.1, 0.1, 2.0), 0.0),
        ("비등방 막대+스핀", Vec3::new(0.1, 0.1, 2.0), 6.0),
    ] {
        let mut w = World::new(params());
        ground_flat(&mut w);
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 12f32.to_radians());
        let h = w.add_body(Body::new_dynamic(
            Shape::Box { half, radius: 0.0 },
            Pose {
                pos: Vec3::new(0.0, 1.1, 0.0),
                rot,
            },
            1.0,
        ));
        w.set_linear_velocity(h, Vec3::new(2.0, 0.0, 1.3)).unwrap();
        if spin != 0.0 {
            w.set_angular_velocity(h, Vec3::new(0.3, spin, 0.2))
                .unwrap();
        }
        // 관성 텐서를 모르므로 회전 에너지는 |w|^2 대용이 아니라 실제 관성으로 재야 정확하다.
        // 여기서는 **상대 비교**가 목적이라 균질 직육면체 관성으로 직접 계산한다.
        let (hx, hy, hz) = (half.x, half.y, half.z);
        let m = 1.0f32;
        let i = Vec3::new(
            m / 3.0 * (hy * hy + hz * hz),
            m / 3.0 * (hx * hx + hz * hz),
            m / 3.0 * (hx * hx + hy * hy),
        );
        let energy = |w: &World| {
            let b = w.body(h);
            let v = b.linvel();
            let om = b.angvel(); // 월드 각속도 — 근사(주축 정렬 가정)라 절대값이 아닌 추세를 본다
            0.5 * m * v.dot(v)
                + 0.5 * (i.x * om.x * om.x + i.y * om.y * om.y + i.z * om.z * om.z)
                + m * g * b.pose().pos.y
        };
        let mut prev = energy(&w);
        let (mut max_gain, mut at) = (0.0f32, 0usize);
        let mut total_gain = 0.0f32;
        for s in 0..400 {
            w.step(DT);
            let e = energy(&w);
            let d = e - prev;
            if d > max_gain {
                max_gain = d;
                at = s + 1;
            }
            if d > 0.0 {
                total_gain += d;
            }
            prev = e;
        }
        println!(
            "  {label:<16} 최대 스텝 증가 {max_gain:>9.5} J (step {at:>3}) · 양의 증가 합 {total_gain:>9.5} J"
        );
    }
}
