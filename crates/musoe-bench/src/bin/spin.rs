//! 회전이 빠른 **이방성** 바디의 접촉 — stale effective mass가 실제로 무는지 재는 장면.
//!
//! `prepare`가 프레임 시작 자세로 유효질량을 구하고 솔버가 서브스텝마다 갱신된
//! `inv_i`로 임펄스를 적용하면 K⁻¹와 K가 어긋난다. 쌓인 박스는 프레임당 회전이 거의
//! 없어 이 오차가 안 드러난다 — 그래서 **프레임당 크게 도는** 가늘고 긴 박스를 만든다.
//! 각속도 50 rad/s면 dt=1/60에 0.83 rad(48°) 돌아 프레임 시작 K가 크게 틀린다.
//!
//! 판정용 출력: 유한성 · 최대 침투 · 총 에너지 궤적 · 최종 각속도.
//! 실행: cargo run --release -p musoe-bench --bin spin -- [steps] [omega]

use musoe_collide::{Pose, Shape};
use musoe_dynamics::{Body, BodyHandle, World, WorldParams};
use musoe_math::{Quat, Vec3};

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);
    let omega: f32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50.0);

    let d = WorldParams::default();
    let mut w = World::new(WorldParams {
        // 잠들면 회전이 멈춰 관측이 끝난다 — 수면 OFF.
        sleep_linear_threshold: 0.0,
        sleep_angular_threshold: 0.0,
        ..d
    });
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(40.0, 1.0, 40.0),
            radius: 0.0,
        },
        Pose {
            pos: Vec3::new(0.0, -1.0, 0.0),
            rot: Quat::IDENTITY,
        },
    ));

    // 가늘고 긴 박스 = 주관성 모멘트가 축마다 크게 다르다(이방성).
    // 주축이 아닌 축으로 돌려 자세가 프레임마다 크게 바뀌게 한다.
    let mut handles = Vec::new();
    for i in 0..24 {
        let x = (i % 6) as f32 * 3.0 - 7.5;
        let z = (i / 6) as f32 * 3.0 - 4.5;
        let h = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.18, 1.6, 0.32),
                radius: 0.0,
            },
            Pose {
                pos: Vec3::new(x, 0.9, z),
                rot: Quat::from_axis_angle(Vec3::new(1.0, 0.3, 0.2), 0.4),
            },
            1.0,
        ));
        // 주축에서 벗어난 축으로 회전 — 자유세차까지 겹쳐 자세 변화가 커진다.
        w.set_angular_velocity(
            h,
            Vec3::new(omega * 0.6, omega, omega * 0.35) * (if i % 2 == 0 { 1.0 } else { -1.0 }),
        )
        .expect("살아있는 핸들");
        handles.push(h);
    }

    let energy = |w: &World, hs: &[BodyHandle]| -> f64 {
        let mut e = 0.0f64;
        for h in hs {
            let b = w.body(*h);
            let m = 1.0 / b.inv_mass() as f64;
            e += 0.5 * m * b.linvel().length_sq() as f64;
            let ii = b.inv_inertia_local();
            e += 0.5 * (1.0 / ii.x as f64) * b.angvel().length_sq() as f64;
            e += m * 9.81 * b.pose().pos.y as f64;
        }
        e
    };

    let e0 = energy(&w, &handles);
    let mut max_pen = 0.0f32;
    let mut worst_ang = 0.0f32;
    for i in 0..steps {
        let _ = w.try_step(1.0 / 60.0);
        max_pen = max_pen.max(w.diagnostics().max_penetration);
        if i % 150 == 149 {
            println!(
                "  step {:4}  E={:12.1}  (E/E0={:.4})  max_pen={:.5}",
                i + 1,
                energy(&w, &handles),
                energy(&w, &handles) / e0,
                max_pen
            );
        }
    }
    let mut finite = true;
    let mut vmax = 0.0f32;
    for h in &handles {
        let b = w.body(*h);
        finite &= b.pose().pos.y.is_finite() && b.linvel().length().is_finite();
        vmax = vmax.max(b.linvel().length());
        worst_ang = worst_ang.max(b.angvel().length());
    }
    let e1 = energy(&w, &handles);
    println!(
        "omega={omega} steps={steps}  finite={finite}  E0={e0:.1} E1={e1:.1} (E1/E0={:.4})\n\
         max_pen={max_pen:.5}  vmax={vmax:.3}  angmax={worst_ang:.3}  vclamp={}  sat={}",
        e1 / e0,
        w.diagnostics().velocity_clamped,
        w.diagnostics().state_saturations
    );
}
