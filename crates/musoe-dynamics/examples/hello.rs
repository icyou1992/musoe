//! 최소 예제 — README의 빠른 시작이 실제로 도는지 잠그는 자리다.
//! `cargo run -p musoe-dynamics --example hello`

use musoe_collide::{Pose, Shape};
use musoe_dynamics::{Body, World, WorldParams};
use musoe_math::{Quat, Vec3};

fn main() {
    let mut world = World::new(WorldParams::default());

    // 바닥 — static은 질량이 없다.
    world.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(10.0, 0.5, 10.0),
            radius: 0.0,
        },
        Pose {
            pos: Vec3::new(0.0, -0.5, 0.0),
            rot: Quat::IDENTITY,
        },
    ));

    // 떨어지는 상자 — 세 번째 인자는 질량이 아니라 **밀도**다.
    let cube = world.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        Pose {
            pos: Vec3::new(0.0, 4.0, 0.0),
            rot: Quat::IDENTITY,
        },
        1.0,
    ));

    for _ in 0..180 {
        world.step(1.0 / 60.0);
    }

    let b = world.body(cube);
    println!(
        "y = {:.4}, |v| = {:.4}, asleep = {}",
        b.pose().pos.y,
        b.linvel().length(),
        world.is_asleep(cube)
    );
}
