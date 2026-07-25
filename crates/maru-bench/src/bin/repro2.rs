use maru_collide::{contact, gjk, GjkResult, Pose, Shape};
use maru_math::{Quat, Vec3};
fn main() {
    let s = Shape::Box {
        half: Vec3::new(0.5, 0.5, 0.5),
        radius: 0.0,
    };
    // 피라미드에서 포착된 나란한 두 박스 (정확히 맞닿음, 간격 4.8e-6)
    let pa = Pose {
        pos: Vec3::new(-4.500016, 68.47692, 2.6655919e-6),
        rot: Quat::IDENTITY,
    };
    let pb = Pose {
        pos: Vec3::new(-3.5000112, 68.47692, 4.9195756e-7),
        rot: Quat::IDENTITY,
    };
    match gjk(&s, &pa, &s, &pb) {
        GjkResult::Separated {
            distance,
            normal,
            on_a,
            on_b,
        } => {
            println!("GJK Separated d={distance:e} n={normal:?}");
            println!("   on_a={on_a:?} on_b={on_b:?}");
        }
        GjkResult::Intersecting(seed) => {
            let v: Vec<Vec3> = seed.iter().map(|p| p.w).collect();
            println!(
                "GJK Intersecting vol={:e}",
                (v[1] - v[0]).cross(v[2] - v[0]).dot(v[3] - v[0])
            );
        }
    }
    if let Some(m) = contact(&s, &pa, &s, &pb, 0.05) {
        println!("manifold n={:?} pts={}", m.normal, m.points.len());
        for p in &m.points {
            println!("   sep={:.6} pos={:?}", p.separation, p.position);
        }
    } else {
        println!("no contact");
    }
}
