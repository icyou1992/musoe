//! maru-dynamics — 강체 동역학 (M2).
//!
//! 설계 계약(02-architecture.md): §4.4 TGS-soft, §4.6 슬리핑, §8 결정론·denormal 위생, §9 바디 3종.

mod body;
mod solver;
mod world;

pub use body::{Body, BodyHandle, BodyType};
pub use solver::SolverParams;
pub use world::{World, WorldParams};

#[cfg(test)]
mod tests {
    use super::*;
    use maru_collide::{Pose, Shape};
    use maru_math::{Quat, Vec3};

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

    /// 해석 검증: 자유낙하 v=gt, y=y0−gt²/2 (Symplectic Euler 오차 허용)
    #[test]
    fn free_fall_analytic() {
        let mut w = World::new(WorldParams::default());
        let ball = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(0.0, 100.0, 0.0),
            1.0,
        ));
        for _ in 0..60 {
            w.step(DT);
        }
        let b = w.body(ball);
        let t = 1.0;
        assert!((b.linvel.y + 9.81 * t).abs() < 0.01, "v={}", b.linvel.y);
        // Symplectic Euler는 반스텝 오차: |y − 해석| < g·dt·t
        let analytic = 100.0 - 0.5 * 9.81 * t * t;
        assert!(
            (b.pose.pos.y - analytic).abs() < 9.81 * DT * t + 0.01,
            "y={}",
            b.pose.pos.y
        );
    }

    /// 정착: 박스가 바닥에 얹혀 침투·점프 없이 정지 상태 유지
    #[test]
    fn box_rests_on_ground_without_sinking() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let bx = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.5, 0.0), // 정확히 얹힘
            1.0,
        ));
        for _ in 0..300 {
            w.step(DT);
        }
        let b = w.body(bx);
        assert!((b.pose.pos.y - 0.5).abs() < 0.01, "y={}", b.pose.pos.y);
        assert!(b.linvel.length() < 0.05, "v={:?}", b.linvel);
        assert!(b.angvel.length() < 0.05, "w={:?}", b.angvel);
    }

    /// 스택 5단 안정 (M2 피라미드 100단의 축소 선행 게이트)
    #[test]
    fn stack_of_5_boxes_stays_up() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let mut handles = Vec::new();
        for i in 0..5 {
            handles.push(w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.5, 0.5, 0.5),
                    radius: 0.0,
                },
                pose(0.0, 0.5 + i as f32 * 1.0, 0.0),
                1.0,
            )));
        }
        for _ in 0..600 {
            w.step(DT);
        }
        for (i, h) in handles.iter().enumerate() {
            let b = w.body(*h);
            let expect_y = 0.5 + i as f32 * 1.0;
            assert!(
                (b.pose.pos.y - expect_y).abs() < 0.05,
                "box {i}: y={} expect {expect_y}",
                b.pose.pos.y
            );
            assert!(
                b.pose.pos.x.abs() < 0.05 && b.pose.pos.z.abs() < 0.05,
                "box {i} drifted: {:?}",
                b.pose.pos
            );
        }
    }

    /// restitution: e=0.8 공 낙하 → 반발 apex ≈ e²·h₀ (임팩트 게이트 경유)
    #[test]
    fn restitution_bounce_apex() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let mut ball = Body::new_dynamic(Shape::Sphere { radius: 0.5 }, pose(0.0, 2.5, 0.0), 1.0);
        ball.restitution = 0.8;
        let h0 = 2.0; // 낙하 높이 (표면 기준)
        let ball = w.add_body(ball);
        let mut apex: f32 = 0.0;
        let mut bounced = false;
        for _ in 0..240 {
            w.step(DT);
            let b = w.body(ball);
            if b.linvel.y > 0.01 {
                bounced = true;
            }
            if bounced {
                apex = apex.max(b.pose.pos.y - 0.5);
            }
        }
        assert!(bounced, "반발 없음");
        let expect = 0.8 * 0.8 * h0;
        assert!(
            (apex - expect).abs() < 0.25 * expect,
            "apex={apex} expect≈{expect}"
        );
    }

    /// 마찰: 기울기 없는 바닥에서 수평 초기속도가 마찰로 감쇠
    #[test]
    fn friction_stops_sliding_box() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let mut bx = Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.5, 0.0),
            1.0,
        );
        bx.linvel = Vec3::new(3.0, 0.0, 0.0);
        let bx = w.add_body(bx);
        for _ in 0..300 {
            w.step(DT);
        }
        let b = w.body(bx);
        assert!(b.linvel.length() < 0.05, "미끄럼 지속: {:?}", b.linvel);
    }

    /// 슬리핑: 정착한 스택은 잠들고, 새 충돌체 도착 시 level-trigger로 깨어난다 (설계 §4.6)
    #[test]
    fn stack_sleeps_then_wakes_on_impact() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let base = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.5, 0.0),
            1.0,
        ));
        for _ in 0..180 {
            w.step(DT);
        }
        assert!(w.is_asleep(base), "3초 후에도 잠들지 않음");
        // 위에서 공 낙하 → 접촉 페어 생성 → wake
        let ball = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 3.0, 0.0),
            1.0,
        ));
        let mut woke = false;
        for _ in 0..120 {
            w.step(DT);
            if !w.is_asleep(base) {
                woke = true;
            }
        }
        assert!(woke, "충격에도 깨어나지 않음");
        let _ = ball;
    }

    /// 결정론 L0 스모크: 같은 씬 2회 실행 → 비트 동일
    #[test]
    fn deterministic_two_runs_bit_equal() {
        let run = || {
            let mut w = World::new(WorldParams::default());
            ground(&mut w);
            let mut hs = Vec::new();
            for i in 0..10 {
                hs.push(w.add_body(Body::new_dynamic(
                    Shape::Box {
                        half: Vec3::new(0.4, 0.4, 0.4),
                        radius: 0.0,
                    },
                    pose(
                        (i as f32) * 0.1 - 0.5,
                        1.0 + i as f32 * 0.9,
                        (i as f32) * 0.05,
                    ),
                    1.0,
                )));
            }
            for _ in 0..120 {
                w.step(DT);
            }
            hs.iter()
                .flat_map(|h| {
                    let b = w.body(*h);
                    [
                        b.pose.pos.x.to_bits(),
                        b.pose.pos.y.to_bits(),
                        b.pose.pos.z.to_bits(),
                    ]
                })
                .collect::<Vec<u32>>()
        };
        assert_eq!(run(), run());
    }
}
