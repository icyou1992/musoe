//! maru-dynamics — 강체 동역학 (M2).
//!
//! 설계 계약(02-architecture.md): §4.4 TGS-soft, §4.6 슬리핑, §8 결정론·denormal 위생, §9 바디 3종.

mod body;
mod joint;
mod snapshot;
mod solver;
mod world;

pub use body::{Body, BodyHandle, BodyType};
pub use joint::{Joint, JointKind};
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

    /// point 조인트: 흔들리는 진자 — 제약 오차(앵커 간 거리)가 계속 작아야 함
    #[test]
    fn point_joint_pendulum_holds() {
        let mut w = World::new(WorldParams::default());
        let anchor = w.add_body(Body::new_static(
            Shape::Sphere { radius: 0.1 },
            pose(0.0, 5.0, 0.0),
        ));
        let bob = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.2, 0.2, 0.2),
                radius: 0.0,
            },
            pose(1.0, 5.0, 0.0), // 수평에서 놓아 흔들림 시작
            1.0,
        ));
        w.add_joint(Joint::point(
            anchor.index as usize,
            bob.index as usize,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0), // bob 로컬 (-1,0,0)이 앵커에 붙음
        ));
        let mut max_err: f32 = 0.0;
        for _ in 0..300 {
            w.step(DT);
            let b = w.body(bob);
            let attach = b.pose.pos + b.pose.rot.rotate(Vec3::new(-1.0, 0.0, 0.0));
            max_err = max_err.max((attach - Vec3::new(0.0, 5.0, 0.0)).length());
        }
        let b = w.body(bob);
        assert!(max_err < 0.03, "제약 오차 {max_err}");
        assert!(
            b.pose.pos.y < 5.0,
            "진자가 흔들려 내려와야 함: {:?}",
            b.pose.pos
        );
    }

    /// 리뷰 지적 검증: hinge 각 제약의 **부호**. cross product는 반평행에서도 0이라
    /// |cos|만 보면 부호를 판별할 수 없다 — 작은 초기 오차(20°)에서 시작해
    /// **오차가 줄어드는지**로 판별한다(부호가 반대면 커져서 반대축으로 넘어간다).
    #[test]
    fn hinge_reduces_small_axis_error() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..WorldParams::default()
        });
        let anchor = w.add_body(Body::new_static(
            Shape::Sphere { radius: 0.05 },
            pose(0.0, 0.0, 0.0),
        ));
        // B를 z축 기준 20° 돌려 놓고, hinge 축은 양쪽 다 +X로 지정
        let tilt = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 20f32.to_radians());
        let b = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.1, 0.1),
                radius: 0.0,
            },
            Pose {
                pos: Vec3::ZERO,
                rot: tilt,
            },
            1.0,
        ));
        w.add_joint(Joint::hinge(
            anchor.index as usize,
            b.index as usize,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ));
        let align = |w: &World| {
            w.body(b)
                .pose
                .rot
                .rotate(Vec3::new(1.0, 0.0, 0.0))
                .dot(Vec3::new(1.0, 0.0, 0.0))
        };
        let start = align(&w); // cos20° ≈ 0.94
        for _ in 0..120 {
            w.step(DT);
        }
        let end = align(&w);
        assert!(
            end > start && end > 0.999,
            "hinge 축 오차가 줄지 않음: {start} → {end} (부호 반대 의심)"
        );
    }

    /// M2 수용 기준: hinge 체인 20링크 + 접촉 혼합 수렴 (설계 §11 M2).
    /// 체인을 **바닥에 눕힌 상태**로 검증한다 — 전 링크가 동시에 접촉 제약과 조인트 제약을
    /// 받는 결정적 시나리오. (공중에서 떨어뜨리면 혼돈계라 더미 형태가 플랫폼마다 달라
    /// 판정 기준으로 쓸 수 없다 — 낙하 케이스는 아래 무발산 테스트가 담당.)
    #[test]
    fn hinge_chain_20_links_on_ground() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let anchor = w.add_body(Body::new_static(
            Shape::Sphere { radius: 0.05 },
            pose(0.0, 0.1, 0.0),
        ));
        let mut prev = anchor.index as usize;
        let mut links = Vec::new();
        for i in 0..20 {
            let x = 0.4 + i as f32 * 0.8;
            let h = w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.4, 0.1, 0.1),
                    radius: 0.0,
                },
                pose(x, 0.1, 0.0), // 바닥 위에 정확히 누움
                1.0,
            ));
            let ap = if i == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(0.4, 0.0, 0.0)
            };
            w.add_joint(Joint::hinge(
                prev,
                h.index as usize,
                ap,
                Vec3::new(-0.4, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ));
            links.push(h);
            prev = h.index as usize;
        }
        for _ in 0..480 {
            w.step(DT);
        }
        for (i, h) in links.iter().enumerate() {
            let b = w.body(*h);
            assert!(b.pose.pos.y.is_finite(), "link {i} 발산");
            assert!(b.linvel.length() < 0.05, "link {i} 미정착: {:?}", b.linvel);
            // 접촉이 실제로 유지되는가 (가라앉지도, 튀어오르지도 않음)
            assert!(
                (b.pose.pos.y - 0.1).abs() < 0.02,
                "link {i} 높이 이탈: {:?}",
                b.pose.pos
            );
        }
        // 조인트가 접촉과 동시에 유지되는가 (혼합 수렴의 핵심 판정)
        for i in 1..20 {
            let a = w.body(links[i - 1]);
            let b = w.body(links[i]);
            let pa = a.pose.pos + a.pose.rot.rotate(Vec3::new(0.4, 0.0, 0.0));
            let pb = b.pose.pos + b.pose.rot.rotate(Vec3::new(-0.4, 0.0, 0.0));
            assert!(
                (pa - pb).length() < 0.02,
                "joint {i} 벌어짐 {}",
                (pa - pb).length()
            );
        }
    }

    /// 낙하 체인(혼돈계): **무발산과 조인트 무결성만** 판정한다.
    /// 12초 내 완전 정착은 물리가 보장하는 성질이 아니다(접힌 체인은 계속 미끄러질 수 있음)
    /// — 정착 판정은 위 '바닥에 누운 체인'(결정적 시나리오)이 담당한다.
    #[test]
    fn hinge_chain_dropped_does_not_diverge() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let anchor = w.add_body(Body::new_static(
            Shape::Sphere { radius: 0.05 },
            pose(0.0, 3.0, 0.0),
        ));
        let mut prev = anchor.index as usize;
        let mut links = Vec::new();
        for i in 0..20 {
            let x = 0.4 + i as f32 * 0.8;
            let h = w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.4, 0.1, 0.1),
                    radius: 0.0,
                },
                pose(x, 3.0, 0.0),
                1.0,
            ));
            let ap = if i == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(0.4, 0.0, 0.0)
            };
            w.add_joint(Joint::hinge(
                prev,
                h.index as usize,
                ap,
                Vec3::new(-0.4, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ));
            links.push(h);
            prev = h.index as usize;
        }
        for _ in 0..720 {
            w.step(DT);
        }
        for (i, h) in links.iter().enumerate() {
            let b = w.body(*h);
            assert!(b.pose.pos.y.is_finite(), "link {i} 발산");
            // 에너지 폭주만 금지 (자유낙하 3m의 종단속도 ~7.7m/s를 크게 넘지 않아야 함)
            assert!(
                b.linvel.length() < 5.0,
                "link {i} 에너지 폭주: {:?}",
                b.linvel
            );
            assert!(b.pose.pos.y > -1.0, "link {i} 바닥 관통: {:?}", b.pose.pos);
        }
        for i in 1..20 {
            let a = w.body(links[i - 1]);
            let b = w.body(links[i]);
            let pa = a.pose.pos + a.pose.rot.rotate(Vec3::new(0.4, 0.0, 0.0));
            let pb = b.pose.pos + b.pose.rot.rotate(Vec3::new(-0.4, 0.0, 0.0));
            assert!((pa - pb).length() < 0.05, "joint {i} 벌어짐");
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

    fn assert_pyramid_stable(w: &World, hs: &[BodyHandle], tiers: usize) {
        for (idx, h) in hs.iter().enumerate() {
            let b = w.body(*h);
            assert!(b.pose.pos.y.is_finite(), "box {idx} 발산");
            assert!(b.linvel.length() < 0.2, "box {idx} 미정착: {:?}", b.linvel);
        }
        // 꼭대기 박스가 제 높이를 유지 (붕괴 판정)
        let top = w.body(*hs.last().unwrap());
        let expect_y = 0.5 + (tiers - 1) as f32;
        assert!(
            (top.pose.pos.y - expect_y).abs() < 0.15,
            "정상 y={} expect {expect_y} — 붕괴",
            top.pose.pos.y
        );
    }

    /// 피라미드 10단 (55박스) — CI 상시 게이트 (100단의 축소판)
    #[test]
    fn pyramid_10_tiers_stable() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let hs = pyramid(&mut w, 10);
        for _ in 0..600 {
            w.step(DT);
        }
        assert_pyramid_stable(&w, &hs, 10);
    }

    /// M2 수용 기준(2026-07-25 실측 재보정): 피라미드 40단(820박스) 안정.
    /// release 전용 — debug는 수 분 걸린다. `cargo test --release -- --ignored`
    /// 봉투 전체(20~100단 침하·ms/step)는 `maru-bench --bin pyramid100` + README 표.
    #[test]
    #[ignore = "release 전용: cargo test --release -p maru-dynamics -- --ignored"]
    fn pyramid_40_tiers_stable_acceptance() {
        let mut w = World::new(WorldParams::default());
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(60.0, 1.0, 20.0),
                radius: 0.0,
            },
            pose(0.0, -1.0, 0.0),
        ));
        let hs = pyramid(&mut w, 40);
        for _ in 0..600 {
            w.step(DT);
        }
        assert_pyramid_stable(&w, &hs, 40);
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
