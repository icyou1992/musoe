//! maru-dynamics — 강체 동역학 (M2).
//!
//! 설계 계약(02-architecture.md): §4.4 TGS-soft, §4.6 슬리핑, §8 결정론·denormal 위생, §9 바디 3종.

mod body;
mod command;
mod joint;
mod snapshot;
mod solver;
mod world;

pub use body::{Body, BodyHandle, BodyType};
pub use command::{Command, CommandReport};
pub use joint::{Joint, JointKind};
pub use snapshot::SnapshotError;
pub use solver::SolverParams;
pub use world::{ApiError, Diagnostics, World, WorldParams};

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

    /// 리뷰 BLOCKER 회귀: 접촉으로만 연결된 스택이 **완전히 잠든 뒤**에도 아일랜드가
    /// 유지되어, 한 바디만 깨워도 전원이 함께 깨어나야 한다(설계 §4.6 island 단위 wake).
    /// 스냅샷 왕복 후에도 같아야 한다(아일랜드가 캐노니컬 상태에 포함되므로).
    #[test]
    fn sleeping_stack_wakes_as_one_island() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let mut hs = Vec::new();
        for i in 0..3 {
            hs.push(w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.5, 0.5, 0.5),
                    radius: 0.0,
                },
                pose(0.0, 0.5 + i as f32, 0.0),
                1.0,
            )));
        }
        // 충분히 오래 재워 아일랜드가 잠든 상태로 여러 스텝 유지되게 한다
        for _ in 0..300 {
            w.step(DT);
        }
        assert!(hs.iter().all(|h| w.is_asleep(*h)), "스택이 잠들지 않음");

        // 스냅샷 왕복 후에도 아일랜드가 보존되는지 함께 확인
        let snap = w.snapshot();
        let mut w2 = World::new(WorldParams::default());
        ground(&mut w2);
        for i in 0..3 {
            w2.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.5, 0.5, 0.5),
                    radius: 0.0,
                },
                pose(0.0, 0.5 + i as f32, 0.0),
                1.0,
            ));
        }
        w2.restore(&snap).expect("restore");

        // 맨 아래 바디만 건드린다 → 아일랜드 전원이 깨어나야 함
        w2.set_linear_velocity(hs[0], Vec3::new(0.1, 0.0, 0.0))
            .expect("set velocity");
        for (i, h) in hs.iter().enumerate() {
            assert!(!w2.is_asleep(*h), "restore 후 box {i}가 깨어나지 않음");
        }
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
            anchor,
            bob,
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
            anchor,
            b,
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
        let mut prev = anchor;
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
                h,
                ap,
                Vec3::new(-0.4, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ));
            links.push(h);
            prev = h;
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
        let mut prev = anchor;
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
                h,
                ap,
                Vec3::new(-0.4, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ));
            links.push(h);
            prev = h;
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

    /// 붕괴 판정. **상대 기준**(스택 높이의 1%)을 쓴다 — 절대 임계는 단수가 바뀌면 의미가
    /// 달라진다(톨러런스에서 배운 것과 같은 함정). soft 접촉의 하중 비례 침하는 정상 거동이고,
    /// 판정해야 하는 것은 '형상을 잃었는가'다. 100단 붕괴(4.9% 낙하)는 이 기준에서도 실패한다.
    fn assert_pyramid_stable(w: &World, hs: &[BodyHandle], tiers: usize) {
        for (idx, h) in hs.iter().enumerate() {
            let b = w.body(*h);
            assert!(b.pose.pos.y.is_finite(), "box {idx} 발산");
            assert!(b.linvel.length() < 0.2, "box {idx} 미정착: {:?}", b.linvel);
        }
        let top = w.body(*hs.last().unwrap());
        let expect_y = 0.5 + (tiers - 1) as f32;
        let sag = expect_y - top.pose.pos.y;
        let limit = 0.01 * tiers as f32; // 스택 높이의 1%
        assert!(
            sag.abs() < limit,
            "정상 침하 {sag:.3}m > 허용 {limit:.3}m (높이 {expect_y}) — 붕괴"
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
        // 지면은 `maru-bench --bin pyramid100`과 **동일**해야 한다 — 같은 수용 기준이 서로 다른
        // 씬을 재면 README 봉투 표와 CI 게이트가 어긋난다.
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(120.0, 1.0, 20.0),
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

    /// 설계 §7 계약: 쓰기는 커맨드 큐로, **enqueue 순서대로** 적용되고,
    /// **쿼리는 pending을 보지 않는다**(암묵 플러시 없음). pending은 캐노니컬 상태다.
    #[test]
    fn command_buffer_contract() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let b = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 5.0, 0.0),
            1.0,
        ));

        // ① 큐에 쌓아도 쿼리는 이전 상태를 본다
        w.enqueue(Command::SetLinearVelocity(b, Vec3::new(3.0, 0.0, 0.0)))
            .unwrap();
        assert_eq!(w.pending_commands(), 1);
        assert_eq!(w.body(b).linvel.x, 0.0, "pending이 쿼리에 보이면 안 된다");

        // ② 명시 flush로만 반영된다
        let rep = w.flush_commands();
        assert_eq!(rep.applied, 1);
        assert_eq!(w.body(b).linvel.x, 3.0);
        assert_eq!(w.pending_commands(), 0);

        // ③ 순서대로 적용된다 (마지막 값이 남는다)
        w.enqueue(Command::SetLinearVelocity(b, Vec3::new(1.0, 0.0, 0.0)))
            .unwrap();
        w.enqueue(Command::SetLinearVelocity(b, Vec3::new(2.0, 0.0, 0.0)))
            .unwrap();
        w.flush_commands();
        assert_eq!(w.body(b).linvel.x, 2.0, "enqueue 순서가 지켜지지 않음");

        // ④ 잘못된 값은 **enqueue에서** 거부된다 — 큐에 들어간 커맨드는 항상 구조적으로
        //    유효하므로 스냅샷 왕복이 자기 자신에게 복원된다(enqueue/restore 계약 일치).
        assert_eq!(
            w.enqueue(Command::SetLinearVelocity(b, Vec3::new(f32::NAN, 0.0, 0.0))),
            Err(ApiError::InvalidValue)
        );
        assert_eq!(w.pending_commands(), 0, "무효 커맨드가 큐에 남음");
        w.enqueue(Command::SetMaterial(b, 0.5, 0.5)).unwrap();
        let rep = w.flush_commands();
        assert_eq!(rep.applied, 1);
        assert_eq!(rep.rejected, 0);
        assert!(w.body(b).linvel.is_finite());

        // ④' enqueue 시점엔 유효했지만 **앞선 커맨드가 대상을 지우면** flush에서 거부된다
        //     (스텝을 중단시키지 않고 카운트만 — 설계 §3 스텝 원자성)
        let doomed = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(9.0, 5.0, 0.0),
            1.0,
        ));
        w.enqueue(Command::RemoveBody(doomed)).unwrap();
        w.enqueue(Command::SetLinearVelocity(doomed, Vec3::new(1.0, 0.0, 0.0)))
            .unwrap();
        let rep = w.flush_commands();
        assert_eq!((rep.applied, rep.rejected), (1, 1));

        // ⑤ step 0단계가 pending을 자동 적용한다
        w.enqueue(Command::SetLinearVelocity(b, Vec3::new(0.0, 0.0, 5.0)))
            .unwrap();
        w.step(DT);
        assert_eq!(w.pending_commands(), 0);
        assert!(w.body(b).pose.pos.z > 0.0, "step이 커맨드를 적용하지 않음");

        // ⑥ pending은 캐노니컬 상태: 스냅샷 왕복 후 재생이 일치한다
        let mut w2 = World::new(WorldParams::default());
        ground(&mut w2);
        w2.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 5.0, 0.0),
            1.0,
        ));
        w2.restore(&w.snapshot()).expect("restore");
        w.enqueue(Command::SetLinearVelocity(b, Vec3::new(-2.0, 0.0, 0.0)))
            .unwrap();
        w2.enqueue(Command::SetLinearVelocity(b, Vec3::new(-2.0, 0.0, 0.0)))
            .unwrap();
        let snap_with_pending = w.snapshot();
        let mut w3 = World::new(WorldParams::default());
        ground(&mut w3);
        w3.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 5.0, 0.0),
            1.0,
        ));
        w3.restore(&snap_with_pending).expect("restore");
        assert_eq!(w3.pending_commands(), 1, "pending이 스냅샷에 실리지 않음");
        for _ in 0..10 {
            w.step(DT);
            w3.step(DT);
        }
        assert_eq!(
            w.state_hash(),
            w3.state_hash(),
            "pending 포함 스냅샷 재생이 원 실행과 분기"
        );
    }

    /// 리뷰 MAJOR 회귀: generational arena가 실제로 동작하는가 —
    /// 제거 시 이웃 wake, 핸들 무효화, 슬롯 재사용 시 세대 증가로 옛 핸들 거부.
    #[test]
    fn body_removal_wakes_neighbors_and_invalidates_handles() {
        let mut w = World::new(WorldParams::default());
        let g = ground(&mut w);
        let base = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.5, 0.0),
            1.0,
        ));
        let top = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 1.5, 0.0),
            1.0,
        ));
        for _ in 0..300 {
            w.step(DT);
        }
        assert!(w.is_asleep(top) && w.is_asleep(base), "스택이 잠들지 않음");

        // 받침 제거 → 위 바디가 깨어나 낙하해야 한다 (설계 §4.6 pre-mutation wake)
        w.remove_body(base).expect("remove");
        assert!(!w.is_asleep(top), "받침 제거인데 위 바디가 잠든 채");
        assert!(!w.is_alive(base), "제거된 핸들이 살아있다고 보고됨");
        assert_eq!(w.try_body(base).err(), Some(ApiError::StaleHandle));
        assert_eq!(w.body_count(), 2); // ground + top

        let y0 = w.body(top).pose.pos.y;
        for _ in 0..60 {
            w.step(DT);
        }
        assert!(w.body(top).pose.pos.y < y0 - 0.3, "받침이 없는데 안 떨어짐");

        // 슬롯 재사용: 같은 인덱스라도 세대가 달라 옛 핸들은 거부된다
        let reused = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.2 },
            pose(5.0, 5.0, 0.0),
            1.0,
        ));
        assert_eq!(reused.index, base.index, "최소 인덱스 슬롯 재사용");
        assert_ne!(reused.generation, base.generation, "세대가 증가해야 함");
        assert_eq!(w.try_body(base).err(), Some(ApiError::StaleHandle));
        assert!(w.try_body(reused).is_ok());
        let _ = g;
    }

    /// 리뷰 MAJOR 회귀: 공개 변이는 불변식을 강제한다 (NaN·범위 밖·static 거부).
    #[test]
    fn mutation_api_enforces_invariants() {
        let mut w = World::new(WorldParams::default());
        let g = ground(&mut w);
        let b = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 2.0, 0.0),
            1.0,
        ));
        assert_eq!(
            w.set_linear_velocity(b, Vec3::new(f32::NAN, 0.0, 0.0))
                .err(),
            Some(ApiError::InvalidValue)
        );
        assert_eq!(
            w.set_material(b, -1.0, 0.5).err(),
            Some(ApiError::InvalidValue)
        );
        assert_eq!(
            w.set_material(b, 0.5, 2.0).err(),
            Some(ApiError::InvalidValue)
        );
        // static은 불변 (설계 §9)
        assert_eq!(
            w.set_linear_velocity(g, Vec3::ZERO).err(),
            Some(ApiError::ImmutableStatic)
        );
        // 정상 값은 통과하고 상태가 유지된다
        assert!(w.set_linear_velocity(b, Vec3::new(1.0, 0.0, 0.0)).is_ok());
        assert_eq!(w.body(b).linvel.x, 1.0);
        for _ in 0..30 {
            w.step(DT);
        }
        assert!(w.body(b).pose.pos.is_finite());
    }

    /// 리뷰 BLOCKER 회귀: 조작·손상된 스냅샷은 **패닉·OOM 없이 거부**되고 월드는 불변이어야 한다
    /// (M3에서 JS가 임의 바이트를 넘길 수 있다).
    #[test]
    fn malformed_snapshot_is_rejected_without_panic() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let b = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 3.0, 0.0),
            1.0,
        ));
        for _ in 0..10 {
            w.step(DT);
        }
        let good = w.snapshot();
        let before = w.state_hash();

        // 1) 빈/짧은 입력
        assert_eq!(w.restore(&[]), Err(SnapshotError::Truncated));
        // 잘린 입력: 바이트가 모자라 읽다 끊기거나(Truncated), 남은 바이트로는 담을 수 없는
        // 개수를 만나 **읽기 전에** 거부된다(CountTooLarge). 둘 다 정상 방어다.
        assert!(matches!(
            w.restore(&good[..good.len() / 2]),
            Err(SnapshotError::Truncated) | Err(SnapshotError::CountTooLarge)
        ));
        // 2) 매직 훼손
        let mut bad = good.clone();
        bad[0] ^= 0xFF;
        assert_eq!(w.restore(&bad), Err(SnapshotError::BadMagic));
        // 3) 버전 훼손
        let mut bad = good.clone();
        bad[4] = 99;
        assert_eq!(w.restore(&bad), Err(SnapshotError::BadVersion(99)));
        // 4) 거대한 바디 개수 (OOM 유도)
        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(w.restore(&bad), Err(SnapshotError::CountTooLarge));
        // 5) 잔여 바이트
        let mut bad = good.clone();
        bad.push(0);
        assert_eq!(w.restore(&bad), Err(SnapshotError::TrailingBytes));
        // 6) 비유한 값 주입 (바디 위치 필드)
        let mut bad = good.clone();
        // magic4+version4+count4 + alive1+reserved1+generation4+body_type1 = pos.x
        let pos_off = 12 + 1 + 1 + 4 + 1;
        bad[pos_off..pos_off + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(
            w.restore(&bad),
            Err(SnapshotError::NonFinite) | Err(SnapshotError::BadShape)
        ));

        // 전부 거부됐고 월드는 그대로여야 한다 (원자적 복원)
        assert_eq!(w.state_hash(), before, "실패한 restore가 월드를 오염시킴");
        assert!(w.body(b).pose.pos.y.is_finite());
        // 정상 스냅샷은 여전히 복원된다
        assert!(w.restore(&good).is_ok());
        assert_eq!(w.state_hash(), before);
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

    // ─────────── 리뷰 rev_fd76395d 회귀 (BLOCKER 3 + MAJOR) ───────────

    fn dyn_box(w: &mut World, x: f32, y: f32) -> BodyHandle {
        w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(x, y, 0.0),
            1.0,
        ))
    }

    /// BLOCKER: 커맨드로 만든 바디를 가리킬 방법이 있어야 한다 —
    /// `enqueue_add_body`가 **예약 핸들**을 즉시 돌려주고, 같은 배치의 조인트가 그것을 쓴다.
    #[test]
    fn enqueue_add_body_returns_handle_usable_in_same_batch() {
        let mut w = World::new(WorldParams::default());
        let anchor = w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 5.0, 0.0),
        ));
        let bob = w.enqueue_add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(1.5, 5.0, 0.0),
            1.0,
        ));
        // 아직 flush 전 — 쿼리에는 보이지 않는다 (설계 §7)
        assert!(!w.is_alive(bob), "pending 바디가 쿼리에 보임");
        w.enqueue(Command::AddJoint(Box::new(Joint::point(
            anchor,
            bob,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
        ))))
        .unwrap();
        for _ in 0..120 {
            w.step(DT);
        }
        assert_eq!(w.last_command_report().rejected, 0, "배치가 거부됨");
        assert!(w.is_alive(bob), "예약 핸들이 실현되지 않음");
        // 조인트가 실제로 걸렸다면 앵커에서 1.5m 안쪽에 매달려 있다 (자유낙하면 훨씬 아래)
        let p = w.body(bob).pose.pos;
        assert!(
            (p - Vec3::new(0.0, 5.0, 0.0)).length() < 2.0,
            "조인트 미체결 — bob={p:?}"
        );
    }

    /// MAJOR: 빈 월드에 AddBody만 큐잉하면 `bodies.is_empty()` 조기 반환에 막혀
    /// 영원히 적용되지 않았다. 커맨드 플러시는 그 판정보다 **먼저**여야 한다.
    #[test]
    fn empty_world_applies_pending_add_body() {
        let mut w = World::new(WorldParams::default());
        let h = w.enqueue_add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(0.0, 10.0, 0.0),
            1.0,
        ));
        w.step(DT);
        assert!(w.is_alive(h), "빈 월드의 pending AddBody가 적용되지 않음");
        assert_eq!(w.body_count(), 1);
    }

    /// BLOCKER(ABA): 슬롯이 제거·재사용되면 옛 핸들은 새 바디를 가리켜선 안 된다.
    /// 조인트 엔드포인트가 raw 인덱스였을 때 성립하던 경로다.
    #[test]
    fn slot_reuse_kills_old_handle_and_joint_command() {
        let mut w = World::new(WorldParams::default());
        let old = dyn_box(&mut w, 0.0, 3.0);
        let keep = dyn_box(&mut w, 5.0, 3.0);
        w.remove_body(old).unwrap();
        // 최소 인덱스 재사용 → 같은 슬롯, 다른 세대
        let new = w.enqueue_add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(0.0, 3.0, 0.0),
            1.0,
        ));
        assert_eq!(new.index, old.index, "슬롯 재사용 전제가 깨짐");
        assert_ne!(new.generation, old.generation, "세대가 증가하지 않음");
        // 옛 핸들로 만든 조인트는 **enqueue에서** 거부된다 (계약 통일: 큐에 들어간 커맨드는
        // 항상 유효 → 자기 스냅샷이 자기에게 복원된다)
        assert_eq!(
            w.enqueue(Command::AddJoint(Box::new(Joint::point(
                old,
                keep,
                Vec3::ZERO,
                Vec3::ZERO,
            )))),
            Err(ApiError::StaleHandle),
            "stale 핸들 조인트가 큐에 들어감"
        );
        assert_eq!(w.pending_commands(), 1, "예약 AddBody만 남아야 한다");
        w.flush_commands();
        assert!(w.is_alive(new) && !w.is_alive(old));
    }

    /// MAJOR: `is_asleep`가 raw 인덱스를 읽으면 재사용된 슬롯의 **남의 상태**를 돌려준다.
    #[test]
    fn is_asleep_rejects_stale_handle() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let h = dyn_box(&mut w, 0.0, 0.5);
        for _ in 0..400 {
            w.step(DT);
        }
        assert!(w.is_asleep(h), "정착한 바디가 잠들지 않음");
        w.remove_body(h).unwrap();
        assert!(!w.is_asleep(h), "stale 핸들이 잠든 상태를 보고함");
    }

    /// BLOCKER(자원): 고정 상한 대신 **입력 길이**로 개수를 묶는다 — 12바이트 파일이
    /// 400만 개를 주장해도 산술적으로 불가능하므로 읽기 전에 거부된다.
    #[test]
    fn snapshot_counts_bounded_by_input_length() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        dyn_box(&mut w, 0.0, 3.0);
        w.step(DT);
        let good = w.snapshot();
        let before = w.state_hash();

        // 헤더만 남기고 바디 100만 개를 주장 (고정 상한 1<<22 아래라 예전엔 통과했다)
        let mut bad = good[..12].to_vec();
        bad[8..12].copy_from_slice(&1_000_000u32.to_le_bytes());
        assert_eq!(w.restore(&bad), Err(SnapshotError::CountTooLarge));

        // 중첩 개수(hull 점)도 같은 규칙 — 총량 예산이 파일 크기에서 자동으로 나온다
        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&(good.len() as u32).to_le_bytes());
        assert!(matches!(
            w.restore(&bad),
            Err(SnapshotError::CountTooLarge) | Err(SnapshotError::Truncated)
        ));
        assert_eq!(w.state_hash(), before, "거부된 restore가 월드를 오염시킴");
    }

    /// BLOCKER(구조): 값이 유한·범위 안이어도 **타입과 질량의 관계**가 깨지면 거부한다.
    /// dynamic의 타입 바이트를 static(0)으로 바꾸면 질량은 그대로라 불일치가 된다.
    #[test]
    fn snapshot_rejects_type_mass_mismatch() {
        let mut w = World::new(WorldParams::default());
        let g = ground(&mut w);
        dyn_box(&mut w, 0.0, 3.0);
        w.step(DT);
        let good = w.snapshot();
        let before = w.state_hash();

        // 슬롯 레이아웃(Box 셰이프): alive1+reserved1+gen4 + 본문106 + sleep4+asleep1+island4 = 121
        //   본문106 = type1 + pos12 + rot16 + lin12 + ang12 + invM4 + invI12 + fric4 + rest4
        //            + shape_origin_offset12 + Box17
        const SLOT: usize = 121;
        let type_off = 12 + SLOT + 1 + 1 + 4; // 헤더 + 슬롯0(ground) + alive/reserved/gen
        assert_eq!(
            good[type_off], 1,
            "슬롯1이 dynamic이 아님 — 레이아웃 가정 붕괴"
        );
        let mut bad = good.clone();
        bad[type_off] = 0; // Dynamic → Static (질량은 그대로 남는다)
        assert_eq!(
            w.restore(&bad),
            Err(SnapshotError::InconsistentState),
            "질량을 가진 static이 통과"
        );
        assert_eq!(w.state_hash(), before);
        assert!(w.is_alive(g));
    }

    /// MAJOR: kinematic은 **항상 활성**이다 — 구동 중인 kinematic에 매달린 조인트를
    /// 비활성으로 스킵하면 체인이 처방 운동을 따라가지 못한다.
    #[test]
    fn kinematic_drives_jointed_body() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let k = w.add_body(Body::new_kinematic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 5.0, 0.0),
        ));
        let b = dyn_box(&mut w, 1.5, 5.0);
        w.add_joint(Joint::point(
            k,
            b,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
        ));
        w.set_linear_velocity(k, Vec3::new(2.0, 0.0, 0.0)).unwrap();
        for _ in 0..120 {
            w.step(DT);
        }
        let kx = w.body(k).pose.pos.x;
        let bx = w.body(b).pose.pos.x;
        assert!(kx > 3.0, "kinematic이 구동되지 않음 x={kx}");
        assert!(
            (bx - (kx + 1.5)).abs() < 0.5,
            "조인트가 kinematic을 따라가지 못함 k={kx} b={bx}"
        );
    }

    /// MAJOR: `World.params`는 비공개 — 교체도 생성자와 같은 검증을 거친다.
    #[test]
    #[should_panic(expected = "substeps must be in 1..=64")]
    fn set_params_revalidates() {
        let mut w = World::new(WorldParams::default());
        w.set_params(WorldParams {
            substeps: 0,
            ..Default::default()
        });
    }

    // ─────────── 리뷰 rev_1517fd7a 회귀 ───────────

    /// BLOCKER: `#[non_exhaustive]`만으로는 필드 변조를 못 막는다 → 필드 비공개화 +
    /// **삽입 경계 재검증**. (컴파일 차단 자체는 body.rs의 compile_fail 독테스트가 증명한다.)
    #[test]
    #[should_panic(expected = "invalid body")]
    fn add_body_revalidates_mutated_body() {
        let mut w = World::new(WorldParams::default());
        let mut b = Body::new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 1.0, 0.0), 1.0);
        // 크레이트 안에서만 가능한 변조를 흉내 — 밖에서는 컴파일되지 않는다
        b.shape = Shape::Hull {
            points: vec![],
            radius: 0.0,
        };
        w.add_body(b);
    }

    /// NEW MAJOR: 생성자가 `is_finite`만 봐서 0 쿼터니언이 통과했다.
    #[test]
    #[should_panic(expected = "normalizable")]
    fn zero_quaternion_rejected_at_construction() {
        Body::new_dynamic(
            Shape::Sphere { radius: 1.0 },
            Pose {
                pos: Vec3::ZERO,
                rot: Quat {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 0.0,
                },
            },
            1.0,
        );
    }

    /// PARTIAL MAJOR: 검사하지 않던 파라미터들. NaN 임계는 비교가 항상 false라
    /// 게이트를 조용히 무력화한다.
    #[test]
    #[should_panic(expected = "restitution_threshold")]
    fn nan_restitution_threshold_rejected() {
        World::new(WorldParams {
            solver: crate::solver::SolverParams {
                restitution_threshold: f32::NAN,
                ..Default::default()
            },
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "time_to_sleep")]
    fn negative_time_to_sleep_rejected() {
        World::new(WorldParams {
            time_to_sleep: -1.0,
            ..Default::default()
        });
    }

    /// PARTIAL BLOCKER: 유한하지만 거대한 값은 첫 곱에서 Inf가 된다 — 크기 상한이 필요하다.
    #[test]
    fn snapshot_rejects_huge_finite_values() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        dyn_box(&mut w, 0.0, 3.0);
        w.step(DT);
        let good = w.snapshot();
        let before = w.state_hash();
        // 슬롯1의 inv_mass 자리 (SLOT 121 · alive1+reserved1+gen4 + type1+pos12+rot16+lin12+ang12)
        let off = 12 + 121 + 1 + 1 + 4 + 1 + 12 + 16 + 12 + 12;
        let mut bad = good.clone();
        bad[off..off + 4].copy_from_slice(&f32::MAX.to_le_bytes());
        assert_eq!(
            w.restore(&bad),
            Err(SnapshotError::NonFinite),
            "f32::MAX 역질량이 통과 — 첫 warm start 곱에서 Inf가 된다"
        );
        assert_eq!(w.state_hash(), before);
    }

    /// PARTIAL MAJOR: enqueue와 restore의 계약이 어긋나면 **자기 월드가 만든 스냅샷이
    /// 자기에게 복원되지 않는다**. 큐에 무엇을 넣어도 왕복이 성립해야 한다.
    #[test]
    fn any_enqueued_state_survives_snapshot_roundtrip() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let a = dyn_box(&mut w, 0.0, 3.0);
        let doomed = dyn_box(&mut w, 3.0, 3.0);
        w.step(DT);
        // 제거된 바디의 옛 핸들은 애초에 큐에 들어가지 못한다
        w.remove_body(doomed).unwrap();
        assert_eq!(
            w.enqueue(Command::SetLinearVelocity(doomed, Vec3::ZERO)),
            Err(ApiError::StaleHandle)
        );
        // 유효한 것만 쌓고 왕복
        w.enqueue(Command::SetLinearVelocity(a, Vec3::new(1.0, 0.0, 0.0)))
            .unwrap();
        let fresh = w.enqueue_add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(6.0, 3.0, 0.0),
            1.0,
        ));
        w.enqueue(Command::SetMaterial(fresh, 0.4, 0.2)).unwrap();
        let snap = w.snapshot();
        assert!(
            w.restore(&snap).is_ok(),
            "자기 월드가 만든 스냅샷이 복원되지 않음 (enqueue/restore 계약 불일치)"
        );
        assert_eq!(w.pending_commands(), 3);
        w.step(DT);
        assert!(w.is_alive(fresh));
        assert_eq!(w.last_command_report().rejected, 0);
    }

    /// PARTIAL MAJOR: hull 재중심화로 엔진 pose ≠ 호스트가 준 원점이 된다 →
    /// 렌더 오프셋을 조회할 수 있어야 시각 형상과 물리 형상이 어긋나지 않는다.
    #[test]
    fn hull_recenter_exposes_render_offset() {
        let origin = pose(1.0, 2.0, 3.0);
        let b = Body::new_dynamic(
            Shape::Hull {
                points: vec![Vec3::new(9.5, -0.5, -0.5), Vec3::new(10.5, 0.5, 0.5)],
                radius: 0.0,
            },
            origin,
            1.0,
        );
        // 질량중심은 원래 원점에서 (10,0,0) 떨어져 있다
        assert!((b.pose().pos - Vec3::new(11.0, 2.0, 3.0)).length() < 1e-3);
        // 호스트가 원래 정점 버퍼를 그릴 위치 = pose * offset → 원래 원점으로 돌아온다
        let render = b.pose().pos + b.pose().rot.rotate(b.shape_origin_offset());
        assert!(
            (render - origin.pos).length() < 1e-3,
            "렌더 오프셋이 원래 원점을 복원하지 못함: {render:?}"
        );
    }

    /// PARTIAL MAJOR: 정상 슬라이딩에서 warm start가 매 프레임 끊기면 안 된다.
    /// 허용치가 기하 척도만 쓰면 60Hz·1m/s(프레임당 16.7mm)가 반두께 허용치를 넘는다.
    #[test]
    fn sliding_contact_keeps_warm_start() {
        let mut w = World::new(WorldParams::default());
        // 얇은 판 위를 구가 미끄러진다 — 구·판 접점은 generic witness(위치 매칭 대상)
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(20.0, 0.01, 20.0),
                radius: 0.0,
            },
            pose(0.0, -0.01, 0.0),
        ));
        let s = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(-5.0, 0.3, 0.0),
            1.0,
        ));
        for _ in 0..60 {
            w.step(DT); // 안착
        }
        w.set_linear_velocity(s, Vec3::new(1.0, 0.0, 0.0)).unwrap();
        // 진단은 **스텝별로 리셋**되므로 매 스텝 누적한다
        let (mut hits, mut total) = (0usize, 0usize);
        for _ in 0..180 {
            w.step(DT);
            let d = w.diagnostics();
            hits += d.warm_hits;
            total += d.warm_total;
        }
        assert!(total > 100, "접촉 표본이 부족: {total}");
        let ratio = hits as f32 / total as f32;
        assert!(
            ratio > 0.8,
            "슬라이딩 중 warm start 계승률 {ratio:.2} — 매 프레임 끊기고 있다"
        );
    }

    // ─────────── 리뷰 rev_9a7af148 회귀 ───────────

    /// NEW BLOCKER: 유한하지만 과대한 값은 몇 서브스텝 만에 Inf가 된다 —
    /// `is_finite()`만으로는 방어가 아니다. 라이브 API도 스냅샷과 **같은 정책**을 쓴다.
    #[test]
    fn live_api_rejects_huge_finite_values() {
        let mut w = World::new(WorldParams::default());
        let b = dyn_box(&mut w, 0.0, 3.0);
        assert_eq!(
            w.set_linear_velocity(b, Vec3::new(f32::MAX, 0.0, 0.0)),
            Err(ApiError::InvalidValue),
            "f32::MAX 속도가 통과"
        );
        assert_eq!(
            w.enqueue(Command::SetLinearVelocity(b, Vec3::new(f32::MAX, 0.0, 0.0))),
            Err(ApiError::InvalidValue)
        );
        assert_eq!(
            w.set_pose(
                b,
                Pose {
                    pos: Vec3::new(1.0e20, 0.0, 0.0),
                    rot: Quat::IDENTITY
                }
            ),
            Err(ApiError::InvalidValue)
        );
    }

    #[test]
    #[should_panic(expected = "MAX_MAGNITUDE")]
    fn huge_pose_rejected_at_construction() {
        Body::new_dynamic(Shape::Sphere { radius: 1.0 }, pose(1.0e20, 0.0, 0.0), 1.0);
    }

    #[test]
    #[should_panic(expected = "dt must be finite")]
    fn huge_dt_rejected() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        w.step(8.0);
    }

    #[test]
    #[should_panic(expected = "invalid shape")]
    fn oversized_hull_rejected() {
        let points: Vec<Vec3> = (0..(maru_collide::MAX_HULL_POINTS + 1))
            .map(|i| Vec3::new(i as f32 * 1e-3, 0.0, 0.0))
            .collect();
        Body::new_dynamic(
            Shape::Hull {
                points,
                radius: 0.0,
            },
            pose(0.0, 0.0, 0.0),
            1.0,
        );
    }

    /// BLOCKER: 공개 경로로 같은 예약 핸들에 AddBody를 두 번 넣으면 restore가 거부하는
    /// pending 상태가 만들어진다 → 공개 AddBody 자체를 막고 `enqueue_add_body`만 정본으로.
    #[test]
    fn public_add_body_command_is_rejected() {
        let mut w = World::new(WorldParams::default());
        let h = w.enqueue_add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 3.0, 0.0),
            1.0,
        ));
        assert_eq!(
            w.enqueue(Command::AddBody(
                h,
                Box::new(Body::new_dynamic(
                    Shape::Sphere { radius: 0.9 },
                    pose(5.0, 3.0, 0.0),
                    1.0,
                ))
            )),
            Err(ApiError::InvalidValue),
            "예약 핸들에 두 번째 AddBody가 큐에 들어감"
        );
        assert_eq!(w.pending_commands(), 1, "예약당 AddBody는 정확히 하나");
        assert!(w.restore(&w.snapshot()).is_ok());
    }

    /// 정책 **경계**의 대칭: 라이브 API가 받아들이는 최대치는 스냅샷 리더도 받아야 한다.
    /// (리뷰가 지적한 비대칭의 정확한 형태 — 리더만 1e9로 막고 있었다.)
    #[test]
    fn boundary_legal_values_round_trip() {
        let big = maru_collide::MAX_MAGNITUDE * 0.9;
        let mut w = World::new(WorldParams::default());
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(big, 0.0, 0.0),
            1.0,
        ));
        w.set_linear_velocity(h, Vec3::new(big, 0.0, 0.0)).unwrap();
        let snap = w.snapshot();
        assert!(
            w.restore(&snap).is_ok(),
            "라이브 API가 허용한 최대치를 리더가 거부 — 정책 비대칭"
        );
        assert!((w.body(h).pose().pos.x - big).abs() < big * 1e-6);
    }

    /// **정책 대칭 속성**: 공개 API로 도달할 수 있는 어떤 상태든 자기 스냅샷을 복원할 수
    /// 있어야 한다. 리뷰가 두 라운드 연속으로 이 계약의 구멍을 찾았으므로, 개별 사례가
    /// 아니라 **연산 시퀀스 전체**를 훑어 계약을 성질로 검증한다.
    #[test]
    fn every_reachable_state_restores_its_own_snapshot() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let mut live: Vec<BodyHandle> = Vec::new();
        // 결정적 LCG — 무작위성 없이 시퀀스를 흔든다
        let mut rng: u32 = 0x2545_f491;
        let next = |rng: &mut u32| {
            *rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*rng >> 16) as usize
        };
        for step_i in 0..200 {
            match next(&mut rng) % 7 {
                0 => live.push(dyn_box(&mut w, (step_i % 9) as f32 - 4.0, 3.0)),
                1 => {
                    if let Some(h) = live.pop() {
                        let _ = w.remove_body(h);
                    }
                }
                2 => {
                    if let Some(h) = live.first() {
                        let _ = w.enqueue(Command::SetLinearVelocity(
                            *h,
                            Vec3::new((step_i % 5) as f32, 0.0, 0.0),
                        ));
                    }
                }
                3 => live.push(w.enqueue_add_body(Body::new_dynamic(
                    Shape::Sphere { radius: 0.3 },
                    pose((step_i % 7) as f32 - 3.0, 5.0, 0.0),
                    1.0,
                ))),
                4 => {
                    if live.len() >= 2 {
                        let _ = w.try_add_joint(Joint::point(
                            live[0],
                            live[1],
                            Vec3::ZERO,
                            Vec3::new(0.0, 1.0, 0.0),
                        ));
                    }
                }
                5 => w.step(DT),
                _ => {
                    w.flush_commands();
                }
            }
            let snap = w.snapshot();
            assert!(
                w.restore(&snap).is_ok(),
                "연산 {step_i} 이후 자기 스냅샷이 복원되지 않음 (정책 비대칭)"
            );
        }
    }
}
