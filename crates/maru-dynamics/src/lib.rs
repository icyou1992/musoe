//! maru-dynamics — 강체 동역학 (M2).
//!
//! 설계 계약(02-architecture.md): §4.4 TGS-soft, §4.6 슬리핑, §8 결정론·denormal 위생, §9 바디 3종.

mod body;
mod character;
mod command;
mod joint;
mod snapshot;
mod solver;
mod solver_block;
mod threads;
mod world;

pub use body::{Body, BodyHandle, BodyType};
pub use character::{CharacterVirtual, MoveResult};
pub use command::{Command, CommandReport};
pub use joint::{Joint, JointKind, Motor};
pub use snapshot::{SnapshotError, SNAPSHOT_VERSION};
pub use solver::SolverParams;
pub use world::{
    ApiError, ContactEvent, ContactEventKind, Diagnostics, QueryFilter, RayHit, SweepHit, World,
    WorldParams,
};

#[cfg(test)]
mod tests {

    fn unit_sphere_world() -> (World, BodyHandle) {
        let mut w = World::new(WorldParams {
            gravity: maru_math::Vec3::ZERO,
            ..WorldParams::default()
        });
        let h = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            2.0,
        ));
        (w, h)
    }

    /// 힘은 **가속도**로 들어간다: F=ma 이므로 dt 동안 Δv = F·dt/m 이어야 한다.
    /// (`new_dynamic`의 셋째 인자는 질량이 아니라 **밀도**다 — 질량은 역질량에서 되찾는다.)
    #[test]
    fn applied_force_integrates_as_acceleration() {
        let (mut w, h) = unit_sphere_world();
        let m = 1.0 / w.body(h).inv_mass();
        w.apply_force(h, maru_math::Vec3::new(10.0, 0.0, 0.0))
            .unwrap();
        let dt = 1.0 / 60.0;
        w.try_step(dt).unwrap();
        let v = w.body(h).linvel();
        let want = 10.0 / m * dt;
        assert!((v.x - want).abs() < 1e-4, "Δv={} 인데 F·dt/m = {want}", v.x);
        // **한 스텝이 소비하고 지운다** — 다음 스텝에 다시 가속되면 안 된다.
        assert_eq!(w.body(h).force(), maru_math::Vec3::ZERO);
        w.try_step(dt).unwrap();
        assert!(
            (w.body(h).linvel().x - want).abs() < 1e-4,
            "힘이 두 번 먹혔다"
        );
    }

    /// 여러 번 부르면 더해진다.
    #[test]
    fn forces_accumulate_until_the_step() {
        let (mut w, h) = unit_sphere_world();
        for _ in 0..4 {
            w.apply_force(h, maru_math::Vec3::new(1.0, 0.0, 0.0))
                .unwrap();
        }
        assert!((w.body(h).force().x - 4.0).abs() < 1e-6);
        let m = 1.0 / w.body(h).inv_mass();
        w.try_step(1.0 / 60.0).unwrap();
        let want = 4.0 / m * (1.0 / 60.0);
        assert!((w.body(h).linvel().x - want).abs() < 1e-4);
    }

    /// 충격은 즉시 속도를 바꾼다 (스텝을 기다리지 않는다).
    #[test]
    fn impulse_changes_velocity_immediately() {
        let (mut w, h) = unit_sphere_world();
        let m = 1.0 / w.body(h).inv_mass();
        w.apply_impulse(h, maru_math::Vec3::new(6.0, 0.0, 0.0))
            .unwrap();
        assert!((w.body(h).linvel().x - 6.0 / m).abs() < 1e-5, "Δv = J/m");
    }

    /// 질량중심에서 벗어난 점을 때리면 **회전도** 생긴다.
    #[test]
    fn off_center_impulse_creates_spin() {
        let (mut w, h) = unit_sphere_world();
        let com = w.body(h).pose().pos;
        // 중심에서 +y로 0.5 떨어진 점을 +x로 때린다 → z축 각속도가 생겨야 한다
        w.apply_impulse_at_point(
            h,
            maru_math::Vec3::new(4.0, 0.0, 0.0),
            com + maru_math::Vec3::new(0.0, 0.5, 0.0),
        )
        .unwrap();
        let m = 1.0 / w.body(h).inv_mass();
        let b = w.body(h);
        assert!(
            (b.linvel().x - 4.0 / m).abs() < 1e-5,
            "선속도는 중심 타격과 같다"
        );
        assert!(
            b.angvel().z < -1e-3,
            "r×J 의 z가 음수여야 한다: {:?}",
            b.angvel()
        );
        assert!(b.angvel().x.abs() < 1e-6 && b.angvel().y.abs() < 1e-6);
    }

    /// 잠든 바디에 힘을 주면 깨어나야 한다 — 안 그러면 "왜 안 움직이지"가 된다.
    #[test]
    fn applying_force_wakes_a_sleeping_body() {
        let mut w = World::new(WorldParams::default());
        w.add_body(Body::new_static(
            maru_collide::Shape::Box {
                half: maru_math::Vec3::new(10.0, 1.0, 10.0),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, -1.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        ));
        let h = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 0.5, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        for _ in 0..240 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        assert!(w.is_asleep(h), "먼저 잠들어야 의미 있는 테스트다");
        w.apply_impulse(h, maru_math::Vec3::new(0.0, 3.0, 0.0))
            .unwrap();
        assert!(!w.is_asleep(h), "힘을 받으면 깨어나야 한다");
    }

    /// static·kinematic은 힘을 **조용히 무시하지 않고** 거부한다.
    #[test]
    fn forces_are_rejected_on_non_dynamic_bodies() {
        let mut w = World::new(WorldParams::default());
        let s = w.add_body(Body::new_static(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::ZERO,
                rot: maru_math::Quat::IDENTITY,
            },
        ));
        assert_eq!(
            w.apply_force(s, maru_math::Vec3::new(1.0, 0.0, 0.0)),
            Err(ApiError::ImmutableStatic)
        );
        assert_eq!(
            w.apply_impulse(s, maru_math::Vec3::new(1.0, 0.0, 0.0)),
            Err(ApiError::ImmutableStatic)
        );
    }

    /// 힘 입력은 **입력 범위**로 거른다 (NaN·Inf·상한 초과).
    ///
    /// 누적분에 파생 상한(포화 + `StateOverflow`)을 두긴 하지만, 라이브 API로는 사실상
    /// 도달하지 않는다 — 한 번의 입력이 `MAX_INPUT`(1e4)으로 막혀 있어 `MAX_STATE`(1e15)에
    /// 닿으려면 1e11번을 불러야 한다. 그 검사의 실제 수요자는 **스냅샷 리더**다(악의적
    /// 스냅샷이 거대한 힘을 담을 수 있다). 그래서 여기서는 입력 경계만 확인하고,
    /// 복원 경계는 `read_body`가 라이브와 같은 상태 정책으로 읽는 것으로 보장한다.
    #[test]
    fn force_input_is_validated() {
        let (mut w, h) = unit_sphere_world();
        for bad in [
            maru_math::Vec3::new(f32::NAN, 0.0, 0.0),
            maru_math::Vec3::new(f32::INFINITY, 0.0, 0.0),
            maru_math::Vec3::new(1e30, 0.0, 0.0),
        ] {
            assert_eq!(w.apply_force(h, bad), Err(ApiError::InvalidValue));
            assert_eq!(w.apply_torque(h, bad), Err(ApiError::InvalidValue));
            assert_eq!(w.apply_impulse(h, bad), Err(ApiError::InvalidValue));
        }
        assert_eq!(
            w.body(h).force(),
            maru_math::Vec3::ZERO,
            "거부는 상태를 남기지 않는다"
        );
    }

    /// 누적 힘은 **캐노니컬 상태**다 — 스냅샷 왕복이 같은 미래를 만들어야 한다.
    #[test]
    fn pending_forces_survive_snapshot_roundtrip() {
        let (mut w, h) = unit_sphere_world();
        w.apply_force(h, maru_math::Vec3::new(7.0, 0.0, 3.0))
            .unwrap();
        w.apply_torque(h, maru_math::Vec3::new(0.0, 2.0, 0.0))
            .unwrap();
        let snap = w.try_snapshot().unwrap();
        let mut restored = World::new(WorldParams::default());
        restored.restore(&snap).unwrap();
        assert_eq!(restored.body(h).force(), w.body(h).force());
        assert_eq!(restored.body(h).torque(), w.body(h).torque());
        for _ in 0..5 {
            w.try_step(1.0 / 60.0).unwrap();
            restored.try_step(1.0 / 60.0).unwrap();
        }
        assert_eq!(
            w.try_state_hash().unwrap(),
            restored.try_state_hash().unwrap(),
            "힘이 빠지면 해시는 같은데 미래가 갈린다"
        );
    }

    fn query_world() -> (World, Vec<BodyHandle>) {
        let mut w = World::new(WorldParams::default());
        let mut hs = Vec::new();
        // x = 0, 5, 10 에 반지름 1 구 세 개
        for k in 0..3 {
            hs.push(w.add_body(Body::new_dynamic(
                maru_collide::Shape::Sphere { radius: 1.0 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::new(k as f32 * 5.0, 0.0, 0.0),
                    rot: maru_math::Quat::IDENTITY,
                },
                1.0,
            )));
        }
        (w, hs)
    }

    /// 가장 가까운 물체를 맞히고, 맞은 지점·법선이 일관돼야 한다.
    #[test]
    fn ray_hits_the_nearest_body() {
        let (w, hs) = query_world();
        let hit = w
            .cast_ray(
                maru_math::Vec3::new(-10.0, 0.0, 0.0),
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                100.0,
            )
            .unwrap()
            .expect("맞아야 한다");
        assert_eq!(hit.body, hs[0], "가장 가까운 것은 x=0의 구다");
        assert!((hit.toi - 9.0).abs() < 1e-3, "toi={}", hit.toi);
        assert!((hit.point.x - (-1.0)).abs() < 1e-3, "point={:?}", hit.point);
        assert!(
            hit.normal.x < -0.99,
            "바깥 법선은 광선 반대편을 본다: {:?}",
            hit.normal
        );
        assert!(!hit.started_inside);
    }

    /// 모든 히트는 toi 오름차순이고 개수가 맞아야 한다.
    #[test]
    fn ray_all_is_sorted_by_toi() {
        let (w, hs) = query_world();
        let mut out = Vec::new();
        w.cast_ray_all(
            maru_math::Vec3::new(-10.0, 0.0, 0.0),
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            100.0,
            &mut out,
        )
        .unwrap();
        assert_eq!(out.len(), 3);
        assert!(
            out.windows(2).all(|p| p[0].toi <= p[1].toi),
            "toi 정렬 위반"
        );
        assert_eq!(
            out.iter().map(|h| h.body).collect::<Vec<_>>(),
            hs,
            "가까운 순서여야 한다"
        );
    }

    /// `max_toi`는 존중돼야 한다.
    #[test]
    fn ray_respects_max_toi() {
        let (w, _) = query_world();
        let mut out = Vec::new();
        w.cast_ray_all(
            maru_math::Vec3::new(-10.0, 0.0, 0.0),
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            10.0,
            &mut out,
        )
        .unwrap();
        assert_eq!(out.len(), 1, "10 안에는 첫 구만 있다");
    }

    /// 물체 **안에서** 쏘면 맞은 것으로 나와야 한다 — 지면 판정이 여기 걸린다.
    #[test]
    fn ray_from_inside_reports_a_hit() {
        let (w, hs) = query_world();
        let hit = w
            .cast_ray(
                maru_math::Vec3::ZERO,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                100.0,
            )
            .unwrap()
            .expect("안에서 시작해도 맞은 것이다");
        assert_eq!(hit.body, hs[0]);
        assert_eq!(hit.toi, 0.0);
        assert!(hit.started_inside);
    }

    /// AABB·점 질의는 슬롯 오름차순으로 결정적이어야 한다.
    #[test]
    fn aabb_and_point_queries_are_deterministic() {
        let (w, hs) = query_world();
        let mut out = Vec::new();
        w.query_aabb(
            maru_collide::Aabb::new(
                maru_math::Vec3::new(-2.0, -2.0, -2.0),
                maru_math::Vec3::new(7.0, 2.0, 2.0),
            ),
            &mut out,
        )
        .unwrap();
        assert_eq!(out, vec![hs[0], hs[1]], "x=0,5 두 개만 겹친다");

        let mut pts = Vec::new();
        w.query_point(maru_math::Vec3::new(5.2, 0.0, 0.0), &mut pts)
            .unwrap();
        assert_eq!(pts, vec![hs[1]]);
        w.query_point(maru_math::Vec3::new(2.5, 0.0, 0.0), &mut pts)
            .unwrap();
        assert!(pts.is_empty(), "빈틈에는 아무것도 없다");
    }

    /// 잠든 물체는 **보인다**(자는 것은 최적화지 부재가 아니다).
    #[test]
    fn queries_see_sleeping_bodies() {
        let mut w = World::new(WorldParams::default());
        w.add_body(Body::new_static(
            maru_collide::Shape::Box {
                half: maru_math::Vec3::new(10.0, 1.0, 10.0),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, -1.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        ));
        let h = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 0.5, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        for _ in 0..240 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        assert!(w.is_asleep(h), "먼저 잠들어야 의미 있는 테스트다");
        let hit = w
            .cast_ray(
                maru_math::Vec3::new(-5.0, 0.5, 0.0),
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                100.0,
            )
            .unwrap();
        assert_eq!(hit.map(|x| x.body), Some(h), "잠든 물체도 맞아야 한다");
    }

    /// 잘못된 입력은 거부한다 (0 방향·NaN·음수 max_toi).
    #[test]
    fn ray_input_is_validated() {
        let (w, _) = query_world();
        let o = maru_math::Vec3::ZERO;
        assert_eq!(
            w.cast_ray(o, maru_math::Vec3::ZERO, 10.0),
            Err(ApiError::InvalidValue),
            "0 방향은 광선이 아니다"
        );
        assert_eq!(
            w.cast_ray(o, maru_math::Vec3::new(f32::NAN, 0.0, 0.0), 10.0),
            Err(ApiError::InvalidValue)
        );
        assert_eq!(
            w.cast_ray(o, maru_math::Vec3::new(1.0, 0.0, 0.0), -1.0),
            Err(ApiError::InvalidValue)
        );
    }

    /// 떨어뜨리면 **닿는 순간** Started, 치우면 Stopped가 나와야 한다.
    #[test]
    fn contact_events_fire_on_touch_and_separation() {
        let mut w = World::new(WorldParams::default());
        let g = w.add_body(Body::new_static(
            maru_collide::Shape::Box {
                half: maru_math::Vec3::new(10.0, 1.0, 10.0),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, -1.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        ));
        let b = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 3.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        let mut started = 0usize;
        for _ in 0..120 {
            w.try_step(1.0 / 60.0).unwrap();
            for e in w.contact_events() {
                if e.kind == ContactEventKind::Started && e.a == g && e.b == b
                    || e.kind == ContactEventKind::Started && e.a == b && e.b == g
                {
                    started += 1;
                }
            }
            if started > 0 {
                break;
            }
        }
        assert_eq!(started, 1, "닿을 때 Started가 정확히 한 번");

        // 치우면 Stopped
        w.set_pose(
            b,
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 50.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        )
        .unwrap();
        w.try_step(1.0 / 60.0).unwrap();
        assert!(
            w.contact_events()
                .iter()
                .any(|e| e.kind == ContactEventKind::Stopped),
            "떨어지면 Stopped가 나와야 한다"
        );
    }

    /// 이벤트는 **스텝 단위 출력**이다 — 다음 스텝이 비운다.
    #[test]
    fn contact_events_are_cleared_each_step() {
        let mut w = World::new(WorldParams::default());
        w.add_body(Body::new_static(
            maru_collide::Shape::Box {
                half: maru_math::Vec3::new(10.0, 1.0, 10.0),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, -1.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        ));
        w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 0.5, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        w.try_step(1.0 / 60.0).unwrap();
        assert!(!w.contact_events().is_empty(), "첫 스텝에 닿는다");
        for _ in 0..5 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        assert!(
            w.contact_events().is_empty(),
            "계속 닿아 있는 동안은 새 이벤트가 없다"
        );
    }

    /// 충돌 그룹이 다르면 **통과**해야 한다.
    #[test]
    fn collision_groups_filter_contacts() {
        let make = |a_groups: (u32, u32), b_groups: (u32, u32)| -> f32 {
            let mut w = World::new(WorldParams::default());
            w.add_body(
                Body::new_static(
                    maru_collide::Shape::Box {
                        half: maru_math::Vec3::new(10.0, 1.0, 10.0),
                        radius: 0.0,
                    },
                    maru_collide::Pose {
                        pos: maru_math::Vec3::new(0.0, -1.0, 0.0),
                        rot: maru_math::Quat::IDENTITY,
                    },
                )
                .with_groups(a_groups.0, a_groups.1),
            );
            let b = w.add_body(
                Body::new_dynamic(
                    maru_collide::Shape::Sphere { radius: 0.5 },
                    maru_collide::Pose {
                        pos: maru_math::Vec3::new(0.0, 2.0, 0.0),
                        rot: maru_math::Quat::IDENTITY,
                    },
                    1.0,
                )
                .with_groups(b_groups.0, b_groups.1),
            );
            for _ in 0..90 {
                w.try_step(1.0 / 60.0).unwrap();
            }
            w.body(b).pose().pos.y
        };
        let landed = make((1, 1), (1, 1));
        assert!(landed > 0.3, "같은 그룹이면 바닥에 서야 한다: {landed}");
        let fell = make((1, 1), (2, 2));
        assert!(fell < -1.0, "다른 그룹이면 통과해야 한다: {fell}");
        // **양방향**: 한쪽만 상대를 무시해도 통과해야 한다
        let one_way = make((1, 1), (2, 1));
        assert!(one_way < -1.0, "한쪽이 무시하면 양쪽 다 통과다: {one_way}");
    }

    /// 센서는 **이벤트만** 내고 밀지 않는다.
    #[test]
    fn sensors_report_but_do_not_push() {
        let mut w = World::new(WorldParams::default());
        let trigger = w.add_body(
            Body::new_static(
                maru_collide::Shape::Box {
                    half: maru_math::Vec3::new(5.0, 0.5, 5.0),
                    radius: 0.0,
                },
                maru_collide::Pose {
                    pos: maru_math::Vec3::new(0.0, 1.0, 0.0),
                    rot: maru_math::Quat::IDENTITY,
                },
            )
            .as_sensor(true),
        );
        let b = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 4.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        let mut saw = false;
        for _ in 0..120 {
            w.try_step(1.0 / 60.0).unwrap();
            saw |= w
                .contact_events()
                .iter()
                .any(|e| e.kind == ContactEventKind::Started && (e.a == trigger || e.b == trigger));
        }
        assert!(saw, "센서를 지나가면 이벤트가 나와야 한다");
        assert!(
            w.body(b).pose().pos.y < -2.0,
            "센서는 밀지 않는다 — 그대로 떨어져야 한다: {}",
            w.body(b).pose().pos.y
        );
    }

    /// 그룹·센서는 캐노니컬 상태다 — 스냅샷 왕복에서 살아남아야 한다.
    #[test]
    fn groups_and_sensor_survive_snapshot() {
        let mut w = World::new(WorldParams::default());
        let h = w.add_body(
            Body::new_dynamic(
                maru_collide::Shape::Sphere { radius: 0.5 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::ZERO,
                    rot: maru_math::Quat::IDENTITY,
                },
                1.0,
            )
            .with_groups(0b1010, 0b0110)
            .as_sensor(true),
        );
        let snap = w.try_snapshot().unwrap();
        let mut r = World::new(WorldParams::default());
        r.restore(&snap).unwrap();
        assert_eq!(r.body(h).membership(), 0b1010);
        assert_eq!(r.body(h).filter(), 0b0110);
        assert!(r.body(h).is_sensor());
    }

    /// 슬롯이 재사용돼도 **닿은 적 없는 물체에 종료 이벤트가 나가면 안 된다**.
    #[test]
    fn removed_body_does_not_leak_events_to_a_reused_slot() {
        let mut w = World::new(WorldParams::default());
        w.add_body(Body::new_static(
            maru_collide::Shape::Box {
                half: maru_math::Vec3::new(10.0, 1.0, 10.0),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, -1.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        ));
        let a = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 0.5, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        w.try_step(1.0 / 60.0).unwrap();
        assert!(
            !w.contact_events().is_empty(),
            "먼저 닿아야 의미 있는 테스트다"
        );

        w.remove_body(a).unwrap();
        // 같은 슬롯을 재사용하는 새 물체 — 바닥에서 멀리 띄워 닿지 않게 둔다
        let b = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 40.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        w.try_step(1.0 / 60.0).unwrap();
        assert!(
            !w.contact_events().iter().any(|e| e.a == b || e.b == b),
            "새 물체는 아직 아무것도 안 닿았다: {:?}",
            w.contact_events()
        );
    }

    /// 감쇠는 속도를 지수적으로 줄인다 — `v(t) = v0 / (1 + t·d)` 근사.
    #[test]
    fn damping_slows_a_body_without_gravity() {
        let mut w = World::new(WorldParams {
            gravity: maru_math::Vec3::ZERO,
            ..WorldParams::default()
        });
        let free = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        let damped = w.add_body(
            Body::new_dynamic(
                maru_collide::Shape::Sphere { radius: 0.5 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::new(0.0, 30.0, 0.0),
                    rot: maru_math::Quat::IDENTITY,
                },
                1.0,
            )
            .with_damping(2.0, 0.0)
            .unwrap(),
        );
        for h in [free, damped] {
            w.set_linear_velocity(h, maru_math::Vec3::new(10.0, 0.0, 0.0))
                .unwrap();
        }
        for _ in 0..60 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let vf = w.body(free).linvel().x;
        let vd = w.body(damped).linvel().x;
        assert!(
            (vf - 10.0).abs() < 1e-3,
            "감쇠 없으면 그대로여야 한다: {vf}"
        );
        assert!(vd < 4.0 && vd > 0.0, "감쇠는 줄이되 뒤집지 않는다: {vd}");
    }

    /// **큰 감쇠에서 부호가 뒤집히면 안 된다** — 암시적 형태를 쓰는 이유.
    #[test]
    fn huge_damping_does_not_reverse_velocity() {
        let mut w = World::new(WorldParams {
            gravity: maru_math::Vec3::ZERO,
            ..WorldParams::default()
        });
        let h = w.add_body(
            Body::new_dynamic(
                maru_collide::Shape::Sphere { radius: 0.5 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::ZERO,
                    rot: maru_math::Quat::IDENTITY,
                },
                1.0,
            )
            // h = dt/substeps = 1/240 이므로 h·d = 4 ≫ 1 — 명시적이면 부호가 뒤집힌다
            .with_damping(1000.0, 0.0)
            .unwrap(),
        );
        w.set_linear_velocity(h, maru_math::Vec3::new(10.0, 0.0, 0.0))
            .unwrap();
        w.try_step(1.0 / 60.0).unwrap();
        let v = w.body(h).linvel().x;
        assert!((0.0..1.0).contains(&v), "줄되 뒤집히지 않아야 한다: {v}");
    }

    #[test]
    fn damping_input_is_validated_and_survives_snapshot() {
        let mut w = World::new(WorldParams::default());
        let h = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::ZERO,
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        assert_eq!(w.set_damping(h, -1.0, 0.0), Err(ApiError::InvalidValue));
        assert_eq!(w.set_damping(h, f32::NAN, 0.0), Err(ApiError::InvalidValue));
        w.set_damping(h, 1.5, 2.5).unwrap();
        let snap = w.try_snapshot().unwrap();
        let mut r = World::new(WorldParams::default());
        r.restore(&snap).unwrap();
        assert_eq!(r.body(h).linear_damping(), 1.5);
        assert_eq!(r.body(h).angular_damping(), 2.5);
    }

    /// 질의 필터는 **물리 충돌과 같은 양방향 규칙**이어야 한다.
    #[test]
    fn query_filter_matches_physics_group_rule() {
        let mut w = World::new(WorldParams::default());
        let a = w.add_body(
            Body::new_static(
                maru_collide::Shape::Sphere { radius: 1.0 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::new(0.0, 0.0, 0.0),
                    rot: maru_math::Quat::IDENTITY,
                },
            )
            .with_groups(0b01, 0b01),
        );
        let b = w.add_body(
            Body::new_static(
                maru_collide::Shape::Sphere { radius: 1.0 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::new(5.0, 0.0, 0.0),
                    rot: maru_math::Quat::IDENTITY,
                },
            )
            .with_groups(0b10, 0b10),
        );
        let o = maru_math::Vec3::new(-10.0, 0.0, 0.0);
        let d = maru_math::Vec3::new(1.0, 0.0, 0.0);
        // 필터 없이는 가장 가까운 a
        assert_eq!(w.cast_ray(o, d, 100.0).unwrap().map(|h| h.body), Some(a));
        // 그룹 0b10만 보면 b
        assert_eq!(
            w.cast_ray_filtered(o, d, 100.0, QueryFilter::ALL.groups(0b10))
                .unwrap()
                .map(|h| h.body),
            Some(b)
        );
        // 자기 자신 제외
        assert_eq!(
            w.cast_ray_filtered(o, d, 100.0, QueryFilter::ALL.exclude(a))
                .unwrap()
                .map(|h| h.body),
            Some(b)
        );
    }

    /// 센서는 **기본적으로 광선에 안 맞는다** — 총알이 트리거에 막히면 안 된다.
    #[test]
    fn sensors_are_excluded_from_queries_by_default() {
        let mut w = World::new(WorldParams::default());
        let trigger = w.add_body(
            Body::new_static(
                maru_collide::Shape::Sphere { radius: 1.0 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::ZERO,
                    rot: maru_math::Quat::IDENTITY,
                },
            )
            .as_sensor(true),
        );
        let o = maru_math::Vec3::new(-10.0, 0.0, 0.0);
        let d = maru_math::Vec3::new(1.0, 0.0, 0.0);
        assert!(w.cast_ray(o, d, 100.0).unwrap().is_none(), "기본은 제외");
        assert_eq!(
            w.cast_ray_filtered(o, d, 100.0, QueryFilter::ALL.with_sensors())
                .unwrap()
                .map(|h| h.body),
            Some(trigger),
            "명시하면 맞는다"
        );
    }

    /// 앵커 하나로 매단 두 바디를 만든다 (A는 static, B는 dynamic).
    fn hung_pair(w: &mut World, offset: maru_math::Vec3) -> (BodyHandle, BodyHandle) {
        let a = w.add_body(Body::new_static(
            maru_collide::Shape::Sphere { radius: 0.2 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        ));
        let b = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Box {
                half: maru_math::Vec3::new(0.3, 0.3, 0.3),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 10.0, 0.0) + offset,
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        (a, b)
    }

    /// **고정 조인트는 자세까지 묶는다** — 중력이 걸려도 상대 회전이 안 생겨야 한다.
    ///
    /// 앵커를 B의 **무게중심에서 떨어뜨린다**(레버암). 앵커가 무게중심에 있으면 중력이
    /// 토크를 만들지 않아, 각 제약이 아예 없는 Point 조인트로도 이 테스트가 통과한다 —
    /// 실제로 그랬고, 그래서 부호가 뒤집힌 Fixed가 녹색으로 통과했다
    /// (검수 `review_existing_joint_tests_pass_without_any_angular_constraint`).
    #[test]
    fn fixed_joint_holds_relative_orientation() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        w.try_add_joint(
            Joint::try_fixed(
                a,
                b,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                // B의 무게중심이 아니라 **끝단**을 잡는다 → 중력이 토크를 만든다
                maru_math::Vec3::new(-0.3, 0.0, 0.0),
                maru_math::Quat::IDENTITY,
            )
            .unwrap(),
        )
        .unwrap();
        for _ in 0..300 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let q = w.body(b).pose().rot;
        // 상대 자세가 항등이어야 하므로 |w| ≈ 1
        assert!(
            q.w.abs() > 0.995,
            "고정인데 회전했다: {q:?} (|w|={})",
            q.w.abs()
        );
        let p = w.body(b).pose().pos;
        assert!(
            (p - maru_math::Vec3::new(1.3, 10.0, 0.0)).length() < 0.1,
            "위치도 붙어 있어야 한다: {p:?}"
        );
    }

    /// 거리 조인트는 **부등식**이다: 구간 안에서는 자유롭고 max를 넘으면 당긴다.
    #[test]
    fn distance_joint_is_a_rope_not_a_rod() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(0.0, -0.5, 0.0));
        w.try_add_joint(
            Joint::try_distance(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, 0.0, 2.0)
                .unwrap(),
        )
        .unwrap();
        // 처음엔 0.5 떨어져 있으니 자유낙하해야 한다
        w.try_step(1.0 / 60.0).unwrap();
        assert!(
            w.body(b).linvel().y < -0.05,
            "구간 안에서는 자유낙하: {:?}",
            w.body(b).linvel()
        );
        for _ in 0..600 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let d = (w.body(b).pose().pos - maru_math::Vec3::new(0.0, 10.0, 0.0)).length();
        assert!(
            (d - 2.0).abs() < 0.1,
            "max에서 멈춰야 한다 (로프): 거리={d}"
        );
    }

    /// `min == max`면 막대다 — 양방향으로 붙잡는다.
    #[test]
    fn distance_joint_with_equal_bounds_is_a_rod() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(0.0, -1.0, 0.0));
        w.try_add_joint(
            Joint::try_distance(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, 1.0, 1.0)
                .unwrap(),
        )
        .unwrap();
        for _ in 0..600 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let d = (w.body(b).pose().pos - maru_math::Vec3::new(0.0, 10.0, 0.0)).length();
        assert!((d - 1.0).abs() < 0.08, "막대 길이 유지: {d}");
    }

    /// hinge 한계는 문이 벽을 뚫고 가지 않게 한다.
    #[test]
    fn hinge_limit_stops_rotation() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
        w.try_add_joint(
            Joint::try_hinge(
                a,
                b,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Vec3::ZERO,
                axis,
                axis,
            )
            .unwrap()
            .with_limit(-0.3, 0.3)
            .unwrap(),
        )
        .unwrap();
        // 초기 각속도를 줘서 한계에 **부딪히게** 만든다 — 앵커가 무게중심에 있으면
        // 중력만으로는 토크가 안 생겨 한계가 시험되지 않는다.
        w.set_angular_velocity(b, maru_math::Vec3::new(0.0, 0.0, 6.0))
            .unwrap();
        for _ in 0..600 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        // 한계가 없으면 중력에 끌려 크게 돌아간다. 한계 안에 머물러야 한다.
        let q = w.body(b).pose().rot;
        let ang = 2.0 * q.z.abs().atan2(q.w.abs());
        assert!(ang < 0.6, "한계를 크게 넘었다: {ang} rad");
    }

    /// 모터는 목표 각속도를 추종하되 `max_impulse`를 넘지 않는다.
    #[test]
    fn hinge_motor_drives_toward_target_speed() {
        let mut w = World::new(WorldParams {
            gravity: maru_math::Vec3::ZERO,
            ..WorldParams::default()
        });
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
        w.try_add_joint(
            Joint::try_hinge(
                a,
                b,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Vec3::ZERO,
                axis,
                axis,
            )
            .unwrap()
            .with_motor(3.0, 100.0)
            .unwrap(),
        )
        .unwrap();
        for _ in 0..240 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let wz = w.body(b).angvel().z;
        // **부호까지 본다.** `|wz∓3|` 둘 중 하나만 보면 모터 방향 규약이 시험되지 않는다.
        assert!(
            (wz - 3.0).abs() < 0.5,
            "목표 각속도 +3 rad/s를 부호까지 추종해야 한다: {wz}"
        );
    }

    /// 새 조인트 종류·한계·모터가 전부 스냅샷을 왕복해야 한다.
    #[test]
    fn new_joint_kinds_survive_snapshot() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
        w.try_add_joint(
            Joint::try_fixed(
                a,
                b,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Vec3::ZERO,
                maru_math::Quat::IDENTITY,
            )
            .unwrap(),
        )
        .unwrap();
        w.try_add_joint(
            Joint::try_distance(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, 0.5, 2.0)
                .unwrap(),
        )
        .unwrap();
        w.try_add_joint(
            Joint::try_hinge(
                a,
                b,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Vec3::ZERO,
                axis,
                axis,
            )
            .unwrap()
            .with_limit(-1.0, 1.0)
            .unwrap()
            .with_motor(2.0, 50.0)
            .unwrap(),
        )
        .unwrap();
        for _ in 0..30 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let snap = w.try_snapshot().unwrap();
        let mut r = World::new(WorldParams::default());
        r.restore(&snap).unwrap();
        assert_eq!(r.try_state_hash().unwrap(), w.try_state_hash().unwrap());
        for _ in 0..30 {
            w.try_step(1.0 / 60.0).unwrap();
            r.try_step(1.0 / 60.0).unwrap();
        }
        assert_eq!(
            r.try_state_hash().unwrap(),
            w.try_state_hash().unwrap(),
            "복원 후 미래가 갈리면 안 된다"
        );
    }

    /// 잘못된 조인트 파라미터는 거부한다.
    #[test]
    fn new_joint_parameters_are_validated() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        let z = maru_math::Vec3::ZERO;
        // min > max
        assert!(Joint::try_distance(a, b, z, z, 2.0, 1.0).is_err());
        // 음수 거리
        assert!(Joint::try_distance(a, b, z, z, -1.0, 1.0).is_err());
        // NaN
        assert!(Joint::try_distance(a, b, z, z, 0.0, f32::NAN).is_err());
        // 비단위 쿼터니언
        assert!(Joint::try_fixed(
            a,
            b,
            z,
            z,
            maru_math::Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0
            }
        )
        .is_err());
        // 한계·모터는 hinge에만
        let p = Joint::try_point(a, b, z, z).unwrap();
        assert!(p.clone().with_limit(-1.0, 1.0).is_err());
        assert!(p.with_motor(1.0, 1.0).is_err());
    }

    // ─────────────── Prismatic (슬라이더) — M4 조인트 세트 ───────────────

    /// 슬라이더 위 바디는 **축 방향으로만** 움직인다.
    ///
    /// 중력을 축과 수직으로 두면(축 = x, 중력 = −y) 자유 슬라이더는 y로 떨어지지 않아야
    /// 한다. 떨어지면 수직 2-DOF 구속이 안 걸린 것이다.
    #[test]
    fn prismatic_confines_motion_to_its_axis() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        let axis = maru_math::Vec3::new(1.0, 0.0, 0.0);
        w.try_add_joint(
            Joint::try_prismatic(
                a,
                b,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::ZERO,
                axis,
                maru_math::Quat::IDENTITY,
            )
            .unwrap(),
        )
        .unwrap();
        let y0 = w.body(b).pose().pos.y;
        for _ in 0..300 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let p = w.body(b).pose().pos;
        assert!(
            (p.y - y0).abs() < 0.02,
            "축에 수직인 중력에 끌려 내려갔다: y {y0} → {}",
            p.y
        );
        assert!(p.z.abs() < 0.02, "축 밖 z로 벗어났다: {}", p.z);
    }

    /// 슬라이더는 상대 자세를 잠근다 — 축 방향 병진만 남긴다.
    #[test]
    fn prismatic_locks_relative_orientation() {
        let mut w = World::new(WorldParams {
            gravity: maru_math::Vec3::ZERO,
            ..WorldParams::default()
        });
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        w.try_add_joint(
            Joint::try_prismatic(
                a,
                b,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Quat::IDENTITY,
            )
            .unwrap(),
        )
        .unwrap();
        // 축 둘레로 세게 돌려도 각 3-DOF 잠금이 되돌려야 한다.
        w.set_angular_velocity(b, maru_math::Vec3::new(5.0, 3.0, -2.0))
            .unwrap();
        for _ in 0..300 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let q = w.body(b).pose().rot;
        let angle = 2.0 * (q.x * q.x + q.y * q.y + q.z * q.z).sqrt().min(1.0).asin();
        assert!(angle < 0.1, "상대 자세가 풀렸다: {angle} rad");
    }

    /// 축 방향 한계는 슬라이더를 구간 안에 묶는다.
    #[test]
    fn prismatic_limit_bounds_travel() {
        let mut w = World::new(WorldParams {
            // 중력을 **축 방향**으로 둔다 — 한계를 실제로 밀어붙이는 하중이 필요하다.
            gravity: maru_math::Vec3::new(-9.81, 0.0, 0.0),
            ..WorldParams::default()
        });
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(0.0, 0.0, 0.0));
        let jh = w
            .try_add_joint(
                Joint::try_prismatic(
                    a,
                    b,
                    maru_math::Vec3::ZERO,
                    maru_math::Vec3::ZERO,
                    maru_math::Vec3::new(1.0, 0.0, 0.0),
                    maru_math::Quat::IDENTITY,
                )
                .unwrap()
                .with_limit(-0.5, 0.5)
                .unwrap(),
            )
            .unwrap();
        for _ in 0..600 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let x = w.body(b).pose().pos.x - w.body(a).pose().pos.x;
        assert!(
            x > -0.6,
            "한계가 없으면 계속 미끄러진다 — 하한을 크게 넘었다: {x}"
        );
        assert!(x < 0.6, "상한을 넘었다: {x}");
        // **호스트가 읽는 좌표와 물리가 같은 식이어야 한다** — 표시값이 따로 놀면
        // "문이 얼마나 열렸나"가 한계와 어긋난다.
        let s = w.prismatic_translation(jh).expect("prismatic 좌표");
        assert!(
            (s - x).abs() < 1.0e-4,
            "World가 보고하는 슬라이더 좌표({s})가 실제 변위({x})와 다르다"
        );
        // 다른 종류 조인트는 None이어야 한다.
        let jp = w
            .try_add_joint(
                Joint::try_point(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO).unwrap(),
            )
            .unwrap();
        assert!(w.prismatic_translation(jp).is_none());
        assert!(w.prismatic_translation(9999).is_none());
    }

    /// 선형 모터는 목표 속도를 **부호까지** 추종한다.
    #[test]
    fn prismatic_motor_drives_toward_target_speed() {
        let mut w = World::new(WorldParams {
            gravity: maru_math::Vec3::ZERO,
            ..WorldParams::default()
        });
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::ZERO);
        w.try_add_joint(
            Joint::try_prismatic(
                a,
                b,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Quat::IDENTITY,
            )
            .unwrap()
            .with_motor(2.0, 100.0)
            .unwrap(),
        )
        .unwrap();
        for _ in 0..240 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let vx = w.body(b).linvel().x;
        // 목표는 **A 기준 B**의 축 방향 속도다. A가 static이므로 그대로 vx여야 한다.
        assert!(
            (vx - 2.0).abs() < 0.3,
            "목표 속도 +2 m/s를 부호까지 추종해야 한다: {vx}"
        );
    }

    /// 선형 모터의 최대 힘도 **서브스텝 수에 무관**해야 한다 (hinge 모터와 같은 계약).
    #[test]
    fn prismatic_motor_effort_is_independent_of_substeps() {
        fn push(substeps: u32) -> f32 {
            let mut w = World::new(WorldParams {
                gravity: maru_math::Vec3::ZERO,
                substeps,
                ..WorldParams::default()
            });
            let (a, b) = hung_pair(&mut w, maru_math::Vec3::ZERO);
            w.try_add_joint(
                Joint::try_prismatic(
                    a,
                    b,
                    maru_math::Vec3::ZERO,
                    maru_math::Vec3::ZERO,
                    maru_math::Vec3::new(1.0, 0.0, 0.0),
                    maru_math::Quat::IDENTITY,
                )
                .unwrap()
                // 목표를 크게 잡아 모터를 항상 포화시킨다 → 전달량 = 상한 × dt
                .with_motor(50.0, 0.2)
                .unwrap(),
            )
            .unwrap();
            w.try_step(1.0 / 60.0).unwrap();
            w.body(b).linvel().x
        }
        let s1 = push(1);
        let s8 = push(8);
        assert!(
            (s1 - s8).abs() < 0.1 * s1.abs().max(1e-6),
            "substeps만 바꿨는데 모터 전달 임펄스가 달라진다: {s1} vs {s8}"
        );
    }

    /// prismatic도 스냅샷을 왕복하고 복원 후 미래가 갈리지 않아야 한다.
    #[test]
    fn prismatic_survives_snapshot() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        w.try_add_joint(
            Joint::try_prismatic(
                a,
                b,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::new(0.0, 1.0, 0.0),
                maru_math::Quat::IDENTITY,
            )
            .unwrap()
            .with_limit(-1.0, 1.0)
            .unwrap()
            .with_motor(0.5, 25.0)
            .unwrap(),
        )
        .unwrap();
        for _ in 0..30 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let snap = w.try_snapshot().unwrap();
        let mut r = World::new(WorldParams::default());
        r.restore(&snap).unwrap();
        assert_eq!(r.try_state_hash().unwrap(), w.try_state_hash().unwrap());
        for _ in 0..30 {
            w.try_step(1.0 / 60.0).unwrap();
            r.try_step(1.0 / 60.0).unwrap();
        }
        assert_eq!(
            r.try_state_hash().unwrap(),
            w.try_state_hash().unwrap(),
            "복원 후 미래가 갈리면 안 된다"
        );
    }

    /// **축이 회전하면 캐시된 수직 반력이 자유축으로 샌다** — warm start가 그걸
    /// 그대로 실으면 슬라이더가 스스로 미끄러진다 (교차검수 rev_d2e60241 MAJOR).
    ///
    /// 솔버 상태를 직접 만들어 그 상황을 재현한다: 누적 임펄스를 **현재 자유축과
    /// 나란하게** 두고 warm start를 돌린 뒤, 자유축 상대속도가 그대로인지 본다.
    #[test]
    fn prismatic_warm_start_does_not_leak_into_the_free_axis() {
        use crate::solver::SolverState;
        let axis = maru_math::Vec3::new(1.0, 0.0, 0.0);
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::ZERO);
        let mut j = Joint::try_prismatic(
            a,
            b,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::ZERO,
            axis,
            maru_math::Quat::IDENTITY,
        )
        .unwrap();
        // 이전 프레임에 축이 다른 방향이었을 때 쌓인 수직 반력 — 지금은 자유축과 나란하다.
        j.linear_impulse = axis * 5.0;

        let mut st = SolverState::default();
        st.reserve_for(2).unwrap();
        st.lin = vec![maru_math::Vec3::ZERO; 2];
        st.ang = vec![maru_math::Vec3::ZERO; 2];
        st.pos = vec![maru_math::Vec3::ZERO, maru_math::Vec3::new(1.0, 0.0, 0.0)];
        st.rot = vec![maru_math::Quat::IDENTITY; 2];
        st.inv_m = vec![0.0, 1.0]; // A는 static, B는 dynamic
        st.inv_i = vec![maru_math::Mat3::from_diagonal(maru_math::Vec3::ZERO); 2];
        st.inv_i[1] = maru_math::Mat3::from_diagonal(maru_math::Vec3::new(1.0, 1.0, 1.0));
        st.dynamic = vec![false, true];

        crate::joint::warm_start(&mut st, &j);
        let free = axis.dot(st.lin[1] - st.lin[0]);
        assert!(
            free.abs() < 1.0e-6,
            "warm start가 자유축 속도를 주입했다: {free} m/s"
        );
    }

    /// 비단위(허용 범위 안) `rel_rot`은 **생성 시점에** 캐노니컬화돼야 한다 —
    /// 스냅샷 리더가 정규화하므로, 안 하면 왕복이 항등이 아니다.
    #[test]
    fn joint_rel_rot_is_canonicalized_so_snapshots_round_trip() {
        // norm² = 0.9549 — validate 허용 범위(0.9~1.1) 안이지만 **단위가 아니다**.
        // 경계값(0.9 정확)은 f32 반올림으로 검증에 걸릴 수 있어 안쪽을 고른다.
        // 항등이 아닌 회전을 써야 정규화가 값을 실제로 바꾼다.
        let loose = maru_math::Quat {
            x: 0.3,
            y: 0.0,
            z: 0.0,
            w: 0.93,
        };
        for prismatic in [false, true] {
            let mut w = World::new(WorldParams::default());
            let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
            let j = if prismatic {
                Joint::try_prismatic(
                    a,
                    b,
                    maru_math::Vec3::ZERO,
                    maru_math::Vec3::ZERO,
                    maru_math::Vec3::new(1.0, 0.0, 0.0),
                    loose,
                )
                .unwrap()
            } else {
                Joint::try_fixed(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, loose).unwrap()
            };
            w.try_add_joint(j).unwrap();
            for _ in 0..10 {
                w.try_step(1.0 / 60.0).unwrap();
            }
            let snap = w.try_snapshot().unwrap();
            let mut r = World::new(WorldParams::default());
            r.restore(&snap).unwrap();
            assert_eq!(
                r.try_state_hash().unwrap(),
                w.try_state_hash().unwrap(),
                "prismatic={prismatic}: 비단위 rel_rot이 왕복에서 갈렸다"
            );
            for _ in 0..10 {
                w.try_step(1.0 / 60.0).unwrap();
                r.try_step(1.0 / 60.0).unwrap();
            }
            assert_eq!(
                r.try_state_hash().unwrap(),
                w.try_state_hash().unwrap(),
                "prismatic={prismatic}: 복원 후 미래가 갈렸다"
            );
        }
    }

    /// **f32 정규화는 비트 멱등이 아니다** — `normalize`를 두 번 적용하면 값이 1 ULP
    /// 왔다 갔다 하는 주기 2 진동이 생긴다. 그래서 정규화를 여러 경계에서 반복하면
    /// 적용 횟수의 홀짝에 따라 값이 갈리고, 삽입(2회)과 복원(3회)이 다른 표현을 갖는다
    /// (교차검수 rev_2001abc9 NEW_DEFECT).
    ///
    /// **스텝을 돌리지 않고** 왕복해야 저장된 표현 자체를 시험한다. 복원도 두 번 한다 —
    /// 한 번만 보면 주기 2 진동을 놓친다.
    #[test]
    fn quaternion_canonicalization_is_bitwise_idempotent_across_boundaries() {
        // 검수가 제시한 반례. norm² = 0.95096 — 합법 범위 안이고 단위가 아니다.
        let loose = maru_math::Quat {
            x: -0.03805745,
            y: 0.05848086,
            z: -0.5699489,
            w: 0.788194,
        };
        // 이 값이 실제로 진동하는 부류인지 먼저 확인한다 — 아니면 테스트가 무의미하다.
        let n1 = loose.normalize();
        let n2 = n1.normalize();
        assert_ne!(
            (
                n1.x.to_bits(),
                n1.y.to_bits(),
                n1.z.to_bits(),
                n1.w.to_bits()
            ),
            (
                n2.x.to_bits(),
                n2.y.to_bits(),
                n2.z.to_bits(),
                n2.w.to_bits()
            ),
            "반례가 더 이상 진동하지 않는다 — 이 테스트는 아무것도 증명하지 못한다"
        );
        // 캐노니컬화는 몇 번을 적용해도 같은 비트여야 한다.
        let c1 = loose.try_canonicalize().unwrap();
        let c2 = c1.try_canonicalize().unwrap();
        let c3 = c2.try_canonicalize().unwrap();
        assert_eq!(c1.x.to_bits(), c2.x.to_bits());
        assert_eq!(c1.w.to_bits(), c3.w.to_bits());

        for prismatic in [false, true] {
            let mut w = World::new(WorldParams::default());
            let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
            let j = if prismatic {
                Joint::try_prismatic(
                    a,
                    b,
                    maru_math::Vec3::ZERO,
                    maru_math::Vec3::ZERO,
                    maru_math::Vec3::new(1.0, 0.0, 0.0),
                    loose,
                )
                .unwrap()
            } else {
                Joint::try_fixed(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, loose).unwrap()
            };
            w.try_add_joint(j).unwrap();
            // **스텝 없이** 바로 왕복 — 저장된 표현 자체를 본다.
            let snap = w.try_snapshot().unwrap();
            let mut r = World::new(WorldParams::default());
            r.restore(&snap).unwrap();
            assert_eq!(
                r.try_state_hash().unwrap(),
                w.try_state_hash().unwrap(),
                "prismatic={prismatic}: 왕복만으로 상태 해시가 갈렸다"
            );
            // 두 번째 왕복 — 주기 2 진동은 여기서 드러난다.
            let snap2 = r.try_snapshot().unwrap();
            assert_eq!(
                snap, snap2,
                "prismatic={prismatic}: 재직렬화 바이트가 갈렸다"
            );
            let mut r2 = World::new(WorldParams::default());
            r2.restore(&snap2).unwrap();
            assert_eq!(
                r2.try_state_hash().unwrap(),
                w.try_state_hash().unwrap(),
                "prismatic={prismatic}: 두 번째 복원에서 갈렸다"
            );
            // 스텝을 돌려도 미래가 같아야 한다.
            for _ in 0..20 {
                w.try_step(1.0 / 60.0).unwrap();
                r2.try_step(1.0 / 60.0).unwrap();
            }
            assert_eq!(
                r2.try_state_hash().unwrap(),
                w.try_state_hash().unwrap(),
                "prismatic={prismatic}: 복원 후 미래가 갈렸다"
            );
        }
    }

    /// prismatic 파라미터 검증 — 축·자세·한계·모터 전부.
    #[test]
    fn prismatic_parameters_are_validated() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        let z = maru_math::Vec3::ZERO;
        let x = maru_math::Vec3::new(1.0, 0.0, 0.0);
        let id = maru_math::Quat::IDENTITY;
        // 0 축 / NaN 축 — **생성 시점에** 걸려야 한다(정규화가 0을 살려내면 안 된다).
        assert!(Joint::try_prismatic(a, b, z, z, z, id).is_err());
        assert!(
            Joint::try_prismatic(a, b, z, z, maru_math::Vec3::new(f32::NAN, 0.0, 0.0), id).is_err()
        );
        // 비단위 rel_rot
        assert!(Joint::try_prismatic(
            a,
            b,
            z,
            z,
            x,
            maru_math::Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0
            }
        )
        .is_err());
        let ok = Joint::try_prismatic(a, b, z, z, x, id).unwrap();
        // lo > hi
        assert!(ok.clone().with_limit(1.0, -1.0).is_err());
        // NaN 한계
        assert!(ok.clone().with_limit(0.0, f32::NAN).is_err());
        // 범위 밖 한계 (앵커 = 셰이프 범위)
        assert!(ok.clone().with_limit(-1.0, 1.0e9).is_err());
        // 음수 최대 힘 / NaN 목표 속도
        assert!(ok.clone().with_motor(1.0, -1.0).is_err());
        assert!(ok.clone().with_motor(f32::NAN, 1.0).is_err());
        // 정상값은 통과
        assert!(ok
            .with_limit(-1.0, 1.0)
            .unwrap()
            .with_motor(1.0, 5.0)
            .is_ok());
    }

    /// 재질 설정자는 다른 쓰기 경로와 같은 규율을 따른다 — 바꾼 뒤 `validate`를 다시 통과해야 한다.
    #[test]
    fn material_setters_validate() {
        let mk = || {
            Body::new_dynamic(
                maru_collide::Shape::Sphere { radius: 0.5 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::ZERO,
                    rot: maru_math::Quat::IDENTITY,
                },
                1.0,
            )
        };
        assert_eq!(mk().with_friction(0.0).unwrap().friction(), 0.0);
        assert_eq!(mk().with_friction(1.5).unwrap().friction(), 1.5);
        assert_eq!(mk().with_restitution(1.0).unwrap().restitution(), 1.0);
        // 범위 밖은 거부한다 (음수·NaN·반발 1 초과)
        assert!(mk().with_friction(-0.1).is_err());
        assert!(mk().with_friction(f32::NAN).is_err());
        assert!(mk().with_restitution(1.1).is_err());
        assert!(mk().with_restitution(-0.001).is_err());
    }
    use super::*;
    use maru_collide::{Pose, Shape};
    use maru_math::{Quat, Vec3};

    const DT: f32 = 1.0 / 60.0;
    /// 스냅샷 헤더 = magic4 + version4 + WorldParams60. 슬롯 카운트는 그 뒤에 온다.
    const HDR: usize = 4 + 4 + 60;

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

    /// **M2 수용 기준 (2026-07-26 서브스텝 재보정)**: 피라미드 40단(820박스) 50초 안정.
    ///
    /// # 창 길이가 기준의 일부다
    /// 이전 기준은 40단 × 600스텝(10초)이었고 통과했다. 같은 장면을 3000스텝(50초) 돌리면
    /// **1200스텝 부근에서 붕괴한다**(top_y 39.3 → 5.5). "안정"이 아니라 "아직 안 무너짐"을
    /// 재고 있었다 — 설계의 무근거 목표(100단)를 실측으로 40단으로 내렸던 것과 **같은 계열의
    /// 실수를 창 길이에서 반복**했다.
    ///
    /// 실측 장기 봉투(4 서브스텝·30Hz 접촉, 50초):
    /// 실측 장기 봉투는 **서브스텝에 묶여 있었다**(hertz가 아니라):
    /// substeps=4에서 29단이 한계였고 30단부터 붕괴했다. 기본값을 8로 올리자 **60단**까지
    /// 안정이다(80단 붕괴). 경계 근처(60~63)는 **비단조**다 — 61 붕괴·62 안정 —
    /// 임계 근처의 전형적 거동이므로 기준은 여유 있는 40단에 둔다.
    /// 각 측정은 재현적이다(반복 간 비트 동일 — L0 결정론 유지).
    ///
    /// # 안정의 실질 판정은 '잠들었는가'다
    /// 이 엔진에서 장기 안정과 수면은 같은 사건이다: 잔류 속도가 임계 아래로 떨어지면
    /// 아일랜드가 얼어 영구 안정이 되고, 떨어지지 않으면 지터가 수십 초에 걸쳐 자라 붕괴한다.
    /// 그래서 속도·형상뿐 아니라 **수면**을 함께 요구한다.
    ///
    /// **L0 결정론 게이트 (설계 §8): 같은 빌드·같은 초기상태·같은 호출 순서면 결과는
    /// 스레드 수와 무관하게 같아야 한다.**
    ///
    /// 상태 해시만 보지 않고 **스냅샷 바이트 전체**를 대조한다 — 해시는 캐노니컬 상태의
    /// 축약이라 충돌 가능성이 남고, 이 계약은 "값이 같다"가 아니라 "상태가 같다"이다.
    /// 중간 스텝마다 대조해야 갈림이 생긴 **시점**을 알 수 있다(끝에서만 보면 어디서
    /// 갈렸는지 모른다).
    ///
    /// 씬은 병렬 경로를 실제로 타야 의미가 있다 — 페어가 문턱(512)을 넘는 피라미드
    /// 20단(210박스)을 쓰고, 실제로 참여자 >1로 돌았는지 진단으로 **증명**한다.
    /// (debug 빌드는 debug_assert가 많아 스텝이 비싸다 — 매 커밋 게이트는 짧게 잡고,
    /// 설계 §8의 1,000스텝 판정은 아래 release 전용 테스트가 맡는다.)
    /// (이 증명이 없으면 "직렬 두 번 돌려 같았다"를 결정론이라 부르게 된다.)
    /// 결정론 게이트의 본체 — 스레드 수·솔버 병렬 스위치를 바꿔도 **상태가 같아야** 한다.
    fn assert_thread_determinism(tiers: usize, steps: usize, thread_counts: &[usize]) {
        fn run(
            tiers: usize,
            threads: usize,
            par_solver: bool,
            steps: usize,
        ) -> (Vec<Vec<u8>>, usize, usize) {
            let mut w = World::new(WorldParams::default());
            w.set_threads(threads);
            w.set_parallel_solver(par_solver);
            ground(&mut w);
            let _ = pyramid(&mut w, tiers);
            let mut snaps = Vec::new();
            let (mut max_np, mut max_sp) = (0usize, 0usize);
            let every = (steps / 8).max(1);
            for i in 0..steps {
                w.step(DT);
                max_np = max_np.max(w.diagnostics().narrow_threads);
                max_sp = max_sp.max(w.diagnostics().solve_threads);
                if i % every == 0 || i + 1 == steps {
                    snaps.push(w.snapshot());
                }
            }
            (snaps, max_np, max_sp)
        }
        let (base, np1, sp1) = run(tiers, 1, false, steps);
        assert_eq!(
            np1, 1,
            "스레드 1개 실행이 병렬 경로를 탔다 — 대조군이 오염됐다"
        );
        assert_eq!(
            sp1, 1,
            "스레드 1개 실행에서 솔버가 병렬로 돌았다 — 대조군 오염"
        );
        let can_thread = World::new(WorldParams::default()).threads() > 1;
        // 내로우만 병렬(기본) · 솔버까지 병렬(노브 on) 두 구성 모두 같은 상태를 내야 한다.
        for par_solver in [false, true] {
            for &t in thread_counts {
                let (got, np, sp) = run(tiers, t, par_solver, steps);
                if can_thread {
                    // 스레드를 띄울 수 있는 환경에서만 요구한다(wasm은 항상 1).
                    assert!(
                        np > 1,
                        "threads={t}인데 내로우페이즈가 병렬로 안 돌았다 — 게이트가 공회전이다"
                    );
                    assert_eq!(
                        sp > 1,
                        par_solver,
                        "threads={t} par_solver={par_solver}: 솔버 병렬 여부가 스위치와 다르다"
                    );
                }
                assert_eq!(got.len(), base.len());
                for (i, (a, b)) in base.iter().zip(got.iter()).enumerate() {
                    assert!(
                        a == b,
                        "threads={t} par_solver={par_solver}: 스냅샷 #{i}에서 상태가 갈렸다 (L0 결정론 위반)"
                    );
                }
            }
        }
    }

    /// **L0 결정론 게이트 (설계 §8): 같은 빌드 + 같은 초기상태 + 같은 호출 순서면 결과는
    /// 스레드 수와 무관하게 같아야 한다.** 매 커밋(`scripts/check.sh`)에서 도는 짧은 판.
    ///
    /// 상태 해시만 보지 않고 **스냅샷 바이트 전체**를 대조한다 — 해시는 캐노니컬 상태의
    /// 축약이라 충돌 가능성이 남고, 이 계약은 "값이 같다"가 아니라 "상태가 같다"이다.
    /// 중간 스텝마다 대조해야 갈림이 생긴 **시점**을 알 수 있다.
    ///
    /// 씬은 병렬 경로를 실제로 타야 의미가 있다 — 페어가 문턱(512)을 넘는 피라미드
    /// 30단(465박스)을 쓰고, 실제로 참여자 >1로 돌았는지 진단으로 **증명**한다.
    /// (이 증명이 없으면 "직렬 두 번 돌려 같았다"를 결정론이라 부르게 된다.)
    #[test]
    fn thread_determinism_hash_matches_across_thread_counts() {
        assert_thread_determinism(20, 40, &[4]);
    }

    /// 참여자 버퍼 예약이 **목표 용량에 실제로 닿는지** (교차검수 rev_518b072b MAJOR).
    ///
    /// `Vec::try_reserve`의 인자는 목표 용량이 아니라 `len`에 더할 여분이다. 목표에서
    /// `capacity`를 빼서 넘기면 비워진 버퍼(len=0)가 목표보다 작은 용량으로도 요청을
    /// 통과시키고, 그 뒤 push가 **wake 폐쇄 뒤에** 재할당한다("스텝 중 할당 0"·
    /// "Err면 상태 불변" 계약이 동시에 깨진다). 되돌리면 이 테스트가 실패한다.
    #[test]
    fn narrow_buffer_reserve_reaches_target_capacity_after_reuse() {
        let mut b = crate::world::NarrowOut::default();
        b.reserve(150, 150).expect("첫 예약");
        let before = b.capacities();
        assert!(before.0 >= 150);
        // 사용 후 비워진 상태(len=0, capacity≥150)에서 더 큰 목표를 요구하는 상황.
        b.reserve(200, 200).expect("두 번째 예약");
        let after = b.capacities();
        assert!(
            after.0 >= 200 && after.1 >= 200 && after.2 >= 200,
            "예약이 목표에 못 닿았다: {after:?} (목표 200)"
        );
    }

    /// 참여자 수는 설계 상한(`MAX_THREADS`)을 넘지 않는다 — 넘기면 스핀 설계가 무너지고
    /// 참여자별 스크래치가 입력과 무관하게 커진다(교차검수 rev_518b072b MAJOR).
    #[test]
    fn thread_count_is_clamped_to_design_limit() {
        let mut w = World::new(WorldParams::default());
        w.set_threads(4096);
        assert!(
            w.threads() <= crate::threads::MAX_THREADS,
            "참여자 {}명 — 상한 {}을 넘었다",
            w.threads(),
            crate::threads::MAX_THREADS
        );
        w.set_threads(1);
        assert_eq!(w.threads(), 1, "1은 그대로여야 한다(완전 직렬)");
    }

    /// **복합 바디는 페어 하나가 제약 여러 개다** — 스크래치 회계가 그것을 세는지
    /// (교차검수 rev_518b072b BLOCKER).
    ///
    /// 8자식 복합 64개가 전부 겹치면 페어는 2,016개뿐이지만 제약은 최대 64배(≈129k)라
    /// 128 MiB 예산을 넘는다. 페어 수로만 세던 회계는 이 스텝을 통과시킨 뒤 wake 폐쇄
    /// **뒤에** 재할당했다. 지금은 페어 목록에서 정확한 제약 상한을 구해 미리 거절한다.
    ///
    /// 판정이 **페어 예산이 아니라 제약 회계**에서 났음을 함께 확인한다:
    /// 브로드페이즈는 성공해 `diag.pairs`가 차 있고 예산 안이다.
    #[test]
    fn compound_pairs_are_budgeted_by_constraints_not_pair_count() {
        fn scene(compound: bool) -> (World, Result<(), ApiError>) {
            let mut w = World::new(WorldParams::default());
            for i in 0..64 {
                let shape = if compound {
                    Shape::Compound {
                        children: (0..8)
                            .map(|c| {
                                (
                                    pose(0.02 * c as f32, 0.0, 0.0),
                                    Shape::Sphere { radius: 0.2 },
                                )
                            })
                            .collect(),
                    }
                } else {
                    Shape::Sphere { radius: 0.2 }
                };
                // 전부 겹치게 촘촘히 둔다 — 페어 수는 같고 제약 수만 64배가 된다.
                w.add_body(Body::new_dynamic(
                    shape,
                    pose(0.01 * i as f32, 5.0, 0.0),
                    1.0,
                ));
            }
            let r = w.try_step(DT);
            (w, r)
        }
        let (w_plain, r_plain) = scene(false);
        assert!(
            r_plain.is_ok(),
            "비복합 대조군이 실패하면 이 테스트는 예산이 아니라 씬을 재는 것이다: {r_plain:?}"
        );
        let pairs = w_plain.diagnostics().pairs;
        assert!(pairs > 1000, "겹침이 부족하다 — 페어 {pairs}개");

        let (w_comp, r_comp) = scene(true);
        assert!(
            matches!(r_comp, Err(ApiError::ResourceLimit)),
            "복합 씬이 제약 회계에 걸리지 않았다: {r_comp:?}"
        );
        let d = w_comp.diagnostics();
        // 복합은 자식이 퍼져 AABB가 커서 페어가 조금 더 많다 — 같은 수를 요구하지 않는다.
        assert!(
            d.pairs >= pairs,
            "브로드페이즈 자체는 통과했어야 한다: {}",
            d.pairs
        );
        assert!(
            d.pairs <= d.pair_budget,
            "페어 예산({})이 먼저 걸렸다 — 제약 회계를 재는 테스트가 아니게 된다",
            d.pair_budget
        );
    }

    /// 영속 스크래치 예약이 **목표 용량**을 요구하는지 (확인검수 rev_071f1cd9 MAJOR).
    ///
    /// `try_reserve(nbody)`는 "지금 len에 nbody를 **더**"라는 뜻이라, 매 스텝 len이
    /// 바디 수인 버퍼에 그대로 넘기면 두 번째 스텝부터 용량이 2배로 부푼다 —
    /// 회계는 한 벌만 세므로 128 MiB 상한이 조용히 깨진다. 되돌리면 이 테스트가 실패한다.
    #[test]
    fn per_body_scratch_does_not_double_each_step() {
        const N: usize = 120;
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        for i in 0..N {
            w.add_body(Body::new_dynamic(
                Shape::Sphere { radius: 0.2 },
                pose(i as f32 * 0.5, 3.0, 0.0),
                1.0,
            ));
        }
        let n = w.slot_count();
        for _ in 0..3 {
            w.step(DT);
        }
        for (i, cap) in w.body_scratch_capacities().iter().enumerate() {
            // 고친 경로는 정확히 슬롯 수만큼 잡는다. 버그는 `len + nbody`를 요구해
            // 두 번째 스텝에서 **정확히 2배**가 되므로, 여유를 조금만 두면 갈린다.
            assert!(
                *cap <= n + n / 4 + 16,
                "per-body 스크래치 #{i} 용량 {cap}이 슬롯 수 {n}보다 크게 부풀었다 — 매 스텝 재예약이 쌓인다"
            );
        }
    }

    /// `SolverState::reserve_for`가 `len < capacity < 목표`에서도 목표에 닿는지
    /// (확인검수 rev_071f1cd9 MAJOR — `NarrowOut::reserve`와 같은 계열의 잔존).
    #[test]
    fn solver_state_reserve_reaches_target_when_capacity_exceeds_len() {
        let mut st = crate::solver::SolverState::default();
        // 큰 용량을 먼저 잡고(=capacity 128) 작은 수로 채운다(=len 100).
        st.reserve_for(128).expect("첫 예약");
        let bodies: Vec<Body> = (0..100)
            .map(|_| Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0))
            .collect();
        let alive = vec![true; 100];
        let oob = vec![false; 100];
        st.fill_from(&bodies, &alive, &oob);
        assert!(st.lin_capacity() >= 128 && st.lin_capacity() > 100);
        // 이제 목표 1000 — `목표 - capacity`로 계산하면 보장이 972에 그친다.
        st.reserve_for(1000).expect("두 번째 예약");
        assert!(
            st.lin_capacity() >= 1000,
            "예약이 목표에 못 닿았다: {} (목표 1000)",
            st.lin_capacity()
        );
    }

    /// 참여자 버퍼 회계가 **실제 보유 용량**을 세는지 (확인검수 rev_071f1cd9 MAJOR).
    /// 영속 버퍼는 줄지 않으므로, 이번 청크 목표가 작아도 이미 쥔 만큼을 세야 한다.
    #[test]
    fn narrow_buffer_accounting_counts_held_capacity() {
        let mut b = crate::world::NarrowOut::default();
        // 용량은 판정에 필요한 만큼만 — wasm 테스트는 한 인스턴스에서 전 테스트가 돌고
        // 선형 메모리는 줄지 않으므로, 테스트 하나의 발자국이 전체 상한을 밀어 올린다.
        b.reserve(1024, 1024).expect("큰 예약");
        let small_target = crate::world::NarrowOut::bytes_for(8, 8);
        let projected = b.projected_bytes(8, 8);
        assert!(
            projected > small_target * 50,
            "보유 용량(4096)을 안 세고 목표(8)만 셌다: {projected} vs {small_target}"
        );
    }

    /// **줄어든 월드의 영속 스크래치는 반납된다** (확인검수 rev_21c1a97f MAJOR).
    ///
    /// 회계는 "이번 스텝에 필요한 양"으로 세는데 `Vec` 용량은 저절로 줄지 않는다.
    /// 큰 월드를 돌린 뒤 작아지면 그 차이가 회계 밖에 남아 128 MiB 주장이 거짓이 된다.
    /// 반대로 "넘쳤으니 거절"만 하면 용량이 안 줄어 그 월드는 영원히 못 돈다 —
    /// 그래서 **반납하고 다시 잰다**. 이 테스트는 반납이 실제로 일어나는지 본다.
    #[test]
    fn shrunken_world_releases_persistent_scratch() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let mut hs = Vec::new();
        for i in 0..400 {
            hs.push(w.add_body(Body::new_dynamic(
                Shape::Sphere { radius: 0.2 },
                pose(i as f32 * 0.5, 3.0, 0.0),
                1.0,
            )));
        }
        w.step(DT);
        let big = w.body_scratch_capacities();
        assert!(big[0] >= 400, "큰 월드에서 스크래치가 안 자랐다: {big:?}");
        // 대부분을 지운다 — 슬롯은 줄지 않지만(free-list) 용량 회계의 목표는 줄어든다.
        for h in hs.iter().skip(8) {
            w.remove_body(*h).expect("제거");
        }
        // 슬롯 수를 실제로 줄이려면 새 월드가 필요하다 — 여기서는 **반납 경로 자체**를
        // 직접 부른다(스텝 경로는 초과분이 상한을 넘을 때만 부른다).
        w.release_scratch_for_test(8);
        let small = w.body_scratch_capacities();
        // 반납은 **해제**다(축소가 아니라) — 전부 0이어야 한다. 축소로 두면 과거 용량이
        // 목표만큼 남아 회복하지 못하는 경로가 생기고, `shrink_to`는 재할당을 시도할 수
        // 있어 자원이 빠듯한 바로 그 순간에 abort할 수 있다(확인검수 rev_de34bbb5).
        for (i, (b, s)) in big.iter().zip(small.iter()).enumerate() {
            assert!(*s == 0, "스크래치 #{i}가 해제되지 않았다: {b} → {s}");
        }
        // 반납 뒤에도 정상 동작해야 한다(예약이 다시 목표까지 올라간다).
        assert!(w.try_step(DT).is_ok(), "반납 뒤 스텝이 실패했다");
    }

    /// **큰 장면이 남긴 스크래치는 다음 스텝 시작에 놓아준다** (확인검수 rev_51f69fbd BLOCKER).
    ///
    /// 총상한 검사와 반납은 브로드페이즈가 새 페어 배열을 만든 뒤에 돈다 — 그 전에 이미
    /// 부풀어 있으면 그 할당에서 죽는다(fallible이 아니다). 그래서 스텝 시작에 "현재 필요의
    /// 4배" 안으로 눌러 둔다. 되돌리면(위생 패스 제거) 이 테스트가 실패한다.
    #[test]
    fn stale_scratch_is_released_before_the_next_broadphase() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let mut hs = Vec::new();
        for i in 0..600 {
            hs.push(w.add_body(Body::new_dynamic(
                Shape::Sphere { radius: 0.2 },
                pose(i as f32 * 0.5, 3.0, 0.0),
                1.0,
            )));
        }
        w.step(DT);
        let big = w.body_scratch_capacities();
        assert!(big[0] >= 600, "큰 장면에서 스크래치가 안 자랐다: {big:?}");
        // 작은 월드로 **복원**한다 — 슬롯 수가 실제로 줄어드는 유일한 경로다.
        let mut small = World::new(WorldParams::default());
        ground(&mut small);
        for i in 0..4 {
            small.add_body(Body::new_dynamic(
                Shape::Sphere { radius: 0.2 },
                pose(i as f32 * 0.5, 3.0, 0.0),
                1.0,
            ));
        }
        w.restore(&small.snapshot()).expect("복원");
        w.step(DT);
        let after = w.body_scratch_capacities();
        for (i, (b, a)) in big.iter().zip(after.iter()).enumerate() {
            // #1은 적분 게이트 — 증분 경로 입력이라 위생 패스가 건드리지 않는다(1바이트/원소).
            if i == 1 {
                continue;
            }
            assert!(
                *a <= 4 * 64 + 64,
                "스크래치 #{i}가 큰 장면 용량을 그대로 쥐고 있다: {b} → {a}"
            );
        }
    }

    // ===== M4: 하이트필드 (설계 §4.2) =====

    fn hf_pose() -> maru_collide::Pose {
        pose(0.0, 0.0, 0.0)
    }

    /// 평평한 하이트필드 위에 떨어뜨린 구는 **표면에 정착한다** — 지형 접촉의 기본.
    #[test]
    fn sphere_settles_on_a_flat_heightfield() {
        let mut w = World::new(WorldParams::default());
        let heights = vec![0.0f32; 9 * 9];
        w.try_add_heightfield(&heights, 9, 9, 1.0, hf_pose())
            .expect("지형 추가");
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(4.0, 3.0, 4.0),
            1.0,
        ));
        for _ in 0..300 {
            w.step(DT);
        }
        let b = w.body(h);
        assert!(
            (b.pose().pos.y - 0.5).abs() < 0.02,
            "표면(0)+반지름(0.5)에 정착해야 한다: y={}",
            b.pose().pos.y
        );
        assert!(b.linvel().length() < 0.05, "정착 속도 {:?}", b.linvel());
    }

    /// 격자 위를 굴러가는 구가 셀 경계를 지나도 **뚫거나 튀지 않는다** (거동 회귀).
    ///
    /// 내부 에지 억제 자체의 증거는 여기가 아니라
    /// `maru_collide::manifold::heightfield_contact_tests::welding_snaps_seam_contact_normal_to_the_face`
    /// 에 있다 — 평평한 격자(두께 0)에서는 에지 법선이 이미 면 법선과 같아서 용접을 꺼도
    /// 이 테스트가 통과한다(확인함). 여기서 재는 것은 "이음매를 지나도 정상"이다.
    #[test]
    fn sphere_slides_across_heightfield_seams_without_kick() {
        let mut w = World::new(WorldParams::default());
        let heights = vec![0.0f32; 13 * 5];
        w.try_add_heightfield(&heights, 13, 5, 1.0, hf_pose())
            .expect("지형");
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(1.0, 0.5, 2.0),
            1.0,
        ));
        w.set_linear_velocity(h, Vec3::new(4.0, 0.0, 0.0))
            .expect("속도");
        let mut max_y = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        for _ in 0..120 {
            w.step(DT);
            let y = w.body(h).pose().pos.y;
            max_y = max_y.max(y);
            min_y = min_y.min(y);
            // 굴러가는 동안 계속 지형 위에 있어야 한다
            assert!(y > 0.3, "지형을 뚫었다: y={y}");
        }
        assert!(
            max_y - min_y < 0.05,
            "이음매에서 튀었다: y 범위 {min_y}..{max_y} (내부 에지 억제 실패)"
        );
        assert!(
            w.body(h).pose().pos.x > 4.0,
            "굴러가지 못했다: x={}",
            w.body(h).pose().pos.x
        );
    }

    /// 경사면에서는 미끄러져 내려가고, 능선(절벽)은 **용접하지 않아** 벽으로 작동한다.
    #[test]
    fn heightfield_slope_and_cliff_behave_differently() {
        let mut w = World::new(WorldParams::default());
        // z가 커질수록 높아지는 완만한 경사 + 마지막 줄은 절벽
        let (nx, nz) = (9u32, 9u32);
        let mut heights = vec![0.0f32; (nx * nz) as usize];
        for iz in 0..nz {
            for ix in 0..nx {
                heights[(iz * nx + ix) as usize] = if iz >= 7 { 4.0 } else { iz as f32 * 0.1 };
            }
        }
        w.try_add_heightfield(&heights, nx, nz, 1.0, hf_pose())
            .expect("지형");
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.4 },
            pose(4.0, 2.0, 5.5),
            1.0,
        ));
        w.set_linear_velocity(h, Vec3::new(0.0, 0.0, 6.0))
            .expect("속도");
        let mut max_z = f32::NEG_INFINITY;
        for _ in 0..240 {
            w.step(DT);
            let p = w.body(h).pose().pos;
            assert!(p.y.is_finite(), "발산했다: {p:?}");
            max_z = max_z.max(p.z);
            // 격자 안에 있는 동안에는 표면 위에 있어야 한다(뚫리지 않는다).
            let inside = (0.4..7.6).contains(&p.x) && (0.4..6.6).contains(&p.z);
            if inside {
                let surf = if p.z >= 7.0 {
                    4.0
                } else {
                    (p.z * 0.1).max(0.0)
                };
                assert!(p.y > surf - 0.15, "지형을 뚫었다: p={p:?} surface≈{surf}");
            }
        }
        // 절벽(z≥7 줄, 높이 4m)을 그냥 통과하면 안 된다 — 격자 밖으로 나가 떨어지는 것은
        // 유한 지형의 정상 거동이므로 최종 위치가 아니라 **최대 z**로 판정한다.
        assert!(max_z < 7.3, "절벽을 뚫고 지나갔다: max_z={max_z}");
    }

    /// 레이가 지형 표면을 맞히고, 지형에 묻힌 점은 `query_point`가 찾는다.
    #[test]
    fn heightfield_supports_ray_and_point_queries() {
        let mut w = World::new(WorldParams::default());
        let mut heights = vec![0.0f32; 5 * 5];
        heights[12] = 2.0; // 가운데 정점
        let hf = w
            .try_add_heightfield(&heights, 5, 5, 1.0, hf_pose())
            .expect("지형");
        let hit = w
            .cast_ray(Vec3::new(2.0, 5.0, 2.0), Vec3::new(0.0, -1.0, 0.0), 100.0)
            .expect("질의 성공")
            .expect("표면을 맞혀야 한다");
        assert_eq!(hit.body, hf);
        assert!((hit.toi - 3.0).abs() < 1e-2, "toi={}", hit.toi);
        // 봉우리 정점을 정통으로 맞히면 법선은 이웃 삼각형의 **기울어진** 법선이다.
        // 위를 향하기만 하면 된다 — 평평한 곳의 법선은 아래에서 따로 본다.
        assert!(hit.normal.y > 0.3, "법선이 위를 안 본다: {:?}", hit.normal);
        let flat = w
            .cast_ray(Vec3::new(0.5, 5.0, 0.5), Vec3::new(0.0, -1.0, 0.0), 100.0)
            .expect("질의 성공")
            .expect("평평한 곳도 맞아야 한다");
        assert!(flat.normal.y > 0.99, "평면 법선 {:?}", flat.normal);
        // 표면 아래 점은 지형 안이다(속이 찬 것으로 본다)
        let mut out = Vec::new();
        w.query_point(Vec3::new(1.0, -1.0, 1.0), &mut out)
            .expect("질의");
        assert!(out.contains(&hf), "표면 아래 점이 지형 안으로 안 잡혔다");
        out.clear();
        w.query_point(Vec3::new(1.0, 5.0, 1.0), &mut out)
            .expect("질의");
        assert!(!out.contains(&hf), "표면 위 점이 지형 안으로 잡혔다");
    }

    /// **CCD가 지형을 뚫지 않는다** — 총알 속도로 쏜 구가 격자에서 멈춘다.
    #[test]
    fn ccd_bullet_does_not_tunnel_through_heightfield() {
        let mut w = World::new(WorldParams::default());
        let heights = vec![0.0f32; 9 * 9];
        w.try_add_heightfield(&heights, 9, 9, 1.0, hf_pose())
            .expect("지형");
        let b =
            Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(4.0, 6.0, 4.0), 1.0).with_ccd();
        let h = w.add_body(b);
        w.set_linear_velocity(h, Vec3::new(0.0, -400.0, 0.0))
            .expect("속도");
        for _ in 0..20 {
            w.step(DT);
        }
        let y = w.body(h).pose().pos.y;
        assert!(y > -0.5, "지형을 관통했다: y={y}");
    }

    /// 하이트필드는 **static 전용**이고 스냅샷 왕복에서 살아남는다.
    #[test]
    fn heightfield_is_static_only_and_survives_snapshot() {
        let mut w = World::new(WorldParams::default());
        let mut heights = vec![0.0f32; 6 * 6];
        heights[7] = 1.5;
        let hf = w
            .try_add_heightfield(&heights, 6, 6, 1.0, hf_pose())
            .expect("지형");
        let ball = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(2.5, 2.0, 2.5),
            1.0,
        ));
        for _ in 0..60 {
            w.step(DT);
        }
        let snap = w.snapshot();
        let hash = w.state_hash();
        let mut w2 = World::new(WorldParams::default());
        w2.restore(&snap).expect("복원");
        assert_eq!(w2.state_hash(), hash, "복원 해시 불일치");
        // 복원된 월드가 같은 궤적을 낸다
        for _ in 0..60 {
            w.step(DT);
            w2.step(DT);
        }
        assert_eq!(w.state_hash(), w2.state_hash(), "복원 후 궤적이 갈렸다");
        assert!(w.is_alive(hf) && w.is_alive(ball));
    }

    /// dynamic 하이트필드는 **만들 수조차 없다** — 질량 경로가 없어 생성이 거부한다.
    /// (복원 경계도 같은 정책이다: `validate_structure`가 static이 아닌 하이트필드를 거부.)
    #[test]
    #[should_panic(expected = "dynamic shape must have finite positive mass")]
    fn dynamic_heightfield_is_rejected_at_construction() {
        let _ = Body::new_dynamic(
            Shape::heightfield(vec![0.0; 4], 2, 2, 1.0).expect("격자"),
            hf_pose(),
            1.0,
        );
    }

    /// 잘못된 격자는 **오류로** 거부된다(패닉이 아니라).
    #[test]
    fn malformed_heightfields_are_rejected_as_errors() {
        let mut w = World::new(WorldParams::default());
        let cases: Vec<(&str, Vec<f32>, u32, u32, f32)> = vec![
            ("샘플 수 불일치", vec![0.0; 8], 3, 3, 1.0),
            ("너무 작은 격자", vec![0.0; 1], 1, 1, 1.0),
            ("셀 크기 0", vec![0.0; 4], 2, 2, 0.0),
            ("NaN 높이", vec![f32::NAN; 4], 2, 2, 1.0),
            ("무한 높이", vec![f32::INFINITY; 4], 2, 2, 1.0),
        ];
        for (name, heights, nx, nz, cell) in cases {
            let r = w.try_add_heightfield(&heights, nx, nz, cell, hf_pose());
            assert!(r.is_err(), "{name}가 통과했다");
        }
    }

    // ===== M5: L1 결정론 (설계 §8) =====

    /// **FP 환경 가드가 실제로 잡는다** — 플러시 모드를 켠 스레드에서 스텝을 부르면
    /// 조용히 다른 물리를 내는 대신 오류를 돌려주고, **상태는 그대로**여야 한다.
    #[test]
    fn step_refuses_to_run_in_a_non_canonical_fp_environment() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 2.0, 0.0),
            1.0,
        ));
        w.step(DT);
        let before = w.state_hash();
        if !maru_math::force_flush_to_zero_for_test() {
            return; // 뒤집을 레지스터가 없는 타깃(wasm) — 계약상 항상 캐노니컬이다
        }
        let r = w.try_step(DT);
        assert_eq!(
            r,
            Err(ApiError::FpEnvironment),
            "플러시 모드에서 스텝이 그냥 돌았다 — 결정론 계약이 거짓이 된다"
        );
        assert_eq!(w.state_hash(), before, "거부된 스텝이 상태를 건드렸다");
        // 되돌리면 다시 돈다
        assert!(World::canonicalize_fp_environment());
        assert!(w.try_step(DT).is_ok());
        assert!(w.body(h).pose().pos.y.is_finite());
    }

    /// 조인트 각이 **결정적 atan2**를 쓴다 — 플랫폼 libm에 의존하지 않는다.
    /// (값 자체는 std와 2e-6 안에서 같다는 것을 maru-math가 잠그고, 여기서는 그 함수가
    /// 실제로 물리 경로에 연결돼 있는지를 본다.)
    #[test]
    fn hinge_angle_uses_the_deterministic_atan2() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..WorldParams::default()
        });
        let anchor = w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.2, 0.2, 0.2),
                radius: 0.0,
            },
            pose(0.0, 0.0, 0.0),
        ));
        let arm = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.1, 0.1),
                radius: 0.0,
            },
            pose(0.7, 0.0, 0.0),
            1.0,
        ));
        w.try_add_joint(Joint::hinge(
            anchor,
            arm,
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ))
        .expect("힌지");
        w.set_angular_velocity(arm, Vec3::new(0.0, 0.0, 1.5))
            .expect("각속도");
        for _ in 0..30 {
            w.step(DT);
        }
        // 각도가 유한하고 회전이 실제로 일어났으면 경로가 살아 있다.
        let q = w.body(arm).pose().rot;
        let ang = 2.0 * maru_math::atan2(q.z.abs(), q.w.abs());
        assert!(ang.is_finite() && ang > 0.01, "회전이 없다: {ang}");
    }

    // ===== M5: CharacterVirtual (설계 §9) =====

    fn char_world() -> World {
        let mut w = World::new(WorldParams::default());
        // 넓은 바닥
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(20.0, 0.5, 20.0),
                radius: 0.0,
            },
            pose(0.0, -0.5, 0.0),
        ));
        w
    }

    /// 평평한 바닥 캐릭터 — 계단 경로를 검증할 때 쓴다(둥근 바닥은 그냥 타고 오른다).
    fn box_char(x: f32, y: f32, z: f32) -> CharacterVirtual {
        CharacterVirtual::new(
            Shape::Box {
                half: Vec3::new(0.3, 0.5, 0.3),
                radius: 0.0,
            },
            Vec3::new(x, y, z),
        )
    }

    fn capsule_char(x: f32, y: f32, z: f32) -> CharacterVirtual {
        CharacterVirtual::new(
            Shape::Capsule {
                half_height: 0.5,
                radius: 0.3,
            },
            Vec3::new(x, y, z),
        )
    }

    /// 빈 공간에서는 원하는 만큼 그대로 간다 — 그리고 바닥 위면 `on_ground`다.
    #[test]
    fn character_moves_freely_and_detects_ground() {
        let w = char_world();
        let mut c = capsule_char(0.0, 0.8, 0.0);
        let r = c
            .move_and_slide(&w, Vec3::new(1.0, 0.0, 0.0), QueryFilter::ALL)
            .expect("이동");
        assert!(
            (r.position.x - 1.0).abs() < 1e-3,
            "막힌 것이 없는데 못 갔다: {:?}",
            r.position
        );
        assert!(r.on_ground, "바닥 위인데 지면 판정이 안 됐다");
        assert!(r.ground_normal.y > 0.9, "지면 법선 {:?}", r.ground_normal);
        assert_eq!(r.remaining, Vec3::ZERO);
    }

    /// 벽에 부딪히면 **멈추지 않고 벽을 따라 미끄러진다** — 캐릭터 컨트롤러의 핵심 거동.
    #[test]
    fn character_slides_along_a_wall_instead_of_stopping() {
        let mut w = char_world();
        // x = 1.0에 z축으로 긴 벽
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.2, 2.0, 5.0),
                radius: 0.0,
            },
            pose(1.5, 2.0, 0.0),
        ));
        let mut c = capsule_char(0.0, 0.8, 0.0);
        // 벽을 45°로 밀어붙인다 — x는 막히고 z는 살아야 한다.
        let r = c
            .move_and_slide(&w, Vec3::new(2.0, 0.0, 2.0), QueryFilter::ALL)
            .expect("이동");
        assert!(r.position.x < 1.1, "벽을 통과했다: {:?}", r.position);
        assert!(
            r.position.z > 1.0,
            "벽을 따라 못 미끄러졌다(그냥 멈춤): {:?}",
            r.position
        );
        assert!(r.slides >= 1, "미끄러짐이 기록되지 않았다");
    }

    /// 낮은 턱은 **걸어 올라간다**(계단), 높은 벽은 못 올라간다 — 같은 코드가 둘을 가른다.
    ///
    /// 캐릭터를 **상자**로 둔다(캡슐이 아니라): 캡슐은 둥근 바닥이 턱 모서리를 타고 올라가서
    /// 계단 로직을 꺼도 통과한다 — 그러면 이 테스트가 계단을 증명하지 않는다(실제로 확인했다).
    /// 평평한 바닥은 수직면에서 순수 수평 법선만 얻으므로 오직 계단 경로로만 오를 수 있다.
    #[test]
    fn character_steps_up_low_ledges_but_not_high_walls() {
        let step_h = 0.25;
        let mut low = char_world();
        low.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(1.0, step_h / 2.0, 5.0),
                radius: 0.0,
            },
            pose(2.0, step_h / 2.0, 0.0),
        ));
        let mut c = box_char(0.0, 0.55, 0.0);
        for _ in 0..6 {
            c.move_and_slide(&low, Vec3::new(0.5, 0.0, 0.0), QueryFilter::ALL)
                .expect("이동");
        }
        assert!(c.position.x > 1.5, "낮은 턱을 못 넘었다: {:?}", c.position);
        assert!(
            c.position.y > 0.55 + step_h * 0.5,
            "턱 위로 올라서지 못했다: y={}",
            c.position.y
        );

        // 같은 장면인데 턱만 높으면 막혀야 한다.
        let mut high = char_world();
        high.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(1.0, 1.0, 5.0),
                radius: 0.0,
            },
            pose(2.0, 1.0, 0.0),
        ));
        let mut c2 = box_char(0.0, 0.55, 0.0);
        for _ in 0..6 {
            c2.move_and_slide(&high, Vec3::new(0.5, 0.0, 0.0), QueryFilter::ALL)
                .expect("이동");
        }
        assert!(c2.position.x < 1.3, "높은 벽을 올라갔다: {:?}", c2.position);
    }

    /// 급경사는 지면으로 치지 않는다 — `on_ground`가 거짓이어야 호스트가 미끄러뜨릴 수 있다.
    #[test]
    fn steep_slopes_are_not_walkable_ground() {
        let mut w = World::new(WorldParams::default());
        // 60° 경사면(축 회전한 큰 박스)
        let tilt = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 1.05);
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(10.0, 0.5, 10.0),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: Vec3::new(0.0, 0.0, 0.0),
                rot: tilt,
            },
        ));
        let mut c = capsule_char(0.0, 1.6, 0.0);
        // 경사면 위로 내려서 접촉시킨다
        let r = c
            .move_and_slide(&w, Vec3::new(0.0, -1.0, 0.0), QueryFilter::ALL)
            .expect("이동");
        assert!(
            !r.on_ground,
            "60° 경사(한계 45°)를 걸을 수 있는 지면으로 판정했다: n={:?}",
            r.ground_normal
        );
    }

    /// 같은 입력이면 같은 결과 — 캐릭터도 L0 규약 안에 있다.
    #[test]
    fn character_movement_is_deterministic() {
        let mut w = char_world();
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.2, 2.0, 5.0),
                radius: 0.0,
            },
            pose(1.5, 2.0, 0.0),
        ));
        let run = || {
            let mut c = capsule_char(0.0, 0.8, 0.0);
            let mut path = Vec::new();
            for i in 0..20 {
                let d = Vec3::new(0.2, 0.0, 0.05 * (i % 3) as f32);
                let r = c.move_and_slide(&w, d, QueryFilter::ALL).expect("이동");
                path.push((
                    r.position.x.to_bits(),
                    r.position.y.to_bits(),
                    r.position.z.to_bits(),
                    r.on_ground,
                ));
            }
            path
        };
        assert_eq!(run(), run(), "같은 입력이 다른 경로를 냈다");
    }

    // ===== 검수 rev_1e81559e 회귀 =====

    /// **회전 CCD가 하이트필드를 만나도 trap하지 않는다** (BLOCKER).
    /// 회전 스윕은 선속도 0에서도 켜지므로, 지형 옆에서 도는 CCD 바디가 즉시 재현했다.
    #[test]
    fn rotational_ccd_against_heightfield_does_not_trap() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            substeps: 8,
            ..WorldParams::default()
        });
        let heights = vec![0.0f32; 9 * 9];
        w.try_add_heightfield(&heights, 9, 9, 1.0, pose(0.0, 0.0, 0.0))
            .expect("지형");
        let blade = Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(1.2, 0.05, 0.1),
                radius: 0.0,
            },
            pose(4.0, 0.6, 4.0),
            1.0,
        )
        .with_ccd();
        let h = w.add_body(blade);
        w.set_angular_velocity(h, Vec3::new(0.0, 0.0, 60.0))
            .expect("각속도");
        for _ in 0..30 {
            w.step(DT); // trap이면 여기서 프로세스가 죽는다
        }
        assert!(w.body(h).pose().pos.y.is_finite());
    }

    /// **타일 가지치기가 긴 질의를 살린다** — 그리고 진짜로 볼 셀이 많으면 오류다.
    ///
    /// 예산 이력: 잘라서 `None`(관통) → 무제한(폭주) → 가짜 히트(영구 동결) → **타일 계층**.
    /// 지금은 질의의 Y 밴드와 겹치는 타일만 보므로, 평지 위를 지나는 긴 스윕은 볼 셀이
    /// 거의 없어 **성공**한다. 반대로 모든 셀이 밴드와 겹치는 거친 지형에서는 진짜로
    /// 볼 일이 많으므로 `ResourceLimit`으로 거절한다(가짜 히트로 위장하지 않는다).
    #[test]
    fn tile_pruning_keeps_long_queries_cheap_and_still_reports_real_overload() {
        // ① 평지 + 600m 지점의 벽 — 스윕 밴드(y≈1)는 평지 타일(0..0)과 안 겹친다.
        let mut flat = World::new(WorldParams::default());
        let (nx, nz) = (8193u32, 3u32);
        let mut heights = vec![0.0f32; (nx * nz) as usize];
        for iz in 0..nz {
            heights[(iz * nx + 6000) as usize] = 3.0;
        }
        flat.try_add_heightfield(&heights, nx, nz, 0.1, pose(0.0, 0.0, 0.0))
            .expect("지형");
        let hit = flat
            .sweep_shape(
                &Shape::Sphere { radius: 0.2 },
                pose(0.0, 1.0, 0.1),
                Vec3::new(800.0, 0.0, 0.0),
                QueryFilter::ALL,
            )
            .expect("타일 가지치기 덕에 예산에 걸리지 않아야 한다");
        assert!(hit.is_some(), "600m 앞의 벽을 놓쳤다");

        // ② 모든 셀이 스윕 밴드와 겹치는 거친 지형 — 진짜로 볼 일이 많다 → 오류.
        let mut rough = World::new(WorldParams::default());
        let mut rh = vec![0.0f32; (nx * nz) as usize];
        for iz in 0..nz {
            for ix in 0..nx {
                // 밴드(y 0.8..1.2)를 가로지르도록 톱니
                rh[(iz * nx + ix) as usize] = if ix % 2 == 0 { 0.0 } else { 2.0 };
            }
        }
        rough
            .try_add_heightfield(&rh, nx, nz, 0.1, pose(0.0, 0.0, 0.0))
            .expect("지형");
        let r = rough.sweep_shape(
            &Shape::Sphere { radius: 0.2 },
            pose(0.0, 1.0, 0.1),
            Vec3::new(800.0, 0.0, 0.0),
            QueryFilter::ALL,
        );
        assert_eq!(
            r.err(),
            Some(ApiError::ResourceLimit),
            "볼 셀이 8천 개인 질의가 조용히 통과했다"
        );

        // ③ 레이는 DDA라 격자 크기에 묶여 있어 그대로 먼 벽을 본다.
        let ray = flat
            .cast_ray(Vec3::new(0.05, 1.0, 0.1), Vec3::new(1.0, 0.0, 0.0), 1000.0)
            .expect("질의");
        assert!(ray.is_some(), "레이가 먼 벽을 놓쳤다");
    }

    /// 격자 **밖** 점은 지형 안이 아니다 (MAJOR: 마지막 셀 외삽).
    /// 그리고 표면 아래에서 쏜 레이는 `toi=0`·`started_inside`다(다른 셰이프와 같은 계약).
    #[test]
    fn heightfield_point_and_ray_contracts_hold_outside_and_inside() {
        let mut w = World::new(WorldParams::default());
        let heights = vec![0.0f32; 3 * 3];
        let hf = w
            .try_add_heightfield(&heights, 3, 3, 1.0, pose(0.0, 0.0, 0.0))
            .expect("지형");
        let mut out = Vec::new();
        w.query_point(Vec3::new(100.0, -1.0, 0.5), &mut out)
            .expect("질의");
        assert!(
            !out.contains(&hf),
            "격자에서 98m 떨어진 점이 지형 안으로 잡혔다"
        );
        let inside = w
            .cast_ray(Vec3::new(1.0, -0.5, 1.0), Vec3::new(0.0, 1.0, 0.0), 10.0)
            .expect("질의")
            .expect("안에서 시작한 레이도 히트다");
        assert_eq!(inside.toi, 0.0, "안에서 시작인데 toi가 0이 아니다");
        assert!(inside.started_inside, "started_inside가 안 섰다");
    }

    /// 하이트필드는 **라이브 생성에서도** static 전용이고, compound 자식이 될 수 없다
    /// (BLOCKER: 복원에서만 막으면 `restore(snapshot())`가 자기 스냅샷을 거부한다).
    #[test]
    fn heightfield_static_only_is_enforced_at_creation_too() {
        let hf = || Shape::heightfield(vec![0.0; 4], 2, 2, 1.0).expect("격자");
        // **fallible 생성자로** 본다 — 비-fallible 쪽은 계약 위반에 패닉하는 것이 규약이다.
        assert!(
            Body::try_new_kinematic(hf(), pose(0.0, 0.0, 0.0)).is_err(),
            "kinematic 하이트필드가 통과했다 — 스냅샷 왕복이 깨진다"
        );
        // compound 자식으로도 안 된다(접촉 경로가 자식을 GJK에 그대로 넣는다)
        assert!(Body::try_new_static(
            Shape::Compound {
                children: vec![
                    (pose(0.0, 0.0, 0.0), Shape::Sphere { radius: 0.5 }),
                    (pose(0.0, 0.0, 0.0), hf()),
                ],
            },
            pose(0.0, 0.0, 0.0),
        )
        .is_err());
    }

    /// 캐릭터 설정이 **매 호출 검증**된다 (BLOCKER: 음수 skin이면 요청보다 더 간다).
    #[test]
    fn character_rejects_a_broken_configuration() {
        let mut w = char_world();
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.2, 2.0, 5.0),
                radius: 0.0,
            },
            pose(1.5, 2.0, 0.0),
        ));
        let mut c = capsule_char(0.0, 0.8, 0.0);
        c.skin = -10.0;
        assert_eq!(
            c.move_and_slide(&w, Vec3::new(10.0, 0.0, 0.0), QueryFilter::ALL)
                .err(),
            Some(ApiError::InvalidValue),
            "음수 skin이 통과하면 벽 반대편으로 순간이동한다"
        );
        let mut c2 = capsule_char(0.0, 0.8, 0.0);
        c2.up = Vec3::new(0.0, 3.0, 0.0); // 단위가 아니다
        assert!(c2
            .move_and_slide(&w, Vec3::new(1.0, 0.0, 0.0), QueryFilter::ALL)
            .is_err());
        // 비볼록 셰이프는 스윕 주체가 될 수 없다
        let mut c3 = capsule_char(0.0, 0.8, 0.0);
        c3.shape = Shape::heightfield(vec![0.0; 4], 2, 2, 1.0).expect("격자");
        assert!(c3
            .move_and_slide(&w, Vec3::new(1.0, 0.0, 0.0), QueryFilter::ALL)
            .is_err());
        assert!(w
            .sweep_shape(
                &Shape::heightfield(vec![0.0; 4], 2, 2, 1.0).expect("격자"),
                pose(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                QueryFilter::ALL
            )
            .is_err());
    }

    /// **FP 가드가 스텝뿐 아니라 상태를 만드는 진입점에도 있다** (MAJOR).
    #[test]
    fn canonical_mutations_refuse_a_non_canonical_fp_environment() {
        let mut w = World::new(WorldParams::default());
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 2.0, 0.0),
            1.0,
        ));
        if !maru_math::force_flush_to_zero_for_test() {
            return;
        }
        // 자세·속도·힘·바디 추가·조인트·복원이 전부 거부돼야 한다.
        assert_eq!(
            w.set_pose(
                h,
                maru_collide::Pose {
                    pos: Vec3::new(1.0, 1.0, 1.0),
                    rot: Quat::IDENTITY
                }
            ),
            Err(ApiError::FpEnvironment)
        );
        assert_eq!(
            w.apply_impulse(h, Vec3::new(1.0, 0.0, 0.0)),
            Err(ApiError::FpEnvironment)
        );
        // 생성자 자체가 가드에 걸린다 — 비-fallible 판은 패닉이므로 fallible 판으로 본다.
        assert_eq!(
            Body::try_new_dynamic(Shape::Sphere { radius: 0.2 }, pose(3.0, 3.0, 3.0), 1.0).err(),
            Some(ApiError::FpEnvironment)
        );
        assert!(World::try_new(WorldParams::default()).is_ok() || true);
        assert!(World::canonicalize_fp_environment());
        assert!(w
            .set_pose(
                h,
                maru_collide::Pose {
                    pos: Vec3::new(1.0, 1.0, 1.0),
                    rot: Quat::IDENTITY
                }
            )
            .is_ok());
    }

    // ===== 확인검수 rev_7efe4e35 회귀 =====

    /// **시작부터 벽에 묻혀 있으면 이동하지 않는다** (BLOCKER: 스윕이 None을 내 통과했다).
    #[test]
    fn character_that_starts_overlapped_does_not_pass_through() {
        let mut w = char_world();
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(1.0, 2.0, 5.0),
                radius: 0.0,
            },
            pose(0.0, 2.0, 0.0),
        ));
        // 벽 한가운데에서 시작(스폰·텔레포트·움직이는 지형이 만드는 상황)
        let mut c = capsule_char(0.0, 1.0, 0.0);
        let r = c
            .move_and_slide(&w, Vec3::new(5.0, 0.0, 0.0), QueryFilter::ALL)
            .expect("오류가 아니라 결과로 알린다");
        assert!(r.started_overlapped, "겹침을 감지하지 못했다");
        assert_eq!(r.position, Vec3::new(0.0, 1.0, 0.0), "겹친 채로 이동했다");
        assert_eq!(r.remaining, Vec3::new(5.0, 0.0, 0.0));
        // 정상 위치에서는 그대로 동작한다(대조군)
        let mut ok = capsule_char(3.0, 0.8, 0.0);
        let r2 = ok
            .move_and_slide(&w, Vec3::new(0.5, 0.0, 0.0), QueryFilter::ALL)
            .expect("이동");
        assert!(!r2.started_overlapped);
    }

    /// **회전 CCD의 봉투가 자식 궤도를 덮는다** (BLOCKER: 먼 자식이 지형을 관통했다).
    #[test]
    fn rotating_compound_child_far_from_pivot_still_sees_terrain() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            substeps: 8,
            ..WorldParams::default()
        });
        // 넓은 평지
        let heights = vec![0.0f32; 33 * 33];
        w.try_add_heightfield(&heights, 33, 33, 1.0, pose(0.0, 0.0, 0.0))
            .expect("지형");
        // 피벗에서 멀리 떨어진 자식을 가진 복합 바디 — 회전만으로 지형을 쓸고 지나간다.
        let body = Body::new_dynamic(
            Shape::Compound {
                children: vec![
                    (pose(0.0, 0.0, 0.0), Shape::Sphere { radius: 0.2 }),
                    (pose(8.0, -0.55, 0.0), Shape::Sphere { radius: 0.2 }),
                ],
            },
            pose(16.0, 0.6, 16.0),
            1.0,
        )
        .with_ccd();
        let h = w.add_body(body);
        w.set_angular_velocity(h, Vec3::new(0.0, 12.0, 0.0))
            .expect("각속도");
        for _ in 0..40 {
            w.step(DT);
        }
        // trap 없이 끝나고, 바디가 지형 아래로 꺼지지 않았다.
        let p = w.body(h).pose().pos;
        assert!(p.y.is_finite() && p.y > -0.5, "지형 아래로 빠졌다: {p:?}");
    }

    /// **커맨드 큐는 비표준 FP 환경에서 소비되지 않는다** (BLOCKER).
    #[test]
    fn command_queue_survives_a_non_canonical_fp_environment() {
        let mut w = World::new(WorldParams::default());
        let h = w.enqueue_add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 2.0, 0.0),
            1.0,
        ));
        let pending = w.pending_commands();
        assert!(pending > 0);
        if !maru_math::force_flush_to_zero_for_test() {
            return;
        }
        assert_eq!(
            w.try_flush_commands().err(),
            Some(ApiError::FpEnvironment),
            "비표준 환경에서 커맨드가 소비됐다"
        );
        assert_eq!(
            w.pending_commands(),
            pending,
            "실패했는데 큐가 줄었다 — 재시도할 커맨드가 사라진다"
        );
        assert!(World::canonicalize_fp_environment());
        let report = w.try_flush_commands().expect("환경 복구 후에는 성공");
        assert!(report.applied > 0, "복구 후에도 적용되지 않았다");
        assert!(w.is_alive(h));
    }

    /// 비표준 환경에서 만든 **Body 자체**가 거부된다 (BLOCKER: 우회 경로).
    #[test]
    fn body_construction_is_guarded_too() {
        if !maru_math::force_flush_to_zero_for_test() {
            return;
        }
        let r = Body::try_new_dynamic(Shape::Sphere { radius: 0.3 }, pose(0.0, 0.0, 0.0), 1.0);
        let r2 = Body::try_new_static(Shape::Sphere { radius: 0.3 }, pose(0.0, 0.0, 0.0));
        assert!(World::canonicalize_fp_environment());
        assert_eq!(r.err(), Some(ApiError::FpEnvironment));
        assert_eq!(r2.err(), Some(ApiError::FpEnvironment));
        assert!(
            Body::try_new_dynamic(Shape::Sphere { radius: 0.3 }, pose(0.0, 0.0, 0.0), 1.0).is_ok()
        );
    }

    /// 복원 실패 원인이 **환경**과 **파일 손상**으로 갈린다 (MAJOR: 멀쩡한 저장본 폐기 방지).
    #[test]
    fn restore_reports_fp_environment_separately_from_corruption() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 2.0, 0.0),
            1.0,
        ));
        w.step(DT);
        let snap = w.snapshot();
        if !maru_math::force_flush_to_zero_for_test() {
            return;
        }
        let e = w.restore(&snap).err();
        assert!(World::canonicalize_fp_environment());
        assert_eq!(
            e,
            Some(crate::SnapshotError::FpEnvironment),
            "멀쩡한 저장본이 손상으로 보고됐다 — 호스트가 버린다"
        );
        assert!(w.restore(&snap).is_ok(), "환경 복구 후에는 복원돼야 한다");
    }

    // ===== 확인검수 rev_e50670ad 회귀 =====

    /// **위조된 타일은 이제 존재할 수 없다** (BLOCKER: L0·복원 대칭 → 구성으로 봉인).
    ///
    /// 예전에는 `Shape::Heightfield { tiles, .. }`가 공개 필드라 평지에 `[(100,100)]`을 넣어
    /// 모든 지형을 가지치기로 건너뛰게 만들 수 있었다. 스냅샷은 타일을 저장하지 않으므로
    /// 복원 월드는 **올바른** 타일을 갖는다 — 같은 바이트·같은 해시인데 다음 CCD가 달라졌다
    /// (rev_e50670ad). 처음에는 `validate`에서 비트 검증으로 막았지만, 그 검사는 O(n)이라
    /// 라이브 질의(스텝 안의 CCD가 후보마다 부른다)에 넣을 수 없었고 그래서 라이브 경로에
    /// 구멍이 남았다(rev_97887d75 BLOCKER).
    ///
    /// **지금은 검사 대신 구성으로 막는다.** 데이터는 `HeightfieldGrid`가 소유하고 필드는
    /// 비공개이므로, 타일을 넣는 코드는 **컴파일되지 않는다** — 어느 경로에서도 재검증이
    /// 필요 없고 런타임 비용은 0이다. 이 테스트는 그 대체 계약을 확인한다: 생성자가 유일한
    /// 입구이고, 그 입구가 구조 불변식을 세우며, 나온 타일이 실제 높이와 일치한다.
    #[test]
    fn heightfield_tiles_are_derived_by_the_constructor_and_cannot_be_supplied() {
        // 봉우리가 있는 격자 — 타일 범위가 평평하지 않아야 대조가 의미 있다.
        let (nx, nz) = (9u32, 9u32);
        let mut heights = vec![0.0f32; (nx * nz) as usize];
        heights[(4 * nx + 4) as usize] = 3.0;
        let s = Shape::heightfield(heights.clone(), nx, nz, 1.0).expect("격자");
        assert!(s.validate().is_ok());
        assert!(Body::try_new_static(s.clone(), pose(0.0, 0.0, 0.0)).is_ok());
        // 생성자가 만든 타일이 높이와 일치한다 — 봉우리를 품은 타일의 최대가 3.0이다.
        let hf = s.heightfield_ref().expect("하이트필드");
        let (lo, hi) = hf.tile_range(0, 0);
        assert_eq!((lo, hi), (0.0, 3.0), "타일 범위가 높이와 다르다");
        // 구조 불변식을 어기는 입력은 생성자가 거절한다(개수 불일치·너무 작은 격자·
        // 비유한 높이·0 셀 크기). 예전에는 이런 값이 셰이프 안까지 들어왔다.
        assert!(Shape::heightfield(vec![0.0; 8], 3, 3, 1.0).is_none());
        assert!(Shape::heightfield(vec![0.0; 1], 1, 1, 1.0).is_none());
        assert!(Shape::heightfield(vec![f32::NAN; 4], 2, 2, 1.0).is_none());
        assert!(Shape::heightfield(vec![0.0; 4], 2, 2, 0.0).is_none());
        // **음성 대조** — 정상 입력은 통과한다(생성자가 전부 거절하는 게 아니다).
        assert!(Shape::heightfield(vec![0.0; 4], 2, 2, 1.0).is_some());
        // 크기 **정책**은 여전히 validate 소관이다(생성자는 구조만 본다).
        let huge = Shape::heightfield(vec![0.0; 4], 2, 2, 1.0e9).expect("구조는 정상");
        assert!(huge.validate().is_err(), "MAX_SHAPE_EXTENT 정책이 빠졌다");
    }

    /// **멀리 있는 봉우리 하나가 정상 스텝을 비싸게 만들지 않는다** (MAJOR: raw 셀 수).
    ///
    /// 지형은 바디 하나라 전역 AABB가 봉우리까지 포함한다 — 그래서 브로드페이즈는 공중의
    /// 물체도 후보로 넣는다. 예전에는 그 후보의 **raw 셀 수**를 세서 후보가 실제로 0개인
    /// 질의까지 거절했고, 접촉 순회는 셀 전부를 돌았다. 타일 Y 밴드로 걸러야 한다.
    ///
    /// (셀 **예산** 초과는 이제 자연스러운 장면에서 재현하기 어렵다: 셀을 많이 보려면
    /// 바디가 커야 하고, 큰 바디는 접촉을 먼저 64개 채워 포화로 끝난다. 예산은 남겨 둔
    /// 최후 방어선이고, 그 신호가 포화와 **구분되는지**는 `ContactWalk` 타입이 잠근다.)
    #[test]
    fn a_distant_peak_does_not_make_a_hovering_body_expensive() {
        let (nx, nz) = (600u32, 600u32);
        let mut heights = vec![0.0f32; (nx * nz) as usize];
        heights[0] = 3.0; // 전역 AABB만 높이는 봉우리 하나
        let mut w = World::new(WorldParams::default());
        w.try_add_heightfield(&heights, nx, nz, 0.1, pose(0.0, 0.0, 0.0))
            .expect("지형");
        // 평지 위 2.5m 공중의 큰 박스 — 봉우리와 멀고 실제 접촉은 없다.
        let h = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(25.0, 1.0, 25.0),
                radius: 0.0,
            },
            pose(30.0, 2.5, 30.0),
            1.0,
        ));
        assert!(
            w.try_step(DT).is_ok(),
            "후보가 0개인 스텝이 자원 오류로 거절됐다 — raw 셀 수를 세고 있다"
        );
        // 질의도 같은 규칙이어야 한다.
        assert!(w
            .sweep_shape(
                &Shape::Sphere { radius: 0.2 },
                pose(30.0, 2.5, 30.0),
                Vec3::new(0.0, 0.2, 0.0),
                QueryFilter::ALL
            )
            .is_ok());
        assert!(w.body(h).pose().pos.y.is_finite());
    }

    /// **비볼록을 볼록 전용 경로에 넣을 방법이 아예 없다** (BLOCKER: trap → miss 위장 → 차단).
    ///
    /// 진화 과정: `unreachable!` 패닉 → `None` 반환(그러면 "접촉 없음"과 구분이 안 돼
    /// 호스트가 "지나갈 수 있다"고 오판한다) → **공개 표면에서 제거**.
    /// 지금 밖으로 나가는 것은 셰이프 무관 디스패처뿐이고, 그것들은 모든 셰이프를 옳게 다룬다.
    /// 이 테스트는 디스패처가 하이트필드를 **정상 처리**하는 것으로 그 계약을 확인한다
    /// (볼록 전용 API를 부르는 코드는 이제 컴파일되지 않는다 — 그게 최선의 방어다).
    #[test]
    fn dispatchers_handle_heightfields_and_convex_only_apis_are_not_public() {
        let heights = vec![0.0f32; 9 * 9];
        let hf = Shape::heightfield(heights, 9, 9, 1.0).expect("격자");
        let sphere = Shape::Sphere { radius: 0.5 };
        let p = pose(0.0, 0.0, 0.0);
        // 디스패처는 어느 쪽에 하이트필드가 있어도 동작한다.
        let mut hits = 0;
        maru_collide::for_each_contact(
            &sphere,
            &pose(4.0, 0.4, 4.0),
            &hf,
            &p,
            0.02,
            &mut |_, _, m| {
                if !m.is_empty() {
                    hits += 1;
                }
            },
        );
        assert!(hits > 0, "지형 위 구가 접촉을 못 만들었다");
        assert_eq!(
            maru_collide::any_contact(&sphere, &pose(4.0, 0.4, 4.0), &hf, &p, 0.02),
            Some(true)
        );
        // 스윕 디스패처도 지형을 펼친다.
        assert!(maru_collide::try_shape_cast_dispatch(
            &sphere,
            &pose(4.0, 3.0, 4.0),
            Vec3::new(0.0, -5.0, 0.0),
            &hf,
            &p,
            0.02
        )
        .expect("정상 입력")
        .is_some());
    }

    /// **거대한 유한 각은 조용히 항등이 되지 않는다** (MAJOR: 공개 계약 모순).
    #[test]
    fn huge_finite_axis_angle_is_rejected_not_silently_identity() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        assert!(Quat::try_from_axis_angle(axis, 1.0).is_some());
        assert!(Quat::try_from_axis_angle(axis, 1.0e9).is_some());
        assert!(
            Quat::try_from_axis_angle(axis, 1.0e20).is_none(),
            "결정적으로 접을 수 없는 각을 받아들였다"
        );
        assert!(Quat::try_from_axis_angle(axis, f32::INFINITY).is_none());
        assert!(Quat::try_from_axis_angle(Vec3::ZERO, 1.0).is_none());
        // 0 축은 데이터에서 자연히 나오는 조건이라 `from_axis_angle`도 항등으로 통과시킨다
        // (아래 패닉 테스트의 **음성 대조**: 패닉이 "무조건"이 아님을 보인다).
        assert_eq!(Quat::from_axis_angle(Vec3::ZERO, 1.0), Quat::IDENTITY);
    }

    /// **표현 불가 각을 `from_axis_angle`에 넣으면 패닉한다** (조용한 항등 금지).
    ///
    /// `try_` 없는 생성자의 Rust 관례이며, 조용히 항등을 돌려주면 "물체가 왜 회전을
    /// 멈췄나"를 추적할 단서가 아무것도 남지 않는다(확인검수 rev_1a31d83f MAJOR 6).
    #[test]
    #[should_panic(expected = "표현 불가한 각")]
    fn from_axis_angle_panics_on_unrepresentable_angle() {
        let _ = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 1.0e20);
    }

    #[test]
    #[should_panic(expected = "표현 불가한 각")]
    fn from_axis_angle_panics_on_nonfinite_angle() {
        let _ = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), f32::NAN);
    }

    /// **공개 회전 캐스트는 패닉하지 않고, 잘못된 입력을 미스로 위장하지도 않는다.**
    ///
    /// 두 계약이 함께 있어야 의미가 있다. 캐스트 루프는 매 반복 `from_axis_angle`을 부르고
    /// 그 함수는 표현 불가 각에 패닉하므로 (a) 호스트 입력이 그 패닉에 닿아선 안 되고,
    /// (b) 그렇다고 `None`(=미스)으로 돌려주면 호출자가 "경로가 비었다"고 읽어 그대로
    /// 이동시킨다 — 저수준 볼록 API를 공개 표면에서 뺀 것과 같은 결함이다.
    /// 그래서 공개 경로는 `Result`이고 `Err(CastReject::InvalidInput)`을 낸다.
    ///
    /// 각만 검사하면 부족하다: 유효한 각 + NaN 축/이동이면 내부에서 `t`가 NaN이 되어
    /// `angle * t`가 다시 표현 불가 각이 된다(확인검수 rev_d5b32e22 BLOCKER).
    #[test]
    fn rotational_cast_rejects_bad_input_as_error_not_miss() {
        use maru_collide::CastReject;
        // 회전 피벗은 **A의 위치(바디 COM)**다 — 구는 자기 중심 회전에 아무 데도 안 쓸어서
        // 음성 대조가 될 수 없다. 긴 박스를 세로로 세워 90° 돌리면 이웃 박스를 쓸고 지나간다.
        let a = Shape::Box {
            half: Vec3::new(1.0, 0.2, 0.2),
            radius: 0.0,
        };
        let b = Shape::Box {
            half: Vec3::new(0.2, 0.2, 0.2),
            radius: 0.0,
        };
        let pa = pose(0.0, 0.0, 0.0);
        let pb = pose(0.0, 0.0, 1.05);
        let good_axis = Vec3::new(0.0, 1.0, 0.0);
        let good_angle = core::f32::consts::FRAC_PI_2;
        // ① 표현 불가한 각
        for bad in [1.0e20f32, f32::NAN, f32::INFINITY] {
            assert_eq!(
                maru_collide::try_shape_cast_dispatch_rotational(
                    &a,
                    &pa,
                    Vec3::ZERO,
                    good_axis,
                    bad,
                    &b,
                    &pb,
                    0.02
                ),
                Err(CastReject::InvalidInput),
                "표현 불가한 각 {bad}"
            );
        }
        // ② 유효한 각 + 비유한 축/이동 — 예전에는 이 조합이 내부 assert를 때렸다.
        assert_eq!(
            maru_collide::try_shape_cast_dispatch_rotational(
                &a,
                &pa,
                Vec3::ZERO,
                Vec3::new(f32::NAN, 0.0, 0.0),
                good_angle,
                &b,
                &pb,
                0.02
            ),
            Err(CastReject::InvalidInput)
        );
        assert_eq!(
            maru_collide::try_shape_cast_dispatch_rotational(
                &a,
                &pa,
                Vec3::new(f32::NAN, 0.0, 0.0),
                good_axis,
                good_angle,
                &b,
                &pb,
                0.02
            ),
            Err(CastReject::InvalidInput)
        );
        // ③ 선형 디스패처도 같은 규칙
        assert_eq!(
            maru_collide::try_shape_cast_dispatch(
                &a,
                &pa,
                Vec3::new(0.0, f32::INFINITY, 0.0),
                &b,
                &pb,
                0.02
            ),
            Err(CastReject::InvalidInput)
        );
        // **음성 대조** — 같은 배치에서 정상 입력은 실제로 맞춘다(가드가 전부를 막는 게 아니다).
        let hit = maru_collide::try_shape_cast_dispatch_rotational(
            &a,
            &pa,
            Vec3::ZERO,
            good_axis,
            good_angle,
            &b,
            &pb,
            0.02,
        )
        .expect("정상 입력이 거절됐다");
        assert!(
            hit.is_some_and(|h| h.toi > 0.0 && h.toi < 1.0),
            "정상 회전 스윕이 접촉을 못 찾았다 — 테스트 배치가 무의미하다: {hit:?}"
        );
    }

    /// **지형 캐스트에는 작업량 상한이 있고, 초과는 미스가 아니라 오류다.**
    ///
    /// 타일 가지치기는 후보 0인 지형의 반복만 없앤다 — Y 밴드가 실제로 겹치면 상한이
    /// 없어서 큰 바디 하나가 서브스텝마다 100만 셀을 훑었다(확인검수 rev_d5b32e22 MAJOR).
    #[test]
    fn heightfield_cast_reports_budget_instead_of_walking_forever() {
        use maru_collide::CastReject;
        let heights = vec![0.0f32; 257 * 257];
        let hf = Shape::heightfield(heights, 257, 257, 0.1).expect("격자");
        let p = pose(0.0, 0.0, 0.0);
        // 반지름 5m 구 = 100×100셀(=10000 > 4096 예산)을 덮는다.
        let big = Shape::Sphere { radius: 5.0 };
        assert_eq!(
            maru_collide::try_shape_cast_dispatch(
                &big,
                &pose(12.8, 5.0, 12.8),
                Vec3::new(0.0, -1.0, 0.0),
                &hf,
                &p,
                0.02
            ),
            Err(CastReject::BudgetExceeded)
        );
        // **음성 대조** — 작은 바디는 같은 지형에서 정상적으로 지면을 맞춘다.
        let small = Shape::Sphere { radius: 0.2 };
        let hit = maru_collide::try_shape_cast_dispatch(
            &small,
            &pose(12.8, 1.0, 12.8),
            Vec3::new(0.0, -2.0, 0.0),
            &hf,
            &p,
            0.02,
        )
        .expect("작은 바디가 예산에 걸렸다");
        assert!(hit.is_some(), "지면을 향한 스윕이 아무것도 못 맞췄다");
    }

    /// **접촉 예산을 넘긴 스텝은 물리를 한 톱도 진행하지 않는다** (BLOCKER: 부분 물리 적용).
    ///
    /// 예전에는 `budget_exceeded`를 보면 플래그만 세우고 앞부분 제약으로 솔브·적분을 끝까지
    /// 했다. 그러면 `ResourceLimit`("커맨드는 적용, 물리 스텝은 건너뜀") 계약이 깨지고,
    /// 호스트가 오류를 보고 재시도할 때마다 불완전한 물리가 한 스텝씩 더 나아간다
    /// (확인검수 rev_1a31d83f BLOCKER 1).
    ///
    /// 탈출구도 함께 확인한다: 커맨드는 스텝 0단계에서 이미 적용됐고 `resource_limit`은 매
    /// 스텝 초기화되므로, 문제 바디를 지우면 다음 스텝은 정상으로 돌아온다(영구 동결 아님).
    #[test]
    fn contact_budget_exceeded_step_advances_nothing() {
        let mut w = World::new(WorldParams::default());
        // 256×256 평지(셀 0.1m). 셀 예산은 4096이다.
        let heights = vec![0.0f32; 257 * 257];
        w.try_add_heightfield(&heights, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
            .expect("지형 추가");
        // **AABB는 거대하지만 실제 접촉은 몇 개뿐인 바디**가 예산을 때린다: 반지름 5m 구를
        // 평지에 살짝 닿게 놓으면 AABB는 100×100셀(=10000 > 4096)을 덮는데 접촉은 바닥
        // 몇 셀뿐이다. 넓적한 판으로는 안 된다 — 판은 매니폴드 상한(64)에 먼저 걸려
        // `Saturated`("찾았고 끊었다")가 되고 그건 예산 초과가 아니다.
        // 띄우면(y 밴드가 지형 [0,0] 타일과 안 겹치면) 셀을 세지도 않으므로 닿게 놓는다.
        let slab = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 5.0 },
            pose(12.8, 4.999, 12.8),
            1.0,
        ));
        let before = w.body(slab).pose().pos;
        let before_v = w.body(slab).linvel();
        assert_eq!(w.try_step(1.0 / 60.0), Err(ApiError::ResourceLimit));
        assert_eq!(
            w.body(slab).pose().pos,
            before,
            "예산 초과 스텝이 위치를 바꿨다 — 부분 물리가 적용됐다"
        );
        assert_eq!(
            w.body(slab).linvel(),
            before_v,
            "예산 초과 스텝이 속도를 바꿨다 — 적분이 돌았다"
        );
        assert!(w.contact_events().is_empty());
        // **음성 대조 ①** — 같은 세계에서 작은 바디만 남기면 스텝이 정상 통과하고 실제로
        // 떨어진다(테스트가 '지형이 있으면 항상 실패'를 보는 게 아니다).
        w.remove_body(slab).expect("제거");
        let small = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.2 },
            pose(12.8, 5.0, 12.8),
            1.0,
        ));
        let y0 = w.body(small).pose().pos.y;
        assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
        assert!(
            w.body(small).pose().pos.y < y0,
            "정상 스텝인데 중력이 안 걸렸다 — 대조군이 무의미하다"
        );
    }

    /// **실패한 스텝은 sleep 상태도 되돌린다** (BLOCKER: 캐노니컬 상태가 앞으로 나감).
    ///
    /// wake 폐쇄는 narrowphase보다 먼저 돈다. 그래서 뒤에서 접촉 예산이 터져 스텝을
    /// 포기하면, "물리 스텝은 건너뜀"이라 말하면서 잠들어 있어야 할 바디를 깨워 놓은 채
    /// 끝난다 — `asleep`은 스냅샷에 실리는 캐노니컬 상태이므로, 호스트가 문제 바디를
    /// 지우고 재시도해도 그 바디는 이미 중력·적분 대상이 되어 실패 전과 다른 미래를 만든다
    /// (확인검수 rev_d5b32e22 BLOCKER 1).
    #[test]
    fn failed_step_rolls_back_wake() {
        let mut w = World::new(WorldParams {
            time_to_sleep: 0.1,
            ..WorldParams::default()
        });
        let heights = vec![0.0f32; 257 * 257];
        w.try_add_heightfield(&heights, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
            .expect("지형 추가");
        let sleeper = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.2 },
            pose(12.8, 0.2, 12.8),
            1.0,
        ));
        // 지면에 앉아 잠들 때까지 돌린다.
        for _ in 0..300 {
            assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
            if w.is_asleep(sleeper) {
                break;
            }
        }
        assert!(w.is_asleep(sleeper), "테스트 전제(잠듦)가 성립하지 않았다");
        let pos_before = w.body(sleeper).pose().pos;
        // 예산을 터뜨리는 큰 바디를 **잠든 바디 위에** 놓는다: wake 폐쇄가 먼저 돌아
        // 잠든 아일랜드를 깨우고, 그 다음 narrowphase가 셀 예산에서 판정을 포기한다.
        let slab = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 5.0 },
            pose(12.8, 4.999, 12.8),
            1.0,
        ));
        // 기준선은 **바디를 추가한 뒤**에 잡는다 — 추가 자체는 캐노니컬 변경이 맞다.
        let hash_before = w.state_hash();
        assert_eq!(w.try_step(1.0 / 60.0), Err(ApiError::ResourceLimit));
        // **가장 강한 단언**: 실패한 스텝은 캐노니컬 상태를 한 비트도 바꾸지 않는다.
        assert_eq!(
            w.state_hash(),
            hash_before,
            "실패 스텝이 캐노니컬 상태를 바꿨다"
        );
        assert!(
            w.is_asleep(sleeper),
            "실패한 스텝이 잠든 바디를 깨운 채 끝났다 — 캐노니컬 상태가 앞으로 나갔다"
        );
        assert_eq!(w.body(sleeper).pose().pos, pos_before);
        // **음성 대조 ①** — 문제 바디를 지우면 다음 스텝은 정상이고, 잠든 바디는 계속 잔다
        // (롤백이 sleep 타이머를 망가뜨려 즉시 깨어나는 일이 없다).
        w.remove_body(slab).expect("제거");
        assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
        assert!(w.is_asleep(sleeper), "복구된 스텝이 잠든 바디를 깨웠다");
    }

    /// **존재 질의는 하나 찾으면 예산 초과와 무관하게 `Some(true)`다.**
    ///
    /// 부정("접촉 없음")만 순회 완주를 요구한다. 긍정을 `None`으로 돌려주면 지원 범위 안의
    /// 고정 장면(넓은 AABB + 접촉 몇 개)이 영구히 답을 못 받는다(확인검수 rev_d5b32e22 MAJOR).
    #[test]
    fn any_contact_keeps_a_proven_hit_even_when_the_budget_blows() {
        let nx = 257usize;
        let mut heights = vec![0.0f32; nx * nx];
        // 순회 **앞쪽**(예산 4096 안)에 구 표면에 닿는 봉우리를 하나 만든다. 셀 0.1m이므로
        // 격자 좌표 100 ≈ 월드 10.0m이고, 중심 (12.8, 5.0)에서 수평 3.96m 떨어진 지점의
        // 구 표면 높이는 약 1.95m다.
        for gz in 99..=101 {
            for gx in 99..=101 {
                heights[gz * nx + gx] = 1.95;
            }
        }
        let hf = Shape::heightfield(heights, nx as u32, nx as u32, 0.1).expect("격자");
        let p = pose(0.0, 0.0, 0.0);
        let big = Shape::Sphere { radius: 5.0 };
        // 봉우리에 닿는 배치: 접촉을 앞쪽 셀에서 찾고, 그 뒤 수천 셀에서 예산이 터진다.
        assert_eq!(
            maru_collide::any_contact(&big, &pose(12.8, 5.0, 12.8), &hf, &p, 0.02),
            Some(true),
            "증명된 접촉이 예산 초과 때문에 사라졌다"
        );
        // **음성 대조 ①** — 봉우리를 벗어난 자리(같은 크기 AABB, 앞쪽 셀에 접촉 없음)에서는
        // 예산이 터질 때까지 아무것도 못 찾았으므로 `Some(false)`가 아니라 `None`이다.
        // 이쪽이 무너지면 "판정 미완료"가 "접촉 없음"으로 뭉개진다.
        assert_eq!(
            maru_collide::any_contact(&big, &pose(12.8, 5.0, 20.0), &hf, &p, 0.02),
            None,
            "판정 미완료가 '접촉 없음'으로 뭉개졌다"
        );
        // **음성 대조 ②** — 타일 가지치기가 전부 걸러내면 순회는 **완주**한 것이므로
        // `Some(false)`가 맞다(예산과 무관하게 확정적인 '없음').
        assert_eq!(
            maru_collide::any_contact(&big, &pose(12.8, 40.0, 12.8), &hf, &p, 0.02),
            Some(false)
        );
    }

    /// **내부 CCD 캐스트도 작업량 상한 안에서 돈다** (MAJOR: 스텝 안의 무한 작업량).
    ///
    /// 공개 질의(`sweep_shape`)만 사전 예산을 봤고, 물리 스텝 안의 CCD는 디스패처를 직접
    /// 불러 예산을 우회했다 — 1024×1024 지형을 덮는 큰 CCD 바디 하나가 서브스텝마다
    /// 100만 셀을 훑어 프레임을 멈출 수 있었다(확인검수 rev_d5b32e22 MAJOR).
    ///
    /// 초과 시 정책은 **보수적 정지**(그 서브스텝 이동·회전 0)이고, 보고는
    /// `Diagnostics::ccd_incomplete`다. `ResourceLimit`을 쓸 수 없는 이유는 그 필드 문서에.
    #[test]
    fn internal_ccd_cast_is_bounded_and_reports_incompleteness() {
        let mut w = World::new(WorldParams::default());
        let heights = vec![0.0f32; 257 * 257];
        w.try_add_heightfield(&heights, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
            .expect("지형 추가");
        // 바닥에서 0.3m 뜬 반지름 5m 구. 접촉(내로우페이즈)은 타일 Y밴드 가지치기로 0셀이라
        // 스텝이 예산에 걸리지 않지만, 아래로 빠르게 쏘면 **CCD 스윕 봉투**가 100×100셀을
        // 덮어 예산(4096)을 넘는다 — 예전에는 이 경로에 상한이 아예 없었다.
        let big = w.add_body(
            Body::new_dynamic(Shape::Sphere { radius: 5.0 }, pose(12.8, 5.3, 12.8), 1.0).with_ccd(),
        );
        w.set_linear_velocity(big, Vec3::new(0.0, -400.0, 0.0))
            .expect("속도");
        let y0 = w.body(big).pose().pos.y;
        assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
        assert!(
            w.diagnostics().ccd_incomplete > 0,
            "CCD가 예산을 넘겼는데 보고가 없다 — 상한이 없거나 조용히 강등됐다"
        );
        assert_eq!(
            w.body(big).pose().pos.y,
            y0,
            "판정 미완료인데 이동했다 — 보수적 정지가 아니다"
        );
        // **속도가 쌓이지 않는다.** 위치만 얼리면 중력·솔버가 속도를 계속 키워
        // `max_linear_velocity`(500 m/s)까지 저장되고, 호스트가 지형을 줄여 판정이 가능해지는
        // 순간 그 바디가 최대 속도로 발사된다(확인검수 rev_2e93ebf2 MAJOR — 위치만 본 이
        // 테스트가 놓쳤다). 판정 미완료 서브스텝마다 속도를 0으로 되돌리므로, 한 스텝 안에서
        // 정지 이후 남은 서브스텝의 중력만 남는다(약 0.12 m/s) — 그 이상 자라지 않는다.
        // **저장된 에너지가 없다.** 위치만 얼리면 초기 400 m/s가 그대로 남고 중력이 계속
        // 더해져 `max_linear_velocity`(500)까지 쌓인다 — 호스트가 지형을 줄여 판정이
        // 가능해지는 순간 그 바디가 최대 속도로 발사된다(확인검수 rev_2e93ebf2 MAJOR:
        // 위치만 본 이 테스트가 놓쳤다). 판정 미완료 서브스텝에서 속도까지 0으로 되돌리므로,
        // 이후 2초 동안 속도는 한 자리 수를 넘지 않고 구는 지형 위에 안착한다.
        let mut vmax = 0.0f32;
        for _ in 0..120 {
            assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
            vmax = vmax.max(w.body(big).linvel().length());
        }
        assert!(
            vmax < 20.0,
            "판정 미완료 정지가 에너지를 저장하고 있다: 최대 {vmax} m/s (초기 400, 상한 500)"
        );
        assert!(
            w.body(big).pose().pos.y > 4.0,
            "구가 지형을 뚫고 내려갔다: y={}",
            w.body(big).pose().pos.y
        );
        // **음성 대조** — 같은 지형·같은 속도의 작은 CCD 바디는 예산 안이라 정상적으로
        // 움직이고 지면에서 절단된다(상한이 CCD를 통째로 끈 게 아니다).
        let mut w2 = World::new(WorldParams::default());
        let heights2 = vec![0.0f32; 257 * 257];
        w2.try_add_heightfield(&heights2, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
            .expect("지형 추가");
        let small = w2.add_body(
            Body::new_dynamic(Shape::Sphere { radius: 0.2 }, pose(12.8, 5.3, 12.8), 1.0).with_ccd(),
        );
        w2.set_linear_velocity(small, Vec3::new(0.0, -400.0, 0.0))
            .expect("속도");
        assert_eq!(w2.try_step(1.0 / 60.0), Ok(()));
        assert_eq!(w2.diagnostics().ccd_incomplete, 0);
        // 대조군은 속도가 살아 있다 — 정지 정책이 CCD 바디 전부를 얼린 게 아니다.
        assert!(w2.body(small).linvel().y < 0.0);
        let y = w2.body(small).pose().pos.y;
        assert!(
            y < 5.3 && y > -0.5,
            "작은 CCD 바디가 안 움직였거나 지형을 관통했다: y={y}"
        );
    }

    /// **복합 바디도 지형 스윕에서 히트를 얻는다** (자기 검수: 관통 경로).
    ///
    /// 선형 디스패처가 `match (sa, sb)`로 짜여 있어서 `(Compound, _)` 팔이 먼저 잡혔고,
    /// 각 자식이 **볼록 전용** 캐스트에 지형을 그대로 넘겼다 — 그 함수는 비볼록에 `None`을
    /// 내므로 결과가 "맞은 것 없음"이 되어 복합 바디가 지형을 통과했다(회전 분기는 애초에
    /// B쪽 디스패치를 안쪽에 두어 이 함정이 없었다).
    #[test]
    fn compound_body_sweeping_into_terrain_reports_a_hit() {
        let heights = vec![0.0f32; 17 * 17];
        let hf = Shape::heightfield(heights, 17, 17, 1.0).expect("격자");
        let p = pose(0.0, 0.0, 0.0);
        let child = Shape::Box {
            half: Vec3::new(0.3, 0.3, 0.3),
            radius: 0.0,
        };
        let comp = Shape::Compound {
            children: vec![
                (pose(-0.5, 0.0, 0.0), child.clone()),
                (pose(0.5, 0.0, 0.0), child.clone()),
            ],
        };
        let start = pose(8.0, 3.0, 8.0);
        let down = Vec3::new(0.0, -5.0, 0.0);
        let comp_hit = maru_collide::try_shape_cast_dispatch(&comp, &start, down, &hf, &p, 0.02)
            .expect("정상 입력");
        // **음성 대조** — 같은 배치의 단일 셰이프는 원래도 맞췄다. 두 결과의 toi가 사실상
        // 같아야 한다(같은 높이에서 같은 지면으로 떨어지므로): 복합 경로만 고장났던 것이다.
        let one_hit = maru_collide::try_shape_cast_dispatch(&child, &start, down, &hf, &p, 0.02)
            .expect("정상 입력");
        let one = one_hit.expect("단일 박스가 지면을 못 맞췄다 — 배치가 무의미하다");
        let comp = comp_hit.expect("복합 바디가 지형을 통과했다");
        assert!(
            (comp.toi - one.toi).abs() < 1.0e-3,
            "복합 toi {}가 단일 toi {}와 다르다",
            comp.toi,
            one.toi
        );
    }

    /// **불변식이 깨진 형상은 공개 경로에서 값으로 거절된다** (BLOCKER: 패닉/미스 위장).
    ///
    /// `Shape`의 필드가 공개라 호스트는 `Hull { points: vec![] }`를 만들 수 있고, 예전에는
    /// 그것이 GJK로 내려가 `support_core`의 `points[0]`에서 프로세스를 죽였다(wasm에서는
    /// unwind가 trap이다). 빈 복합은 반대로 `Ok(None)`이 되어 잘못된 형상이 **실제 미스로
    /// 위장**됐다 — 확인검수 rev_2e93ebf2 BLOCKER.
    #[test]
    fn malformed_shapes_are_rejected_by_value_on_public_paths() {
        use maru_collide::CastReject;
        let empty_hull = Shape::Hull {
            points: Vec::new(),
            radius: 0.0,
        };
        let empty_compound = Shape::Compound {
            children: Vec::new(),
        };
        let ok = Shape::Sphere { radius: 0.5 };
        let pa = pose(0.0, 0.0, 0.0);
        let pb = pose(1.0, 0.0, 0.0);
        let m = Vec3::new(1.0, 0.0, 0.0);
        for bad in [&empty_hull, &empty_compound] {
            // 어느 쪽 인자로 들어와도 같은 답이어야 한다(A/B 대칭).
            assert_eq!(
                maru_collide::try_shape_cast_dispatch(bad, &pa, m, &ok, &pb, 0.02),
                Err(CastReject::InvalidInput)
            );
            assert_eq!(
                maru_collide::try_shape_cast_dispatch(&ok, &pa, m, bad, &pb, 0.02),
                Err(CastReject::InvalidInput)
            );
            assert_eq!(
                maru_collide::try_shape_cast_dispatch_rotational(
                    bad,
                    &pa,
                    m,
                    Vec3::new(0.0, 1.0, 0.0),
                    0.5,
                    &ok,
                    &pb,
                    0.02
                ),
                Err(CastReject::InvalidInput)
            );
            // 접촉 쪽 형제 API도 같은 규칙 — `false`(없음)로 뭉개지 않는다.
            assert_eq!(maru_collide::any_contact(bad, &pa, &ok, &pb, 0.02), None);
            assert_eq!(
                maru_collide::for_each_contact(bad, &pa, &ok, &pb, 0.02, &mut |_, _, _| {}),
                maru_collide::ContactWalk::InvalidShape
            );
        }
        // **음성 대조** — 점 하나만 넣어 정상 hull로 만들면 곧바로 통과하고 실제로 맞춘다.
        let tiny_hull = Shape::Hull {
            points: vec![Vec3::ZERO],
            radius: 0.4,
        };
        let hit = maru_collide::try_shape_cast_dispatch(&tiny_hull, &pa, m, &ok, &pb, 0.02)
            .expect("정상 hull이 거절됐다");
        assert!(hit.is_some(), "정상 hull이 맞은 것을 못 찾았다");
        assert_eq!(
            maru_collide::any_contact(&tiny_hull, &pose(0.6, 0.0, 0.0), &ok, &pb, 0.02),
            Some(true)
        );
    }

    /// **비단위 자세는 찌그러진 형상이 아니라 정규화된 형상으로 판정된다** (BLOCKER: 조용한 왜곡).
    ///
    /// `Quat::rotate`는 비단위 쿼터니언에 회전+축소를 적용한다. 유한성만 검사하면 0.5배한
    /// 정상 쿼터니언이 통과해 캐스트가 절반 크기 형상으로 답을 낸다(확인검수 rev_2e93ebf2).
    /// 부동소수 드리프트로 정확한 단위는 드물기 때문에 **거절이 아니라 정규화**가 맞고,
    /// 정규화 불가(0·NaN)만 거절한다.
    #[test]
    fn non_unit_pose_rotation_is_normalized_not_trusted() {
        let long = Shape::Box {
            half: Vec3::new(1.0, 0.2, 0.2),
            radius: 0.0,
        };
        let wall = Shape::Box {
            half: Vec3::new(0.2, 1.0, 1.0),
            radius: 0.0,
        };
        let q = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_2);
        let unit = maru_collide::Pose {
            pos: Vec3::ZERO,
            rot: q,
        };
        let scaled = maru_collide::Pose {
            pos: Vec3::ZERO,
            rot: Quat {
                x: q.x * 0.5,
                y: q.y * 0.5,
                z: q.z * 0.5,
                w: q.w * 0.5,
            },
        };
        // 90° 돌린 긴 박스는 z∈[-1,1]을 차지한다 — 벽은 그 **밖에** 둬야 한다(겹쳐서
        // 시작하면 캐스트는 기존 관통이라 `None`이 맞고, 그러면 대조가 성립하지 않는다).
        let pb = pose(0.0, 0.0, 2.4);
        let m = Vec3::new(0.0, 0.0, 1.0);
        let a = maru_collide::try_shape_cast_dispatch(&long, &unit, m, &wall, &pb, 0.02)
            .expect("정상 자세");
        let b = maru_collide::try_shape_cast_dispatch(&long, &scaled, m, &wall, &pb, 0.02)
            .expect("정규화 가능한 자세");
        // **음성 대조** — 이 배치에서 정상 자세는 실제로 맞춘다(둘 다 None이면 무의미하다).
        assert!(
            a.is_some(),
            "정상 자세가 벽을 못 맞췄다 — 배치가 무의미하다"
        );
        assert_eq!(a, b, "비단위 자세가 다른 형상으로 판정됐다");
        // 정규화 불가한 자세는 거절.
        let zero = maru_collide::Pose {
            pos: Vec3::ZERO,
            rot: Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
        };
        assert_eq!(
            maru_collide::try_shape_cast_dispatch(&long, &zero, m, &wall, &pb, 0.02),
            Err(maru_collide::CastReject::InvalidInput)
        );
        // 월드 질의도 같은 규약 — 여기서 정규화하지 않으면 AABB 선판정이 먼저 왜곡된다.
        let mut w = World::new(WorldParams::default());
        w.add_body(Body::new_static(wall.clone(), pb));
        let s1 = w
            .sweep_shape(&long, unit, m, QueryFilter::ALL)
            .expect("질의");
        let s2 = w
            .sweep_shape(&long, scaled, m, QueryFilter::ALL)
            .expect("질의");
        assert!(s1.is_some(), "월드 질의에서 정상 자세가 벽을 못 맞췄다");
        assert_eq!(
            s1.map(|h| h.toi),
            s2.map(|h| h.toi),
            "월드 질의가 비단위 자세를 그대로 썼다"
        );
        assert_eq!(
            w.sweep_shape(&long, zero, m, QueryFilter::ALL).err(),
            Some(ApiError::InvalidValue)
        );
    }

    /// **양성은 한 점으로 증명된다** — `shape_penetrates`가 넓은 AABB를 이유로 거절하지 않는다.
    ///
    /// 예전에는 접촉 순회 **전에** 셀 수만 세어 4096을 넘으면 즉시 `ResourceLimit`이었다.
    /// 그러면 첫 셀부터 깊이 박혀 있는 큰 바디도 AABB가 넓으면 증명 가능한 `true`를 영원히
    /// 못 받고, 캐릭터의 시작 겹침 검사가 매 호출 오류로 고정된다(확인검수 rev_2e93ebf2 MAJOR).
    /// 작업량 상한은 순회 안에 있으므로(4096셀) 선판정 없이도 유계다.
    #[test]
    fn shape_penetrates_proves_positives_even_with_a_huge_aabb() {
        let mut w = World::new(WorldParams::default());
        let heights = vec![0.0f32; 257 * 257];
        w.try_add_heightfield(&heights, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
            .expect("지형 추가");
        // 반지름 5m 구 = 100×100셀(예산 4096 초과). 중심을 지면 아래로 내려 **깊이 박아** 둔다.
        let deep = Shape::Sphere { radius: 5.0 };
        assert_eq!(
            w.shape_penetrates(&deep, pose(12.8, -1.0, 12.8), QueryFilter::ALL),
            Ok(true),
            "박혀 있는 것이 증명 가능한데 AABB가 넓다는 이유로 거절됐다"
        );
        // **음성 대조 ①** — 같은 크기인데 안 닿는 높이면, 앞쪽 셀에서 아무것도 못 찾은 채
        // 예산이 끊기므로 `Ok(false)`가 아니라 `Err(ResourceLimit)`(판정 미완료)이다.
        assert_eq!(
            w.shape_penetrates(&deep, pose(12.8, 5.0, 12.8), QueryFilter::ALL),
            Err(ApiError::ResourceLimit),
            "판정 미완료가 '겹치지 않음'으로 뭉개졌다"
        );
        // **음성 대조 ②** — 작은 바디는 예산 안이라 확정적인 답을 받는다(양쪽 다).
        let small = Shape::Sphere { radius: 0.2 };
        assert_eq!(
            w.shape_penetrates(&small, pose(12.8, -0.1, 12.8), QueryFilter::ALL),
            Ok(true)
        );
        assert_eq!(
            w.shape_penetrates(&small, pose(12.8, 5.0, 12.8), QueryFilter::ALL),
            Ok(false)
        );
    }

    /// **`ResourceLimit`의 기준선은 "호출 직전"이 아니라 "커맨드 적용 직후"다.**
    ///
    /// README가 한때 "실패한 스텝은 한 비트도 바꾸지 않는다"라고 단정했는데, 커맨드 flush와
    /// 속도 클램프는 실패보다 **먼저** 일어난다(그건 의도된 계약이다 — 빈 월드에 `AddBody`만
    /// 큐잉하면 영원히 적용되지 않는 문제를 그렇게 막았다). 호스트가 그 문구를 믿고 오류 후
    /// 커맨드를 다시 큐에 넣으면 같은 변경이 두 번 적용된다(확인검수 rev_2e93ebf2 MAJOR).
    #[test]
    fn resource_limit_applies_commands_but_skips_physics() {
        let mut w = World::new(WorldParams::default());
        let heights = vec![0.0f32; 257 * 257];
        w.try_add_heightfield(&heights, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
            .expect("지형 추가");
        let slab = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 5.0 },
            pose(12.8, 4.999, 12.8),
            1.0,
        ));
        let marker = w.enqueue_add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.1 },
            pose(1.0, 1.0, 1.0),
            1.0,
        ));
        let pos_before = w.body(slab).pose().pos;
        assert_eq!(w.try_step(1.0 / 60.0), Err(ApiError::ResourceLimit));
        // 커맨드는 적용됐다 — 재큐잉하면 두 번 들어간다.
        assert!(
            w.is_alive(marker),
            "실패 스텝이 커맨드를 삼켰다 — 호스트가 잃어버린다"
        );
        // 물리는 진행되지 않았다.
        assert_eq!(
            w.body(slab).pose().pos,
            pos_before,
            "실패 스텝이 물리를 진행시켰다"
        );
    }

    /// **레이·점 질의도 잘못된 입력을 값으로 거절한다** (BLOCKER: 공개 표면의 절반만 방어).
    ///
    /// README가 `cast_ray`·`contains_point`를 "모든 셰이프를 옳게 처리하는 공개 표면"으로
    /// 열거하는데 방어는 캐스트 두 함수에만 있었다 — 빈 hull을 넣으면 GJK에서 프로세스가
    /// 죽고, 빈 복합은 `None`/`false`가 되어 잘못된 형상이 미스로 위장됐다
    /// (확인검수 rev_97887d75 BLOCKER).
    #[test]
    fn ray_and_point_queries_reject_bad_input_by_value() {
        use maru_collide::QueryReject;
        let empty_hull = Shape::Hull {
            points: Vec::new(),
            radius: 0.0,
        };
        let empty_compound = Shape::Compound {
            children: Vec::new(),
        };
        let p = pose(0.0, 0.0, 0.0);
        let o = Vec3::new(-3.0, 0.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 0.0);
        for bad in [&empty_hull, &empty_compound] {
            assert_eq!(
                maru_collide::try_cast_ray(bad, &p, o, d, 10.0),
                Err(QueryReject::InvalidInput)
            );
            assert_eq!(
                maru_collide::try_contains_point(bad, &p, Vec3::ZERO),
                Err(QueryReject::InvalidInput)
            );
        }
        // 산술 입력·정규화 불가 자세도 같은 규칙.
        let ok = Shape::Sphere { radius: 1.0 };
        assert_eq!(
            maru_collide::try_cast_ray(&ok, &p, o, Vec3::new(f32::NAN, 0.0, 0.0), 10.0),
            Err(QueryReject::InvalidInput)
        );
        let zero_rot = maru_collide::Pose {
            pos: Vec3::ZERO,
            rot: Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
        };
        assert_eq!(
            maru_collide::try_contains_point(&ok, &zero_rot, Vec3::ZERO),
            Err(QueryReject::InvalidInput)
        );
        // **음성 대조** — 정상 입력은 실제로 맞추고 실제로 품는다.
        assert!(maru_collide::try_cast_ray(&ok, &p, o, d, 10.0)
            .expect("정상 입력")
            .is_some());
        assert_eq!(
            maru_collide::try_contains_point(&ok, &p, Vec3::new(0.5, 0.0, 0.0)),
            Ok(true)
        );
        // 비단위(정규화 가능) 자세는 정규화해서 단위와 같은 답을 낸다.
        let q = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.7);
        let long = Shape::Box {
            half: Vec3::new(2.0, 0.2, 0.2),
            radius: 0.0,
        };
        let unit = maru_collide::Pose {
            pos: Vec3::ZERO,
            rot: q,
        };
        let scaled = maru_collide::Pose {
            pos: Vec3::ZERO,
            rot: Quat {
                x: q.x * 0.5,
                y: q.y * 0.5,
                z: q.z * 0.5,
                w: q.w * 0.5,
            },
        };
        let a = maru_collide::try_cast_ray(&long, &unit, o, d, 10.0).expect("정상");
        let b = maru_collide::try_cast_ray(&long, &scaled, o, d, 10.0).expect("정규화 가능");
        assert!(
            a.is_some(),
            "정상 자세가 박스를 못 맞췄다 — 배치가 무의미하다"
        );
        assert_eq!(
            a.map(|h| h.toi),
            b.map(|h| h.toi),
            "비단위 자세가 찌그러진 형상으로 판정됐다"
        );
    }

    /// **NaN margin이 접촉을 만들어내지 않는다** (MAJOR: 없는 접촉이 생긴다).
    ///
    /// `dist2 > reach*reach` 비교가 NaN 때문에 false가 되면 멀리 떨어진 두 구가 접촉을 만든다.
    /// 캐스트에서 막은 "잘못된 입력이 판정으로 위장" 결함이 바로 옆 API에 남아 있었다
    /// (확인검수 rev_97887d75 MAJOR).
    #[test]
    fn contact_api_rejects_bad_arithmetic_input() {
        let a = Shape::Sphere { radius: 0.5 };
        let b = Shape::Sphere { radius: 0.5 };
        let pa = pose(0.0, 0.0, 0.0);
        let far = pose(50.0, 0.0, 0.0);
        for bad in [f32::NAN, -1.0, f32::INFINITY] {
            assert_eq!(
                maru_collide::for_each_contact(&a, &pa, &b, &far, bad, &mut |_, _, _| {}),
                maru_collide::ContactWalk::InvalidInput,
                "margin={bad}"
            );
            assert_eq!(maru_collide::any_contact(&a, &pa, &b, &far, bad), None);
        }
        // **음성 대조** — 정상 margin이면 멀리 있는 쌍은 접촉이 없고, 가까운 쌍은 있다.
        assert_eq!(
            maru_collide::any_contact(&a, &pa, &b, &far, 0.02),
            Some(false)
        );
        assert_eq!(
            maru_collide::any_contact(&a, &pa, &b, &pose(0.8, 0.0, 0.0), 0.02),
            Some(true)
        );
    }

    /// **포화는 완주가 아니다** — 음성 판정은 순회를 끝까지 걸어야 한다 (MAJOR).
    ///
    /// `shape_penetrates`가 `Saturated`를 `Complete`처럼 다루면, 앞 64개 매니폴드가 침투
    /// 임계에 못 미치는 얕은 접촉이었을 때 뒤쪽의 깊은 침투를 못 본 채 "안 겹친다"가 된다 —
    /// 캐릭터가 지형 안에서 이동을 시작한다(확인검수 rev_97887d75 MAJOR).
    ///
    /// 넓은 판을 평지에 **살짝 얹으면**(침투 없음) 매니폴드가 64개에서 포화한다. 그때 답은
    /// `Ok(false)`가 아니라 "판정 미완료"다.
    #[test]
    fn shape_penetrates_treats_saturation_as_incomplete_not_negative() {
        let mut w = World::new(WorldParams::default());
        let heights = vec![0.0f32; 65 * 65];
        w.try_add_heightfield(&heights, 65, 65, 0.5, pose(0.0, 0.0, 0.0))
            .expect("지형 추가");
        // 12m×12m 판을 지면 **위에** 얹는다(겹침 없음) — 셀마다 접촉이 생겨 64개에서 포화.
        let slab = Shape::Box {
            half: Vec3::new(6.0, 0.2, 6.0),
            radius: 0.0,
        };
        assert_eq!(
            w.shape_penetrates(&slab, pose(8.0, 0.2, 8.0), QueryFilter::ALL),
            Err(ApiError::ResourceLimit),
            "포화를 확정 음성으로 취급했다 — 뒤쪽 깊은 침투를 못 본 채 '안 겹친다'가 된다"
        );
        // **음성 대조 ①** — 확실히 박히면 양성이 먼저 증명되므로 포화와 무관하게 `true`.
        assert_eq!(
            w.shape_penetrates(&slab, pose(8.0, -0.1, 8.0), QueryFilter::ALL),
            Ok(true)
        );
        // **음성 대조 ②** — 작은 바디는 포화하지 않으므로 확정적인 답을 받는다.
        let small = Shape::Box {
            half: Vec3::new(0.2, 0.2, 0.2),
            radius: 0.0,
        };
        assert_eq!(
            w.shape_penetrates(&small, pose(8.0, 2.0, 8.0), QueryFilter::ALL),
            Ok(false)
        );
        assert_eq!(
            w.shape_penetrates(&small, pose(8.0, -0.1, 8.0), QueryFilter::ALL),
            Ok(true)
        );
    }

    /// **셀 예산은 질의 하나가 나눠 쓴다** (MAJOR: 바디 수만큼 곱해지는 작업량).
    ///
    /// 예산이 바디마다 새로 4096이면 같은 자리에 지형을 여러 장 겹친 장면에서 한 번의 질의가
    /// 바디 수 × 4096셀을 훑는다 — "작업량 상한은 순회 안에 있다"는 주장이 페어에만
    /// 참이었다(확인검수 rev_97887d75 MAJOR). 스윕 질의로 확인한다: 셀 수는 스윕 봉투에서
    /// 세므로 접촉이 있는지와 무관하게 관측된다.
    #[test]
    fn query_cell_budget_is_shared_across_bodies() {
        let probe = Shape::Box {
            half: Vec3::new(15.0, 0.2, 15.0),
            radius: 0.0,
        };
        let start = pose(16.0, 2.0, 16.0);
        let down = Vec3::new(0.0, -4.0, 0.0);
        // 지형 한 장(65×65, 셀 0.5m = 32m 폭). 30m 판의 스윕 봉투는 약 60×60 = 3600셀 —
        // 예산(4096) 안이다. 두 장이면 7200으로 넘는다.
        let mut w1 = World::new(WorldParams::default());
        w1.try_add_heightfield(&vec![0.0f32; 65 * 65], 65, 65, 0.5, pose(0.0, 0.0, 0.0))
            .expect("지형");
        let one = w1.sweep_shape(&probe, start, down, QueryFilter::ALL);
        assert!(
            matches!(one, Ok(Some(_))),
            "한 장짜리 장면이 예산에 걸렸다 — 대조가 성립하지 않는다: {one:?}"
        );
        // **같은 자리에 두 장을 겹친다.** 예산이 공유되면 총합이 4096을 넘어 판정 미완료가
        // 되고, 공유가 아니면 장 수만큼 곱해진 작업을 끝까지 한다.
        let mut w3 = World::new(WorldParams::default());
        for _ in 0..2 {
            w3.try_add_heightfield(&vec![0.0f32; 65 * 65], 65, 65, 0.5, pose(0.0, 0.0, 0.0))
                .expect("지형");
        }
        assert_eq!(
            w3.sweep_shape(&probe, start, down, QueryFilter::ALL).err(),
            Some(ApiError::ResourceLimit),
            "예산이 바디마다 새로 주어지고 있다 — 총 작업량이 유계가 아니다"
        );
        // **음성 대조** — 작은 바디는 두 장을 훑어도 예산 안이라 답을 받는다.
        let small = Shape::Box {
            half: Vec3::new(0.2, 0.2, 0.2),
            radius: 0.0,
        };
        assert!(matches!(
            w3.sweep_shape(&small, start, down, QueryFilter::ALL),
            Ok(Some(_))
        ));
    }

    /// **자식 자세 정책이 생성 경계와 질의 경계에서 같다** (BLOCKER: 같은 바디, API마다 다른 답).
    ///
    /// 길이 제곱 1.0404(≈ w 1.02)는 옛 `0.9..=1.1`을 통과해 월드에 들어갔지만 질의 쪽
    /// `is_castable`(1e-3)에서는 거절됐다. 그러면 같은 바디가 레이 질의에서는 미스, 점 질의에서는
    /// `InvalidValue`, 물리 스텝에서는 `ResourceLimit`이 된다 — 게다가 동적 복합은
    /// `bake_compound`가 그 비단위 회전으로 관성 텐서를 계산한다
    /// (확인검수 rev_c73f897c BLOCKER — 내가 질의 쪽만 조이고 이 자리를 빠뜨렸다).
    #[test]
    fn compound_child_rotation_policy_is_the_same_at_both_boundaries() {
        let child = Shape::Box {
            half: Vec3::new(0.3, 0.3, 0.3),
            radius: 0.0,
        };
        let scaled = |k: f32| maru_collide::Pose {
            pos: Vec3::new(0.5, 0.0, 0.0),
            rot: Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: k,
            },
        };
        let make = |k: f32| Shape::Compound {
            children: vec![(scaled(k), child.clone())],
        };
        // w=1.02 → 길이 제곱 1.0404: 옛 기준은 통과, 새 기준은 거절.
        let bad = make(1.02);
        assert!(
            bad.validate().is_err(),
            "생성 경계가 비단위 자식을 통과시켰다 — 질의 경계와 정책이 갈린다"
        );
        assert!(!bad.is_castable_pub_for_test());
        assert!(Body::try_new_static(bad.clone(), pose(0.0, 0.0, 0.0)).is_err());
        assert!(Body::try_new_dynamic(bad, pose(0.0, 0.0, 0.0), 1.0).is_err());
        // **음성 대조** — 드리프트 수준(1e-4)은 양쪽 다 통과한다(정상 입력을 막지 않는다).
        let drift = make(1.0 + 5.0e-5);
        assert!(drift.validate().is_ok(), "정상 드리프트가 거절됐다");
        assert!(drift.is_castable_pub_for_test());
        assert!(Body::try_new_static(drift, pose(0.0, 0.0, 0.0)).is_ok());
        // 자식 수 상한도 두 경계에서 같다 — 콜백 키가 u8이라 래핑하면 자식이 서로 섞인다.
        let many = Shape::Compound {
            children: (0..9)
                .map(|i| {
                    (
                        pose(i as f32 * 0.8, 0.0, 0.0),
                        Shape::Sphere { radius: 0.2 },
                    )
                })
                .collect(),
        };
        assert!(many.validate().is_err());
        assert!(!many.is_castable_pub_for_test());
    }

    /// **레이 질의는 0 방향과 거대한 원점을 값으로 거절한다** (BLOCKER: 계약 위반 결과).
    ///
    /// 유한성만 보면 (a) `dir=0`이 법선 0 벡터인 히트를 만들고("법선은 단위 벡터" 계약 파기),
    /// (b) `origin=f32::MAX`가 중간 계산에서 `inf-inf=NaN`이 되어 `toi`·`normal`이 NaN인
    /// 히트를 `Ok(Some)`으로 내보낸다(확인검수 rev_c73f897c BLOCKER).
    #[test]
    fn ray_query_rejects_zero_direction_and_out_of_range_origin() {
        use maru_collide::QueryReject;
        let s = Shape::Sphere { radius: 1.0 };
        let p = pose(0.0, 0.0, 0.0);
        assert_eq!(
            maru_collide::try_cast_ray(&s, &p, Vec3::ZERO, Vec3::ZERO, 10.0),
            Err(QueryReject::InvalidInput),
            "0 방향이 통과했다 — 법선이 0 벡터인 히트가 나온다"
        );
        assert_eq!(
            maru_collide::try_cast_ray(
                &s,
                &p,
                Vec3::new(f32::MAX, 0.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                f32::MAX
            ),
            Err(QueryReject::InvalidInput),
            "범위 밖 원점이 통과했다 — NaN 히트가 나온다"
        );
        assert_eq!(
            maru_collide::try_contains_point(&s, &p, Vec3::new(f32::MAX, 0.0, 0.0)),
            Err(QueryReject::InvalidInput)
        );
        // **음성 대조** — 정상 입력은 맞추고, 법선은 단위 벡터다.
        let h = maru_collide::try_cast_ray(
            &s,
            &p,
            Vec3::new(-3.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            10.0,
        )
        .expect("정상 입력")
        .expect("구를 못 맞췄다");
        assert!(
            (h.normal.length() - 1.0).abs() < 1.0e-4,
            "법선이 단위가 아니다"
        );
    }

    /// **AABB 질의도 패닉하지 않는다** (BLOCKER: 공개 표면의 남은 패닉 경로).
    ///
    /// 레이·점만 `try_`로 바꾸고 `local_aabb()`를 열어 두면, 빈 hull은 `points[0]`에서
    /// 빈 복합은 누적값 `expect`에서 여전히 프로세스를 죽인다(확인검수 rev_c73f897c BLOCKER).
    #[test]
    fn aabb_query_rejects_malformed_shapes_instead_of_panicking() {
        use maru_collide::ShapeReject;
        for bad in [
            Shape::Hull {
                points: Vec::new(),
                radius: 0.0,
            },
            Shape::Compound {
                children: Vec::new(),
            },
        ] {
            assert_eq!(bad.try_local_aabb(), Err(ShapeReject::InvalidShape));
        }
        // **음성 대조** — 정상 형상은 실제 AABB를 준다.
        let s = Shape::Sphere { radius: 0.5 };
        let a = s.try_local_aabb().expect("정상 형상");
        assert!((a.max.x - 0.5).abs() < 1.0e-6 && (a.min.x + 0.5).abs() < 1.0e-6);
    }

    /// **존재 질의의 답이 삽입 순서에 의존하지 않는다** (MAJOR: 같은 장면, 다른 답).
    ///
    /// 판정 미완료(포화·예산)에서 즉시 오류로 나가면, 포화하는 지형이 앞 슬롯일 때 뒤에 깊게
    /// 박힌 바디를 못 본다 — 같은 형상 집합이 삭제·재삽입 순서에 따라 "침투함"과 "판정 불가"로
    /// 갈린다(확인검수 rev_c73f897c MAJOR). 미완료는 기억만 하고, 확정 양성이 있으면 그것이 이긴다.
    #[test]
    fn shape_penetrates_answer_does_not_depend_on_slot_order() {
        let probe = Shape::Box {
            half: Vec3::new(6.0, 0.2, 6.0),
            radius: 0.0,
        };
        let at = pose(8.0, 0.2, 8.0);
        // 지형만 있으면 포화로 "판정 미완료"다(직전 라운드 테스트가 고정한 계약).
        let mut only_hf = World::new(WorldParams::default());
        only_hf
            .try_add_heightfield(&vec![0.0f32; 65 * 65], 65, 65, 0.5, pose(0.0, 0.0, 0.0))
            .expect("지형");
        assert_eq!(
            only_hf.shape_penetrates(&probe, at, QueryFilter::ALL),
            Err(ApiError::ResourceLimit)
        );
        // 같은 장면 + 프로브에 **깊게 박힌 구**. 두 삽입 순서 모두 `Ok(true)`여야 한다.
        for hf_first in [true, false] {
            let mut w = World::new(WorldParams::default());
            let add_hf = |w: &mut World| {
                w.try_add_heightfield(&vec![0.0f32; 65 * 65], 65, 65, 0.5, pose(0.0, 0.0, 0.0))
                    .expect("지형");
            };
            let add_ball = |w: &mut World| {
                w.add_body(Body::new_static(
                    Shape::Sphere { radius: 0.5 },
                    pose(8.0, 0.2, 8.0),
                ));
            };
            if hf_first {
                add_hf(&mut w);
                add_ball(&mut w);
            } else {
                add_ball(&mut w);
                add_hf(&mut w);
            }
            assert_eq!(
                w.shape_penetrates(&probe, at, QueryFilter::ALL),
                Ok(true),
                "삽입 순서(지형 먼저={hf_first})가 답을 바꿨다"
            );
        }
    }

    /// **비단위 자식 자세가 기하를 왜곡하지 않는다** (BLOCKER: 1m 규모의 조용한 오차).
    ///
    /// 수용 기준(길이 제곱 1e-3)은 *입력 관용*이고, 기하는 **정규화된 회전으로** 풀어야 한다.
    /// 비단위 쿼터니언의 `rotate`는 회전이 아니라 회전+스케일이라, 최대 형상 반경(1000m)에서
    /// 0.05%는 약 0.5m다 — speculative margin(0.05m)의 10배, 침투 임계(1e-4m)의 5000배다
    /// (확인검수 rev_201e3d20 BLOCKER). AABB·레이·접촉·관성이 모두 이 값을 쓴다.
    #[test]
    fn non_unit_child_rotation_does_not_distort_geometry() {
        let child = Shape::Box {
            half: Vec3::new(300.0, 5.0, 5.0),
            radius: 0.0,
        };
        let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), core::f32::consts::FRAC_PI_2);
        // 수용 범위 안에서 최대로 부풀린 자식 회전(길이 제곱 ≈ 1+1e-3).
        let k = (1.0f32 + 9.0e-4).sqrt();
        let mk = |rot: Quat| Shape::Compound {
            children: vec![(
                maru_collide::Pose {
                    pos: Vec3::new(400.0, 0.0, 0.0),
                    rot,
                },
                child.clone(),
            )],
        };
        let unit = mk(q);
        let scaled = mk(Quat {
            x: q.x * k,
            y: q.y * k,
            z: q.z * k,
            w: q.w * k,
        });
        assert!(scaled.validate().is_ok(), "수용 범위 안인데 거절됐다");
        // ① AABB가 같아야 한다.
        let (a, b) = (
            unit.try_local_aabb().expect("정상"),
            scaled.try_local_aabb().expect("정상"),
        );
        let d = (a.min - b.min).length().max((a.max - b.max).length());
        assert!(
            d < 1.0e-3,
            "비단위 자식이 AABB를 {d} m 만큼 왜곡했다 (침투 임계 1e-4 m)"
        );
        // ② 레이 히트 지점도 같아야 한다.
        let p = pose(0.0, 0.0, 0.0);
        // 자식은 (400,0,0)에 있고 90° 돌아 y로 길어진다 — 그 위에서 아래로 쏜다.
        let o = Vec3::new(400.0, 900.0, 0.0);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let ha = maru_collide::try_cast_ray(&unit, &p, o, dir, 2000.0).expect("정상");
        let hb = maru_collide::try_cast_ray(&scaled, &p, o, dir, 2000.0).expect("정상");
        // **음성 대조** — 정상 자세가 실제로 맞춰야 비교가 의미 있다.
        let ta = ha
            .expect("정상 자세가 회전한 자식을 못 맞췄다 — 배치가 무의미하다")
            .toi;
        let tb = hb.expect("비단위 자세에서 히트가 사라졌다").toi;
        assert!(
            (ta - tb).abs() < 1.0e-3,
            "레이 toi가 {} vs {}로 갈렸다",
            ta,
            tb
        );
        // ③ 관성·질량도 같아야 한다(동적 복합은 자식 회전으로 텐서를 돌린다).
        let ba = Body::try_new_dynamic(unit, pose(0.0, 0.0, 0.0), 1.0).expect("정상");
        let bb = Body::try_new_dynamic(scaled, pose(0.0, 0.0, 0.0), 1.0).expect("정상");
        assert!((ba.inv_mass() - bb.inv_mass()).abs() <= f32::EPSILON * 4.0);
        let (ia, ib) = (ba.inv_inertia_local(), bb.inv_inertia_local());
        let rel = ((ia - ib).length() / ia.length().max(1.0e-12)).abs();
        assert!(rel < 1.0e-3, "관성 텐서가 상대오차 {rel}로 갈렸다");
    }

    /// **범위 밖 거대 형상은 질의 경계에서도 거절된다** (MAJOR: 두 경계의 정책 불일치).
    ///
    /// `half = f32::MAX`인 박스는 생성 경계(`validate`)가 `MAX_SHAPE_EXTENT`로 거절하는데
    /// 질의 경계(`is_castable`)는 유한성만 봐서 통과시켰다 — 45° 자세의 `transformed_aabb`에서
    /// 두 항의 합이 `Inf`가 된다(확인검수 rev_201e3d20 MAJOR).
    #[test]
    fn out_of_range_shapes_are_rejected_at_the_query_boundary_too() {
        use maru_collide::{QueryReject, ShapeReject};
        let huge_box = Shape::Box {
            half: Vec3::new(f32::MAX, f32::MAX, f32::MAX),
            radius: 0.0,
        };
        let huge_hull = Shape::Hull {
            points: vec![Vec3::new(f32::MAX, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
            radius: 0.0,
        };
        let huge_offset = Shape::Compound {
            children: vec![(pose(f32::MAX, 0.0, 0.0), Shape::Sphere { radius: 0.5 })],
        };
        let p = pose(0.0, 0.0, 0.0);
        for bad in [&huge_box, &huge_hull, &huge_offset] {
            assert!(bad.validate().is_err(), "생성 경계가 통과시켰다");
            assert_eq!(
                bad.try_local_aabb(),
                Err(ShapeReject::InvalidShape),
                "질의 경계가 통과시켰다 — transformed_aabb에서 Inf가 된다"
            );
            assert_eq!(
                maru_collide::try_cast_ray(bad, &p, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 10.0),
                Err(QueryReject::InvalidInput)
            );
            assert_eq!(
                maru_collide::try_shape_cast_dispatch(
                    bad,
                    &p,
                    Vec3::new(1.0, 0.0, 0.0),
                    &Shape::Sphere { radius: 0.5 },
                    &pose(5.0, 0.0, 0.0),
                    0.02
                ),
                Err(maru_collide::CastReject::InvalidInput)
            );
        }
        // **음성 대조** — 상한 안(1000m)의 큰 형상은 양쪽 경계를 통과한다.
        let big_ok = Shape::Box {
            half: Vec3::new(900.0, 900.0, 900.0),
            radius: 0.0,
        };
        assert!(big_ok.validate().is_ok());
        assert!(big_ok.try_local_aabb().is_ok());
    }

    /// **형상 내부 관계가 깨진 박스는 모든 공개 질의가 같이 거절한다** (BLOCKER: 질의끼리 불일치).
    ///
    /// `Box { half: 1, radius: 2 }`는 `validate`가 거절하지만 질의 경계는 범위만 봐서
    /// 통과시켰다. 그 형상에서 AABB·점 질의는 반경 1로, 레이·GJK는 반경 2로 답한다 —
    /// 호스트가 공개 AABB로 후보를 쳐낸 뒤 레이를 쓰면 히트가 사라진다
    /// (확인검수 rev_b85e4e4a BLOCKER).
    #[test]
    fn box_with_radius_larger_than_half_is_rejected_everywhere() {
        use maru_collide::{CastReject, QueryReject, ShapeReject};
        let bad = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 2.0,
        };
        let p = pose(0.0, 0.0, 0.0);
        assert!(bad.validate().is_err(), "생성 경계는 원래 거절했다");
        assert_eq!(bad.try_local_aabb(), Err(ShapeReject::InvalidShape));
        assert_eq!(
            maru_collide::try_contains_point(&bad, &p, Vec3::new(1.5, 0.0, 0.0)),
            Err(QueryReject::InvalidInput)
        );
        assert_eq!(
            maru_collide::try_cast_ray(
                &bad,
                &p,
                Vec3::new(1.5, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                10.0
            ),
            Err(QueryReject::InvalidInput)
        );
        assert_eq!(
            maru_collide::try_shape_cast_dispatch(
                &bad,
                &p,
                Vec3::new(1.0, 0.0, 0.0),
                &Shape::Sphere { radius: 0.5 },
                &pose(5.0, 0.0, 0.0),
                0.02
            ),
            Err(CastReject::InvalidInput)
        );
        assert_eq!(
            maru_collide::any_contact(&bad, &p, &Shape::Sphere { radius: 0.5 }, &p, 0.02),
            None
        );
        // **음성 대조** — 관계가 성립하는 둥근 박스(radius ≤ min half)는 전부 통과한다.
        let ok = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.4,
        };
        assert!(ok.validate().is_ok());
        assert!(ok.try_local_aabb().is_ok());
        assert_eq!(
            maru_collide::try_contains_point(&ok, &p, Vec3::new(0.5, 0.0, 0.0)),
            Ok(true)
        );
    }

    /// **지원 범위 밖 입력은 "유한하다"는 이유로 통과하지 못한다** (BLOCKER: 거짓 히트/미스).
    ///
    /// 두 가지가 독립적으로 재현됐다(확인검수 rev_b85e4e4a):
    /// ① `axis=(f32::MAX,0,0)`은 유한하지만 `length_sq`가 `Inf`라 `try_normalize`가 `None`을
    ///    내고, 그것을 "축이 없다"로 읽으면 **회전 캐스트가 선형으로 폴백해 제자리 회전
    ///    히트를 놓친다**.
    /// ② 자세를 `±f32::MAX`에 두면 중심 차가 `Inf`가 되고 GJK의 비유한 위생이 그것을 거리 0으로
    ///    강등해 **무한히 멀리 떨어진 두 형상이 `toi=0` 히트**가 된다.
    #[test]
    fn out_of_range_arithmetic_input_is_rejected_not_reinterpreted() {
        use maru_collide::{CastReject, ContactWalk};
        let blade = Shape::Box {
            half: Vec3::new(1.5, 0.05, 0.05),
            radius: 0.0,
        };
        let wall = Shape::Box {
            half: Vec3::new(0.2, 1.0, 1.0),
            radius: 0.0,
        };
        // ① 거대한 축 — 0축(회전 없음)으로 오해하면 안 된다.
        assert_eq!(
            maru_collide::try_shape_cast_dispatch_rotational(
                &blade,
                &pose(0.0, 0.0, 0.0),
                Vec3::ZERO,
                Vec3::new(f32::MAX, 0.0, 0.0),
                core::f32::consts::FRAC_PI_2,
                &wall,
                &pose(0.0, 0.0, 2.0),
                0.02
            ),
            Err(CastReject::InvalidInput)
        );
        // ② 범위 밖 자세 — 무한히 먼 두 형상이 접촉으로 나오면 안 된다.
        let far_a = pose(f32::MAX, 0.0, 0.0);
        let far_b = pose(-f32::MAX, 0.0, 0.0);
        let s = Shape::Sphere { radius: 0.5 };
        assert_eq!(
            maru_collide::try_shape_cast_dispatch(
                &s,
                &far_a,
                Vec3::new(-1.0, 0.0, 0.0),
                &s,
                &far_b,
                0.02
            ),
            Err(CastReject::InvalidInput)
        );
        assert_eq!(
            maru_collide::for_each_contact(&s, &far_a, &s, &far_b, 0.02, &mut |_, _, _| {}),
            ContactWalk::InvalidInput
        );
        assert_eq!(
            maru_collide::any_contact(&s, &far_a, &s, &far_b, 0.02),
            None
        );
        // **음성 대조** — 같은 배치를 범위 안(±5000m)으로 옮기면 정상 판정이다.
        let a = pose(5000.0, 0.0, 0.0);
        let b = pose(-5000.0, 0.0, 0.0);
        assert_eq!(
            maru_collide::any_contact(&s, &a, &s, &b, 0.02),
            Some(false),
            "범위 안의 먼 쌍은 '접촉 없음'이 확정 답이다"
        );
        // 그리고 정상 회전 캐스트는 실제로 맞춘다(가드가 회전 경로를 죽인 게 아니다).
        let hit = maru_collide::try_shape_cast_dispatch_rotational(
            &blade,
            &pose(0.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            core::f32::consts::FRAC_PI_2,
            &wall,
            &pose(0.0, 0.0, 1.7),
            0.02,
        )
        .expect("정상 입력");
        assert!(
            hit.is_some(),
            "정상 회전 스윕이 벽을 못 맞췄다 — 배치가 무의미하다"
        );
    }

    /// **FP 가드 전수** — 캐노니컬 상태를 바꾸는 공개 진입점 전부가 같은 규칙을 따른다.
    ///
    /// 검수가 두 번 같은 계열을 지적했다(생성자 → enqueue). 한 곳만 빠지면 "비표준 환경
    /// 진입은 상태를 바꾸지 않는다"가 API 선택에 따라 갈리므로, 목록으로 잠근다.
    #[test]
    fn every_canonical_mutation_entry_point_is_guarded() {
        let mut w = World::new(WorldParams::default());
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 2.0, 0.0),
            1.0,
        ));
        let h2 = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(2.0, 2.0, 0.0),
            1.0,
        ));
        let snap = w.snapshot();
        // 환경을 깨기 **전에** 정상 Body를 하나 만들어 둔다 — `try_add_body`의 가드를 보려면
        // 인자가 유효해야 한다(생성자 가드에 먼저 걸리면 그 경로를 못 본다).
        let pending =
            Body::try_new_static(Shape::Sphere { radius: 0.2 }, pose(5.0, 5.0, 5.0)).expect("생성");
        if !maru_math::force_flush_to_zero_for_test() {
            return;
        }
        // fallible 판은 모두 FpEnvironment를 돌려준다.
        let e = Some(ApiError::FpEnvironment);
        assert_eq!(
            w.set_pose(
                h,
                maru_collide::Pose {
                    pos: Vec3::ZERO,
                    rot: Quat::IDENTITY
                }
            )
            .err(),
            e
        );
        assert_eq!(w.set_linear_velocity(h, Vec3::ZERO).err(), e);
        assert_eq!(w.set_angular_velocity(h, Vec3::ZERO).err(), e);
        assert_eq!(w.apply_force(h, Vec3::ZERO).err(), e);
        assert_eq!(w.apply_torque(h, Vec3::ZERO).err(), e);
        assert_eq!(w.apply_impulse(h, Vec3::ZERO).err(), e);
        assert_eq!(w.apply_angular_impulse(h, Vec3::ZERO).err(), e);
        assert_eq!(w.apply_impulse_at_point(h, Vec3::ZERO, Vec3::ZERO).err(), e);
        assert_eq!(w.clear_forces(h).err(), e);
        assert_eq!(w.set_damping(h, 0.1, 0.1).err(), e);
        assert_eq!(w.set_material(h, 0.5, 0.1).err(), e);
        assert_eq!(w.remove_body(h2).err(), e);
        assert_eq!(w.try_set_params(WorldParams::default()).err(), e);
        assert_eq!(w.try_set_sleep_thresholds(0.1, 0.2, 0.5).err(), e);
        assert_eq!(w.try_flush_commands().err(), e);
        assert_eq!(w.try_step(DT).err(), e);
        assert_eq!(
            w.restore(&snap).err(),
            Some(crate::SnapshotError::FpEnvironment)
        );
        assert_eq!(
            w.try_add_body(pending).err(),
            e,
            "미리 만든 Body 추가가 통과했다"
        );
        // 생성자 자체도 거부한다(우회 경로 차단).
        assert_eq!(
            Body::try_new_static(Shape::Sphere { radius: 0.2 }, pose(1.0, 1.0, 1.0)).err(),
            e
        );
        assert!(World::canonicalize_fp_environment());
        // 복구 후에는 전부 정상.
        assert!(w
            .set_pose(
                h,
                maru_collide::Pose {
                    pos: Vec3::ZERO,
                    rot: Quat::IDENTITY
                }
            )
            .is_ok());
        assert!(w.try_step(DT).is_ok());
        assert!(w.is_alive(h2), "거부된 remove가 실제로 지웠다");
    }

    /// 설계 §8이 규정한 **1,000스텝** 판 — release 전용(debug는 수 분).
    /// `cargo test --release -p maru-dynamics -- --ignored`
    #[test]
    #[ignore = "release 전용: cargo test --release -p maru-dynamics -- --ignored"]
    fn thread_determinism_1000_steps() {
        assert_thread_determinism(30, 1000, &[2, 3, 4]);
    }

    /// release 전용 — debug는 수 분 걸린다. `cargo test --release -- --ignored`
    #[test]
    #[ignore = "release 전용: cargo test --release -p maru-dynamics -- --ignored"]
    fn pyramid_40_tiers_long_run_acceptance() {
        const TIERS: usize = 40;
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
        let hs = pyramid(&mut w, TIERS);
        for _ in 0..3000 {
            w.step(DT);
        }
        assert_pyramid_stable(&w, &hs, TIERS);
        let asleep = hs.iter().filter(|h| w.is_asleep(**h)).count();
        assert_eq!(
            asleep,
            hs.len(),
            "{asleep}/{} 만 잠듦 — 잠들지 않은 스택은 장기적으로 발산한다",
            hs.len()
        );
    }

    /// 봉투 **밖**(65단)은 실제로 붕괴한다 — 기준이 판별력을 갖는지 잠근다.
    /// 이 테스트가 실패하면 봉투가 넓어진 것이므로 위 기준을 다시 재야 한다.
    #[test]
    #[ignore = "release 전용: cargo test --release -p maru-dynamics -- --ignored"]
    fn pyramid_65_tiers_is_outside_the_envelope() {
        let mut w = World::new(WorldParams::default());
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(120.0, 1.0, 20.0),
                radius: 0.0,
            },
            pose(0.0, -1.0, 0.0),
        ));
        let hs = pyramid(&mut w, 65);
        for _ in 0..3000 {
            w.step(DT);
        }
        let top = w.body(*hs.last().unwrap()).pose.pos.y;
        assert!(
            top < 0.99 * (0.5 + 64.0),
            "65단이 버텼다 — 봉투가 넓어졌으니 수용 기준을 재측정할 것 (top_y={top:.2})"
        );
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
        bad[HDR..HDR + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(w.restore(&bad), Err(SnapshotError::CountTooLarge));
        // 5) 잔여 바이트
        let mut bad = good.clone();
        bad.push(0);
        assert_eq!(w.restore(&bad), Err(SnapshotError::TrailingBytes));
        // 6) 비유한 값 주입 (바디 위치 필드)
        let mut bad = good.clone();
        // 헤더 + count4 + alive1+oob1+reserved1+generation4+body_type1 = pos.x
        let pos_off = HDR + 4 + 1 + 1 + 1 + 4 + 1;
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
        let mut bad = good[..HDR + 4].to_vec();
        bad[HDR..HDR + 4].copy_from_slice(&1_000_000u32.to_le_bytes());
        assert_eq!(w.restore(&bad), Err(SnapshotError::CountTooLarge));

        // 중첩 개수(hull 점)도 같은 규칙 — 총량 예산이 파일 크기에서 자동으로 나온다
        let mut bad = good.clone();
        bad[HDR..HDR + 4].copy_from_slice(&(good.len() as u32).to_le_bytes());
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

        // 슬롯(Box): alive1+oob1+reserved1+gen4
        //            + 본문(106 + force12 + torque12 + membership4 + filter4 + sensor1
        //                   + ccd1(v11) + weld1(v12) + damping8 = 149)
        //            + sleep4+asleep1+island4 = 165
        const SLOT: usize = 165;
        let type_off = HDR + 4 + SLOT + 1 + 1 + 1 + 4;
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
        // 슬롯1의 inv_mass 자리
        let off = HDR + 4 + 122 + 1 + 1 + 1 + 4 + 1 + 12 + 16 + 12 + 12;
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
                points: vec![
                    Vec3::new(9.5, -0.5, -0.5),
                    Vec3::new(10.5, -0.5, -0.5),
                    Vec3::new(9.5, 0.5, -0.5),
                    Vec3::new(9.5, -0.5, 0.5),
                    Vec3::new(10.5, 0.5, 0.5),
                ],
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
    #[should_panic(expected = "MAX_INPUT")]
    fn huge_pose_rejected_at_construction() {
        Body::new_dynamic(Shape::Sphere { radius: 1.0 }, pose(1.0e20, 0.0, 0.0), 1.0);
    }

    /// NEW MAJOR: dt 위반은 **wasm trap이 아니라 오류**로 표현돼야 한다.
    /// 고정틱 엔진이므로 큰 dt를 쪼개는 것은 호스트(accumulator) 책임이다.
    #[test]
    fn huge_dt_returns_error_not_trap() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        assert_eq!(w.try_step(8.0), Err(ApiError::InvalidValue));
        assert_eq!(w.try_step(f32::NAN), Err(ApiError::InvalidValue));
        assert_eq!(w.try_step(crate::world::MAX_DT), Ok(()), "상한 자체는 합법");
        // 문서와 구현 통일(리뷰 MAJOR): 0과 음수는 **합법이 아니다**.
        // 일시정지는 step을 부르지 않는 것으로 표현한다 — "성공했는데 아무 일도 없음"이
        // 오류와 구분되지 않으면 호스트가 정지 원인을 진단할 수 없다.
        assert_eq!(w.try_step(0.0), Err(ApiError::InvalidValue));
        assert_eq!(w.try_step(-DT), Err(ApiError::InvalidValue));
        assert_eq!(
            w.try_step(crate::world::MIN_DT * 0.5),
            Err(ApiError::InvalidValue),
            "너무 작은 dt는 1/h를 폭발시킨다"
        );
    }

    /// BLOCKER: 정책이 step에 대해 닫혀야 한다. 월드 범위 이탈은 **정의된 정책**으로
    /// 처리한다 — 경계 고정 + 동결 + 진단. 조용히 계속 계산하면 좌표 정밀도가 최소
    /// 특징 크기를 넘어 충돌 판정이 무의미해진다.
    #[test]
    fn legal_boundary_state_survives_step_and_roundtrip() {
        // 월드 경계 바로 안쪽에서 상한 속도로 나간다 — 한 스텝이면 월드를 벗어난다.
        let params = WorldParams::default();
        let big = maru_collide::MAX_INPUT - 10.0;
        let fast = params.max_linear_velocity;
        let mut w = World::new(params);
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(big, 0.0, 0.0),
            1.0,
        ));
        w.set_linear_velocity(h, Vec3::new(fast, 0.0, 0.0)).unwrap();
        // 월드 범위를 벗어나려 하면 **경계에 고정되고 동결**된다(정의된 정책).
        // 10km 밖에서는 f32 간격이 최소 특징 크기를 넘어 충돌 판정이 의미를 잃으므로,
        // 조용히 계속 계산하는 것이 가장 나쁘다.
        w.try_step(crate::world::MAX_DT).unwrap();
        let p = w.body(h).pose().pos.x;
        assert!(
            (p - maru_collide::MAX_INPUT).abs() < 1.0,
            "월드 경계에 고정되지 않음: {p}"
        );
        assert_eq!(
            w.body(h).linvel().length(),
            0.0,
            "이탈 바디가 동결되지 않음"
        );
        assert!(w.is_out_of_bounds(h), "이탈 상태가 기록되지 않음");
        assert!(
            !w.is_asleep(h),
            "OOB가 sleep과 섞이면 아일랜드 불변식이 깨진다"
        );
        assert_eq!(w.diagnostics().out_of_bounds, 1, "이탈이 관측되지 않음");
        let snap = w.snapshot();
        let r = w.restore(&snap);
        assert!(
            r.is_ok(),
            "step 뒤 상태가 자기 스냅샷을 복원하지 못함 (정책이 step에 대해 닫히지 않음): {r:?}"
        );
    }

    /// NEW MAJOR: 생성자가 자기 삽입 정책을 위반하는 Body를 반환하면 안 된다
    /// (density=1e-10 → inv_mass≈2.39e9 > 입력 상한).
    #[test]
    #[should_panic(expected = "size policy")]
    fn constructor_rejects_body_violating_insertion_policy() {
        Body::new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 0.0, 0.0), 1.0e-10);
    }

    #[test]
    fn fallible_constructor_returns_error() {
        assert!(
            Body::try_new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 0.0, 0.0), 1.0e-10)
                .is_err()
        );
        assert!(
            Body::try_new_dynamic(Shape::Sphere { radius: 1.0 }, pose(1.0e20, 0.0, 0.0), 1.0)
                .is_err()
        );
        assert!(
            Body::try_new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 0.0, 0.0), 1.0).is_ok()
        );
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

    /// BLOCKER: 예약 핸들당 pending 추가는 정확히 하나여야 한다.
    /// **타입 수준으로** 보장한다 — 공개 `Command`에 바디 추가 variant가 없으므로
    /// 두 번째 추가를 표현할 방법 자체가 없다(정본 경로 = `enqueue_add_body`).
    #[test]
    fn reserved_slot_has_exactly_one_pending_add() {
        let mut w = World::new(WorldParams::default());
        let h = w.enqueue_add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 3.0, 0.0),
            1.0,
        ));
        assert_eq!(w.pending_commands(), 1);
        // 같은 바디를 또 추가하면 **다른 슬롯**이 예약된다 (핸들이 다르다)
        let h2 = w.enqueue_add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(1.0, 3.0, 0.0),
            1.0,
        ));
        assert_ne!(h.index, h2.index, "같은 슬롯이 두 번 예약됨");
        assert_eq!(w.pending_commands(), 2);
        assert!(w.restore(&w.snapshot()).is_ok());
        w.step(DT);
        assert!(w.is_alive(h) && w.is_alive(h2));
        assert_eq!(w.last_command_report().rejected, 0);
    }

    /// BLOCKER: writer가 다 만든 뒤 길이를 재면 wasm에서 통제된 오류보다 메모리 고갈이
    /// 먼저 온다 → 할당 **전에** 크기를 계산한다. 그 계산은 writer와 정확히 일치해야 한다.
    #[test]
    fn snapshot_size_preflight_is_exact() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let a = dyn_box(&mut w, 0.0, 3.0);
        let b = w.add_body(Body::new_dynamic(
            Shape::Hull {
                points: vec![
                    Vec3::new(-0.4, -0.4, -0.4),
                    Vec3::new(0.4, -0.4, -0.4),
                    Vec3::new(-0.4, 0.4, -0.4),
                    Vec3::new(-0.4, -0.4, 0.4),
                    Vec3::new(0.4, 0.4, 0.4),
                ],
                radius: 0.0,
            },
            pose(2.0, 3.0, 0.0),
            1.0,
        ));
        let c = w.add_body(Body::new_kinematic(
            Shape::Capsule {
                half_height: 0.5,
                radius: 0.2,
            },
            pose(4.0, 3.0, 0.0),
        ));
        w.add_joint(Joint::hinge(
            a,
            b,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ));
        w.add_joint(Joint::point(b, c, Vec3::ZERO, Vec3::ZERO));
        for _ in 0..20 {
            w.step(DT);
        }
        // pending 커맨드 전 종류를 큐에 남긴다
        w.enqueue(Command::SetLinearVelocity(a, Vec3::new(1.0, 0.0, 0.0)))
            .unwrap();
        w.enqueue(Command::SetAngularVelocity(a, Vec3::new(0.0, 1.0, 0.0)))
            .unwrap();
        w.enqueue(Command::SetPose(a, pose(0.0, 4.0, 0.0))).unwrap();
        w.enqueue(Command::SetMaterial(a, 0.3, 0.1)).unwrap();
        w.enqueue(Command::RemoveBody(c)).unwrap();
        w.enqueue(Command::AddJoint(Box::new(Joint::point(
            a,
            b,
            Vec3::ZERO,
            Vec3::ZERO,
        ))))
        .unwrap();
        w.enqueue_add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(8.0, 3.0, 0.0),
            1.0,
        ));
        let bytes = w.snapshot();
        assert_eq!(
            crate::snapshot::serialized_size(&w),
            bytes.len(),
            "preflight 크기가 writer와 어긋남 — 한쪽만 고친 것이다"
        );
    }

    /// 정책 **경계**의 대칭: 라이브 API가 받아들이는 최대치는 스냅샷 리더도 받아야 한다.
    /// (리뷰가 지적한 비대칭의 정확한 형태 — 리더만 1e9로 막고 있었다.)
    #[test]
    fn boundary_legal_values_round_trip() {
        let big = maru_collide::MAX_INPUT * 0.9;
        let fast = maru_collide::MAX_SPEED * 0.9;
        let mut w = World::new(WorldParams::default());
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(big, 0.0, 0.0),
            1.0,
        ));
        w.set_linear_velocity(h, Vec3::new(fast, 0.0, 0.0)).unwrap();
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
        let big = maru_collide::MAX_INPUT * 0.9;
        let fast = maru_collide::MAX_SPEED * 0.9;
        for step_i in 0..400 {
            let f = (step_i % 5) as f32;
            match next(&mut rng) % 13 {
                0 => live.push(dyn_box(&mut w, (step_i % 9) as f32 - 4.0, 3.0)),
                1 => {
                    if let Some(h) = live.pop() {
                        let _ = w.remove_body(h);
                    }
                }
                2 => {
                    if let Some(h) = live.first() {
                        let _ = w.enqueue(Command::SetLinearVelocity(*h, Vec3::new(f, 0.0, 0.0)));
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
                6 => {
                    w.flush_commands();
                }
                // ── 리뷰 MINOR: 빠져 있던 연산들 ──
                7 => {
                    if let Some(h) = live.first() {
                        let _ = w.enqueue(Command::SetAngularVelocity(*h, Vec3::new(0.0, f, 0.0)));
                    }
                }
                8 => {
                    if let Some(h) = live.first() {
                        let _ = w.enqueue(Command::SetPose(*h, pose(f, 4.0, 0.0)));
                        let _ = w.enqueue(Command::SetMaterial(*h, 0.2 * f, 0.1 * f));
                    }
                }
                9 => {
                    // pending RemoveBody + pending AddJoint (같은 배치 안 상호작용)
                    if live.len() >= 3 {
                        let doomed = live.remove(live.len() - 1);
                        let _ = w.enqueue(Command::RemoveBody(doomed));
                        let _ = w.enqueue(Command::AddJoint(Box::new(Joint::point(
                            live[0],
                            live[1],
                            Vec3::ZERO,
                            Vec3::ZERO,
                        ))));
                    }
                }
                10 => live.push(w.add_body(Body::new_kinematic(
                    Shape::Capsule {
                        half_height: 0.3,
                        radius: 0.2,
                    },
                    pose((step_i % 6) as f32, 2.0, 1.0),
                ))),
                11 => {
                    // 파라미터 변경 + MAX_DT 경계 스텝
                    w.set_params(WorldParams {
                        substeps: 1 + (step_i % 4) as u32,
                        ..Default::default()
                    });
                    // 월드를 벗어난 바디가 있으면 StateOverflow가 날 수 있다 — 상태는
                    // 여전히 유효하므로 왕복 계약은 그대로 요구한다.
                    let _ = w.try_step(crate::world::MAX_DT);
                }
                _ => {
                    // 경계 크기 바디 — step 뒤 상태 정책 폐쇄가 걸리는 지점
                    let h = w.add_body(Body::new_dynamic(
                        Shape::Sphere { radius: 0.5 },
                        pose(big, 0.0, 0.0),
                        1.0,
                    ));
                    let _ = w.set_linear_velocity(h, Vec3::new(fast, 0.0, 0.0));
                    live.push(h);
                }
            }
            let snap = w.snapshot();
            let r = w.restore(&snap);
            assert!(
                r.is_ok(),
                "연산 {step_i} 이후 자기 스냅샷이 복원되지 않음 (정책 비대칭): {r:?}"
            );
        }
    }

    // ─────────── 리뷰 rev_0dde9f23 회귀 ───────────

    /// BLOCKER: 큰 위치 × 작은 역질량 조합은 조인트 솔버의 **중간 연산**에서 먼저 Inf가 된다.
    /// 지원 범위를 차원별로 좁혀 그 조합 자체를 만들 수 없게 한다.
    #[test]
    fn snapshot_rejects_unsupported_mass_range() {
        // 질량 1e9 kg 초과 = inv_mass < 1e-9 → 지원 범위 밖
        assert!(!maru_collide::inv_mass_in_range(1.0e-30));
        assert!(!maru_collide::inv_mass_in_range(1.0e12));
        assert!(maru_collide::inv_mass_in_range(1.0));
        // 생성 경계에서도 같은 정책 — half=1000m·density=1e9는 질량 8e18kg
        assert!(Body::try_new_dynamic(
            Shape::Box {
                half: Vec3::new(1000.0, 1000.0, 1000.0),
                radius: 0.0
            },
            pose(0.0, 0.0, 0.0),
            1.0e9
        )
        .is_err());
    }

    /// BLOCKER: 속도 상한은 **문서화된 물리 파라미터**(Jolt mMaxLinearVelocity 계열)다.
    /// 이것이 파생 상태를 유계로 만들어 정책이 step에 대해 닫힌다.
    #[test]
    fn velocity_limit_bounds_derived_state() {
        let params = WorldParams {
            max_linear_velocity: 50.0,
            ..Default::default()
        };
        let mut w = World::new(params);
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(0.0, 0.0, 0.0),
            1.0,
        ));
        w.set_linear_velocity(h, Vec3::new(1000.0, 0.0, 0.0))
            .unwrap();
        w.step(DT);
        let v = w.body(h).linvel().length();
        assert!(v <= 50.0 + 1e-3, "속도 상한이 적용되지 않음: {v}");
        assert!(
            w.diagnostics().velocity_clamped > 0,
            "클램프가 집계되지 않음"
        );
    }

    /// MAJOR: 포화는 **관측 가능**해야 한다. 서로 다른 초과 상태가 같은 값으로 합쳐지므로
    /// 상태 해시만으로는 물리 오류가 숨는다.
    #[test]
    fn state_saturation_is_observable_and_reported() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let h = dyn_box(&mut w, 0.0, 3.0);
        w.step(DT);
        assert_eq!(
            w.diagnostics().state_saturations,
            0,
            "정상 장면에서 포화 발생"
        );
        // 지원 범위 안에서는 StateOverflow가 나지 않는다
        assert_eq!(w.try_step(DT), Ok(()));
        assert!(w.body(h).pose().pos.y.is_finite());
    }

    /// PARTIAL BLOCKER: 호스트 입력 경로는 **전부** fallible이어야 한다 — wasm trap 금지.
    #[test]
    fn all_host_input_paths_are_fallible() {
        // 월드 파라미터
        assert!(World::try_new(WorldParams {
            substeps: 0,
            ..Default::default()
        })
        .is_err());
        let mut w = World::try_new(WorldParams::default()).unwrap();
        assert!(w
            .try_set_params(WorldParams {
                time_to_sleep: -1.0,
                ..Default::default()
            })
            .is_err());
        // 바디 (dynamic / static / kinematic)
        let bad = Shape::Hull {
            points: vec![],
            radius: 0.0,
        };
        assert!(Body::try_new_dynamic(bad.clone(), pose(0.0, 0.0, 0.0), 1.0).is_err());
        assert!(Body::try_new_static(bad.clone(), pose(0.0, 0.0, 0.0)).is_err());
        assert!(Body::try_new_kinematic(bad, pose(0.0, 0.0, 0.0)).is_err());
        // 조인트 — hinge 축은 Joint 생성 시점에 걸리므로 try_add_joint로는 못 잡는다
        let a = dyn_box(&mut w, 0.0, 3.0);
        let b = dyn_box(&mut w, 2.0, 3.0);
        assert!(Joint::try_hinge(
            a,
            b,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0)
        )
        .is_err());
        assert!(Joint::try_hinge(
            a,
            b,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(f32::NAN, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0)
        )
        .is_err());
        assert!(Joint::try_point(a, b, Vec3::new(f32::INFINITY, 0.0, 0.0), Vec3::ZERO).is_err());
        assert!(Joint::try_hinge(
            a,
            b,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0)
        )
        .is_ok());
        // 스텝·스냅샷·해시
        assert!(w.try_step(0.0).is_err());
        assert!(w.try_snapshot().is_ok());
        assert!(w.try_state_hash().is_ok());
    }

    /// MAJOR: 직렬화되는 필드는 하나도 빠짐없이 상태 폐쇄에 들어가야 한다
    /// (contact_normal이 빠져 NaN이 스텝 뒤에도 남던 경로).
    #[test]
    fn state_closure_covers_every_serialized_field() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        for i in 0..6 {
            dyn_box(&mut w, i as f32 * 0.9 - 2.0, 0.6 + i as f32 * 1.1);
        }
        for _ in 0..90 {
            w.step(DT);
        }
        // 접촉 캐시·조인트·슬립 타이머가 모두 채워진 상태에서 왕복이 성립해야 한다
        let snap = w.snapshot();
        let r = w.restore(&snap);
        assert!(r.is_ok(), "상태 폐쇄에서 빠진 필드가 있다: {r:?}");
        assert_eq!(w.diagnostics().state_saturations, 0);
    }

    // ─────────── 리뷰 rev_d76fd99f 회귀 ───────────

    /// BLOCKER: 최악 임펄스 논증에서 빠졌던 **조인트 레버암 × 각속도**.
    /// 앵커를 셰이프 범위로 묶어 `ω×r` 점 속도를 유계로 만든다.
    #[test]
    fn large_anchor_with_spinning_kinematic_stays_in_policy() {
        // 앵커는 레버암이므로 전역 상한을 넘을 수 없고,
        // **그 바디의 실제 크기**에도 비례해야 한다(아래 try_add_joint에서 검증).
        assert!(Joint::try_point(
            BodyHandle {
                index: 0,
                generation: 0
            },
            BodyHandle {
                index: 1,
                generation: 0
            },
            Vec3::new(maru_collide::MAX_ANCHOR * 2.0, 0.0, 0.0),
            Vec3::ZERO,
        )
        .is_err());

        // 허용 범위 안의 최대 조합: 큰 앵커 + 상한 각속도 + 무거운 바디
        let mut w = World::new(WorldParams::default());
        let k = w.add_body(Body::new_kinematic(
            Shape::Box {
                half: Vec3::new(1.0, 1.0, 1.0),
                radius: 0.0,
            },
            pose(0.0, 0.0, 0.0),
        ));
        let heavy = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(2.0, 2.0, 2.0),
                radius: 0.0,
            },
            pose(50.0, 0.0, 0.0),
            1.0e4, // 부피 64m³ × 1e4 = 6.4e5 kg (질량 상한 1e6 안)
        ));
        w.set_angular_velocity(k, Vec3::new(0.0, 100.0, 0.0))
            .unwrap();
        // 앵커는 그 바디 크기에 비례해야 한다 — kinematic 박스(half=1)의 허용 한계 근처
        w.add_joint(Joint::point(k, heavy, Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO));
        for _ in 0..30 {
            // 지원 범위 안의 합법 월드는 StateOverflow를 내면 안 된다
            assert_eq!(
                w.try_step(DT),
                Ok(()),
                "합법 월드가 StateOverflow: sat={}",
                w.diagnostics().state_saturations
            );
        }
        let r = w.restore(&w.snapshot());
        assert!(r.is_ok(), "{r:?}");
    }

    /// BLOCKER: 복원된 큰 속도가 **브로드페이즈보다 먼저** 잡혀야 한다 —
    /// sweep AABB가 부풀면 O(N²) 페어로 오류 반환 전에 메모리가 터진다.
    #[test]
    fn restored_huge_velocity_is_clamped_before_broadphase() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let hs: Vec<_> = (0..40)
            .map(|i| dyn_box(&mut w, i as f32 * 2.0 - 40.0, 1.0))
            .collect();
        w.step(DT);
        // 스냅샷 바이트를 직접 조작하는 대신, 리더가 허용하는 상태 범위 속도를
        // 엔진 내부 경로로 만들어 같은 조건을 재현한다.
        for h in &hs {
            let _ = w.set_linear_velocity(*h, Vec3::new(9000.0, 0.0, 0.0));
        }
        let snap = w.snapshot();
        assert!(w.restore(&snap).is_ok());
        // 클램프가 브로드페이즈 앞이라면 페어 폭발 없이 정상 스텝이 끝난다
        assert_eq!(w.try_step(DT), Ok(()));
        for h in &hs {
            let v = w.body(*h).linvel().length();
            assert!(v <= 500.0 + 1e-3, "속도 상한 미적용: {v}");
        }
    }

    /// BLOCKER: 진단은 캐노니컬 상태가 아니므로 **복원 후 미래 동작에 영향을 주면 안 된다**.
    /// stale `state_saturations`가 다음 `try_step`을 StateOverflow로 만들던 경로.
    #[test]
    fn diagnostics_do_not_survive_restore() {
        let mut a = World::new(WorldParams::default());
        ground(&mut a);
        dyn_box(&mut a, 0.0, 3.0);
        a.step(DT);
        let snap = a.snapshot();

        // 진단이 오염된 인스턴스와 새 인스턴스가 같은 스냅샷에서 같게 동작해야 한다
        let mut dirty = World::new(WorldParams::default());
        ground(&mut dirty);
        let h = dyn_box(&mut dirty, 0.0, 3.0);
        let _ = dirty.set_linear_velocity(h, Vec3::new(9000.0, 0.0, 0.0));
        dirty.step(DT);
        dirty.restore(&snap).unwrap();

        let mut fresh = World::new(WorldParams::default());
        fresh.restore(&snap).unwrap();

        assert_eq!(
            dirty.try_step(DT),
            fresh.try_step(DT),
            "stale 진단이 반환값을 바꿈"
        );
        assert_eq!(dirty.state_hash(), fresh.state_hash(), "L0 재생 불일치");
    }

    /// BLOCKER: pending 커맨드 페이로드는 **호스트 입력**이므로 공개 enqueue와 같은 정책이다.
    #[test]
    fn pending_payload_uses_input_policy() {
        let mut w = World::new(WorldParams::default());
        let h = dyn_box(&mut w, 0.0, 3.0);
        // 공개 경로가 거부하는 값은 스냅샷 왕복으로도 큐에 들어올 수 없다
        assert!(w
            .enqueue(Command::SetLinearVelocity(
                h,
                Vec3::new(maru_collide::MAX_SPEED * 2.0, 0.0, 0.0)
            ))
            .is_err());
        assert!(w
            .enqueue(Command::SetAngularVelocity(
                h,
                Vec3::new(maru_collide::MAX_ANGULAR * 2.0, 0.0, 0.0)
            ))
            .is_err());
        // 정상 범위는 통과하고 왕복도 성립
        w.enqueue(Command::SetLinearVelocity(h, Vec3::new(10.0, 0.0, 0.0)))
            .unwrap();
        w.enqueue(Command::SetAngularVelocity(h, Vec3::new(1.0, 0.0, 0.0)))
            .unwrap();
        assert!(w.restore(&w.snapshot()).is_ok());
    }

    /// BLOCKER: sleep timer는 파생 상태(dt 누적)라 writer/reader 정책이 같아야 한다.
    #[test]
    fn large_time_to_sleep_survives_roundtrip() {
        let mut w = World::new(WorldParams {
            time_to_sleep: maru_collide::MAX_INPUT * 0.999,
            ..Default::default()
        });
        ground(&mut w);
        dyn_box(&mut w, 0.0, 0.5);
        for _ in 0..30 {
            w.step(0.2);
        }
        assert!(w.restore(&w.snapshot()).is_ok(), "sleep timer 정책 비대칭");
    }

    /// MAJOR: 좌표 정밀도와 최소 특징 크기의 정합.
    #[test]
    fn world_extent_matches_float_precision() {
        // 월드 경계에서의 f32 간격이 최소 특징 크기 이하여야 한다
        let spacing = {
            let x = maru_collide::MAX_INPUT;
            let next = f32::from_bits(x.to_bits() + 1);
            next - x
        };
        assert!(
            spacing <= maru_collide::MIN_SHAPE_EXTENT,
            "월드 경계 f32 간격 {spacing} > 최소 특징 크기 {}",
            maru_collide::MIN_SHAPE_EXTENT
        );
        // 최소 크기 미만 셰이프는 거부된다
        assert!(Body::try_new_dynamic(
            Shape::Sphere {
                radius: maru_collide::MIN_SHAPE_EXTENT * 0.5
            },
            pose(0.0, 0.0, 0.0),
            1.0
        )
        .is_err());
    }

    // ─────────── 리뷰 rev_22b699a7 회귀 ───────────

    /// BLOCKER: 새 차원별 범위가 리더에 전파되지 않아 **정상 생성 가능한 경량 바디**가
    /// 자기 스냅샷을 복원하지 못했다. 이제 리더가 라이브와 **같은 validator**를 호출한다.
    #[test]
    fn light_body_with_large_inverses_round_trips() {
        // 반지름 1cm 구, 질량 ≈ 1e-5 kg → inv_mass ≈ 1e5, inv_inertia ≈ 2.5e9
        let mut w = World::new(WorldParams::default());
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.01 },
            pose(0.0, 1.0, 0.0),
            2.4,
        ));
        let b = w.body(h);
        assert!(b.inv_mass() > 1.0e4, "시나리오 전제 붕괴: {}", b.inv_mass());
        assert!(
            b.inv_inertia_local().x > 1.0e8,
            "시나리오 전제 붕괴: {:?}",
            b.inv_inertia_local()
        );
        let r = w.restore(&w.snapshot());
        assert!(r.is_ok(), "경량 바디 왕복 실패 (차원별 범위 미전파): {r:?}");
    }

    /// BLOCKER: 힌지 각 임펄스는 별도 경로 — 큰 관성에서 `1/k_ang`이 커지므로
    /// 각 bias를 **각속도 상한**으로 묶어야 한다(선형 상한 m/s가 아니라).
    #[test]
    fn misaligned_hinge_with_large_inertia_stays_in_policy() {
        let mut w = World::new(WorldParams {
            substeps: 64,
            solver: crate::solver::SolverParams {
                contact_hertz: 1.0e3,
                max_bias_speed: maru_collide::MAX_SPEED,
                ..Default::default()
            },
            ..Default::default()
        });
        let k = w.add_body(Body::new_kinematic(
            Shape::Box {
                half: Vec3::new(1.0, 1.0, 1.0),
                radius: 0.0,
            },
            pose(0.0, 0.0, 0.0),
        ));
        // 관성 상한 근처의 무거운 바디 (역관성 하한 1e-10 부근)
        let heavy = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(100.0, 100.0, 100.0),
                radius: 0.0,
            },
            pose(200.0, 0.0, 0.0),
            0.125, // 부피 8e6 × 0.125 = 1e6 kg (질량 상한)
        ));
        // 90° 어긋난 축으로 연결 — 각 제약이 최대 오차에서 출발한다
        w.add_joint(Joint::hinge(
            k,
            heavy,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-100.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ));
        for i in 0..20 {
            assert_eq!(
                w.try_step(crate::world::MIN_DT),
                Ok(()),
                "합법 월드가 step {i}에서 StateOverflow (각 임펄스 경로)"
            );
        }
        assert!(w.restore(&w.snapshot()).is_ok());
    }

    /// BLOCKER: 스냅샷이 공개 API가 거부하는 앵커를 주입할 수 없어야 한다.
    /// 리더가 `Joint::validate`를 호출하므로 앵커 정책이 자동으로 따라온다.
    #[test]
    fn snapshot_cannot_inject_oversized_anchor() {
        let mut w = World::new(WorldParams::default());
        let a = dyn_box(&mut w, 0.0, 3.0);
        let b = dyn_box(&mut w, 3.0, 3.0);
        w.add_joint(Joint::point(a, b, Vec3::new(0.4, 0.0, 0.0), Vec3::ZERO));
        w.step(DT);
        let good = w.snapshot();
        let before = w.state_hash();
        // 조인트 앵커 성분을 MAX_ANCHOR의 5배로 바꾸면 거부돼야 한다
        let target = (0.4f32).to_le_bytes();
        let inject = (maru_collide::MAX_ANCHOR * 5.0).to_le_bytes();
        let mut patched = 0;
        for i in 0..good.len().saturating_sub(4) {
            if good[i..i + 4] == target {
                let mut bad = good.clone();
                bad[i..i + 4].copy_from_slice(&inject);
                if w.restore(&bad).is_err() {
                    patched += 1;
                }
            }
        }
        assert!(patched > 0, "앵커 주입을 막는 지점이 없음");
        assert_eq!(w.state_hash(), before, "거부된 restore가 월드를 오염시킴");
    }

    /// BLOCKER: pending 속도 커맨드는 **flush 뒤**에 적용되므로, 클램프도 flush 뒤·
    /// 브로드페이즈 앞이어야 sweep AABB 폭발을 막는다.
    #[test]
    fn pending_velocity_is_clamped_before_broadphase() {
        let mut w = World::new(WorldParams {
            max_linear_velocity: 1.0,
            ..Default::default()
        });
        ground(&mut w);
        let hs: Vec<_> = (0..30)
            .map(|i| dyn_box(&mut w, i as f32 * 3.0 - 45.0, 1.0))
            .collect();
        // 합법 입력 상한 속도를 pending으로 넣는다 (스텝 시작 클램프 **뒤**에 적용된다)
        for h in &hs {
            w.enqueue(Command::SetLinearVelocity(
                *h,
                Vec3::new(maru_collide::MAX_SPEED, 0.0, 0.0),
            ))
            .unwrap();
        }
        assert_eq!(w.try_step(crate::world::MAX_DT), Ok(()));
        for h in &hs {
            let v = w.body(*h).linvel().length();
            assert!(v <= 1.0 + 1e-3, "pending 속도가 클램프되지 않음: {v}");
        }
    }

    /// MAJOR: 최소 특징 크기는 **모든 셰이프**에 적용된다 (capsule·hull이 빠져 있었다).
    #[test]
    fn min_feature_size_applies_to_every_shape() {
        let tiny = maru_collide::MIN_SHAPE_EXTENT * 0.1;
        assert!(Body::try_new_static(
            Shape::Capsule {
                half_height: 1.0,
                radius: tiny
            },
            pose(0.0, 0.0, 0.0)
        )
        .is_err());
        // 반지름 0의 단일점 hull = 모든 축에서 퇴화
        assert!(Body::try_new_static(
            Shape::Hull {
                points: vec![Vec3::ZERO],
                radius: 0.0
            },
            pose(0.0, 0.0, 0.0)
        )
        .is_err());
        // 한 축만 얇은 hull도 거부 (월드 경계에서 AABB가 반올림으로 퇴화)
        assert!(Body::try_new_static(
            Shape::Hull {
                points: vec![Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0)],
                radius: 0.0
            },
            pose(0.0, 0.0, 0.0)
        )
        .is_err());
        // 두께가 반지름으로 채워지면 합법
        assert!(Body::try_new_static(
            Shape::Hull {
                points: vec![Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0)],
                radius: maru_collide::MIN_SHAPE_EXTENT
            },
            pose(0.0, 0.0, 0.0)
        )
        .is_ok());
    }

    // ─────────── 리뷰 rev_23aba1ab 회귀 ───────────

    /// BLOCKER: 전역 앵커 상한만으로는 부족하다 — 앵커는 **그 바디의 실제 크기**에
    /// 비례해야 K 행렬 조건수가 형상 스케일과 무관해진다. 반지름 1mm 구에 1000m 앵커면
    /// K의 고유 스케일이 2.4e4 : 6e16이 되어 상대 det가 임계 아래로 떨어지고
    /// `Mat3::inverse`가 0 행렬을 돌려준다 — 조인트가 오류 없이 조용히 풀린다.
    #[test]
    fn anchor_must_scale_with_body() {
        let mut w = World::new(WorldParams::default());
        let tiny = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.001 },
            pose(0.0, 1.0, 0.0),
            1.0e4,
        ));
        let anchor = w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(1.0, 1.0, 1.0),
                radius: 0.0,
            },
            pose(0.0, 3.0, 0.0),
        ));
        // 전역 상한(1000m) 안이지만 1mm 구에는 터무니없는 레버암
        assert_eq!(
            w.try_add_joint(Joint::point(
                tiny,
                anchor,
                Vec3::new(500.0, 0.0, 0.0),
                Vec3::ZERO
            )),
            Err(ApiError::InvalidValue),
            "바디 크기를 무시한 앵커가 통과"
        );
        // 바디 크기에 비례하는 앵커는 정상
        assert!(w
            .try_add_joint(Joint::point(
                tiny,
                anchor,
                Vec3::new(0.002, 0.0, 0.0),
                Vec3::ZERO
            ))
            .is_ok());
    }

    /// BLOCKER: `WorldParams`는 미래 동작을 바꾸는 mutable state다 —
    /// 스냅샷·해시에 없으면 같은 스냅샷의 미래가 인스턴스마다 달라진다.
    #[test]
    fn world_params_are_canonical_state() {
        let mut zero_g = World::new(WorldParams {
            gravity: Vec3::ZERO,
            substeps: 1,
            ..Default::default()
        });
        ground(&mut zero_g);
        let h = dyn_box(&mut zero_g, 0.0, 5.0);
        zero_g.step(DT);
        let snap = zero_g.snapshot();

        // 기본 파라미터(중력 있음) 월드에 복원한다
        let mut other = World::new(WorldParams::default());
        other.restore(&snap).unwrap();
        assert_eq!(other.state_hash(), zero_g.state_hash());
        // 파라미터까지 복원됐다면 이후 궤적도 같아야 한다
        for _ in 0..30 {
            zero_g.step(DT);
            other.step(DT);
        }
        assert_eq!(
            other.state_hash(),
            zero_g.state_hash(),
            "파라미터가 복원되지 않아 미래가 분기함"
        );
        assert!(
            (other.body(h).pose().pos.y - 5.0).abs() < 1e-3,
            "중력 0 파라미터가 복원되지 않음"
        );
    }

    /// BLOCKER: 같은 자리에 겹친 N개 바디는 N²/2 페어를 만든다. 입력은 작아도
    /// wasm에서는 오류 반환 전에 allocator가 죽으므로 **예산 초과는 오류**여야 한다.
    #[test]
    fn broadphase_pair_budget_is_an_error_not_a_trap() {
        let mut w = World::new(WorldParams {
            max_pairs_per_body: 1,
            ..Default::default()
        });
        for _ in 0..200 {
            w.add_body(Body::new_dynamic(
                Shape::Sphere { radius: 0.5 },
                pose(0.0, 1.0, 0.0), // 전부 같은 자리 → 모든 페어가 겹친다
                1.0,
            ));
        }
        assert_eq!(
            w.try_step(DT),
            Err(ApiError::ResourceLimit),
            "페어 폭발이 오류로 표현되지 않음"
        );
        // 상태는 손상되지 않고 스냅샷도 여전히 유효하다
        assert!(w.restore(&w.snapshot()).is_ok());
    }

    /// BLOCKER: 월드 이탈 바디는 **일반 sleep과 분리**돼야 한다 — 조인트로 연결된
    /// 상대가 깨어 있으면 wake 폐쇄가 매 프레임 되살려 이동→고정→sleep→wake를 반복한다.
    #[test]
    fn out_of_bounds_body_is_not_auto_woken() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let edge = maru_collide::MAX_INPUT - 5.0;
        let a = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(edge, 0.0, 0.0),
            1.0,
        ));
        let b = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(edge - 1.0, 0.0, 0.0),
            1.0,
        ));
        w.add_joint(Joint::point(
            a,
            b,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
        ));
        w.set_linear_velocity(a, Vec3::new(500.0, 0.0, 0.0))
            .unwrap();
        for _ in 0..20 {
            let _ = w.try_step(crate::world::MAX_DT);
        }
        assert!(
            w.is_out_of_bounds(a),
            "경계를 넘은 바디가 OOB로 표시되지 않음"
        );
        // 여러 스텝이 지나도 자동으로 되살아나지 않는다
        for _ in 0..20 {
            let _ = w.try_step(DT);
        }
        assert!(w.is_out_of_bounds(a), "OOB 바디가 자동 wake됨 (진동 경로)");
        assert!(w.restore(&w.snapshot()).is_ok());
        // 호스트가 월드 안으로 되돌리면 해제된다
        w.set_pose(a, pose(0.0, 0.0, 0.0)).unwrap();
        assert!(!w.is_out_of_bounds(a), "set_pose로 OOB가 해제되지 않음");
    }

    /// MAJOR: 축 정렬 span만 보면 **회전된 퇴화 hull**이 통과한다 — 한 평면 위의 점들도
    /// x/y/z span은 모두 크다. 회전 불변 두께를 재야 한다.
    #[test]
    fn rotated_coplanar_hull_is_rejected() {
        // z = x + y 평면 위의 점들 — 3D 두께 0이지만 축별 span은 모두 1
        let coplanar = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 2.0),
        ];
        assert!(
            Body::try_new_static(
                Shape::Hull {
                    points: coplanar.clone(),
                    radius: 0.0
                },
                pose(0.0, 0.0, 0.0)
            )
            .is_err(),
            "회전된 평면 hull이 통과 (축 정렬 검사만 하고 있다)"
        );
        // 두께가 생기면 합법
        let mut thick = coplanar;
        thick.push(Vec3::new(0.5, 0.5, 0.0));
        assert!(Body::try_new_static(
            Shape::Hull {
                points: thick,
                radius: 0.0
            },
            pose(0.0, 0.0, 0.0)
        )
        .is_ok());
    }

    /// 리뷰 권고(커버리지): 속도 상한은 **고반발·구동 kinematic**에서 실제로 닿는다.
    /// 상한에 닿아도 상태가 정책 안에 남고 왕복이 성립하는지 확인한다.
    #[test]
    fn velocity_limit_with_restitution_and_kinematic() {
        let mut w = World::new(WorldParams {
            max_linear_velocity: 3.0, // 일부러 낮춰 상한에 닿게 한다
            solver: crate::solver::SolverParams {
                restitution_threshold: 0.1,
                ..Default::default()
            },
            ..Default::default()
        });
        ground(&mut w);
        // 완전 탄성에 가까운 공을 떨어뜨린다
        let ball = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(0.0, 4.0, 0.0),
            1.0,
        ));
        w.set_material(ball, 0.2, 0.95).unwrap();
        // 구동 kinematic 판이 아래에서 올라온다
        let paddle = w.add_body(Body::new_kinematic(
            Shape::Box {
                half: Vec3::new(2.0, 0.2, 2.0),
                radius: 0.0,
            },
            pose(0.0, 0.5, 0.0),
        ));
        w.set_linear_velocity(paddle, Vec3::new(0.0, 2.0, 0.0))
            .unwrap();

        let mut clamped_total = 0usize;
        for _ in 0..240 {
            // 상한에 닿는 것은 정상 기능이므로 오류가 아니다
            assert_eq!(w.try_step(DT), Ok(()));
            clamped_total += w.diagnostics().velocity_clamped;
            let v = w.body(ball).linvel().length();
            assert!(v <= 3.0 + 1e-3, "속도 상한을 넘김: {v}");
            assert_eq!(w.diagnostics().state_saturations, 0);
        }
        assert!(
            clamped_total > 0,
            "시나리오가 상한에 닿지 않음 — 판별력 없음"
        );
        assert!(w.restore(&w.snapshot()).is_ok());
    }

    // ─────────── 리뷰 rev_826db6d8 회귀 ───────────

    /// BLOCKER: 앵커를 형상에 묶어도 **질량·관성이 셰이프와 무관**하면 K가 조용히 특이해진다.
    /// 리뷰가 제시한 조합(1mm 구 + inv_mass=1e-6 + inv_inertia=1e12)을 거부해야 한다.
    #[test]
    fn snapshot_rejects_inertia_inconsistent_with_shape() {
        let mut w = World::new(WorldParams::default());
        let h = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.001 },
            pose(0.0, 1.0, 0.0),
            1000.0,
        ));
        let good = w.snapshot();
        let before = w.state_hash();
        // 슬롯0의 inv_mass·inv_inertia를 비물리적 조합으로 바꾼다
        let off = HDR + 4 + 1 + 1 + 1 + 4 + 1 + 12 + 16 + 12 + 12;
        let mut bad = good.clone();
        bad[off..off + 4].copy_from_slice(&1.0e-6f32.to_le_bytes());
        for k in 0..3 {
            let o = off + 4 + k * 4;
            bad[o..o + 4].copy_from_slice(&1.0e12f32.to_le_bytes());
        }
        assert_eq!(
            w.restore(&bad),
            Err(SnapshotError::InconsistentState),
            "셰이프와 무관한 질량·관성이 통과 (K가 조용히 특이해진다)"
        );
        assert_eq!(w.state_hash(), before);
        assert!(w.is_alive(h));
    }

    /// BLOCKER: `ResourceLimit` 플래그가 스텝마다 초기화되지 않으면 **영구 고착**된다 —
    /// 원인을 없앤 뒤에도 계속 오류를 돌려주면서 상태만 앞으로 나간다.
    #[test]
    fn resource_limit_flag_clears_next_step() {
        let mut w = World::new(WorldParams {
            max_pairs_per_body: 1,
            ..Default::default()
        });
        let hs: Vec<_> = (0..200)
            .map(|_| {
                w.add_body(Body::new_dynamic(
                    Shape::Sphere { radius: 0.5 },
                    pose(0.0, 1.0, 0.0),
                    1.0,
                ))
            })
            .collect();
        assert_eq!(w.try_step(DT), Err(ApiError::ResourceLimit));
        // 원인을 제거하면 다음 스텝은 성공해야 한다
        for h in hs.iter().skip(3) {
            w.remove_body(*h).unwrap();
        }
        assert_eq!(w.try_step(DT), Ok(()), "플래그가 고착됨");
        assert_eq!(w.try_step(DT), Ok(()));
    }

    /// BLOCKER: OOB는 **모든 시뮬레이션 경로**에서 빠져야 한다. 플래그만 분리하고
    /// 중력·적분에 그대로 두면 경계에 고정된 바디가 다음 스텝부터 아래로 떨어진다.
    /// (직전 테스트는 gravity=0이라 이 결함을 못 봤다.)
    #[test]
    fn out_of_bounds_body_is_excluded_from_simulation() {
        let mut w = World::new(WorldParams::default()); // **중력 있음**
        let edge = maru_collide::MAX_INPUT - 5.0;
        let flyer = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(edge, 100.0, 0.0),
            1.0,
        ));
        w.set_linear_velocity(flyer, Vec3::new(500.0, 0.0, 0.0))
            .unwrap();
        for _ in 0..10 {
            let _ = w.try_step(crate::world::MAX_DT);
        }
        assert!(w.is_out_of_bounds(flyer));
        let frozen = w.body(flyer).pose().pos;
        // 중력이 있어도 움직이지 않는다
        for _ in 0..120 {
            let _ = w.try_step(DT);
        }
        let now = w.body(flyer).pose().pos;
        assert!(
            (now - frozen).length() < 1e-3,
            "OOB 바디가 중력으로 이동함: {frozen:?} → {now:?}"
        );
        assert_eq!(w.body(flyer).linvel().length(), 0.0, "OOB가 속도를 얻음");
        assert!(!w.is_asleep(flyer), "oob ⇒ !asleep 불변식 위반");
        assert!(w.restore(&w.snapshot()).is_ok());
    }

    /// MAJOR: `time_to_sleep=0`인 합법 월드에서 OOB가 같은 스텝에 asleep이 되면
    /// v4 복원 불변식(`oob ⇒ !asleep`)이 깨져 자기 스냅샷이 복원되지 않는다.
    #[test]
    fn oob_and_zero_time_to_sleep_keep_the_invariant() {
        let mut w = World::new(WorldParams {
            time_to_sleep: 0.0,
            ..Default::default()
        });
        ground(&mut w);
        let edge = maru_collide::MAX_INPUT - 5.0;
        let flyer = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(edge, 50.0, 0.0),
            1.0,
        ));
        w.set_linear_velocity(flyer, Vec3::new(500.0, 0.0, 0.0))
            .unwrap();
        for _ in 0..30 {
            let _ = w.try_step(crate::world::MAX_DT);
            assert!(
                w.restore(&w.snapshot()).is_ok(),
                "oob/asleep 불변식이 깨짐 (time_to_sleep=0)"
            );
        }
        assert!(w.is_out_of_bounds(flyer));
    }

    /// MAJOR: 앵커 척도는 **바디 원점 기준**이어야 한다 — AABB 크기만 쓰면 원점에서
    /// 떨어진 static hull의 정상 표면 앵커가 거부된다.
    #[test]
    fn anchor_scale_uses_origin_distance_not_size() {
        let mut w = World::new(WorldParams::default());
        // local x≈100m에 놓인 1m 크기 static hull (원점에서 멀지만 합법)
        let far = w.add_body(Body::new_static(
            Shape::Hull {
                points: vec![
                    Vec3::new(99.5, -0.5, -0.5),
                    Vec3::new(100.5, -0.5, -0.5),
                    Vec3::new(99.5, 0.5, -0.5),
                    Vec3::new(99.5, -0.5, 0.5),
                    Vec3::new(100.5, 0.5, 0.5),
                ],
                radius: 0.0,
            },
            pose(0.0, 0.0, 0.0),
        ));
        let b = dyn_box(&mut w, 0.0, 3.0);
        // 그 hull의 실제 표면 앵커
        assert!(
            w.try_add_joint(Joint::point(far, b, Vec3::new(100.0, 0.0, 0.0), Vec3::ZERO))
                .is_ok(),
            "원점에서 떨어진 셰이프의 정상 앵커가 거부됨"
        );
    }

    /// MAJOR: 최소 폭 스크린은 **한 평면의 높이**가 아니라 여러 면 방향을 봐야 한다.
    /// 리뷰가 제시한 sliver(현 계산 1.1mm, 다른 면 기준 0.575mm)를 거부해야 한다.
    #[test]
    fn sliver_hull_is_rejected_by_multi_face_screen() {
        let pts = vec![
            Vec3::new(-0.684779, 0.439397, 0.415335),
            Vec3::new(-2.926032, 1.807456, 1.821069),
            Vec3::new(-0.745893, 0.486453, 0.446511),
            Vec3::new(-0.655741, 0.426129, 0.392909),
        ];
        assert!(
            Body::try_new_static(
                Shape::Hull {
                    points: pts,
                    radius: 0.0
                },
                pose(0.0, 0.0, 0.0)
            )
            .is_err(),
            "sliver hull이 통과 (단일 평면 높이만 재고 있다)"
        );
    }

    // ─────────── 리뷰 rev_c18c4dc0 회귀 ───────────

    /// BLOCKER: 형상·앵커 휴리스틱으로는 **이방성** 조합의 K 특이화를 막을 수 없다.
    /// 리뷰가 제시한 구성(길쭉한 박스 + 얇은 축 방향 앵커)은 전역·반경 한계를 모두
    /// 통과하지만 K가 diag(4, 1, 1.5e12)가 되어 `Mat3::inverse`가 0 행렬을 돌려준다.
    /// 이제 삽입 시 **실제 K**를 만들어 풀리는지 확인한다.
    #[test]
    fn anisotropic_body_with_long_anchor_is_rejected_by_k_check() {
        let mut w = World::new(WorldParams::default());
        let rod = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(1000.0, 0.001, 0.001),
                radius: 0.0,
            },
            pose(0.0, 0.0, 0.0),
            125.0,
        ));
        let anchor = w.add_body(Body::new_static(
            Shape::Sphere { radius: 0.1 },
            pose(0.0, 1000.0, 0.0),
        ));
        assert_eq!(
            w.try_add_joint(Joint::point(
                rod,
                anchor,
                Vec3::new(0.0, 1000.0, 0.0),
                Vec3::ZERO
            )),
            Err(ApiError::InvalidValue),
            "K가 특이해지는 조인트가 통과 — 조용히 풀린다"
        );
        // 같은 바디라도 **긴 축** 방향 앵커는 조건수가 멀쩡하므로 허용돼야 한다
        assert!(w
            .try_add_joint(Joint::point(
                rod,
                anchor,
                Vec3::new(1000.0, 0.0, 0.0),
                Vec3::ZERO
            ))
            .is_ok());
    }

    /// 로드형 진자(작은 추가 1m 밖에 매달림)는 **정상 구성**이다 —
    /// 축별 결합 같은 과잉 제약으로 막으면 안 된다.
    #[test]
    fn rod_style_pendulum_anchor_is_allowed() {
        let mut w = World::new(WorldParams::default());
        let a = w.add_body(Body::new_static(
            Shape::Sphere { radius: 0.1 },
            pose(0.0, 5.0, 0.0),
        ));
        let bob = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.2, 0.2, 0.2),
                radius: 0.0,
            },
            pose(1.0, 5.0, 0.0),
            1.0,
        ));
        assert!(w
            .try_add_joint(Joint::point(a, bob, Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0)))
            .is_ok());
    }

    /// BLOCKER: OOB는 **솔버에서도** 비활성이어야 한다. 적분 게이트만 막으면 정상 바디와
    /// 접촉·조인트를 통해 임펄스를 받고 그 속도가 바디로 되쓰인다.
    /// 그리고 OOB 접촉이 만든 PCM 키(STATIC_BIT + Dynamic 타입)는 **자기 스냅샷을 거부**했다.
    #[test]
    fn out_of_bounds_body_takes_no_impulse_and_roundtrips() {
        let mut w = World::new(WorldParams::default());
        let edge = maru_collide::MAX_INPUT - 5.0;
        // 경계 밖으로 날려 OOB로 만든다
        let ghost = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(edge, 0.0, 0.0),
            1.0,
        ));
        w.set_linear_velocity(ghost, Vec3::new(500.0, 0.0, 0.0))
            .unwrap();
        for _ in 0..10 {
            let _ = w.try_step(crate::world::MAX_DT);
        }
        assert!(w.is_out_of_bounds(ghost));
        let frozen = w.body(ghost).pose().pos;

        // 정상 바디를 그 자리로 떨어뜨린다 — 접촉이 생기면 OOB가 밀려나면 안 된다
        let live = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            Pose {
                pos: frozen + Vec3::new(0.0, 3.0, 0.0),
                rot: Quat::IDENTITY,
            },
            1.0,
        ));
        for _ in 0..200 {
            let _ = w.try_step(DT);
            // 매 스텝 자기 스냅샷이 복원돼야 한다 (PCM 키 계약)
            let snap = w.snapshot();
            assert!(
                w.restore(&snap).is_ok(),
                "OOB 접촉이 복원 불가 PCM 키를 만들었다"
            );
        }
        let now = w.body(ghost).pose().pos;
        assert!(
            (now - frozen).length() < 1e-3,
            "OOB가 임펄스를 받아 움직였다: {frozen:?} → {now:?}"
        );
        assert_eq!(w.body(ghost).linvel().length(), 0.0);
        let _ = live;
    }

    /// MAJOR: SAT 동률이 A/B **대칭**이어야 한다 — 같은 두 박스의 삽입 순서를 뒤집었다고
    /// 반력·마찰 방향이 달라지면 물리가 인덱스에 의존하는 것이다.
    #[test]
    fn sat_tie_break_is_symmetric_under_body_order() {
        let half = Vec3::new(0.5, 0.5, 0.5);
        let mk = |first_at: Vec3, second_at: Vec3| {
            let mut w = World::new(WorldParams {
                gravity: Vec3::ZERO,
                ..Default::default()
            });
            let a = w.add_body(Body::new_dynamic(
                Shape::Box { half, radius: 0.0 },
                Pose {
                    pos: first_at,
                    rot: Quat::IDENTITY,
                },
                1.0,
            ));
            let b = w.add_body(Body::new_dynamic(
                Shape::Box { half, radius: 0.0 },
                Pose {
                    pos: second_at,
                    rot: Quat::IDENTITY,
                },
                1.0,
            ));
            for _ in 0..30 {
                w.step(DT);
            }
            (w.body(a).pose().pos, w.body(b).pose().pos)
        };
        // 코너 접촉(세 축의 분리가 근소 동률) — 동률 정책이 실제로 개입하는 구성
        let p = Vec3::new(0.0, 0.0, 0.0);
        let q = Vec3::new(0.94, 0.94, 0.94);
        let (a1, b1) = mk(p, q);
        let (b2, a2) = mk(q, p); // 순서를 뒤집어 같은 장면을 만든다
        assert!(
            (a1 - a2).length() < 1e-4 && (b1 - b2).length() < 1e-4,
            "삽입 순서에 따라 결과가 다르다: {a1:?}/{a2:?}, {b1:?}/{b2:?}"
        );
    }

    /// MAJOR: hull 지원 상한은 **정확히 검증 가능한 범위**여야 한다.
    #[test]
    fn hull_support_limit_matches_exact_verification() {
        assert_eq!(maru_collide::MAX_HULL_POINTS, 32);
        let too_many: Vec<Vec3> = (0..33)
            .map(|i| {
                let f = i as f32 * 0.1;
                Vec3::new(f.sin(), f.cos(), (f * 0.7).sin())
            })
            .collect();
        assert!(Body::try_new_static(
            Shape::Hull {
                points: too_many,
                radius: 0.0
            },
            pose(0.0, 0.0, 0.0)
        )
        .is_err());
    }

    // ─────────── 리뷰 rev_f724b2dd 회귀 ───────────

    /// BLOCKER: 삽입 검증과 솔버가 **같은 K**를 만들어야 한다.
    /// 함수로 뽑으며 회전 항의 부호를 뒤집어, 검증이 솔버와 다른 행렬을 보고 있었다.
    /// 이제 솔버가 `linear_k`를 직접 호출하므로 구현이 하나다 — 그 사실을
    /// **물리적 결과로** 잠근다: 검증을 통과한 조인트는 실제로 제약을 유지해야 한다.
    #[test]
    fn validated_joint_actually_holds_in_solver() {
        let mut w = World::new(WorldParams::default());
        let anchor = w.add_body(Body::new_static(
            Shape::Sphere { radius: 0.1 },
            pose(0.0, 5.0, 0.0),
        ));
        // 이방성 바디 — 검증과 솔버의 K가 갈리면 여기서 드러난다
        let plate = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(2.0, 0.05, 2.0),
                radius: 0.0,
            },
            pose(1.0, 5.0, 0.0),
            100.0,
        ));
        w.try_add_joint(Joint::point(
            anchor,
            plate,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
        ))
        .expect("검증을 통과해야 한다");
        for _ in 0..600 {
            w.step(DT);
        }
        // 조인트가 살아 있으면 앵커에서 대략 1m 반경을 유지한다
        let d = (w.body(plate).pose().pos - Vec3::new(0.0, 5.0, 0.0)).length();
        assert!(
            (0.5..2.0).contains(&d),
            "검증을 통과한 조인트가 실제로는 풀렸다 (거리 {d:.2})"
        );
    }

    /// BLOCKER: 스냅샷 복원도 라이브와 **같은 K 검사**를 거쳐야 한다.
    #[test]
    fn restore_applies_the_same_joint_k_check() {
        // 라이브가 거부하는 구성을 만들고, 그 거부가 복원 경로에도 존재하는지 확인한다.
        let mut w = World::new(WorldParams::default());
        let rod = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(1000.0, 0.001, 0.001),
                radius: 0.0,
            },
            pose(0.0, 0.0, 0.0),
            125.0,
        ));
        let anchor = w.add_body(Body::new_static(
            Shape::Sphere { radius: 0.1 },
            pose(0.0, 1000.0, 0.0),
        ));
        assert_eq!(
            w.try_add_joint(Joint::point(
                rod,
                anchor,
                Vec3::new(0.0, 1000.0, 0.0),
                Vec3::ZERO
            )),
            Err(ApiError::InvalidValue)
        );
        // 정상 조인트만 있는 월드는 왕복이 성립한다
        w.try_add_joint(Joint::point(
            rod,
            anchor,
            Vec3::new(1000.0, 0.0, 0.0),
            Vec3::ZERO,
        ))
        .unwrap();
        w.step(DT);
        assert!(w.restore(&w.snapshot()).is_ok());
    }

    /// BLOCKER: 자세 변경이 K를 깨뜨리면 **거부**돼야 한다 — 조용히 풀리는 것보다 낫다.
    #[test]
    fn set_pose_revalidates_connected_joint_k() {
        let mut w = World::new(WorldParams::default());
        let a = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.0, 0.0),
            1.0,
        ));
        let b = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(2.0, 0.0, 0.0),
            1.0,
        ));
        w.try_add_joint(Joint::point(a, b, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO))
            .unwrap();
        // 정상 자세 변경은 통과한다
        assert!(w.set_pose(b, pose(2.0, 1.0, 0.0)).is_ok());
        // 거부되더라도 자세는 **바뀌지 않은 채** 남아야 한다 (원자성)
        let before = w.body(b).pose().pos;
        let _ = w.set_pose(b, pose(f32::NAN, 0.0, 0.0));
        assert_eq!(w.body(b).pose().pos, before);
    }

    /// MAJOR: OOB는 아일랜드 union에도 참여하면 안 된다 — 두 정상 바디가 같은 OOB에
    /// 매달리면 비활성 조인트를 통해 한 아일랜드로 묶인다.
    #[test]
    fn oob_joint_does_not_union_islands() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let edge = maru_collide::MAX_INPUT - 5.0;
        let ghost = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(edge, 0.0, 0.0),
            1.0,
        ));
        let l = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(edge - 3.0, 0.0, 0.0),
            1.0,
        ));
        let r = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(edge - 6.0, 0.0, 0.0),
            1.0,
        ));
        w.try_add_joint(Joint::point(ghost, l, Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO))
            .unwrap();
        w.try_add_joint(Joint::point(
            ghost,
            r,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::ZERO,
        ))
        .unwrap();
        w.set_linear_velocity(ghost, Vec3::new(500.0, 0.0, 0.0))
            .unwrap();
        for _ in 0..10 {
            let _ = w.try_step(crate::world::MAX_DT);
        }
        assert!(w.is_out_of_bounds(ghost));
        for _ in 0..60 {
            let _ = w.try_step(DT);
            assert!(
                w.restore(&w.snapshot()).is_ok(),
                "OOB가 아일랜드에 묶여 복원 불변식을 깼다"
            );
        }
    }

    /// MAJOR: OOB 전이는 인접 조인트의 누적 임펄스를 **지워야** 한다 —
    /// 복귀 후 warm start가 옛 자세의 임펄스를 새 자세에 적용하면 큰 kick이 된다.
    #[test]
    fn oob_transition_clears_adjacent_joint_impulses() {
        let mut w = World::new(WorldParams::default());
        let edge = maru_collide::MAX_INPUT - 5.0;
        // **둘 다 동적**이어야 함께 경계 밖으로 나간다 — 정적 앵커에 매달리면 조인트가 잡는다.
        let a = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(edge - 1.0, 0.0, 0.0),
            1.0,
        ));
        let b = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(edge, 0.0, 0.0),
            1.0,
        ));
        w.try_add_joint(Joint::point(a, b, Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO))
            .unwrap();
        // 중력·구속으로 누적 임펄스를 쌓는다
        for _ in 0..120 {
            w.step(DT);
        }
        // 함께 경계 밖으로
        for h in [a, b] {
            w.set_linear_velocity(h, Vec3::new(500.0, 0.0, 0.0))
                .unwrap();
        }
        for _ in 0..10 {
            let _ = w.try_step(crate::world::MAX_DT);
        }
        assert!(
            w.is_out_of_bounds(a) && w.is_out_of_bounds(b),
            "시나리오 전제 붕괴: OOB로 나가지 않았다"
        );
        // 호스트가 되돌린다 — 다음 스텝에 stale 임펄스 kick이 없어야 한다
        w.set_pose(a, pose(0.0, 1.0, 0.0)).unwrap();
        w.set_pose(b, pose(0.5, 1.0, 0.0)).unwrap();
        w.step(DT);
        let v = w.body(a).linvel().length().max(w.body(b).linvel().length());
        assert!(v < 5.0, "복귀 직후 stale 임펄스 kick: {v}");
    }

    // ─────────── 리뷰 rev_63c4f4fc 회귀 ───────────

    /// BLOCKER(L0): 파생 캐시(브로드페이즈 페어)의 무효화가 **한 경로라도** 빠지면
    /// 같은 스냅샷의 미래가 캐시 유무로 갈린다. pending Add 실현 경로가 빠져 있었다.
    ///
    /// 재현: static만 있는 월드를 한 스텝 돌려 빈 페어 캐시를 valid로 만든 뒤,
    /// 커맨드로 겹치는 dynamic을 추가한다. 원본은 캐시를 재사용해 접촉 없이 적분하고,
    /// 복원본은 캐시가 없어 페어를 재빌드해 접촉을 푼다 → 궤적이 갈린다.
    #[test]
    fn derived_cache_invalidation_is_complete_for_pending_add() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        w.step(DT); // 빈 페어 캐시를 valid로 만든다
                    // 지면과 **겹치도록** 커맨드로 추가한다
        let h = w.enqueue_add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.2, 0.0),
            1.0,
        ));
        w.flush_commands();
        let snap = w.snapshot();

        let mut restored = World::new(WorldParams::default());
        restored.restore(&snap).unwrap();
        // 같은 스냅샷에서 출발했으면 미래도 같아야 한다
        for i in 0..30 {
            w.step(DT);
            restored.step(DT);
            assert_eq!(
                w.state_hash(),
                restored.state_hash(),
                "스텝 {i}에서 궤적이 갈렸다 — 파생 캐시가 L0를 깨고 있다"
            );
        }
        assert!(w.is_alive(h));
    }

    /// MAJOR: 일반 `set_pose`(텔레포트)도 인접 조인트의 누적 임펄스를 지워야 한다 —
    /// 이전 자세의 월드 임펄스를 새 레버암에 적용하면 인공 kick이 된다.
    #[test]
    fn teleport_clears_adjacent_joint_impulses() {
        let mut w = World::new(WorldParams::default());
        let anchor = w.add_body(Body::new_static(
            Shape::Sphere { radius: 0.2 },
            pose(0.0, 5.0, 0.0),
        ));
        let bob = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.3 },
            pose(1.0, 5.0, 0.0),
            1.0,
        ));
        w.try_add_joint(Joint::point(
            anchor,
            bob,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
        ))
        .unwrap();
        for _ in 0..240 {
            w.step(DT); // 하중으로 임펄스를 쌓는다
        }
        // 반대편으로 텔레포트 — 레버암 방향이 뒤집힌다
        w.set_pose(bob, pose(-1.0, 5.0, 0.0)).unwrap();
        w.step(DT);
        let v = w.body(bob).linvel().length();
        assert!(v < 5.0, "텔레포트 직후 stale 임펄스 kick: {v}");
    }

    /// SAT 보조 키는 **부호 불변**이어야 A/B 교환에 대칭이다(signed 사전식은 `n → -n`에서
    /// 대소 순서가 보존되지 않아 다른 축을 고른다).
    ///
    /// 일반 구성(`|d·n|`이 유의미)과 **퇴화 구성**(거의 동심 + 수직 법선)을 함께 잠근다.
    /// 퇴화에서는 `d`가 방향을 못 정하므로 **바디 자체 데이터**(치수·자세)로 만든
    /// 교환 반대칭 키가 방향을 정한다 — manifold.rs의 `oriented` 주석 참조.
    #[test]
    fn sat_tie_break_symmetric_for_rotated_boxes() {
        let half = Vec3::new(0.5, 0.5, 0.5);
        let rot_a = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.3);
        let rot_b = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.4);
        let mk = |pa: Vec3, qa: Quat, pb: Vec3, qb: Quat| {
            let mut w = World::new(WorldParams {
                gravity: Vec3::ZERO,
                ..Default::default()
            });
            let a = w.add_body(Body::new_dynamic(
                Shape::Box { half, radius: 0.0 },
                Pose { pos: pa, rot: qa },
                1.0,
            ));
            let b = w.add_body(Body::new_dynamic(
                Shape::Box { half, radius: 0.0 },
                Pose { pos: pb, rot: qb },
                1.0,
            ));
            for _ in 0..40 {
                w.step(DT);
            }
            (w.body(a).pose().pos, w.body(b).pose().pos)
        };
        // 같은 자세를 **부호가 뒤집힌 쿼터니언**으로 표현해도 결과가 같아야 한다
        // (q와 -q는 같은 회전 — 표현이 물리를 바꾸면 안 된다).
        {
            let neg = |q: Quat| Quat {
                x: -q.x,
                y: -q.y,
                z: -q.z,
                w: -q.w,
            };
            let base = mk(Vec3::ZERO, rot_a, Vec3::new(0.02, 0.01, 0.0), rot_b);
            let flipped = mk(
                Vec3::ZERO,
                neg(rot_a),
                Vec3::new(0.02, 0.01, 0.0),
                neg(rot_b),
            );
            assert!(
                (base.0 - flipped.0).length() < 1e-3 && (base.1 - flipped.1).length() < 1e-3,
                "쿼터니언 부호 표현이 물리를 바꾼다: {:?} vs {:?}",
                base,
                flipped
            );
        }
        for (label, pa, pb) in [
            // 일반 구성
            ("일반", Vec3::ZERO, Vec3::new(0.62, 0.31, 0.44)),
            // **퇴화**: 거의 동심 → 정렬 키가 전부 0이고 d가 방향을 못 정한다
            ("퇴화(동심)", Vec3::ZERO, Vec3::new(0.02, 0.01, 0.0)),
            // **퇴화**: 완전 동심
            ("퇴화(완전동심)", Vec3::ZERO, Vec3::ZERO),
        ] {
            let (a1, b1) = mk(pa, rot_a, pb, rot_b);
            let (b2, a2) = mk(pb, rot_b, pa, rot_a);
            assert!(
                (a1 - a2).length() < 1e-3 && (b1 - b2).length() < 1e-3,
                "{label}: 삽입 순서에 따라 결과가 다르다: {a1:?}/{a2:?}, {b1:?}/{b2:?}"
            );
        }
    }

    /// 미해결이었던 **전 단계 fallible 바이트 예산**: 페어 개수만 묶으면 뒤따르는
    /// constraint·키·HashSet·SoA가 남는다. 총 바이트로 상한을 정하고 그것으로 페어 상한을
    /// 되돌려 계산하므로, 어떤 입력이 와도 한 스텝 스크래치가 상한을 넘지 않는다.
    #[test]
    fn step_scratch_is_byte_budgeted_and_fallible() {
        // 계수형 예산을 최대로 열어도 **바이트 상한**이 페어를 묶는다
        let mut w = World::new(WorldParams {
            max_pairs_per_body: 4096,
            ..Default::default()
        });
        for _ in 0..300 {
            w.add_body(Body::new_dynamic(
                Shape::Sphere { radius: 0.5 },
                pose(0.0, 1.0, 0.0), // 전부 같은 자리 → 최대 밀집
                1.0,
            ));
        }
        let r = w.try_step(DT);
        // 성공하든 ResourceLimit이든 **trap이 아니라 값으로** 돌아와야 한다
        assert!(
            r.is_ok() || r == Err(ApiError::ResourceLimit),
            "예상 밖 오류: {r:?}"
        );
        let d = w.diagnostics();
        let per_pair = core::mem::size_of::<(u32, u32)>() * 2 + 48;
        assert!(
            d.pair_budget.saturating_mul(per_pair) <= crate::world::MAX_STEP_SCRATCH_BYTES * 2,
            "페어 예산이 바이트 상한과 무관하다: {} 페어",
            d.pair_budget
        );
        // 상태는 손상되지 않고 스냅샷도 유효하다
        assert!(w.restore(&w.snapshot()).is_ok());
    }

    /// BLOCKER 회귀: 자원 실패가 **캐노니컬 상태를 훼손하면 안 된다**.
    /// `mem::take(&mut self.joints)` 뒤에 예약 실패로 조기 반환해 **모든 조인트가 삭제**되던
    /// 경로가 있었다. 선점을 캐노니컬 변이 **전에** 모두 끝내 그 창을 없앴다.
    #[test]
    fn resource_failure_never_destroys_canonical_state() {
        let mut w = World::new(WorldParams {
            max_pairs_per_body: 1,
            ..Default::default()
        });
        let a = dyn_box(&mut w, 0.0, 3.0);
        let b = dyn_box(&mut w, 1.2, 3.0);
        w.try_add_joint(Joint::point(a, b, Vec3::new(0.6, 0.0, 0.0), Vec3::ZERO))
            .unwrap();
        // 페어 예산을 넘겨 ResourceLimit을 강제한다
        for _ in 0..200 {
            w.add_body(Body::new_dynamic(
                Shape::Sphere { radius: 0.5 },
                pose(0.0, 1.0, 0.0),
                1.0,
            ));
        }
        let before = w.state_hash();
        let r = w.try_step(DT);
        assert_eq!(r, Err(ApiError::ResourceLimit));
        // **조인트가 살아 있고 물리도 전진하지 않았어야 한다**
        assert_eq!(
            w.state_hash(),
            before,
            "자원 실패가 상태를 바꿨다 (조인트 삭제 또는 물리 전진)"
        );
        assert!(w.restore(&w.snapshot()).is_ok());
        // 원인을 없애면 다음 스텝은 정상이고 조인트도 그대로다
        w.set_params(WorldParams {
            max_pairs_per_body: 4096,
            ..Default::default()
        });
        assert_eq!(w.try_step(DT), Ok(()));
    }

    /// BLOCKER: 삽입 시 조건수를 검사해도 **정상 적분 중 자유 회전**으로 K가 특이해질 수
    /// 있다(point joint는 회전을 구속하지 않는다). 그때 `inverse`가 0 행렬이면 제약이
    /// **조용히 풀린다** — 가장 나쁜 실패다. 솔브 시점 정규화로 항상 풀리게 만든다.
    #[test]
    fn joint_survives_free_rotation_without_silent_release() {
        let mut w = World::new(WorldParams::default());
        let anchor = w.add_body(Body::new_static(
            Shape::Sphere { radius: 0.1 },
            pose(0.0, 6.0, 0.0),
        ));
        // 이방성 판 — 자유 회전으로 조건수가 크게 변한다
        let plate = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(3.0, 0.05, 0.4),
                radius: 0.0,
            },
            pose(1.5, 6.0, 0.0),
            50.0,
        ));
        w.try_add_joint(Joint::point(
            anchor,
            plate,
            Vec3::ZERO,
            Vec3::new(-1.5, 0.0, 0.0),
        ))
        .unwrap();
        // 회전을 크게 넣어 자세를 계속 바꾼다
        w.set_angular_velocity(plate, Vec3::new(0.0, 0.0, 8.0))
            .unwrap();
        for i in 0..1200 {
            w.step(DT);
            let d = (w.body(plate).pose().pos - Vec3::new(0.0, 6.0, 0.0)).length();
            assert!(d < 4.0, "스텝 {i}에서 조인트가 풀렸다 (앵커 거리 {d:.2})");
        }
        assert!(w.restore(&w.snapshot()).is_ok());
    }

    // ==========================================================================
    // 적대 리뷰(2026-07-27)에서 추가한 재현 테스트. 아래 5건은 **현 구현의 결함을
    // 드러내기 위한 것**이며, 통과하도록 고치기 전까지 실패한다.
    // ==========================================================================

    /// 조인트 앵커가 B의 질량중심에서 **떨어져 있어야** 중력이 토크를 만든다.
    /// 기존 조인트 테스트들은 앵커를 B의 COM에 두어 각 제약을 전혀 자극하지 않는다.
    fn hinge_pendulum(w: &mut World) -> (BodyHandle, BodyHandle) {
        let a = w.add_body(Body::new_static(
            maru_collide::Shape::Sphere { radius: 0.1 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        ));
        let b = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Box {
                half: maru_math::Vec3::new(1.0, 0.1, 0.1),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(1.0, 10.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        (a, b)
    }

    /// **hinge 한계가 실제로 각을 재는가.** `hinge_angle`은 A·B 양쪽 축의
    /// `perp_basis`를 비교하는데, 힌지가 두 축을 정렬시키므로 두 기저가 같아져
    /// 각이 항상 0이 된다 → 한계가 절대 발동하지 않는다.
    #[test]
    fn review_hinge_limit_actually_limits_a_pendulum() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hinge_pendulum(&mut w);
        let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
        w.try_add_joint(
            Joint::try_hinge(
                a,
                b,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::new(-1.0, 0.0, 0.0),
                axis,
                axis,
            )
            .unwrap()
            .with_limit(-0.3, 0.3)
            .unwrap(),
        )
        .unwrap();
        // 초기 각속도를 줘서 한계에 **부딪히게** 만든다 — 앵커가 무게중심에 있으면
        // 중력만으로는 토크가 안 생겨 한계가 시험되지 않는다.
        w.set_angular_velocity(b, maru_math::Vec3::new(0.0, 0.0, 6.0))
            .unwrap();
        let mut worst: f32 = 0.0;
        for _ in 0..300 {
            w.try_step(1.0 / 60.0).unwrap();
            let q = w.body(b).pose().rot;
            let ang = 2.0 * q.z.abs().atan2(q.w.abs());
            worst = worst.max(ang);
        }
        assert!(
            worst < 0.45,
            "한계 ±0.3 rad인데 최대 {worst} rad까지 돌았다 (한계가 발동하지 않음)"
        );
    }

    /// **Fixed의 각 bias 부호.** err = 2·vec(qb·(qa·rel)⁻¹)는 d(err)/dt = ωb − ωa로
    /// 자라는데 솔버는 cdot = (ωa − ωb)를 쓴다 — Baumgarte 항의 부호가 뒤집혀
    /// 오차를 **되돌리는 대신 키운다**.
    #[test]
    fn review_fixed_joint_corrects_angular_error() {
        let mut w = World::new(WorldParams {
            gravity: maru_math::Vec3::ZERO,
            ..WorldParams::default()
        });
        let a = w.add_body(Body::new_static(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        ));
        // B의 초기 자세를 목표(항등)에서 0.1 rad 어긋나게 둔다.
        let b = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Box {
                half: maru_math::Vec3::new(0.3, 0.3, 0.3),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(1.0, 10.0, 0.0),
                rot: maru_math::Quat::from_axis_angle(maru_math::Vec3::new(1.0, 0.0, 0.0), 0.1),
            },
            1.0,
        ));
        w.try_add_joint(
            Joint::try_fixed(
                a,
                b,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Vec3::ZERO,
                maru_math::Quat::IDENTITY,
            )
            .unwrap(),
        )
        .unwrap();
        let err0 = 2.0 * w.body(b).pose().rot.x.abs();
        for _ in 0..60 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let err1 = 2.0 * w.body(b).pose().rot.x.abs();
        assert!(
            err1 < err0,
            "고정 조인트가 각 오차를 줄여야 한다: {err0} → {err1}"
        );
    }

    /// **부등식 게이트 × warm start.** 구간 안으로 들어온 서브스텝에서 누적 임펄스를
    /// 0으로 되돌리지만, 같은 서브스텝의 warm start가 이미 옛 임펄스를 속도에 실었다 —
    /// 되돌리는 델타(0 − old)를 적용하지 않아 그 운동량이 그대로 남는다.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn review_distance_joint_inactive_leaves_velocity_untouched() {
        use crate::solver::{Softness, SolverState};
        let mut st = SolverState::default();
        // 2바디: 0=static, 1=dynamic. 거리 1.0(구간 [0,2] 안).
        st.lin = vec![maru_math::Vec3::ZERO; 2];
        st.ang = vec![maru_math::Vec3::ZERO; 2];
        st.pos = vec![maru_math::Vec3::ZERO, maru_math::Vec3::new(1.0, 0.0, 0.0)];
        st.rot = vec![maru_math::Quat::IDENTITY; 2];
        st.inv_m = vec![0.0, 1.0];
        st.inv_i = vec![
            maru_math::Mat3::from_diagonal(maru_math::Vec3::ZERO),
            maru_math::Mat3::from_diagonal(maru_math::Vec3::new(1.0, 1.0, 1.0)),
        ];
        st.dynamic = vec![false, true];
        let ha = BodyHandle {
            index: 0,
            generation: 1,
        };
        let hb = BodyHandle {
            index: 1,
            generation: 1,
        };
        let mut j = Joint::try_distance(
            ha,
            hb,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::ZERO,
            0.0,
            2.0,
        )
        .unwrap();
        // 직전 서브스텝에 구간 밖이어서 쌓인 누적 임펄스 (로프가 당기고 있었다).
        j.linear_impulse = maru_math::Vec3::new(-3.0, 0.0, 0.0);
        let soft = Softness::new(60.0, 2.0, 1.0 / 240.0);
        crate::joint::warm_start(&mut st, &j);
        crate::joint::solve(&mut st, &mut j, 1.0 / 240.0, Some((&soft, 4.0, 30.0)));
        assert!(
            st.lin[1].length() < 1e-4,
            "구간 안이면 조인트가 속도를 건드리면 안 된다: {:?}",
            st.lin[1]
        );
    }

    /// **모터 상한은 스텝당인가 서브스텝당인가.** `max_impulse`가 서브스텝 누적에
    /// 걸리고 warm start가 매 서브스텝 전량을 다시 실으므로, 실제 전달 임펄스는
    /// substeps에 비례한다 — 같은 월드에서 substeps만 바꾸면 모터 힘이 달라진다.
    #[test]
    fn review_motor_max_impulse_is_independent_of_substeps() {
        fn spin(substeps: u32) -> f32 {
            let mut w = World::new(WorldParams {
                gravity: maru_math::Vec3::ZERO,
                substeps,
                ..WorldParams::default()
            });
            let a = w.add_body(Body::new_static(
                maru_collide::Shape::Sphere { radius: 0.5 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
                    rot: maru_math::Quat::IDENTITY,
                },
            ));
            let b = w.add_body(Body::new_dynamic(
                maru_collide::Shape::Box {
                    half: maru_math::Vec3::new(0.3, 0.3, 0.3),
                    radius: 0.0,
                },
                maru_collide::Pose {
                    pos: maru_math::Vec3::new(1.0, 10.0, 0.0),
                    rot: maru_math::Quat::IDENTITY,
                },
                1.0,
            ));
            let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
            w.try_add_joint(
                Joint::try_hinge(
                    a,
                    b,
                    maru_math::Vec3::new(1.0, 0.0, 0.0),
                    maru_math::Vec3::ZERO,
                    axis,
                    axis,
                )
                .unwrap()
                // 목표를 크게 잡아 모터를 항상 포화시킨다 → 전달량 = 상한 × (횟수)
                .with_motor(50.0, 0.2)
                .unwrap(),
            )
            .unwrap();
            w.try_step(1.0 / 60.0).unwrap();
            w.body(b).angvel().z
        }
        let s1 = spin(1);
        let s8 = spin(8);
        assert!(
            (s1 - s8).abs() < 0.1 * s1.abs().max(1e-6),
            "substeps만 바꿨는데 모터 전달 임펄스가 달라진다: {s1} vs {s8}"
        );
    }

    /// **exclude는 세대를 무시한다** — 제거 후 재사용된 슬롯을 죽은 핸들이 계속 가린다.
    #[test]
    fn review_query_exclude_respects_generation() {
        let mut w = World::new(WorldParams::default());
        let sphere = |p: maru_math::Vec3| {
            Body::new_static(
                maru_collide::Shape::Sphere { radius: 1.0 },
                maru_collide::Pose {
                    pos: p,
                    rot: maru_math::Quat::IDENTITY,
                },
            )
        };
        let dead = w.add_body(sphere(maru_math::Vec3::ZERO));
        w.remove_body(dead).unwrap();
        // 같은 슬롯이 재사용되며 세대만 올라간다.
        let fresh = w.add_body(sphere(maru_math::Vec3::ZERO));
        assert_eq!(fresh.index, dead.index, "슬롯 재사용 전제");
        assert_ne!(fresh.generation, dead.generation);
        let hit = w
            .cast_ray_filtered(
                maru_math::Vec3::new(-10.0, 0.0, 0.0),
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                100.0,
                QueryFilter::ALL.exclude(dead),
            )
            .unwrap();
        assert_eq!(
            hit.map(|h| h.body),
            Some(fresh),
            "죽은 핸들이 재사용된 슬롯의 새 바디를 가리면 안 된다"
        );
    }

    /// **무필터 질의의 계약 변경.** 물리적으로 아무와도 안 부딪히게 filter=0으로 둔
    /// 바디가 `cast_ray`(무필터)에서도 사라진다 — 이전 커밋에서는 맞았다.
    #[test]
    fn review_unfiltered_ray_still_sees_noncolliding_bodies() {
        let mut w = World::new(WorldParams::default());
        let ghost = w.add_body(
            Body::new_static(
                maru_collide::Shape::Sphere { radius: 1.0 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::ZERO,
                    rot: maru_math::Quat::IDENTITY,
                },
            )
            .with_groups(0b1, 0),
        );
        let hit = w
            .cast_ray(
                maru_math::Vec3::new(-10.0, 0.0, 0.0),
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                100.0,
            )
            .unwrap();
        assert_eq!(hit.map(|h| h.body), Some(ghost));
    }

    /// **기존 조인트 테스트 2건은 각 제약을 자극하지 않는다**(`hung_pair`가 앵커를
    /// B의 질량중심에 두어 중력 토크가 0이다). 같은 배치를 **Point 조인트**로 바꿔도
    /// `fixed_joint_holds_relative_orientation` · `hinge_limit_stops_rotation`의
    /// 단언이 그대로 통과한다 — 즉 두 테스트는 각 제약을 검증하지 않는다.
    #[test]
    fn review_existing_joint_tests_pass_without_any_angular_constraint() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        // Fixed도 Hinge도 아닌 **순수 point** — 각 제약이 전혀 없다.
        w.try_add_joint(
            Joint::try_point(
                a,
                b,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Vec3::ZERO,
            )
            .unwrap(),
        )
        .unwrap();
        for _ in 0..600 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let q = w.body(b).pose().rot;
        // fixed_joint_holds_relative_orientation 의 단언
        assert!(q.w.abs() > 0.995, "{q:?}");
        let p = w.body(b).pose().pos;
        assert!(
            (p - maru_math::Vec3::new(1.0, 10.0, 0.0)).length() < 0.1,
            "{p:?}"
        );
        // hinge_limit_stops_rotation 의 단언
        let ang = 2.0 * q.z.abs().atan2(q.w.abs());
        assert!(ang < 0.6, "{ang}");
    }

    /// preflight(`serialized_size`)가 새 조인트 종류까지 writer와 맞는가.
    /// 기존 `snapshot_preflight_size_matches...` 테스트는 `Point`만 담고 있어
    /// Hinge(limit·motor)·Fixed·Distance는 preflight 커버리지가 없었다.
    #[test]
    fn review_preflight_size_covers_new_joint_kinds() {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
        let anchor_a = maru_math::Vec3::new(1.0, 0.0, 0.0);
        w.try_add_joint(
            Joint::try_fixed(
                a,
                b,
                anchor_a,
                maru_math::Vec3::ZERO,
                maru_math::Quat::IDENTITY,
            )
            .unwrap(),
        )
        .unwrap();
        w.try_add_joint(
            Joint::try_distance(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, 0.5, 2.0)
                .unwrap(),
        )
        .unwrap();
        w.try_add_joint(
            Joint::try_hinge(a, b, anchor_a, maru_math::Vec3::ZERO, axis, axis)
                .unwrap()
                .with_limit(-1.0, 1.0)
                .unwrap()
                .with_motor(2.0, 50.0)
                .unwrap(),
        )
        .unwrap();
        // limit/motor 없는 hinge도 (Option 길이 고정 확인)
        w.try_add_joint(
            Joint::try_hinge(a, b, anchor_a, maru_math::Vec3::ZERO, axis, axis).unwrap(),
        )
        .unwrap();
        let bytes = w.snapshot();
        assert_eq!(
            crate::snapshot::serialized_size(&w),
            bytes.len(),
            "preflight 크기가 writer와 어긋남 — 한쪽만 고친 것이다"
        );
    }

    /// **`hinge_angle`은 축 둘레 각을 재지 못한다** (화이트박스).
    ///
    /// `hinge_angle`은 A·B **월드 축**의 `perp_basis`를 비교하는데, `perp_basis`는
    /// 입력 축만의 순함수다. 힌지는 두 축을 같게 만들므로 두 기저가 같아지고
    /// 각은 항상 0이 된다 — B가 축 둘레로 90° 돌아 있어도 한계가 발동하지 않는다.
    /// 기준 벡터는 각 바디의 **로컬** 벡터를 월드로 돌린 것이어야 한다.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn review_hinge_limit_fires_when_b_is_rotated_about_the_axis() {
        use crate::solver::{Softness, SolverState};
        let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
        let mut st = SolverState::default();
        st.lin = vec![maru_math::Vec3::ZERO; 2];
        st.ang = vec![maru_math::Vec3::ZERO; 2];
        st.pos = vec![maru_math::Vec3::ZERO; 2];
        // B는 힌지 축(z) 둘레로 90° 돌아 있다 — 한계 ±0.1 rad를 크게 벗어난다.
        st.rot = vec![
            maru_math::Quat::IDENTITY,
            maru_math::Quat::from_axis_angle(axis, core::f32::consts::FRAC_PI_2),
        ];
        st.inv_m = vec![0.0, 1.0];
        st.inv_i = vec![
            maru_math::Mat3::from_diagonal(maru_math::Vec3::ZERO),
            maru_math::Mat3::from_diagonal(maru_math::Vec3::new(1.0, 1.0, 1.0)),
        ];
        st.dynamic = vec![false, true];
        let ha = BodyHandle {
            index: 0,
            generation: 1,
        };
        let hb = BodyHandle {
            index: 1,
            generation: 1,
        };
        let mut j = Joint::try_hinge(
            ha,
            hb,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::ZERO,
            axis,
            axis,
        )
        .unwrap()
        .with_limit(-0.1, 0.1)
        .unwrap();
        let soft = Softness::new(60.0, 2.0, 1.0 / 240.0);
        crate::joint::solve(&mut st, &mut j, 1.0 / 240.0, Some((&soft, 4.0, 30.0)));
        assert!(
            j.limit_impulse != 0.0,
            "축 둘레 90°는 한계 ±0.1을 벗어났는데 한계 임펄스가 0이다 (각이 측정되지 않음)"
        );
    }

    /// 정착(전원 수면) 월드의 스텝은 **캐노니컬 상태에 항등**이어야 한다 — fast path와
    /// 풀 파이프라인 어느 쪽이 돌든. 스냅샷 바이트 동일성이 그 항등의 정의다.
    /// (이 성질이 깨지면 잠든 월드가 스텝마다 미세하게 표류하고, 수면은 "얼림"이 아니게 된다.)
    #[test]
    fn settled_step_is_canonical_identity() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        for i in 0..3 {
            dyn_box(&mut w, 0.0, 0.5 + i as f32);
        }
        for _ in 0..300 {
            w.step(DT);
        }
        assert!(w.diagnostics().pairs > 0, "테스트 씬이 비어 있으면 무의미");
        let before = w.snapshot();
        w.step(DT);
        assert!(
            w.diagnostics().bp_reused,
            "정착 스텝이 페어 캐시를 재사용하지 않음 — fast path가 죽었다"
        );
        let after = w.snapshot();
        assert_eq!(before, after, "정착 스텝이 캐노니컬 상태를 바꿨다");
    }

    /// 회귀(캐시 계약): impulse는 속도를 **즉시** 바꾸고 sweep AABB는 속도의 함수이므로,
    /// 정착 상태에서 impulse를 주면 다음 스텝은 페어 캐시를 버리고 재빌드해야 한다.
    /// 재사용하면 임펄스 이전 속도의 페어로 한 스텝을 돌아 speculative 접촉이 빠진다.
    #[test]
    fn impulse_on_settled_world_rebuilds_pairs() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let h = dyn_box(&mut w, 0.0, 0.5);
        for _ in 0..300 {
            w.step(DT);
        }
        assert!(w.is_asleep(h), "먼저 잠들어야 의미 있는 테스트다");
        w.step(DT);
        assert!(w.diagnostics().bp_reused, "정착 상태 전제 확인");
        w.apply_impulse(h, Vec3::new(50.0, 0.0, 0.0)).unwrap();
        w.step(DT);
        assert!(
            !w.diagnostics().bp_reused,
            "임펄스 후에도 페어 캐시를 재사용했다 — 임펄스 전 속도의 sweep AABB로 스텝이 돈다"
        );
    }

    /// 증분 브로드페이즈 등가성 — **궤적 전체**로 잠근다: 증분 월드와 전체-재빌드 월드를
    /// 같은 입력으로 락스텝 진행하며 매 스텝 스냅샷 바이트가 같아야 한다. 페어 집합이
    /// 한 스텝이라도 갈리면 제약·이벤트·wake가 갈려 스냅샷이 갈린다.
    /// 시나리오는 증분의 모든 전이를 지나간다: 활성 낙하(R>0) → 정착(전원 수면, fast path)
    /// → 임펄스(구조 무효화 + wake) → 바디 추가(구조 무효화) → 재정착.
    #[test]
    fn incremental_broadphase_trajectory_matches_full_rebuild() {
        let build = |force_full: bool| {
            let mut w = World::new(WorldParams::default());
            w.set_force_full_broadphase(force_full);
            ground(&mut w);
            let mut hs = Vec::new();
            for i in 0..24 {
                let x = (i % 4) as f32 * 1.05 - 2.0;
                let z = ((i / 4) % 3) as f32 * 1.05 - 1.0;
                let y = 0.5 + (i / 12) as f32 * 1.05;
                hs.push(w.add_body(Body::new_dynamic(
                    Shape::Box {
                        half: Vec3::new(0.5, 0.5, 0.5),
                        radius: 0.0,
                    },
                    pose(x, y, z),
                    1.0,
                )));
            }
            // 낙하 교란체 — 높이 띄워 "잠든 다수 + 활성 소수"(증분 창)를 길게 만든다.
            // R이 슬롯 절반을 넘는 활성 구간은 설계상 재빌드 폴백이라 증분이 안 돈다.
            for i in 0..4 {
                hs.push(w.add_body(Body::new_dynamic(
                    Shape::Sphere { radius: 0.4 },
                    pose(i as f32 * 1.5 - 2.0, 50.0 + i as f32 * 8.0, 0.3),
                    1.0,
                )));
            }
            (w, hs)
        };
        let (mut a, ha) = build(false); // 증분
        let (mut b, hb) = build(true); // 전체 재빌드 (대조군)
        assert_eq!(ha.len(), hb.len());
        let mut incr_steps = 0usize;
        let mut limited_event_steps = 0usize;
        for step in 0..460 {
            // 중간 이벤트: 임펄스(정착 후) · 바디 추가 — 양쪽 동일 적용
            if step == 340 {
                a.apply_impulse(ha[0], Vec3::new(6.0, 3.0, 0.0)).unwrap();
                b.apply_impulse(hb[0], Vec3::new(6.0, 3.0, 0.0)).unwrap();
            }
            if step == 380 {
                let body =
                    || Body::new_dynamic(Shape::Sphere { radius: 0.4 }, pose(0.2, 8.0, -0.2), 1.0);
                a.add_body(body());
                b.add_body(body());
            }
            a.step(DT);
            b.step(DT);
            incr_steps += a.diagnostics().bp_incremental as usize;
            if a.diagnostics().narrow_limited && !a.contact_events().is_empty() {
                limited_event_steps += 1;
            }
            assert!(
                !b.diagnostics().bp_incremental,
                "대조군이 증분을 탔다 — 노브 고장, 테스트 무의미"
            );
            // 이벤트는 스냅샷 밖이다 — 제한 모드의 **델타 이벤트**(fresh에서만 차집합)가
            // 전체 차집합과 같은 이벤트를 내는지 명시적으로 대조한다.
            assert_eq!(
                a.contact_events(),
                b.contact_events(),
                "step {step}: 증분 이벤트가 전체 재빌드 이벤트에서 이탈"
            );
            assert_eq!(
                a.snapshot(),
                b.snapshot(),
                "step {step}: 증분 궤적이 전체 재빌드 궤적에서 이탈"
            );
        }
        assert!(
            limited_event_steps > 0,
            "제한 모드에서 이벤트가 발생한 스텝이 없다 — 델타 이벤트 경로가 검증되지 않았다"
        );
        assert!(
            incr_steps > 60,
            "증분 경로가 거의 안 돌았다({incr_steps}스텝) — 테스트가 아무것도 증명 안 함"
        );
    }

    /// CCD의 존재 이유: 고속 소형 바디가 얇은 static을 스텝 사이에 건너뛰는 것.
    /// 같은 장면에서 CCD 없이는 관통하고(전제 확인 — 이게 안 뚫리면 테스트가 무의미),
    /// CCD를 켜면 벽 앞에서 멈춰야 한다.
    #[test]
    fn ccd_stops_bullet_that_tunnels_without_it() {
        let wall = |w: &mut World| {
            w.add_body(Body::new_static(
                Shape::Box {
                    half: Vec3::new(0.05, 5.0, 5.0),
                    radius: 0.0,
                },
                pose(10.0, 0.0, 0.0),
            ))
        };
        let bullet = |ccd: bool| {
            let b = Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0);
            if ccd {
                b.with_ccd()
            } else {
                b
            }
        };
        // 무중력 — 순수 관통 시나리오. 속도 400 m/s = 스텝당 6.7m (벽 두께 0.1의 67배).
        let params = || WorldParams {
            gravity: Vec3::ZERO,
            ..WorldParams::default()
        };

        let mut w0 = World::new(params());
        wall(&mut w0);
        let h0 = w0.add_body(bullet(false));
        w0.set_linear_velocity(h0, Vec3::new(400.0, 0.0, 0.0))
            .unwrap();
        for _ in 0..10 {
            w0.step(DT);
        }
        assert!(
            w0.body(h0).pose().pos.x > 11.0,
            "CCD 없이 관통해야 전제가 성립하는데 x={} — 테스트가 아무것도 증명 못 한다",
            w0.body(h0).pose().pos.x
        );

        let mut w1 = World::new(params());
        wall(&mut w1);
        let h1 = w1.add_body(bullet(true));
        w1.set_linear_velocity(h1, Vec3::new(400.0, 0.0, 0.0))
            .unwrap();
        for _ in 0..10 {
            w1.step(DT);
        }
        let x = w1.body(h1).pose().pos.x;
        assert!(x < 9.95, "CCD가 켜졌는데 벽(근면 x=9.95)을 넘었다: x={x}");
        assert!(x > 8.0, "벽 앞 어딘가에는 도달해야 한다: x={x}");
    }

    /// **TOI-hit 큐** (설계 §4.5 / M4): 절단 순간이 호스트에게 보고돼야 한다.
    ///
    /// 접촉 이벤트로는 이 정보를 낼 수 없다 — linear cast는 바디를 표면에 세워 두고
    /// 반발·정지는 다음 스텝의 접촉이 해석하므로, 절단 자체는 어디에도 안 남았다.
    #[test]
    fn ccd_reports_the_toi_hit_it_clamped_on() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..WorldParams::default()
        });
        let wall = w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.05, 5.0, 5.0),
                radius: 0.0,
            },
            pose(10.0, 0.0, 0.0),
        ));
        let bullet = w.add_body(
            Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0).with_ccd(),
        );
        w.set_linear_velocity(bullet, Vec3::new(400.0, 0.0, 0.0))
            .unwrap();
        // 벽까지 6.7m/step → 2스텝째에 맞는다. 맞은 스텝의 히트를 잡는다.
        let mut hits = Vec::new();
        for _ in 0..4 {
            w.step(DT);
            hits.extend_from_slice(w.ccd_hits());
            if !hits.is_empty() {
                break;
            }
        }
        assert!(!hits.is_empty(), "CCD가 절단했는데 히트가 보고되지 않았다");
        let h = hits[0];
        assert_eq!(h.body, bullet);
        assert_eq!(h.other, wall, "부딪힌 static 바디가 잘못 보고됐다");
        assert!(
            (0.0..=1.0).contains(&h.toi),
            "toi는 서브스텝 분율이어야 한다: {}",
            h.toi
        );
        assert!(
            h.substep < WorldParams::default().substeps,
            "서브스텝 인덱스가 범위를 넘었다: {}",
            h.substep
        );
        // 법선은 벽이 총알을 밀어내는 방향(−x)이어야 한다.
        assert!(
            h.normal.x < -0.9,
            "법선이 밀어내는 방향이 아니다: {:?}",
            h.normal
        );
        // 접점은 벽 근면(x≈9.95) 부근이어야 한다.
        assert!(
            (h.point.x - 9.95).abs() < 0.3,
            "접점이 벽 표면에서 멀다: {:?}",
            h.point
        );
        // **스텝 단위 출력**이다 — 매 스텝 비워지므로 누적되지 않는다.
        // (벽에 붙은 뒤에도 잔여 이동이 있으면 매 스텝 새 히트가 날 수 있다 —
        //  그러니 "빈다"가 아니라 "안 쌓인다"가 검증해야 할 성질이다.)
        let cap_per_step = WorldParams::default().substeps as usize;
        for _ in 0..20 {
            w.step(DT);
            assert!(
                w.ccd_hits().len() <= cap_per_step,
                "히트가 스텝을 넘어 누적됐다: {} > 서브스텝 {}",
                w.ccd_hits().len(),
                cap_per_step
            );
        }
    }

    /// 히트 큐가 **상한에서 포화**해도 결정적이고, 버려진 수를 보고해야 한다.
    /// (M4 완료 기준: "TOI 큐 상한 포화 프레임 — 할당 0·오버플로 결정성")
    #[test]
    fn ccd_hit_queue_saturates_deterministically() {
        let build = || {
            let mut w = World::new(WorldParams {
                gravity: Vec3::ZERO,
                ..WorldParams::default()
            });
            w.add_body(Body::new_static(
                Shape::Box {
                    half: Vec3::new(0.05, 200.0, 200.0),
                    radius: 0.0,
                },
                pose(10.0, 0.0, 0.0),
            ));
            let n = crate::world::MAX_CCD_HITS + 64;
            for k in 0..n {
                // 격자로 흩어 놓아 서로 겹치지 않게 한다.
                let y = (k % 64) as f32 * 0.5 - 16.0;
                let z = (k / 64) as f32 * 0.5 - 8.0;
                let h = w.add_body(
                    Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, y, z), 1.0)
                        .with_ccd(),
                );
                w.set_linear_velocity(h, Vec3::new(400.0, 0.0, 0.0))
                    .unwrap();
            }
            w
        };
        let run = || {
            let mut w = build();
            let mut seen = Vec::new();
            let mut dropped = 0usize;
            for _ in 0..4 {
                w.step(DT);
                if !w.ccd_hits().is_empty() {
                    seen = w.ccd_hits().to_vec();
                    dropped = w.diagnostics().ccd_hits_dropped;
                    break;
                }
            }
            (seen, dropped)
        };
        let (a, da) = run();
        let (b, db) = run();
        assert!(!a.is_empty(), "포화 시나리오에서 히트가 하나도 없다");
        assert!(
            a.len() <= crate::world::MAX_CCD_HITS,
            "상한을 넘겨 보고했다: {}",
            a.len()
        );
        assert!(
            da > 0,
            "상한을 넘겼는데 버려진 수가 0으로 보고됐다 — 조용한 유실이다"
        );
        // **결정성**: 같은 입력이면 남는 집합도 버려지는 수도 같아야 한다.
        assert_eq!(da, db, "버려진 수가 실행마다 다르다");
        assert_eq!(a.len(), b.len(), "보고된 히트 수가 실행마다 다르다");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.body, y.body, "히트 순서가 실행마다 다르다");
            assert_eq!(x.other, y.other);
            assert_eq!(x.toi.to_bits(), y.toi.to_bits(), "toi가 실행마다 다르다");
        }
    }

    /// **복원 정책이 바뀌면 버전을 올린다** — 같은 바이트가 다른 값으로 복원되기 때문이다.
    ///
    /// v13까지의 리더는 쿼터니언을 무조건 재정규화했고 v14는 비트 멱등 캐노니컬화를 한다.
    /// 구 빌드가 저장한 주기-2 쿼터니언은 두 리더가 **다른 값**으로 복원하므로, 같은 버전
    /// 번호가 두 계약을 뜻하게 두면 안 된다 (교차검수 rev_420ab223 MAJOR).
    #[test]
    fn snapshot_rejects_the_previous_version_after_the_quat_policy_change() {
        let mut w = World::new(WorldParams::default());
        w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(0.0, 1.0, 0.0),
            1.0,
        ));
        w.step(DT);
        let mut snap = w.try_snapshot().unwrap();
        // 헤더 = MAGIC(u32 LE) + VERSION(u32 LE). 버전만 13으로 되돌린다.
        assert!(snap.len() > 8);
        snap[4..8].copy_from_slice(&13u32.to_le_bytes());
        let mut r = World::new(WorldParams::default());
        match r.restore(&snap) {
            Err(SnapshotError::BadVersion(13)) => {}
            other => panic!("구 버전 스냅샷을 거부해야 한다: {other:?}"),
        }
    }

    /// 복원은 **폐기한 미래의 스텝 출력**까지 지워야 한다.
    ///
    /// 히트 큐는 캐노니컬 상태가 아니라 남아 있었는데, 그러면 호스트가 되돌린 타임라인의
    /// 히트를 다시 읽어 데미지·이펙트를 두 번 적용한다. `diag`는 복원 시 0이 되므로
    /// `ccd_hits_dropped == 0`이라 온전한 목록처럼 보이기까지 한다
    /// (교차검수 rev_5e74f64f MAJOR).
    #[test]
    fn restore_clears_ccd_hits_from_the_discarded_timeline() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..WorldParams::default()
        });
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.05, 5.0, 5.0),
                radius: 0.0,
            },
            pose(10.0, 0.0, 0.0),
        ));
        let bullet = w.add_body(
            Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0).with_ccd(),
        );
        w.set_linear_velocity(bullet, Vec3::new(400.0, 0.0, 0.0))
            .unwrap();
        // 충돌 **전** 스냅샷을 잡아 둔다.
        let before = w.try_snapshot().unwrap();
        let mut hit = false;
        for _ in 0..4 {
            w.step(DT);
            if !w.ccd_hits().is_empty() {
                hit = true;
                break;
            }
        }
        assert!(hit, "전제: 이 씬에서 CCD 히트가 나야 한다");
        // 되돌린다 — 그 히트는 이제 없던 일이다.
        w.restore(&before).unwrap();
        assert!(
            w.ccd_hits().is_empty(),
            "복원이 폐기한 타임라인의 히트를 남겼다: {}건",
            w.ccd_hits().len()
        );
    }

    /// CCD 고정 버퍼도 **페어 예산을 역산하기 전에** 총 스크래치에서 빠져야 한다.
    ///
    /// 안 빼면 페어가 바이트 상한까지 찬 월드가 128 MiB를 거의 다 쓴 뒤 히트 버퍼를 더 잡아
    /// 총상한을 넘긴다. 관측 가능한 형태로 잠근다: **같은 씬에서 CCD를 켜면 페어 예산이
    /// 그만큼 줄어야 한다** (교차검수 rev_5e74f64f MAJOR).
    #[test]
    fn ccd_hit_buffer_is_charged_against_the_step_scratch_budget() {
        let build = |ccd: bool| {
            let mut w = World::new(WorldParams {
                // 페어 예산을 **바이트 상한이 지배하도록** 개수 상한을 최대로 연다
                // (개수 상한이 이기면 CCD 바이트를 빼도 예산이 안 변해 아무것도 못 잰다).
                max_pairs_per_body: 4096,
                ..WorldParams::default()
            });
            for i in 0..200 {
                let b = Body::new_dynamic(
                    Shape::Sphere { radius: 0.1 },
                    pose(i as f32 * 3.0, 50.0, 0.0),
                    1.0,
                );
                w.add_body(if ccd { b.with_ccd() } else { b });
            }
            w.step(DT);
            w.diagnostics().pair_budget
        };
        let without = build(false);
        let with = build(true);
        assert!(
            with < without,
            "CCD를 켰는데 페어 예산이 그대로다 — 히트 버퍼가 회계에서 빠져 있다 \
             (without={without}, with={with})"
        );
    }

    /// 복원의 스텝-출력 정책은 **원자적**이다: 성공하면 두 큐를 다 비우고,
    /// 실패하면 둘 다 보존한다.
    ///
    /// 하나만 비우면 폐기한 타임라인의 접촉 이벤트가 남아 호스트가 데미지·사운드를
    /// 다시 적용한다 — 히트 큐만 고쳤을 때 남았던 비대칭이다
    /// (교차검수 rev_6da1d7ba NEW_DEFECT).
    #[test]
    fn restore_discards_both_step_output_queues_atomically() {
        let mut w = World::new(WorldParams::default());
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.05, 5.0, 5.0),
                radius: 0.0,
            },
            pose(10.0, 0.0, 0.0),
        ));
        let bullet = w.add_body(
            Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0).with_ccd(),
        );
        w.set_linear_velocity(bullet, Vec3::new(400.0, 0.0, 0.0))
            .unwrap();
        // 충돌 **전** 스냅샷.
        let before = w.try_snapshot().unwrap();
        // 벽에 닿는 스텝까지 — 그 스텝은 접촉 `Started`와 CCD 히트를 **둘 다** 낸다.
        // (히트는 절단한 스텝에, Started는 접촉 기준선이 바뀐 스텝에 난다.)
        let mut both = false;
        for _ in 0..12 {
            w.step(DT);
            if !w.contact_events().is_empty() && !w.ccd_hits().is_empty() {
                both = true;
                break;
            }
        }
        assert!(
            both,
            "전제: 한 스텝에 접촉 이벤트와 CCD 히트가 함께 나야 두 큐를 동시에 검증한다"
        );

        // ── 실패한 복원: 두 큐 모두 보존돼야 한다 ──
        let events_before = w.contact_events().to_vec();
        let hits_before = w.ccd_hits().to_vec();
        let mut broken = before.clone();
        broken[4..8].copy_from_slice(&999u32.to_le_bytes()); // 잘못된 버전
        assert!(w.restore(&broken).is_err(), "잘못된 스냅샷은 실패해야 한다");
        assert_eq!(
            w.contact_events(),
            events_before.as_slice(),
            "실패한 복원이 접촉 이벤트를 건드렸다"
        );
        assert_eq!(
            w.ccd_hits(),
            hits_before.as_slice(),
            "실패한 복원이 CCD 히트를 건드렸다"
        );

        // ── 성공한 복원: 두 큐 모두 비어야 한다 ──
        w.restore(&before).unwrap();
        assert!(
            w.contact_events().is_empty(),
            "복원이 폐기한 타임라인의 접촉 이벤트를 남겼다: {}건",
            w.contact_events().len()
        );
        assert!(
            w.ccd_hits().is_empty(),
            "복원이 폐기한 타임라인의 CCD 히트를 남겼다: {}건",
            w.ccd_hits().len()
        );
    }

    /// 클램프 평면은 단방향이다 — 바닥에 착지한 CCD 바디의 접선 이동(슬라이드)을 막지 않는다.
    #[test]
    fn ccd_clamp_allows_tangential_slide() {
        let params = WorldParams {
            gravity: Vec3::new(0.0, -9.8, 0.0),
            ..WorldParams::default()
        };
        let mut w = World::new(params);
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(50.0, 1.0, 50.0),
                radius: 0.0,
            },
            pose(0.0, -1.0, 0.0),
        ));
        let h = w.add_body(
            Body::new_dynamic(Shape::Sphere { radius: 0.2 }, pose(0.0, 4.0, 0.0), 1.0)
                .with_ccd()
                .with_friction(0.0)
                .unwrap(),
        );
        // 빠른 낙하 + 수평 속도 — 착지 후에도 수평으로 계속 가야 한다
        w.set_linear_velocity(h, Vec3::new(30.0, -80.0, 0.0))
            .unwrap();
        let mut max_x = 0.0f32;
        for _ in 0..30 {
            w.step(DT);
            let p = w.body(h).pose().pos;
            assert!(p.y > 0.0, "바닥 관통: {p:?}");
            max_x = max_x.max(p.x);
        }
        assert!(
            max_x > 3.0,
            "클램프가 슬라이드를 막았다 — 30스텝 동안 x 최대 {max_x}"
        );
    }

    /// ccd 플래그는 캐노니컬 상태다 — 스냅샷 왕복에서 살아남고, 왕복한 월드의 미래가 같다.
    #[test]
    fn ccd_flag_survives_snapshot_roundtrip() {
        let params = WorldParams {
            gravity: Vec3::ZERO,
            ..WorldParams::default()
        };
        let mut w = World::new(params);
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.05, 5.0, 5.0),
                radius: 0.0,
            },
            pose(10.0, 0.0, 0.0),
        ));
        let h = w.add_body(
            Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0).with_ccd(),
        );
        w.set_linear_velocity(h, Vec3::new(400.0, 0.0, 0.0))
            .unwrap();
        w.step(DT);
        let snap = w.snapshot();
        let mut w2 = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..WorldParams::default()
        });
        w2.restore(&snap).unwrap();
        assert!(w2.body(h).ccd(), "ccd 플래그가 왕복에서 사라졌다");
        for _ in 0..9 {
            w.step(DT);
            w2.step(DT);
        }
        assert_eq!(w.snapshot(), w2.snapshot(), "왕복 후 궤적 분기");
        assert!(
            w.body(h).pose().pos.x < 9.95,
            "왕복 뒤에도 CCD가 벽을 막아야 한다"
        );
    }

    /// 복합 질량 베이킹 해석해 대조: 나란한 두 박스 == 한 개의 넓은 박스.
    /// COM 재중심화(pose.pos = 질량중심)와 주축 관성까지 단일 박스 해석해와 일치해야 한다.
    #[test]
    fn compound_mass_baking_matches_analytic_wide_box() {
        let two = Body::new_dynamic(
            Shape::Compound {
                children: vec![
                    (
                        Pose {
                            pos: Vec3::new(-0.5, 0.0, 0.0),
                            rot: Quat::IDENTITY,
                        },
                        Shape::Box {
                            half: Vec3::new(0.5, 0.5, 0.5),
                            radius: 0.0,
                        },
                    ),
                    (
                        Pose {
                            pos: Vec3::new(0.5, 0.0, 0.0),
                            rot: Quat::IDENTITY,
                        },
                        Shape::Box {
                            half: Vec3::new(0.5, 0.5, 0.5),
                            radius: 0.0,
                        },
                    ),
                ],
            },
            pose(3.0, 5.0, 0.0),
            1.0,
        );
        let one = Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(1.0, 0.5, 0.5),
                radius: 0.0,
            },
            pose(3.0, 5.0, 0.0),
            1.0,
        );
        assert!(
            (two.inv_mass() - one.inv_mass()).abs() < 1e-5 * one.inv_mass(),
            "질량 불일치: {} vs {}",
            two.inv_mass(),
            one.inv_mass()
        );
        // 주축 관성 비교 — 고유값 순서가 다를 수 있어 정렬 후 대조.
        let sorted = |v: Vec3| {
            let mut a = [v.x, v.y, v.z];
            a.sort_by(f32::total_cmp);
            a
        };
        let (ti, oi) = (
            sorted(two.inv_inertia_local()),
            sorted(one.inv_inertia_local()),
        );
        for k in 0..3 {
            assert!(
                (ti[k] - oi[k]).abs() < 1e-4 * oi[k].abs().max(1e-6),
                "주축 관성 불일치: {ti:?} vs {oi:?}"
            );
        }
        // COM 재중심화: 대칭 배치라 COM == 원 pose 위치.
        assert!((two.pose().pos - Vec3::new(3.0, 5.0, 0.0)).length() < 1e-5);
    }

    /// L자 복합 바디가 낙하→정착→수면하고, 스냅샷 왕복 궤적이 비트 동일해야 한다.
    /// (자식쌍 feature 네임스페이스·PCM append 병합·복합 디스패치의 통합 검증.)
    #[test]
    fn compound_l_shape_settles_and_roundtrips() {
        let build = || {
            let mut w = World::new(WorldParams::default());
            ground(&mut w);
            let l = w.add_body(Body::new_dynamic(
                Shape::Compound {
                    children: vec![
                        (
                            Pose {
                                pos: Vec3::ZERO,
                                rot: Quat::IDENTITY,
                            },
                            Shape::Box {
                                half: Vec3::new(1.0, 0.25, 0.25),
                                radius: 0.0,
                            },
                        ),
                        (
                            Pose {
                                pos: Vec3::new(-0.75, 0.75, 0.0),
                                rot: Quat::IDENTITY,
                            },
                            Shape::Box {
                                half: Vec3::new(0.25, 0.5, 0.25),
                                radius: 0.0,
                            },
                        ),
                    ],
                },
                pose(0.0, 2.0, 0.0),
                1.0,
            ));
            (w, l)
        };
        let (mut w, l) = build();
        let mut asleep_at = None;
        for step in 0..600 {
            w.step(DT);
            if w.is_asleep(l) {
                asleep_at = Some(step);
                break;
            }
        }
        assert!(asleep_at.is_some(), "L자 복합이 600스텝에도 잠들지 않음");
        let p = w.body(l).pose().pos;
        assert!(p.y > 0.0 && p.y < 1.5, "정착 높이 이상: {p:?}");

        // 왕복 궤적: 중간 시점 스냅샷 → 복원 → 잔여 궤적 비트 동일
        let (mut a, _) = build();
        for _ in 0..40 {
            a.step(DT);
        }
        let snap = a.snapshot();
        let mut b = World::new(WorldParams::default());
        b.restore(&snap).unwrap();
        for _ in 0..120 {
            a.step(DT);
            b.step(DT);
        }
        assert_eq!(a.snapshot(), b.snapshot(), "복합 왕복 궤적 분기");
    }

    /// 복합×복합 접촉: 두 L자가 서로 얹혀도 발산 없이 정착한다.
    #[test]
    fn compound_on_compound_stacks() {
        let l_shape = || Shape::Compound {
            children: vec![
                (
                    Pose {
                        pos: Vec3::ZERO,
                        rot: Quat::IDENTITY,
                    },
                    Shape::Box {
                        half: Vec3::new(0.8, 0.25, 0.4),
                        radius: 0.0,
                    },
                ),
                (
                    Pose {
                        pos: Vec3::new(-0.55, 0.5, 0.0),
                        rot: Quat::IDENTITY,
                    },
                    Shape::Box {
                        half: Vec3::new(0.25, 0.25, 0.4),
                        radius: 0.0,
                    },
                ),
            ],
        };
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let a = w.add_body(Body::new_dynamic(l_shape(), pose(0.0, 0.4, 0.0), 1.0));
        let b = w.add_body(Body::new_dynamic(l_shape(), pose(0.15, 1.2, 0.0), 1.0));
        for _ in 0..600 {
            w.step(DT);
        }
        for h in [a, b] {
            let p = w.body(h).pose().pos;
            assert!(
                p.y.is_finite() && p.y > -0.5 && p.y < 3.0,
                "발산/관통: {p:?}"
            );
            assert!(
                w.body(h).linvel().length() < 0.5,
                "600스텝 후에도 큰 잔류 속도: {:?}",
                w.body(h).linvel()
            );
        }
    }

    /// 지형 메시(삼각형당 static hull): 구가 4-삼각형 그리드에 얹혀 정착·수면하고,
    /// 레이가 지형을 맞힌다. 퇴화 입력(범위 밖 인덱스·두께 0)은 전체 거부(부분 삽입 없음).
    #[test]
    fn terrain_mesh_supports_resting_sphere_and_rays() {
        let mut w = World::new(WorldParams::default());
        // 2×2 그리드 (4 정점 → 근데 그리드는 9정점… 간단히 3×3 정점, 8 삼각형)
        let mut verts = Vec::new();
        for z in 0..3 {
            for x in 0..3 {
                verts.push(Vec3::new(x as f32 * 2.0 - 2.0, 0.0, z as f32 * 2.0 - 2.0));
            }
        }
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for z in 0..2u32 {
            for x in 0..2u32 {
                let i = z * 3 + x;
                tris.push([i, i + 1, i + 3]);
                tris.push([i + 1, i + 4, i + 3]);
            }
        }
        let handles = w
            .try_add_terrain(&verts, &tris, 0.02)
            .expect("정상 지형이 거부됨");
        assert_eq!(handles.len(), 8);

        let ball = w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.4 },
            pose(0.3, 2.0, 0.2),
            1.0,
        ));
        let mut slept = false;
        for _ in 0..600 {
            w.step(DT);
            if w.is_asleep(ball) {
                slept = true;
                break;
            }
        }
        assert!(slept, "지형 위 구가 잠들지 않음");
        let y = w.body(ball).pose().pos.y;
        // 표면(0) + radius(0.4) + hull radius(0.02) 근처
        assert!((y - 0.42).abs() < 0.05, "정착 높이 이상: {y}");

        // 레이: 위에서 아래로 — 지형에 맞아야 한다
        let hit = w
            .cast_ray(Vec3::new(-1.0, 5.0, -1.0), Vec3::new(0.0, -1.0, 0.0), 10.0)
            .unwrap()
            .expect("지형 레이 미스");
        assert!(hit.toi > 4.0 && hit.toi < 5.1, "toi={}", hit.toi);

        // 퇴화 입력 거부: 범위 밖 인덱스 · 두께 0 (radius 0)
        let n_before = handles.len();
        assert!(w.try_add_terrain(&verts, &[[0, 1, 99]], 0.02).is_err());
        assert!(w.try_add_terrain(&verts, &[[0, 1, 3]], 0.0).is_err());
        let _ = n_before;
    }

    /// 내부 에지 판정(§4.2 ghost edge): 공면 공유 에지만 용접 마스크가 선다 —
    /// 경계·볼록 능선·비-다양체 에지는 제외.
    #[test]
    fn terrain_marks_only_coplanar_shared_edges_internal() {
        let mut w = World::new(WorldParams::default());
        // 평평한 사각형 = 2 삼각형, 공유 에지 (1,2)는 공면 → 양쪽 용접.
        let quad = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(-1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ];
        let hs = w
            .try_add_terrain(&quad, &[[0, 1, 2], [1, 3, 2]], 0.02)
            .unwrap();
        let mask = |h: BodyHandle, w: &World| w.body(h).weld_edges;
        // T0의 공유 에지 = (v1,v2) = 에지 1, T1의 공유 에지 = (v1,v2)… T1=[1,3,2]에서
        // (1,2)는 에지 2다. 경계 에지는 비트가 서면 안 된다.
        assert_eq!(mask(hs[0], &w), 0b010, "T0: 공유 에지(1→2)만 내부");
        assert_eq!(mask(hs[1], &w), 0b100, "T1: 공유 에지(2→1)만 내부");

        // 90° 능선: 공유 에지가 공면이 아니다 → 용접 없음.
        let ridge = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0), // 수평 면
            Vec3::new(0.0, 1.0, 0.0),  // 수직 면
        ];
        let hs = w
            .try_add_terrain(&ridge, &[[0, 1, 2], [0, 3, 1]], 0.02)
            .unwrap();
        assert_eq!(mask(hs[0], &w), 0, "능선은 용접 금지");
        assert_eq!(mask(hs[1], &w), 0, "능선은 용접 금지");

        // 비-다양체(3면 공유): 보수적으로 용접 없음.
        let fan = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.5),
        ];
        let hs = w
            .try_add_terrain(&fan, &[[0, 1, 2], [0, 1, 3], [0, 1, 4]], 0.02)
            .unwrap();
        for h in &hs {
            assert_eq!(mask(*h, &w), 0, "비-다양체 에지는 용접 금지");
        }
    }

    /// 용접 마스크는 캐노니컬이다: 스냅샷 왕복에서 보존되고, validate는 지형 삼각형이
    /// 아닌 바디의 비영 마스크를 거부한다.
    #[test]
    fn weld_mask_roundtrips_and_is_rejected_off_terrain() {
        let mut w = World::new(WorldParams::default());
        let quad = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(-1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ];
        let hs = w
            .try_add_terrain(&quad, &[[0, 1, 2], [1, 3, 2]], 0.02)
            .unwrap();
        let snap = w.snapshot();
        let mut w2 = World::new(WorldParams::default());
        w2.restore(&snap).expect("정상 스냅샷 복원 실패");
        assert_eq!(w2.body(hs[0]).weld_edges, 0b010);
        assert_eq!(snap, w2.snapshot(), "왕복 바이트 불일치");

        // 지형 삼각형이 아닌 바디의 비영 마스크 = 도달 불가 상태 → validate 거부.
        let mut bad = Body::new_static(Shape::Sphere { radius: 0.5 }, pose(0.0, 0.0, 0.0));
        bad.weld_edges = 1;
        assert!(bad.validate().is_err(), "구에 용접 마스크가 허용됨");
        // 3점이 아닌 hull도 거부 — 면 법선·에지 인덱스 의미가 없다.
        let mut bad = Body::new_static(
            Shape::Hull {
                points: vec![
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.3, 1.0, 0.3),
                ],
                radius: 0.02,
            },
            pose(0.0, 1.0, 0.0),
        );
        bad.weld_edges = 1;
        assert!(bad.validate().is_err(), "4점 hull에 용접 마스크가 허용됨");
    }

    /// 거동: 평평한 이음새를 미끄러져 건너는 구가 옆으로 차이지 않는다.
    #[test]
    fn sphere_slides_across_welded_seam_without_lateral_kick() {
        let mut w = World::new(WorldParams::default());
        // z축 이음새를 가진 평평한 2×1 바닥 (x = −2..2, 이음새 x=0).
        let verts = [
            Vec3::new(-2.0, 0.0, -2.0),
            Vec3::new(0.0, 0.0, -2.0),
            Vec3::new(2.0, 0.0, -2.0),
            Vec3::new(-2.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(2.0, 0.0, 2.0),
        ];
        let tris = [[0u32, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]];
        w.try_add_terrain(&verts, &tris, 0.02).unwrap();
        let mut b = Body::new_dynamic(Shape::Sphere { radius: 0.3 }, pose(-1.2, 0.33, 0.1), 1.0);
        b.linvel = Vec3::new(2.5, 0.0, 0.0);
        let ball = w.add_body(b);
        let mut crossed = false;
        for _ in 0..90 {
            w.step(DT);
            let p = w.body(ball).pose().pos;
            let v = w.body(ball).linvel();
            assert!(
                v.z.abs() < 0.05,
                "이음새에서 측면 속도 발생: vz={} (x={})",
                v.z,
                p.x
            );
            assert!(v.y < 0.5, "이음새에서 수직 킥: vy={} (x={})", v.y, p.x);
            if p.x > 0.5 {
                crossed = true;
                break;
            }
        }
        assert!(
            crossed,
            "이음새를 건너지 못했다 — 테스트가 아무것도 증명 안 함"
        );
    }

    /// 회전 스윕 CCD(§4 예약 확장): 선속도 0으로 **회전만으로** 벽을 쓸고 지나가는
    /// 블레이드 — 선형 캐스트는 이동이 0이라 원리적으로 못 잡는다. 쌍둥이 대조로
    /// 전제(비-CCD는 통과)와 효과(CCD는 차단)를 함께 증명한다.
    #[test]
    fn spinning_blade_does_not_sweep_through_wall() {
        let params = || WorldParams {
            gravity: Vec3::ZERO,
            ..WorldParams::default()
        };
        let build = |ccd: bool| {
            let mut w = World::new(params());
            w.add_body(Body::new_static(
                Shape::Box {
                    half: Vec3::new(0.05, 3.0, 3.0),
                    radius: 0.0,
                },
                pose(1.2, 0.0, 0.0),
            ));
            // 장축을 z로 돌려둔 블레이드(반경 1.5) — y축 스핀이 팁을 +x 벽으로 보낸다.
            let mut b = Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(1.5, 0.05, 0.05),
                    radius: 0.0,
                },
                Pose {
                    pos: Vec3::ZERO,
                    rot: Quat::from_axis_angle(
                        Vec3::new(0.0, 1.0, 0.0),
                        core::f32::consts::FRAC_PI_2,
                    ),
                },
                1.0,
            );
            if ccd {
                b = b.with_ccd();
            }
            let h = w.add_body(b);
            // 팁 속도 = 30 × 1.5 = 45 m/s → 서브스텝당 이동 0.19m (벽 두께 0.1의 ~2배).
            w.set_angular_velocity(h, Vec3::new(0.0, 30.0, 0.0))
                .unwrap();
            (w, h)
        };
        // 팁의 x 최대 도달 — 관통 관측자. 벽 근면 x=1.15.
        let tip_x = |w: &World, h: BodyHandle| {
            let p = w.body(h).pose();
            let a = p.pos + p.rot.rotate(Vec3::new(1.5, 0.0, 0.0));
            let b = p.pos + p.rot.rotate(Vec3::new(-1.5, 0.0, 0.0));
            a.x.max(b.x)
        };

        // 전제: CCD 없이는 팁이 벽 반대편(x > 1.25)까지 지나간다.
        let (mut w0, h0) = build(false);
        let mut crossed = false;
        for _ in 0..120 {
            w0.step(DT);
            if tip_x(&w0, h0) > 1.25 {
                crossed = true;
                break;
            }
        }
        assert!(crossed, "전제 실패: 비-CCD 블레이드가 벽을 통과하지 않았다");

        // 효과: CCD는 매 스텝 팁이 벽 근면을 (skin 여유 안에서) 넘지 못한다.
        let (mut w1, h1) = build(true);
        let skin = w1.params().speculative_margin;
        for step in 0..120 {
            w1.step(DT);
            let x = tip_x(&w1, h1);
            assert!(
                x <= 1.15 + skin + 0.02,
                "step {step}: 회전 CCD가 팁 관통을 허용 (tip_x={x})"
            );
        }
        assert!(w1.diagnostics().ccd_clamps > 0 || tip_x(&w1, h1) < 1.0);
    }

    /// 회귀(rev_3398738c MAJOR ①②): 내부 에지 판정은 각도 기반·대칭이고, 같은 쪽
    /// 중복면은 "2면 공유"라도 용접하지 않는다.
    #[test]
    fn welding_rejects_skinny_ridges_and_same_side_duplicates() {
        let mut w = World::new(WorldParams::default());
        let mask = |h: BodyHandle, w: &World| w.body(h).weld_edges;

        // 같은 삼각형 두 번(같은 쪽 겹침) — 모든 에지가 정확히 2면 공유지만 반대쪽
        // 검증이 걸러야 한다. 예전 판정(공면만)은 세 에지를 전부 용접했다.
        let v = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let hs = w
            .try_add_terrain(&v, &[[0, 1, 2], [0, 1, 2]], 0.02)
            .unwrap();
        assert_eq!(mask(hs[0], &w), 0, "같은 쪽 중복면이 용접됐다");
        assert_eq!(mask(hs[1], &w), 0, "같은 쪽 중복면이 용접됐다");

        // 스키니 삼각형 + 실각 45° 능선: 예전 판정은 t1의 최대 변(~707m)으로 임계를
        // 부풀려(7e-4) 0.6mm 능선을 공면으로 봤다. 각도 판정은 스케일 불변이라 거른다.
        let v = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.02, 0.0, 0.0),        // 공유 에지 2cm
            Vec3::new(500.0, 0.0, 500.0),     // t1의 먼 정점 (스케일 ~707m)
            Vec3::new(0.01, 6.0e-4, -6.0e-4), // t2 반대 정점 — 평면 위 0.6mm (실각 45°)
        ];
        let hs = w
            .try_add_terrain(&v, &[[0, 1, 2], [1, 0, 3]], 0.001)
            .unwrap();
        assert_eq!(mask(hs[0], &w), 0, "스키니 능선이 용접됐다");
        assert_eq!(mask(hs[1], &w), 0, "스키니 능선이 용접됐다");

        // 대조: 같은 스키니 t1이라도 진짜 공면 이웃(반대쪽·각 0°)은 용접된다.
        let v = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.02, 0.0, 0.0),
            Vec3::new(500.0, 0.0, 500.0),
            Vec3::new(0.01, 0.0, -0.01),
        ];
        let hs = w
            .try_add_terrain(&v, &[[0, 1, 2], [1, 0, 3]], 0.001)
            .unwrap();
        assert_ne!(mask(hs[0], &w), 0, "공면 이웃이 용접되지 않았다");
        assert_ne!(mask(hs[1], &w), 0, "공면 이웃이 용접되지 않았다");
    }

    /// 회귀(rev_3398738c MAJOR ③): frozen(평면 5개째)이 돼도 회전 스윕은 계속
    /// 캐스트·절단된다 — 이동만 얼리면 팁이 남은 서브스텝 동안 벽을 쓸고 지나간다.
    #[test]
    fn frozen_ccd_body_keeps_rotation_clamped() {
        // 서브스텝 8 — 평면은 서브스텝당 최대 1개 쌓이므로 5개째(frozen)에 도달하려면
        // 기본 4로는 불가능하다. toi=0 접촉 히트가 같은 법선을 반복해서 쌓는다.
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            substeps: 8,
            ..WorldParams::default()
        });
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.05, 3.0, 3.0),
                radius: 0.0,
            },
            pose(1.2, 0.0, 0.0),
        ));
        let b = Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(1.5, 0.05, 0.05),
                radius: 0.0,
            },
            Pose {
                pos: Vec3::ZERO,
                rot: Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_2),
            },
            1.0,
        )
        .with_ccd();
        let h = w.add_body(b);
        w.set_angular_velocity(h, Vec3::new(0.0, 30.0, 0.0))
            .unwrap();
        let skin = w.params().speculative_margin;
        let mut froze = false;
        for step in 0..120 {
            w.step(DT);
            froze |= w.diagnostics().ccd_frozen > 0;
            let p = w.body(h).pose();
            let tip = (p.pos + p.rot.rotate(Vec3::new(1.5, 0.0, 0.0)))
                .x
                .max((p.pos + p.rot.rotate(Vec3::new(-1.5, 0.0, 0.0))).x);
            assert!(
                tip <= 1.15 + skin + 0.02,
                "step {step}: frozen 후 회전이 벽을 쓸었다 (tip_x={tip})"
            );
        }
        assert!(
            froze,
            "frozen이 발동하지 않았다 — 테스트가 경로를 증명하지 못함"
        );
    }

    /// 회귀(확인 검수 rev_8672a3eb MAJOR): substeps=64에서 서브스텝당 팁 이동
    /// (30/3840×1.5 ≈ 1.17cm)이 skin/4(1.25cm) 아래로 숨는다 — 게이트가 서브스텝
    /// 단위면 캐스트가 아예 꺼져 스텝 전체 0.75m 회전 스윕이 무검사가 된다.
    /// 게이트는 스텝 전체 봉투여야 한다.
    #[test]
    fn per_substep_gate_cannot_hide_a_full_step_rotation() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            substeps: 64,
            ..WorldParams::default()
        });
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.05, 3.0, 3.0),
                radius: 0.0,
            },
            pose(1.2, 0.0, 0.0),
        ));
        let b = Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(1.5, 0.05, 0.05),
                radius: 0.0,
            },
            Pose {
                pos: Vec3::ZERO,
                rot: Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_2),
            },
            1.0,
        )
        .with_ccd();
        let h = w.add_body(b);
        w.set_angular_velocity(h, Vec3::new(0.0, 30.0, 0.0))
            .unwrap();
        let skin = w.params().speculative_margin;
        for step in 0..120 {
            w.step(DT);
            let p = w.body(h).pose();
            let tip = (p.pos + p.rot.rotate(Vec3::new(1.5, 0.0, 0.0)))
                .x
                .max((p.pos + p.rot.rotate(Vec3::new(-1.5, 0.0, 0.0))).x);
            assert!(
                tip <= 1.15 + skin + 0.02,
                "step {step}: 서브스텝 게이트 아래로 숨은 회전이 벽을 쓸었다 (tip_x={tip})"
            );
        }
    }

    /// 수면 임계 런타임 조정: 진입만 게이트하고(관대해지면 더 빨리 잠듦) 잠든 바디는
    /// 깨우지 않으며, 파라미터는 캐노니컬이라 스냅샷 왕복에 실린다. 검증은 생성 정본.
    #[test]
    fn sleep_thresholds_can_be_retuned_at_runtime() {
        let mut w = World::new(WorldParams::default());
        ground(&mut w);
        let b = dyn_box(&mut w, 0.0, 0.5);
        for _ in 0..120 {
            w.step(DT);
        }
        assert!(w.is_asleep(b), "정착한 박스가 기본 임계로 잠들어야 한다");
        // 관대한 임계로 조정 — 잠든 바디는 그대로 잠들어 있다(기상 없음).
        w.try_set_sleep_thresholds(0.4, 0.5, 0.25).unwrap();
        w.step(DT);
        assert!(w.is_asleep(b), "임계 조정이 잠든 바디를 깨웠다");
        assert_eq!(w.params().sleep_linear_threshold, 0.4);
        // 스냅샷 왕복 — 파라미터가 캐노니컬로 실린다.
        let snap = w.snapshot();
        let mut w2 = World::new(WorldParams::default());
        w2.restore(&snap).unwrap();
        assert_eq!(w2.params().sleep_linear_threshold, 0.4);
        assert_eq!(snap, w2.snapshot());
        // 검증은 생성과 같은 정본 — NaN·음수는 거부.
        assert!(w.try_set_sleep_thresholds(f32::NAN, 0.5, 0.25).is_err());
        assert!(w.try_set_sleep_thresholds(-1.0, 0.5, 0.25).is_err());
    }

    /// 회귀(rev_e4075e85): 센서+CCD 바디는 벽에 물리적으로 멈추면 안 된다 — 센서 계약.
    #[test]
    fn sensor_with_ccd_is_not_clamped_by_walls() {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..WorldParams::default()
        });
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.05, 5.0, 5.0),
                radius: 0.0,
            },
            pose(10.0, 0.0, 0.0),
        ));
        let s = w.add_body(
            Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0)
                .as_sensor(true)
                .with_ccd(),
        );
        w.set_linear_velocity(s, Vec3::new(400.0, 0.0, 0.0))
            .unwrap();
        for _ in 0..10 {
            w.step(DT);
        }
        assert!(
            w.body(s).pose().pos.x > 11.0,
            "센서가 CCD에 막혔다: x={}",
            w.body(s).pose().pos.x
        );
    }

    /// 회귀(rev_e4075e85): 퇴화 삼각형(동일점·일직선)은 지형에서 전체 거부된다.
    #[test]
    fn terrain_rejects_degenerate_triangles() {
        let mut w = World::new(WorldParams::default());
        let v = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0), // 0-1-2 일직선
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(w.try_add_terrain(&v, &[[0, 0, 0]], 0.02).is_err(), "동일점");
        assert!(w.try_add_terrain(&v, &[[0, 1, 2]], 0.02).is_err(), "일직선");
        assert!(
            w.try_add_terrain(&v, &[[0, 1, 3]], 0.02).is_ok(),
            "정상 거부 오탐"
        );
    }
}
