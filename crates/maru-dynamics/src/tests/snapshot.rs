//! snapshot 관련 테스트.

use super::*;

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

/// 메시 스냅샷 왕복 — 파생물(BVH·용접 마스크)은 저장하지 않고 복원에서 다시 만든다.
#[test]
fn trimesh_survives_snapshot_roundtrip() {
    let mut w = World::new(WorldParams::default());
    let mesh = maru_collide::Shape::trimesh(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(2.0, 0.0, 2.0),
        ],
        vec![[0, 1, 2], [1, 3, 2]],
    )
    .expect("정상 메시");
    w.add_body(Body::new_static(mesh, pose(0.0, -1.0, 0.0)));
    w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.4 },
        pose(1.0, 2.0, 1.0),
        1.0,
    ));
    for _ in 0..30 {
        w.step(1.0 / 60.0);
    }
    let snap = w.snapshot();
    let mut w2 = World::new(WorldParams::default());
    w2.restore(&snap).expect("복원");
    // **복원 뒤에도 같은 물리가 이어져야 한다** — 파생물을 저장하지 않으므로
    // 복원이 BVH를 다시 만들고, 그게 틀리면 여기서 궤적이 갈린다.
    assert_eq!(w2.snapshot(), snap, "왕복이 바이트 동일이 아니다");
    let mut a = w;
    let mut b = w2;
    for i in 0..60 {
        a.step(1.0 / 60.0);
        b.step(1.0 / 60.0);
        assert_eq!(
            a.snapshot(),
            b.snapshot(),
            "복원 후 step {i}에서 궤적이 갈렸다"
        );
    }
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
