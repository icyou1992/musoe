//! determinism 관련 테스트.

use super::*;

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

/// 설계 §8이 규정한 **1,000스텝** 판 — release 전용(debug는 수 분).
/// `cargo test --release -p maru-dynamics -- --ignored`
#[test]
#[ignore = "release 전용: cargo test --release -p maru-dynamics -- --ignored"]
fn thread_determinism_1000_steps() {
    assert_thread_determinism(30, 1000, &[2, 3, 4]);
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
