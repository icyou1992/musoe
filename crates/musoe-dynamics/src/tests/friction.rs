//! friction 관련 테스트.

use super::*;

/// theta(도)로 기울인 바닥 + 그 위에 같은 자세로 놓인 상자. `steps` 뒤의 내리막 변위.
fn slope_slide_distance(theta_deg: f32, steps: usize) -> f32 {
    let (mut w, h, down) = slope_scene(theta_deg);
    let p0 = w.body(h).pose().pos;
    for _ in 0..steps {
        w.step(DT);
    }
    (w.body(h).pose().pos - p0).dot(down)
}

/// 같은 장면의 `steps` 뒤 내리막 속도.
fn slope_slide_velocity(theta_deg: f32, steps: usize) -> f32 {
    let (mut w, h, down) = slope_scene(theta_deg);
    for _ in 0..steps {
        w.step(DT);
    }
    w.body(h).linvel().dot(down)
}

fn slope_scene(theta_deg: f32) -> (World, BodyHandle, Vec3) {
    let theta = theta_deg.to_radians();
    let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), theta);
    let mut w = World::new(no_sleep_params());
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
    // 바닥 윗면은 로컬 +y=1 — 법선 방향으로 1(반두께) + 0.5(상자 반두께)만큼 띄운다.
    let n = rot.rotate(Vec3::new(0.0, 1.0, 0.0));
    let h = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        Pose { pos: n * 1.5, rot },
        1.0,
    ));
    (w, h, rot.rotate(Vec3::new(-1.0, 0.0, 0.0)))
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

// ── 마찰 품질 게이트 ────────────────────────────────────────────────────────
//
// 위 `friction_stops_sliding_box`는 "마찰이 **있다**"만 잠근다 — mu를 10배로 키워도
// 통과한다. 마찰 솔버를 **바꾸려면**(마찰 2축을 2x2 블록으로 동시에 푸는 안) 크기·
// 방향·임계까지 잠근 기준선이 필요하다. 아래 넷이 그 기준선이고, 숫자는 계측기
// (`cargo run --release -p musoe-bench --bin friction`)로 뽑은 실측에서 왔다.

/// 경사면 위 물체의 정지 조건은 `tan(theta) <= mu`다 (mu=0.6 → 임계 30.96도).
///
/// **양 끝을 본다.** 정지 쪽만 보면 "마찰이 무한대"인 버그가 통과하고, 미끄럼 쪽만
/// 보면 "마찰이 0"인 버그가 통과한다. 구간 불변식은 두 끝을 다 관측해야 성립한다.
/// 실측 임계는 29도(정지)와 30.5도(미끄럼 시작) 사이 — 이론에서 1.5도 안쪽이다.
#[test]
fn friction_slope_threshold_brackets_atan_mu() {
    // 임계 아래: 3초 동안 사실상 제자리
    let d = slope_slide_distance(29.0, 180);
    assert!(
        d.abs() < 0.01,
        "임계 아래(29도)인데 미끄러졌다: {d:.4} m (이론 임계 30.96도)"
    );
    // 임계 위: 뚜렷하게 미끄러진다
    let d = slope_slide_distance(31.5, 180);
    assert!(
        d > 0.1,
        "임계 위(31.5도)인데 안 미끄러졌다: {d:.4} m (이론 임계 30.96도)"
    );
}

/// 미끄러질 때의 가속도는 `g(sin θ − mu cos θ)`여야 한다 — 마찰의 **크기**를 잠근다.
/// 정지 임계만 보면 mu가 맞아도 미끄럼 중 마찰이 틀릴 수 있다(정지 임계는 부등식이라
/// 한쪽만 규정한다).
#[test]
fn sliding_acceleration_matches_coulomb_theory() {
    // 45·55도까지 넣는다 — 순차 GS 때는 각각 +9.6%·+4.6% 벗어나 못 넣었는데,
    // 두 축 동시 해로 바꾸자 전 각도가 이론과 0.1% 안에서 맞는다.
    for deg in [37.0f32, 40.0, 45.0, 55.0] {
        let th = deg.to_radians();
        let a_theory = 9.81 * (th.sin() - 0.6 * th.cos());
        // 접촉 안정화 20스텝을 버리고 그 뒤 1초의 평균 가속도를 본다.
        let v0 = slope_slide_velocity(deg, 20);
        let v1 = slope_slide_velocity(deg, 80);
        let a = (v1 - v0) / (60.0 * DT);
        let ratio = a / a_theory;
        assert!(
            (0.97..=1.03).contains(&ratio),
            "{deg}도 미끄럼 가속도가 이론과 어긋난다: 실측 {a:.4} / 이론 {a_theory:.4} = {ratio:.3}"
        );
    }
}

/// 마찰은 접선면에서 **방향에 무관**해야 한다. 축0 → 축1 **순차** Gauss-Seidel이던
/// 시절에는 먼저 푸는 축이 유리해 폭이 6.85%였다(축 정렬 0.7584 m 대 직교 0.8104 m).
/// 두 축을 **동시에** 풀고 클램프를 적용 앞으로 옮기자 8방향이 전부 0.7584로 모였다.
/// 그래서 상한을 8% → **0.5%**로 조인다 — 느슨한 채로 두면 순차 GS로 되돌아가도 통과한다.
///
/// **뮤테이션 기록**: 기본 마찰을 5.0으로 키우면 발화한다. 그러나 **0.0으로 만들면
/// 발화하지 않는다** — 마찰이 없으면 모든 방향이 똑같이 멀리 미끄러져 "완벽하게
/// 등방"이기 때문이다. 이 테스트가 재는 것은 방향 의존성뿐이고, 마찰의 **크기**는
/// `sliding_acceleration_matches_coulomb_theory`와
/// `friction_slope_threshold_brackets_atan_mu`가 잠근다(둘 다 mu=0에서 발화한다).
#[test]
fn tangent_plane_friction_is_nearly_isotropic() {
    // **한 월드에 8개를 나란히 놓는다.** 처음엔 방향마다 월드를 따로 만들었는데,
    // 그러면 접촉 제약이 스텝당 하나뿐이라 컬러링이 블록을 만들지 못해 **SIMD 블록
    // 경로(`solver_block::solve_block`)를 한 번도 안 탄다** — 마찰 2x2 변경은 스칼라와
    // 블록 **양쪽**에 들어가는데 게이트가 스칼라만 덮고 있었다(블록 경로에 panic을 심어
    // 확인함: 이 파일의 마찰 게이트 중 스택 테스트만 도달했다).
    // 상자끼리 바닥을 공유하면 색이 갈리므로 **바닥도 각자 따로** 둔다.
    let mut w = World::new(no_sleep_params());
    let mut hs = Vec::new();
    for k in 0..8 {
        let ang = k as f32 * core::f32::consts::TAU / 8.0;
        let cx = k as f32 * 100.0; // 서로 접촉·브로드페이즈 간섭이 없을 만큼 멀리
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(20.0, 1.0, 20.0),
                radius: 0.0,
            },
            pose(cx, -1.0, 0.0),
        ));
        let h = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(cx, 0.5, 0.0),
            1.0,
        ));
        w.set_linear_velocity(h, Vec3::new(ang.cos() * 3.0, 0.0, ang.sin() * 3.0))
            .unwrap();
        hs.push((h, Vec3::new(cx, 0.5, 0.0)));
    }
    for _ in 0..300 {
        w.step(DT);
    }
    let dists: Vec<f32> = hs
        .iter()
        .map(|&(h, p0)| (w.body(h).pose().pos - p0).length())
        .collect();
    let dmin = dists.iter().cloned().fold(f32::MAX, f32::min);
    let dmax = dists.iter().cloned().fold(0.0f32, f32::max);
    // 이동 자체가 없으면 비율이 무의미하다 — 분모가 살아 있는지 먼저 본다.
    assert!(
        dmin > 0.3,
        "8방향 모두에서 충분히 미끄러져야 한다: min {dmin:.4} m"
    );
    let spread = dmax / dmin - 1.0;
    assert!(
        spread < 0.005,
        "접선 마찰 비등방이 커졌다: {:.3}% (기준선 ~0.01%, 상한 0.5%) · min {dmin:.4} max {dmax:.4}",
        100.0 * spread
    );
}

/// 제자리 회전(스핀)은 마찰이 죽여야 한다 — 접선 임펄스가 레버암을 통해 토크로 간다.
///
/// **양 끝을 본다** (교차검수 rev_ac00206f MAJOR). 처음엔 "0.5초 뒤 0.1 아래"라는
/// **상한만** 뒀는데, 그러면 마찰 토크가 **지나치게 큰** 회귀가 통과한다: 2x2 역행렬의
/// 교차항 부호·배율이 틀려 선형 임펄스는 상쇄되고 토크만 배로 커지면 5 rad/s가 첫 스텝에
/// 0으로 스냅되는데, 경사·이동거리 테스트는 그대로 통과한다. 그래서 넷을 함께 잠근다:
/// 이른 시점의 **하한**, 늦은 시점의 **상한**, 부호 역전 금지, 각속력 증가(에너지 주입) 금지.
///
/// 기준선(계측기 실측): 스텝당 약 0.408 rad/s씩 선형 감소해 12~13스텝에 소멸.
/// 5스텝 2.966 · 12스텝 0.106 · 20스텝 −0.00001(부호 역전 아님, 수치 잔여).
#[test]
fn spin_decays_gradually_without_overshoot() {
    // 등방성 게이트와 같은 이유로 **4개를 나란히** 놓아 SIMD 블록 경로까지 덮는다
    // (하나만 두면 제약이 1개라 스칼라 경로만 돈다 — 검수가 든 "토크만 2배" 시나리오는
    //  블록 경로에 심었을 때 하나짜리 씬을 그대로 통과했다).
    let mut w = World::new(no_sleep_params());
    let mut hs = Vec::new();
    for k in 0..4 {
        let cx = k as f32 * 100.0;
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(20.0, 1.0, 20.0),
                radius: 0.0,
            },
            pose(cx, -1.0, 0.0),
        ));
        let h = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(cx, 0.5, 0.0),
            1.0,
        ));
        hs.push(h);
    }
    const W0: f32 = 5.0;
    for &h in &hs {
        w.set_angular_velocity(h, Vec3::new(0.0, W0, 0.0)).unwrap();
    }
    let mut at5 = vec![f32::NAN; hs.len()];
    for i in 0..30 {
        w.step(DT);
        for (k, &h) in hs.iter().enumerate() {
            let wy = w.body(h).angvel().y;
            // 에너지 주입 금지 — 마찰은 각속력을 키울 수 없다.
            assert!(
                wy.abs() <= W0 + 1e-3,
                "바디 {k}, {}스텝에서 각속력이 초기값을 넘었다: {wy:.5} > {W0}",
                i + 1
            );
            // 부호 역전 금지 — 마찰이 스핀을 반대로 돌리면 안 된다.
            assert!(
                wy > -0.05,
                "바디 {k}, {}스텝에서 스핀이 뒤집혔다: {wy:.5}",
                i + 1
            );
            if i == 4 {
                at5[k] = wy;
            }
        }
    }
    for (k, &v) in at5.iter().enumerate() {
        // 이른 시점 **하한**: 너무 빨리 죽으면 마찰 토크가 과하다는 뜻이다(기준선 2.966).
        assert!(
            v > 2.0,
            "바디 {k}: 5스텝 만에 스핀이 과하게 꺾였다: {v:.5} rad/s (기준선 2.97) — 마찰 토크 과다"
        );
    }
    for (k, &h) in hs.iter().enumerate() {
        // 늦은 시점 **상한**: 남아 있으면 마찰 토크가 부족하다.
        let wy = w.body(h).angvel().y.abs();
        assert!(
            wy < 0.01,
            "바디 {k}: 0.5초 뒤에도 스핀이 남았다: {wy:.5} rad/s"
        );
    }
}

/// **비등방 관성 + 모서리 접촉** — 2x2 접선 유효질량의 교차항이 크고 조건수가 높은 영역
/// (교차검수 rev_ac00206f MAJOR: 기존 넷은 전부 질량 1의 정육면체 면접촉이라 이 영역을
/// 건드리지 않았다). 길쭉한 막대를 기울여 떨어뜨리면 한쪽 끝(모서리)으로 먼저 닿고,
/// 관성비가 극단적이라 레버암이 길다 — 행렬식이 작아지는 조건이다.
///
/// 여기서 `det` 근0 처리나 역행렬 식이 틀리면 NaN이나 거대 임펄스로 나타난다.
/// 기준선: 최종 정지(|v|=0.00000), 최대 침투 0.02823 m, 최대 |v| 5.4275(낙하 가속 포함).
#[test]
fn anisotropic_rod_on_edge_contact_stays_finite_and_settles() {
    let mut w = World::new(no_sleep_params());
    ground(&mut w);
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
    let (mut max_pen, mut max_v) = (0.0f32, 0.0f32);
    for _ in 0..400 {
        w.step(DT);
        max_pen = max_pen.max(w.diagnostics().max_penetration);
        max_v = max_v.max(w.body(h).linvel().length());
        assert!(
            w.body(h).pose().pos.y.is_finite() && w.body(h).linvel().length().is_finite(),
            "막대 상태가 유한하지 않다 — 접선 행렬이 터졌다"
        );
    }
    assert!(
        max_v < 8.0,
        "속도가 튀었다: 최대 |v| {max_v:.4} (기준선 5.43) — 거대 임펄스 신호"
    );
    assert!(
        max_pen < 0.05,
        "침투가 커졌다: {max_pen:.5} m (기준선 0.02823)"
    );
    let b = w.body(h);
    assert!(
        b.linvel().length() < 0.02 && b.angvel().length() < 0.02,
        "막대가 정지하지 않았다: |v|={:.5} |w|={:.5}",
        b.linvel().length(),
        b.angvel().length()
    );
}

/// **마찰이 역학적 에너지를 주입하지 않는가** (교차검수 rev_87748fe6 MAJOR).
///
/// 지금 마찰은 "대각 질량 스텝 → 유클리드 원판 투영"인데, 이건 `K`가 등방일 때만
/// 제약된 최대소산 해와 일치한다. 비등방 접촉에서는 원판 안의 이전 임펄스에서
/// **에너지를 늘리는** 갱신이 가능하다 — 검수가 준 반례를 산술로 확인했다
/// (`K=diag(0.5,2)`, `muN=1`, `lambda=(-0.75,-0.5)`, `v=(1.5,1.5)` → `dE=+0.0213`).
///
/// **이건 순차 GS 시절부터 있던 성질이고, 두 축 동시 해로 바꾸면서 오히려 줄었다**
/// (같은 계측기·같은 장면으로 두 바이너리 비교: 양의 증가 합 0.04211 → 0.04172 J,
///  스핀 포함 장면 54.61 → 50.80 J). 그래서 이 커밋의 회귀는 아니지만, **알려진 결함**
/// 이므로 현재 수준을 상한으로 못박아 더 나빠지는 것을 막는다. 제대로 고치려면 활성
/// 원판에서 `(K + alpha I) lambda = -q`, `|lambda| = mu*N`을 풀어야 한다(별도 작업).
///
/// 등방(정육면체)은 **정확히 0**이어야 한다 — 그쪽이 깨지면 투영 자체가 잘못된 것이다.
///
/// **뮤테이션 기록 — 이 게이트는 민감하지 않다.** 마찰 임펄스를 1.3배로 키우는 뮤턴트와
/// 접선면에서 20도 회전시키는 뮤턴트(검수가 지목한 방향 오류 계열) **둘 다 통과했다**.
/// 즉 이 테스트는 "현재 수준보다 나빠지지 않는다"는 **래칫**이지, 에너지 주입을 잡아내는
/// 탐지기가 아니다. 이 장면의 증가분은 접촉 소프트니스 등 다른 몫이 섞여 있어 마찰
/// 오차가 묻힌다. 제대로 잡으려면 접촉을 하나로 고정하고 임펄스 단위로 `dE = v·d + ½dᵀKd`를
/// 직접 계산하는 단위 테스트가 필요하다(별도 작업 — 제약 해 자체를 고칠 때 함께).
///
/// 에너지 식 함정: 처음에는 **월드 각속도에 몸체 관성**을 곱했다. 회전한 물체에서 그 값은
/// 틀리고, 오차가 "에너지 증가"로 잡혀 게이트가 자기 식의 오차를 재게 된다.
/// 각속도를 몸체 프레임으로 되돌린 뒤 주축 관성을 곱해야 한다.
#[test]
fn friction_does_not_inject_mechanical_energy() {
    for (label, half, limit) in [
        ("등방 정육면체", Vec3::new(0.5, 0.5, 0.5), 0.0f32),
        ("비등방 막대", Vec3::new(0.1, 0.1, 2.0), 0.045),
    ] {
        let mut w = World::new(no_sleep_params());
        ground(&mut w);
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
        // 균질 직육면체의 주축 관성 (질량 1). 회전 에너지 항이 없으면 스핀이 마찰로
        // 죽는 것을 "에너지 감소"로 오독한다.
        let (hx, hy, hz) = (half.x, half.y, half.z);
        let inertia = Vec3::new(
            (hy * hy + hz * hz) / 3.0,
            (hx * hx + hz * hz) / 3.0,
            (hx * hx + hy * hy) / 3.0,
        );
        let energy = |w: &World| {
            let b = w.body(h);
            let (v, om) = (b.linvel(), b.angvel());
            // **각속도를 몸체 프레임으로 되돌려서** 주축 관성을 곱한다. 월드 각속도에
            // 몸체 관성을 그대로 곱하면 회전한 물체에서 값이 틀리고, 그 오차가 "에너지
            // 증가"로 잘못 잡혀 게이트가 자기 식의 오차를 재게 된다(처음에 그랬다).
            let ob = b.pose().rot.conjugate().rotate(om);
            0.5 * v.dot(v)
                + 0.5
                    * (inertia.x * ob.x * ob.x + inertia.y * ob.y * ob.y + inertia.z * ob.z * ob.z)
                + 9.81 * b.pose().pos.y
        };
        let mut prev = energy(&w);
        let mut gain = 0.0f32;
        for _ in 0..400 {
            w.step(DT);
            let e = energy(&w);
            if e > prev {
                gain += e - prev;
            }
            prev = e;
        }
        assert!(
            gain <= limit,
            "{label}: 마찰이 에너지를 주입했다 — 양의 증가 합 {gain:.5} J (상한 {limit})"
        );
    }
}

/// 격자 위를 굴러가는 구가 셀 경계를 지나도 **뚫거나 튀지 않는다** (거동 회귀).
///
/// 내부 에지 억제 자체의 증거는 여기가 아니라
/// `musoe_collide::manifold::heightfield_contact_tests::welding_snaps_seam_contact_normal_to_the_face`
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
        musoe_collide::Pose {
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

/// 거동: 평평한 이음새를 미끄러져 건너는 구가 옆으로 차이지 않는다.
///
/// **이 테스트는 용접(내부 에지 억제)을 검증하지 않는다** — 2026-07-30에 실측으로 확인했다:
/// `weld_internal_edges` 호출을 통째로 비활성화해도 통과하고, 같은 지형을 용접 없는 개별
/// static hull로 깔면 측면 속도가 오히려 **더 작다**(0.019 vs 0.037). 평평한 격자에서는
/// 이음새 법선이 이미 면 법선과 같아 용접이 no-op이기 때문이다(같은 함정이 하이트필드
/// 이음매 테스트에서도 한 번 잡혔다 — 기제 증거는 **경사 격자에 구를 파묻어야** 나온다).
/// 용접의 진짜 가드는 `musoe-collide`의 `welding_snaps_seam_contact_normal_to_the_face`다.
/// 여기 남기는 이유는 **거동 스모크**(이음새를 건너다 튀지 않는다)로는 유효하기 때문이다.
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
    // **착지 전이와 이음새를 나눠 잰다.** 예전에는 전 구간에 절대 임계(0.05) 하나를 걸었는데,
    // 실측하니 측면 속도의 최댓값은 **이음새가 아니라 착지**에서 나온다(궤적 계측:
    // 착지 0.046 / 이음새 0.037, substeps=4 기준). 즉 그 단언은 이름과 달리 착지 과도응답을
    // 재고 있었고, 서브스텝을 4→2로 낮추자(2026-07-30 "속도 우선") 착지 0.058로 임계를
    // 넘겨 실패했다 — 이음새는 0.046으로 여전히 착지보다 낮았다.
    //
    // 그래서 **관계로** 검사한다: 이음새를 건너는 것이 착지보다 더 차면 안 된다.
    // 절대 임계보다 강한 진술이고(용접이 깨지면 이음새가 착지를 넘어선다), 서브스텝 수에
    // 의존하지 않는다. 절대 상한은 폭발 감지용으로만 느슨하게 남긴다.
    let mut max_landing = 0.0f32;
    let mut max_seam = 0.0f32;
    for _ in 0..90 {
        w.step(DT);
        let p = w.body(ball).pose().pos;
        let v = w.body(ball).linvel();
        if p.x < -0.5 {
            max_landing = max_landing.max(v.z.abs());
        } else {
            max_seam = max_seam.max(v.z.abs());
        }
        assert!(v.z.abs() < 0.1, "측면 폭발: vz={} (x={})", v.z, p.x);
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
    // **음성 대조**: 착지에서 실제로 측면 성분이 생겨야 이 비교가 의미 있다.
    assert!(
        max_landing > 0.01,
        "착지 전이가 없어 비교가 무의미하다: {max_landing}"
    );
    assert!(
        max_seam <= max_landing,
        "이음새가 착지보다 더 찼다 — 내부 에지 억제가 깨졌다 (이음새 {max_seam} > 착지 {max_landing})"
    );
}
