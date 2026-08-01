//! 셰이프 쌍 → 알고리즘 **디스패치**와 순회 API.
//!
//! `contact`(단일 매니폴드)와 `for_each_contact*`(컨테이너까지 펼치는 순회)가 여기 있다.
//! 비볼록 셰이프를 볼록 경로로 내려보내지 않는 정책 가드(`is_non_convex`)도 이 층이다 —
//! 새 셰이프 변종을 더할 때 반드시 훑어야 하는 자리다.

use super::clip::clip_box_box;
use super::container::{heightfield_contacts, trimesh_contacts};
use super::sat::{box_box_axis, sphere_box, sphere_sphere, SatResult};
use super::*;

/// 접촉 존재 판정 — wake 스윕용 (복합·하이트필드 디스패치 포함).
///
/// **`bool`이 아니라 `Option<bool>`이다**: 하이트필드 순회는 셀 예산에서 끊길 수 있고,
/// 그때 `false`("접촉 없음")를 돌려주면 **실제로 닿아 있는 잠든 바디를 깨우지 않는다** —
/// 접촉 제약이 빠진 채 스텝이 진행된다(확인검수 rev_1a31d83f BLOCKER).
/// `None` = "판정하지 못했다"이고, 호출자는 그 스텝을 자원 오류로 끝내야 한다.
pub fn any_contact(sa: &Shape, pa: &Pose, sb: &Shape, pb: &Pose, margin: f32) -> Option<bool> {
    let mut found = false;
    let walk = for_each_contact(sa, pa, sb, pb, margin, &mut |_, _, m| {
        if !m.is_empty() {
            found = true;
        }
    });
    match walk {
        // 포화는 접촉을 **찾고** 끊은 것이라 존재 판정에는 충분하다.
        ContactWalk::Complete | ContactWalk::Saturated => Some(found),
        // 형상·입력이 깨졌으면 판정 자체가 성립하지 않는다 — `false`(=없음)로 뭉개지 않는다.
        ContactWalk::InvalidShape | ContactWalk::InvalidInput => None,
        // 예산이 터졌어도 **이미 하나 찾았으면 존재는 증명됐다** — 그때 `None`을 돌려주면
        // 지원 범위 안의 고정 장면(넓은 AABB + 접촉 몇 개)이 영구히 답을 못 받는다
        // (확인검수 rev_d5b32e22 MAJOR). 부정("없다")만 순회 완주를 요구한다.
        ContactWalk::BudgetExceeded if found => Some(true),
        ContactWalk::BudgetExceeded => None,
    }
}

/// 접촉 디스패처 — compound를 자식으로 펼쳐 자식쌍마다 매니폴드를 콜백에 준다.
/// 비복합 쌍은 (0,0) 키로 1회. 자식 순서는 목록 순 — 결정적. 콜백이 받은 매니폴드의
/// feature 네임스페이싱(자식쌍 키 접기)은 **호출자 책임**이다(키 조립 규약이 호출자에 있다).
/// 접촉 순회 결과 — **포화와 예산 초과는 다른 사건**이다.
///
/// 포화는 "접촉을 64개 찾고 끊었다"(뒤에 더 있을 수 있다)이고, 예산 초과는 "볼 셀이 너무
/// 많아 **판정 자체를 못 했다**"다. 후자를 포화로 뭉개면 접촉 없는 장면이 정상으로 보이는데
/// 실제로는 순회가 잘렸다 — 뒤쪽의 진짜 접촉이 조용히 사라진다(확인검수 rev_e50670ad BLOCKER).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContactWalk {
    /// 끝까지 걸었다.
    Complete,
    /// **산술 입력이 잘못돼 판정을 시작하지 못했다** (margin 비유한·음수, 자세 정규화 불가).
    ///
    /// NaN margin은 특히 조용하다: `dist2 > reach*reach` 비교가 NaN 때문에 false가 되어
    /// **멀리 떨어진 두 구가 접촉을 만든다**(확인검수 rev_97887d75 MAJOR).
    InvalidInput,
    /// **입력 형상의 불변식이 깨져 판정을 시작하지 못했다** (빈 hull·빈 복합·격자 개수 불일치).
    ///
    /// `Shape`의 필드가 공개라 호스트가 만들 수 있고, 그대로 GJK에 넣으면 `support_core`의
    /// 인덱싱에서 프로세스가 죽는다. `Complete`로 뭉개면 잘못된 형상이 "접촉 없음"으로
    /// 위장된다(확인검수 rev_2e93ebf2 BLOCKER의 접촉 쪽 형제 결함).
    InvalidShape,
    /// 매니폴드 상한(`MAX_HEIGHTFIELD_MANIFOLDS`)에서 끊었다.
    Saturated,
    /// 셀 예산을 넘어 순회를 포기했다 — 호출자는 이 스텝을 **자원 오류**로 다뤄야 한다.
    BudgetExceeded,
}

pub fn for_each_contact(
    sa: &Shape,
    pa: &Pose,
    sb: &Shape,
    pb: &Pose,
    margin: f32,
    f: &mut dyn FnMut(u8, u8, Manifold),
) -> ContactWalk {
    let mut budget = crate::MAX_HEIGHTFIELD_CELLS_PER_QUERY;
    for_each_contact_with_budget(sa, pa, sb, pb, margin, &mut budget, f)
}

/// `for_each_contact` + **호출자가 소유하는 셀 예산**.
///
/// 예산이 호출마다 새로 4096이면 한 질의의 **총** 작업량은 유계가 아니다: 같은 자리에 지형
/// 100개를 겹치면 `shape_penetrates` 한 번이 40만 셀을 훑는다(확인검수 rev_97887d75 MAJOR).
/// 여러 바디를 도는 질의는 예산 하나를 만들어 이 함수로 넘겨 **질의 단위**로 묶는다.
pub fn for_each_contact_with_budget(
    sa: &Shape,
    pa: &Pose,
    sb: &Shape,
    pb: &Pose,
    margin: f32,
    budget: &mut usize,
    f: &mut dyn FnMut(u8, u8, Manifold),
) -> ContactWalk {
    // 캐스트와 **같은 구조 검사**(O(1)) — 불변식이 깨진 형상은 GJK에서 패닉하거나
    // 미스로 위장된다. 정책 전수 검사(`validate`)는 O(n)이라 여기서 부르지 않는다.
    if !sa.is_castable() || !sb.is_castable() {
        return ContactWalk::InvalidShape;
    }
    // 산술 입력 검사와 자세 정규화 — 캐스트 디스패처와 같은 규약(그쪽
    // `canonical_cast_inputs` 주석 참조). margin이 NaN이면 거리 비교가 전부 false가 되어
    // 접촉이 **없는 곳에서 생긴다**.
    // 캐스트와 같은 규약 — 유한성이 아니라 지원 범위로 잰다(그쪽 주석의 `Inf` 차 논증).
    if !crate::in_range(margin)
        || margin < 0.0
        || !crate::vec_in_range(pa.pos)
        || !crate::vec_in_range(pb.pos)
    {
        return ContactWalk::InvalidInput;
    }
    let (Some(ra), Some(rb)) = (pa.rot.try_normalize(), pb.rot.try_normalize()) else {
        return ContactWalk::InvalidInput;
    };
    let (npa, npb) = (
        Pose {
            pos: pa.pos,
            rot: ra,
        },
        Pose {
            pos: pb.pos,
            rot: rb,
        },
    );
    let (pa, pb) = (&npa, &npb);
    let compose = |parent: &Pose, child: &Pose| Pose {
        pos: parent.pos + parent.rot.rotate(child.pos),
        // 자식 회전 정규화 — 근거는 `cast.rs`의 같은 자리 주석에.
        rot: parent.rot * child.rot.normalize(),
    };
    // 하이트필드는 바디 하나가 삼각형 수천 개다 — 겹치는 셀만 펼치고 **상한에서 끊는다**.
    // 끊었다는 사실은 반환값으로 나간다(조용한 누락 금지 — 호출자가 진단에 싣는다).
    if matches!(sb, Shape::Heightfield { .. }) {
        return heightfield_contacts(sa, pa, sb, pb, margin, false, budget, f);
    }
    if matches!(sa, Shape::Heightfield { .. }) {
        // 하이트필드가 A쪽인 경우(질의 경로) — 역할을 바꿔 풀고 매니폴드를 뒤집는다.
        return heightfield_contacts(sb, pb, sa, pa, margin, true, budget, f);
    }
    // 메시도 같다 — 가지치기 구조만 격자에서 BVH로 바뀐다.
    if matches!(sb, Shape::TriMesh { .. }) {
        return trimesh_contacts(sa, pa, sb, pb, margin, false, budget, f);
    }
    if matches!(sa, Shape::TriMesh { .. }) {
        return trimesh_contacts(sb, pb, sa, pa, margin, true, budget, f);
    }
    match (sa, sb) {
        (Shape::Compound { children: ca }, Shape::Compound { children: cb }) => {
            for (i, (cpa, csa)) in ca.iter().enumerate() {
                let wpa = compose(pa, cpa);
                for (j, (cpb, csb)) in cb.iter().enumerate() {
                    if let Some(m) = contact(csa, &wpa, csb, &compose(pb, cpb), margin) {
                        f(i as u8, j as u8, m);
                    }
                }
            }
        }
        (Shape::Compound { children }, _) => {
            for (i, (cp, cs)) in children.iter().enumerate() {
                if let Some(m) = contact(cs, &compose(pa, cp), sb, pb, margin) {
                    f(i as u8, 0, m);
                }
            }
        }
        (_, Shape::Compound { children }) => {
            for (j, (cp, cs)) in children.iter().enumerate() {
                if let Some(m) = contact(sa, pa, cs, &compose(pb, cp), margin) {
                    f(0, j as u8, m);
                }
            }
        }
        _ => {
            if let Some(m) = contact(sa, pa, sb, pb, margin) {
                f(0, 0, m);
            }
        }
    }
    ContactWalk::Complete
}

/// 볼록 전용 경로에 비볼록 셰이프가 들어왔는가 — 들어오면 GJK가 `unreachable!`에 닿는다.
#[inline]
pub(super) fn is_non_convex(s: &Shape) -> bool {
    // **비볼록 변종을 하나라도 빠뜨리면 GJK가 `support_core`의 `unreachable!`에 닿는다** —
    // 공개 API라 호스트가 한 줄로 프로세스를 죽일 수 있고 wasm에서는 trap이다.
    // TriMesh를 빠뜨렸던 것을 교차검수 rev_953115e8이 BLOCKER로 잡았다.
    // `Shape`에 비볼록 변종을 더할 때 **여기와 `cast.rs::is_non_convex`를 함께** 고쳐야 한다
    // (둘 다 `matches!`라 컴파일러가 안 잡아 준다 — 그래서 회귀 테스트로 잠갔다:
    // `mesh_in_a_convex_slot_does_not_trap`).
    matches!(
        s,
        Shape::Compound { .. } | Shape::Heightfield { .. } | Shape::TriMesh { .. }
    )
}

pub fn contact(sa: &Shape, pa: &Pose, sb: &Shape, pb: &Pose, margin: f32) -> Option<Manifold> {
    // **비볼록은 여기서 끝낸다.** 이 함수는 볼록 전용이고(디스패치는 `for_each_contact`),
    // 그냥 넘기면 `support_core`의 `unreachable!`이 프로세스를 죽인다 — 라이브러리가
    // 잘못된 입력에 trap하는 것은 이 저장소의 계약 위반이다(확인검수 rev_e50670ad BLOCKER).
    // **디버그에서도 패닉하지 않는다**: 공개 API의 거동이 빌드 프로파일에 따라 달라지면
    // "라이브러리는 trap하지 않는다"가 조건부 약속이 된다.
    if is_non_convex(sa) || is_non_convex(sb) {
        return None;
    }
    debug_assert!(
        !matches!(sa, Shape::Compound { .. }) && !matches!(sb, Shape::Compound { .. }),
        "compound는 for_each_contact로 디스패치해야 한다"
    );
    // **전용 경로 우선**: 구-구는 해석적으로 정확하고 5 flops면 끝난다. 이걸 GJK+EPA로
    // 돌리면 페어당 마이크로초가 든다(대조 실측: 내로우페이즈가 스텝의 58%).
    // 정확도도 전용 경로가 낫다 — 반복 수렴 오차가 없고 접점 ID가 스텝 간 안정적이라
    // warm start가 위치 근접 게이트에 의존하지 않는다.
    if let (Shape::Sphere { radius: ra }, Shape::Sphere { radius: rb }) = (sa, sb) {
        return sphere_sphere(*ra, pa.pos, *rb, pb.pos, margin);
    }
    // **구-박스도 해석 경로**: 박스 로컬로 옮겨 클램프하면 끝이다.
    // 이게 없을 때 실측으로 페어당 6.0us였다 — 같은 씬의 구-구가 0.014us니까 428배다.
    // 구 더미 씬에서 바닥판 접촉이 전부 이 경로를 타 내로우페이즈를 지배했다.
    if let (Shape::Sphere { radius: rs }, Shape::Box { half, radius: rbox }) = (sa, sb) {
        return sphere_box(*rs, pa.pos, *half, *rbox, pb, margin, true);
    }
    if let (Shape::Box { half, radius: rbox }, Shape::Sphere { radius: rs }) = (sa, sb) {
        return sphere_box(*rs, pb.pos, *half, *rbox, pa, margin, false);
    }
    // 박스 쌍의 회전행렬 — SAT·클리퍼·GJK 폴백 클리핑이 **모두 이 하나**를 공유한다.
    let mut box_mats: Option<(Mat3, Mat3)> = None;
    // **박스-박스는 SAT로 법선만 구한다.** 클리퍼는 법선만 요구하므로(깊이는 안 쓴다)
    // GJK+EPA 전체를 돌릴 이유가 없다. 클리퍼 자체는 M1에서 검증된 코드 그대로 쓰고
    // 법선의 출처만 바꾼다 — 실패하면(에지-에지 지배·퇴화) 아래 GJK 경로로 떨어진다.
    if let (
        Shape::Box {
            half: ha,
            radius: rab,
        },
        Shape::Box {
            half: hb,
            radius: rbb,
        },
    ) = (sa, sb)
    {
        // **회전행렬은 페어당 한 번**이고 SAT와 클리퍼가 공유한다. 예전에는 같은 자세로
        // 쿼터니언 회전을 SAT에서 6번·클리퍼에서 21번 다시 돌렸다(페어당 27회).
        // `Mat3::from_quat`이 하는 일이 바로 그 축 3개 회전이므로 SAT 쪽은 **비트 동일**이고
        // (`sat_axes_match_quaternion_rotation_bit_for_bit`가 잠근다), 클리퍼 쪽은
        // 행렬-벡터 곱으로 바뀌어 값이 마지막 비트에서 달라진다(아래 클리퍼 주석).
        // **폴백까지 살려서 재사용한다.** 아래 GJK/EPA 경로의 클리핑 재시도가 같은 자세로
        // 행렬을 다시 만들면 퇴화·에지 지배 페어만 회전을 12회 더 돌게 되고, "페어당 한 번"이
        // 그 경로에서 거짓이 된다(교차검수 rev_7a05abc1 MINOR).
        box_mats = Some((Mat3::from_quat(pa.rot), Mat3::from_quat(pb.rot)));
        let (ma, mb) = box_mats.as_ref().expect("방금 넣었다");
        match box_box_axis(*ha, *rab, pa, ma, *hb, *rbb, pb, mb, margin) {
            SatResult::Apart => return None,
            SatResult::Axis(n) => {
                let char_len = characteristic_length(sa, sb);
                let dup_len = min_feature_size(sa, sb);
                if let Some(m) = clip_box_box(
                    char_len, dup_len, *ha, *rab, pa, ma, *hb, *rbb, pb, mb, n, margin,
                ) {
                    return Some(m);
                }
            }
            SatResult::Degenerate => {}
        }
    }
    let (ra, rb) = (sa.radius(), sb.radius());
    let (normal, core_sep, on_a, on_b) = match gjk(sa, pa, sb, pb) {
        GjkResult::Separated {
            distance,
            on_a,
            on_b,
            normal,
        } => {
            let sep = distance - ra - rb;
            if sep > margin {
                return None;
            }
            (normal, sep, on_a, on_b)
        }
        GjkResult::Intersecting(seed) => {
            let p = epa(seed, sa, pa, sb, pb);
            (p.normal, -(p.depth + ra + rb), p.on_a, p.on_b)
        }
    };

    // box-box + 면 지배 → 클리핑 매니폴드
    if let (
        Shape::Box {
            half: ha,
            radius: ra_box,
        },
        Shape::Box {
            half: hb,
            radius: rb_box,
        },
    ) = (sa, sb)
    {
        let char_len = characteristic_length(sa, sb);
        let dup_len = min_feature_size(sa, sb);
        // 위 SAT 경로에서 만든 행렬을 그대로 쓴다. 자세는 그 사이에 바뀌지 않는다(같은 호출의
        // 같은 `pa`/`pb`다). 박스 쌍인데 위 블록을 안 지났을 수는 없지만, 없으면 만든다.
        let (ma, mb) = match &box_mats {
            Some(m) => (m.0, m.1),
            None => (Mat3::from_quat(pa.rot), Mat3::from_quat(pb.rot)),
        };
        if let Some(m) = clip_box_box(
            char_len, dup_len, *ha, *ra_box, pa, &ma, *hb, *rb_box, pb, &mb, normal, margin,
        ) {
            return Some(m);
        }
    }

    // **일반 볼록 면 접촉** — hull(·박스) 쌍의 다접점 매니폴드. 박스-박스는 위 전용
    // 경로가 이미 처리했고(비트 동일 보장) 여기서도 한 번 더 걸러진다. 면을 못 내거나
    // 에지-에지 지배면 None이라 아래 witness 단일점으로 떨어진다 — 기존 동작 그대로다.
    if let Some(m) = genclip::convex_face_manifold(sa, pa, sb, pb, normal, margin) {
        return Some(m);
    }

    // 폴백: witness 단일점 (radius 재팽창 — 설계 shrink-then-reinflate의 reinflate 지점)
    let surface_a = on_a - normal * ra; // A 표면: 코어점에서 −normal(=B쪽)로 radius만큼
    let surface_b = on_b + normal * rb;
    Some(Manifold::from_points(
        normal,
        &[ContactPoint {
            position: (surface_a + surface_b) * 0.5,
            separation: core_sep,
            // witness 단일점: 계보가 없다(곡면·에지-에지). 상위 비트로 **generic 표식**을 달아
            // 솔버가 ID 대신 위치 근접으로 warm start를 매칭하게 한다(리뷰 MAJOR).
            feature: FeatureId(FEATURE_GENERIC),
        }],
    ))
}
