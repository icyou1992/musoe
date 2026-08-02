//! **분리축 탐색(SAT)** 과 해석적 쌍 — 어느 축으로 얼마나 떨어져 있나.
//!
//! 접점을 만드는 일은 `clip`이 한다. 여기는 "어느 면/에지가 참조면인가"까지다.
//! 구-구·구-박스는 해석해가 있어 SAT를 거치지 않는다.

use super::*;

pub(super) enum SatResult {
    /// 여유(margin)를 넘어 분리 — 접촉 없음.
    Apart,
    /// A를 +normal로 밀면 분리되는 축.
    Axis(Vec3),
    /// 축이 퇴화(평행 에지 등) — 호출자가 GJK로 폴백.
    Degenerate,
}

/// 면 축과 에지 축의 **근소 동률**을 깨는 허용치 (특성 길이 대비).
///
/// **이 편향이 없으면 정지 스택이 무너진다**: 면 접촉인데 수치 오차로 에지 축이 근소하게
/// 이기면 스텝마다 참조면이 바뀌어 feature ID가 요동치고 warm start가 끊긴다.
/// M2에서 SAT를 "실측 회귀"로 기각했던 실제 원인이 이것이었다 — 결론이
/// "SAT를 쓰지 말자"가 아니라 "동률을 깨자"였어야 했다.
///
/// **비율 편향(예: 5%)은 안 된다**: 진짜 에지 지배 접촉까지 면으로 끌어당겨 법선이 틀리고
/// 옆으로 미는 힘이 생긴다(40단 피라미드에서 z축 잔류속도 0.2 m/s로 실측됐다).
/// 동률 구간만 덮는 **작은 절대 허용치**여야 한다 — Box2D·Jolt도 같은 형태다.
pub(super) const FACE_TIE_REL: f32 = 1.0e-3;
/// 정렬 키의 동률 폭 (특성 길이 대비 — 절대값이면 장면 스케일에 따라 진입 여부가 달라진다).
pub(super) const ALIGN_TIE_REL: f32 = 1.0e-6;

/// 박스-박스 분리축 (코어 기준). 반환 법선은 **A를 +normal로 밀면 분리**되는 방향이다.
#[allow(clippy::too_many_arguments)]
pub(super) fn box_box_axis(
    ha: Vec3,
    ra: f32,
    pa: &Pose,
    ma: &Mat3,
    hb: Vec3,
    rb: f32,
    pb: &Pose,
    mb: &Mat3,
    margin: f32,
) -> SatResult {
    let core =
        |h: Vec3, r: f32| Vec3::new((h.x - r).max(0.0), (h.y - r).max(0.0), (h.z - r).max(0.0));
    let (ca, cb) = (core(ha, ra), core(hb, rb));
    // 회전행렬의 열 = 회전된 단위축. `Mat3::from_quat`이 정확히 `rot.rotate(e_i)`를 계산하므로
    // 이 대입은 **비트 동일**이다(값이 아니라 계산 위치만 옮긴다).
    let axes_a = ma.c;
    let axes_b = mb.c;
    let d = pb.pos - pa.pos;
    let extent = |axes: &[Vec3; 3], c: Vec3, n: Vec3| -> f32 {
        (axes[0].dot(n) * c.x).abs() + (axes[1].dot(n) * c.y).abs() + (axes[2].dot(n) * c.z).abs()
    };
    // 코어 표면이 여유 안에 있어야 접촉이다: core_sep - ra - rb <= margin
    let reach = margin + ra + rb;
    // 평행 에지 판정은 **스케일 무관**하다 — 단위 축의 cross 길이는 sin(각)이다.
    const PARALLEL_SIN2: f32 = 1.0e-8;

    // 축 후보를 **세 부류로 따로** 추적한 뒤 마지막에 동률을 깬다: A면 / B면 / 에지.
    // 부류 안에서는 최대값, 부류 사이에서는 허용치를 넘어야 승계된다.
    let (mut a_sep, mut a_axis) = (f32::NEG_INFINITY, Vec3::ZERO);
    let (mut b_sep, mut b_axis) = (f32::NEG_INFINITY, Vec3::ZERO);
    let (mut edge_sep, mut edge_axis) = (f32::NEG_INFINITY, Vec3::ZERO);
    // 방향 규칙: A를 +normal로 밀면 분리된다.
    //
    // `d·n ≈ 0`(법선이 중심 방향에 수직)이면 `d`만으로는 방향이 정해지지 않고, 그대로 두면
    // A/B 교환에서 부호가 뒤집히지 않아 물리가 삽입 순서에 의존한다.
    // **비대칭 원천은 바디 자체의 데이터**다 — 교환하면 두 바디의 (치수, 자세) 역할이
    // 뒤바뀌므로, 그것으로 만든 스칼라 키의 대소가 뒤집힌다. 인덱스(월드마다 다름)가 아니라
    // 형상·자세에서 뽑으므로 소유자 무관이면서 교환 반대칭이다.
    // 두 키가 정확히 같은 경우는 구성 자체가 교환 대칭이므로 어느 쪽을 골라도 동등하다.
    // **쿼터니언 부호를 정규화한다**: `q`와 `-q`는 같은 자세인데 키가 뒤집히면, 표현만 다른
    // 동일 장면이 다른 법선을 고른다(자세는 같은데 물리가 달라진다 — 실측으로 확인했다).
    // 대표원을 골라 표현 불변으로 만든다: 마지막 비영 성분이 음수면 전체 부호를 뒤집는다.
    let canon = |q: Quat| -> [f32; 4] {
        let v = [q.x, q.y, q.z, q.w];
        let flip = v
            .iter()
            .rev()
            .find(|c| **c != 0.0)
            .is_some_and(|c| *c < 0.0);
        if flip {
            [-v[0], -v[1], -v[2], -v[3]]
        } else {
            v
        }
    };
    // 키에 **convex radius와 위치**까지 넣는다(리뷰 PARTIAL): half/rotation만 쓰면
    // `ra=0` vs `rb=0.1`인 rounded-box 쌍이나 미세하게 다른 위치가 같은 키를 가져
    // 교환에서 부호가 뒤집히지 않는다.
    let body_key = |h: Vec3, r: f32, p: &Pose| -> [f32; 11] {
        let q = canon(p.rot);
        [
            h.x, h.y, h.z, r, q[0], q[1], q[2], q[3], p.pos.x, p.pos.y, p.pos.z,
        ]
    };
    let (ka, kb) = (body_key(ha, ra, pa), body_key(hb, rb, pb));
    let tie_sign = {
        let mut s = 1.0f32;
        for i in 0..11 {
            if ka[i] != kb[i] {
                s = if ka[i] > kb[i] { 1.0 } else { -1.0 };
                break;
            }
        }
        s
    };
    let orient_tie = ALIGN_TIE_REL * ca.length().min(cb.length()).max(1.0e-6);
    let oriented = |n: Vec3| {
        let dn = d.dot(n);
        if dn.abs() > orient_tie {
            if dn < 0.0 {
                n
            } else {
                -n
            }
        } else {
            n * tie_sign
        }
    };

    for a in axes_a.iter() {
        let sep = d.dot(*a).abs() - extent(&axes_a, ca, *a) - extent(&axes_b, cb, *a);
        if sep > reach {
            return SatResult::Apart;
        }
        if sep > a_sep {
            a_sep = sep;
            a_axis = oriented(*a);
        }
    }
    for b in axes_b.iter() {
        let sep = d.dot(*b).abs() - extent(&axes_a, ca, *b) - extent(&axes_b, cb, *b);
        if sep > reach {
            return SatResult::Apart;
        }
        if sep > b_sep {
            b_sep = sep;
            b_axis = oriented(*b);
        }
    }
    for a in axes_a.iter() {
        for b in axes_b.iter() {
            let n = a.cross(*b);
            let l2 = n.length_sq();
            if l2 <= PARALLEL_SIN2 {
                continue; // 평행 에지 — 축이 퇴화한다(면 축이 이미 덮는다)
            }
            let n = n * (1.0 / l2.sqrt());
            let sep = d.dot(n).abs() - extent(&axes_a, ca, n) - extent(&axes_b, cb, n);
            if sep > reach {
                return SatResult::Apart;
            }
            if sep > edge_sep {
                edge_sep = sep;
                edge_axis = oriented(n);
            }
        }
    }
    if a_axis == Vec3::ZERO && b_axis == Vec3::ZERO {
        return SatResult::Degenerate;
    }
    let scale = ca.length().min(cb.length()).max(1.0e-6);
    let tol = FACE_TIE_REL * scale;
    // **A면을 기본으로 두고, B면·에지는 허용치를 넘어야 이긴다.**
    // 부류 안이 아니라 부류 **사이**의 동률이 실제 문제였다: 적층된 두 박스가 하중으로
    // 미세하게 흔들리면 A의 Y축과 B의 Y축 분리값이 1e-7 수준으로 엎치락뒤치락하고,
    // 그때마다 클리퍼의 참조면이 바뀌어 feature ID가 갈린다 → warm start 단절 → 지터.
    // (40단 피라미드에서 z축 잔류속도로 실측됐다. 법선 자체는 GJK/EPA와 0.44° 안에서 일치한다.)
    // **A/B 대칭 동률**(리뷰 MAJOR): "A면 우선"은 같은 두 박스의 삽입 순서를 뒤집으면
    // 다른 법선을 고르게 만든다(결정론은 유지되지만 물리가 인덱스에 의존한다).
    // 동률 구간에서는 소유자가 아니라 **법선 자체의 캐노니컬 키**로 고른다.
    // 동률 키는 **중심 방향과의 정렬**이다. `d`와 법선이 A/B 교환에서 함께 부호가 뒤집히므로
    // `|n·d|`는 교환 불변이고(대칭성), 바디가 흔들려도 매끄럽게 변한다(안정성).
    // 성분 사전식 키는 대칭이지만 **불연속**이라 근소 동률에서 축이 튀어 봉투가 28→22단으로
    // 좁아졌다(실측) — 대칭성과 안정성을 동시에 만족하는 키가 필요하다.
    let align = |n: Vec3| n.dot(d).abs();
    // `d ≈ 0`(중심 일치)이면 정렬 키가 전부 0이라 다시 "먼저 온 축"(=A)으로 퇴화한다.
    // 보조 키는 **부호 불변**이어야 한다(리뷰 NOT_RESOLVED): A/B를 교환하면 후보 법선이
    // `n → -n`으로 뒤집히는데 signed 사전식은 대소 순서가 보존되지 않아 다른 축을 고른다.
    // 성분 절대값 사전식은 교환에 불변이므로 **같은 축(직선)** 을 고르고, 방향은 `d` 규칙이
    // 정한다(d=0에서는 임의이지만 결정적이다).
    // **부호 정준 방향**으로 만든 뒤 전 성분을 비교한다(리뷰 PARTIAL): 성분 절대값만 쓰면
    // `(1,1,0)`과 `(1,-1,0)`처럼 **서로 다른 축 직선**이 같은 키를 가져 전순서가 되지 않는다.
    // 첫 비영 성분이 양수가 되도록 뒤집으면 `n`과 `-n`만 같은 키를 갖는다.
    let lex = |n: Vec3| -> (f32, f32, f32) {
        let v = [n.x, n.y, n.z];
        let flip = v.iter().find(|c| **c != 0.0).is_some_and(|c| *c < 0.0);
        if flip {
            (-v[0], -v[1], -v[2])
        } else {
            (v[0], v[1], v[2])
        }
    };
    let align_tie = ALIGN_TIE_REL * scale;
    let better = |sep_new: f32, n_new: Vec3, sep_cur: f32, n_cur: Vec3| -> bool {
        if sep_new > sep_cur + tol {
            return true;
        }
        if sep_cur > sep_new + tol {
            return false;
        }
        let (an, ac) = (align(n_new), align(n_cur));
        if an > ac + align_tie {
            return true;
        }
        if ac > an + align_tie {
            return false;
        }
        lex(n_new) > lex(n_cur)
    };
    let (mut best_sep, mut best_axis) = (a_sep, a_axis);
    if b_axis != Vec3::ZERO
        && (best_axis == Vec3::ZERO || better(b_sep, b_axis, best_sep, best_axis))
    {
        best_sep = b_sep;
        best_axis = b_axis;
    }
    // 에지 축은 면 축과 **동률이면 면을 유지**한다(정지 스택 안정성 — 위 주석).
    if edge_axis != Vec3::ZERO && edge_sep > best_sep + tol {
        best_axis = edge_axis;
    }
    SatResult::Axis(best_axis)
}

/// 구-구 해석 접촉. 중심이 겹친 퇴화는 결정적 축(+Y)으로 분리한다.
pub(super) fn sphere_sphere(ra: f32, ca: Vec3, rb: f32, cb: Vec3, margin: f32) -> Option<Manifold> {
    let d = cb - ca;
    let dist2 = d.length_sq();
    let sum = ra + rb;
    let reach = sum + margin;
    if dist2 > reach * reach {
        return None;
    }
    let dist = dist2.sqrt();
    // A를 +normal로 밀면 분리 (musoe-collide 관례) → normal은 B에서 A로 향한다
    let normal = if dist > 0.0 {
        -d * (1.0 / dist)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let surface_a = ca - normal * ra;
    let surface_b = cb + normal * rb;
    Some(Manifold::from_points(
        normal,
        &[ContactPoint {
            position: (surface_a + surface_b) * 0.5,
            separation: dist - sum,
            feature: FeatureId(FEATURE_ANALYTIC),
        }],
    ))
}

/// 구-박스 해석 경로.
///
/// GJK+EPA를 돌리면 페어당 6.0us지만(실측) 해석해는 30 flops 남짓이다: 구 중심을 박스
/// 로컬로 옮기고 half-extent로 클램프하면 박스 코어 위의 최근접점이 나온다.
/// 정확도도 낫다 — 반복 수렴 오차가 없고 접점이 스텝 간 안정적이다.
///
/// `sphere_is_a`: 호출부의 A가 구면 true. 법선 부호 계약이 "A를 +normal로 밀면 분리"라
/// A가 어느 쪽인지 알아야 한다.
pub(super) fn sphere_box(
    rs: f32,
    cs: Vec3,
    half: Vec3,
    rbox: f32,
    pbox: &Pose,
    margin: f32,
    sphere_is_a: bool,
) -> Option<Manifold> {
    let sum = rs + rbox;
    let reach = sum + margin;
    // 구 중심을 박스 로컬 프레임으로 (자세는 단위 쿼터니언 — 입력 검증이 보장한다)
    let l = pbox.rot.conjugate().rotate(cs - pbox.pos);
    let cl = Vec3::new(
        l.x.clamp(-half.x, half.x),
        l.y.clamp(-half.y, half.y),
        l.z.clamp(-half.z, half.z),
    );
    let d = l - cl;
    let d2 = d.length_sq();
    // **조기 기각은 제곱 비교로** — 안 닿는 페어가 대다수인데 sqrt까지 갈 이유가 없다.
    if d2 > reach * reach {
        return None;
    }
    // 스케일 의존 epsilon (설계 §8 — 절대 상수는 스케일 버그다). d2는 길이²이므로 L².
    let scale = half.x.max(half.y).max(half.z).max(rs);
    let eps2 = crate::tolerance::length_sq_eps(scale);

    // 중심이 박스 코어 **밖**: 최근접점에서 중심으로 가는 방향이 법선이다.
    // `core_pt`는 법선 방향으로 잰 박스 **코어** 표면 위의 접점(겉면은 여기서 rbox 만큼 밖).
    let (n_local, sep, core_pt) = if d2 > eps2 {
        let dist = d2.sqrt();
        (d * (1.0 / dist), dist - sum, cl)
    } else {
        // 중심이 코어 **안**(또는 면 위): 가장 얕은 면으로 밀어낸다.
        // 동률은 낮은 축 인덱스, 축 위 0은 양의 방향 — 결정적 타이브레이크.
        let depth = [half.x - l.x.abs(), half.y - l.y.abs(), half.z - l.z.abs()];
        let mut axis = 0usize;
        for (i, &v) in depth.iter().enumerate().skip(1) {
            if v < depth[axis] {
                axis = i;
            }
        }
        let comp = [l.x, l.y, l.z][axis];
        let sign = if comp >= 0.0 { 1.0 } else { -1.0 };
        let mut n = Vec3::ZERO;
        match axis {
            0 => n.x = sign,
            1 => n.y = sign,
            _ => n.z = sign,
        }
        // 겉면 기준 침투: 면까지 depth + 구 반지름 + 박스 convex radius
        (n, -(depth[axis] + sum), l + n * depth[axis])
    };
    if sep > margin {
        return None;
    }
    // 겉면 위의 두 점 (로컬) → 월드
    let surf_box_local = core_pt + n_local * rbox;
    let surf_sphere_local = l - n_local * rs;
    let mid_local = (surf_box_local + surf_sphere_local) * 0.5;
    let position = pbox.pos + pbox.rot.rotate(mid_local);
    // n_local은 박스 → 구 방향. A가 구면 그대로, A가 박스면 뒤집는다.
    let n_world = pbox.rot.rotate(n_local);
    let normal = if sphere_is_a { n_world } else { -n_world };
    Some(Manifold::from_points(
        normal,
        &[ContactPoint {
            position,
            separation: sep,
            feature: FeatureId(FEATURE_ANALYTIC),
        }],
    ))
}

// ---------- box face 유틸 ----------
