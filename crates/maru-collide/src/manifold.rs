//! 접촉 매니폴드 생성 — face clipping + canonical feature ID (설계 §4.2).
//!
//! - GJK/EPA가 준 접촉 법선으로 reference/incident 면을 고르고(동률=낮은 면 인덱스,
//!   ref 소속 동률=A — 결정론), incident 면을 ref 측면 평면들로 클리핑해 최대 4점 생성.
//! - **feature ID는 단일 정본 함수**: 클리핑 계보(원본 정점/에지×평면)에서 결정적으로 부여.
//!   PCM 캐시 매칭·중복 판정 키(설계 §4)가 이 ID를 소비한다.
//! - box-box 면 접촉이 4점을 내는 것이 M1 완료 기준. 면 지배적이 아닌 접촉(에지-에지 등)과
//!   비박스 쌍은 witness 단일점(v1 명시 범위 — hull 면 클리핑은 hull 토폴로지 빌더와 함께).

use crate::epa::epa;
use crate::gjk::{gjk, GjkResult};
use crate::tolerance::{area_eps, characteristic_length};
use crate::{Pose, Shape};
use maru_math::{Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct FeatureId(pub u32);

#[derive(Clone, Copy, Debug, Default)]
pub struct ContactPoint {
    /// 월드 좌표 (B 표면 기준).
    pub position: Vec3,
    /// 부호 있는 간격: 음수 = 침투. (겉면 기준 — box는 half가 겉면)
    pub separation: f32,
    pub feature: FeatureId,
}

/// 매니폴드 접점 상한. 클리핑이 `reduce_to_4`로 4점을 넘기지 않고, 해석 경로는 1점이다.
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// 접촉 매니폴드.
///
/// 접점을 **인라인 배열**로 담는다(리뷰 MAJOR: 스텝 중 힙 할당). `Vec`이면 접촉마다
/// 할당이 하나씩 생기는데, 3000구 더미의 5,538 접촉이면 스텝당 수천 회다 —
/// 5,538 접촉 규모에서는 산술보다 allocator·포인터 추적이 지배할 수 있다.
#[derive(Clone, Copy, Debug)]
pub struct Manifold {
    /// A를 +normal로 밀면 분리 (GJK/EPA 관례 승계).
    pub normal: Vec3,
    points: [ContactPoint; MAX_MANIFOLD_POINTS],
    count: u8,
}

impl Default for Manifold {
    fn default() -> Self {
        Self {
            normal: Vec3::ZERO,
            points: [ContactPoint::default(); MAX_MANIFOLD_POINTS],
            count: 0,
        }
    }
}

impl Manifold {
    #[inline]
    pub fn points(&self) -> &[ContactPoint] {
        &self.points[..self.count as usize]
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    #[inline]
    pub fn push(&mut self, p: ContactPoint) {
        if (self.count as usize) < MAX_MANIFOLD_POINTS {
            self.points[self.count as usize] = p;
            self.count += 1;
        }
    }
    /// 테스트·조정용 가변 접근.
    #[inline]
    pub fn points_mut(&mut self) -> &mut [ContactPoint] {
        &mut self.points[..self.count as usize]
    }
    pub fn from_points(normal: Vec3, pts: &[ContactPoint]) -> Self {
        let mut m = Self {
            normal,
            ..Default::default()
        };
        for p in pts {
            m.push(*p);
        }
        m
    }
}

/// 면 지배 판정 임계: |면법선·접촉법선| 이 이 값 이상이면 face contact로 취급.
const FACE_DOT_MIN: f32 = 0.95;

/// 계보 없는 witness 접점 표식. 클리핑 접점의 feature 인코딩(상위 비트=ref_is_a, 면·계보)과
/// 겹치지 않는 예약값이며, 솔버는 이 값을 보면 **위치 근접**으로 캐시를 매칭한다.
pub const FEATURE_GENERIC: u32 = 0xFFFF_0000;
/// 전용(analytic) 경로가 만든 **안정적 단일 접점** ID. generic과 달리 스텝 간 동일 접촉이
/// 같은 ID를 유지하므로 솔버가 위치 근접 게이트 없이 ID로 warm start를 잇는다.
pub const FEATURE_ANALYTIC: u32 = 0xFFFE_0000;

/// 클리핑 정점의 계보 (리뷰 MAJOR 반영 — 구 태그 재사용 인코딩은 다단 클리핑에서 서로 다른
/// 계보가 같은 태그로 충돌했다): incident 원본 정점 비트합 ∪ 관여한 reference 측면 비트합.
#[derive(Clone, Copy)]
struct ClipVertex {
    position: Vec3,
    incident_mask: u8,
    reference_mask: u8,
}

/// 접촉 생성 진입점. `margin` = speculative margin (양의 separation 허용 한계).
pub fn contact(sa: &Shape, pa: &Pose, sb: &Shape, pb: &Pose, margin: f32) -> Option<Manifold> {
    // **전용 경로 우선**: 구-구는 해석적으로 정확하고 5 flops면 끝난다. 이걸 GJK+EPA로
    // 돌리면 페어당 마이크로초가 든다(대조 실측: 내로우페이즈가 스텝의 58%).
    // 정확도도 전용 경로가 낫다 — 반복 수렴 오차가 없고 접점 ID가 스텝 간 안정적이라
    // warm start가 위치 근접 게이트에 의존하지 않는다.
    if let (Shape::Sphere { radius: ra }, Shape::Sphere { radius: rb }) = (sa, sb) {
        return sphere_sphere(*ra, pa.pos, *rb, pb.pos, margin);
    }
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
        match box_box_axis(*ha, *rab, pa, *hb, *rbb, pb, margin) {
            SatResult::Apart => return None,
            SatResult::Axis(n) => {
                let eps_area = area_eps(characteristic_length(sa, sb));
                if let Some(m) = clip_box_box(eps_area, *ha, *rab, pa, *hb, *rbb, pb, n, margin) {
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
        let eps_area = area_eps(characteristic_length(sa, sb));
        if let Some(m) = clip_box_box(eps_area, *ha, *ra_box, pa, *hb, *rb_box, pb, normal, margin)
        {
            return Some(m);
        }
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

enum SatResult {
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
const FACE_TIE_REL: f32 = 1.0e-3;
/// 정렬 키의 동률 폭 (특성 길이 대비 — 절대값이면 장면 스케일에 따라 진입 여부가 달라진다).
const ALIGN_TIE_REL: f32 = 1.0e-6;

/// 박스-박스 분리축 (코어 기준). 반환 법선은 **A를 +normal로 밀면 분리**되는 방향이다.
#[allow(clippy::too_many_arguments)]
fn box_box_axis(
    ha: Vec3,
    ra: f32,
    pa: &Pose,
    hb: Vec3,
    rb: f32,
    pb: &Pose,
    margin: f32,
) -> SatResult {
    let core =
        |h: Vec3, r: f32| Vec3::new((h.x - r).max(0.0), (h.y - r).max(0.0), (h.z - r).max(0.0));
    let (ca, cb) = (core(ha, ra), core(hb, rb));
    let axes_a = [
        pa.rot.rotate(Vec3::new(1.0, 0.0, 0.0)),
        pa.rot.rotate(Vec3::new(0.0, 1.0, 0.0)),
        pa.rot.rotate(Vec3::new(0.0, 0.0, 1.0)),
    ];
    let axes_b = [
        pb.rot.rotate(Vec3::new(1.0, 0.0, 0.0)),
        pb.rot.rotate(Vec3::new(0.0, 1.0, 0.0)),
        pb.rot.rotate(Vec3::new(0.0, 0.0, 1.0)),
    ];
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
fn sphere_sphere(ra: f32, ca: Vec3, rb: f32, cb: Vec3, margin: f32) -> Option<Manifold> {
    let d = cb - ca;
    let dist2 = d.length_sq();
    let sum = ra + rb;
    let reach = sum + margin;
    if dist2 > reach * reach {
        return None;
    }
    let dist = dist2.sqrt();
    // A를 +normal로 밀면 분리 (maru-collide 관례) → normal은 B에서 A로 향한다
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

// ---------- box face 유틸 ----------

/// 면 f = axis*2 + (양의 방향이면 1). 월드 법선.
fn box_face_normal(f: usize, rot: Quat) -> Vec3 {
    let axis = f / 2;
    let sign = if f % 2 == 1 { 1.0 } else { -1.0 };
    let local = match axis {
        0 => Vec3::new(sign, 0.0, 0.0),
        1 => Vec3::new(0.0, sign, 0.0),
        _ => Vec3::new(0.0, 0.0, sign),
    };
    rot.rotate(local)
}

/// 면 f의 4개 꼭짓점 (로컬, 바깥에서 볼 때 CCW — 결정적 순서).
/// rounded box(리뷰 MAJOR 반영): 실제 평면 패치는 접선 방향으로 radius만큼 수축되고
/// 법선 방향은 겉면(half)에 놓인다 — 겉 코너를 접점으로 쓰면 곡면 밖 점이 솔버로 간다.
fn box_face_verts(f: usize, half: Vec3, radius: f32) -> [Vec3; 4] {
    let axis = f / 2;
    let s = if f % 2 == 1 { 1.0 } else { -1.0 };
    let core = Vec3::new(
        (half.x - radius).max(0.0),
        (half.y - radius).max(0.0),
        (half.z - radius).max(0.0),
    );
    let (h, u, v) = match axis {
        0 => (
            core.x,
            Vec3::new(0.0, core.y, 0.0),
            Vec3::new(0.0, 0.0, core.z),
        ),
        1 => (
            core.y,
            Vec3::new(0.0, 0.0, core.z),
            Vec3::new(core.x, 0.0, 0.0),
        ),
        _ => (
            core.z,
            Vec3::new(core.x, 0.0, 0.0),
            Vec3::new(0.0, core.y, 0.0),
        ),
    };
    let n = match axis {
        0 => Vec3::new(s, 0.0, 0.0),
        1 => Vec3::new(0.0, s, 0.0),
        _ => Vec3::new(0.0, 0.0, s),
    };
    let c = n * (h + radius);
    // s<0이면 u,v 순서를 바꿔 CCW 유지
    if s > 0.0 {
        [c + u + v, c - u + v, c - u - v, c + u - v]
    } else {
        [c + u + v, c + u - v, c - u - v, c - u + v]
    }
}

/// 법선과 가장 정렬된 면 선택 (동률: 낮은 면 인덱스 — 결정론). 반환 (face, dot).
fn best_face(rot: Quat, want: Vec3) -> (usize, f32) {
    let mut best = 0;
    let mut best_dot = -f32::INFINITY;
    for f in 0..6 {
        let d = box_face_normal(f, rot).dot(want);
        if d > best_dot + 1.0e-6 {
            best_dot = d;
            best = f;
        }
    }
    (best, best_dot)
}

/// box-box 면 클리핑. 법선이 면 지배적이 아니면 None(→ witness 폴백).
#[allow(clippy::too_many_arguments)]
fn clip_box_box(
    eps_area: f32,
    ha: Vec3,
    ra: f32,
    pa: &Pose,
    hb: Vec3,
    rb: f32,
    pb: &Pose,
    normal: Vec3,
    margin: f32,
) -> Option<Manifold> {
    // A에서 B를 향하는 면(법선 ≈ −normal), B에서 A를 향하는 면(≈ +normal)
    let (fa, da) = best_face(pa.rot, -normal);
    let (fb, db) = best_face(pb.rot, normal);
    if da.max(db) < FACE_DOT_MIN {
        return None; // 에지-에지 지배 — witness 단일점 경로
    }
    // reference = 더 정렬된 쪽 (동률 = A — 결정론)
    let ref_is_a = da >= db;
    let (rf, rh, rr, rp, ih, ir, ip) = if ref_is_a {
        (fa, ha, ra, pa, hb, rb, pb)
    } else {
        (fb, hb, rb, pb, ha, ra, pa)
    };

    let rn = box_face_normal(rf, rp.rot); // reference 바깥 법선 (월드)
    let rverts: Vec<Vec3> = box_face_verts(rf, rh, rr)
        .iter()
        .map(|v| rp.pos + rp.rot.rotate(*v))
        .collect();
    // incident = 상대 박스에서 rn과 가장 반대인 면
    let (inc_f, _) = best_face(ip.rot, -rn);
    let iverts: Vec<Vec3> = box_face_verts(inc_f, ih, ir)
        .iter()
        .map(|v| ip.pos + ip.rot.rotate(*v))
        .collect();
    // radius가 half를 잠식해 평면 패치가 선/점으로 퇴화하면 클리핑 불가 → witness 폴백
    // cross 길이 = 면적(L²) → L² 임계와 비교 (length_sq는 L⁴)
    if (rverts[1] - rverts[0])
        .cross(rverts[2] - rverts[1])
        .length()
        <= eps_area
        || (iverts[1] - iverts[0])
            .cross(iverts[2] - iverts[1])
            .length()
            <= eps_area
    {
        return None;
    }

    // Sutherland–Hodgman: incident 폴리곤을 reference 4 측면 평면으로 클리핑 (계보 추적)
    let mut poly: Vec<ClipVertex> = iverts
        .iter()
        .enumerate()
        .map(|(i, v)| ClipVertex {
            position: *v,
            incident_mask: 1u8 << i,
            reference_mask: 0,
        })
        .collect();
    for side in 0..4 {
        let e0 = rverts[side];
        let e1 = rverts[(side + 1) % 4];
        // 측면 평면: 법선 = rn × edge (안쪽을 향하도록)
        let mut pn = rn.cross(e1 - e0);
        let center = (rverts[0] + rverts[1] + rverts[2] + rverts[3]) * 0.25;
        if pn.dot(center - e0) < 0.0 {
            pn = -pn;
        }
        let mut out: Vec<ClipVertex> = Vec::with_capacity(8);
        for i in 0..poly.len() {
            let v0 = poly[i];
            let v1 = poly[(i + 1) % poly.len()];
            let d0 = pn.dot(v0.position - e0);
            let d1 = pn.dot(v1.position - e0);
            if d0 >= 0.0 {
                out.push(v0);
            }
            // 교점은 **엄격한 부호 반대**일 때만 생성 (확인 라운드 MAJOR 반영 — d0==0인 경계
            // 정점은 위에서 이미 보존되므로 t=0 교점까지 만들면 같은 위치가 서로 다른 계보
            // 마스크로 중복 진입해 솔버가 동일 지점에 제약을 중복 적용한다).
            if (d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0) {
                let t = d0 / (d0 - d1);
                let x = v0.position + (v1.position - v0.position) * t;
                out.push(ClipVertex {
                    position: x,
                    incident_mask: v0.incident_mask | v1.incident_mask,
                    reference_mask: v0.reference_mask | v1.reference_mask | (1u8 << side),
                });
            }
        }
        poly = out;
        if poly.is_empty() {
            return None;
        }
    }

    // reference 평면 아래(margin 이내)만 채택, separation = 평면까지 부호 거리
    let mut pts: Vec<ContactPoint> = Vec::with_capacity(8);
    for v in poly {
        let sep = rn.dot(v.position - rverts[0]);
        if sep <= margin {
            // 접점 위치는 reference 면 위로 투영한 점과 incident 점의 중간
            let proj = v.position - rn * sep;
            let feature = ((ref_is_a as u32) << 31)
                | ((rf as u32) << 28)
                | ((inc_f as u32) << 25)
                | ((v.reference_mask as u32) << 4)
                | v.incident_mask as u32;
            pts.push(ContactPoint {
                position: (v.position + proj) * 0.5,
                separation: sep,
                feature: FeatureId(feature),
            });
        }
    }
    if pts.is_empty() {
        return None;
    }

    // 4점 축소 (설계 §4.2): 최심점 고정 + 나머지는 면적 최대 조합(그리디, 결정적)
    if pts.len() > 4 {
        pts = reduce_to_4(pts, rn);
    }
    // manifold 법선은 push-A 관례로: reference가 A면 rn은 A→B → 뒤집기
    let m_normal = if ref_is_a { -rn } else { rn };
    Some(Manifold::from_points(m_normal, &pts))
}

/// 최심점 고정 → 나머지 3점 전수 조합으로 접촉 사각형 면적 최대 선택 (설계 §4.2 원문,
/// 리뷰 MAJOR 반영 — 구 그리디(최원점 선고정)는 면적 최대 조합을 보장하지 않았다).
/// pts는 클리핑 다각형의 순환 순서 — 인덱스 정렬로 부분집합의 순환 순서를 보존해 면적 계산.
fn reduce_to_4(pts: Vec<ContactPoint>, n: Vec3) -> Vec<ContactPoint> {
    use core::cmp::Ordering;
    let deepest = pts
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.separation
                .total_cmp(&b.separation)
                .then(a.feature.cmp(&b.feature))
        })
        .map(|(i, _)| i)
        .unwrap();
    let candidates: Vec<usize> = (0..pts.len()).filter(|i| *i != deepest).collect();
    let mut best_indices = [deepest, candidates[0], candidates[1], candidates[2]];
    let mut best_area = -1.0f32;
    let mut best_key = [FeatureId(u32::MAX); 4];

    for a in 0..(candidates.len() - 2) {
        for b in (a + 1)..(candidates.len() - 1) {
            for c in (b + 1)..candidates.len() {
                let mut indices = [deepest, candidates[a], candidates[b], candidates[c]];
                indices.sort_unstable(); // 순환 순서 보존
                let origin = pts[indices[0]].position;
                let twice_area = (pts[indices[1]].position - origin)
                    .cross(pts[indices[2]].position - origin)
                    .dot(n)
                    + (pts[indices[2]].position - origin)
                        .cross(pts[indices[3]].position - origin)
                        .dot(n);
                let area = twice_area.abs();
                let mut key = [
                    pts[indices[0]].feature,
                    pts[indices[1]].feature,
                    pts[indices[2]].feature,
                    pts[indices[3]].feature,
                ];
                key.sort_unstable();
                let better = match area.total_cmp(&best_area) {
                    Ordering::Greater => true,
                    Ordering::Equal => key < best_key,
                    Ordering::Less => false,
                };
                if better {
                    best_area = area;
                    best_indices = indices;
                    best_key = key;
                }
            }
        }
    }

    let mut out: Vec<ContactPoint> = best_indices.iter().map(|&i| pts[i]).collect();
    out.sort_by_key(|p| p.feature);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(x: f32, y: f32, z: f32) -> Pose {
        Pose {
            pos: Vec3::new(x, y, z),
            rot: Quat::IDENTITY,
        }
    }

    /// M1 완료 기준: box 면 접촉이 4점 매니폴드를 낸다.
    #[test]
    fn box_on_box_face_gives_4_points() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        // top이 ground 위에 딱 얹힘 (y: 1+1=2)
        let m = contact(
            &top,
            &pose(0.0, 2.0, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .expect("contact expected");
        assert_eq!(m.points().len(), 4, "{:?}", m.points);
        for p in m.points() {
            assert!(p.separation.abs() < 1e-4, "{:?}", p);
        }
        assert!(
            (m.normal.y - 1.0).abs() < 1e-4,
            "top을 +y로 밀면 분리: {:?}",
            m.normal
        );
        // 4점의 feature는 서로 달라야 함
        let mut feats: Vec<_> = m.points().iter().map(|p| p.feature).collect();
        feats.sort_unstable();
        feats.dedup();
        assert_eq!(feats.len(), 4);
    }

    /// M1 완료 기준: feature ID 안정성 — 미세 섭동에도 같은 feature 집합.
    #[test]
    fn feature_ids_stable_under_perturbation() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let base = contact(
            &top,
            &pose(0.0, 1.999, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        let perturbed = contact(
            &top,
            &pose(1.0e-4, 1.999, -2.0e-4),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        let f0: Vec<_> = base.points.iter().map(|p| p.feature).collect();
        let f1: Vec<_> = perturbed.points.iter().map(|p| p.feature).collect();
        assert_eq!(f0, f1, "feature 집합이 섭동에 흔들림");
    }

    /// 회전 슬라이드 케이스: yaw 회전한 박스가 얹혀도 ≤4점 + 전부 접촉권.
    #[test]
    fn rotated_box_on_ground_clipped() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.6);
        let pa = Pose {
            pos: Vec3::new(0.0, 1.995, 0.0),
            rot,
        };
        let m = contact(&top, &pa, &ground, &pose(0.0, 0.0, 0.0), 0.05).unwrap();
        assert!(
            !m.points.is_empty() && m.points().len() <= 4,
            "{:?}",
            m.points().len()
        );
        for p in m.points() {
            assert!(p.separation < 0.05);
        }
    }

    /// 침투 케이스: EPA 경로에서도 클리핑 4점 + 음의 separation.
    #[test]
    fn penetrating_face_contact() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let m = contact(
            &top,
            &pose(0.0, 1.9, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        assert_eq!(m.points().len(), 4);
        for p in m.points() {
            assert!((p.separation + 0.1).abs() < 1e-3, "sep={}", p.separation);
        }
    }

    /// 확인 라운드 MAJOR 회귀: 경계 정점이 클리핑 평면 위에 정확히 놓여도 중복 접점 금지.
    #[test]
    fn boundary_vertex_no_duplicate_points() {
        // incident 꼭짓점이 reference 측면 평면(x=1) 위에 정확히 놓이는 45° 구성
        let ground = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let top = Shape::Box {
            half: Vec3::new(
                core::f32::consts::FRAC_1_SQRT_2,
                1.0,
                core::f32::consts::FRAC_1_SQRT_2,
            ),
            radius: 0.0,
        };
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_4);
        let pa = Pose {
            pos: Vec3::new(1.0, 2.0, 0.0),
            rot,
        };
        let m = contact(&top, &pa, &ground, &pose(0.0, 0.0, 0.0), 0.05).unwrap();
        assert!(!m.points.is_empty() && m.points().len() <= 4);
        for (i, p) in m.points().iter().enumerate() {
            for q in m.points().iter().skip(i + 1) {
                let d = (p.position - q.position).length_sq();
                assert!(d > 1.0e-8, "중복 접점: {p:?} vs {q:?}");
            }
        }
    }

    /// 리뷰 요구: **metamorphic 스케일 테스트**. 같은 씬을 1e-3~1e2배로 변환해도
    /// 접촉 판정·법선·상대 침투가 유지되어야 한다(절대 임계가 남아 있으면 여기서 깨진다).
    #[test]
    fn sat_normal_agrees_with_gjk_epa() {
        // 정지 스택에서 실제로 나타나는 배치(축정렬 + 미세 회전)를 결정적으로 흔들며
        // SAT 법선과 GJK/EPA 법선의 각도차를 잰다. 두 경로가 다른 축을 고르면
        // 참조면이 바뀌어 feature ID가 요동치고 warm start가 끊긴다.
        let half = Vec3::new(0.5, 0.5, 0.5);
        let mut rng: u32 = 0x1234_5678;
        let mut next = || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5 // [-0.5, 0.5)
        };
        let (mut worst_deg, mut disagree, mut total) = (0.0f32, 0, 0);
        for case in 0..600 {
            // 수직 적층 / 수평 인접 / 대각 이웃을 번갈아
            let base = match case % 3 {
                0 => Vec3::new(0.0, 1.0, 0.0),
                1 => Vec3::new(1.02, 0.0, 0.0),
                _ => Vec3::new(0.51, 1.0, 0.0),
            };
            let jitter = Vec3::new(next(), next(), next()) * 0.02;
            let mut tilt = |a: f32, b: f32, c: f32| {
                Quat::from_axis_angle(Vec3::new(a, b, c).normalize_or_zero(), 0.01 * next())
            };
            let pa = Pose {
                pos: Vec3::ZERO,
                rot: tilt(1.0, 0.0, 0.0),
            };
            let pb = Pose {
                pos: base + jitter,
                rot: tilt(0.0, 0.0, 1.0),
            };
            let sat = match box_box_axis(half, 0.0, &pa, half, 0.0, &pb, 0.05) {
                SatResult::Axis(n) => n,
                _ => continue,
            };
            let sa = Shape::Box { half, radius: 0.0 };
            let reference = match gjk(&sa, &pa, &sa, &pb) {
                GjkResult::Separated { normal, .. } => normal,
                GjkResult::Intersecting(seed) => epa(seed, &sa, &pa, &sa, &pb).normal,
            };
            total += 1;
            let cos = sat.dot(reference).clamp(-1.0, 1.0);
            let deg = cos.acos().to_degrees();
            if deg > worst_deg {
                worst_deg = deg;
            }
            if deg > 1.0 {
                disagree += 1;
            }
        }
        println!("SAT vs GJK/EPA: 표본 {total}, 1° 초과 불일치 {disagree}, 최대 {worst_deg:.2}°");
        assert!(
            disagree == 0,
            "SAT 법선이 {disagree}/{total} 케이스에서 기존 경로와 다르다 (최대 {worst_deg:.2}°)"
        );
    }

    #[test]
    fn contact_is_scale_invariant() {
        for &k in &[1.0e-3f32, 1.0e-1, 1.0, 1.0e1, 1.0e2] {
            let top = Shape::Box {
                half: Vec3::new(k, k, k),
                radius: 0.0,
            };
            let ground = Shape::Box {
                half: Vec3::new(4.0 * k, k, 4.0 * k),
                radius: 0.0,
            };
            // 상대 침투 1% 로 동일하게 구성
            let m = contact(
                &top,
                &pose(0.0, 1.99 * k, 0.0),
                &ground,
                &pose(0.0, 0.0, 0.0),
                0.05 * k,
            )
            .unwrap_or_else(|| panic!("scale {k}: 접촉 소실"));
            assert_eq!(
                m.points().len(),
                4,
                "scale {k}: 매니폴드 점수 {}",
                m.points().len()
            );
            assert!(
                (m.normal.y - 1.0).abs() < 1.0e-3,
                "scale {k}: 법선 {:?}",
                m.normal
            );
            for p in m.points() {
                let rel = p.separation / k;
                assert!(
                    (rel + 0.01).abs() < 0.05,
                    "scale {k}: 상대 침투 {rel} (절대 {})",
                    p.separation
                );
            }
        }
    }

    /// 리뷰 MAJOR 회귀: rounded box의 접점은 평면 패치(접선 ±(half−radius)) 안에 있어야 한다.
    #[test]
    fn rounded_box_face_points_stay_on_flat_patch() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.25,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let m = contact(
            &top,
            &pose(0.0, 2.0, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        assert_eq!(m.points().len(), 4);
        for p in m.points() {
            assert!(p.position.x.abs() <= 0.7501, "point={p:?}");
            assert!(p.position.z.abs() <= 0.7501, "point={p:?}");
            assert!(p.separation.abs() < 1.0e-4, "point={p:?}");
        }
    }

    /// 구-박스: witness 단일점 + radius 재팽창 separation.
    #[test]
    fn sphere_box_single_point() {
        let ball = Shape::Sphere { radius: 0.5 };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let m = contact(
            &ball,
            &pose(0.0, 1.52, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.1,
        )
        .unwrap();
        assert_eq!(m.points().len(), 1);
        assert!(
            (m.points()[0].separation - 0.02).abs() < 1e-3,
            "{:?}",
            m.points()[0]
        );
        assert!((m.normal.y - 1.0).abs() < 1e-3);
    }

    /// speculative: margin 밖이면 None, 안이면 양의 separation 접촉.
    #[test]
    fn speculative_margin_gate() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        assert!(contact(&s, &pose(0.0, 2.3, 0.0), &s, &pose(0.0, 0.0, 0.0), 0.05).is_none());
        let m = contact(&s, &pose(0.0, 2.3, 0.0), &s, &pose(0.0, 0.0, 0.0), 0.5).unwrap();
        assert!(m.points().iter().all(|p| (p.separation - 0.3).abs() < 1e-3));
    }
}
