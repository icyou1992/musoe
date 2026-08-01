//! **점 투영** — 임의의 점에서 셰이프 겉면 위의 최근접점 (rapier `project_point` 대응).
//!
//! 레이 캐스트가 "이 방향으로 쏘면 어디에 닿나"라면 이건 "이 점에서 가장 가까운 겉면이
//! 어디냐"다. 호스트는 이걸로 스냅·집기·자석 배치·최단거리 판정을 만든다.
//!
//! **반환은 항상 겉면 위 점이다** — 질의점이 셰이프 안에 있어도 그렇다. rapier의
//! `solid=true`(안에 있으면 점 자신) 의미가 필요하면 호출자가
//! `if proj.inside { p } else { proj.point }`로 만들면 된다. 반대 방향(겉면 점에서
//! solid 의미를 복원하는 것)은 불가능하므로 **더 많은 정보를 주는 쪽**을 원시 연산으로 둔다.
//!
//! 안팎 판정(`inside`)은 `contains_point`와 **같은 답**이어야 한다 — 두 질의가 갈리면
//! 호스트가 "안에 있다는데 투영은 밖을 가리킨다"를 보게 된다. 테스트가 셰이프별로 잠근다.

use crate::gjk::{gjk, GjkResult};
use crate::ray::QueryReject;
use crate::{Aabb, Pose, Shape};
use maru_math::{Quat, Vec3};

/// 방향이 없을 때 쓰는 **결정적** 축 — 임의로 고른 값이 아니라 고정 규약이다.
/// (구의 중심·박스 코어 안·평면 위 점처럼 "가장 가까운 겉면 방향"이 유일하지 않은 자리.)
const TIE_X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
const TIE_Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);

#[inline]
fn splat(v: f32) -> Vec3 {
    Vec3::new(v, v, v)
}

/// 점 투영 결과.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointProjection {
    /// 셰이프 **겉면**(convex radius 포함) 위의 최근접점 — 월드 좌표.
    pub point: Vec3,
    /// 질의점이 셰이프 안에 있었는가 (경계 위는 안으로 친다 — `contains_point`와 동일).
    pub inside: bool,
}

/// 공개 진입 — 형상·산술 입력을 검사하고 자세를 정규화한 뒤 푼다.
///
/// `try_cast_ray`·`try_contains_point`와 같은 계약이다: 잘못된 입력은 "먼 점"으로
/// 위장하지 않고 오류로 나온다.
pub fn try_project_point(
    shape: &Shape,
    pose: &Pose,
    p: Vec3,
) -> Result<PointProjection, QueryReject> {
    if !shape.is_castable() || !crate::vec_in_range(p) || !crate::vec_in_range(pose.pos) {
        return Err(QueryReject::InvalidInput);
    }
    let pose = Pose {
        pos: pose.pos,
        rot: pose.rot.try_normalize().ok_or(QueryReject::InvalidInput)?,
    };
    let local = pose.rot.conjugate().rotate(p - pose.pos);
    let proj = project_local(shape, local)?;
    Ok(PointProjection {
        point: pose.pos + pose.rot.rotate(proj.point),
        inside: proj.inside,
    })
}

/// 로컬 좌표 투영 — **검증된 입력 전용**(크레이트 내부·월드 경계 통과분).
pub(crate) fn project_local(shape: &Shape, p: Vec3) -> Result<PointProjection, QueryReject> {
    let r = shape.radius();
    match *shape {
        Shape::Sphere { radius } => Ok(inflate(Vec3::ZERO, p, radius, TIE_X)),
        Shape::Capsule {
            half_height,
            radius,
        } => {
            let c = Vec3::new(0.0, p.y.clamp(-half_height, half_height), 0.0);
            Ok(inflate(c, p, radius, TIE_X))
        }
        Shape::Box { half, radius } => Ok(project_box(half, radius, p)),
        Shape::Triangle { points, radius } => {
            let q = closest_on_triangle(p, points);
            // fallback은 **아핀 여공간**이다 — 면적이 0이면 삼각형 법선이 0이라 고정축으로
            // 마감되는데, 그 축이 퇴화한 선분과 나란할 수 있다(검수 rev_ef5d1868 MAJOR:
            // (-1,0,0)·(1,0,0)·(0,0,0)에 radius 0.1, 원점 질의가 (0.1,0,0)을 냈다 —
            // 여전히 코어 선분 안이다). 세 점의 아핀 랭크는 2 이하라 항상 여공간이 있다.
            let fallback = affine_complement_dir(&points).unwrap_or(TIE_X);
            let mut proj = inflate(q, p, radius, fallback);
            // 삼각형은 **부피가 없다** — radius가 0이면 어떤 점도 안이 아니다
            // (`contains_point`의 삼각형 분기와 같은 계약).
            proj.inside &= radius > 0.0;
            Ok(proj)
        }
        Shape::Hull { .. } => Ok(project_convex_core(shape, p, r)),
        Shape::Compound { ref children } => project_compound(children, p),
        Shape::TriMesh(ref mesh) => project_trimesh(mesh, p, r),
        Shape::Heightfield { .. } => {
            let hf = shape.heightfield_ref().expect("Heightfield 변종");
            project_heightfield(&hf, p, r)
        }
    }
}

/// 코어 위 최근접점 `q`에서 **겉면**으로 반지름만큼 밀어낸다.
///
/// `p == q`(코어 안 또는 코어 표면 위)면 방향이 없다 — 그때만 `fallback`을 쓴다.
/// 방향이 없다고 0 벡터를 내보내면 "겉면 위 점"이라는 계약이 깨진다.
#[inline]
fn inflate(q: Vec3, p: Vec3, radius: f32, fallback: Vec3) -> PointProjection {
    let v = p - q;
    let l2 = v.length_sq();
    if l2 > 0.0 {
        PointProjection {
            point: q + v * (radius / l2.sqrt()),
            inside: l2 <= radius * radius,
        }
    } else {
        PointProjection {
            point: q + fallback * radius,
            inside: true,
        }
    }
}

/// 박스 — 코어(겉면 half에서 radius를 뺀 상자)에 클램프한 뒤 부풀린다.
///
/// 코어 **안쪽** 점은 클램프가 자기 자신을 돌려주므로 방향이 없다. 그때는 축별로 코어
/// 면까지의 여유를 재서 **가장 가까운 면**으로 내보낸다(동률은 낮은 축 번호 — 결정론).
fn project_box(half: Vec3, radius: f32, p: Vec3) -> PointProjection {
    let core = Vec3::new(
        (half.x - radius).max(0.0),
        (half.y - radius).max(0.0),
        (half.z - radius).max(0.0),
    );
    let q = Vec3::new(
        p.x.clamp(-core.x, core.x),
        p.y.clamp(-core.y, core.y),
        p.z.clamp(-core.z, core.z),
    );
    if (p - q).length_sq() > 0.0 {
        return inflate(q, p, radius, TIE_X);
    }
    let pa = [p.x, p.y, p.z];
    let ca = [core.x, core.y, core.z];
    let mut axis = 0usize;
    let mut best = ca[0] - pa[0].abs();
    for (i, (c, v)) in ca.iter().zip(pa.iter()).enumerate().skip(1) {
        let gap = c - v.abs();
        if gap < best {
            best = gap;
            axis = i;
        }
    }
    // 0은 +로 민다 — 부호가 없는 축에서 결정적으로 한 쪽을 고른다.
    let sign = if pa[axis] < 0.0 { -1.0 } else { 1.0 };
    let mut out = pa;
    out[axis] = sign * (ca[axis] + radius);
    PointProjection {
        point: Vec3::new(out[0], out[1], out[2]),
        inside: true,
    }
}

/// 일반 convex 코어(hull) — 검증된 GJK/EPA를 재사용한다.
///
/// 점을 반지름 0 구로 모델링하면 점-볼록체 거리는 셰이프-셰이프 거리의 특수한 경우다
/// (`ray::closest_core_dist`와 같은 기법 — 여기서는 witness 점까지 받는다).
/// 코어 **안쪽** 점은 GJK가 교차를 주므로 EPA로 최소 이탈 방향을 구한다: 최소 관통
/// 깊이의 반대 방향이 곧 최근접 겉면 방향이다.
fn project_convex_core(shape: &Shape, p: Vec3, radius: f32) -> PointProjection {
    let point = Shape::Sphere { radius: 0.0 };
    let at = Pose {
        pos: p,
        rot: Quat::IDENTITY,
    };
    let ident = Pose {
        pos: Vec3::ZERO,
        rot: Quat::IDENTITY,
    };
    match gjk(&point, &at, shape, &ident) {
        GjkResult::Separated { distance, on_b, .. } => {
            let inside = distance <= radius;
            if radius <= 0.0 {
                // 두께가 없으면 코어가 곧 겉면이다 — 방향이 필요 없다.
                return PointProjection {
                    point: on_b,
                    inside,
                };
            }
            // **방향은 GJK 법선이 아니라 위치 차이로 만든다.**
            //
            // 코어가 평면·선처럼 낮은 차원일 수 있다(그래도 convex radius가 두께를 주므로
            // 합법 셰이프다). 그런 코어 **안**의 점에 대해 GJK는 거리를 거의 0으로 주는데,
            // 그때 법선은 반올림이 만든 **코어 평면 안** 방향일 수 있다. 그 방향으로 radius
            // 만큼 밀면 여전히 물체 안이라 "겉면 위 점" 계약이 깨진다
            // (검수 rev_6429a223 BLOCKER: z=0 사각형 hull + radius 0.1의 중심 투영이
            //  (-0.07, 0.07, 0)으로 나왔다 — 겉면은 z=±0.1이다).
            //
            // 위치 차이는 그런 착시가 없다: 코어 밖이면 진짜 바깥 방향이고, 코어 안이면
            // 길이가 0에 붙어 **여기서 걸러진다**. 그때만 이탈 방향을 따로 구한다.
            let v = p - on_b;
            let l = v.length();
            let eps = crate::length_eps(crate::characteristic_length(shape, shape));
            if l > eps {
                return PointProjection {
                    point: on_b + v * (radius / l),
                    inside,
                };
            }
            // 코어 **위/안**이라 방향이 없다 — 지지 함수로 이탈 방향과 여유를 구한다.
            let (n, gap) = core_exit(shape, p);
            PointProjection {
                point: p + n * (gap + radius),
                inside,
            }
        }
        GjkResult::Intersecting(_) => {
            // 코어 안이다. **EPA를 쓰지 않는다** — EPA는 GJK가 부피 있는 사면체 시드를
            // 만들 수 있을 때만 돌고, 그 판정은 최대 에지 길이에 **상대적**이라 얇고 긴
            // 코어(예: 반크기 (1000, 0.002, 0.002))에서는 전체 차원인데도 시드를 못 만든다.
            // 지지 함수 경로는 그 문턱이 없다.
            let (n, gap) = core_exit(shape, p);
            PointProjection {
                point: p + n * (gap + radius),
                inside: true,
            }
        }
    }
}

/// 코어 **위/안**의 점 `q`에서 나가는 방향과, 그 방향으로 코어 표면까지의 여유.
///
/// 겉면 위 점은 `q + dir * (gap + radius)`다.
///
/// 왜 GJK/EPA를 쓰지 않나: 둘 다 **상대 임계**에 걸린다. 코어가 평면·선이면 법선이
/// 코어 안 방향으로 나오고(검수 rev_6429a223), 전체 차원이어도 종횡비가 크면 EPA 시드
/// 조건(`부피 > 1e-6·최대에지³`)을 못 넘겨 관통 경로 자체가 안 열린다(rev_ef5d1868).
/// 두 실패 모두 "그럴듯하게 틀린" 값이라 0-검사로는 안 걸린다.
///
/// 대신 **지지 함수**로 푼다. 볼록체 안의 점에서 표면까지의 거리는
/// `min_n ( max_p n·(p − q) )` 이고, 최솟값을 주는 `n`이 곧 가장 가까운 면의 바깥 법선이다.
/// 임의의 `n`에 대해 `max_p n·(p − q)`는 그 방향 지지 평면까지의 거리이므로 **항상 답 이상**
/// 이다 — 후보에 진짜 면 법선만 들어 있으면 최솟값이 정확한 답이 된다. 그리고 볼록 껍질의
/// 모든 면은 껍질 점 3개가 이룬다. 임계도 분기도 없다.
///
/// 비용은 점 수의 4제곱(`C(m,3)·m`)이지만 상한이 32개로 잠겨 있고, 이 경로는 질의점이
/// 코어 **안**일 때만 돈다(밖이면 위치 차이가 방향을 준다).
fn core_exit(shape: &Shape, q: Vec3) -> (Vec3, f32) {
    let Shape::Hull { ref points, .. } = *shape else {
        // 다른 셰이프는 해석 경로가 따로 있어 여기 오지 않는다.
        return (TIE_X, 0.0);
    };
    // 코어가 저차원이면 표면까지 여유가 0이고, 방향은 아핀 껍질의 직교 여공간이다
    // (그쪽으로는 지지 평면이 무한히 많아 위 최소화가 방향을 못 고른다).
    if let Some(d) = affine_complement_dir(points) {
        return (d, 0.0);
    }
    let mut best: Option<(f32, Vec3)> = None;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            for k in (j + 1)..points.len() {
                let Some(n) = (points[j] - points[i])
                    .cross(points[k] - points[i])
                    .try_normalize()
                else {
                    continue; // 세 점이 한 직선 위 — 면을 이루지 못한다
                };
                for cand in [n, n * -1.0] {
                    // `p − q`를 먼저 빼서 지지 거리를 한 번의 반올림으로 만든다.
                    let mut h = f32::NEG_INFINITY;
                    for pt in points {
                        h = h.max(cand.dot(*pt - q));
                    }
                    // 동률은 먼저 본 후보가 이긴다 — 순회가 결정적이므로 답도 결정적이다.
                    if h.is_finite() && best.is_none_or(|(bh, _)| h < bh) {
                        best = Some((h, cand));
                    }
                }
            }
        }
    }
    match best {
        // 여유는 음수가 될 수 없다(볼록체 안의 점) — 반올림이 만든 음수는 0으로 본다.
        Some((h, n)) => (n, h.max(0.0)),
        // 도달 불가(전체 차원인데 면이 하나도 안 나올 수 없다) — 계약을 지켜 결정적 축.
        None => (TIE_X, 0.0),
    }
}

/// 코어 **위**에 있는 점에서 겉면으로 나가는 결정적 방향.
///
/// 코어의 아핀 껍질이 3차원보다 낮으면(평면·선·점) 여공간이 있고, 그 방향이 답이다.
/// 3차원이면 여공간이 없다 — 그건 점이 코어 **경계 위**에 정확히 놓인 경우이므로
/// (안쪽이었다면 GJK가 교차를 줬다) 중심 쪽으로 아주 조금 당겨 교차 상태를 만들고
/// EPA에 최소 이탈 방향을 묻는다.
/// 점 집합의 아핀 껍질이 3차원보다 낮으면 그 **직교 여공간**의 단위 벡터, 아니면 `None`.
///
/// 임계는 점 집합의 지름에 상대적이다 — 절대 상수는 장면 스케일에서 깨진다(설계 §8).
fn affine_complement_dir(points: &[Vec3]) -> Option<Vec3> {
    let &p0 = points.first()?;
    let mut span = 0.0f32;
    for q in points {
        span = span.max((*q - p0).length());
    }
    if span <= 0.0 || span.is_nan() {
        return Some(TIE_X); // 모든 점이 한 자리(또는 비유한) — 코어는 점으로 본다
    }
    let eps = crate::length_eps(span);
    let mut basis = [Vec3::ZERO; 2];
    let mut rank = 0usize;
    for q in points {
        let mut v = *q - p0;
        for e in basis.iter().take(rank) {
            v = v - *e * e.dot(v);
        }
        let Some(u) = v.try_normalize() else { continue };
        if v.length() <= eps {
            continue;
        }
        if rank == 2 {
            return None; // 세 번째 독립 방향 — 전체 차원이다
        }
        basis[rank] = u;
        rank += 1;
    }
    Some(match rank {
        0 => TIE_X,
        1 => perpendicular_to(basis[0]),
        _ => basis[0].cross(basis[1]).try_normalize().unwrap_or(TIE_X),
    })
}

/// `d`에 수직인 결정적 단위 벡터 — 가장 덜 정렬된 축과 외적한다(퇴화 없음).
fn perpendicular_to(d: Vec3) -> Vec3 {
    let a = if d.x.abs() <= d.y.abs() && d.x.abs() <= d.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if d.y.abs() <= d.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    d.cross(a).try_normalize().unwrap_or(TIE_X)
}

/// 복합 — **자식 겉면들의 합집합** 위 최근접점.
///
/// 자식이 서로 겹치지 않으면 이것이 곧 복합 전체의 겉면이다(정상적인 복합 셰이프).
/// 겹치는 자식을 준 경우 반환점이 다른 자식 안쪽일 수 있다 — 그건 "합집합의 경계"가
/// 아니라 "경계들의 합집합"이며, rapier도 같은 정의다. 계약으로 명시하고 숨기지 않는다.
/// (진짜 합집합 경계는 자식 쌍마다 곡면 교선을 풀어야 한다.)
fn project_compound(children: &[(Pose, Shape)], p: Vec3) -> Result<PointProjection, QueryReject> {
    // 빈 복합은 답이 없다 — "무한히 먼 점"으로 위장하면 호스트가 NaN을 먹는다.
    // (`validate`가 1..=8을 강제하지만 이 함수는 그걸 가정하지 않는다.)
    let mut best: Option<(f32, Vec3)> = None;
    let mut inside = false;
    for (cp, cs) in children {
        // **자식 회전을 정규화한다** — `contains_point`·`cast_ray`가 쓰는 것과 같은
        // 변환이어야 한다. 형상 검증은 단위에서 길이제곱 1e-3까지 벗어난 자식 자세를
        // 통과시키므로, 정규화를 빼면 같은 합법 형상에서 안팎 판정이 다른 질의와 갈린다
        // (검수 rev_6429a223 MAJOR).
        let rot = cp.rot.try_normalize().ok_or(QueryReject::InvalidInput)?;
        let local = rot.conjugate().rotate(p - cp.pos);
        let sub = project_local(cs, local)?;
        let world = cp.pos + rot.rotate(sub.point);
        inside |= sub.inside;
        let d2 = (world - p).length_sq();
        // 동률은 **먼저 온 자식**이 이긴다(엄격 부등호) — 순서만이 결정한다.
        if best.is_none_or(|(bd, _)| d2 < bd) {
            best = Some((d2, world));
        }
    }
    match best {
        Some((_, point)) => Ok(PointProjection { point, inside }),
        None => Err(QueryReject::InvalidInput),
    }
}

/// 삼각형 다발(메시·하이트필드) 공통 — **반경을 넓혀 가며** 후보를 훑는다.
///
/// 전수 순회는 메시가 커지면 못 쓴다. 그렇다고 반경 하나를 찍으면 그 안에 삼각형이 없을
/// 때 답이 없다. 그래서 `p`에서 셰이프 AABB까지의 거리(하한)에서 시작해 **답을 찾고 그
/// 답이 현재 반경 안에 들어올 때까지** 배로 넓힌다 — 넓힌 반경 밖에 더 가까운 삼각형은
/// 있을 수 없으므로 그 시점의 답이 정확하다. AABB 전체를 덮으면 반드시 끝난다.
struct NearestTri {
    best_d2: f32,
    best: Vec3,
}

impl NearestTri {
    fn new() -> Self {
        Self {
            best_d2: f32::INFINITY,
            best: Vec3::ZERO,
        }
    }
    #[inline]
    fn feed(&mut self, p: Vec3, tri: [Vec3; 3]) {
        let q = closest_on_triangle(p, tri);
        let d2 = (q - p).length_sq();
        // 동률은 먼저 본 삼각형이 이긴다 — 순회 순서가 결정론이므로 답도 결정적이다.
        if d2 < self.best_d2 {
            self.best_d2 = d2;
            self.best = q;
        }
    }
}

/// 확장 검색 **전체**의 후보 예산 — 라운드마다가 아니라 누적이다.
///
/// 없으면 합법적인 단일 질의 하나가 백만 삼각형을 훑는다: 평평한 지형에 아주 높은 점을
/// 질의하면 첫 반경이 이미 AABB 전체를 덮어 모든 셀이 후보가 된다(검수 rev_6429a223 MAJOR).
/// BVH 스택 포화는 **깊이**만 막지 후보 수를 막지 않는다. 캐스트·매니폴드 경로와 같은
/// 상수를 써서 엔진 전체가 한 예산 정책을 쓰게 한다.
const PROJECT_TRI_BUDGET: usize = crate::MAX_HEIGHTFIELD_CELLS_PER_QUERY;

/// 확장 검색의 반경 수열: AABB까지의 거리에서 시작해 배로 늘리고, 전체를 덮으면 멈춘다.
fn search_radii(p: Vec3, bounds: Aabb) -> (f32, f32) {
    let clamped = Vec3::new(
        p.x.clamp(bounds.min.x, bounds.max.x),
        p.y.clamp(bounds.min.y, bounds.max.y),
        p.z.clamp(bounds.min.z, bounds.max.z),
    );
    let d0 = (clamped - p).length();
    let ext = bounds.max - bounds.min;
    let full = d0 + ext.x.max(ext.y).max(ext.z);
    // 첫 반경은 0이면 안 된다(0 AABB 질의는 경계 삼각형을 놓칠 수 있다).
    let seed = (full * 0.0625).max(f32::MIN_POSITIVE);
    (d0.max(seed), full.max(seed))
}

fn project_trimesh(
    mesh: &crate::trimesh::TriMeshData,
    p: Vec3,
    radius: f32,
) -> Result<PointProjection, QueryReject> {
    let bounds = mesh.local_aabb();
    let (mut r, full) = search_radii(p, bounds);
    let mut acc;
    let mut budget = PROJECT_TRI_BUDGET;
    loop {
        acc = NearestTri::new();
        let q = Aabb::new(p - splat(r), p + splat(r));
        let mut over = false;
        // 순회 스택 포화도 예산 소진도 **후보 누락**이다 — 그대로 답을 내면 "더 먼 겉면"을
        // 최근접이라 우기게 된다. 오답 대신 자원 오류로 드러낸다.
        let complete = mesh.for_each_candidate(q, |i| {
            if budget == 0 {
                over = true;
                return false;
            }
            budget -= 1;
            acc.feed(p, mesh.triangle(i));
            true
        });
        if over || !complete {
            return Err(QueryReject::ResourceLimit);
        }
        if acc.best_d2.sqrt() <= r || r >= full {
            break;
        }
        r = (r * 2.0).min(full);
    }
    if !acc.best_d2.is_finite() {
        // 삼각형이 하나도 없는 메시 — `TriMeshData::new`가 막지만 가정하지 않는다.
        return Err(QueryReject::InvalidInput);
    }
    // 메시는 표면이지 부피가 아니다 — `contains_point`와 같이 항상 밖이다.
    let mut proj = inflate(acc.best, p, radius, TIE_X);
    proj.inside = false;
    Ok(proj)
}

fn project_heightfield(
    hf: &crate::heightfield::HeightfieldRef<'_>,
    p: Vec3,
    radius: f32,
) -> Result<PointProjection, QueryReject> {
    let bounds = hf.local_aabb();
    let (mut r, full) = search_radii(p, bounds);
    let mut acc;
    let mut budget = PROJECT_TRI_BUDGET;
    loop {
        acc = NearestTri::new();
        let q = Aabb::new(p - splat(r), p + splat(r));
        let mut over = false;
        if let Some(range) = hf.cell_range(q) {
            crate::heightfield::for_each_overlapping_cell(hf, range, p.y - r, p.y + r, |cx, cz| {
                if budget == 0 {
                    over = true;
                    return false;
                }
                budget -= 1;
                acc.feed(p, hf.triangle(cx, cz, 0));
                acc.feed(p, hf.triangle(cx, cz, 1));
                true
            });
        }
        if over {
            return Err(QueryReject::ResourceLimit);
        }
        if acc.best_d2.sqrt() <= r || r >= full {
            break;
        }
        r = (r * 2.0).min(full);
    }
    if !acc.best_d2.is_finite() {
        return Err(QueryReject::InvalidInput);
    }
    let mut proj = inflate(acc.best, p, radius, TIE_Y);
    // 하이트필드는 **표면 아래를 속이 찬 것으로** 본다 — `contains_point`와 같은 계약이라
    // 반지름 근방 판정이 아니라 높이 비교가 답이다.
    proj.inside = hf.height_at(p.x, p.z).is_some_and(|h| p.y <= h);
    Ok(proj)
}

/// 점 → 삼각형 최근접점 (보로노이 영역 분해).
///
/// 정점 3 · 에지 3 · 내부 1의 일곱 영역을 순서대로 배제한다. 분모가 0이 되는 퇴화
/// 삼각형에서도 정점을 돌려주므로 NaN이 나가지 않는다.
#[inline]
pub(crate) fn closest_on_triangle(p: Vec3, tri: [Vec3; 3]) -> Vec3 {
    let (a, b, c) = (tri[0], tri[1], tri[2]);
    let ab = b - a;
    let ac = c - a;
    let d1 = ab.dot(p - a);
    let d2 = ac.dot(p - a);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let d3 = ab.dot(p - b);
    let d4 = ac.dot(p - b);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        return if denom > 0.0 {
            a + ab * (d1 / denom)
        } else {
            a
        };
    }
    let d5 = ab.dot(p - c);
    let d6 = ac.dot(p - c);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        return if denom > 0.0 {
            a + ac * (d2 / denom)
        } else {
            a
        };
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        return if denom > 0.0 {
            b + (c - b) * ((d4 - d3) / denom)
        } else {
            b
        };
    }
    let sum = va + vb + vc;
    if sum > 0.0 {
        let inv = 1.0 / sum;
        a + ab * (vb * inv) + ac * (vc * inv)
    } else {
        a // 퇴화(면적 0·NaN) — 내부 영역이 없다. 정점으로 마감해 NaN을 내보내지 않는다.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray::try_contains_point;

    fn ident() -> Pose {
        Pose {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        }
    }

    /// 회전+평행이동이 섞인 자세 — 로컬↔월드 왕복이 빠졌는지 잡는다.
    fn tilted() -> Pose {
        Pose {
            pos: Vec3::new(3.0, -2.0, 7.0),
            rot: Quat::from_axis_angle(Vec3::new(1.0, 2.0, -0.5), 0.9),
        }
    }

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1u32 << 24) as f32
        }
        fn point(&mut self, span: f32) -> Vec3 {
            v(
                (self.next() - 0.5) * span,
                (self.next() - 0.5) * span,
                (self.next() - 0.5) * span,
            )
        }
    }

    fn shapes() -> Vec<(&'static str, Shape)> {
        let hull_pts = vec![
            v(-0.8, -0.5, -0.4),
            v(0.9, -0.5, -0.4),
            v(0.0, 1.1, -0.4),
            v(0.0, 0.0, 0.7),
        ];
        vec![
            ("구", Shape::Sphere { radius: 1.3 }),
            (
                "캡슐",
                Shape::Capsule {
                    half_height: 0.7,
                    radius: 0.4,
                },
            ),
            (
                "각진 박스",
                Shape::Box {
                    half: v(0.6, 1.0, 0.35),
                    radius: 0.0,
                },
            ),
            (
                "둥근 박스",
                Shape::Box {
                    half: v(0.9, 0.7, 0.5),
                    radius: 0.25,
                },
            ),
            (
                "hull(각)",
                Shape::Hull {
                    points: hull_pts.clone(),
                    radius: 0.0,
                },
            ),
            (
                "hull(둥근)",
                Shape::Hull {
                    points: hull_pts,
                    radius: 0.2,
                },
            ),
            (
                "hull(평면 코어+두께)",
                Shape::Hull {
                    points: vec![
                        v(-0.9, 0.0, -0.6),
                        v(0.9, 0.0, -0.6),
                        v(0.9, 0.0, 0.6),
                        v(-0.9, 0.0, 0.6),
                    ],
                    radius: 0.18,
                },
            ),
            (
                "삼각형(얇음)",
                Shape::Triangle {
                    points: [v(-1.0, 0.0, -0.6), v(1.2, 0.0, -0.6), v(0.0, 0.0, 0.9)],
                    radius: 0.15,
                },
            ),
            (
                "복합",
                Shape::Compound {
                    children: vec![
                        (ident(), Shape::Sphere { radius: 0.5 }),
                        (
                            Pose {
                                pos: v(1.4, 0.0, 0.0),
                                rot: Quat::IDENTITY,
                            },
                            Shape::Box {
                                half: v(0.4, 0.4, 0.4),
                                radius: 0.0,
                            },
                        ),
                    ],
                },
            ),
        ]
    }

    // ---------- 해석적 대조 ----------

    #[test]
    fn sphere_projection_is_analytic() {
        let s = Shape::Sphere { radius: 2.0 };
        let pose = tilted();
        // 밖: 중심에서 같은 방향, 거리 정확히 radius.
        let p = pose.pos + v(0.0, 5.0, 0.0);
        let pr = try_project_point(&s, &pose, p).unwrap();
        assert!(!pr.inside);
        assert!((pr.point - pose.pos).length() - 2.0 < 1e-5);
        assert!((pr.point - (pose.pos + v(0.0, 2.0, 0.0))).length() < 1e-4);
        // 안: 여전히 겉면 위.
        let p = pose.pos + v(0.5, 0.0, 0.0);
        let pr = try_project_point(&s, &pose, p).unwrap();
        assert!(pr.inside);
        assert!(((pr.point - pose.pos).length() - 2.0).abs() < 1e-4);
        // 중심: 방향이 없다 — 겉면 위 결정적 한 점.
        let pr = try_project_point(&s, &pose, pose.pos).unwrap();
        assert!(pr.inside);
        assert!(((pr.point - pose.pos).length() - 2.0).abs() < 1e-4);
    }

    #[test]
    fn capsule_projection_is_analytic() {
        let s = Shape::Capsule {
            half_height: 1.0,
            radius: 0.5,
        };
        // 옆면: 축에서 수직으로 radius.
        let pr = try_project_point(&s, &ident(), v(3.0, 0.5, 0.0)).unwrap();
        assert!(!pr.inside);
        assert!((pr.point - v(0.5, 0.5, 0.0)).length() < 1e-5);
        // 반구 끝: 축 끝점에서 radius.
        let pr = try_project_point(&s, &ident(), v(0.0, 4.0, 0.0)).unwrap();
        assert!((pr.point - v(0.0, 1.5, 0.0)).length() < 1e-5);
        // 축 위(방향 없음): 겉면까지 정확히 radius.
        let pr = try_project_point(&s, &ident(), v(0.0, 0.3, 0.0)).unwrap();
        assert!(pr.inside);
        assert!(((pr.point - v(0.0, 0.3, 0.0)).length() - 0.5).abs() < 1e-5);
    }

    /// 둥근 박스의 **모서리는 깎여 있다** — half 상자로 재면 안이라고 우기는 자리다.
    /// `radius == half`면 그 박스는 구다.
    #[test]
    fn rounded_box_corner_is_cut_not_square() {
        let s = Shape::Box {
            half: v(1.0, 1.0, 1.0),
            radius: 1.0,
        };
        let corner = v(1.0, 1.0, 1.0); // 중심에서 √3 — 반지름 1인 구의 밖
        let pr = try_project_point(&s, &ident(), corner).unwrap();
        assert!(!pr.inside, "깎인 모서리를 안이라고 판정했다");
        assert!(
            (pr.point.length() - 1.0).abs() < 1e-5,
            "겉면(반지름 1)이 아니다: {}",
            pr.point.length()
        );
        // 안팎 질의도 같은 답이어야 한다.
        assert!(!try_contains_point(&s, &ident(), corner).unwrap());
        // 면 중앙은 여전히 안이다.
        assert!(try_contains_point(&s, &ident(), v(0.99, 0.0, 0.0)).unwrap());
    }

    #[test]
    fn box_interior_projects_to_nearest_face() {
        let s = Shape::Box {
            half: v(2.0, 0.5, 3.0),
            radius: 0.0,
        };
        // Y가 가장 얇다 — 안쪽 점은 Y면으로 나간다.
        let pr = try_project_point(&s, &ident(), v(0.1, 0.2, -0.3)).unwrap();
        assert!(pr.inside);
        assert!(
            (pr.point - v(0.1, 0.5, -0.3)).length() < 1e-6,
            "{:?}",
            pr.point
        );
        // 아래쪽이면 −Y로.
        let pr = try_project_point(&s, &ident(), v(0.1, -0.2, -0.3)).unwrap();
        assert!((pr.point - v(0.1, -0.5, -0.3)).length() < 1e-6);
        // X가 가장 가까운 자리.
        let pr = try_project_point(&s, &ident(), v(1.9, 0.0, 0.0)).unwrap();
        assert!(
            (pr.point - v(2.0, 0.0, 0.0)).length() < 1e-6,
            "{:?}",
            pr.point
        );
    }

    /// hull은 GJK/EPA 경로다 — 같은 기하를 박스로 준 해석 경로와 답이 같아야 한다.
    #[test]
    fn hull_matches_analytic_box() {
        let h = v(0.7, 0.5, 0.9);
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(h.x * sx, h.y * sy, h.z * sz));
                }
            }
        }
        let hull = Shape::Hull {
            points: pts,
            radius: 0.0,
        };
        let boxy = Shape::Box {
            half: h,
            radius: 0.0,
        };
        let mut rng = Lcg(7);
        for _ in 0..200 {
            let p = rng.point(4.0);
            let a = try_project_point(&hull, &ident(), p).unwrap();
            let b = try_project_point(&boxy, &ident(), p).unwrap();
            assert_eq!(a.inside, b.inside, "안팎이 갈렸다 p={p:?}");
            // 내부 점은 등거리 면이 여러 개일 수 있으므로 **거리**로 비교한다.
            let da = (a.point - p).length();
            let db = (b.point - p).length();
            assert!(
                (da - db).abs() < 2e-3,
                "hull {da} vs box {db} (p={p:?}, inside={})",
                a.inside
            );
        }
    }

    #[test]
    fn triangle_without_radius_is_never_inside() {
        let tri = Shape::Triangle {
            points: [v(-1.0, 0.0, -1.0), v(1.0, 0.0, -1.0), v(0.0, 0.0, 1.0)],
            radius: 0.0,
        };
        // 삼각형 위 점 — 겉면 위지만 부피가 없으므로 안이 아니다.
        let pr = try_project_point(&tri, &ident(), v(0.0, 0.0, 0.0)).unwrap();
        assert!(!pr.inside);
        assert!((pr.point - v(0.0, 0.0, 0.0)).length() < 1e-6);
        assert_eq!(
            pr.inside,
            try_contains_point(&tri, &ident(), Vec3::ZERO).unwrap()
        );
        // 정점 밖은 그 정점으로.
        let pr = try_project_point(&tri, &ident(), v(5.0, 0.0, -5.0)).unwrap();
        assert!(
            (pr.point - v(1.0, 0.0, -1.0)).length() < 1e-5,
            "{:?}",
            pr.point
        );
    }

    #[test]
    fn compound_picks_nearest_child_surface() {
        let s = Shape::Compound {
            children: vec![
                (ident(), Shape::Sphere { radius: 0.5 }),
                (
                    Pose {
                        pos: v(5.0, 0.0, 0.0),
                        rot: Quat::IDENTITY,
                    },
                    Shape::Sphere { radius: 0.5 },
                ),
            ],
        };
        let pr = try_project_point(&s, &ident(), v(-3.0, 0.0, 0.0)).unwrap();
        assert!(
            (pr.point - v(-0.5, 0.0, 0.0)).length() < 1e-5,
            "{:?}",
            pr.point
        );
        let pr = try_project_point(&s, &ident(), v(8.0, 0.0, 0.0)).unwrap();
        assert!(
            (pr.point - v(5.5, 0.0, 0.0)).length() < 1e-5,
            "{:?}",
            pr.point
        );
        // 둘째 자식 안 → inside
        let pr = try_project_point(&s, &ident(), v(5.1, 0.0, 0.0)).unwrap();
        assert!(pr.inside);
    }

    // ---------- 불변식 (전 셰이프) ----------

    /// **안팎 판정은 `contains_point`와 같은 답이어야 한다.**
    /// 두 질의가 갈리면 호스트가 "안에 있다는데 투영은 밖"을 보게 된다.
    #[test]
    fn inside_agrees_with_contains_point() {
        let mut rng = Lcg(31);
        for (name, s) in shapes() {
            for pose in [ident(), tilted()] {
                for _ in 0..300 {
                    let p = pose.pos + rng.point(5.0);
                    let a = try_project_point(&s, &pose, p).unwrap().inside;
                    let b = try_contains_point(&s, &pose, p).unwrap();
                    assert_eq!(a, b, "{name}: 안팎이 갈렸다 p={p:?}");
                }
            }
        }
    }

    /// **겉면 위 점을 다시 투영하면 자기 자신이다** — 반환점이 진짜 겉면 위라는 증명이다.
    /// (반환점이 코어 위거나 안쪽이면 두 번째 투영이 그걸 밖으로 밀어내 어긋난다.)
    #[test]
    fn projection_is_idempotent_on_surface() {
        let mut rng = Lcg(1009);
        for (name, s) in shapes() {
            for _ in 0..300 {
                let p = rng.point(6.0);
                let once = try_project_point(&s, &ident(), p).unwrap();
                let twice = try_project_point(&s, &ident(), once.point).unwrap();
                let drift = (twice.point - once.point).length();
                assert!(
                    drift < 3e-3,
                    "{name}: 겉면 위가 아니다 (드리프트 {drift}, p={p:?})"
                );
            }
        }
    }

    /// **더 가까운 겉면 점은 없다** — 겉면을 조밀히 표본해 반환값이 최소인지 확인한다.
    /// 표본은 상한이므로 "반환 거리 ≤ 표본 최소 거리 + 여유"만 요구한다.
    #[test]
    fn no_sampled_surface_point_is_closer() {
        let mut rng = Lcg(4242);
        for (name, s) in shapes() {
            // 겉면 표본: 사방에서 투영해 모은다(투영이 겉면 위라는 건 위 테스트가 잠근다).
            let mut surface = Vec::new();
            let mut s_rng = Lcg(77);
            for _ in 0..2000 {
                let far = s_rng.point(1.0);
                if far.length_sq() <= 0.0 {
                    continue;
                }
                let far = far * (12.0 / far.length());
                surface.push(try_project_point(&s, &ident(), far).unwrap().point);
            }
            for _ in 0..60 {
                let p = rng.point(7.0);
                let got = try_project_point(&s, &ident(), p).unwrap();
                let mine = (got.point - p).length();
                let best = surface
                    .iter()
                    .map(|q| (*q - p).length())
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    mine <= best + 1e-3,
                    "{name}: 표본이 더 가깝다 ({mine} > {best}, p={p:?})"
                );
            }
        }
    }

    // ---------- 메시·하이트필드 ----------

    fn brute_force_nearest(tris: &[[Vec3; 3]], p: Vec3) -> Vec3 {
        let mut best = (f32::INFINITY, Vec3::ZERO);
        for t in tris {
            let q = closest_on_triangle(p, *t);
            let d2 = (q - p).length_sq();
            if d2 < best.0 {
                best = (d2, q);
            }
        }
        best.1
    }

    #[test]
    fn trimesh_matches_brute_force() {
        // 계단형 리본 — BVH가 여러 갈래로 갈리도록.
        let mut verts = Vec::new();
        let mut tris = Vec::new();
        for i in 0..12u32 {
            let x = i as f32 * 0.9;
            let y = ((i % 3) as f32) * 0.6;
            verts.push(v(x, y, -1.0));
            verts.push(v(x, y, 1.0));
        }
        for i in 0..11u32 {
            let (a, b, c, d) = (i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 3);
            tris.push([a, b, c]);
            tris.push([b, d, c]);
        }
        let mesh = Shape::trimesh(verts.clone(), tris.clone()).expect("메시");
        let flat: Vec<[Vec3; 3]> = tris
            .iter()
            .map(|t| {
                [
                    verts[t[0] as usize],
                    verts[t[1] as usize],
                    verts[t[2] as usize],
                ]
            })
            .collect();
        let mut rng = Lcg(555);
        for _ in 0..400 {
            let p = v(
                rng.next() * 14.0 - 2.0,
                rng.next() * 6.0 - 3.0,
                rng.next() * 6.0 - 3.0,
            );
            let got = try_project_point(&mesh, &ident(), p).unwrap();
            let want = brute_force_nearest(&flat, p);
            assert!(!got.inside, "메시는 표면이지 부피가 아니다");
            let (dg, dw) = ((got.point - p).length(), (want - p).length());
            assert!(
                (dg - dw).abs() < 1e-4,
                "확장 검색이 더 가까운 삼각형을 놓쳤다 {dg} vs {dw} (p={p:?})"
            );
        }
    }

    #[test]
    fn heightfield_matches_brute_force_and_contains() {
        let (nx, nz, cell) = (9u32, 7u32, 0.8f32);
        let heights: Vec<f32> = (0..(nx * nz))
            .map(|i| ((i % 5) as f32) * 0.3 - 0.4)
            .collect();
        let hf = Shape::heightfield(heights.clone(), nx, nz, cell).expect("하이트필드");
        let href = hf.heightfield_ref().unwrap();
        let mut flat = Vec::new();
        let (ncx, ncz) = href.cells();
        for cz in 0..ncz {
            for cx in 0..ncx {
                flat.push(href.triangle(cx, cz, 0));
                flat.push(href.triangle(cx, cz, 1));
            }
        }
        let mut rng = Lcg(9001);
        for _ in 0..400 {
            let p = v(
                rng.next() * 10.0 - 2.0,
                rng.next() * 4.0 - 2.0,
                rng.next() * 8.0 - 2.0,
            );
            let got = try_project_point(&hf, &ident(), p).unwrap();
            let want = brute_force_nearest(&flat, p);
            let (dg, dw) = ((got.point - p).length(), (want - p).length());
            assert!(
                (dg - dw).abs() < 1e-4,
                "확장 검색 누락 {dg} vs {dw} (p={p:?})"
            );
            assert_eq!(
                got.inside,
                try_contains_point(&hf, &ident(), p).unwrap(),
                "하이트필드 안팎이 갈렸다 p={p:?}"
            );
        }
    }

    // ---------- 입력 검증 ----------

    // ---------- 검수 rev_6429a223 회귀 ----------

    /// **저차원 코어 + 두께**: z=0 평면 hull에 convex radius를 주면 겉면은 z=±r인데,
    /// GJK 법선을 그대로 쓰면 반올림이 만든 **평면 안** 방향으로 밀어 물체 안에 머문다.
    #[test]
    fn flat_hull_with_radius_escapes_the_plane() {
        let s = Shape::Hull {
            points: vec![
                v(-1.0, -1.0, 0.0),
                v(1.0, -1.0, 0.0),
                v(1.0, 1.0, 0.0),
                v(-1.0, 1.0, 0.0),
            ],
            radius: 0.1,
        };
        assert!(s.validate().is_ok(), "합법 셰이프라는 전제가 깨졌다");
        for p in [v(0.0, 0.0, 0.0), v(0.5, -0.3, 0.0), v(-0.9, 0.9, 0.0)] {
            let pr = try_project_point(&s, &ident(), p).unwrap();
            assert!(pr.inside, "코어 안인데 밖이라고 했다 p={p:?}");
            assert!(
                (pr.point.z.abs() - 0.1).abs() < 1e-4,
                "겉면(z=±0.1)이 아니다: {:?} (p={p:?})",
                pr.point
            );
            assert!(
                ((pr.point - p).length() - 0.1).abs() < 1e-4,
                "겉면까지 거리가 radius가 아니다: {} (p={p:?})",
                (pr.point - p).length()
            );
        }
        // 선형 코어(두 점)도 같은 계약이다 — 여공간이 2차원이라 방향 선택이 결정적이어야 한다.
        let seg = Shape::Hull {
            points: vec![v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
            radius: 0.3,
        };
        let pr = try_project_point(&seg, &ident(), v(0.2, 0.0, 0.0)).unwrap();
        assert!(pr.inside);
        assert!(
            ((pr.point - v(0.2, 0.0, 0.0)).length() - 0.3).abs() < 1e-4,
            "{:?}",
            pr.point
        );
    }

    /// **전체 차원이지만 얇고 긴 코어**: EPA 시드 조건(`부피 > 1e-6·최대에지³`)이
    /// 최대 에지에 상대적이라, 종횡비가 크면 관통 경로가 아예 안 열린다. 지지 함수
    /// 경로는 그 문턱이 없어야 한다(검수 rev_ef5d1868 BLOCKER).
    #[test]
    fn sliver_hull_exits_through_the_thin_axis() {
        let h = v(1000.0, 0.002, 0.002);
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(h.x * sx, h.y * sy, h.z * sz));
                }
            }
        }
        let s = Shape::Hull {
            points: pts,
            radius: 0.1,
        };
        assert!(s.validate().is_ok(), "합법 셰이프라는 전제가 깨졌다");
        let pr = try_project_point(&s, &ident(), Vec3::ZERO).unwrap();
        assert!(pr.inside);
        // 얇은 축(±Y·±Z)으로 나가야 한다 — 긴 축으로 나가면 코어 안에 그대로 남는다.
        assert!(
            pr.point.x.abs() < 1.0e-4,
            "긴 축으로 나갔다(코어 안에 남는다): {:?}",
            pr.point
        );
        let d = pr.point.length();
        assert!(
            (d - 0.102).abs() < 1.0e-4,
            "겉면까지 거리가 반두께+radius가 아니다: {d}"
        );
    }

    /// 면적 0 삼각형(사실상 선분)도 `is_castable`을 통과한다 — fallback이 코어와
    /// **나란한** 축이면 반환점이 코어 안에 남는다(검수 rev_ef5d1868 MAJOR).
    #[test]
    fn degenerate_triangle_exits_perpendicular_to_its_core() {
        let s = Shape::Triangle {
            points: [v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 0.0, 0.0)],
            radius: 0.1,
        };
        assert!(
            s.is_castable_pub_for_test(),
            "질의 입력으로 수용된다는 전제"
        );
        for q in [Vec3::ZERO, v(0.5, 0.0, 0.0), v(-1.0, 0.0, 0.0)] {
            let pr = try_project_point(&s, &ident(), q).unwrap();
            let d = pr.point - q;
            assert!(
                (d.length() - 0.1).abs() < 1.0e-5,
                "겉면까지 거리가 radius가 아니다: {} (q={q:?})",
                d.length()
            );
            assert!(
                d.x.abs() < 1.0e-5,
                "코어 선분과 나란한 방향으로 나갔다: {d:?} (q={q:?})"
            );
            // 겉면 위 점이므로 재투영이 제자리여야 한다.
            let again = try_project_point(&s, &ident(), pr.point).unwrap();
            assert!((again.point - pr.point).length() < 1.0e-5, "q={q:?}");
        }
    }

    /// 복합 자식 회전은 **다른 질의와 같은 정규화**를 거쳐야 한다 — 검증이 허용하는
    /// 비단위 자세에서 안팎 판정이 갈리면 안 된다.
    #[test]
    fn compound_child_rotation_is_normalized_like_other_queries() {
        let k = (1.0f32 + 9.0e-4).sqrt();
        let q = Quat::from_axis_angle(v(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_4);
        let scaled = Quat {
            x: q.x * k,
            y: q.y * k,
            z: q.z * k,
            w: q.w * k,
        };
        let s = Shape::Compound {
            children: vec![(
                Pose {
                    pos: Vec3::ZERO,
                    rot: scaled,
                },
                Shape::Box {
                    half: v(900.0, 1.0, 1.0),
                    radius: 0.0,
                },
            )],
        };
        assert!(s.validate().is_ok(), "합법 셰이프라는 전제가 깨졌다");
        let mut rng = Lcg(2024);
        for _ in 0..400 {
            // 상자 표면 근처를 훑는다 — 정규화 차이가 드러나는 자리다.
            let local = v(
                (rng.next() - 0.5) * 1900.0,
                (rng.next() - 0.5) * 2.4,
                (rng.next() - 0.5) * 2.4,
            );
            let p = q.rotate(local);
            let a = try_project_point(&s, &ident(), p).unwrap();
            let b = try_contains_point(&s, &ident(), p).unwrap();
            assert_eq!(a.inside, b, "안팎이 갈렸다 local={local:?}");
        }
    }

    /// 확장 검색은 **누적 예산**을 넘으면 답을 지어내지 않고 자원 오류를 낸다.
    #[test]
    fn oversized_mesh_query_reports_resource_limit() {
        // 예산(4096)을 넘는 삼각형을 한 반경 안에 두고, 멀리서 질의해 전부 후보가 되게 한다.
        let (nx, nz) = (64u32, 64u32);
        let mut verts = Vec::new();
        for iz in 0..nz {
            for ix in 0..nx {
                verts.push(v(ix as f32, 0.0, iz as f32));
            }
        }
        let mut tris = Vec::new();
        for iz in 0..(nz - 1) {
            for ix in 0..(nx - 1) {
                let a = iz * nx + ix;
                tris.push([a, a + 1, a + nx]);
                tris.push([a + 1, a + nx + 1, a + nx]);
            }
        }
        assert!(
            tris.len() > super::PROJECT_TRI_BUDGET,
            "전제: 예산 초과 규모"
        );
        let mesh = Shape::trimesh(verts, tris).expect("메시");
        // 격자 한참 위 — 첫 반경이 이미 전체를 덮는다.
        let r = try_project_point(&mesh, &ident(), v(32.0, 5000.0, 32.0));
        assert_eq!(r, Err(QueryReject::ResourceLimit));
    }

    #[test]
    fn invalid_input_is_an_error_not_a_far_point() {
        let s = Shape::Sphere { radius: 1.0 };
        assert_eq!(
            try_project_point(&s, &ident(), v(f32::NAN, 0.0, 0.0)),
            Err(QueryReject::InvalidInput)
        );
        let bad_pose = Pose {
            pos: Vec3::ZERO,
            rot: Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
        };
        assert_eq!(
            try_project_point(&s, &bad_pose, Vec3::ZERO),
            Err(QueryReject::InvalidInput)
        );
        // 빈 hull은 GJK를 죽인다 — 질의 경계에서 막혀야 한다.
        let empty = Shape::Hull {
            points: vec![],
            radius: 0.0,
        };
        assert_eq!(
            try_project_point(&empty, &ident(), Vec3::ZERO),
            Err(QueryReject::InvalidInput)
        );
    }
}
