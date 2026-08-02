//! 광선·점 질의 — 셰이프 단위 primitive (월드 순회는 musoe-dynamics).
//!
//! 계약:
//! - `toi`는 **광선 파라미터**다: 맞은 지점 = `origin + dir * toi`. `dir`은 정규화하지
//!   않아도 되지만, 정규화하면 `toi`가 곧 거리다.
//! - 광선이 셰이프 **안에서 시작**하면 `toi = 0`, 법선은 `-dir` 방향(관례: Bullet·rapier와 동일).
//!   "이미 안에 있다"를 못 맞은 것으로 처리하면 지면 판정이 조용히 실패한다.
//! - 반환 법선은 **바깥쪽 단위 벡터**다.
//! - convex radius가 있는 셰이프(둥근 박스·hull)는 코어에서 반지름만큼 부푼 형상으로 잰다.

use crate::{Pose, Shape};
use musoe_math::Vec3;

/// 셰이프 하나와의 교차.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayCast {
    /// 광선 파라미터. 맞은 지점 = `origin + dir * toi`.
    pub toi: f32,
    /// 맞은 지점의 바깥쪽 단위 법선.
    pub normal: Vec3,
    /// 광선 시작점이 셰이프 내부였는가 (그 경우 `toi == 0`).
    pub started_inside: bool,
}

/// 광선 × 셰이프. `max_toi`를 넘는 교차는 없는 것으로 본다.
///
/// 셰이프별 해석해가 있으면 그것을 쓰고(정확하고 빠르다), 없으면 support 함수 기반
/// 보수적 전진(GJK ray cast)으로 떨어진다 — hull과 둥근 박스가 그 경로다.
/// **질의가 답을 낼 수 없는 이유** — "맞은 것이 없다"와 구분된다.
///
/// 캐스트 쪽 `CastReject`와 같은 취지다: 잘못된 입력을 `None`/`false`로 돌려주면 호출자가
/// "비어 있다"고 읽어 그대로 진행한다(확인검수 rev_97887d75 BLOCKER — README가 이 두 함수를
/// "모든 셰이프를 옳게 처리하는 공개 표면"으로 열거하는데 방어가 캐스트에만 있었다).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QueryReject {
    /// 형상 불변식이 깨졌거나(빈 hull → GJK 패닉, 빈 복합 → 미스 위장) 산술 입력이
    /// 비유한이거나 자세를 정규화할 수 없다.
    InvalidInput,
    /// 가지치기 구조 순회가 **포화**해 후보를 다 못 봤다 — 답을 낼 수 없다.
    ///
    /// 이걸 "가장 가까운 건 못 찾았지만 이 정도"로 뭉개면 안 된다: 최근접 질의는 못 본
    /// 후보가 더 가까울 수 있으므로 조용한 오답이 된다(레이는 "안 맞음"이 보수적이지만
    /// 최근접은 보수적인 방향이 아예 없다).
    ResourceLimit,
}

/// 공개 레이 캐스트 — 형상·산술 입력을 검사하고 자세를 정규화한 뒤 푼다.
pub fn try_cast_ray(
    shape: &Shape,
    pose: &Pose,
    origin: Vec3,
    dir: Vec3,
    max_toi: f32,
) -> Result<Option<RayCast>, QueryReject> {
    // **유한성만으로는 부족하다**(확인검수 rev_c73f897c BLOCKER):
    // ① `dir = 0`이면 히트의 법선이 0 벡터가 되어 "법선은 단위 벡터"라는 반환 계약이 깨진다.
    // ② `origin = f32::MAX`는 유한하지만 중간 계산에서 `inf - inf = NaN`이 되고, NaN은 모든
    //    비교를 false로 만들어 `toi`·`normal`이 NaN인 히트가 `Ok(Some)`으로 나간다.
    // 그래서 범위(MAX_INPUT)로 재고 방향은 길이로 본다.
    let dir_l2 = dir.length_sq();
    if !shape.is_castable()
        || !crate::vec_in_range(origin)
        || !crate::vec_in_range(dir)
        || !dir_l2.is_finite()
        || dir_l2 <= 0.0
        || !max_toi.is_finite()
        || max_toi < 0.0
        || !crate::vec_in_range(pose.pos)
    {
        return Err(QueryReject::InvalidInput);
    }
    // 자세 정규화 — 비단위 쿼터니언은 회전+축소라 찌그러진 형상으로 답이 나온다.
    let pose = Pose {
        pos: pose.pos,
        rot: pose.rot.try_normalize().ok_or(QueryReject::InvalidInput)?,
    };
    Ok(cast_ray(shape, &pose, origin, dir, max_toi))
}

/// 공개 점 포함 판정 — `try_cast_ray`와 같은 검증 규약.
pub fn try_contains_point(shape: &Shape, pose: &Pose, p: Vec3) -> Result<bool, QueryReject> {
    let mut work = crate::MAX_PROJECT_WORK;
    try_contains_point_with_budget(shape, pose, p, &mut work)
}

/// **작업 예산을 공유하는** 포함 판정 — 여러 물체를 훑는 호출자용.
///
/// hull 포함 판정은 GJK를 믿을 수 없어 지지 함수를 전수로 돌 수 있다(`project::core_contains`).
/// 물체마다 새 예산을 주면 "질의점을 품는 hull 수천 개"인 월드가 한 번의 동기 질의에서
/// 수억 번 내적을 돈다(검수 rev_b16c028d MAJOR). `try_project_point_with_budget`과
/// 같은 계약이다: 소진되면 답을 지어내지 않고 `ResourceLimit`.
pub fn try_contains_point_with_budget(
    shape: &Shape,
    pose: &Pose,
    p: Vec3,
    budget: &mut usize,
) -> Result<bool, QueryReject> {
    if !shape.is_castable() || !crate::vec_in_range(p) || !crate::vec_in_range(pose.pos) {
        return Err(QueryReject::InvalidInput);
    }
    let pose = Pose {
        pos: pose.pos,
        rot: pose.rot.try_normalize().ok_or(QueryReject::InvalidInput)?,
    };
    contains_point(shape, &pose, p, budget)
}

/// **검증된 입력 전용**(크레이트 내부·월드 경계 통과분). 공개 진입은 `try_cast_ray`다 —
/// 자세가 단위 쿼터니언이고 형상 불변식이 성립한다고 가정한다.
pub(crate) fn cast_ray(
    shape: &Shape,
    pose: &Pose,
    origin: Vec3,
    dir: Vec3,
    max_toi: f32,
) -> Option<RayCast> {
    // 셰이프 로컬로 옮겨 푼다 (자세는 단위 쿼터니언 — 입력 검증이 보장한다).
    let inv = pose.rot.conjugate();
    let o = inv.rotate(origin - pose.pos);
    let d = inv.rotate(dir);
    let local = match *shape {
        Shape::Sphere { radius } => ray_sphere(o, d, Vec3::ZERO, radius, max_toi),
        Shape::Capsule {
            half_height,
            radius,
        } => ray_capsule(o, d, half_height, radius, max_toi),
        Shape::Box { half, radius } => {
            if radius <= 0.0 {
                ray_box(o, d, half, max_toi)
            } else {
                ray_convex_ca(shape, o, d, max_toi)
            }
        }
        Shape::Cylinder {
            half_height,
            radius,
        } => ray_cylinder(o, d, half_height, radius, max_toi),
        Shape::Hull { .. } => ray_convex_ca(shape, o, d, max_toi),
        Shape::Triangle { points, .. } => crate::heightfield::ray_triangle(o, d, points, max_toi)
            .map(|(toi, n)| Local {
                toi,
                normal: n,
                started_inside: false,
            }),
        // 메시: BVH로 **선분 AABB**와 겹치는 삼각형만 시험한다.
        //
        // 레이 전용 BVH 하강(슬래브 순서 순회)이 아니라 선분 AABB 질의를 쓴다 — 후보가
        // 상위집합이라 정확성은 같고, 트리 API를 하나로 유지한다. 긴 레이는 후보가 늘지만
        // 지형 레이는 대개 짧다(길어지면 그때 하강을 넣는다 — 지금 넣으면 근거 없는 복잡도다).
        //
        // **순회가 포화되면 `None`이 아니라 "안 맞음"으로 뭉개지 않는다**: 조용히 후보를
        // 빠뜨리면 레이가 지형을 통과한 것처럼 보인다. 포화는 상한을 넘긴 메시에서만 나고,
        // 그 경우 히트 없음 대신 **그때까지 찾은 최근접**을 돌려준다(보수적).
        Shape::TriMesh(ref mesh) => {
            let end = o + d * max_toi;
            let lo = Vec3::new(o.x.min(end.x), o.y.min(end.y), o.z.min(end.z));
            let hi = Vec3::new(o.x.max(end.x), o.y.max(end.y), o.z.max(end.z));
            let mut best: Option<Local> = None;
            mesh.for_each_candidate(crate::Aabb::new(lo, hi), |i| {
                let tri = mesh.triangle(i);
                if let Some((toi, n)) = crate::heightfield::ray_triangle(o, d, tri, max_toi) {
                    if best.as_ref().is_none_or(|b| toi < b.toi) {
                        best = Some(Local {
                            toi,
                            normal: n,
                            started_inside: false,
                        });
                    }
                }
                true
            });
            best
        }
        // 하이트필드: 격자 DDA — 지나는 셀만 삼각형으로 시험한다.
        Shape::Heightfield { .. } => {
            let hf = shape.heightfield_ref().expect("Heightfield 변종");
            // **표면 아래에서 시작하면 `toi=0, started_inside`** — 하이트필드는 속이 찬
            // 것으로 정의했으므로(`contains_point`) 레이도 같은 계약을 따라야 한다.
            // 안 그러면 "안에 있는데 안 맞았다"가 되어 다른 셰이프와 규약이 갈린다
            // (검수 rev_1e81559e MAJOR).
            if hf.height_at(o.x, o.z).is_some_and(|h| o.y <= h) {
                return Some(RayCast {
                    toi: 0.0,
                    normal: -dir.normalize_or_zero(),
                    started_inside: true,
                });
            }
            crate::heightfield::ray_heightfield(&hf, o, d, max_toi).map(|(toi, n)| Local {
                toi,
                normal: n,
                started_inside: false,
            })
        }
        // 복합: 자식별 최소 TOI. 동률은 앞선 자식(낮은 인덱스) — 결정론.
        Shape::Compound { ref children } => {
            let mut best: Option<Local> = None;
            for (cp, child) in children {
                // 자식 로컬로 다시 변환해 재귀 대신 직접 호출 (중첩은 validate가 금지).
                let world_child = Pose {
                    pos: pose.pos + pose.rot.rotate(cp.pos),
                    // 자식 회전 정규화 — 근거는 `cast.rs`의 compose 주석에.
                    rot: pose.rot * cp.rot.normalize(),
                };
                if let Some(hit) = cast_ray(child, &world_child, origin, dir, max_toi) {
                    let better = best.as_ref().is_none_or(|b| hit.toi < b.toi);
                    if better {
                        // 자식 히트는 이미 월드 법선 — 아래 공통 회전을 건너뛰기 위해
                        // 로컬 형식으로 역변환한다.
                        best = Some(Local {
                            toi: hit.toi,
                            normal: pose.rot.conjugate().rotate(hit.normal),
                            started_inside: hit.started_inside,
                        });
                    }
                }
            }
            best
        }
    }?;
    Some(RayCast {
        toi: local.toi,
        normal: pose.rot.rotate(local.normal),
        started_inside: local.started_inside,
    })
}

/// 점이 셰이프 안에 있는가 (겉면 기준 — convex radius 포함).
/// **검증된 입력 전용** — 공개 진입은 `try_contains_point`다.
pub(crate) fn contains_point(
    shape: &Shape,
    pose: &Pose,
    p: Vec3,
    budget: &mut usize,
) -> Result<bool, QueryReject> {
    let local = pose.rot.conjugate().rotate(p - pose.pos);
    let r = shape.radius();
    Ok(match *shape {
        Shape::Sphere { radius } => local.length_sq() <= radius * radius,
        Shape::Capsule {
            half_height,
            radius,
        } => {
            let y = local.y.clamp(-half_height, half_height);
            (local - Vec3::new(0.0, y, 0.0)).length_sq() <= radius * radius
        }
        Shape::Cylinder {
            half_height,
            radius,
        } => {
            local.y.abs() <= half_height && local.x * local.x + local.z * local.z <= radius * radius
        }
        Shape::Box { half, radius } => {
            // 겉면 기준 half가 곧 사용자 관점 치수다 (코어+radius = half).
            //
            // **모서리는 깎여 있다**: 둥근 박스의 겉면은 "코어 상자 ⊕ 반지름 공"이라
            // 꼭짓점 근방이 구의 8분면이다. half 상자로만 재면 그 깎인 자리를 안이라고
            // 우긴다(예: half=(1,1,1)·radius=1인 구에서 (1,1,1)은 중심에서 √3 떨어져
            // 밖인데 half 판정은 통과한다). 코어까지의 거리로 재야 옳고, 그래야
            // `project_point`의 안팎 판정과도 같은 답이 된다.
            let core = Vec3::new(
                (half.x - radius).max(0.0),
                (half.y - radius).max(0.0),
                (half.z - radius).max(0.0),
            );
            let q = Vec3::new(
                local.x.clamp(-core.x, core.x),
                local.y.clamp(-core.y, core.y),
                local.z.clamp(-core.z, core.z),
            );
            (local - q).length_sq() <= radius * radius
        }
        Shape::Hull { .. } => {
            // 코어 안이거나, 코어까지의 거리가 radius 이하면 안이다.
            //
            // **코어 안 판정에 GJK 거리를 쓰지 않는다** — GJK는 코어 안의 점에도 양의
            // 거리를 줄 수 있고(같은 support 재등장 시 조기 종료), 그러면 안에 있는 점을
            // 밖이라고 한다. `project`와 **같은 판정**을 써서 두 질의가 갈리지 않게 한다
            // (검수 rev_230afeaf BLOCKER).
            let d = closest_core_dist(shape, local).0;
            crate::project::core_contains(shape, local, budget)? || d <= r
        }
        // 삼각형은 부피가 없다 — radius가 있을 때만 코어까지 거리로 판정(hull과 같은 경로).
        Shape::Triangle { .. } => r > 0.0 && closest_core_dist(shape, local).0 <= r,
        // **메시는 표면이지 부피가 아니다 — 포함은 항상 거짓이다.**
        //
        // 하이트필드는 "높이 아래 = 속이 참"으로 정의할 수 있다. 격자가 **높이 함수**라
        // 위/아래가 전역적으로 정해지기 때문이다. 일반 메시에는 그런 것이 없다: 열린
        // 지형 패치일 수도, 닫힌 방일 수도 있고, 안팎 판정은 레이 패리티 계산이 필요한데
        // 그건 열린 메시에서 답이 아예 정의되지 않는다.
        //
        // 그래서 **정의되지 않은 것을 추측하지 않는다**. rapier의 TriMesh도 기본이 표면이다.
        // 부피 질의가 필요하면 닫힌 볼록 셰이프나 복합으로 표현하는 것이 옳다.
        Shape::TriMesh(_) => false,
        // 하이트필드는 **표면 아래를 속이 찬 것으로** 본다(Jolt 동형): 격자 범위 안이고
        // 보간 높이 아래면 포함이다. 그래야 `query_point`가 지형에 묻힌 물체를 찾는다.
        Shape::Heightfield { .. } => {
            let hf = shape.heightfield_ref().expect("Heightfield 변종");
            match hf.height_at(local.x, local.z) {
                Some(h) => local.y <= h,
                None => false,
            }
        }
        // 복합: 어느 자식이든 포함하면 포함. 예산은 자식 사이에서 **공유**한다.
        Shape::Compound { ref children } => {
            for (cp, child) in children {
                let world_child = Pose {
                    pos: pose.pos + pose.rot.rotate(cp.pos),
                    // 자식 회전 정규화 — 근거는 `cast.rs`의 compose 주석에.
                    rot: pose.rot * cp.rot.normalize(),
                };
                if contains_point(child, &world_child, p, budget)? {
                    return Ok(true);
                }
            }
            false
        }
    })
}

// ---------- 셰이프별 해석해 (로컬 좌표) ----------

struct Local {
    toi: f32,
    normal: Vec3,
    started_inside: bool,
}

fn ray_sphere(o: Vec3, d: Vec3, c: Vec3, r: f32, max_toi: f32) -> Option<Local> {
    let m = o - c;
    if m.length_sq() <= r * r {
        return Some(Local {
            toi: 0.0,
            normal: -d.normalize_or_zero(),
            started_inside: true,
        });
    }
    let a = d.length_sq();
    if a <= 0.0 {
        return None;
    }
    let b = m.dot(d);
    if b >= 0.0 {
        return None; // 구에서 멀어지는 방향
    }
    let cc = m.length_sq() - r * r;
    let disc = b * b - a * cc;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / a;
    if t < 0.0 || t > max_toi {
        return None;
    }
    let hit = o + d * t;
    Some(Local {
        toi: t,
        normal: (hit - c).normalize_or_zero(),
        started_inside: false,
    })
}

/// 축 정렬 박스 슬랩 판정 (radius 0 전용 — 둥근 박스는 보수적 전진으로 간다).
fn ray_box(o: Vec3, d: Vec3, half: Vec3, max_toi: f32) -> Option<Local> {
    if o.x.abs() <= half.x && o.y.abs() <= half.y && o.z.abs() <= half.z {
        return Some(Local {
            toi: 0.0,
            normal: -d.normalize_or_zero(),
            started_inside: true,
        });
    }
    let (mut tmin, mut tmax) = (0.0f32, max_toi);
    let mut axis = 0usize;
    let mut sign = 1.0f32;
    let (od, dd, hd) = ([o.x, o.y, o.z], [d.x, d.y, d.z], [half.x, half.y, half.z]);
    for i in 0..3 {
        if dd[i].abs() < 1e-20 {
            // 축에 평행 — 그 축의 슬랩 밖이면 절대 못 맞는다
            if od[i].abs() > hd[i] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / dd[i];
        let mut t1 = (-hd[i] - od[i]) * inv;
        let mut t2 = (hd[i] - od[i]) * inv;
        let mut s = -1.0f32;
        if t1 > t2 {
            core::mem::swap(&mut t1, &mut t2);
            s = 1.0;
        }
        if t1 > tmin {
            tmin = t1;
            axis = i;
            sign = s;
        }
        tmax = tmax.min(t2);
        if tmin > tmax {
            return None;
        }
    }
    let mut n = Vec3::ZERO;
    match axis {
        0 => n.x = sign,
        1 => n.y = sign,
        _ => n.z = sign,
    }
    Some(Local {
        toi: tmin,
        normal: n,
        started_inside: false,
    })
}

/// 광선 × **원기둥**(Y축 정렬, 날카로운 끝면).
///
/// 두 조각을 각각 풀어 가장 이른 히트를 고른다: 무한 원기둥의 옆면(XZ 평면의 원과의
/// 2차식)과 위·아래 끝면(평면 교차 뒤 반경 확인). 조각마다 **다른 조각의 범위 조건**을
/// 확인해야 한다 — 옆면 해는 |y| ≤ half_height, 끝면 해는 반경 ≤ radius여야 진짜 히트다.
///
/// 광선이 안에서 시작하면 `started_inside`로 표시하고 toi 0을 준다(다른 셰이프와 같은 규약).
fn ray_cylinder(o: Vec3, d: Vec3, half_height: f32, radius: f32, max_toi: f32) -> Option<Local> {
    if o.y.abs() <= half_height && o.x * o.x + o.z * o.z <= radius * radius {
        return Some(Local {
            toi: 0.0,
            // **0 벡터를 내보내지 않는다.** 공개 계약은 "법선은 단위 벡터"이고, 0을 주면
            // 호스트의 반사·슬라이딩이 NaN이 되거나 충돌을 무시한다(검수 rev_0e075168 MAJOR).
            // 안에서 시작하면 캡슐·구와 같이 **광선 반대 방향**을 준다.
            normal: d
                .try_normalize()
                .map_or(Vec3::new(0.0, 1.0, 0.0), |u| u * -1.0),
            started_inside: true,
        });
    }
    let mut best: Option<Local> = None;
    let mut take = |toi: f32, normal: Vec3| {
        if toi < 0.0 || toi > max_toi {
            return;
        }
        // 동률은 먼저 본 조각이 이긴다 — 옆면을 먼저 시험하므로 규약이 결정적이다.
        if best.as_ref().is_none_or(|b| toi < b.toi) {
            best = Some(Local {
                toi,
                normal,
                started_inside: false,
            });
        }
    };
    // ① 옆면: (o.xz + t·d.xz)² = radius²
    let a = d.x * d.x + d.z * d.z;
    if a > 0.0 {
        let b = o.x * d.x + o.z * d.z;
        let c = o.x * o.x + o.z * o.z - radius * radius;
        let disc = b * b - a * c;
        if disc >= 0.0 {
            let sd = disc.sqrt();
            // 두 근 모두 본다 — 밖에서 시작해도 접선 근처에서는 작은 근이 음수일 수 있다.
            for t in [(-b - sd) / a, (-b + sd) / a] {
                let y = o.y + t * d.y;
                if y.abs() <= half_height {
                    let px = o.x + t * d.x;
                    let pz = o.z + t * d.z;
                    let n = Vec3::new(px, 0.0, pz)
                        .try_normalize()
                        .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                    take(t, n);
                }
            }
        }
    }
    // ② 끝면: y = ±half_height 평면과의 교차 뒤 반경 확인.
    if d.y != 0.0 {
        for sy in [-1.0f32, 1.0] {
            let t = (sy * half_height - o.y) / d.y;
            let px = o.x + t * d.x;
            let pz = o.z + t * d.z;
            if px * px + pz * pz <= radius * radius {
                take(t, Vec3::new(0.0, sy, 0.0));
            }
        }
    }
    best
}

/// 광선 × 캡슐 (Y축 세그먼트 + radius).
fn ray_capsule(o: Vec3, d: Vec3, half_height: f32, r: f32, max_toi: f32) -> Option<Local> {
    // 내부에서 시작?
    let y0 = o.y.clamp(-half_height, half_height);
    if (o - Vec3::new(0.0, y0, 0.0)).length_sq() <= r * r {
        return Some(Local {
            toi: 0.0,
            normal: -d.normalize_or_zero(),
            started_inside: true,
        });
    }
    // 원통부: xz 평면에서 원과의 교차 → y가 세그먼트 안이면 채택
    let mut best: Option<Local> = None;
    let mut consider = |t: f32, n: Vec3| {
        if t >= 0.0 && t <= max_toi && best.as_ref().is_none_or(|b| t < b.toi) {
            best = Some(Local {
                toi: t,
                normal: n,
                started_inside: false,
            });
        }
    };
    let a = d.x * d.x + d.z * d.z;
    if a > 1e-20 {
        let b = o.x * d.x + o.z * d.z;
        let c = o.x * o.x + o.z * o.z - r * r;
        let disc = b * b - a * c;
        if disc >= 0.0 {
            let t = (-b - disc.sqrt()) / a;
            let y = o.y + d.y * t;
            if y >= -half_height && y <= half_height {
                let p = o + d * t;
                consider(t, Vec3::new(p.x, 0.0, p.z).normalize_or_zero());
            }
        }
    }
    // 양 끝 반구
    for cy in [-half_height, half_height] {
        let c = Vec3::new(0.0, cy, 0.0);
        if let Some(l) = ray_sphere(o, d, c, r, max_toi) {
            // 반구 쪽만 채택 (원통부와 겹치는 절반은 위에서 이미 봤다)
            let p = o + d * l.toi;
            let keep = if cy > 0.0 { p.y >= cy } else { p.y <= cy };
            if keep {
                consider(l.toi, l.normal);
            }
        }
    }
    best
}

// ---------- 일반 convex: support 기반 보수적 전진 ----------

/// 로컬 점에서 **코어**까지의 거리와 바깥 법선.
///
/// 새로 GJK를 쓰지 않고 **크레이트의 검증된 GJK를 그대로 재사용한다** — 점을 반지름 0
/// 구로 모델링하면 점-볼록체 거리는 셰이프-셰이프 거리의 특수한 경우다.
/// (처음엔 점 전용 GJK를 따로 썼는데 심플렉스 축약을 대충 만들어 면 위 최근접점을
///  놓쳤고, 보수적 전진이 통째로 실패했다. 검증된 코드를 부르는 편이 짧고 옳다.)
fn closest_core_dist(shape: &Shape, p: Vec3) -> (f32, Vec3) {
    let point = Shape::Sphere { radius: 0.0 };
    let at = Pose {
        pos: p,
        rot: musoe_math::Quat::IDENTITY,
    };
    let ident = Pose {
        pos: Vec3::ZERO,
        rot: musoe_math::Quat::IDENTITY,
    };
    match crate::gjk::gjk(&point, &at, shape, &ident) {
        crate::gjk::GjkResult::Separated {
            distance, normal, ..
        } => (distance, normal),
        // 코어 안 — 거리 0
        crate::gjk::GjkResult::Intersecting(_) => (0.0, Vec3::ZERO),
    }
}

/// 보수적 전진 광선 캐스트 — 임의 convex(코어 + radius)에 쓴다.
///
/// 매 반복에서 현재 점의 표면까지 거리 `d`를 구하고, 광선이 그만큼은 안전하게 나아갈 수
/// 있으므로 `t += d / |dir 성분|`으로 전진한다. 표면에 닿으면 종료한다.
fn ray_convex_ca(shape: &Shape, o: Vec3, d: Vec3, max_toi: f32) -> Option<Local> {
    let r = shape.radius();
    let dlen = d.length();
    if dlen <= 0.0 {
        return None;
    }
    // 스케일 의존 종료 임계 — 절대 상수는 장면 스케일에서 깨진다(설계 §8).
    let scale = crate::characteristic_length(shape, shape).max(1e-6);
    let eps = 1e-4 * scale;

    let (d0, n0) = closest_core_dist(shape, o);
    if d0 <= r {
        return Some(Local {
            toi: 0.0,
            normal: -d * (1.0 / dlen),
            started_inside: true,
        });
    }
    let mut t = 0.0f32;
    let mut normal = n0;
    for _ in 0..64 {
        let p = o + d * t;
        let (dist, n) = closest_core_dist(shape, p);
        let gap = dist - r;
        if gap <= eps {
            // **0 법선을 내보내지 않는다.** 전진이 반올림으로 코어 안으로 한 발 넘어가면
            // GJK가 `Intersecting`을 주고 `closest_core_dist`가 방향을 0으로 돌려준다.
            // 그대로 반환하면 "법선은 바깥쪽 단위 벡터"라는 공개 계약이 깨지고
            // 호스트의 반사·슬라이딩이 0/NaN이 된다 (검수 rev_710b92d2 MAJOR).
            // 직전 반복의 법선이 여전히 유효한 바깥 방향이다.
            let hit_normal = if n.length_sq() > 0.0 { n } else { normal };
            return Some(Local {
                toi: t,
                normal: hit_normal,
                started_inside: false,
            });
        }
        if n.length_sq() > 0.0 {
            normal = n;
        }
        // 광선이 표면 쪽으로 다가가는 속도. 멀어지면 절대 못 맞는다.
        let approach = -d.dot(n);
        if approach <= 1e-20 {
            return None;
        }
        t += gap / approach;
        if t > max_toi {
            return None;
        }
    }
    // 반복 상한 — 수렴 못 했으면 못 맞은 것으로 본다(거짓 히트보다 낫다).
    let _ = normal;
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use musoe_math::Quat;

    fn ident() -> Pose {
        Pose {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        }
    }

    #[test]
    fn sphere_hit_and_miss() {
        let s = Shape::Sphere { radius: 1.0 };
        let h = cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            100.0,
        )
        .unwrap();
        assert!((h.toi - 4.0).abs() < 1e-4, "toi={}", h.toi);
        assert!((h.normal - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-4);
        assert!(!h.started_inside);
        assert!(cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 3.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            100.0
        )
        .is_none());
        // 뒤로 쏘면 안 맞는다
        assert!(cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            100.0
        )
        .is_none());
    }

    #[test]
    fn ray_starting_inside_reports_zero_toi() {
        for s in [
            Shape::Sphere { radius: 1.0 },
            Shape::Capsule {
                half_height: 1.0,
                radius: 0.5,
            },
            Shape::Box {
                half: Vec3::new(1.0, 1.0, 1.0),
                radius: 0.0,
            },
        ] {
            let h = cast_ray(&s, &ident(), Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 10.0)
                .unwrap_or_else(|| panic!("내부 시작인데 못 맞았다: {s:?}"));
            assert_eq!(h.toi, 0.0);
            assert!(h.started_inside);
        }
    }

    #[test]
    fn box_face_normals_are_axis_aligned() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.0,
        };
        let cases = [
            (
                Vec3::new(-9.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                8.0,
                Vec3::new(-1.0, 0.0, 0.0),
            ),
            (
                Vec3::new(0.0, 9.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                7.0,
                Vec3::new(0.0, 1.0, 0.0),
            ),
            (
                Vec3::new(0.0, 0.0, -9.0),
                Vec3::new(0.0, 0.0, 1.0),
                6.0,
                Vec3::new(0.0, 0.0, -1.0),
            ),
        ];
        for (o, d, want_t, want_n) in cases {
            let h = cast_ray(&s, &ident(), o, d, 100.0).unwrap();
            assert!((h.toi - want_t).abs() < 1e-4, "toi={} want {want_t}", h.toi);
            assert!((h.normal - want_n).length() < 1e-4, "n={:?}", h.normal);
        }
    }

    /// 회전한 자세에서도 맞아야 한다 — 로컬 변환이 맞는지 가른다.
    #[test]
    fn rotated_pose_is_handled() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let p = Pose {
            pos: Vec3::new(5.0, 0.0, 0.0),
            rot: Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_4),
        };
        let h = cast_ray(&s, &p, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 100.0).unwrap();
        // 45° 돌린 정육면체의 x쪽 최근접점은 모서리 → 중심에서 √2 떨어져 있다
        let want = 5.0 - 2.0f32.sqrt();
        assert!((h.toi - want).abs() < 1e-3, "toi={} want {want}", h.toi);
    }

    /// 보수적 전진 경로(둥근 박스·hull)가 해석해와 어긋나지 않아야 한다.
    #[test]
    fn conservative_advancement_matches_analytic_on_a_sphere_like_hull() {
        // 정팔면체 hull + radius → 코어가 작아 구에 가까운 형상
        let pts = vec![
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, -0.5),
        ];
        let hull = Shape::Hull {
            data: crate::HullData::new(pts),
            radius: 0.25,
        };
        // +x로 쏘면 코어 최원점 0.5 + radius 0.25 = 0.75 에서 맞는다
        let h = cast_ray(
            &hull,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            100.0,
        )
        .unwrap();
        assert!((h.toi - 4.25).abs() < 5e-3, "toi={}", h.toi);
        assert!(h.normal.x < -0.99, "n={:?}", h.normal);
    }

    /// 둥근 박스: 면 중앙은 half까지, 모서리 방향은 더 가깝다.
    #[test]
    fn rounded_box_face_distance() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.25,
        };
        let h = cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            100.0,
        )
        .unwrap();
        assert!(
            (h.toi - 4.0).abs() < 5e-3,
            "면 중앙은 half=1.0에서 맞아야 한다: {}",
            h.toi
        );
    }

    /// **성공 결과의 법선은 항상 단위 벡터**여야 한다 (공개 계약).
    ///
    /// 보수적 전진이 코어 안으로 한 발 넘어가면 GJK가 `Intersecting`을 주고 방향이
    /// 0이 된다 — 그대로 내보내면 호스트의 반사 계산이 0/NaN이 된다.
    #[test]
    fn conservative_advancement_never_returns_a_zero_normal() {
        let mut rng = 0x1234_5678u32;
        let mut f = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng >> 8) as f32 / (1u32 << 24) as f32
        };
        let shapes = [
            Shape::Box {
                half: Vec3::new(1.0, 0.05, 1.0), // 얇은 판 — 스치는 각이 잘 나온다
                radius: 0.02,
            },
            Shape::Hull {
                data: crate::HullData::new(vec![
                    Vec3::new(0.6, 0.0, 0.0),
                    Vec3::new(-0.6, 0.0, 0.0),
                    Vec3::new(0.0, 0.03, 0.5),
                    Vec3::new(0.0, -0.03, -0.5),
                ]),
                radius: 0.05,
            },
        ];
        let mut hits = 0;
        for s in &shapes {
            for _ in 0..4000 {
                let o = Vec3::new((f() - 0.5) * 8.0, (f() - 0.5) * 8.0, (f() - 0.5) * 8.0);
                // 셰이프 근처를 스치도록 목표점을 형상 경계 부근으로 잡는다
                let target = Vec3::new((f() - 0.5) * 2.2, (f() - 0.5) * 0.3, (f() - 0.5) * 2.2);
                let d = target - o;
                if d.length_sq() < 1e-6 {
                    continue;
                }
                let Some(h) = cast_ray(s, &ident(), o, d, 100.0) else {
                    continue;
                };
                hits += 1;
                let l2 = h.normal.length_sq();
                assert!(
                    (0.9..=1.1).contains(&l2),
                    "법선이 단위가 아니다: {:?} (|n|²={l2}) shape={s:?} o={o:?} d={d:?}",
                    h.normal
                );
            }
        }
        assert!(hits > 500, "표본이 너무 적다: {hits}");
    }

    #[test]
    fn max_toi_is_respected() {
        let s = Shape::Sphere { radius: 1.0 };
        assert!(cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            3.0
        )
        .is_none());
        assert!(cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            4.5
        )
        .is_some());
    }

    #[test]
    fn point_containment() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.0,
        };
        let mut w = usize::MAX;
        assert!(contains_point(&s, &ident(), Vec3::new(0.9, 1.9, 2.9), &mut w).unwrap());
        assert!(!contains_point(&s, &ident(), Vec3::new(1.1, 0.0, 0.0), &mut w).unwrap());
        let sp = Shape::Sphere { radius: 1.0 };
        assert!(contains_point(&sp, &ident(), Vec3::new(0.5, 0.5, 0.5), &mut w).unwrap());
        assert!(!contains_point(&sp, &ident(), Vec3::new(0.7, 0.7, 0.7), &mut w).unwrap());
    }
}
