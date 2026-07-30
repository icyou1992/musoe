//! 선형 셰이프 캐스트 — CCD linear cast의 기하 원시 (설계 §4 CCD 계약 v0.5).
//!
//! A를 `motion`만큼 **평행이동**시키며 정지한 B와 처음 닿는 시각을 conservative
//! advancement로 구한다. 회전은 캐스트 중 무시한다(linear cast — Jolt LinearCast
//! stop-at-TOI와 동형, 설계 근거). 볼록체 + 선형 이동이므로 "최근접 방향 접근 속도로
//! 최근접 거리를 나눈 전진"은 관통을 건너뛸 수 없다 — CA의 건전성 근거.

use crate::gjk::{gjk, GjkResult};
use crate::{Pose, Shape};
use maru_math::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeCastHit {
    /// `motion`의 분율 [0,1] — 이 시각에 표면 거리 ≤ skin.
    pub toi: f32,
    /// B→A 단위 법선 (A를 밀어내는 방향).
    pub normal: Vec3,
    /// B **표면**의 접점 근사 (witness + 법선 × convex radius).
    pub point: Vec3,
}

/// 볼록 전용 경로 판정 (`manifold`의 것과 같은 규칙).
#[inline]
fn is_non_convex(s: &Shape) -> bool {
    matches!(s, Shape::Compound { .. } | Shape::Heightfield { .. })
}

/// 선형 셰이프 캐스트. `skin` = 이 거리 안이면 "닿았다"로 판정하는 피부 두께 —
/// 호출자는 보통 speculative margin을 넘겨, 정지 위치가 다음 스텝의 일반 접촉
/// 파이프라인(speculative 봉투)에 자연히 잡히게 한다.
///
/// 반환 규약:
/// - 시작부터 **코어가 겹치면** `None` — 기존 관통은 일반 파이프라인(EPA·bias) 소관이고,
///   CCD는 "이번 이동이 만드는 새 관통"만 막는다(설계: 임팩트 단일 소비).
/// - 시작이 표면 접촉(코어 분리·표면 거리 ≤ skin)이고 **접근 중**이면 `toi=0` 히트.
///   멀어지는 중이면 `None`(슬라이드·이탈을 막지 않는다).
/// - CA 반복 상한 도달은 **미스가 아니라 마지막 t의 히트**다 — 미세 전진의 누적은 접촉
///   직전이라는 뜻이고, 여기서 미스를 돌려주면 그 스텝의 나머지 이동이 관통한다.
pub fn shape_cast(
    sa: &Shape,
    pa: &Pose,
    motion: Vec3,
    sb: &Shape,
    pb: &Pose,
    skin: f32,
) -> Option<ShapeCastHit> {
    // **비볼록은 여기서 끝낸다** — 볼록 전용 경로다(디스패치는 `shape_cast_dispatch`).
    // 넘기면 GJK가 `support_core`의 `unreachable!`에 닿아 프로세스가 죽는다
    // (확인검수 rev_e50670ad BLOCKER: 공개 저수준 API가 trap 경로였다).
    if is_non_convex(sa) || is_non_convex(sb) {
        return None;
    }
    let (ra, rb) = (sa.radius(), sb.radius());
    let mut t = 0.0f32;
    let mut pose_a = *pa;
    let mut last: Option<(Vec3, Vec3)> = None; // (normal, point) — 상한 도달 방어용
    for _ in 0..64 {
        match gjk(sa, &pose_a, sb, pb) {
            GjkResult::Intersecting(_) => {
                // 코어 겹침. t=0이면 기존 관통 — CCD 소관 아님. t>0이면 전진식이
                // 과전진한 극단(이론상 비도달·수치 방어) — 직전 표면 정보로 히트 처리.
                return if t > 0.0 {
                    last.map(|(normal, point)| ShapeCastHit {
                        toi: t,
                        normal,
                        point,
                    })
                } else {
                    None
                };
            }
            GjkResult::Separated {
                distance,
                on_b,
                normal,
                ..
            } => {
                let sep = distance - ra - rb; // 코어 거리 → 표면 거리
                let point = on_b + normal * rb;
                if sep <= skin {
                    // 접촉 판정. 시작 접촉(t=0)에서 멀어지는 중이면 히트가 아니다.
                    if t == 0.0 && motion.dot(normal) >= 0.0 {
                        return None;
                    }
                    return Some(ShapeCastHit {
                        toi: t,
                        normal,
                        point,
                    });
                }
                last = Some((normal, point));
                // 법선 방향 접근 속도 (t 단위당). 볼록체 + 선형 이동에서 최근접 거리는
                // 이 속도보다 빨리 줄 수 없다 — 전진이 관통을 건너뛰지 못하는 이유.
                let approach = -motion.dot(normal);
                if approach <= 0.0 {
                    return None; // 최근접 방향으로 멀어지거나 평행 — 닿을 수 없다
                }
                // skin의 절반을 남기고 전진 — 다음 반복에서 `sep ≤ skin`으로 종결되도록.
                let advance = (sep - skin * 0.5) / approach;
                t += advance.max(0.0);
                if t > 1.0 {
                    return None; // 이번 이동 범위 안에서는 닿지 않는다
                }
                pose_a.pos = pa.pos + motion * t;
            }
        }
    }
    // 반복 상한 — 접촉 직전의 미세 전진 누적. 보수적으로 히트.
    last.map(|(normal, point)| ShapeCastHit {
        toi: t,
        normal,
        point,
    })
}

/// **회전 인지 셰이프 캐스트** (§4 CCD 예약 확장 — 회전 보수적 전진).
///
/// A가 `motion`(평행이동)과 함께 `pivot`을 중심으로 축 `axis`(단위)·총각 `angle`만큼
/// 회전하며 정지한 B와 처음 닿는 시각을 구한다. `pivot`은 회전 중심의 시작 위치
/// (바디 COM — 복합 자식은 자기 원점이 아니라 바디 중심으로 돈다):
/// `pos(t) = pivot + motion·t + R(angle·t)·(pa.pos − pivot)`, `rot(t) = R(angle·t)·pa.rot`.
///
/// 건전성: 표면 임의 점의 t-단위 이동은 `|motion·n|`(선형) + `|angle|·r_arc`(각) 이하다 —
/// `r_arc = |pa.pos − pivot| + bounding_radius(A)` (Mirtich 보수적 전진의 각 상한).
/// 따라서 `advance = gap / (선형 접근 + |angle|·r_arc)`는 관통을 건너뛸 수 없다.
///
/// 시작 표면 접촉의 접근 판정은 **witness 순간 속도**로 한다(각 상한은 여기선 과잉 —
/// 구르는 접촉이 전부 toi=0이 되어 슬라이드를 막는다): witness 점 속도
/// `motion + angle·(axis × (on_a − pivot))`의 법선 성분이 접근일 때만 toi=0.
///
/// 반환 규약은 [`shape_cast`]와 동일(시작 코어겹침=None·반복 상한=보수적 히트).
#[allow(clippy::too_many_arguments)]
pub fn shape_cast_rotational(
    sa: &Shape,
    pa: &Pose,
    motion: Vec3,
    axis: Vec3,
    angle: f32,
    pivot: Vec3,
    sb: &Shape,
    pb: &Pose,
    skin: f32,
) -> Option<ShapeCastHit> {
    // 볼록 전용 — 비볼록은 `shape_cast_dispatch_rotational`이 펼친다.
    if is_non_convex(sa) || is_non_convex(sb) {
        return None;
    }
    // **표현 불가한 각은 캐스트를 하지 않는다.** 아래 루프는 매 반복
    // `Quat::from_axis_angle`을 부르고 그 함수는 비유한·초대형 각에 패닉한다 —
    // 라이브러리가 호스트 입력 때문에 죽으면 안 되므로 여기서 값으로 거른다.
    // (패닉 계약을 각 생성자에 두는 이유는 그쪽 문서 참조: 조용한 항등 회전 금지.)
    if !angle.is_finite() || angle.abs() > maru_math::Quat::MAX_AXIS_ANGLE {
        return None;
    }
    let (ra, rb) = (sa.radius(), sb.radius());
    let offset = pa.pos - pivot;
    // 각 기여 상한 = 축 수직 오프셋 + **각 스윕 반지름**(코어의 축 수직 거리 —
    // 표면 = 코어 ⊕ 반지름 볼이고 볼은 중심 기준 회전 불변이라 이것으로 충분하다).
    // 외접 반지름보다 조여서 회전 불변 형상(구)의 전진이 각 항으로 불필요하게
    // 잘게 쪼개지지 않는다. 축 대비 로컬 기하는 그 축 회전 아래 불변이므로
    // t=0에 잰 값이 캐스트 내내 유효하다.
    let axis_local = pa.rot.conjugate().rotate(axis);
    let off_perp = (offset - axis * offset.dot(axis)).length();
    let r_arc = off_perp + sa.angular_sweep_radius(axis_local);
    let ang_bound = angle.abs() * r_arc;
    let mut t = 0.0f32;
    let mut pose_a = *pa;
    let mut last: Option<(Vec3, Vec3)> = None;
    for _ in 0..64 {
        match gjk(sa, &pose_a, sb, pb) {
            GjkResult::Intersecting(_) => {
                return if t > 0.0 {
                    last.map(|(normal, point)| ShapeCastHit {
                        toi: t,
                        normal,
                        point,
                    })
                } else {
                    None
                };
            }
            GjkResult::Separated {
                distance,
                on_a,
                on_b,
                normal,
            } => {
                let sep = distance - ra - rb;
                let point = on_b + normal * rb;
                if sep <= skin {
                    if t == 0.0 {
                        // witness 순간 접근 판정 — 각 상한이 아니라 실제 점 속도.
                        let w_vel = motion + axis.cross(on_a - pivot) * angle;
                        if w_vel.dot(normal) >= 0.0 {
                            return None;
                        }
                    }
                    return Some(ShapeCastHit {
                        toi: t,
                        normal,
                        point,
                    });
                }
                last = Some((normal, point));
                let approach = -motion.dot(normal) + ang_bound;
                if approach <= 0.0 {
                    return None;
                }
                let advance = (sep - skin * 0.5) / approach;
                t += advance.max(0.0);
                if t > 1.0 {
                    return None;
                }
                let rot_t = maru_math::Quat::from_axis_angle(axis, angle * t);
                pose_a.pos = pivot + motion * t + rot_t.rotate(offset);
                pose_a.rot = rot_t * pa.rot;
            }
        }
    }
    last.map(|(normal, point)| ShapeCastHit {
        toi: t,
        normal,
        point,
    })
}

/// 캐스트가 **답을 낼 수 없는** 두 가지 이유 — 어느 쪽도 "맞은 것이 없다"가 아니다.
///
/// 이 구분이 없으면 잘못된 입력이나 미완료 순회가 `None`(=미스)으로 위장되고, 호출자는
/// "지나갈 수 있다"고 오판해 지형을 통과한다 — 저수준 볼록 API를 공개 표면에서 뺀 것과
/// 같은 이유다(확인검수 rev_d5b32e22 BLOCKER/MAJOR).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CastReject {
    /// 산술 입력이 비유한이거나 표현 범위를 벗어났다(축·이동·skin·각). **호출자 버그**다.
    InvalidInput,
    /// 지형 셀 예산(`MAX_HEIGHTFIELD_CELLS_PER_QUERY`)을 넘겨 순회를 완료하지 못했다.
    BudgetExceeded,
}

/// 두 디스패처가 공유하는 산술 입력 검증 — **패닉 대신 값**.
///
/// 각만 검사하던 시절, `axis=(NaN,0,0)`에 유효한 각을 넣으면 내부에서 `t`가 NaN이 되어
/// `angle * t`가 NaN이 되고 `Quat::from_axis_angle`의 assert가 프로세스를 죽였다
/// (확인검수 rev_d5b32e22 BLOCKER — 패닉 입력 공간이 각에서 축·이동으로 옮겨갔을 뿐이었다).
fn cast_inputs_ok(motion: Vec3, skin: f32, pa: &Pose, pb: &Pose) -> bool {
    motion.is_finite()
        && skin.is_finite()
        && skin >= 0.0
        && pa.pos.is_finite()
        && pa.rot.is_finite()
        && pb.pos.is_finite()
        && pb.rot.is_finite()
}

/// 회전 인지 복합 디스패치 — 자식은 **바디 COM(pivot)** 중심으로 함께 돈다.
/// 비복합은 자기 자신 1건으로 동작이 같다.
///
/// `Err`는 **미스가 아니다** — `CastReject` 참조.
#[allow(clippy::too_many_arguments)]
pub fn try_shape_cast_dispatch_rotational(
    sa: &Shape,
    pa: &Pose,
    motion: Vec3,
    axis: Vec3,
    angle: f32,
    sb: &Shape,
    pb: &Pose,
    skin: f32,
) -> Result<Option<ShapeCastHit>, CastReject> {
    // 산술 입력은 **격자를 걷기 전에** 전부 검사한다. 축은 여기서 한 번 정규화해
    // 전 경로가 같은 값을 쓰게 한다(내부에서 각각 정규화하면 NaN 축이 살아남는다).
    if !cast_inputs_ok(motion, skin, pa, pb)
        // **비유한 축은 "축이 없음"이 아니라 잘못된 입력이다.** 0 축(회전 없음)과 달리
        // NaN 축은 호출자 버그이고, 걸러내지 않으면 내부 `t`가 NaN이 되어 각 생성자의
        // assert에 닿는다(확인검수 rev_d5b32e22 BLOCKER).
        || !axis.is_finite()
        || !angle.is_finite()
        || angle.abs() > maru_math::Quat::MAX_AXIS_ANGLE
    {
        return Err(CastReject::InvalidInput);
    }
    let axis = match axis.try_normalize() {
        Some(a) => a,
        // 축 길이가 0이면 회전이 없다 — 선형 캐스트와 같은 답이다(오류가 아니다).
        None => return try_shape_cast_dispatch(sa, pa, motion, sb, pb, skin),
    };
    let mut budget = crate::MAX_HEIGHTFIELD_CELLS_PER_QUERY;
    let compose = |parent: &Pose, child: &Pose| Pose {
        pos: parent.pos + parent.rot.rotate(child.pos),
        rot: parent.rot * child.rot,
    };
    let cast_one =
        |csa: &Shape, cpa: &Pose, budget: &mut usize| -> Result<Option<ShapeCastHit>, CastReject> {
            Ok(match sb {
                // **하이트필드는 볼록이 아니다** — 그냥 넘기면 GJK가 `support_core`의
                // `unreachable!`에 닿아 trap이 된다(검수 rev_1e81559e BLOCKER: 회전 CCD를 켠
                // 블레이드를 지형 옆에 두면 선속도 0에서도 즉시 재현). 선형 경로와 같은 규칙으로
                // 스윕 봉투가 덮는 셀만 삼각형으로 펼친다. 회전 봉투는 외접 반지름으로 넓힌다.
                Shape::Heightfield { .. } => {
                    let hf = sb.heightfield_ref().expect("Heightfield 변종");
                    // **자식은 자기 원점이 아니라 부모 피벗 주위를 돈다.** 자식 반경만 넓히면
                    // COM에서 10m 떨어진 자식이 90° 도는 궤도를 못 덮어, 캐스트가 `None`을
                    // 내면서 자식이 지형을 관통한다(확인검수 rev_7efe4e35 BLOCKER).
                    // 봉투는 피벗 중심의 궤도 반경(|자식-피벗| + 자식 반경)으로 잡는다.
                    let orbit_r = (cpa.pos - pa.pos).length() + csa.bounding_radius();
                    let e = Vec3::new(orbit_r, orbit_r, orbit_r);
                    let a0 = crate::Aabb::new(pa.pos - e, pa.pos + e);
                    let sweep = a0.merge(crate::Aabb::new(a0.min + motion, a0.max + motion));
                    let expanded = crate::Aabb::new(
                        sweep.min - Vec3::new(skin, skin, skin),
                        sweep.max + Vec3::new(skin, skin, skin),
                    );
                    let local = crate::heightfield::world_aabb_to_local(expanded, pb);
                    let Some((cx0, cz0, cx1, cz1)) = hf.cell_range(local) else {
                        return Ok(None);
                    };
                    let mut best: Option<ShapeCastHit> = None;
                    // 선형 경로와 **같은 가지치기**: 타일 높이 범위가 스윕 Y 밴드와 안 겹치면
                    // 그 타일의 셀은 보지 않는다. 궤도 봉투는 넓으므로 이게 없으면 최악이
                    // 격자 전체다(확인검수 rev_d390e73d BLOCKER — 회전 분기만 예산 밖이었다).
                    let (y_lo, y_hi) = (local.min.y, local.max.y);
                    // **타일을 바깥 루프로.** 셀을 돌면서 타일을 조회하면 일은 줄어도 반복은
                    // 그대로다 — 후보가 0개인 1024×1024 격자에서 100만 번 돈다
                    // (확인검수 rev_1a31d83f: 내가 고쳤다고 보고했는데 치환이 조용히 실패했고
                    //  확인하지 않았다. 이제 헬퍼 하나만 쓴다 — 같은 코드를 두 번 적지 않는다).
                    // **작업량 상한.** 타일 가지치기는 후보가 0인 지형의 100만 셀 반복을
                    // 없애지만, Y 밴드가 실제로 겹치면 상한이 없다 — 1024×1024 지형을 덮는
                    // CCD 바디 하나가 서브스텝마다 100만 셀·200만 삼각형 캐스트를 돌려
                    // wasm 프레임을 멈춘다(확인검수 rev_d5b32e22 MAJOR). 예산을 넘으면
                    // **자르지 않고 오류로 올린다**: 잘라서 `None`을 내면 진짜 접촉을 놓쳐
                    // 지형을 통과하고, 가짜 `toi=0` 히트를 만들면 정상 장면이 영구 동결된다
                    // (rev_d390e73d에서 실제로 그랬다). 보수적 정지 정책은 **소비자**가 고른다.
                    let complete = crate::heightfield::for_each_overlapping_cell(
                        &hf,
                        (cx0, cz0, cx1, cz1),
                        y_lo,
                        y_hi,
                        |cx, cz| {
                            if *budget == 0 {
                                return false;
                            }
                            *budget -= 1;
                            for t in 0..2u8 {
                                let tri = Shape::Triangle {
                                    points: hf.triangle(cx, cz, t),
                                    radius: 0.0,
                                };
                                if let Some(h) = shape_cast_rotational(
                                    csa, cpa, motion, axis, angle, pa.pos, &tri, pb, skin,
                                ) {
                                    if best.as_ref().is_none_or(|b| h.toi < b.toi) {
                                        best = Some(h);
                                    }
                                }
                            }
                            true
                        },
                    );
                    if !complete {
                        return Err(CastReject::BudgetExceeded);
                    }
                    best
                }
                Shape::Compound { children } => children
                    .iter()
                    .filter_map(|(cpb, csb)| {
                        shape_cast_rotational(
                            csa,
                            cpa,
                            motion,
                            axis,
                            angle,
                            pa.pos,
                            csb,
                            &compose(pb, cpb),
                            skin,
                        )
                    })
                    .min_by(|x, y| x.toi.total_cmp(&y.toi)),
                _ => shape_cast_rotational(csa, cpa, motion, axis, angle, pa.pos, sb, pb, skin),
            })
        };
    match sa {
        Shape::Compound { children } => {
            let mut best: Option<ShapeCastHit> = None;
            for (cp, cs) in children {
                if let Some(h) = cast_one(cs, &compose(pa, cp), &mut budget)? {
                    if best.as_ref().is_none_or(|b| h.toi < b.toi) {
                        best = Some(h);
                    }
                }
            }
            Ok(best)
        }
        _ => cast_one(sa, pa, &mut budget),
    }
}

/// 복합 디스패치 셰이프 캐스트 — 자식쌍의 earliest TOI (동률은 자식 순서 — 결정적).
///
/// `Err`는 **미스가 아니다** — `CastReject` 참조.
pub fn try_shape_cast_dispatch(
    sa: &Shape,
    pa: &Pose,
    motion: Vec3,
    sb: &Shape,
    pb: &Pose,
    skin: f32,
) -> Result<Option<ShapeCastHit>, CastReject> {
    if !cast_inputs_ok(motion, skin, pa, pb) {
        return Err(CastReject::InvalidInput);
    }
    let compose = |parent: &Pose, child: &Pose| Pose {
        pos: parent.pos + parent.rot.rotate(child.pos),
        rot: parent.rot * child.rot,
    };
    let mut best: Option<ShapeCastHit> = None;
    let mut budget = crate::MAX_HEIGHTFIELD_CELLS_PER_QUERY;
    // **B쪽 디스패치를 A쪽 루프에서 분리한다.** 예전에는 `match (sa, sb)`가 `(Compound, _)`를
    // 먼저 잡아, 복합 바디가 지형으로 스윕하면 각 자식이 **볼록 전용** `shape_cast`에 지형을
    // 그대로 넘겼다 — 그 함수는 비볼록에 `None`을 돌려주므로 "맞은 것 없음"이 되어 복합
    // 바디가 지형을 그냥 통과했다(자기 검수 2026-07-30, 회전 분기는 이미 이 구조였다).
    let cast_one = |csa: &Shape,
                    cpa: &Pose,
                    best: &mut Option<ShapeCastHit>,
                    budget: &mut usize|
     -> Result<(), CastReject> {
        let mut consider = |h: ShapeCastHit| {
            if best.as_ref().is_none_or(|b| h.toi < b.toi) {
                *best = Some(h);
            }
        };
        match sb {
            Shape::Compound { children } => {
                for (cpb, csb) in children {
                    if let Some(h) = shape_cast(csa, cpa, motion, csb, &compose(pb, cpb), skin) {
                        consider(h);
                    }
                }
            }
            // 하이트필드가 대상이면 스윕 봉투가 덮는 셀만 삼각형으로 펼친다.
            // CCD의 유일한 대상이 static이므로(설계 §4 linear cast) 이 경로가 지형 관통을 막는다.
            Shape::Heightfield { .. } => {
                let hf = sb.heightfield_ref().expect("Heightfield 변종");
                let a0 = crate::transformed_aabb(csa.local_aabb(), cpa);
                let sweep = a0.merge(crate::Aabb::new(a0.min + motion, a0.max + motion));
                let expanded = crate::Aabb::new(
                    sweep.min - Vec3::new(skin, skin, skin),
                    sweep.max + Vec3::new(skin, skin, skin),
                );
                let local = crate::heightfield::world_aabb_to_local(expanded, pb);
                let Some(range) = hf.cell_range(local) else {
                    return Ok(());
                };
                // **타일 높이 범위로 먼저 쳐낸다.** 스윕의 Y 밴드와 안 겹치는 타일은 셀을
                // 하나도 볼 필요가 없다 — 이게 없으면 총알이 평평한 통로를 지나가는 것만으로
                // 수천 셀을 검사하고, 그 비용을 피하려던 "예산 초과 = 가짜 히트"가 정상 장면을
                // 영구 동결시켰다(확인검수 rev_d390e73d BLOCKER). 가지치기는 근사가 아니라
                // 정확하다: 타일 밖 높이는 그 범위 안에 있다는 것이 build_tiles의 계약이다.
                let (y_lo, y_hi) = (local.min.y, local.max.y);
                // **작업량 상한.** 예산을 넘으면 자르지 않고 오류로 올린다 — 근거는
                // 회전 분기의 같은 자리 주석에 있다(자르면 관통, 가짜 히트면 영구 동결).
                let complete = crate::heightfield::for_each_overlapping_cell(
                    &hf,
                    range,
                    y_lo,
                    y_hi,
                    |cx, cz| {
                        if *budget == 0 {
                            return false;
                        }
                        *budget -= 1;
                        for t in 0..2u8 {
                            let tri = Shape::Triangle {
                                points: hf.triangle(cx, cz, t),
                                radius: 0.0,
                            };
                            if let Some(h) = shape_cast(csa, cpa, motion, &tri, pb, skin) {
                                consider(h);
                            }
                        }
                        true
                    },
                );
                if !complete {
                    return Err(CastReject::BudgetExceeded);
                }
            }
            _ => {
                if let Some(h) = shape_cast(csa, cpa, motion, sb, pb, skin) {
                    consider(h);
                }
            }
        }
        Ok(())
    };
    match sa {
        Shape::Compound { children } => {
            for (cp, cs) in children {
                cast_one(cs, &compose(pa, cp), &mut best, &mut budget)?;
            }
        }
        _ => cast_one(sa, pa, &mut best, &mut budget)?,
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use maru_math::Quat;

    fn pose(x: f32, y: f32, z: f32) -> Pose {
        Pose {
            pos: Vec3::new(x, y, z),
            rot: Quat::IDENTITY,
        }
    }

    /// 순수 회전 스윕: z축 정렬 블레이드가 y축으로 90° 돌며 +x 벽에 닿는다 —
    /// 선형 캐스트(motion=0)는 원리적으로 잡을 수 없는 히트.
    #[test]
    fn pure_rotation_sweep_hits_wall() {
        let blade = Shape::Box {
            half: Vec3::new(1.5, 0.05, 0.05),
            radius: 0.0,
        };
        let wall = Shape::Box {
            half: Vec3::new(0.05, 5.0, 5.0),
            radius: 0.0,
        };
        // 블레이드 장축을 z로 돌려두고, y축 -90°(z→x) 회전 스윕.
        let pa = Pose {
            pos: Vec3::ZERO,
            rot: Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_2),
        };
        let pb = pose(1.2, 0.0, 0.0);
        // 선형 캐스트는 이동이 0이라 아무것도 못 잡는다.
        assert!(shape_cast(&blade, &pa, Vec3::ZERO, &wall, &pb, 0.02).is_none());
        let hit = shape_cast_rotational(
            &blade,
            &pa,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            core::f32::consts::FRAC_PI_2,
            pa.pos,
            &wall,
            &pb,
            0.02,
        )
        .expect("회전 스윕이 벽을 놓쳤다");
        // 팁 반경 1.5가 x=1.15 근면에 닿는 각도: cos보다 훨씬 전 — toi ∈ (0,1).
        assert!(hit.toi > 0.0 && hit.toi < 1.0, "toi={}", hit.toi);
        assert!(
            hit.normal.x < -0.9,
            "벽 법선이 −x여야 한다: {:?}",
            hit.normal
        );
    }

    /// 회전 0이면 선형 캐스트와 같은 답 (같은 원시 — 게이트 없는 경로 검증).
    #[test]
    fn zero_rotation_matches_linear_cast() {
        let s = Shape::Sphere { radius: 0.5 };
        let wall = Shape::Box {
            half: Vec3::new(0.05, 5.0, 5.0),
            radius: 0.0,
        };
        let motion = Vec3::new(20.0, 0.0, 0.0);
        let lin = shape_cast(
            &s,
            &pose(0.0, 0.0, 0.0),
            motion,
            &wall,
            &pose(10.0, 0.0, 0.0),
            0.02,
        )
        .expect("선형 미스");
        let rot = shape_cast_rotational(
            &s,
            &pose(0.0, 0.0, 0.0),
            motion,
            Vec3::new(0.0, 1.0, 0.0),
            0.0,
            Vec3::ZERO,
            &wall,
            &pose(10.0, 0.0, 0.0),
            0.02,
        )
        .expect("회전0 미스");
        assert!(
            (lin.toi - rot.toi).abs() < 1e-4,
            "{} vs {}",
            lin.toi,
            rot.toi
        );
    }

    /// 구르는 접촉은 toi=0으로 막지 않는다 — witness 순간 속도가 접선이면 None.
    #[test]
    fn rolling_surface_contact_is_not_a_hit() {
        let s = Shape::Sphere { radius: 0.5 };
        let ground = Shape::Box {
            half: Vec3::new(10.0, 0.5, 10.0),
            radius: 0.0,
        };
        // 표면 접촉(간격 0.01 ≤ skin 0.02), 접선 이동 + 구름 회전.
        let pa = pose(0.0, 1.01, 0.0);
        let pb = pose(0.0, 0.0, 0.0);
        let hit = shape_cast_rotational(
            &s,
            &pa,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            1.0,
            pa.pos,
            &ground,
            &pb,
            0.02,
        );
        assert!(hit.is_none(), "구르는 접촉이 toi=0으로 잡혔다: {hit:?}");
        // 같은 자리에서 법선 방향으로 접근하면 toi=0 히트다.
        let hit = shape_cast_rotational(
            &s,
            &pa,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            1.0,
            pa.pos,
            &ground,
            &pb,
            0.02,
        );
        assert!(hit.is_some_and(|h| h.toi == 0.0), "누르는 접촉이 놓쳤다");
    }

    /// 해석해 대조: 반지름 0.5 구가 x=+10에 있는 두께 0.1 벽(박스)을 향해 20만큼 이동.
    /// 벽 근면은 x=9.95 → 구 표면이 닿는 중심 위치 x=9.45 → toi=(9.45-0)/20=0.4725.
    #[test]
    fn sphere_into_thin_wall_matches_analytic_toi() {
        let s = Shape::Sphere { radius: 0.5 };
        let wall = Shape::Box {
            half: Vec3::new(0.05, 5.0, 5.0),
            radius: 0.0,
        };
        let hit = shape_cast(
            &s,
            &pose(0.0, 0.0, 0.0),
            Vec3::new(20.0, 0.0, 0.0),
            &wall,
            &pose(10.0, 0.0, 0.0),
            1e-3,
        )
        .expect("정면 캐스트가 벽을 놓쳤다");
        assert!(
            (hit.toi - 0.4725).abs() < 2e-3,
            "toi={} (해석해 0.4725)",
            hit.toi
        );
        assert!(
            (hit.normal.x + 1.0).abs() < 1e-3,
            "법선이 -x여야 한다: {:?}",
            hit.normal
        );
    }

    /// 빗나가는 캐스트 · 멀어지는 캐스트는 None.
    #[test]
    fn miss_and_receding_return_none() {
        let s = Shape::Sphere { radius: 0.5 };
        let wall = Shape::Box {
            half: Vec3::new(0.05, 1.0, 1.0),
            radius: 0.0,
        };
        // 옆으로 빗나감 (벽 y 범위 1.0, 구는 y=5)
        assert!(shape_cast(
            &s,
            &pose(0.0, 5.0, 0.0),
            Vec3::new(20.0, 0.0, 0.0),
            &wall,
            &pose(10.0, 0.0, 0.0),
            1e-3
        )
        .is_none());
        // 반대 방향
        assert!(shape_cast(
            &s,
            &pose(0.0, 0.0, 0.0),
            Vec3::new(-20.0, 0.0, 0.0),
            &wall,
            &pose(10.0, 0.0, 0.0),
            1e-3
        )
        .is_none());
    }

    /// 시작이 표면 접촉이고 접근 중이면 toi=0, 멀어지면 None (슬라이드 비차단).
    #[test]
    fn touching_start_hits_only_when_approaching() {
        let s = Shape::Sphere { radius: 0.5 };
        let floor = Shape::Box {
            half: Vec3::new(10.0, 1.0, 10.0),
            radius: 0.0,
        };
        // 바닥 위에 정확히 얹힘 (표면 거리 0)
        let on = pose(0.0, 1.5, 0.0);
        let fp = pose(0.0, 0.0, 0.0);
        let hit = shape_cast(&s, &on, Vec3::new(0.0, -1.0, 0.0), &floor, &fp, 1e-3)
            .expect("접근 중 접촉 시작이 히트가 아니다");
        assert_eq!(hit.toi, 0.0);
        assert!(shape_cast(&s, &on, Vec3::new(0.0, 2.0, 0.0), &floor, &fp, 1e-3).is_none());
        // 순수 슬라이드(접선 이동)도 막지 않는다 — motion·normal = 0 은 "멀어짐" 쪽.
        assert!(shape_cast(&s, &on, Vec3::new(3.0, 0.0, 0.0), &floor, &fp, 1e-3).is_none());
    }

    /// 코어가 이미 겹친 시작은 None — 일반 접촉 파이프라인 소관 (계약).
    #[test]
    fn initially_intersecting_core_is_not_a_cast_hit() {
        let s = Shape::Sphere { radius: 0.5 };
        let b = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        assert!(shape_cast(
            &s,
            &pose(0.0, 0.5, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            &b,
            &pose(0.0, 0.0, 0.0),
            1e-3
        )
        .is_none());
    }

    /// 박스-박스 (회전된 대상 포함) — CA가 볼록 일반 쌍에서 도는지.
    #[test]
    fn box_cast_onto_rotated_box_hits() {
        let a = Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        };
        let b = Shape::Box {
            half: Vec3::new(2.0, 0.5, 2.0),
            radius: 0.0,
        };
        let target = Pose {
            pos: Vec3::new(8.0, 0.0, 0.0),
            rot: Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.3),
        };
        let hit = shape_cast(
            &a,
            &pose(0.0, 0.0, 0.0),
            Vec3::new(12.0, 0.0, 0.0),
            &b,
            &target,
            1e-3,
        )
        .expect("회전 박스 캐스트 미스");
        assert!(hit.toi > 0.0 && hit.toi < 1.0, "toi={}", hit.toi);
        assert!(hit.normal.x < -0.5, "대체로 -x 법선: {:?}", hit.normal);
    }
}
