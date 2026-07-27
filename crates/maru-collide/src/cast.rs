//! 선형 셰이프 캐스트 — CCD linear cast의 기하 원시 (설계 §4 CCD 계약 v0.5).
//!
//! A를 `motion`만큼 **평행이동**시키며 정지한 B와 처음 닿는 시각을 conservative
//! advancement로 구한다. 회전은 캐스트 중 무시한다(linear cast — Jolt LinearCast
//! stop-at-TOI와 동형, 설계 근거). 볼록체 + 선형 이동이므로 "최근접 방향 접근 속도로
//! 최근접 거리를 나눈 전진"은 관통을 건너뛸 수 없다 — CA의 건전성 근거.

use crate::gjk::{gjk, GjkResult};
use crate::{Pose, Shape};
use maru_math::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct ShapeCastHit {
    /// `motion`의 분율 [0,1] — 이 시각에 표면 거리 ≤ skin.
    pub toi: f32,
    /// B→A 단위 법선 (A를 밀어내는 방향).
    pub normal: Vec3,
    /// B **표면**의 접점 근사 (witness + 법선 × convex radius).
    pub point: Vec3,
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

/// 복합 디스패치 셰이프 캐스트 — 자식쌍의 earliest TOI (동률은 자식 순서 — 결정적).
pub fn shape_cast_dispatch(
    sa: &Shape,
    pa: &Pose,
    motion: Vec3,
    sb: &Shape,
    pb: &Pose,
    skin: f32,
) -> Option<ShapeCastHit> {
    let compose = |parent: &Pose, child: &Pose| Pose {
        pos: parent.pos + parent.rot.rotate(child.pos),
        rot: parent.rot * child.rot,
    };
    let mut best: Option<ShapeCastHit> = None;
    let mut consider = |h: ShapeCastHit| {
        if best.as_ref().is_none_or(|b| h.toi < b.toi) {
            best = Some(h);
        }
    };
    match (sa, sb) {
        (Shape::Compound { children: ca }, Shape::Compound { children: cb }) => {
            for (cpa, csa) in ca {
                let wpa = compose(pa, cpa);
                for (cpb, csb) in cb {
                    if let Some(h) = shape_cast(csa, &wpa, motion, csb, &compose(pb, cpb), skin) {
                        consider(h);
                    }
                }
            }
        }
        (Shape::Compound { children }, _) => {
            for (cp, cs) in children {
                if let Some(h) = shape_cast(cs, &compose(pa, cp), motion, sb, pb, skin) {
                    consider(h);
                }
            }
        }
        (_, Shape::Compound { children }) => {
            for (cp, cs) in children {
                if let Some(h) = shape_cast(sa, pa, motion, cs, &compose(pb, cp), skin) {
                    consider(h);
                }
            }
        }
        _ => return shape_cast(sa, pa, motion, sb, pb, skin),
    }
    best
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
