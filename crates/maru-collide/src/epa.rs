//! EPA — 교차 쌍의 침투 깊이/법선 (설계 §4.2).
//!
//! GJK의 Intersecting 사면체를 시드로 민코프스키 차 표면을 확장, 원점 최근접 면을 찾는다.
//! 반환 `normal`은 **A를 +normal 방향으로 depth만큼 밀면 분리**되는 방향(= GJK Separated의
//! B→A 관례와 동일 — 두 경로의 접촉 생성이 부호 분기 없이 합류).
//! 결정론: 최근접 면 동률은 낮은 face 인덱스 우선, 반복 상한 고정.

use crate::gjk::SupportPoint;
use crate::tolerance::{area_eps, bary_denom_eps, characteristic_length, length_eps};
use crate::{Pose, Shape};
use maru_math::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct Penetration {
    pub depth: f32,
    /// A를 이 방향(+)으로 밀면 분리 (단위 벡터).
    pub normal: Vec3,
    pub on_a: Vec3,
    pub on_b: Vec3,
}

const MAX_ITER: usize = 64;
/// 폴리토프 크기 하드 상한 — 퇴화 시드가 들어와도 **절대 무한 확장하지 않는다**(설계 §6
/// "유한 자원" 규율). 초과 시 현재 최근접 면으로 결정적 마감.
const MAX_FACES: usize = 512;

#[derive(Clone, Copy)]
struct Face {
    v: [usize; 3],
    n: Vec3, // 바깥쪽 단위 법선
    d: f32,  // 원점→면 거리 (>= 0)
    alive: bool,
}

fn make_face(
    verts: &[SupportPoint],
    i: usize,
    j: usize,
    k: usize,
    interior: Vec3,
    eps_area: f32,
) -> Option<Face> {
    let (a, b, c) = (verts[i].w, verts[j].w, verts[k].w);
    let mut n = (b - a).cross(c - a);
    // cross 길이 = 면적(L²) → L² 임계와 비교 (length_sq는 L⁴이라 차원 불일치)
    if n.length() < eps_area {
        return None; // 퇴화 면
    }
    let len = n.length();
    n = n * (1.0 / len);
    // 바깥 법선 + **CCW 와인딩 일관성 강제**: horizon 에지 상쇄는 인접 면이 공유 에지를
    // 서로 반대 방향으로 갖는다는 가정에 선다 — 와인딩이 뒤집힌 면이 섞이면 상쇄가 실패해
    // 균열/퇴화 면이 생기고 d=0 조기 수렴한다(박스 겹침 depth=0 버그로 실증).
    let mut v = [i, j, k];
    if n.dot(a - interior) < 0.0 {
        n = -n;
        v.swap(1, 2);
    }
    let d = n.dot(a).max(0.0);
    Some(Face {
        v,
        n,
        d,
        alive: true,
    })
}

pub fn epa(seed: [SupportPoint; 4], sa: &Shape, pa: &Pose, sb: &Shape, pb: &Pose) -> Penetration {
    let scale = characteristic_length(sa, sb);
    let eps_len = length_eps(scale);
    let eps_area = area_eps(scale);
    let denom_eps = bary_denom_eps(scale);
    let mut verts: Vec<SupportPoint> = seed.to_vec();
    let interior = (verts[0].w + verts[1].w + verts[2].w + verts[3].w) * 0.25;
    let mut faces: Vec<Face> = Vec::with_capacity(32);
    for (i, j, k) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
        if let Some(f) = make_face(&verts, i, j, k, interior, eps_area) {
            faces.push(f);
        }
    }
    // 상한 종료 폴백용 — 마지막으로 확장을 시도한(=당시 최근접) 면
    let mut last_best = faces[0];

    for _ in 0..MAX_ITER {
        // 최근접 면 (동률: 낮은 인덱스 — 결정론)
        let mut best = usize::MAX;
        let mut best_d = f32::INFINITY;
        for (fi, f) in faces.iter().enumerate() {
            if f.alive && f.d < best_d {
                best_d = f.d;
                best = fi;
            }
        }
        if best == usize::MAX {
            break; // 방어: 면 소진
        }
        let f = faces[best];
        last_best = f;
        let on_a = pa.support_core(sa, f.n);
        let on_b = pb.support_core(sb, -f.n);
        let w = SupportPoint {
            w: on_a - on_b,
            on_a,
            on_b,
        };
        // 확장량이 없으면 수렴
        if w.w.dot(f.n) - f.d < eps_len {
            return finish(&verts, f, denom_eps);
        }
        if faces.len() > MAX_FACES {
            break; // 상한 도달 → 아래 alive 최근접 면으로 마감
        }
        // 새 정점에서 보이는 면 제거, horizon 에지 수집
        let wi = verts.len();
        verts.push(w);
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for face in faces.iter_mut() {
            if face.alive && face.n.dot(w.w - verts[face.v[0]].w) > 0.0 {
                face.alive = false;
                for e in [
                    (face.v[0], face.v[1]),
                    (face.v[1], face.v[2]),
                    (face.v[2], face.v[0]),
                ] {
                    // 역방향 에지가 이미 있으면 내부 에지 → 상쇄
                    if let Some(pos) = horizon.iter().position(|&(a, b)| a == e.1 && b == e.0) {
                        horizon.swap_remove(pos);
                    } else {
                        horizon.push(e);
                    }
                }
            }
        }
        for &(a, b) in &horizon {
            if let Some(nf) = make_face(&verts, a, b, wi, interior, eps_area) {
                faces.push(nf);
            }
        }
    }

    // 상한 도달: **alive 면 중** 최근접으로 마감 (리뷰 MAJOR 반영 — 구 코드는 best=0에서
    // 시작해 죽은 face 0의 작은 d가 기준이 되면 어떤 alive 면도 선택되지 않아 제거된 면의
    // depth/witness를 반환했다). alive가 하나도 없으면 마지막 확장 대상 면으로 폴백.
    let mut best = usize::MAX;
    let mut best_d = f32::INFINITY;
    for (fi, f) in faces.iter().enumerate() {
        if f.alive && f.d < best_d {
            best_d = f.d;
            best = fi;
        }
    }
    if best != usize::MAX {
        finish(&verts, faces[best], denom_eps)
    } else {
        finish(&verts, last_best, denom_eps)
    }
}

fn finish(verts: &[SupportPoint], f: Face, denom_eps: f32) -> Penetration {
    // 원점을 면 평면에 투영한 점 p = n·d 의 바리센트릭으로 witness 복원
    let (pa, pb, pc) = (verts[f.v[0]], verts[f.v[1]], verts[f.v[2]]);
    let p = f.n * f.d;
    let (v0, v1, v2) = (pb.w - pa.w, pc.w - pa.w, p - pa.w);
    let (d00, d01, d11) = (v0.dot(v0), v0.dot(v1), v1.dot(v1));
    let (d20, d21) = (v2.dot(v0), v2.dot(v1));
    let denom = d00 * d11 - d01 * d01;
    let (bv, cv) = if denom.abs() > denom_eps {
        (
            (d11 * d20 - d01 * d21) / denom,
            (d00 * d21 - d01 * d20) / denom,
        )
    } else {
        (0.0, 0.0)
    };
    let av = 1.0 - bv - cv;
    Penetration {
        depth: f.d,
        normal: -f.n, // 바깥 법선의 반대 = A를 밀어낼 방향 (B→A 관례)
        on_a: pa.on_a * av + pb.on_a * bv + pc.on_a * cv,
        on_b: pa.on_b * av + pb.on_b * bv + pc.on_b * cv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gjk::{gjk, GjkResult};
    use maru_math::Quat;

    fn pose(x: f32, y: f32, z: f32) -> Pose {
        Pose {
            pos: Vec3::new(x, y, z),
            rot: Quat::IDENTITY,
        }
    }

    fn run_epa(sa: &Shape, pa: &Pose, sb: &Shape, pb: &Pose) -> Penetration {
        match gjk(sa, pa, sb, pb) {
            GjkResult::Intersecting(seed) => epa(seed, sa, pa, sb, pb),
            r => panic!("expected intersecting, got {r:?}"),
        }
    }

    #[test]
    fn box_box_x_overlap_depth() {
        // half=1 두 박스, 중심距 x=1.4 → 겹침 0.6, 법선 = A를 −x로 밀기(A가 왼쪽)
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let p = run_epa(&s, &pose(0.0, 0.0, 0.0), &s, &pose(1.4, 0.0, 0.0));
        assert!((p.depth - 0.6).abs() < 1e-3, "depth={}", p.depth);
        assert!((p.normal.x + 1.0).abs() < 1e-3, "normal={:?}", p.normal);
        // 검산: A를 normal*depth만큼 밀면 접촉 상태여야 함
        let moved = pose(
            p.normal.x * p.depth,
            p.normal.y * p.depth,
            p.normal.z * p.depth,
        );
        match gjk(&s, &moved, &s, &pose(1.4, 0.0, 0.0)) {
            GjkResult::Separated { distance, .. } => assert!(distance < 1e-2),
            GjkResult::Intersecting(_) => {} // 수치 경계 — 허용
        }
    }

    #[test]
    fn box_box_y_shallow() {
        let s = Shape::Box {
            half: Vec3::new(2.0, 0.5, 2.0),
            radius: 0.0,
        };
        let p = run_epa(&s, &pose(0.0, 0.9, 0.0), &s, &pose(0.0, 0.0, 0.0));
        assert!((p.depth - 0.1).abs() < 1e-3, "depth={}", p.depth);
        assert!((p.normal.y - 1.0).abs() < 1e-3, "normal={:?}", p.normal); // A는 위 → +y로 탈출
    }

    /// 리뷰 지적: 구 버전은 이름과 달리 **witness에 대해 아무것도 단정하지 않았다**.
    /// witness는 두 표면 위의 대응점이어야 하며, 그 차이를 법선에 투영하면 침투 깊이가 나온다.
    #[test]
    fn witness_points_satisfy_depth_relation() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let pa = pose(0.2, 0.1, -0.15);
        let pb = pose(0.0, 0.0, 0.0);
        let p = run_epa(&s, &pa, &s, &pb);
        assert!(p.depth > 0.0, "깊이 {}", p.depth);
        assert!((p.normal.length() - 1.0).abs() < 1e-4);
        // 계약: (on_b − on_a)·n = depth  (A를 +n으로 depth만큼 밀면 표면이 만난다)
        let proj = (p.on_b - p.on_a).dot(p.normal);
        assert!(
            (proj - p.depth).abs() < 1.0e-3,
            "witness 관계 위반: (on_b−on_a)·n={proj} vs depth={}",
            p.depth
        );
        // witness는 각 박스의 표면 위 (로컬 좌표 성분 하나가 half와 일치)
        for (w, pose_) in [(p.on_a, pa), (p.on_b, pb)] {
            let local = pose_.rot.conjugate().rotate(w - pose_.pos);
            let on_surface = (local.x.abs() - 1.0).abs() < 1e-3
                || (local.y.abs() - 1.0).abs() < 1e-3
                || (local.z.abs() - 1.0).abs() < 1e-3;
            assert!(on_surface, "witness가 표면 위가 아님: local={local:?}");
        }
    }

    #[test]
    fn deterministic_repeat() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 0.7, 1.3),
            radius: 0.0,
        };
        let pb = Pose {
            pos: Vec3::new(0.6, 0.4, -0.3),
            rot: Quat::from_axis_angle(Vec3::new(1.0, 0.4, 0.2), 0.5),
        };
        let a = run_epa(&s, &pose(0.0, 0.0, 0.0), &s, &pb);
        let b = run_epa(&s, &pose(0.0, 0.0, 0.0), &s, &pb);
        assert_eq!(a.depth.to_bits(), b.depth.to_bits());
        assert_eq!(a.normal.x.to_bits(), b.normal.x.to_bits());
    }
}
