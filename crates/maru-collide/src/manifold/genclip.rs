//! 변수 정점 볼록 면 클리퍼 — 일반 볼록(hull·원기둥) 면 접촉용.
//!
//! `clip.rs`의 클리퍼는 **박스 전용**이다: 참조면·입사면 모두 정점 4개 고정(`[Vec3; 4]`),
//! 측면 루프 `0..4`, 계보를 4비트 마스크 둘로 담는다. hull 면은 정점이 최대 32개라
//! 그 어느 것도 성립하지 않는다. 여기서는 정점 수를 변수로 두고, 계보를 인덱스로 담는다
//! (`feature.rs` 코덱).
//!
//! **박스 경로는 건드리지 않는다** — 박스-박스 접점의 비트 동일성이 절대 조건이라
//! 새 경로를 따로 둔다(같은 바디 페어가 두 경로를 동시에 타지 않는다: 디스패치가
//! 셰이프 쌍으로 갈린다).
//!
//! ## 계보 추적이 정점이 아니라 **에지**에 붙는 이유
//! Sutherland–Hodgman은 다단이다. 2번째 측면을 자를 때 만나는 정점 v0·v1은 이미
//! 1번째 측면이 만든 교점일 수 있다. 그래서 "v0과 v1의 계보를 합친다"(박스 경로의
//! 비트마스크 합집합)로는 무엇을 잘랐는지 알 수 없다. 대신 **폴리곤의 각 에지가
//! 어디서 왔는지**(입사면의 원본 에지 j, 또는 참조면 측면 t)를 들고 다닌다. 볼록
//! 폴리곤을 반평면으로 자르면 이 불변식이 유지된다 — 새로 생기는 에지는 자른 평면
//! 자신뿐이다. 교점의 계보는 곧 **잘린 에지의 출처**로 결정된다:
//!
//! - 입사 에지 j를 측면 s로 자름 → `EdgeCross { a: j, b: s }`
//! - 측면 t를 측면 s로 자름 → 두 측면이 만나는 **참조면 정점** → `ReferenceVertex { a: v }`
//!
//! 뒤엣것은 t·s가 인접할 때만 참조면 정점이다. 부동소수 퇴화로 비인접 측면이 만나면
//! 계보를 특정할 수 없으므로 **모른다고 표시한다**(`UNKNOWN_INDEX`) — 코덱이 범위 밖으로
//! 거절하고 호출자가 `FEATURE_GENERIC`으로 떨어뜨린다. 지어내면 서로 다른 접점이 같은
//! ID를 받아 warm start가 엉뚱한 곳에 붙는다.

use super::feature::ClipKind;
use maru_math::Vec3;

/// 면 하나의 정점 상한. `MAX_HULL_POINTS`와 같다 — hull 면 링이 그보다 길 수 없다.
pub(super) const MAX_FACE_VERTS: usize = crate::MAX_HULL_POINTS;

/// 클리핑 결과 상한. 볼록 m각형을 반평면 k개로 자르면 정점은 최대 m + k다
/// (측면마다 최대 +1). 힙을 쓰지 않으므로 상한이 곧 스택 배열 크기다.
pub(super) const GEN_CLIP_CAP: usize = 2 * MAX_FACE_VERTS;

/// 계보를 특정할 수 없다는 표시. 코덱 인덱스 상한(31)을 넘으므로 `pack()`이 거절한다 —
/// 조용히 0번으로 지어내면 서로 다른 접점이 같은 ID를 받는다.
pub(super) const UNKNOWN_INDEX: u8 = u8::MAX;

/// 폴리곤 에지의 출처. `Incident(j)` = 입사면 원본 에지 j, `RefSide(t)` = 참조면 측면 t.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EdgeSrc {
    Incident(u8),
    RefSide(u8),
}

/// 클리핑된 폴리곤의 정점 하나.
#[derive(Clone, Copy, Debug)]
pub(super) struct GenVertex {
    pub position: Vec3,
    pub kind: ClipKind,
    /// 계보 첫째 인덱스. `UNKNOWN_INDEX`면 특정 실패(→ GENERIC 폴백).
    pub a: u8,
    /// 계보 둘째 인덱스. `EdgeCross`에서만 의미가 있고 나머지는 0.
    pub b: u8,
    /// 이 정점에서 **시작하는** 에지의 출처 (다음 단계 클리핑이 소비한다).
    edge: EdgeSrc,
}

/// 클리핑 결과 버퍼. 반환값으로 넘기면 1.5KiB를 복사하므로 out-param으로 채운다.
pub(super) struct GenClip {
    verts: [GenVertex; GEN_CLIP_CAP],
    len: usize,
}

impl GenClip {
    pub(super) fn new() -> Self {
        Self {
            verts: [GenVertex {
                position: Vec3::ZERO,
                kind: ClipKind::IncidentVertex,
                a: 0,
                b: 0,
                edge: EdgeSrc::Incident(0),
            }; GEN_CLIP_CAP],
            len: 0,
        }
    }
    pub(super) fn as_slice(&self) -> &[GenVertex] {
        &self.verts[..self.len]
    }
}

/// 측면 t와 측면 s가 공유하는 참조면 정점. 측면 s는 `rverts[s] -> rverts[(s+1)%k]`이므로
/// 측면 t와 t+1이 정점 t+1을 공유한다. 인접하지 않으면 `None`.
fn shared_ref_vertex(t: usize, s: usize, k: usize) -> Option<usize> {
    if (t + 1) % k == s {
        Some(s)
    } else if (s + 1) % k == t {
        Some(t)
    } else {
        None
    }
}

/// 입사면 폴리곤을 참조면의 측면 평면들로 클리핑한다.
///
/// - `rverts`: 참조면 정점(월드, 순환 순서). `rn`: 참조면 바깥 법선.
/// - `iverts`: 입사면 정점(월드, 순환 순서).
/// - 성공하면 `out`에 계보가 붙은 폴리곤을 채우고 `true`. 폴리곤이 비거나 입력이
///   퇴화(정점 3 미만 / 상한 초과)면 `false`(→ 호출자가 witness 단일점으로 폴백).
///
/// 힙을 쓰지 않는다. 감기 방향을 가정하지 않는다 — 측면 평면의 안쪽 방향을 참조면
/// 중심으로 맞춘다(박스 경로와 같은 규칙).
pub(super) fn clip_convex_face(
    rverts: &[Vec3],
    rn: Vec3,
    iverts: &[Vec3],
    out: &mut GenClip,
) -> bool {
    let k = rverts.len();
    let m = iverts.len();
    if !(3..=MAX_FACE_VERTS).contains(&k) || !(3..=MAX_FACE_VERTS).contains(&m) {
        return false;
    }

    let mut poly = [GenVertex {
        position: Vec3::ZERO,
        kind: ClipKind::IncidentVertex,
        a: 0,
        b: 0,
        edge: EdgeSrc::Incident(0),
    }; GEN_CLIP_CAP];
    for (i, v) in iverts.iter().enumerate() {
        poly[i] = GenVertex {
            position: *v,
            kind: ClipKind::IncidentVertex,
            a: i as u8,
            b: 0,
            // 정점 i에서 시작하는 에지 = 입사면 원본 에지 i (i -> i+1)
            edge: EdgeSrc::Incident(i as u8),
        };
    }
    let mut poly_len = m;

    // 박스 경로와 같이 두 버퍼의 **역할만 바꾼다** — 측면마다 배열을 통째로 복사하지 않는다.
    let mut buf = poly;
    let mut flip = false;

    // 참조면 중심은 측면 전체에 대해 같은 값이라 루프 밖에서 한 번만 만든다.
    let mut center = Vec3::ZERO;
    for v in rverts {
        center += *v;
    }
    center = center * (1.0 / k as f32);

    for side in 0..k {
        let e0 = rverts[side];
        let e1 = rverts[(side + 1) % k];
        let mut pn = rn.cross(e1 - e0);
        if pn.dot(center - e0) < 0.0 {
            pn = -pn;
        }
        let (src, dst) = if flip {
            (&buf, &mut poly)
        } else {
            (&poly, &mut buf)
        };
        let mut out_len = 0usize;
        for i in 0..poly_len {
            let v0 = src[i];
            let v1 = src[(i + 1) % poly_len];
            let d0 = pn.dot(v0.position - e0);
            let d1 = pn.dot(v1.position - e0);
            if d0 >= 0.0 && out_len < GEN_CLIP_CAP {
                let mut keep = v0;
                // **경계 위 정점의 출발 에지는 갈아끼워야 한다** (검수 rev_68687b10 MAJOR).
                // d0 == 0이면 v0는 자르는 평면 위에 있다. 그 다음이 밖(또는 역시 평면 위)이면
                // 엄격 부호 규칙 때문에 교점을 안 만들고, 출력 폴리곤에서 v0부터 재진입점까지의
                // 새 에지는 **이 측면 평면 위**에 놓인다. 그런데 옛 입사 에지 출처를 그대로 두면
                // 그 다음 측면이 만드는 교점이 실제로는 참조면 코너인데도 `EdgeCross`로 찍힌다
                // (실측: 참조 정사각형 [0,1]²과 오각형에서 코너 (1,1)이 ReferenceVertex(2)가
                // 아니라 EdgeCross(1,2)를 받았다). 같은 코너가 미세한 자세 변화 전후로 다른
                // ID를 받으면 warm start가 끊긴다.
                if d0 == 0.0 && d1 <= 0.0 {
                    keep.edge = EdgeSrc::RefSide(side as u8);
                }
                dst[out_len] = keep;
                out_len += 1;
            }
            // **엄격한 부호 반대**일 때만 교점을 만든다 — d0 == 0인 경계 정점은 위에서
            // 이미 보존됐고, t=0 교점까지 만들면 같은 자리가 서로 다른 계보로 중복 진입한다
            // (박스 경로가 같은 이유로 같은 규칙을 쓴다).
            if ((d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0)) && out_len < GEN_CLIP_CAP {
                let t = d0 / (d0 - d1);
                let x = v0.position + (v1.position - v0.position) * t;
                // 계보는 **잘린 에지**(v0에서 출발하는 에지)의 출처가 결정한다.
                let (kind, a, b) = match v0.edge {
                    EdgeSrc::Incident(j) => (ClipKind::EdgeCross, j, side as u8),
                    EdgeSrc::RefSide(t_side) => match shared_ref_vertex(t_side as usize, side, k) {
                        Some(v) => (ClipKind::ReferenceVertex, v as u8, 0),
                        // 비인접 측면끼리 만났다 = 볼록 가정이 부동소수로 깨졌다.
                        // 지어내지 않고 모른다고 표시한다.
                        None => (ClipKind::ReferenceVertex, UNKNOWN_INDEX, 0),
                    },
                };
                // 나가는 교차(d0>0)면 x 다음은 이 측면을 따라 간다 → 출발 에지 = 측면 s.
                // 들어오는 교차(d0<0)면 x 다음은 v1이다 → 출발 에지 = 잘린 에지 그대로.
                let edge = if d0 > 0.0 {
                    EdgeSrc::RefSide(side as u8)
                } else {
                    v0.edge
                };
                dst[out_len] = GenVertex {
                    position: x,
                    kind,
                    a,
                    b,
                    edge,
                };
                out_len += 1;
            }
        }
        flip = !flip;
        poly_len = out_len;
        if poly_len == 0 {
            return false;
        }
    }

    let poly = if flip { &buf } else { &poly };
    out.verts[..poly_len].copy_from_slice(&poly[..poly_len]);
    out.len = poly_len;
    true
}

/// **선분** 입사 특징을 참조면 측면 평면들로 자른다 (매개변수 클리핑).
///
/// 다각형 클리퍼는 정점을 순환으로 돌기 때문에 정점 2개를 넣으면 같은 에지를 두 번
/// 훑는다 — 선분은 닫힌 폴리곤이 아니라 **열린 구간**이라 전용 경로가 맞다.
/// 남은 구간의 두 끝점과 각 끝점의 계보를 준다. 구간이 통째로 밖이면 `None`.
///
/// 계보: 원래 끝점이 살아남으면 `IncidentVertex { a: 끝점 번호 }`, 측면 s가 잘랐으면
/// `EdgeCross { a: 0(선분은 에지 하나), b: s }`.
fn clip_segment(rverts: &[Vec3], rn: Vec3, p0: Vec3, p1: Vec3) -> Option<[GenVertex; 2]> {
    let k = rverts.len();
    if !(3..=MAX_FACE_VERTS).contains(&k) {
        return None;
    }
    let mut center = Vec3::ZERO;
    for v in rverts {
        center += *v;
    }
    center = center * (1.0 / k as f32);

    let dir = p1 - p0;
    let (mut t_lo, mut t_hi) = (0.0f32, 1.0f32);
    // 각 끝을 무엇이 잘랐나. `None`이면 원래 끝점 그대로다.
    let (mut lo_side, mut hi_side): (Option<u8>, Option<u8>) = (None, None);
    // 코너 판정용 길이 척도 — 참조면 크기에 상대적이어야 한다(절대 상수는 장면 스케일에서 깨진다).
    let mut span = 0.0f32;
    for v in rverts {
        span = span.max((*v - rverts[0]).length());
    }
    let corner_eps = crate::tolerance::length_eps(span);

    for side in 0..k {
        let e0 = rverts[side];
        let e1 = rverts[(side + 1) % k];
        let mut pn = rn.cross(e1 - e0);
        if pn.dot(center - e0) < 0.0 {
            pn = -pn;
        }
        let d0 = pn.dot(p0 - e0);
        let denom = pn.dot(dir);
        if denom == 0.0 {
            // 측면과 평행하다 — 밖이면 구간 전체가 밖이고, 안이면 이 측면은 안 자른다.
            if d0 < 0.0 {
                return None;
            }
            continue;
        }
        let t = -d0 / denom;
        if denom > 0.0 {
            // 들어오는 쪽: 이 t보다 앞은 밖이다.
            if t > t_lo {
                t_lo = t;
                lo_side = Some(side as u8);
            }
        } else if t < t_hi {
            t_hi = t;
            hi_side = Some(side as u8);
        }
        if t_lo > t_hi {
            return None;
        }
    }

    // **코너 판정은 `t`의 비트 동일성이 아니라 위치로 한다** (검수 rev_83950af0 MAJOR).
    // 선분이 참조면 정점을 정확히 지나면 인접한 두 측면이 같은 t를 내야 하지만, 실제로는
    // dot/나눗셈 반올림이 달라 1 ulp 어긋난다(실측: 평행사변형 정점에서 0.5 vs 0.49999997).
    // 그러면 `t == t_lo` 비교는 절대 발동하지 않고, 사실상 같은 자리에 `EdgeCross` 두 개가
    // 서로 다른 측면 ID로 생겨 1 ulp 자세 변화에 ID가 뒤집힌다.
    // 잘린 점이 그 측면의 **양 끝 정점 중 하나**에 붙어 있으면 코너로 정규화한다.
    let make = |t: f32, cut: Option<u8>, end: u8| {
        let position = p0 + dir * t;
        let (kind, a, b) = match cut {
            Some(s) => {
                let si = s as usize;
                // **참조 평면 안에서 잰다** — 입사 선분은 참조면과 다른 평면에 있으므로
                // 3D 거리로 재면 법선 방향 간격이 섞여 코너를 절대 못 알아본다.
                let corner = [si, (si + 1) % k].into_iter().find(|&c| {
                    let d = position - rverts[c];
                    let inplane = d - rn * d.dot(rn);
                    inplane.length() <= corner_eps
                });
                match corner {
                    Some(c) => (ClipKind::ReferenceVertex, c as u8, 0),
                    None => (ClipKind::EdgeCross, 0, s), // 선분은 에지가 하나다
                }
            }
            None => (ClipKind::IncidentVertex, end, 0),
        };
        GenVertex {
            position,
            kind,
            a,
            b,
            edge: EdgeSrc::Incident(0),
        }
    };
    Some([make(t_lo, lo_side, 0), make(t_hi, hi_side, 1)])
}

/// 면 패치가 클리핑에 쓸 수 없을 만큼 퇴화했나 — 넓이가 임계 이하면 참.
///
/// 정점 2개(원기둥 옆면 선분)는 애초에 넓이가 0이므로 **여기서 판정하지 않는다**:
/// 선분은 전용 경로가 처리하고, 길이 0은 셰이프 검증(`half_height >= MIN_SHAPE_EXTENT`)이
/// 막는다. 넓이는 Newell 합(= 2 × 다각형 넓이)의 크기로 잰다 — 삼각형 하나만 보면
/// 나머지가 퇴화한 경우를 놓친다.
fn polygon_is_degenerate(v: &[Vec3], area_eps: f32) -> bool {
    if v.len() < 3 {
        return false;
    }
    let mut acc = Vec3::ZERO;
    for i in 0..v.len() {
        acc += (v[i] - v[0]).cross(v[(i + 1) % v.len()] - v[0]);
    }
    acc.length() <= area_eps
}

/// 셰이프의 `dir`(월드) 방향 지지면을 월드 좌표 정점 배열로 채운다.
///
/// 성공하면 `(정점 수, 면 코드, 면 바깥 법선(월드))`. 면을 낼 수 없으면 `None`:
/// 구·캡슐처럼 평면 패치가 없는 셰이프, 퇴화 hull(면 0개), 그리고 **둥근 hull**
/// (`radius > 0`)이 그렇다.
///
/// 둥근 hull을 제외하는 이유: 실제 접촉 패치는 면을 접선 방향으로 radius만큼 **안쪽으로
/// 오프셋한** 다각형이다. 박스는 `box_face_verts`가 축 정렬이라 성분별 축소로 정확히
/// 해내지만, 일반 다각형의 내향 오프셋은 에지 직선을 각각 밀어 다시 교차시켜야 하고
/// 폭이 2*radius보다 좁으면 퇴화한다. 근사(중심 방향 축소)는 비정다각형에서 틀린다.
/// 그래서 지금은 둥근 hull을 기존 witness 단일점 경로에 그대로 둔다(회귀 없음).
/// 알려진 한계이며 README '매니폴드 생성'에 적어 두었다.
fn support_face(
    shape: &crate::Shape,
    pose: &crate::Pose,
    m: &maru_math::Mat3,
    dir: Vec3,
    out: &mut [Vec3; MAX_FACE_VERTS],
) -> Option<(usize, u32, Vec3)> {
    match shape {
        crate::Shape::Box { half, radius } => {
            // 박스는 기존 헬퍼를 그대로 쓴다 — 면 인덱스 0..5, 정점 4개.
            let (f, _) = super::clip::best_face(m, dir);
            let vs = super::clip::box_face_verts(f, *half, *radius);
            for (i, v) in vs.iter().enumerate() {
                out[i] = pose.pos + m.mul_vec(*v);
            }
            Some((4, f as u32, super::clip::box_face_normal(f, m)))
        }
        crate::Shape::Hull { data, radius } => {
            if *radius > 0.0 {
                return None;
            }
            // 면 법선은 로컬이다 — 방향을 로컬로 돌리는 편이 면마다 법선을 돌리는 것보다 싸다.
            let local_dir = m.transpose().mul_vec(dir);
            let (f, _) = data.faces().best_face(local_dir)?;
            let face = &data.faces().faces()[f];
            let ring = &face.verts;
            if !(3..=MAX_FACE_VERTS).contains(&ring.len()) {
                return None;
            }
            let pts = data.points();
            for (i, vi) in ring.iter().enumerate() {
                out[i] = pose.pos + m.mul_vec(pts[*vi as usize]);
            }
            Some((ring.len(), f as u32, m.mul_vec(face.normal)))
        }
        crate::Shape::Cylinder {
            half_height,
            radius,
        } => {
            // 축은 로컬 Y다(`support_core`와 같은 규약). 뚜껑이 지지 특징이 아니면
            // 여기서 거절한다 — 옆면 접촉인데 뚜껑을 입사면으로 넘기면 모서리로 선
            // 다각형이 참조면에 눕는 꼴이 된다. 임계는 바깥 면 지배 판정과 같은 값을
            // 써서 문턱이 한 군데만 있게 한다.
            let axis = m.c[1];
            let d = axis.dot(dir);
            if d.abs() < super::FACE_DOT_MIN {
                // **옆면 지지 특징은 선분이다** — 면이 아니다. 방향 `dir`의 축 수직 성분이
                // 지배적이면 옆면 모선(생성선) 하나가 통째로 지지 집합이므로, 정점 2개짜리
                // "면"으로 넘긴다(호출자가 선분 전용 클리핑으로 보낸다). 둘 다 지배적이지
                // 않은 테두리(rim) 접촉은 계보를 특정할 수 없어 기존 witness 경로로 둔다.
                let perp = dir - Vec3::new(axis.x * d, axis.y * d, axis.z * d);
                let pl = perp.length();
                if pl < super::FACE_DOT_MIN {
                    return None; // 테두리 지배 — witness 단일점
                }
                let u = perp * (1.0 / pl);
                out[0] = pose.pos - axis * *half_height + u * *radius;
                out[1] = pose.pos + axis * *half_height + u * *radius;
                // 면 코드 2 = 옆면. 뚜껑(0·1)과 겹치지 않는다.
                return Some((2, 2, u));
            }
            let s = if d >= 0.0 { 1.0 } else { -1.0 };
            let cap_n = Vec3::new(axis.x * s, axis.y * s, axis.z * s);
            let center = pose.pos + cap_n * *half_height;
            // **링을 접촉 방향에 맞춘다** (검수 rev_83950af0 MAJOR). 원판의 `dir` 방향
            // 지지점은 `center + r·normalize(dir의 접평면 성분)`이다. 링 정점 0을 그 방향에
            // 두면 **최심 테두리점이 정확히 다각형 정점**이 되어, 기울어 닿을 때 생기던
            // 침투 방향 누락(`r·(1−cos(π/N))·sin θ`)이 0이 된다. 고정 N으로는 그 오차가
            // 반지름에 비례해서(≈0.006·r) 큰 원기둥에서 접점을 통째로 잃었다 —
            // 반지름 10 m·sin θ=0.312면 59 mm로 기본 margin 50 mm를 넘는다.
            //
            // 접평면 성분이 0이면(뚜껑이 평면과 평행) 방향을 못 정하는데, 그때는 오차도
            // 0이므로 **물체에 붙은 기저**로 떨어진다 — 세워 둔 원기둥에서 링이 프레임마다
            // 흔들려 ID가 튀는 것을 막는다.
            let u = (dir - cap_n * dir.dot(cap_n))
                .try_normalize()
                .unwrap_or(m.c[0]);
            let v = cap_n.cross(u);
            for (i, (cx, cz)) in CAP_RING.iter().enumerate() {
                out[i] = center + u * (radius * cx) + v * (radius * cz);
            }
            // 면 코드: +Y 뚜껑 0, −Y 뚜껑 1. (옆면은 뒤에 2..를 쓴다)
            let code = if s > 0.0 { 0 } else { 1 };
            Some((
                CAP_RING.len(),
                code,
                Vec3::new(axis.x * s, axis.y * s, axis.z * s),
            ))
        }
        _ => None,
    }
}

/// 원기둥 뚜껑 다각형의 단위 좌표(접평면 기저 기준). 삼각함수를 부르지 않는다
/// (결정론 정책 L1: 상태에 닿는 경로에서 libm 금지 — native와 wasm의 마지막 ulp가 다르다).
///
/// **링은 접촉 방향에 정렬해 놓는다**(`support_face` 참조). 그래서 정점 0이 원판의 지지점과
/// 정확히 일치하고, **침투 방향 오차가 0**이다 — 기울기·반지름과 무관하다. 남는 근사는
/// 정점 사이를 잇는 현이 호보다 안쪽이라는 것뿐이고, 그건 뚜껑이 참조면 **모서리에
/// 걸쳤을 때의 클리핑 범위**에만 영향을 준다(최대 `r·(1 − cos(π/16)) = 0.019·r`).
///
/// N=16인 이유: 접점 상한이 4라 축약이 어차피 4개를 고르므로 안정성은 N에 안 변한다.
/// 남은 것은 위의 모서리 클리핑 정확도와 클리핑 비용(측면 루프 N회)의 교환이다.
const CAP_RING: [(f32, f32); 16] = {
    const C1: f32 = 0.923_879_5; // cos(22.5°)
    const C2: f32 = core::f32::consts::FRAC_1_SQRT_2; // cos(45°)
    const C3: f32 = 0.382_683_43; // sin(22.5°)
    [
        (1.0, 0.0),
        (C1, C3),
        (C2, C2),
        (C3, C1),
        (0.0, 1.0),
        (-C3, C1),
        (-C2, C2),
        (-C1, C3),
        (-1.0, 0.0),
        (-C1, -C3),
        (-C2, -C2),
        (-C3, -C1),
        (0.0, -1.0),
        (C3, -C1),
        (C2, -C2),
        (C1, -C3),
    ]
};

/// 일반 볼록 면 접촉 매니폴드. 실패하면 `None`(호출자가 witness 단일점으로 폴백).
///
/// 박스-박스는 **여기 오지 않는다** — 전용 경로가 비트 동일을 보장해야 하므로 호출자가
/// 먼저 걸러 낸다. 여기서도 한 번 더 막아 둔다(호출 순서가 바뀌어도 계약이 유지되게).
pub(super) fn convex_face_manifold(
    sa: &crate::Shape,
    pa: &crate::Pose,
    sb: &crate::Shape,
    pb: &crate::Pose,
    normal: Vec3,
    margin: f32,
) -> Option<super::Manifold> {
    use maru_math::Mat3;
    if matches!(sa, crate::Shape::Box { .. }) && matches!(sb, crate::Shape::Box { .. }) {
        return None;
    }
    let ma = Mat3::from_quat(pa.rot);
    let mb = Mat3::from_quat(pb.rot);
    let area_eps = crate::tolerance::area_eps(crate::tolerance::characteristic_length(sa, sb));
    let mut va = [Vec3::ZERO; MAX_FACE_VERTS];
    let mut vb = [Vec3::ZERO; MAX_FACE_VERTS];
    // A에서 B를 향하는 면(법선 ≈ −normal), B에서 A를 향하는 면(≈ +normal)
    let (na, fa, nrm_a) = support_face(sa, pa, &ma, -normal, &mut va)?;
    let (nb, fb, nrm_b) = support_face(sb, pb, &mb, normal, &mut vb)?;
    let da = nrm_a.dot(-normal);
    let db = nrm_b.dot(normal);
    if da.max(db) < super::FACE_DOT_MIN {
        return None; // 에지-에지 지배 — witness 단일점 경로
    }
    // **참조면은 다각형이어야 한다** — 선분(원기둥 옆면)은 자를 측면 평면을 못 만든다.
    // 그래서 정렬도보다 이 조건이 먼저다. 둘 다 다각형이면 더 정렬된 쪽(동률 = A — 결정론).
    let ref_is_a = match (na >= 3, nb >= 3) {
        (true, true) => da >= db,
        (true, false) => true,
        (false, true) => false,
        // 선분 대 선분(원기둥 둘이 옆으로 맞닿음) — witness 단일점.
        // **이 갈래는 아래 `clip_segment`의 `k >= 3` 검사와 등가다**(참조면이 선분이면
        // 거기서 거절된다). 뮤테이션으로 관측되지 않는다 — 의도를 여기 남기려고 둔다.
        (false, false) => return None,
    };
    // **참조면 자신이 정렬돼 있어야 한다** (검수 rev_7992e5a3 BLOCKER).
    // 바깥 게이트는 `da.max(db) >= 0.95`만 본다. 한쪽이 선분이면 그 선분의 정렬도만으로
    // 게이트를 통과할 수 있고, 그러면 정렬도가 낮은 다각형이 참조면으로 강제된다.
    // 다각형-다각형에서는 최대를 고르므로 이 검사가 아무것도 바꾸지 않는다.
    //
    // **다만 이 검사가 실제로 발동하는 배치는 못 만들었다** — 검수가 든 시나리오
    // (원기둥 축을 박스 코너 법선에 수직으로 두고 코너에 댐)를 재보면 ref_dot=0.578로
    // 여기 도달은 하지만, 그 앞뒤가 먼저 막는다: ① 입사면 재선택이 `-rn`으로 다시
    // 물으면 뚜껑도 옆면도 지배하지 않아(테두리) `None`이 되고, ② 모서리를 따라 눕힌
    // 변형에서는 재선택은 통과하지만 모선이 참조면 y 범위 **밖**에 떨어져 클리핑이
    // 비어 버린다. 기하가 그렇게 맞물리는 듯하다: 법선이 면에서 45° 벗어나면 그 면
    // 방향의 지지 특징은 면 경계 밖에 놓인다. 그래서 이 검사는 **불변식 선언**이지
    // 관측된 결함의 수리가 아니다(뮤테이션으로 안 잡힌다 — 그렇게 보고했다).
    let ref_dot = if ref_is_a { da } else { db };
    if ref_dot < super::FACE_DOT_MIN {
        return None;
    }
    let (rn, ref_face) = if ref_is_a { (nrm_a, fa) } else { (nrm_b, fb) };

    // **입사면은 참조 법선이 정해진 뒤 다시 고른다** (검수 rev_68687b10 BLOCKER).
    // 위에서 고른 두 면은 GJK/EPA `normal` 기준이다. 참조면이 정해지면 자를 평면의 법선은
    // `rn`이고, 입사면은 그 `-rn`에 가장 반대인 면이어야 한다 — 둘은 다를 수 있다.
    // 예: `normal`이 45°인데 hull 참조면의 반대 방향이 50°이면, 상대 박스는 45°에서
    // +X/+Y가 동률이라 낮은 인덱스 +X를 고르지만 `-rn`(50°)에 가장 반대인 면은 +Y다.
    // 재선택이 없으면 서로 직교하는 엉뚱한 면을 클리핑한다. 기존 박스 클리퍼는 바로
    // 이 재선택을 한다(`clip.rs`의 `best_face(im, -rn)`).
    let neg_rn = Vec3::new(-rn.x, -rn.y, -rn.z);
    let (inc_n, inc_face) = if ref_is_a {
        let (n, f, _) = support_face(sb, pb, &mb, neg_rn, &mut vb)?;
        (n, f)
    } else {
        let (n, f, _) = support_face(sa, pa, &ma, neg_rn, &mut va)?;
        (n, f)
    };
    let (rverts, iverts) = if ref_is_a {
        (&va[..na], &vb[..inc_n])
    } else {
        (&vb[..nb], &va[..inc_n])
    };

    // **퇴화 패치는 클리핑할 수 없다** (검수 rev_68687b10 BLOCKER). rounded box의 코어가
    // 점·선으로 줄어들면(`half == radius`, 즉 실제로는 구) 면 정점이 전부 한 자리에 겹치고,
    // 측면 평면 법선이 전부 0이 되어 **입사 폴리곤이 하나도 안 잘린다**. 실측: 반크기
    // (1,1,1)·radius 1 박스를 넓은 hull 바닥에 올리면 A로 넣었을 때 바닥의 네 꼭짓점
    // (±3,±3)이 그대로 접점이 되고(구가 반경 4.2짜리 발을 갖는다), 인자를 뒤집으면 1점이
    // 된다 — 인자 순서가 물리를 바꾼다. 기존 `clip_box_box`가 같은 이유로 같은 검사를 한다.
    if polygon_is_degenerate(rverts, area_eps) || polygon_is_degenerate(iverts, area_eps) {
        return None;
    }

    let mut clipped = GenClip::new();
    if iverts.len() == 2 {
        let seg = clip_segment(rverts, rn, iverts[0], iverts[1])?;
        clipped.verts[..2].copy_from_slice(&seg);
        clipped.len = 2;
    } else if !clip_convex_face(rverts, rn, iverts, &mut clipped) {
        return None;
    }

    // reference 평면 아래(margin 이내)만 채택. 같은 자리 중복은 접촉 국소 척도로 친다
    // (박스 경로와 같은 이유·같은 척도 — AABB 대각을 쓰면 고종횡비에서 패치가 붕괴한다).
    let dup_eps_sq = crate::tolerance::length_sq_eps(crate::tolerance::min_feature_size(sa, sb));
    let mut pts = [super::ContactPoint::default(); GEN_CLIP_CAP];
    let mut pts_len = 0usize;
    for v in clipped.as_slice() {
        let sep = rn.dot(v.position - rverts[0]);
        if sep > margin {
            continue;
        }
        let proj = v.position - rn * sep;
        let position = (v.position + proj) * 0.5;
        if pts[..pts_len]
            .iter()
            .any(|q| (q.position - position).length_sq() <= dup_eps_sq)
        {
            continue;
        }
        // 계보를 특정 못 했으면(퇴화) 지어내지 않고 generic으로 떨어뜨린다 —
        // 솔버가 ID 대신 위치 근접으로 warm start를 매칭한다.
        let feature = super::feature::pack(super::feature::FaceFeature {
            ref_is_a,
            ref_face,
            inc_face,
            kind: v.kind,
            a: v.a as u32,
            b: v.b as u32,
        })
        .unwrap_or(super::FeatureId(super::FEATURE_GENERIC));
        pts[pts_len] = super::ContactPoint {
            position,
            separation: sep,
            feature,
        };
        pts_len += 1;
    }
    if pts_len == 0 {
        return None;
    }

    let reduced;
    let final_pts: &[super::ContactPoint] = if pts_len > 4 {
        reduced = reduce_to_4_general(&pts[..pts_len], rn);
        &reduced
    } else {
        &pts[..pts_len]
    };
    let m_normal = if ref_is_a { -rn } else { rn };
    Some(super::Manifold::from_points(m_normal, final_pts))
}

/// 접점 다수를 4개로 줄인다 — 박스 경로와 **같은 규칙**(최심점 고정 + 면적 최대 조합).
///
/// 후보가 많으면 그 규칙이 O(n³)이라 정점 64개에서 조합이 39,711개가 된다. 그래서 8개를
/// 넘으면 후보를 먼저 8개로 줄인다. 줄이는 방법이 중요하다:
///
/// **순환 인덱스 균등 간격은 안 된다** — 기하 퍼짐을 보장하지 못한다(검수 rev_68687b10).
/// 정점 8개가 작은 호에 몰리고 하나만 반대편 극점인 9각형이면 그 유일한 극점을 버린다.
///
/// **고정 방향 극점도 안 된다** — "지름 끝점은 8개 고정 방향 중 하나의 극점"이라는 주장은
/// 거짓이고, 방향 사이에 법선 원뿔이 좁은 정점은 빠진다. 게다가 최심점이 첫 칸을 차지하면
/// 방향을 7개만 보게 된다(검수 rev_83950af0 MAJOR — 실측: 정16각형에서 1번만 더 깊게 두면
/// 표본이 `1,4,8,12`가 되어 면적 3.8478, 전수 최대 `0,4,8,12`의 4.0000을 놓쳤다).
///
/// 그래서 **최원점 표본**(farthest-point sampling)을 쓴다: 최심점에서 시작해, 이미 고른
/// 집합까지의 **최소 거리가 최대**인 점을 하나씩 더한다. 두 번째 선택이 곧 최심점에서
/// 가장 먼 점이므로 지름이 구조적으로 보장되고, 고정 방향이 없어 어떤 분포에서도 퍼진다.
/// 비용은 8 × n(≤ 512회)이고 힙을 쓰지 않는다. 동률은 낮은 feature가 이긴다(결정론).
///
/// 최종 4점 선택은 여전히 전수 면적 최대다 — 탐욕은 면적 최대를 보장하지 못해 박스 경로에서
/// 이미 한 번 리뷰로 기각된 규칙이다.
pub(super) fn reduce_to_4_general(
    pts: &[super::ContactPoint],
    n: Vec3,
) -> [super::ContactPoint; 4] {
    const SAMPLE: usize = 8;
    if pts.len() <= SAMPLE {
        return super::clip::reduce_to_4(pts, n);
    }
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
    let mut sel = [deepest; SAMPLE];
    // 각 후보의 "이미 고른 집합까지의 최소 거리". 매번 전수로 다시 재지 않고 갱신한다.
    let mut mind = [f32::INFINITY; GEN_CLIP_CAP];
    for len in 1..SAMPLE {
        let anchor = pts[sel[len - 1]].position;
        let mut best = usize::MAX;
        let mut best_d = -1.0f32;
        for (i, p) in pts.iter().enumerate() {
            let d = (p.position - anchor).length_sq();
            if d < mind[i] {
                mind[i] = d;
            }
            if sel[..len].contains(&i) {
                continue;
            }
            let better = match mind[i].total_cmp(&best_d) {
                core::cmp::Ordering::Greater => true,
                core::cmp::Ordering::Equal => best != usize::MAX && p.feature < pts[best].feature,
                core::cmp::Ordering::Less => false,
            };
            if better {
                best = i;
                best_d = mind[i];
            }
        }
        debug_assert!(best != usize::MAX, "입력이 8을 넘으므로 남은 후보가 있다");
        sel[len] = best;
    }
    sel.sort_unstable(); // 순환 순서 보존 — 면적 계산의 전제다
                         // **힙을 쓰지 않는다** — 스텝 중 할당 0 계약(설계 §6). 길이가 상수라 배열로 족하다.
    let sample: [super::ContactPoint; SAMPLE] = core::array::from_fn(|j| pts[sel[j]]);
    super::clip::reduce_to_4(&sample, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::feature::{pack, FaceFeature};
    use std::collections::HashSet;

    fn square(cx: f32, cy: f32, h: f32, z: f32) -> Vec<Vec3> {
        vec![
            Vec3::new(cx - h, cy - h, z),
            Vec3::new(cx + h, cy - h, z),
            Vec3::new(cx + h, cy + h, z),
            Vec3::new(cx - h, cy + h, z),
        ]
    }

    /// **엄밀 볼록** n각형(n은 짝수). 삼각함수를 쓰지 않는다(결정론 정책 L1: 상태에
    /// 닿는 경로에서 libm 금지 — 테스트 픽스처도 같은 규율로 만든다).
    ///
    /// 포물선 두 개로 렌즈를 만든다: 아래 사슬 `y = x²`, 위 사슬 `y = 2 − x²`, x ∈ [−1,1].
    /// 포물선 위 세 점은 절대 일직선이 아니므로 **좌표가 딱 떨어지지 않는다** — 정다각형
    /// 근사를 정사각형 둘레 균등분할로 만들면 변마다 점이 일직선이 되어 측면 평면이
    /// 겹치고, 클리퍼가 "비인접 측면끼리 만나는" 퇴화에 빠진다(실측으로 겪었다 —
    /// `collinear_reference_edges_mark_lineage_unknown_instead_of_guessing`가 그쪽을 잠근다).
    fn convex_lens(n: usize, r: f32, z: f32) -> Vec<Vec3> {
        assert!(n >= 4 && n.is_multiple_of(2));
        let h = n / 2;
        let x_at = |j: usize| -1.0 + 2.0 * (j as f32) / (h as f32);
        let mut out: Vec<Vec3> = Vec::with_capacity(n);
        for j in 0..=h {
            let x = x_at(j);
            out.push(Vec3::new(x * r, (x * x - 1.0) * r, z));
        }
        for j in (1..h).rev() {
            let x = x_at(j);
            out.push(Vec3::new(x * r, (1.0 - x * x) * r, z));
        }
        assert_eq!(out.len(), n);
        // 픽스처가 진짜 엄밀 볼록인지 스스로 판정한다 — 아니면 시험이 겨누는 것을 못 겨눈다.
        for i in 0..n {
            let a = out[i];
            let b = out[(i + 1) % n];
            let c = out[(i + 2) % n];
            let cross = (b - a).cross(c - b).z;
            assert!(cross > 1e-6, "{i}번 코너가 일직선이거나 오목하다: {cross}");
        }
        out
    }

    fn lineage_ids(c: &GenClip) -> Vec<Option<u32>> {
        c.as_slice()
            .iter()
            .map(|v| {
                pack(FaceFeature {
                    ref_is_a: true,
                    ref_face: 1,
                    inc_face: 2,
                    kind: v.kind,
                    a: v.a as u32,
                    b: v.b as u32,
                })
                .map(|f| f.0)
            })
            .collect()
    }

    /// 입사면이 참조면 안에 완전히 들면 그대로 살아남는다 — 정점 수·계보가 원본이다.
    #[test]
    fn fully_contained_incident_face_survives_unclipped() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        let i = square(0.0, 0.0, 0.4, 0.1);
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        assert_eq!(out.as_slice().len(), 4);
        for (n, v) in out.as_slice().iter().enumerate() {
            assert_eq!(
                v.kind,
                ClipKind::IncidentVertex,
                "{n}번 정점 계보가 바뀌었다"
            );
            assert_eq!(v.a as usize, n);
        }
    }

    /// 어긋난 사각형끼리는 겹친 사각형이 남는다: 원본 정점 1 + 교점 2 + 참조면 정점 1.
    #[test]
    fn offset_squares_clip_to_the_overlap_with_mixed_lineage() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        let i = square(0.6, 0.6, 1.0, 0.1); // 우상단으로 밀린 같은 크기 사각형
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let pts = out.as_slice();
        assert_eq!(pts.len(), 4, "겹침은 사각형이어야 한다: {pts:?}");
        let kinds: HashSet<_> = pts.iter().map(|v| v.kind).collect();
        assert!(
            kinds.contains(&ClipKind::IncidentVertex),
            "입사면 원본 정점이 하나 남아야 한다"
        );
        assert!(
            kinds.contains(&ClipKind::EdgeCross),
            "입사 에지 × 참조 측면 교점이 있어야 한다"
        );
        assert!(
            kinds.contains(&ClipKind::ReferenceVertex),
            "참조면 코너가 겹침 안에 들어야 한다"
        );
        // 참조면 [-1,1]² ∩ 입사면 [-0.4,1.6]² = [-0.4,1]² 사각형이다.
        for v in pts {
            assert!(
                v.position.x >= -0.4 - 1e-5 && v.position.x <= 1.0 + 1e-5,
                "{v:?}"
            );
            assert!(
                v.position.y >= -0.4 - 1e-5 && v.position.y <= 1.0 + 1e-5,
                "{v:?}"
            );
        }
    }

    /// 참조면 코너의 계보는 **정확히 그 코너의 정점 인덱스**여야 한다.
    /// (0.4,0.4)가 겹침의 코너이고 그것은 참조면 정점 2번 (1,1)이 아니라... 를 좌표로 판정.
    #[test]
    fn reference_corner_lineage_names_the_actual_corner() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0); // 정점 0=(-1,-1) 1=(1,-1) 2=(1,1) 3=(-1,1)
        let i = square(0.0, 0.0, 2.0, 0.1); // 참조면을 완전히 덮는다 → 참조면 코너 4개가 남는다
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let pts = out.as_slice();
        assert_eq!(pts.len(), 4, "참조면 사각형이 그대로 남아야 한다: {pts:?}");
        for v in pts {
            assert_eq!(v.kind, ClipKind::ReferenceVertex, "{v:?}");
            assert_ne!(v.a, UNKNOWN_INDEX, "코너 계보를 특정 못 했다: {v:?}");
            // 클리핑 결과는 **입사면 평면 위**에 있으므로 법선 방향 성분은 다르다.
            // 계보 판정은 면 안(xy) 좌표로 한다.
            let named = r[v.a as usize];
            let d = named - v.position;
            assert!(
                (d.x * d.x + d.y * d.y).sqrt() < 1e-5,
                "계보가 가리키는 참조면 정점 {named:?}가 실제 위치 {:?}와 다르다",
                v.position
            );
        }
    }

    /// 계보는 접점마다 달라야 한다 — 같으면 warm start가 엉뚱한 접점에 붙는다.
    #[test]
    fn every_clipped_vertex_gets_a_distinct_feature_id() {
        let mut out = GenClip::new();
        for (ox, oy) in [(0.0, 0.0), (0.6, 0.6), (0.9, 0.1), (0.5, 0.0)] {
            let r = square(0.0, 0.0, 1.0, 0.0);
            let i = square(ox, oy, 1.0, 0.1);
            assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
            let ids = lineage_ids(&out);
            assert!(
                ids.iter().all(|x| x.is_some()),
                "계보 특정 실패: ({ox},{oy}) {:?}",
                out.as_slice()
            );
            let set: HashSet<_> = ids.iter().collect();
            assert_eq!(
                set.len(),
                ids.len(),
                "({ox},{oy})에서 접점 둘이 같은 feature ID를 받았다: {:?}",
                out.as_slice()
            );
        }
    }

    /// 많은 정점 면끼리도 상한을 넘지 않고 계보가 단사다.
    #[test]
    fn many_sided_faces_stay_within_the_cap_and_stay_injective() {
        let mut out = GenClip::new();
        let r = convex_lens(MAX_FACE_VERTS, 1.0, 0.0);
        let i = convex_lens(MAX_FACE_VERTS, 1.0, 0.1);
        // 살짝 어긋나게 밀어 교점이 생기게 한다
        let i: Vec<Vec3> = i
            .iter()
            .map(|v| Vec3::new(v.x + 0.37, v.y + 0.11, v.z))
            .collect();
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let pts = out.as_slice();
        assert!(
            pts.len() <= GEN_CLIP_CAP,
            "상한 {GEN_CLIP_CAP}을 넘었다: {}",
            pts.len()
        );
        let ids = lineage_ids(&out);
        let set: HashSet<_> = ids.iter().collect();
        assert_eq!(set.len(), ids.len(), "계보가 충돌했다");
        assert!(ids.iter().all(|x| x.is_some()), "계보 특정 실패");
    }

    /// **일직선 점이 섞인 참조면**은 측면 평면이 겹쳐 비인접 측면끼리 만난다. 그럴 때
    /// 계보를 지어내지 않고 모른다고 표시해야 한다 — 지어내면 서로 다른 접점이 같은 ID를
    /// 받아 warm start가 엉뚱한 곳에 붙는다. 실제 hull 면 링은 monotone chain이 일직선
    /// 점을 걸러 주지만, 부동소수 퇴화로 여기 도달할 수 있다.
    #[test]
    fn collinear_reference_edges_mark_lineage_unknown_instead_of_guessing() {
        // 정사각형 각 변을 4등분한 16각형 — 변마다 점 5개가 일직선이다.
        let mut r: Vec<Vec3> = Vec::new();
        for side in 0..4 {
            for j in 0..4 {
                let t = j as f32 / 4.0;
                let (u, v) = match side {
                    0 => (-1.0 + 2.0 * t, -1.0),
                    1 => (1.0, -1.0 + 2.0 * t),
                    2 => (1.0 - 2.0 * t, 1.0),
                    _ => (-1.0, 1.0 - 2.0 * t),
                };
                r.push(Vec3::new(u, v, 0.0));
            }
        }
        let mut out = GenClip::new();
        let i = square(0.0, 0.0, 2.0, 0.1); // 참조면을 완전히 덮는다
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let unknown = out
            .as_slice()
            .iter()
            .filter(|v| v.a == UNKNOWN_INDEX)
            .count();
        assert!(
            unknown > 0,
            "퇴화 참조면인데 계보를 전부 특정했다 — 지어낸 것 아닌가: {:?}",
            out.as_slice()
        );
        // 모른다고 표시한 정점은 코덱이 거절해 호출자가 GENERIC으로 떨어뜨린다.
        for v in out.as_slice().iter().filter(|v| v.a == UNKNOWN_INDEX) {
            assert_eq!(
                pack(FaceFeature {
                    ref_is_a: true,
                    ref_face: 1,
                    inc_face: 2,
                    kind: v.kind,
                    a: v.a as u32,
                    b: v.b as u32,
                }),
                None,
                "미특정 계보가 코덱을 통과했다: {v:?}"
            );
        }
    }

    /// **정점이 측면 평면 위에 정확히 놓인** 배치에서 같은 자리에 점이 둘 생기면 안 된다.
    /// 솔버는 접점마다 제약을 걸므로 겹친 점은 그 지점에 제약을 두 번 거는 것과 같다.
    /// 계보는 서로 달라서(원본 정점 vs t=0 교점) 단사성 시험으로는 안 잡힌다 —
    /// **위치**로 판정해야 한다(뮤테이션 실측: 부호 규칙을 `>=`로 풀어도 단사성은 통과).
    #[test]
    fn a_vertex_exactly_on_a_side_plane_does_not_spawn_a_duplicate_point() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        // x = 1(참조면 오른쪽 측면) **위에** 정점 둘이 정확히 놓인 삼각형.
        let i = vec![
            Vec3::new(1.0, -0.5, 0.1),
            Vec3::new(2.0, 0.0, 0.1),
            Vec3::new(1.0, 0.5, 0.1),
        ];
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let pts = out.as_slice();
        for (n, v) in pts.iter().enumerate() {
            for w in &pts[n + 1..] {
                assert!(
                    (w.position - v.position).length() > 1e-6,
                    "같은 자리에 접점이 둘이다: {v:?} / {w:?}"
                );
            }
        }
    }

    /// **입사 에지 하나가 참조면 측면 둘을 가로지르는** 배치. 볼록 영역을 지나는 직선은
    /// 반드시 한 측면으로 들어와 다른 측면으로 나가므로 교점 둘이 같은 에지에서 나온다 —
    /// 계보에 측면 인덱스가 없으면 둘이 같은 ID를 받는다(뮤테이션 실측: `b`를 0으로
    /// 고정해도 다른 시험은 전부 통과했다).
    #[test]
    fn one_incident_edge_crossing_two_reference_sides_gets_two_ids() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        // 에지 (2,-2) -> (-2,0.5)는 아래 측면(y=-1)을 (0.4,-1)에서 통과해
        // 왼쪽 측면(x=-1)을 (-1,-0.125)에서 빠져나간다. 좌표가 딱 떨어지지 않는다.
        let i = vec![
            Vec3::new(2.0, -2.0, 0.1),
            Vec3::new(-2.0, 0.5, 0.1),
            Vec3::new(3.0, 3.0, 0.1),
        ];
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let crossings: Vec<_> = out
            .as_slice()
            .iter()
            .filter(|v| v.kind == ClipKind::EdgeCross && v.a == 0)
            .collect();
        assert!(
            crossings.len() >= 2,
            "입사 에지 0이 측면 둘을 가로질러야 한다: {:?}",
            out.as_slice()
        );
        let sides: HashSet<u8> = crossings.iter().map(|v| v.b).collect();
        assert_eq!(
            sides.len(),
            crossings.len(),
            "같은 에지의 교점들이 같은 측면 인덱스를 받았다: {crossings:?}"
        );
        let ids = lineage_ids(&out);
        let set: HashSet<_> = ids.iter().collect();
        assert_eq!(
            set.len(),
            ids.len(),
            "계보가 충돌했다: {:?}",
            out.as_slice()
        );
    }

    /// 원기둥은 **자세에 따라 다른 지지 특징**을 써야 한다: 세우면 뚜껑(면 코드 0·1),
    /// 누우면 옆면 모선(면 코드 2). 뚜껑 지지 판정을 빼면 모서리로 선 뚜껑 다각형이
    /// 참조면에 눕는 꼴이 되어 접점이 엉뚱한 곳에 생긴다.
    ///
    /// 300스텝 동역학으로는 두 경우가 y=0.2844 대 0.2706으로만 갈려 판정이 아슬아슬했다
    /// — 계보의 **면 코드**로 직접 겨눈다.
    #[test]
    fn cylinder_picks_the_cap_or_the_side_by_its_pose() {
        use crate::manifold::feature::unpack;
        use crate::{Pose, Shape};
        use maru_math::Quat;
        let floor = Shape::Box {
            half: Vec3::new(5.0, 0.5, 5.0),
            radius: 0.0,
        };
        let pf = Pose {
            pos: Vec3::new(0.0, -0.5, 0.0),
            rot: Quat::IDENTITY,
        };
        let cyl = Shape::Cylinder {
            half_height: 0.6,
            radius: 0.3,
        };

        // 세우면: 뚜껑이 참조면(정렬 동률이면 A가 이긴다)이고 면 코드는 0 또는 1이다.
        let upright = Pose {
            pos: Vec3::new(0.0, 0.599, 0.0),
            rot: Quat::IDENTITY,
        };
        let m = crate::manifold::dispatch::contact(&cyl, &upright, &floor, &pf, 0.02)
            .expect("세운 원기둥이 바닥에 닿아야 한다");
        assert!(
            m.points().len() >= 3,
            "세운 원기둥이 다접점을 못 냈다: {}개",
            m.points().len()
        );
        for p in m.points() {
            let f = unpack(p.feature).expect("뚜껑 접점에 계보가 없다");
            assert!(f.ref_is_a, "세운 원기둥이 참조면이 아니다: {f:?}");
            assert!(
                f.ref_face <= 1,
                "뚜껑이 아닌 면 코드를 썼다: {} ({f:?})",
                f.ref_face
            );
        }

        // 눕히면: 옆면 선분이 입사 특징이고 면 코드는 2다. 참조면은 바닥이어야 한다
        // (선분은 자를 측면 평면을 못 만든다).
        let lying = Pose {
            pos: Vec3::new(0.0, 0.299, 0.0),
            rot: Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), core::f32::consts::FRAC_PI_2),
        };
        let m = crate::manifold::dispatch::contact(&cyl, &lying, &floor, &pf, 0.02)
            .expect("누운 원기둥이 바닥에 닿아야 한다");
        assert_eq!(
            m.points().len(),
            2,
            "옆면 선분은 접점 2개다: {:?}",
            m.points()
        );
        for p in m.points() {
            let f = unpack(p.feature).expect("옆면 접점에 계보가 없다");
            assert!(!f.ref_is_a, "선분이 참조면이 됐다: {f:?}");
            assert_eq!(f.inc_face, 2, "옆면 코드가 아니다: {f:?}");
        }
        // 두 접점은 서로 다른 ID여야 한다 — 같으면 warm start가 한쪽에 몰린다.
        assert_ne!(m.points()[0].feature, m.points()[1].feature);
    }

    /// 선분 클리퍼 자체를 직접 겨눈다. 참조면을 **완전히 빗나가는** 구간은 실제 셰이프
    /// 배치로는 만들기 미묘하다 — 접촉 법선 방향의 지지 선분은 늘 최근접 특징 근처에
    /// 내려앉기 때문이다. 그래도 빈 구간 판정이 없으면 참조면 **밖**의 두 점이 접점으로
    /// 나가므로(뒤집힌 구간의 양 끝), 함수 층에서 잠근다.
    #[test]
    fn segment_clipping_covers_inside_clipped_and_missed() {
        let r = square(0.0, 0.0, 1.0, 0.0);
        let n = Vec3::new(0.0, 0.0, 1.0);

        // ① 안에 완전히 들어간 구간 — 양 끝이 원본 끝점 그대로다.
        let seg = clip_segment(&r, n, Vec3::new(-0.5, 0.0, 0.1), Vec3::new(0.5, 0.0, 0.1))
            .expect("안쪽 구간이 거절됐다");
        for (i, v) in seg.iter().enumerate() {
            assert_eq!(v.kind, ClipKind::IncidentVertex, "{v:?}");
            assert_eq!(v.a as usize, i, "끝점 번호가 어긋났다: {v:?}");
        }

        // ② 양쪽으로 삐져나온 구간 — 양 끝이 서로 다른 측면에 잘린다.
        let seg = clip_segment(&r, n, Vec3::new(-3.0, 0.25, 0.1), Vec3::new(3.0, 0.25, 0.1))
            .expect("가로지르는 구간이 거절됐다");
        for v in &seg {
            assert_eq!(v.kind, ClipKind::EdgeCross, "{v:?}");
            assert!(v.position.x.abs() <= 1.0 + 1e-5, "{v:?}");
        }
        assert_ne!(seg[0].b, seg[1].b, "양 끝이 같은 측면에 잘렸다");
        assert!(
            (seg[0].position.x + 1.0).abs() < 1e-5 && (seg[1].position.x - 1.0).abs() < 1e-5,
            "잘린 자리가 참조면 경계가 아니다: {seg:?}"
        );

        // ③ 코너 **바깥을 스치는 대각선** — 어떤 측면과도 평행하지 않아 평행 갈래에
        //    걸리지 않고, 들어오는 t(0.6)가 나가는 t(0.4)를 넘어서야 비로소 거절된다.
        //    축 평행 구간으로 시험하면 평행 갈래가 먼저 잡아 빈 구간 판정이 관측되지
        //    않는다(뮤테이션 실측: 판정을 지워도 통과했다).
        assert!(
            clip_segment(&r, n, Vec3::new(2.5, 0.0, 0.1), Vec3::new(0.0, 2.5, 0.1)).is_none(),
            "코너 밖을 스친 구간이 접점을 냈다"
        );
        // ③' 음성 대조 — 같은 대각선을 안쪽으로 밀면 잘려서 남는다.
        assert!(
            clip_segment(&r, n, Vec3::new(1.5, 0.0, 0.1), Vec3::new(0.0, 1.5, 0.1)).is_some(),
            "안쪽을 지나는 대각선이 거절됐다"
        );
        // ④ 측면과 평행하면서 밖에 있는 구간도 마찬가지다(denom == 0 갈래).
        assert!(
            clip_segment(&r, n, Vec3::new(-5.0, 1.5, 0.1), Vec3::new(5.0, 1.5, 0.1)).is_none(),
            "평행-바깥 구간이 접점을 냈다"
        );
        // ⑤ 퇴화 참조면(정점 2개)은 자를 측면 평면을 못 만든다.
        let two = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        assert!(
            clip_segment(&two, n, Vec3::new(0.0, 0.0, 0.1), Vec3::new(1.0, 0.0, 0.1)).is_none()
        );
    }

    /// **선분이 실제로 잘리는** 배치. 위 시험은 바닥이 넓어 선분이 한 번도 안 잘렸고,
    /// 그래서 매개변수 클리핑 코드에 도달하지 못했다(뮤테이션 실측: `t_hi` 비교를
    /// 뒤집어도, 빈 구간 판정을 지워도 전부 통과했다). 좁은 발판 위에 긴 원기둥을
    /// 가로놓아 양 끝이 서로 다른 측면에 잘리게 한다.
    #[test]
    fn a_side_segment_hanging_off_both_edges_is_clipped_to_the_reference_face() {
        use crate::manifold::feature::unpack;
        use crate::{Pose, Shape};
        use maru_math::Quat;
        // 발판은 x 방향으로 반폭 0.2뿐이다 — 원기둥(반높이 0.6)이 양쪽으로 삐져나온다.
        let ledge = Shape::Box {
            half: Vec3::new(0.2, 0.5, 5.0),
            radius: 0.0,
        };
        let pf = Pose {
            pos: Vec3::new(0.0, -0.5, 0.0),
            rot: Quat::IDENTITY,
        };
        let cyl = Shape::Cylinder {
            half_height: 0.6,
            radius: 0.3,
        };
        // 축(로컬 Y)을 Z축 둘레로 90° 돌리면 축이 월드 X를 향한다.
        let lying = Pose {
            pos: Vec3::new(0.0, 0.299, 0.0),
            rot: Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), core::f32::consts::FRAC_PI_2),
        };
        let m = crate::manifold::dispatch::contact(&cyl, &lying, &ledge, &pf, 0.02)
            .expect("원기둥이 발판에 닿아야 한다");
        assert_eq!(m.points().len(), 2, "{:?}", m.points());
        let mut sides = Vec::new();
        for p in m.points() {
            let f = unpack(p.feature).expect("계보가 없다");
            assert_eq!(
                f.kind,
                ClipKind::EdgeCross,
                "삐져나온 끝이 안 잘렸다: {f:?}"
            );
            sides.push(f.b);
            assert!(
                p.position.x.abs() <= 0.2 + 1e-4,
                "접점이 발판 밖에 있다: {:?}",
                p.position
            );
        }
        assert_ne!(sides[0], sides[1], "양 끝이 같은 측면에 잘렸다: {sides:?}");
        // 잘린 구간은 발판 폭 전체다 — 한쪽으로 쏠리면 클리핑이 틀린 것이다.
        let span = (m.points()[0].position.x - m.points()[1].position.x).abs();
        assert!(span > 0.35, "잘린 구간이 너무 짧다: {span}");
    }

    /// 테두리(rim)로 서면 뚜껑도 옆면도 지지 특징이 아니다 — 기존 witness 단일점으로
    /// 떨어져야 한다. 계보를 지어내면 회전할 때마다 엉뚱한 ID가 생긴다.
    #[test]
    fn a_cylinder_on_its_rim_falls_back_to_the_witness_point() {
        use crate::{Pose, Shape};
        use maru_math::Quat;
        let floor = Shape::Box {
            half: Vec3::new(5.0, 0.5, 5.0),
            radius: 0.0,
        };
        let pf = Pose {
            pos: Vec3::new(0.0, -0.5, 0.0),
            rot: Quat::IDENTITY,
        };
        let cyl = Shape::Cylinder {
            half_height: 0.6,
            radius: 0.3,
        };
        // 45° 기울이면 |축·법선| = 0.707이라 뚜껑(0.95)도 옆면(0.95)도 지배하지 않는다.
        let tilted = Pose {
            pos: Vec3::new(0.0, 0.63, 0.0),
            rot: Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), core::f32::consts::FRAC_PI_4),
        };
        let m = crate::manifold::dispatch::contact(&cyl, &tilted, &floor, &pf, 0.05)
            .expect("기운 원기둥이 바닥에 닿아야 한다");
        assert_eq!(
            m.points().len(),
            1,
            "테두리 접촉이 다접점을 냈다: {:?}",
            m.points()
        );
        assert_eq!(
            m.points()[0].feature.0,
            crate::FEATURE_GENERIC,
            "테두리 접점에 지어낸 계보가 붙었다"
        );
    }

    /// 두 뚜껑은 **서로 다른 면 코드**를 받아야 한다. 같은 코드를 주면 원기둥이 뒤집혀
    /// 반대쪽 뚜껑이 닿을 때 서로 다른 물리 특징이 같은 ID를 받아, 솔버가 직전 임펄스를
    /// 엉뚱한 접점에 잇는다(뮤테이션 실측: 코드를 0으로 고정해도 다른 시험은 전부 통과).
    #[test]
    fn the_two_caps_get_different_face_codes() {
        use crate::{Pose, Shape};
        use maru_math::Quat;
        let floor = Shape::Box {
            half: Vec3::new(5.0, 0.5, 5.0),
            radius: 0.0,
        };
        let pf = Pose {
            pos: Vec3::new(0.0, -0.5, 0.0),
            rot: Quat::IDENTITY,
        };
        let cyl = Shape::Cylinder {
            half_height: 0.6,
            radius: 0.3,
        };
        let ids = |rot: Quat| -> Vec<u32> {
            let p = Pose {
                pos: Vec3::new(0.0, 0.599, 0.0),
                rot,
            };
            let m = crate::manifold::dispatch::contact(&cyl, &p, &floor, &pf, 0.02)
                .expect("원기둥이 바닥에 닿아야 한다");
            let mut v: Vec<u32> = m.points().iter().map(|p| p.feature.0).collect();
            v.sort_unstable();
            v
        };
        // 똑바로 서면 로컬 −Y 뚜껑이, X축 둘레 180°로 뒤집으면 로컬 +Y 뚜껑이 바닥에 닿는다.
        let upright = ids(Quat::IDENTITY);
        let flipped = ids(Quat::from_axis_angle(
            Vec3::new(1.0, 0.0, 0.0),
            core::f32::consts::PI,
        ));
        assert_eq!(upright.len(), flipped.len());
        assert!(
            upright.iter().all(|a| !flipped.contains(a)),
            "서로 다른 뚜껑이 같은 feature ID를 받았다: {upright:?} vs {flipped:?}"
        );
    }

    /// 32점 구형 hull — 면이 촘촘해 참조면 법선과 GJK 법선이 벌어질 수 있다.
    /// (황금나선 배치. 좌표는 미리 계산해 상수로 박는다 — 시험도 삼각함수를 안 부른다.)
    fn sphere_hull() -> crate::Shape {
        crate::Shape::Hull {
            data: crate::HullData::new(vec![
                Vec3::new(0.248039, 0.968750, 0.000000),
                Vec3::new(-0.311717, 0.906250, 0.285558),
                Vec3::new(0.046925, 0.843750, -0.534681),
                Vec3::new(0.379799, 0.781250, 0.495380),
                Vec3::new(-0.684640, 0.718750, -0.121103),
                Vec3::new(0.636650, 0.656250, -0.404985),
                Vec3::new(-0.208890, 0.593750, 0.777062),
                Vec3::new(-0.390487, 0.531250, -0.751860),
                Vec3::new(0.829732, 0.468750, 0.303017),
                Vec3::new(-0.844632, 0.406250, 0.348652),
                Vec3::new(0.398017, 0.343750, -0.850540),
                Vec3::new(0.287203, 0.281250, 0.915649),
                Vec3::new(-0.844257, 0.218750, -0.489264),
                Vec3::new(0.964680, 0.156250, -0.212082),
                Vec3::new(-0.572596, 0.093750, 0.814459),
                Vec3::new(-0.128448, 0.031250, -0.991224),
                Vec3::new(0.764276, -0.031250, 0.644132),
                Vec3::new(-0.994746, -0.093750, 0.041136),
                Vec3::new(0.700123, -0.156250, -0.696716),
                Vec3::new(-0.045073, -0.218750, 0.974739),
                Vec3::new(-0.614847, -0.281250, -0.736792),
                Vec3::new(0.930675, -0.343750, 0.125221),
                Vec3::new(-0.750069, -0.406250, 0.521879),
                Vec3::new(0.193875, -0.468750, -0.861792),
                Vec3::new(0.421219, -0.531250, 0.735084),
                Vec3::new(-0.766584, -0.593750, -0.244561),
                Vec3::new(0.684968, -0.656250, -0.316473),
                Vec3::new(-0.268416, -0.718750, 0.641367),
                Vec3::new(-0.211268, -0.781250, -0.587379),
                Vec3::new(0.475113, -0.843750, 0.249706),
                Vec3::new(-0.408779, -0.906250, 0.107753),
                Vec3::new(0.134149, -0.968750, -0.208632),
            ]),
            radius: 0.0,
        }
    }

    /// **BLOCKER (검수 rev_68687b10)**: 코어가 점으로 퇴화한 rounded box는 실제로는 구다.
    /// 면 정점이 전부 한 자리에 겹쳐 측면 평면 법선이 0이 되고, 그러면 입사 폴리곤이
    /// **하나도 안 잘린 채** 통째로 접점이 된다.
    ///
    /// 실측(고치기 전): 반크기 (1,1,1)·radius 1 박스를 넓은 hull 바닥에 올렸을 때
    /// A로 넣으면 바닥의 네 꼭짓점 (±3,±3)이 접점 4개가 되고(구가 반경 4.2짜리 발을
    /// 갖는다), 인자를 뒤집으면 중앙 1점이 됐다 — **인자 순서가 물리를 바꿨다**.
    #[test]
    fn a_degenerate_rounded_box_patch_falls_back_to_the_witness_point() {
        use crate::{Pose, Shape};
        use maru_math::Quat;
        let sphere_box = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 1.0,
        };
        assert!(
            sphere_box.validate().is_ok(),
            "이 셰이프는 합법이다 — 거절되지 않는다"
        );
        let corners: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 == 0 { -3.0 } else { 3.0 },
                    if i & 2 == 0 { -0.5 } else { 0.5 },
                    if i & 4 == 0 { -3.0 } else { 3.0 },
                )
            })
            .collect();
        let floor = Shape::Hull {
            data: crate::HullData::new(corners),
            radius: 0.0,
        };
        let pa = Pose {
            pos: Vec3::new(0.0, 1.49, 0.0),
            rot: Quat::IDENTITY,
        };
        let pf = Pose {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        };
        let ab = crate::manifold::dispatch::contact(&sphere_box, &pa, &floor, &pf, 0.02)
            .expect("닿아야 한다");
        let ba = crate::manifold::dispatch::contact(&floor, &pf, &sphere_box, &pa, 0.02)
            .expect("닿아야 한다");
        assert_eq!(
            ab.points().len(),
            1,
            "구형 박스가 다접점 지지를 받았다: {:?}",
            ab.points()
        );
        assert_eq!(
            ab.points().len(),
            ba.points().len(),
            "인자 순서가 접점 수를 바꿨다"
        );
        // 접점은 구의 아래쪽, 즉 축 근처여야 한다 — 바닥 꼭짓점이 아니다.
        for m in [&ab, &ba] {
            let p = m.points()[0].position;
            assert!(
                p.x.abs() < 0.1 && p.z.abs() < 0.1,
                "접점이 구 아래가 아니라 바닥 구석에 있다: {p:?}"
            );
        }
    }

    /// **BLOCKER (검수 rev_68687b10)**: 참조면이 정해진 뒤 입사면을 `-rn` 기준으로
    /// **다시 골라야** 한다. 처음 고른 면은 GJK/EPA `normal` 기준인데, 면 지배 임계가
    /// 0.95라 참조면 법선은 그 법선과 최대 18°까지 벌어질 수 있고, 면이 촘촘한 hull에서는
    /// 그 차이가 argmax를 뒤집는다.
    ///
    /// **밖에서는 이 차이를 볼 수 없다** — 매니폴드가 내보내는 법선은 GJK 법선이 아니라
    /// 참조면 법선이라, `contact()` 결과만 보면 동어반복이 된다(그렇게 재다가 32건을
    /// "차이"로 오독했는데, 전부 witness 폴백이라 GJK 법선이 노출된 표본이었다).
    /// 그래서 `convex_face_manifold`를 법선 인자와 함께 직접 부른다.
    ///
    /// 실측: 아래 배치에서 `rn·normal = 0.951`(18°)이고, 최초 선택은 hull 면 29번,
    /// `-rn` 기준 재선택은 28번이다. 같은 성질의 배치가 스윕 3,721개 중 1,211개였다.
    #[test]
    fn the_incident_face_is_reselected_against_the_reference_normal() {
        use crate::manifold::feature::unpack;
        use crate::{Pose, Shape};
        use maru_math::{Mat3, Quat};
        let hull = sphere_hull();
        let bx = Shape::Box {
            half: Vec3::new(0.6, 0.6, 0.6),
            radius: 0.0,
        };
        let pa = Pose {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        };
        let pb = Pose {
            pos: Vec3::new(1.55, 0.0, 0.0),
            rot: Quat::IDENTITY,
        };
        let ma = Mat3::from_quat(pa.rot);
        let mb = Mat3::from_quat(pb.rot);
        let n = Vec3::new(1.0, -0.22, -0.24)
            .try_normalize()
            .expect("정규화 가능");
        let m = super::convex_face_manifold(&hull, &pa, &bx, &pb, n, 0.05)
            .expect("이 배치는 면 매니폴드를 낸다");
        let f = unpack(m.points()[0].feature).expect("면 계보가 있어야 한다");
        assert!(!f.ref_is_a, "이 배치에서는 박스가 참조면이다: {f:?}");

        let mut sc = [Vec3::ZERO; MAX_FACE_VERTS];
        let (_, _, rn) = super::support_face(&bx, &pb, &mb, n, &mut sc).expect("박스 지지면");
        assert!(
            rn.dot(n) < 0.96,
            "이 배치는 참조 법선이 GJK 법선과 벌어져 있어야 한다: {}",
            rn.dot(n)
        );
        let nrn = Vec3::new(-rn.x, -rn.y, -rn.z);
        let (_, want, _) = super::support_face(&hull, &pa, &ma, nrn, &mut sc).expect("hull 지지면");
        let (_, pre, _) =
            super::support_face(&hull, &pa, &ma, Vec3::new(-n.x, -n.y, -n.z), &mut sc)
                .expect("hull 지지면");
        assert_ne!(want, pre, "이 배치는 재선택이 답을 바꾸는 자리여야 한다");
        assert_eq!(
            f.inc_face, want,
            "입사면을 -rn으로 다시 고르지 않았다 (재선택 {want} vs 최초 {pre})"
        );
    }

    /// **MAJOR (검수 rev_68687b10)**: 경계 정점이 측면 평면 **위에** 놓이고 그 다음이
    /// 밖이면, 출력 폴리곤에서 그 정점부터 재진입점까지의 새 에지는 **그 측면 평면 위**에
    /// 놓인다. 출발 에지를 옛 입사 에지로 두면 다음 측면이 만드는 교점이 실제로는 참조면
    /// 코너인데도 `EdgeCross`로 찍힌다.
    ///
    /// 실측(고치기 전): 참조 정사각형 [0,1]²에 오각형을 물리면 코너 (1,1)이
    /// `ReferenceVertex a=2`가 아니라 `EdgeCross a=1 b=2`를 받았다. 같은 코너가 미세한
    /// 자세 변화 전후로 다른 ID를 받으면 warm start가 끊긴다.
    #[test]
    fn a_boundary_vertex_hands_the_new_edge_to_the_reference_side() {
        let r = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        // (1,0)이 참조면 오른쪽 측면 **위에 정확히** 놓이고, 그 다음 (2,0.5)는 밖이다.
        let i = vec![
            Vec3::new(0.0, 0.0, 0.1),
            Vec3::new(1.0, 0.0, 0.1),
            Vec3::new(2.0, 0.5, 0.1),
            Vec3::new(2.0, 2.0, 0.1),
            Vec3::new(0.0, 2.0, 0.1),
        ];
        let mut out = GenClip::new();
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let corner = out
            .as_slice()
            .iter()
            .find(|v| (v.position.x - 1.0).abs() < 1e-5 && (v.position.y - 1.0).abs() < 1e-5)
            .unwrap_or_else(|| panic!("코너 (1,1)이 없다: {:?}", out.as_slice()));
        assert_eq!(
            corner.kind,
            ClipKind::ReferenceVertex,
            "참조면 코너가 교점으로 찍혔다: {corner:?}"
        );
        assert_eq!(corner.a, 2, "참조면 정점 2번(1,1)이어야 한다: {corner:?}");
    }

    /// **MAJOR (검수 rev_68687b10)**: 후보 축약이 순환 인덱스 균등 간격이면 기하 퍼짐을
    /// 보장하지 못한다. 정점 8개가 아주 작은 호에 몰리고 하나만 반대편 극점인 9각형에서
    /// 균등 간격 표본은 그 유일한 극점을 정확히 버린다 — 접촉 패치가 지름 전체인데
    /// 제약이 한쪽에 몰려 회전 저항이 사라진다.
    #[test]
    fn reduction_keeps_the_lone_far_extreme_when_the_rest_cluster() {
        use super::super::{ContactPoint, FeatureId};
        let n = Vec3::new(0.0, 0.0, 1.0);
        let mut pts: Vec<ContactPoint> = Vec::new();
        // 0..7: x≈1 근처 작은 호에 몰린 점들
        for i in 0..8 {
            let t = (i as f32 - 3.5) * 0.01;
            pts.push(ContactPoint {
                position: Vec3::new(1.0, t, 0.0),
                separation: -0.01,
                feature: FeatureId(i as u32),
            });
        }
        // 8: 반대편 극점 — 접촉 패치의 지름을 만드는 유일한 점
        pts.push(ContactPoint {
            position: Vec3::new(-1.0, 0.0, 0.0),
            separation: -0.01,
            feature: FeatureId(8),
        });
        let out = reduce_to_4_general(&pts, n);
        assert!(
            out.iter().any(|p| p.position.x < -0.5),
            "반대편 극점이 버려졌다 — 제약이 한쪽에 몰린다: {:?}",
            out.iter().map(|p| p.position.x).collect::<Vec<_>>()
        );
    }

    /// **BLOCKER (검수 rev_7992e5a3)**: 참조면은 자신이 정렬돼 있어야 한다.
    /// 바깥 게이트는 `da.max(db) >= 0.95`만 보므로, 원기둥 옆면의 정렬도만으로 통과하면
    /// 정렬도가 낮은 다각형이 참조면으로 강제된다.
    ///
    /// 배치: 박스 코너 법선 `(1,1,1)/√3`에 **수직**인 축으로 원기둥을 코너에 댄다.
    /// 실제 특징은 모선 대 박스 **정점**의 단일점인데, 고치기 전에는 박스 +X 면(dot
    /// 0.577)이 참조면이 되어 모선을 그 사각형에 자르고 +X 법선의 가짜 2접점을 냈다 —
    /// Y·Z 침투를 못 푼다.
    #[test]
    fn a_generatrix_against_a_box_corner_does_not_fake_a_face_contact() {
        use crate::manifold::feature::unpack;
        use crate::{Pose, Shape};
        use maru_math::Quat;
        let bx = Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        };
        let pb = Pose {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        };
        let cyl = Shape::Cylinder {
            half_height: 0.6,
            radius: 0.3,
        };
        let inv3 = 1.0f32 / 3.0f32.sqrt();
        let cn = Vec3::new(inv3, inv3, inv3);
        // rot_z(θ)는 (0,1,0)을 (−sinθ, cosθ, 0)으로 보낸다 — θ = −3π/4이면 축이
        // (0.7071, −0.7071, 0)이 되어 코너 법선과 수직이다.
        let rot = Quat::from_axis_angle(
            Vec3::new(0.0, 0.0, 1.0),
            -3.0 * core::f32::consts::FRAC_PI_4,
        );
        assert!(
            rot.rotate(Vec3::new(0.0, 1.0, 0.0)).dot(cn).abs() < 1e-5,
            "픽스처 전제: 축이 코너 법선과 수직이어야 한다"
        );
        for t in [0.285f32, 0.295, 0.305, 0.315] {
            let pa = Pose {
                pos: Vec3::new(0.5, 0.5, 0.5) + cn * t,
                rot,
            };
            let m = crate::manifold::dispatch::contact(&cyl, &pa, &bx, &pb, 0.02)
                .unwrap_or_else(|| panic!("t={t}에서 접촉이 있어야 한다"));
            assert_eq!(
                m.points().len(),
                1,
                "코너 접촉이 가짜 다접점을 냈다 (t={t}): {:?}",
                m.points()
            );
            assert_eq!(
                unpack(m.points()[0].feature),
                None,
                "코너 접촉에 면 계보가 붙었다 (t={t})"
            );
            // 법선은 코너 방향이어야 한다 — 박스 면 법선으로 눌리면 Y·Z 침투가 안 풀린다.
            assert!(
                m.normal.dot(cn).abs() > 0.99,
                "법선이 코너 방향이 아니다 (t={t}): {:?}",
                m.normal
            );
        }
    }

    /// **MAJOR (검수 rev_7992e5a3)**: 선분이 참조면 **코너를 정확히** 통과하면 인접한 두
    /// 측면이 같은 t를 만든다. 순회 순서로 한쪽만 기록하면, 1 ulp 다른 자세에서 어느 t가
    /// 조금 더 큰지가 뒤집히며 `EdgeCross { b: s }`가 `b: s+1`로 바뀐다 — 위치는 사실상
    /// 같은데 ID만 흔들려 warm start가 매 프레임 끊긴다. 코너 자체로 정규화해야 한다.
    #[test]
    fn a_segment_through_a_reference_corner_names_the_corner() {
        let r = square(0.0, 0.0, 1.0, 0.0); // 정점 0=(-1,-1) 1=(1,-1) 2=(1,1) 3=(-1,1)
        let n = Vec3::new(0.0, 0.0, 1.0);
        // 대각선이 코너 (-1,-1)과 (1,1)을 정확히 지난다.
        let seg = clip_segment(&r, n, Vec3::new(-3.0, -3.0, 0.1), Vec3::new(3.0, 3.0, 0.1))
            .expect("대각선이 참조면을 가로지른다");
        assert_eq!(seg[0].kind, ClipKind::ReferenceVertex, "{:?}", seg[0]);
        assert_eq!(
            seg[0].a, 0,
            "코너 (-1,-1)은 참조면 정점 0이다: {:?}",
            seg[0]
        );
        assert_eq!(seg[1].kind, ClipKind::ReferenceVertex, "{:?}", seg[1]);
        assert_eq!(seg[1].a, 2, "코너 (1,1)은 참조면 정점 2다: {:?}", seg[1]);
        // 음성 대조: 코너를 비껴가면 종전대로 에지 교점이다.
        let off = clip_segment(&r, n, Vec3::new(-3.0, -2.6, 0.1), Vec3::new(3.0, 3.4, 0.1))
            .expect("비껴간 대각선도 가로지른다");
        for v in &off {
            assert_eq!(
                v.kind,
                ClipKind::EdgeCross,
                "비껴갔는데 코너로 찍혔다: {v:?}"
            );
        }
        assert_ne!(off[0].b, off[1].b, "양 끝이 같은 측면에 잘렸다");
    }

    /// 뚜껑 링은 **정16각형**이어야 한다. 기울어 닿을 때 현-호 오차가 침투 방향으로
    /// `sin θ`만큼 투영되므로(검수 rev_7992e5a3 MAJOR), 각 간격이 커지면 누락 높이가
    /// 그대로 커진다: `r·(1−cos(π/N))·sin θ`. 허용 끝(sin θ = 0.312)·r=0.3에서
    /// N=8이면 7.1 mm, N=16이면 1.8 mm다.
    #[test]
    fn the_cap_ring_is_a_regular_16_gon() {
        assert_eq!(
            CAP_RING.len(),
            16,
            "N을 줄이면 기울어진 뚜껑의 누락 높이가 커진다"
        );
        // cos(2π/16) = cos(22.5°). 이웃 내적이 전부 이 값이어야 정다각형이다.
        const NEIGHBOR_DOT: f32 = 0.923_879_5;
        for i in 0..CAP_RING.len() {
            let (ax, az) = CAP_RING[i];
            let (bx, bz) = CAP_RING[(i + 1) % CAP_RING.len()];
            assert!(
                ((ax * ax + az * az).sqrt() - 1.0).abs() < 1e-6,
                "{i}번 정점이 단위원 위에 없다"
            );
            assert!(
                (ax * bx + az * bz - NEIGHBOR_DOT).abs() < 1e-5,
                "{i}번 이웃 간격이 22.5°가 아니다: {}",
                ax * bx + az * bz
            );
        }
        // 최악 현-호 오차(sagitta) = 1 − cos(π/16). 그 값이 문서화된 1.92%인지 본다.
        let half_dot = ((1.0 + NEIGHBOR_DOT) * 0.5).sqrt(); // cos(π/16), 반각 공식
        let sagitta = 1.0 - half_dot;
        assert!(
            (sagitta - 0.019_215).abs() < 1e-4,
            "sagitta가 문서값과 다르다: {sagitta}"
        );
    }

    /// **MAJOR (검수 rev_83950af0)**: 코너 판정을 `t`의 비트 동일성으로 하면 안 된다.
    /// 선분이 참조면 정점을 기하학적으로 정확히 지나도 인접 두 측면의 dot/나눗셈 반올림이
    /// 달라 `t`가 1 ulp 어긋난다 — 아래 반례에서 0.5와 0.49999997이 나온다. 그러면 사실상
    /// 같은 자리에 `EdgeCross` 두 개가 서로 다른 측면 ID로 생기고, 1 ulp 자세 변화에
    /// 어느 쪽이 남는지가 뒤집혀 warm start가 끊긴다.
    #[test]
    fn a_corner_hit_is_recognised_even_when_the_two_sides_disagree_by_an_ulp() {
        // 검수가 준 반례: f32 중점이 정확히 첫 정점인 평행사변형과 선분.
        let r = vec![
            Vec3::new(-3.170_453_8, -1.380_239_6, 0.0),
            Vec3::new(-1.958_739, -2.208_727_4, 0.0),
            Vec3::new(3.170_453_8, 1.380_239_6, 0.0),
            Vec3::new(1.958_739, 2.208_727_4, 0.0),
        ];
        let p0 = Vec3::new(-2.903_672_7, 1.612_034, 0.1);
        let p1 = Vec3::new(-3.437_234_9, -4.372_513_3, 0.1);
        let seg =
            clip_segment(&r, Vec3::new(0.0, 0.0, 1.0), p0, p1).expect("선분이 참조면을 가로지른다");
        let corner = seg
            .iter()
            .find(|v| v.kind == ClipKind::ReferenceVertex)
            .unwrap_or_else(|| panic!("코너를 못 알아봤다: {seg:?}"));
        assert_eq!(corner.a, 0, "정점 0을 지난다: {corner:?}");
        // 같은 자리에 접점이 둘이면 솔버가 그 지점에 제약을 두 번 건다.
        let d = seg[0].position - seg[1].position;
        assert!(
            d.length() > 1e-4 || seg[0].kind == seg[1].kind,
            "사실상 같은 자리에 서로 다른 계보가 둘 생겼다: {seg:?}"
        );
    }

    /// **MAJOR (검수 rev_83950af0)**: 후보 축약이 고정 방향 극점이면 최심점이 첫 칸을
    /// 차지할 때 방향을 **7개만** 본다. 실측: 정16각형에서 1번 정점만 조금 더 깊게 두면
    /// 표본이 `1,4,8,12`가 되어 면적 3.8478이 되고, 전수 최대 `0,4,8,12`의 4.0000을 놓쳤다.
    /// 최원점 표본은 지름을 먼저 잡으므로 전수 최대를 되찾는다.
    #[test]
    fn reduction_recovers_the_maximum_area_quad_on_a_16_gon() {
        use super::super::{ContactPoint, FeatureId};
        let n = Vec3::new(0.0, 0.0, 1.0);
        let pts: Vec<ContactPoint> = CAP_RING
            .iter()
            .enumerate()
            .map(|(i, (cx, cz))| ContactPoint {
                position: Vec3::new(*cx, *cz, 0.0),
                // 1번만 조금 더 깊게 — 최심점이 방향 극점과 겹치지 않게 만든다.
                separation: if i == 1 { -0.02 } else { -0.01 },
                feature: FeatureId(i as u32),
            })
            .collect();
        let out = reduce_to_4_general(&pts, n);
        let o = out[0].position;
        let area = ((out[1].position - o).cross(out[2].position - o).dot(n)
            + (out[2].position - o).cross(out[3].position - o).dot(n))
        .abs();
        assert!(
            area > 3.99,
            "전수 최대(4.0)에 못 미친다: {area} — 고른 정점 {:?}",
            out.iter().map(|p| p.feature.0).collect::<Vec<_>>()
        );
        // 최심점은 반드시 들어간다(박스 규칙의 전제).
        assert!(
            out.iter().any(|p| p.feature.0 == 1),
            "최심점이 빠졌다: {:?}",
            out.iter().map(|p| p.feature.0).collect::<Vec<_>>()
        );
    }

    /// **MAJOR (검수 rev_83950af0)**: 고정 N의 뚜껑 다각형은 오차가 **반지름에 비례**한다
    /// (누락 높이 `r·(1−cos(π/N))·sin θ` ≈ 0.006·r). 원기둥 반지름은 1000 m까지 합법이고
    /// speculative margin은 0까지 합법이라, N=16으로 줄여도 계약 전체를 못 덮는다.
    /// 실측: 반지름 10 m 뚜껑을 평면에 1 mm 침투시키면 물체 고정 링에서는 최저 후보가
    /// margin 밖이라 **다접점이 통째로 필터되고 단일 witness로 떨어진다**.
    ///
    /// 그래서 링을 접촉 방향에 정렬한다 — 정점 0이 원판의 지지점과 정확히 일치하므로
    /// 침투 방향 누락이 0이 된다. 아래는 링 정점 사이 한가운데(11.25°)로 원기둥을 축 둘레로
    /// 돌린 최악 배치다. 물체 고정 링이면 spin=0에서만 맞고 11.25°에서는 계보가 사라진다
    /// (실측: sep −0.001 → +0.00372, 계보 None).
    #[test]
    fn a_large_tilted_cap_reports_the_true_deepest_point_at_any_spin() {
        use crate::manifold::feature::unpack;
        use crate::{Pose, Shape};
        use maru_math::Quat;
        let floor = Shape::Box {
            half: Vec3::new(50.0, 0.5, 50.0),
            radius: 0.0,
        };
        let pf = Pose {
            pos: Vec3::new(0.0, -0.5, 0.0),
            rot: Quat::IDENTITY,
        };
        let (hh, r) = (2.0f32, 10.0f32);
        // 뚜껑 경로 안쪽(|축·법선| = 0.96)으로 기울인다. sin θ = 0.28.
        let cost = 0.96f32;
        let sint = (1.0 - cost * cost).sqrt();
        let theta = sint.atan2(cost);
        let cyl = Shape::Cylinder {
            half_height: hh,
            radius: r,
        };
        // spin 0 = 링 정점이 최급강하 방향과 겹치는 최선, 11.25° = 정점 사이 한가운데 최악.
        for spin in [0.0f32, core::f32::consts::PI / 16.0] {
            let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), theta)
                * Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), spin);
            // 최저 테두리점이 바닥(y=0)을 1 mm 뚫도록 높이를 잡는다.
            let y = -0.001 + hh * cost + r * sint;
            let pa = Pose {
                pos: Vec3::new(0.0, y, 0.0),
                rot,
            };
            let m = crate::manifold::dispatch::contact(&cyl, &pa, &floor, &pf, 0.05)
                .unwrap_or_else(|| panic!("spin={spin}에서 접촉이 있어야 한다"));
            let f = unpack(m.points()[0].feature)
                .unwrap_or_else(|| panic!("spin={spin}에서 뚜껑 다접점을 잃고 witness로 떨어졌다"));
            assert_eq!(f.inc_face, 1, "−Y 뚜껑이어야 한다: {f:?}");
            let deepest = m
                .points()
                .iter()
                .map(|p| p.separation)
                .fold(f32::INFINITY, f32::min);
            assert!(
                (deepest + 0.001).abs() < 1e-4,
                "spin={spin}에서 최심 접점이 실제 1 mm 침투와 다르다: {deepest}"
            );
        }
    }

    /// 분리된 면끼리는 폴리곤이 비고 `false` — 호출자가 witness로 폴백한다.
    #[test]
    fn disjoint_faces_produce_no_polygon() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        let i = square(5.0, 5.0, 0.5, 0.1);
        assert!(!clip_convex_face(
            &r,
            Vec3::new(0.0, 0.0, 1.0),
            &i,
            &mut out
        ));
    }

    /// 퇴화 입력(정점 2개 / 상한 초과)은 거절한다. 경계 바로 안쪽은 통과해야 한다.
    #[test]
    fn degenerate_vertex_counts_are_rejected_at_the_boundary() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let two = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.1)];
        assert!(
            !clip_convex_face(&r, n, &two, &mut out),
            "정점 2개가 통과했다"
        );
        assert!(
            !clip_convex_face(&two, n, &r, &mut out),
            "참조면 2개가 통과했다"
        );
        let over = convex_lens(MAX_FACE_VERTS + 2, 1.0, 0.1);
        assert!(
            !clip_convex_face(&r, n, &over, &mut out),
            "상한 초과가 통과했다"
        );
        let at_cap = convex_lens(MAX_FACE_VERTS, 0.5, 0.1);
        assert!(
            clip_convex_face(&r, n, &at_cap, &mut out),
            "경계값(상한 정확히)이 거절됐다"
        );
    }

    /// 참조면 감기 방향이 반대여도 같은 겹침이 나온다 — 측면 안쪽 방향을 중심으로 맞추므로.
    #[test]
    fn reversed_reference_winding_gives_the_same_overlap() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        let i = square(0.6, 0.6, 1.0, 0.1);
        let mut ccw = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        assert!(clip_convex_face(&r, n, &i, &mut ccw));
        let mut cw = GenClip::new();
        let mut rr = r.clone();
        rr.reverse();
        assert!(clip_convex_face(&rr, n, &i, &mut cw));
        assert_eq!(ccw.as_slice().len(), cw.as_slice().len());
        for v in cw.as_slice() {
            assert!(
                ccw.as_slice()
                    .iter()
                    .any(|w| (w.position - v.position).length() < 1e-5),
                "감기를 뒤집었더니 없던 점이 생겼다: {v:?}"
            );
        }
    }
}
