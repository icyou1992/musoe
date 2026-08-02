//! 볼록 껍질의 **면 구조**(canonical face topology) — 생성 시 한 번 만드는 파생물.
//!
//! ## 왜 필요한가
//!
//! 면 클리핑 매니폴드는 "접촉 법선과 가장 잘 맞는 **면**"을 알아야 한다. 지금 엔진은
//! 박스-박스만 그 면을 알고(박스는 면이 6개로 고정) 나머지 볼록 쌍은 GJK/EPA witness
//! **단일점**으로 떨어져 평면 접촉이 불안정하다(실측: 박스 모양 hull이 지면에 가라앉는다).
//!
//! `Shape::Hull`은 점 집합만 들고 있어서, 런타임에 면을 뽑으려면 "`p·dir`가 최대인 점들"을
//! 임계로 모아야 한다. **그 임계가 위험하다** — 이 저장소에서 임계 기반 기하 판정이 반복해서
//! 결함의 원인이었다(퇴화 코어의 법선·저차원 판정·동률 증명). 그래서 면 구조를 **생성 시**
//! 한 번 정확히 만들고 런타임은 그걸 읽기만 한다(교차검수 rev_ab7232ed 승인 설계).
//!
//! ## 파생물이지 캐노니컬 상태가 아니다
//!
//! 스냅샷에는 **점만** 저장하고 복원에서 다시 만든다(`TriMeshData`의 BVH·`HeightfieldGrid`의
//! 타일과 같은 정책). 그래서 빌더는 **결정적**이어야 한다 — 같은 점 배열이면 같은 면 목록,
//! 같은 순서, 같은 정점 순서. 테스트가 그걸 잠근다.
//!
//! ## 알고리즘: 삼중조합 전수 + 지지 검사
//!
//! 점이 **볼록 위치에 있다고 가정**한다(엔진 계약: hull 점은 빌더가 준다). 그러면 면 평면은
//! "세 점이 만든 평면 중 나머지 점이 전부 한쪽에 있는 것"이다. 상한이 32점이라 삼중조합이
//! 4,960개뿐이고 생성은 스텝 중이 아니므로 전수로 돌린다 — **근사가 아니라 열거**다.
//!
//! 같은 면을 여러 삼중조합이 만들므로 **경계 정점 순환열**로 중복을 없앤다(각도 임계를
//! 쓰지 않는다 — 얕게 인접한 면이 합쳐진다). 평면 위 점은 2D 볼록 껍질로 경계만 남긴다.
//!
//! ## 알려진 한계: **거의 평면인 점 집합은 면을 못 만든다**(fail-closed)
//!
//! 지지 평면을 모으는 `hi − eps` 대역은 "거의 공면"인 점들을 후보마다 조금씩 다른 부분집합
//! 으로 잡는다. 그러면 같은 물리적 면이 서로 다른 순환열 = 서로 다른 면으로 들어간다.
//! 실측(지름 2.28, z가 ±1e-6로 흔들리는 8점 링): 아무 방어도 없으면 **42면**, 사후 재검증
//! (공면성+지지 조건)으로 39면, 사전 검증(삼중조합 평면 자체가 지지 평면인가)까지 넣어도
//! **33면**이다 — 오일러 상한 12를 크게 넘는다.
//!
//! 세 방어를 다 두되, **오일러 상한을 넘으면 빈 구조로 포기한다**. 가짜 면 수십 개를 들고
//! 클리핑에 넘기면 hull을 가로지르는 평면이 진짜 접촉 영역을 잘라내기 때문이다. 빈 구조는
//! 호출자가 기존 witness 경로로 가라는 뜻이고, 그건 **이 모듈이 생기기 전과 같은 동작**이라
//! 회귀가 아니다. 근본 해결은 삼중조합 열거가 아니라 quickhull류 위상 구축이고, 그건 별도
//! 작업이다(검수 rev_8e10c17d).

use crate::limits::MAX_HULL_POINTS;
use musoe_math::Vec3;

/// 면 하나 — 바깥 법선과 그 위 정점(원본 인덱스, 법선 기준 CCW).
#[derive(Clone, Debug, PartialEq)]
pub struct HullFace {
    /// 바깥을 향하는 단위 법선.
    pub normal: Vec3,
    /// 평면 오프셋 — 면 위 점 `p`는 `normal·p == offset`.
    pub offset: f32,
    /// 이 면 위의 정점 인덱스, **법선에서 봤을 때 CCW**. 3개 이상이다.
    pub verts: Vec<u8>,
}

/// hull의 **점 + 면 구조**를 불가분으로 묶은 데이터.
///
/// 면은 점에서 유도되는 **파생물**이라 둘이 따로 놀면 안 된다 — 점만 바꾸고 면을 안 고치면
/// 클리핑이 없는 면을 참조한다. 그래서 필드를 비공개로 두고 생성자에서만 함께 만든다
/// (`TriMeshData`가 정점·BVH를 묶는 것과 같은 패턴).
///
/// 스냅샷에는 **점만** 저장하고 복원에서 면을 다시 만든다. 그래서 빌더가 결정적이어야 한다.
#[derive(Clone, Debug)]
pub struct HullData {
    points: Vec<Vec3>,
    faces: HullFaces,
}

/// **캐노니컬 데이터만 비교한다** — 면은 점의 함수이므로 점이 같으면 같은 hull이다
/// (스냅샷 왕복 검증이 이 비교를 쓴다).
impl PartialEq for HullData {
    fn eq(&self, other: &Self) -> bool {
        self.points == other.points
    }
}

impl HullData {
    /// 점에서 hull을 만든다 — 면 구조를 함께 계산한다.
    pub fn new(points: Vec<Vec3>) -> Self {
        let faces = HullFaces::build(&points);
        Self { points, faces }
    }
    #[inline]
    pub fn points(&self) -> &[Vec3] {
        &self.points
    }
    /// 면 구조. 저차원·병리 입력에서는 비어 있다 — 호출자는 witness 경로로 간다.
    #[inline]
    pub fn faces(&self) -> &HullFaces {
        &self.faces
    }
}

/// hull의 면 목록. 인덱스가 곧 **면 id**이고, feature 계보가 이 id를 쓴다.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HullFaces {
    faces: Vec<HullFace>,
}

impl HullFaces {
    pub fn faces(&self) -> &[HullFace] {
        &self.faces
    }
    pub fn len(&self) -> usize {
        self.faces.len()
    }
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// 방향 `dir`에 가장 잘 맞는 면의 인덱스와 그 정렬도(`normal·dir`).
    ///
    /// **임계가 없다** — 최대를 고르고 동률은 낮은 인덱스가 이긴다(면 목록이 결정적이므로
    /// 답도 결정적이다). 면이 없으면 `None`(퇴화 hull).
    pub fn best_face(&self, dir: Vec3) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, f) in self.faces.iter().enumerate() {
            let d = f.normal.dot(dir);
            if best.is_none_or(|(_, bd)| d > bd) {
                best = Some((i, d));
            }
        }
        best
    }

    /// 점 집합에서 면 구조를 만든다. 점이 **볼록 위치**에 있다고 가정하지 않는다 —
    /// 지지 평면 위에서 2D 볼록 껍질을 다시 돌려 경계만 남기므로 내부점이 섞여도 안전하다.
    ///
    /// 3차원이 아니면(모두 한 직선 위·한 자리) 면이 0개인 빈 구조를 돌려준다 — 오류가
    /// 아니다. 호출자는 `is_empty()`를 보고 기존 witness 경로로 간다.
    pub fn build(points: &[Vec3]) -> Self {
        let n = points.len();
        if !(4..=MAX_HULL_POINTS).contains(&n) {
            return Self::default();
        }
        // **임계는 점 집합의 지름에 상대적**이다 — 절대 상수는 장면 스케일에서 깨진다.
        let mut span = 0.0f32;
        for p in points {
            for q in points {
                span = span.max((*p - *q).length());
            }
        }
        if span <= 0.0 || span.is_nan() {
            return Self::default();
        }
        let eps = crate::length_eps(span);

        let mut faces: Vec<HullFace> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let Some(nrm) = (points[j] - points[i])
                        .cross(points[k] - points[i])
                        .try_normalize()
                    else {
                        continue; // 세 점이 한 직선 위 — 평면을 못 만든다
                    };
                    // 두 방향 모두 후보다: 어느 쪽이 바깥인지는 나머지 점이 정한다.
                    for cand in [nrm, nrm * -1.0] {
                        Self::insert_face(&mut faces, points, cand, cand.dot(points[i]), eps);
                        // **메모리 상한**은 절대 최댓값으로 잡는다(입력 상한 32점 → 60면).
                        // 진짜 판정(아래 오일러 검사)은 순회가 끝난 뒤에 한다 — 도중에 하면
                        // "아직 정점을 다 안 본" 초기 상태에서 잘못 발화한다.
                        if faces.len() > 2 * MAX_HULL_POINTS - 4 {
                            return Self::default();
                        }
                    }
                }
            }
        }
        // **오일러 상한 판정은 실제 정점 수 기준이다.** 입력 길이를 쓰면 내부점 패딩으로
        // 상한이 부풀어 가짜 면이 통과한다 — 8극점 + 내부점 24개인 합법 32점 hull에서
        // 상한이 12가 아니라 60이 되어 33개 가짜 면이 살아남는다(검수 rev_8eacf4a5 BLOCKER).
        // 경계 링에 실제로 쓰인 정점만 센다(내부점은 2D 껍질이 이미 걸렀다).
        let mut used = [false; MAX_HULL_POINTS];
        for face in &faces {
            for idx in &face.verts {
                used[*idx as usize] = true;
            }
        }
        let extreme = used.iter().filter(|u| **u).count();
        if extreme < 4 || faces.len() > 2 * extreme - 4 {
            return Self::default();
        }
        // 면 순서를 **입력만의 함수**로 고정한다. 삼중조합 순회 순서가 이미 결정적이지만,
        // 정렬을 명시해 두면 나중에 순회를 바꿔도 id가 안 흔들린다(feature 계보가 이 id다).
        faces.sort_by(|a, b| {
            a.verts
                .cmp(&b.verts)
                .then_with(|| a.normal.x.total_cmp(&b.normal.x))
                .then_with(|| a.normal.y.total_cmp(&b.normal.y))
                .then_with(|| a.normal.z.total_cmp(&b.normal.z))
        });
        Self { faces }
    }

    /// 후보 법선 하나를 면으로 만들어 넣는다(이미 있으면 버린다).
    ///
    /// **중복 판정에 각도 임계를 쓰지 않는다.** 예전에는 `법선 내적 > 0.9999`로 갈랐는데,
    /// 그러면 얕은 각으로 인접한 **서로 다른 면**이 하나로 합쳐진다(검수 rev_f601d4f4
    /// BLOCKER: 사면체 (-1,0,0)·(1,0,0)·(0,1,0)·(0,-1,0.01)이 4면이 아니라 3면이 됐다).
    /// 캐노니컬 키는 **경계 정점 순환열**이다 — 같은 면을 찾은 삼중조합은 같은 열을 만들고,
    /// 얇은 판의 앞뒤 두 면은 순서가 뒤집혀 서로 다른 열이 된다.
    fn insert_face(faces: &mut Vec<HullFace>, points: &[Vec3], normal: Vec3, base: f32, eps: f32) {
        // 지지 평면: 이 방향 최댓값. `points[i]`가 아니라 최댓값을 쓰면 어느 삼중조합에서
        // 왔는지와 무관하게 같은 평면이 나온다.
        let mut hi = f32::NEG_INFINITY;
        for p in points {
            hi = hi.max(normal.dot(*p));
        }
        // **삼중조합의 평면 자체가 지지 평면인가.** 이게 빠지면 hull 내부를 가로지르는
        // 단면이 후보로 들어오고, 그 방향으로 모은 점이 실제 면과 무관한 부분집합이 되어
        // 가짜 면이 무더기로 생긴다. 거의 평면인 8점 링에서 실측 42개(재검증만으로는 39개,
        // 사전 검증을 넣으면 2개)였다 — 검수 rev_8e10c17d가 지적한 자리다.
        if base < hi - eps {
            return;
        }
        // 평면 위(안쪽 eps 이내) 점들. 여기엔 **면 내부 점도 섞인다** — 다음 단계가 거른다.
        let mut on: Vec<u8> = Vec::new();
        for (idx, p) in points.iter().enumerate() {
            if normal.dot(*p) >= hi - eps {
                on.push(idx as u8);
            }
        }
        if on.len() < 3 {
            return;
        }
        // **경계만 남긴다.** 평면 위 점을 그대로 쓰면 면 내부 점이 경계에 끼어들어 오목한
        // 사슬이 되고, 나중 Sutherland–Hodgman이 진짜 면의 일부를 잘라낸다. 게다가 내부
        // 점이 중심과 겹치면 기준축을 못 만들어 면이 통째로 사라졌다(같은 검수 BLOCKER).
        let Some(ring) = convex_ring(points, &on, normal) else {
            return;
        };
        if faces.iter().any(|f| f.verts == ring) {
            return;
        }
        // 법선·오프셋을 **경계에서 다시 계산**한다(Newell) — 삼중조합이 준 값보다 안정적이고
        // 어느 조합에서 왔는지와 무관해진다. 다만 **다시 계산한 평면을 재검증해야 한다**:
        // `hi - eps` 대역은 "거의 평면"인 점들을 한 면으로 끌어모을 수 있고, 그러면 삼중조합
        // 마다 다른 부분집합이 잡혀 서로 다른 순환열 = 서로 다른 "면"이 무더기로 들어간다.
        // 그 면들은 hull을 가로지르므로 클리핑이 진짜 접촉 영역을 잘라낸다
        // (검수 rev_8e10c17d BLOCKER).
        let mut nsum = Vec3::ZERO;
        for a in 0..ring.len() {
            let p0 = points[ring[a] as usize];
            let p1 = points[ring[(a + 1) % ring.len()] as usize];
            nsum += p0.cross(p1);
        }
        let Some(nrm) = nsum.try_normalize() else {
            return; // 면적 0 — 면이 아니다
        };
        let mut off = 0.0f32;
        for idx in &ring {
            off += nrm.dot(points[*idx as usize]);
        }
        off /= ring.len() as f32;
        // ①② 는 **테스트로 관측되지 않는 방어**다(뮤테이션으로 확인: 빼도 모든 테스트가
        // 통과한다). 정상 입력에서는 위 사전 검증이 이미 보장하고, 병리 입력에서는 오일러
        // 상한이 먼저 걸리기 때문이다. 그래도 남기는 이유는 셋을 합친 효과가 **실측**되기
        // 때문이다: 거의 평면인 8점 링에서 방어 없음 42면 → 사후만 39면 → 사전까지 33면.
        // 상한이 12이므로 이 입력은 어차피 포기되지만, 상한 근처(≤12) 입력에서는 이 차이가
        // 가짜 면 한둘을 실제로 걸러낸다.
        //
        // ① 링이 정말 **공면**인가 — 평균 평면에서 eps 안에 있어야 한다.
        for idx in &ring {
            if (nrm.dot(points[*idx as usize]) - off).abs() > eps {
                return;
            }
        }
        // ② 그 평면이 정말 **지지 평면**인가 — 어떤 점도 바깥에 없어야 한다.
        //    (①만 보면 hull을 가로지르는 얇은 판이 통과한다.)
        for p in points {
            if nrm.dot(*p) > off + eps {
                return;
            }
        }
        faces.push(HullFace {
            normal: nrm,
            offset: off,
            verts: ring,
        });
    }
}

/// 법선 `n`에 수직인 **오른손 기저** `(u, v)` — `u × v = n`.
///
/// `n`과 가장 덜 정렬된 축에서 만들어 퇴화를 피한다. 각도 함수를 쓰지 않으므로
/// 플랫폼 간 결과가 같다.
fn plane_basis(n: Vec3) -> Option<(Vec3, Vec3)> {
    let a = if n.x.abs() <= n.y.abs() && n.x.abs() <= n.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if n.y.abs() <= n.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let u = (a - n * n.dot(a)).try_normalize()?;
    Some((u, n.cross(u)))
}

/// 평면 위 점 인덱스들의 **볼록 경계**를 법선 기준 CCW로 돌려준다.
///
/// Andrew monotone chain — 좌표 정렬 + 외적 부호만 쓴다. **각도 함수를 쓰지 않는다**:
/// `atan2`는 native와 wasm에서 마지막 비트가 갈릴 수 있고, 이 순서가 면 id를 정하므로
/// 그러면 복원한 월드의 warm start가 끊긴다(검수 rev_f601d4f4 MAJOR).
///
/// 공선 중간점·중복점·면 내부 점은 여기서 빠진다.
/// 시작 인덱스를 **최소 인덱스로 회전**해 순환열을 캐노니컬하게 만든다(중복 판정 키).
fn convex_ring(points: &[Vec3], on: &[u8], n: Vec3) -> Option<Vec<u8>> {
    let (u, v) = plane_basis(n)?;
    let xy = |idx: u8| {
        let p = points[idx as usize];
        (p.dot(u), p.dot(v))
    };
    let mut idx: Vec<u8> = on.to_vec();
    idx.sort_by(|a, b| {
        let (ax, ay) = xy(*a);
        let (bx, by) = xy(*b);
        ax.total_cmp(&bx)
            .then_with(|| ay.total_cmp(&by))
            .then_with(|| a.cmp(b))
    });
    // 외적 부호로 좌회전 판정. 0(공선)도 버려 중간점을 남기지 않는다.
    let cross = |o: u8, a: u8, b: u8| {
        let (ox, oy) = xy(o);
        let (ax, ay) = xy(a);
        let (bx, by) = xy(b);
        (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
    };
    let mut lower: Vec<u8> = Vec::new();
    for &p in &idx {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<u8> = Vec::new();
    for &p in idx.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    if lower.len() < 3 {
        return None;
    }
    // 최소 인덱스가 앞에 오도록 회전 — 같은 면이면 같은 열이 되어야 중복 판정이 성립한다.
    let start = lower
        .iter()
        .enumerate()
        .min_by_key(|(_, x)| **x)
        .map(|(i, _)| i)?;
    lower.rotate_left(start);
    Some(lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn box_points(hx: f32, hy: f32, hz: f32) -> Vec<Vec3> {
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(hx * sx, hy * sy, hz * sz));
                }
            }
        }
        pts
    }

    #[test]
    fn box_hull_has_six_quad_faces() {
        let f = HullFaces::build(&box_points(1.0, 2.0, 3.0));
        assert_eq!(f.len(), 6, "박스는 면이 6개다");
        for face in f.faces() {
            assert_eq!(face.verts.len(), 4, "박스 면은 사각형이다: {face:?}");
            // 법선은 축 하나에만 성분이 있다.
            let n = face.normal;
            let axes = [n.x.abs(), n.y.abs(), n.z.abs()];
            let big = axes.iter().filter(|a| **a > 0.5).count();
            assert_eq!(big, 1, "박스 면 법선이 축 정렬이 아니다: {n:?}");
        }
    }

    /// **법선은 바깥을 향해야 한다** — 안쪽이면 클리핑의 reference/incident가 뒤집힌다.
    #[test]
    fn face_normals_point_outward() {
        for pts in [box_points(1.0, 1.0, 1.0), box_points(0.1, 5.0, 0.3)] {
            let f = HullFaces::build(&pts);
            for face in f.faces() {
                // 중심(원점)은 모든 면의 안쪽에 있어야 한다.
                assert!(
                    face.normal.dot(Vec3::ZERO) < face.offset,
                    "법선이 안쪽을 본다: {face:?}"
                );
                // 그리고 어떤 점도 면 밖에 없다.
                for p in &pts {
                    assert!(
                        face.normal.dot(*p) <= face.offset + 1e-4,
                        "면 밖에 점이 있다: {p:?} vs {face:?}"
                    );
                }
            }
        }
    }

    /// 면 정점은 **법선 기준 CCW**여야 한다 — 클리핑의 측면 평면 방향이 여기서 나온다.
    #[test]
    fn face_verts_are_ccw_around_the_normal() {
        let pts = box_points(1.0, 2.0, 3.0);
        let f = HullFaces::build(&pts);
        for face in f.faces() {
            let mut area = Vec3::ZERO;
            let m = face.verts.len();
            for i in 0..m {
                let a = pts[face.verts[i] as usize];
                let b = pts[face.verts[(i + 1) % m] as usize];
                area += a.cross(b);
            }
            assert!(
                area.dot(face.normal) > 0.0,
                "정점 순서가 CW다: {face:?} (area·n = {})",
                area.dot(face.normal)
            );
        }
    }

    /// 사면체 — 면 4개, 전부 삼각형.
    #[test]
    fn tetrahedron_has_four_triangular_faces() {
        let pts = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(0.0, 0.0, 1.0),
        ];
        let f = HullFaces::build(&pts);
        assert_eq!(f.len(), 4);
        for face in f.faces() {
            assert_eq!(face.verts.len(), 3);
        }
    }

    /// **면 안쪽 점은 경계에서 빠져야 한다.** 평면 위 점을 그대로 쓰면 오목한 사슬이
    /// 되고, 나중 클리핑이 진짜 면의 일부를 잘라낸다. 게다가 그 점이 배열 앞에 오면
    /// 중심과 겹쳐 면이 통째로 사라졌다(검수 rev_f601d4f4 BLOCKER).
    /// **점 순서와 무관**해야 하므로 앞·뒤 두 배치를 모두 시험한다.
    #[test]
    fn interior_coplanar_point_is_excluded_from_the_face() {
        for center_first in [false, true] {
            let mut pts = if center_first {
                vec![v(0.0, 0.0, 1.0)]
            } else {
                Vec::new()
            };
            pts.extend(box_points(1.0, 1.0, 1.0));
            if !center_first {
                pts.push(v(0.0, 0.0, 1.0));
            }
            let f = HullFaces::build(&pts);
            assert_eq!(
                f.len(),
                6,
                "면 수가 다르다(center_first={center_first}): {}",
                f.len()
            );
            let zface = f
                .faces()
                .iter()
                .find(|f| f.normal.z > 0.9)
                .unwrap_or_else(|| panic!("+Z 면이 사라졌다(center_first={center_first})"));
            assert_eq!(
                zface.verts.len(),
                4,
                "면 안쪽 점이 경계에 끼었다(center_first={center_first}): {zface:?}"
            );
        }
    }

    /// **얕은 각으로 인접한 면이 합쳐지면 안 된다.** 각도 임계로 중복을 판정하면
    /// 사면체가 4면이 아니라 3면이 된다(검수 rev_f601d4f4 BLOCKER).
    #[test]
    fn shallow_adjacent_faces_are_not_merged() {
        let pts = vec![
            v(-1.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(0.0, -1.0, 0.01),
        ];
        let f = HullFaces::build(&pts);
        assert_eq!(f.len(), 4, "면이 합쳐졌다: {} (기대 4)", f.len());
        for a in 0..4 {
            for b in (a + 1)..4 {
                let d = f.faces()[a].normal.dot(f.faces()[b].normal);
                assert!(d < 0.999_99, "면 {a}·{b} 법선이 사실상 같다: {d}");
            }
        }
    }

    /// **평면 점 집합은 앞뒤 두 면**을 만든다 — 얇은 판의 양면이고, convex radius가 두께를
    /// 주므로 실제로 클리핑할 수 있는 면이다. 지지 검사가 ±n 양쪽에서 통과하기 때문이고,
    /// 그게 기하학적으로 맞다(모든 점이 두 반공간 경계 위에 있다).
    #[test]
    fn planar_point_set_has_two_opposite_faces() {
        let planar = vec![
            v(-1.0, 0.0, -1.0),
            v(1.0, 0.0, -1.0),
            v(1.0, 0.0, 1.0),
            v(-1.0, 0.0, 1.0),
        ];
        let f = HullFaces::build(&planar);
        assert_eq!(f.len(), 2, "평면 판은 앞뒤 두 면이다");
        let (a, b) = (&f.faces()[0], &f.faces()[1]);
        assert!(
            (a.normal + b.normal).length() < 1e-5,
            "두 면이 서로 반대가 아니다: {:?} / {:?}",
            a.normal,
            b.normal
        );
        assert_eq!(a.verts.len(), 4);
        assert_eq!(b.verts.len(), 4);
        // 두 면 다 CCW 계약을 지킨다(방향이 반대이므로 정점 순서도 반대여야 한다).
        for face in f.faces() {
            let mut area = Vec3::ZERO;
            let m = face.verts.len();
            for i in 0..m {
                area +=
                    planar[face.verts[i] as usize].cross(planar[face.verts[(i + 1) % m] as usize]);
            }
            assert!(area.dot(face.normal) > 0.0, "CW다: {face:?}");
        }
    }

    /// **좌표가 딱 떨어지지 않는 면**에서도 공면 정점이 모두 한 면에 모여야 한다.
    ///
    /// 축 정렬 박스는 평면 산술이 정확해서 임계가 0이어도 통과한다 — 그걸로는 임계가
    /// 일을 하는지 알 수 없다. 육각기둥은 뚜껑 6점의 법선·오프셋이 마지막 비트에서
    /// 흔들리므로 임계가 있어야 6각형 하나가 된다.
    #[test]
    fn hexagonal_prism_caps_are_single_faces() {
        let mut pts = Vec::new();
        for k in 0..6 {
            let a = k as f32 * std::f32::consts::TAU / 6.0;
            for sy in [-1.0f32, 1.0] {
                pts.push(v(a.cos() * 0.73, sy * 1.37, a.sin() * 0.73));
            }
        }
        let f = HullFaces::build(&pts);
        // 뚜껑 2개 + 옆면 6개.
        assert_eq!(f.len(), 8, "면 수가 다르다: {}", f.len());
        let caps: Vec<&HullFace> = f
            .faces()
            .iter()
            .filter(|f| f.normal.y.abs() > 0.9)
            .collect();
        assert_eq!(caps.len(), 2, "뚜껑이 2개가 아니다");
        for c in caps {
            assert_eq!(c.verts.len(), 6, "뚜껑이 6각형이 아니다: {c:?}");
        }
        for face in f.faces().iter().filter(|f| f.normal.y.abs() <= 0.9) {
            assert_eq!(face.verts.len(), 4, "옆면이 사각형이 아니다: {face:?}");
        }
    }

    /// **면 id는 발견 순서가 아니라 면 내용으로 정해진다.**
    ///
    /// feature 계보가 이 id를 쓰므로, 나중에 삼중조합 순회 순서를 바꿔도 id가 흔들리면
    /// warm start가 통째로 끊긴다. 그래서 빌더가 정해진 키로 정렬해 둔다 — 그 불변식을
    /// 여기서 잠근다(정렬을 빼면 발견 순서가 그대로 id가 된다).
    #[test]
    fn face_ids_follow_a_canonical_order() {
        for pts in [
            box_points(1.0, 2.0, 3.0),
            vec![
                v(0.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(0.0, 1.0, 0.0),
                v(0.0, 0.0, 1.0),
            ],
        ] {
            let f = HullFaces::build(&pts);
            let key = |x: &HullFace| {
                (
                    x.verts.clone(),
                    x.normal.x.to_bits(),
                    x.normal.y.to_bits(),
                    x.normal.z.to_bits(),
                )
            };
            let mut sorted: Vec<_> = f.faces().iter().map(key).collect();
            let asis = sorted.clone();
            sorted.sort();
            assert_eq!(asis, sorted, "면이 캐노니컬 순서가 아니다");
        }
    }

    /// **거의 평면인 점 집합**이 가짜 면을 무더기로 만들면 안 된다.
    ///
    /// 지지 평면 대역(`hi − eps`)은 비공면 점들을 한 면으로 끌어모을 수 있고, 그러면
    /// 삼중조합마다 다른 부분집합이 잡혀 서로 다른 순환열 = 서로 다른 "면"이 들어간다.
    /// 그 면들은 hull을 가로지르므로 클리핑이 진짜 접촉 영역을 잘라낸다
    /// (검수 rev_8e10c17d BLOCKER). 재계산한 평면의 **공면성 + 지지 조건**을 다시 확인해
    /// 막고, 그래도 넘치면 오일러 상한에서 통째로 포기한다.
    #[test]
    fn nearly_planar_ring_does_not_explode_into_fake_faces() {
        // 좌표는 검수가 준 f32 값 그대로다 — **딱 떨어지지 않아야** 이 결함이 재현된다.
        // 비트 패턴으로 넣어 리터럴 반올림·정밀도 린트가 값을 바꾸지 못하게 한다.
        let b = f32::from_bits;
        let pts = vec![
            v(b(0x3f6bc2b5), b(0x00000000), b(0x351bf64c)),
            v(b(0x3f3e92f7), b(0x3f3e92f7), b(0xb4fbba21)),
            v(b(0x00000000), b(0x3f8f8ae0), b(0x356e6252)),
            v(b(0xbf11d2ca), b(0x3f11d2ca), b(0x352161cd)),
            v(b(0xbf93a6f7), b(0x00000000), b(0xb4a10cca)),
            v(b(0xbf3e40e1), b(0xbf3e40e1), b(0x34b13649)),
            v(b(0x00000000), b(0xbf940b56), b(0x346ec85d)),
            v(b(0x3f3ee1c6), b(0xbf3ee1c6), b(0xb56a4ffb)),
        ];
        let f = HullFaces::build(&pts);
        // **fail-closed가 계약이다**: 방어를 다 걸어도 33면이 나오므로(모듈 주석의 실측)
        // 오일러 상한에서 통째로 포기한다. 빈 구조 = witness 경로 = 이 모듈 이전과 같은 동작.
        assert!(
            f.is_empty(),
            "거의 평면인 입력에서 면을 만들었다: {} (오일러 상한 {})",
            f.len(),
            2 * pts.len() - 4
        );
        // 상한이 **무차별 차단이 아니다** — 정상 입력은 여전히 면을 만든다.
        assert_eq!(HullFaces::build(&box_points(1.0, 1.0, 1.0)).len(), 6);
    }

    /// 저장된 면은 전부 **진짜 지지 평면**이어야 한다 — hull을 가로지르면 클리핑이
    /// 진짜 접촉 영역을 잘라낸다. 정상 입력 전반에 대해 잠근다.
    #[test]
    fn every_stored_face_is_a_supporting_plane() {
        let mut hexprism = Vec::new();
        for k in 0..6 {
            let a = k as f32 * std::f32::consts::TAU / 6.0;
            for sy in [-1.0f32, 1.0] {
                hexprism.push(v(a.cos() * 0.73, sy * 1.37, a.sin() * 0.73));
            }
        }
        for pts in [
            box_points(1.0, 2.0, 3.0),
            hexprism,
            vec![
                v(-1.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(0.0, 1.0, 0.0),
                v(0.0, -1.0, 0.01),
            ],
        ] {
            let f = HullFaces::build(&pts);
            assert!(!f.is_empty());
            let mut span = 0.0f32;
            for p in &pts {
                for q in &pts {
                    span = span.max((*p - *q).length());
                }
            }
            let tol = crate::length_eps(span) * 4.0;
            for face in f.faces() {
                for p in &pts {
                    assert!(
                        face.normal.dot(*p) <= face.offset + tol,
                        "면이 hull을 가로지른다: 점 {p:?}이 {face:?} 바깥"
                    );
                }
            }
        }
    }

    /// **내부점이 섞여도 정상 hull은 정확히 잡혀야 한다** — 오일러 상한을 실제 극점 수로
    /// 재기 때문에, 그 계산이 내부점을 세면 상한이 부풀고(가짜 면 통과) 반대로 너무 좁으면
    /// 정상 hull을 거절한다. 여기서 잠그는 것은 **거절하지 않는다** 쪽이다.
    ///
    /// 병리 입력(거의 평면인 링 + 내부점)은 실측 390면이라 절대 메모리 상한(60)에서 이미
    /// 걸린다 — 그 경로는 `nearly_planar_ring_does_not_explode_into_fake_faces`가 잠근다.
    #[test]
    fn interior_padding_does_not_break_a_valid_hull() {
        let mut pts = box_points(1.0, 1.0, 1.0);
        // 중심과 각 꼭짓점의 볼록 조합 — 정의상 hull 안이고 경계에 안 나온다.
        for t in [0.3f32, 0.6] {
            for k in 0..8 {
                pts.push(pts[k] * t);
            }
        }
        assert_eq!(pts.len(), 24);
        let f = HullFaces::build(&pts);
        assert_eq!(f.len(), 6, "내부점 때문에 면이 달라졌다: {}", f.len());
        for face in f.faces() {
            assert_eq!(face.verts.len(), 4, "내부점이 경계에 끼었다: {face:?}");
            // 그리고 쓰인 정점은 전부 원래 꼭짓점(인덱스 0..8)이어야 한다.
            for idx in &face.verts {
                assert!(*idx < 8, "내부점이 면 정점이 됐다: {idx}");
            }
        }
    }

    /// 1차원 이하(선·점)와 점이 너무 적은 경우는 **빈 구조**다 — 오류가 아니라
    /// "면 클리핑 불가"이고, 호출자는 기존 witness 경로로 간다.
    #[test]
    fn degenerate_point_sets_have_no_faces() {
        let planar = [
            v(-1.0, 0.0, -1.0),
            v(1.0, 0.0, -1.0),
            v(1.0, 0.0, 1.0),
            v(-1.0, 0.0, 1.0),
        ];
        // 한 직선 위
        let line: Vec<Vec3> = (0..5).map(|i| v(i as f32, 0.0, 0.0)).collect();
        assert!(HullFaces::build(&line).is_empty());
        // 점이 너무 적다
        assert!(HullFaces::build(&planar[..3]).is_empty());
        // 모든 점이 한 자리
        assert!(HullFaces::build(&[Vec3::ZERO; 5]).is_empty());
    }

    /// **결정적이어야 한다** — 파생물이라 복원에서 다시 만들고, feature 계보가 면 id를 쓴다.
    /// 같은 입력이면 면 목록·순서·정점 순서가 모두 같아야 한다.
    #[test]
    fn build_is_deterministic() {
        let pts = box_points(1.0, 2.0, 3.0);
        let a = HullFaces::build(&pts);
        for _ in 0..8 {
            assert_eq!(a, HullFaces::build(&pts), "같은 입력에 다른 면 구조");
        }
    }

    /// `best_face`는 임계 없이 최대를 고르고, 동률은 낮은 인덱스가 이긴다.
    #[test]
    fn best_face_picks_the_aligned_face() {
        let pts = box_points(1.0, 1.0, 1.0);
        let f = HullFaces::build(&pts);
        for (dir, want) in [
            (v(0.0, 1.0, 0.0), v(0.0, 1.0, 0.0)),
            (v(0.0, -1.0, 0.0), v(0.0, -1.0, 0.0)),
            (v(0.9, 0.1, 0.0), v(1.0, 0.0, 0.0)),
        ] {
            let (i, d) = f.best_face(dir).expect("면이 있다");
            assert!(
                (f.faces()[i].normal - want).length() < 1e-5,
                "{dir:?} → {:?} (기대 {want:?})",
                f.faces()[i].normal
            );
            assert!(d > 0.0);
        }
        // 동률(정육면체를 대각으로 보면 세 면이 같다) — 두 번 물어도 같은 답.
        let diag = v(1.0, 1.0, 1.0).try_normalize().unwrap();
        assert_eq!(f.best_face(diag), f.best_face(diag));
    }

    /// 상한(32점)에서도 만들어지고 면 수가 오일러 상한(2m−4) 안이다.
    #[test]
    fn respects_the_point_cap() {
        // 구 표면 샘플 32점 — 볼록 위치.
        let mut pts = Vec::new();
        let mut rng = 12345u32;
        let mut next = || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        while pts.len() < MAX_HULL_POINTS {
            if let Some(u) = v(next(), next(), next()).try_normalize() {
                pts.push(u);
            }
        }
        let f = HullFaces::build(&pts);
        assert!(!f.is_empty(), "32점 구 샘플에 면이 없다");
        assert!(
            f.len() <= 2 * MAX_HULL_POINTS - 4,
            "면 수가 오일러 상한을 넘었다: {}",
            f.len()
        );
        // 상한을 넘는 입력은 빈 구조(정책값 초과는 거절이다).
        pts.push(v(2.0, 0.0, 0.0));
        assert!(HullFaces::build(&pts).is_empty());
    }
}
