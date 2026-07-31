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
use crate::tolerance::{area_eps, characteristic_length, length_sq_eps, min_feature_size};
use crate::{Pose, Shape};
use maru_math::{Mat3, Quat, Vec3};

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
/// 복합 자식 feature 네임스페이스 슬롯 — bits 9..15 (7비트).
/// 모든 feature 생산자가 이 비트를 쓰지 않는다: 클립 패킹은 31·28..30·25..27·0..8,
/// 센티널(FEATURE_GENERIC/ANALYTIC)은 16..31만 쓴다. 그래서 `fid | (key<<9)`는
/// key(=자식쌍 식별, 1..=64)별로 **단사**다 — 같은 바디 페어의 서로 다른 자식쌍이
/// 같은 feature를 내도 warm start·PCM 매칭이 절대 섞이지 않는다(설계: 중복 판정 키에
/// source collider 포함 — PCM 키 무변경 달성).
pub const FEATURE_CHILD_SHIFT: u32 = 9;
pub const FEATURE_CHILD_MASK: u32 = 0x7F << FEATURE_CHILD_SHIFT;

impl Manifold {
    /// 자식쌍 키를 전 접점 feature에 접는다. 비복합 경로는 호출하지 않는다(key 0 예약).
    /// A/B 역할을 뒤집는다 — 법선 방향 계약(“A를 +normal로 밀면 분리”)을 지키려면
    /// 법선을 뒤집는 것으로 충분하다(접점 위치·침투는 대칭이다).
    pub fn flip(&mut self) {
        self.normal = -self.normal;
    }

    pub fn namespace_features(&mut self, key: u32) {
        debug_assert!((1..=64).contains(&key), "자식쌍 키는 1..=64");
        for i in 0..self.count as usize {
            debug_assert!(
                self.points[i].feature.0 & FEATURE_CHILD_MASK == 0,
                "feature 생산자가 네임스페이스 비트(9..15)를 침범했다"
            );
            self.points[i].feature.0 |= key << FEATURE_CHILD_SHIFT;
        }
    }
}

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

/// 하이트필드 접촉 — 겹치는 셀만 삼각형으로 펼친다.
///
/// `flip`이면 A/B 역할이 원래와 반대라 매니폴드 법선·접점 소유를 되돌린다.
/// 반환 = 상한(`MAX_HEIGHTFIELD_MANIFOLDS`)에서 끊었는가.
///
/// **feature ID는 전부 generic으로 덮는다.** 클리핑이 삼각형마다 부여하는 ID는 삼각형이
/// 다르면 같은 값이 나올 수 있어(같은 페어 안에서 충돌) warm start가 남의 임펄스를
/// 물려받는다. generic은 위치 근접으로 매칭되므로 삼각형이 달라도 옳게 갈린다 —
/// 계보 대신 위치를 쓰는 대가는 슬라이딩 시 계승률 하락뿐이다.
#[allow(clippy::too_many_arguments)]
fn heightfield_contacts(
    sa: &Shape,
    pa: &Pose,
    hf_shape: &Shape,
    hf_pose: &Pose,
    margin: f32,
    flip: bool,
    budget: &mut usize,
    f: &mut dyn FnMut(u8, u8, Manifold),
) -> ContactWalk {
    let hf = hf_shape.heightfield_ref().expect("Heightfield 변종");
    // 볼록 쪽 월드 AABB를 격자 로컬로 옮겨 셀 범위를 얻는다(margin만큼 넓힌다).
    let mut world = crate::transformed_aabb(sa.local_aabb(), pa);
    #[allow(unused_assignments)]
    let pad = Vec3::new(margin, margin, margin);
    world = crate::Aabb::new(world.min - pad, world.max + pad);
    let local = crate::heightfield::world_aabb_to_local(world, hf_pose);
    let Some(range) = hf.cell_range(local) else {
        return ContactWalk::Complete;
    };
    let (y_lo, y_hi) = (local.min.y, local.max.y);
    let mut emitted = 0usize;
    let mut visited = 0usize;
    let mut saturated = false;
    let mut over_budget = false;
    // 복합 자식도 함께 펼친다 — 자식 키는 그대로 콜백에 넘겨 네임스페이싱을 유지한다.
    let children: &[(Pose, Shape)] = match sa {
        Shape::Compound { children } => children,
        _ => &[],
    };
    // **타일 → 셀** 순서로 걷는다. 겹치지 않는 타일의 셀은 방문조차 하지 않고,
    // 방문 수가 예산을 넘으면 순회를 포기해 호출자가 자원 오류로 처리하게 한다.
    crate::heightfield::for_each_overlapping_cell(&hf, range, y_lo, y_hi, |cx, cz| {
        // 예산은 **호출자 소유**다(질의 단위). 페어마다 새로 4096이면 여러 바디를 도는
        // 질의의 총 작업량이 유계가 아니다.
        if *budget == 0 {
            over_budget = true;
            return false;
        }
        *budget -= 1;
        visited += 1;
        if visited > crate::MAX_HEIGHTFIELD_CELLS_PER_QUERY {
            over_budget = true;
            return false;
        }
        {
            for t in 0..2u8 {
                let tri_local = hf.triangle(cx, cz, t);
                let tri_shape = Shape::Triangle {
                    points: tri_local,
                    radius: 0.0,
                };
                let mask = hf.weld_mask(cx, cz, t);
                let mut emit = |ci: u8, cs: &Shape, cp: &Pose| -> bool {
                    let Some(mut m) = contact(cs, cp, &tri_shape, hf_pose, margin) else {
                        return false;
                    };
                    if m.is_empty() {
                        return false;
                    }
                    // 내부 에지 억제 — 월드 좌표 삼각형이 필요하다.
                    if mask != 0 {
                        let tw = [
                            hf_pose.pos + hf_pose.rot.rotate(tri_local[0]),
                            hf_pose.pos + hf_pose.rot.rotate(tri_local[1]),
                            hf_pose.pos + hf_pose.rot.rotate(tri_local[2]),
                        ];
                        // 밴드 폭 = **접촉 허용오차 + 침투 깊이**.
                        //
                        // 하이트필드 삼각형은 두께가 0이라 radius를 밴드로 쓸 수 없다(지형
                        // 등록부는 삼각형 radius가 그 역할을 했다). 그리고 접점은 에지에서
                        // 정확히 0만큼 떨어져 있지 않다 — 깊이 파고들수록 최근접점이 에지에서
                        // 멀어진다(실측: 18cm 침투에서 5cm). 침투를 밴드에 넣지 않으면
                        // **정작 유령 에지가 세게 차는 깊은 침투에서만 용접이 꺼진다**.
                        let pen = m.points().iter().fold(0.0f32, |a, p| a.max(-p.separation));
                        crate::weld_internal_edges(&mut m, tw, margin + pen.max(0.0), mask);
                    }
                    for i in 0..m.count as usize {
                        m.points[i].feature = FeatureId(FEATURE_GENERIC);
                    }
                    if flip {
                        m.flip();
                    }
                    if flip {
                        f(0, ci, m);
                    } else {
                        f(ci, 0, m);
                    }
                    true
                };
                let hit = if children.is_empty() {
                    emit(0, sa, pa)
                } else {
                    let mut any = false;
                    for (i, (cp, cs)) in children.iter().enumerate() {
                        let wp = Pose {
                            pos: pa.pos + pa.rot.rotate(cp.pos),
                            // 자식 회전 정규화 — 근거는 `cast.rs`의 compose 주석에.
                            rot: pa.rot * cp.rot.normalize(),
                        };
                        if emit(i as u8, cs, &wp) {
                            any = true;
                            emitted += 1;
                            if emitted >= crate::MAX_HEIGHTFIELD_MANIFOLDS {
                                saturated = true;
                                return false;
                            }
                        }
                    }
                    any
                };
                if children.is_empty() && hit {
                    emitted += 1;
                    if emitted >= crate::MAX_HEIGHTFIELD_MANIFOLDS {
                        saturated = true;
                        return false;
                    }
                }
            }
        }
        true
    });
    if over_budget {
        return ContactWalk::BudgetExceeded;
    }
    if saturated {
        return ContactWalk::Saturated;
    }
    ContactWalk::Complete
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
fn is_non_convex(s: &Shape) -> bool {
    matches!(s, Shape::Compound { .. } | Shape::Heightfield { .. })
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

/// 구-박스 해석 경로.
///
/// GJK+EPA를 돌리면 페어당 6.0us지만(실측) 해석해는 30 flops 남짓이다: 구 중심을 박스
/// 로컬로 옮기고 half-extent로 클램프하면 박스 코어 위의 최근접점이 나온다.
/// 정확도도 낫다 — 반복 수렴 오차가 없고 접점이 스텝 간 안정적이다.
///
/// `sphere_is_a`: 호출부의 A가 구면 true. 법선 부호 계약이 "A를 +normal로 밀면 분리"라
/// A가 어느 쪽인지 알아야 한다.
fn sphere_box(
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

/// 면 f = axis*2 + (양의 방향이면 1). 월드 법선.
/// 면 f의 바깥 법선 = 회전행렬 **열의 ±**. 회전을 새로 돌리지 않는다.
///
/// 예전 판은 `rot.rotate(±e_axis)`로 면마다 쿼터니언 회전을 돌렸다. 열의 부호를 뒤집는 것은
/// 그것과 **비트 동일이 아니다** — `rotate(-e_x)`의 0 성분은 `+0.0`인데 `-rotate(e_x)`는
/// `-0.0`이고, 그 차이는 내적이 통째로 0일 때 부호로 살아남는다. 여기서는 그 차이를 감수한다:
/// 클리퍼 전체가 이미 행렬-벡터 곱으로 옮겨져 값이 마지막 비트에서 달라지고, 그 대가는
/// 봉투 재검증으로 치렀다(README '매니폴드 생성' 절).
#[inline]
fn box_face_normal(f: usize, m: &Mat3) -> Vec3 {
    let c = m.c[f / 2];
    if f % 2 == 1 {
        c
    } else {
        Vec3::new(-c.x, -c.y, -c.z)
    }
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
/// **기각된 최적화: 회전 6번 → 3번 (2026-07-30 실측, 재시도 금지).**
///
/// 면 6개의 법선은 회전축 3개의 ±이므로 축 3개만 돌리고 짝수 면은 부호를 뒤집으면 될 것처럼
/// 보인다. 두 가지 이유로 하지 않는다.
///
/// ① **비트 동일이 아니다 — 0의 부호에서 깨진다.** `rotate`가 `v`에 선형인 것은 맞지만,
/// `rotate(-e_x)`는 0 성분이 `+0.0`이고 `-rotate(e_x)`는 `-0.0`이다. 그 차이는 내적이
/// 통째로 0일 때(예: `want = 0`) 결과의 부호로 살아남는다: 정본은 `+0.0`, 축 재사용판은
/// `-0.0`. 반환된 `best_dot`은 `da.max(db)`와 `da >= db`로 흘러 참조면 선택을 정하므로,
/// "±0은 어차피 같다"는 **논증이 필요한 주장**이지 자명한 사실이 아니다.
/// (이 함정은 직접 짠 대조 테스트가 잡았다 — 손으로 편 대수는 0의 부호를 놓친다.)
///
/// ② **값이 없다.** 같은 세션 교차 실행(`--bin boxsplit 40`, 8회씩)에서 페어당 median
/// 0.681 → 0.667 us(−2%)인데 min은 오히려 0.627 → 0.640으로 뒤집혔다 — 분포가 겹친다.
/// 매니폴드 생성의 2%는 스텝의 0.3%라, 위 ①의 논증을 지고 갈 값이 아니다.
///
/// 이 함수를 건드리려면 먼저 `boxsplit`으로 SAT/클리핑 비중을 다시 재라(2026-07-30 실측:
/// 페어당 0.67us 중 SAT 0.25 = 38%, 클리핑 0.41 = 62%). 남은 큰 덩어리는 여기가 아니라
/// **클리핑의 면 정점 회전**(페어마다 8회)이고, 그건 바디별 회전행렬을 스텝당 한 번 만들어
/// 내려보내는 **구조 변경**이라야 없어진다.
fn best_face(m: &Mat3, want: Vec3) -> (usize, f32) {
    let mut best = 0;
    let mut best_dot = -f32::INFINITY;
    for f in 0..6 {
        let d = box_face_normal(f, m).dot(want);
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
    char_len: f32,
    dup_len: f32,
    ha: Vec3,
    ra: f32,
    pa: &Pose,
    ma: &Mat3,
    hb: Vec3,
    rb: f32,
    pb: &Pose,
    mb: &Mat3,
    normal: Vec3,
    margin: f32,
) -> Option<Manifold> {
    let eps_area = area_eps(char_len);
    // A에서 B를 향하는 면(법선 ≈ −normal), B에서 A를 향하는 면(≈ +normal)
    let (fa, da) = best_face(ma, -normal);
    let (fb, db) = best_face(mb, normal);
    if da.max(db) < FACE_DOT_MIN {
        return None; // 에지-에지 지배 — witness 단일점 경로
    }
    // reference = 더 정렬된 쪽 (동률 = A — 결정론)
    let ref_is_a = da >= db;
    let (rf, rh, rr, rp, rm, ih, ir, ip, im) = if ref_is_a {
        (fa, ha, ra, pa, ma, hb, rb, pb, mb)
    } else {
        (fb, hb, rb, pb, mb, ha, ra, pa, ma)
    };

    let rn = box_face_normal(rf, rm); // reference 바깥 법선 (월드)
    let rverts: [Vec3; 4] = box_face_verts(rf, rh, rr).map(|v| rp.pos + rm.mul_vec(v));
    // incident = 상대 박스에서 rn과 가장 반대인 면
    let (inc_f, _) = best_face(im, -rn);
    let iverts: [Vec3; 4] = box_face_verts(inc_f, ih, ir).map(|v| ip.pos + im.mul_vec(v));
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

    // Sutherland–Hodgman: incident 폴리곤을 reference 4 측면 평면으로 클리핑 (계보 추적).
    // **힙 할당 없음** — 볼록 4각형을 반평면 4개로 자르면 정점은 단계마다 최대 +1이라
    // 상한이 8이다. 예전에는 여기서 페어당 Vec 6개(poly + 측면마다 out)를 할당해
    // 스텝당 수만 회의 allocator 왕복이 narrow의 상수를 먹었다.
    const CLIP_CAP: usize = 8;
    let mut poly = [ClipVertex {
        position: Vec3::ZERO,
        incident_mask: 0,
        reference_mask: 0,
    }; CLIP_CAP];
    let mut poly_len = 4usize;
    for (i, v) in iverts.iter().enumerate() {
        poly[i] = ClipVertex {
            position: *v,
            incident_mask: 1u8 << i,
            reference_mask: 0,
        };
    }
    // **측면마다 배열을 통째로 복사하지 않는다.** 예전에는 `poly = buf`로 8원소(=128B)를
    // 매 측면 복사했다(페어당 4회). 두 버퍼를 두고 **역할만 바꾸면** 복사가 사라진다.
    // 값·순서는 그대로다(같은 원소를 같은 순서로 읽고 쓴다).
    let mut buf = poly;
    let mut flip = false;
    // reference 면의 중심은 **측면 4개에 대해 같은 값**이다 — 루프 안에 두면 페어마다
    // 4번 다시 만든다(값은 같으므로 밖으로 빼도 비트 동일).
    let center = (rverts[0] + rverts[1] + rverts[2] + rverts[3]) * 0.25;
    for side in 0..4 {
        let e0 = rverts[side];
        let e1 = rverts[(side + 1) % 4];
        // 측면 평면: 법선 = rn × edge (안쪽을 향하도록)
        let mut pn = rn.cross(e1 - e0);
        if pn.dot(center - e0) < 0.0 {
            pn = -pn;
        }
        // src = 이번 측면의 입력 폴리곤, dst = 출력. 다음 측면에서 역할이 뒤바뀐다.
        let (src, dst) = if flip {
            (&buf, &mut poly)
        } else {
            (&poly, &mut buf)
        };
        let mut out_len = 0usize;
        // 볼록 입력에서 상한 초과는 수학적으로 불가 — 부동소수 퇴화 방어로만 잘라낸다
        // (wasm에서 인덱스 패닉 = trap이라, 불가능 케이스도 조용히 버리는 쪽이 계약이다).
        let push = |buf: &mut [ClipVertex; CLIP_CAP], len: &mut usize, v: ClipVertex| {
            if *len < CLIP_CAP {
                buf[*len] = v;
                *len += 1;
            } else {
                debug_assert!(false, "볼록 클리핑이 상한 8을 넘었다");
            }
        };
        for i in 0..poly_len {
            let v0 = src[i];
            let v1 = src[(i + 1) % poly_len];
            let d0 = pn.dot(v0.position - e0);
            let d1 = pn.dot(v1.position - e0);
            if d0 >= 0.0 {
                push(dst, &mut out_len, v0);
            }
            // 교점은 **엄격한 부호 반대**일 때만 생성 (확인 라운드 MAJOR 반영 — d0==0인 경계
            // 정점은 위에서 이미 보존되므로 t=0 교점까지 만들면 같은 위치가 서로 다른 계보
            // 마스크로 중복 진입해 솔버가 동일 지점에 제약을 중복 적용한다).
            //
            // **이 부호 규칙은 마지막 ulp에 기댄다** — 경계 정점이 정확히 `d == 0.0`일 때만
            // 막아 준다. 같은 기하라도 회전을 어떻게 계산했느냐에 따라 `±1e-9`가 되면 중복이
            // 부활한다. 그래서 아래 접점 수집에서 **위치 중복**을 최종 방어선으로 한 번 더 친다.
            if (d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0) {
                let t = d0 / (d0 - d1);
                let x = v0.position + (v1.position - v0.position) * t;
                push(
                    dst,
                    &mut out_len,
                    ClipVertex {
                        position: x,
                        incident_mask: v0.incident_mask | v1.incident_mask,
                        reference_mask: v0.reference_mask | v1.reference_mask | (1u8 << side),
                    },
                );
            }
        }
        flip = !flip;
        poly_len = out_len;
        if poly_len == 0 {
            return None;
        }
    }

    // reference 평면 아래(margin 이내)만 채택, separation = 평면까지 부호 거리
    // 마지막으로 **쓴** 버퍼가 결과다(교체 후이므로 `flip`이 가리키는 쪽).
    let poly = if flip { &buf } else { &poly };
    let mut pts = [ContactPoint::default(); CLIP_CAP];
    let mut pts_len = 0usize;
    // **같은 자리에 접점이 둘 들어가면 솔버가 그 지점에 제약을 두 번 건다.** 위 부호 규칙은
    // 값이 정확히 0일 때만 막으므로, 서로 다른 측면에서 생긴 교점이 같은 코너로 모이는 경우
    // (45° 박스의 코너가 참조면 코너와 만나는 배치)는 통과한다 — 실제로 회전 계산을 바꾸자
    // 6e-8 떨어진 접점 두 개가 나왔다.
    //
    // 임계는 절대 상수가 아니라 형상에서 유도하되, **접촉 국소 척도**(`min_feature_size`)를
    // 쓴다. AABB 대각(`characteristic_length`)을 쓰면 고종횡비 형상에서 임계가 접촉 패치보다
    // 수천 배 커진다 — 1000×0.001×1000 판 두 장의 1mm 패치에서 임계가 2.8mm가 되어 네 모서리가
    // 전부 한 점으로 합쳐지고 면 접촉이 점 접촉으로 붕괴한다(회전 저항 상실).
    // `tolerance.rs`가 바로 이 목적으로 두 척도를 나눠 두었고 주석도 "접촉 국소 판정에는
    // 이 값을 쓴다"고 적어 두었는데 첫 구현이 그것을 어겼다(교차검수 rev_7a05abc1 BLOCKER,
    // `thin_large_boxes_keep_distinct_contact_patch_points`가 잠근다).
    let dup_eps_sq = length_sq_eps(dup_len);
    for v in poly.iter().take(poly_len) {
        let sep = rn.dot(v.position - rverts[0]);
        if sep <= margin {
            // 접점 위치는 reference 면 위로 투영한 점과 incident 점의 중간
            let proj = v.position - rn * sep;
            let feature = ((ref_is_a as u32) << 31)
                | ((rf as u32) << 28)
                | ((inc_f as u32) << 25)
                | ((v.reference_mask as u32) << 4)
                | v.incident_mask as u32;
            let position = (v.position + proj) * 0.5;
            if pts[..pts_len]
                .iter()
                .any(|q| (q.position - position).length_sq() <= dup_eps_sq)
            {
                continue; // 이미 같은 자리에 접점이 있다
            }
            pts[pts_len] = ContactPoint {
                position,
                separation: sep,
                feature: FeatureId(feature),
            };
            pts_len += 1;
        }
    }
    if pts_len == 0 {
        return None;
    }

    // 4점 축소 (설계 §4.2): 최심점 고정 + 나머지는 면적 최대 조합(그리디, 결정적)
    let reduced;
    let final_pts: &[ContactPoint] = if pts_len > 4 {
        reduced = reduce_to_4(&pts[..pts_len], rn);
        &reduced
    } else {
        &pts[..pts_len]
    };
    // manifold 법선은 push-A 관례로: reference가 A면 rn은 A→B → 뒤집기
    let m_normal = if ref_is_a { -rn } else { rn };
    Some(Manifold::from_points(m_normal, final_pts))
}

/// 최심점 고정 → 나머지 3점 전수 조합으로 접촉 사각형 면적 최대 선택 (설계 §4.2 원문,
/// 리뷰 MAJOR 반영 — 구 그리디(최원점 선고정)는 면적 최대 조합을 보장하지 않았다).
/// pts는 클리핑 다각형의 순환 순서 — 인덱스 정렬로 부분집합의 순환 순서를 보존해 면적 계산.
fn reduce_to_4(pts: &[ContactPoint], n: Vec3) -> [ContactPoint; 4] {
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
    // 입력 상한 8 (클리핑 상한) — 후보는 최대 7. 고정 배열로 할당 제거.
    let mut candidates = [0usize; 8];
    let mut cand_len = 0usize;
    for i in 0..pts.len() {
        if i != deepest {
            candidates[cand_len] = i;
            cand_len += 1;
        }
    }
    let candidates = &candidates[..cand_len];
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

    let mut out = [
        pts[best_indices[0]],
        pts[best_indices[1]],
        pts[best_indices[2]],
        pts[best_indices[3]],
    ];
    out.sort_by_key(|p| p.feature);
    out
}

/// **지형 내부 에지 용접** (§4.2 ghost edge, Jolt ActiveEdges 동형).
///
/// 삼각형 지형은 삼각형마다 독립 바디라, 이웃과 **공면으로 공유하는 에지**(내부 에지)의
/// 둥근 밴드에서 접촉이 생기면 법선이 면이 아니라 에지에서 나온다 — 평평한 바닥을
/// 구르는 구가 이음새마다 옆으로 차이는 유령 충돌. 등록부(try_add_terrain)가 인접성 +
/// 공면성으로 내부 에지를 판정해 마스크를 주면, 여기서 그 에지 근방 접점의 법선을
/// 면 법선으로 스냅한다.
///
/// 보정 규칙(전부 보수적 — 확신 없으면 무변경):
/// - 법선이 이미 면 법선 원뿔 안(±5°)이면 무변경 (면 접촉은 원래 맞다).
/// - 접점 중 하나라도 내부 에지 세그먼트에서 `radius + tol` 안이어야 스냅한다 —
///   외부 에지·꼭짓점 접촉은 그대로 (경계에서는 에지 법선이 물리적으로 옳다).
/// - 면 법선 부호는 현재 법선 쪽으로 정렬한다(지형은 양면 — A가 어느 쪽이든 성립).
/// - separation은 재계산하지 않는다: 오차는 O(θ·침투)이고 소프트 위치 보정이 흡수한다.
///   (Jolt도 거리 유지 — 방향만 고친다.)
///
/// 볼록 능선·오목 골짜기는 등록부가 마스크를 세우지 않으므로 여기 오지 않는다.
pub fn weld_internal_edges(m: &mut Manifold, tri: [Vec3; 3], radius: f32, internal: u8) {
    if internal == 0 || m.is_empty() {
        return;
    }
    let n_face = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
    let scale = (tri[1] - tri[0])
        .length()
        .max((tri[2] - tri[0]).length())
        .max((tri[2] - tri[1]).length());
    // cross 길이는 L² — 면적 차원 임계로 퇴화 방지(등록부가 이미 거르지만 방어).
    let len = n_face.length();
    if len <= crate::tolerance::area_eps(scale) {
        return;
    }
    let n_face = n_face * (1.0 / len);
    let n_signed = if n_face.dot(m.normal) >= 0.0 {
        n_face
    } else {
        -n_face
    };
    // 5° 원뿔 — Jolt ActiveEdges 기본 임계와 동급. 면 접촉은 여기서 끝난다.
    const COS_FACE: f32 = 0.996;
    if m.normal.dot(n_signed) >= COS_FACE {
        return;
    }
    let band = radius + crate::tolerance::length_eps(scale);
    let near_internal = m.points().iter().any(|p| {
        (0..3u8).any(|i| {
            internal & (1 << i) != 0
                && dist_point_segment(p.position, tri[i as usize], tri[((i + 1) % 3) as usize])
                    <= band
        })
    });
    if near_internal {
        m.normal = n_signed;
    }
}

/// 점-세그먼트 거리 (용접 밴드 판정 전용).
fn dist_point_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let denom = ab.length_sq();
    // 퇴화 세그먼트(등록부가 거르지만 방어): 점 거리로.
    if denom <= 0.0 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / denom).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

#[cfg(test)]
mod weld_tests {
    use super::*;
    use maru_math::Quat;

    // 바닥 평면의 삼각형: (−1,0,−1)·(1,0,−1)·(0,0,1), 에지 0 = 앞쪽 (−1..1, z=−1).
    fn tri() -> [Vec3; 3] {
        [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
    }

    fn edge_manifold(tilt: Vec3) -> Manifold {
        let mut m = Manifold {
            normal: (Vec3::new(0.0, 1.0, 0.0) + tilt).normalize_or_zero(),
            ..Default::default()
        };
        m.push(ContactPoint {
            position: Vec3::new(0.0, 0.0, -1.0), // 에지 0 위
            separation: -0.01,
            feature: FeatureId(3),
        });
        m
    }

    /// 내부 에지 근방의 기운 법선은 면 법선으로 스냅된다.
    #[test]
    fn tilted_normal_on_internal_edge_snaps_to_face() {
        let mut m = edge_manifold(Vec3::new(0.0, 0.0, -0.6)); // 에지 0 밖으로 기움
        weld_internal_edges(&mut m, tri(), 0.05, 0b001);
        assert!((m.normal - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
    }

    /// 아래에서 온 접촉은 아래쪽 면 법선으로 — 부호는 현재 법선 쪽으로 정렬된다.
    #[test]
    fn snap_keeps_the_side_the_contact_came_from() {
        let mut m = edge_manifold(Vec3::new(0.0, 0.0, -0.6));
        m.normal = Vec3::new(0.0, -1.0, 0.3).normalize_or_zero();
        weld_internal_edges(&mut m, tri(), 0.05, 0b001);
        assert!((m.normal - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-6);
    }

    /// 마스크가 0이거나(용접 없음) 접점이 외부 에지에 있으면 무변경.
    #[test]
    fn external_edges_and_unmasked_triangles_are_untouched() {
        let before = edge_manifold(Vec3::new(0.0, 0.0, -0.6));
        let mut m = before;
        weld_internal_edges(&mut m, tri(), 0.05, 0);
        assert_eq!(m.normal, before.normal);
        // 에지 1·2만 내부라고 표시 — 접점은 에지 0 위라 밴드 밖.
        let mut m = before;
        weld_internal_edges(&mut m, tri(), 0.05, 0b110);
        assert_eq!(m.normal, before.normal);
    }

    /// 면 접촉(법선이 이미 원뿔 안)은 내부 에지 위 접점이라도 무변경 — 보정은 방향만.
    #[test]
    fn face_contacts_are_left_alone() {
        let mut m = edge_manifold(Vec3::ZERO);
        let before = m.normal;
        weld_internal_edges(&mut m, tri(), 0.05, 0b111);
        assert_eq!(m.normal, before);
    }

    /// 실제 협폭 경로로도 성립: 구가 내부 에지의 둥근 밴드에 접촉 → 기운 법선 → 용접.
    #[test]
    fn real_sphere_edge_contact_gets_welded() {
        let hull = Shape::Hull {
            points: tri().to_vec(),
            radius: 0.05,
        };
        let sphere = Shape::Sphere { radius: 0.3 };
        let id = Pose {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        };
        // 에지 0(z=−1) 바로 바깥 위 — 에지 밴드 접촉이라 법선이 −z로 기운다.
        let pa = Pose {
            pos: Vec3::new(0.0, 0.32, -1.12),
            rot: Quat::IDENTITY,
        };
        let m0 = contact(&sphere, &pa, &hull, &id, 0.1).expect("접촉이 있어야 한다");
        assert!(
            m0.normal.z < -0.2,
            "전제: 에지 접촉 법선이 기울어야 한다 (n={:?})",
            m0.normal
        );
        let mut m = m0;
        weld_internal_edges(&mut m, tri(), 0.05, 0b001);
        assert!(
            (m.normal - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6,
            "용접 후 면 법선이어야 한다 (n={:?})",
            m.normal
        );
    }
}

#[cfg(test)]
mod heightfield_contact_tests {
    use super::*;
    use maru_math::Quat;

    fn pose(x: f32, y: f32, z: f32) -> Pose {
        Pose {
            pos: Vec3::new(x, y, z),
            rot: Quat::IDENTITY,
        }
    }

    /// **내부 에지 억제의 직접 증거**: 셀 경계 바로 위에 놓인 구의 접촉 법선이
    /// 에지 법선이 아니라 **면 법선**이어야 한다.
    ///
    /// 완만한 경사(≈4°, 용접 원뿔 안)를 주면 에지에서의 최근접 방향이 두 면 법선 어느
    /// 쪽과도 다르다 — 용접이 없으면 구가 이음매를 지날 때마다 그 방향으로 차인다.
    /// 용접을 끄면(mask=0) 이 테스트가 실패하는 것으로 기제를 잠근다.
    #[test]
    fn welding_snaps_seam_contact_normal_to_the_face() {
        // z 방향으로 셀마다 7cm 오르는 격자(1m 셀 → ≈4°)
        let (nx, nz) = (4u32, 4u32);
        let mut heights = vec![0.0f32; (nx * nz) as usize];
        for iz in 0..nz {
            for ix in 0..nx {
                heights[(iz * nx + ix) as usize] = iz as f32 * 0.07;
            }
        }
        let hf_shape = Shape::heightfield(heights, nx, nz, 1.0).expect("격자");
        let hf = hf_shape.heightfield_ref().unwrap();
        // 이음매(z=2)를 **넘어선 쪽**에 중심을 두고 살짝 파고들게 놓는다.
        // 이러면 뒤 셀(1,1) 삼각형에서 최근접 특징이 **공유 에지**가 되고, 에지에서
        // 중심으로 향하는 법선은 면 법선과 크게 어긋난다 — 용접이 잡아야 하는 바로 그 경우다.
        // (구를 에지 바로 위에 얹으면 에지 법선이 이미 면 법선의 5° 원뿔 안이라 아무 일도
        //  일어나지 않는다 — 그건 용접이 필요 없는 상황이다.)
        let surf = hf.height_at(1.5, 2.15).unwrap();
        let sphere = Shape::Sphere { radius: 0.3 };
        let sp = pose(1.5, surf + 0.12, 2.15);

        let mut normals = Vec::new();
        for_each_contact(
            &sphere,
            &sp,
            &hf_shape,
            &pose(0.0, 0.0, 0.0),
            0.02,
            &mut |_, _, m| {
                if !m.is_empty() {
                    normals.push(m.normal);
                }
            },
        );
        assert!(
            !normals.is_empty(),
            "이음매에서 접촉이 없다 — 테스트가 공회전"
        );
        // 면 법선(두 이웃 삼각형 중 하나) 과 거의 같아야 한다.
        let faces: Vec<Vec3> = [(1u32, 1u32, 1u8), (1, 2, 0)]
            .iter()
            .map(|&(cx, cz, t)| hf.triangle_normal(cx, cz, t).unwrap())
            .collect();
        for n in &normals {
            let best = faces.iter().map(|f| f.dot(*n).abs()).fold(0.0f32, f32::max);
            assert!(
                best > 0.999,
                "이음매 법선이 면 법선과 다르다(용접 실패): n={n:?} faces={faces:?}"
            );
        }
    }
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

    /// 고종횡비 형상의 **국소 접촉 패치**가 중복 판정에 삼켜지지 않는지.
    ///
    /// 교차검수 rev_7a05abc1 BLOCKER: 중복 임계를 `characteristic_length`(작은 쪽 AABB 대각)로
    /// 잡으면 얇고 큰 판에서 임계가 접촉 패치보다 수천 배 커진다. 아래 판 두 장의 겹침은
    /// 약 1mm×1mm인데 대각은 2.8km라 임계가 2.8mm — 네 모서리가 전부 "같은 점"이 되어
    /// 면 접촉이 점 접촉으로 붕괴하고 회전 저항이 사라진다.
    #[test]
    fn thin_large_boxes_keep_distinct_contact_patch_points() {
        let plate = Shape::Box {
            half: Vec3::new(1000.0, 0.001, 1000.0),
            radius: 0.0,
        };
        let m = contact(
            &plate,
            &pose(0.0, 0.0, 0.0),
            &plate,
            &pose(1999.999, 0.0019, 1999.999),
            0.001,
        )
        .expect("1mm×1mm 면 접촉이 존재한다");
        assert_eq!(
            m.points().len(),
            4,
            "중복 임계가 얇은 접촉 패치를 합쳤다: {:?}",
            m.points()
        );
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
            let (ma, mb) = (Mat3::from_quat(pa.rot), Mat3::from_quat(pb.rot));
            let sat = match box_box_axis(half, 0.0, &pa, &ma, half, 0.0, &pb, &mb, 0.05) {
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

#[cfg(test)]
mod sphere_box_tests {
    use super::*;

    /// 고정 시드 xorshift — 재현 가능한 무작위 배치.
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn sym(&mut self, s: f32) -> f32 {
            (self.f() - 0.5) * 2.0 * s
        }
        fn quat(&mut self) -> Quat {
            Quat::from_axis_angle(
                Vec3::new(self.sym(1.0), self.sym(1.0), self.sym(1.0)),
                self.sym(3.2),
            )
        }
    }

    /// 독립 계산과의 대조. GJK를 기준으로 쓰지 않는 이유는 실측 때문이다 — 중심이
    /// 박스 밖 0.05에 있는 배치에서 `gjk`가 `Separated { distance: 0 }`을 돌려줬다.
    /// 그래서 기준을 **회전축 내적으로 구한 점-OBB 거리**로 둔다. 해석 경로가 쓰는
    /// 쿼터니언 켤레 변환과 다른 경로라, 켤레 방향을 뒤집는 류의 오류를 잡는다.
    #[test]
    fn sphere_box_matches_independent_point_obb_distance() {
        let mut rng = Rng(0x5B07);
        let mut checked = 0;
        for _ in 0..12000 {
            let half = Vec3::new(
                0.2 + rng.f() * 2.0,
                0.2 + rng.f() * 2.0,
                0.2 + rng.f() * 2.0,
            );
            let rs = 0.1 + rng.f() * 1.5;
            let pb = Pose {
                pos: Vec3::new(rng.sym(2.0), rng.sym(2.0), rng.sym(2.0)),
                rot: rng.quat(),
            };
            let cs = Vec3::new(rng.sym(6.0), rng.sym(6.0), rng.sym(6.0));

            // 독립 기준: 박스의 월드 축 3개에 내적해 로컬 좌표를 얻고 클램프한다.
            let ax = pb.rot.rotate(Vec3::new(1.0, 0.0, 0.0));
            let ay = pb.rot.rotate(Vec3::new(0.0, 1.0, 0.0));
            let az = pb.rot.rotate(Vec3::new(0.0, 0.0, 1.0));
            let dv = cs - pb.pos;
            let l = Vec3::new(dv.dot(ax), dv.dot(ay), dv.dot(az));
            let cl = Vec3::new(
                l.x.clamp(-half.x, half.x),
                l.y.clamp(-half.y, half.y),
                l.z.clamp(-half.z, half.z),
            );
            let outside = (l - cl).length();
            let inside_depth = if outside > 0.0 {
                0.0
            } else {
                (half.x - l.x.abs())
                    .min(half.y - l.y.abs())
                    .min(half.z - l.z.abs())
            };
            let want_sep = if outside > 0.0 {
                outside - rs
            } else {
                -(inside_depth + rs)
            };
            if want_sep > 1.0 {
                continue; // margin 밖 — 매니폴드가 없는 게 정답이다
            }
            let m = sphere_box(rs, cs, half, 0.0, &pb, 1.0, true)
                .expect("margin 1.0 안이면 매니폴드가 나와야 한다");
            let got = m.points()[0].separation;
            let scale = 1.0 + half.x.max(half.y).max(half.z) + rs;
            assert!(
                (got - want_sep).abs() < 1e-4 * scale,
                "separation 불일치: 해석 {got} vs 독립 {want_sep}  half={half:?} rs={rs}"
            );
            assert!(
                (m.normal.length() - 1.0).abs() < 1e-4,
                "법선이 단위벡터가 아니다"
            );
            // 법선은 박스 → 구 방향이어야 한다 (A=구를 +normal로 밀면 분리)
            if outside > 1e-3 {
                let e = l - cl;
                let want_n = (ax * e.x + ay * e.y + az * e.z) * (1.0 / outside);
                assert!(
                    m.normal.dot(want_n) > 0.999,
                    "법선 방향 불일치: dot={}",
                    m.normal.dot(want_n)
                );
            }
            checked += 1;
        }
        assert!(checked > 1000, "표본이 너무 적다: {checked}");
    }

    /// **GJK 거리 과소평가 회귀 (2026-07-26 수정 완료).**
    ///
    /// 이 배치에서 `gjk`가 `Separated { distance: 0 }`을 돌려줬다 — 구 중심이 박스 코어
    /// 밖 0.0516에 있는데도. 원인은 둘이었다: ① 진전 판정 임계가 f32 상쇄 오차보다
    /// 엄격해 종료하지 못하고 같은 support를 반복 삽입 ② 그렇게 생긴 영부피 사면체의
    /// 복구 경로가 직전 최근접점을 버리고 거리 0을 **강제**(`Vec3::ZERO` 전달).
    /// 이제 gjk도 독립 점-OBB 계산과 일치해야 한다 — 회귀가 다시 열리면 여기서 잡힌다.
    #[test]
    fn gjk_core_distance_matches_independent_calculation() {
        let half = Vec3::new(0.4904285, 2.0158985, 1.071742);
        let rs = 1.4453597f32;
        let pb = Pose {
            pos: Vec3::new(1.4413323, -1.2419696, 1.250838),
            rot: Quat {
                x: -0.047053058,
                y: -0.12782943,
                z: 0.15615694,
                w: 0.9782948,
            },
        };
        let cs = Vec3::new(1.368325, 0.080718756, 2.0537288);

        // 독립 계산 (회전축 내적)
        let ax = pb.rot.rotate(Vec3::new(1.0, 0.0, 0.0));
        let ay = pb.rot.rotate(Vec3::new(0.0, 1.0, 0.0));
        let az = pb.rot.rotate(Vec3::new(0.0, 0.0, 1.0));
        let dv = cs - pb.pos;
        let l = Vec3::new(dv.dot(ax), dv.dot(ay), dv.dot(az));
        let cl = Vec3::new(
            l.x.clamp(-half.x, half.x),
            l.y.clamp(-half.y, half.y),
            l.z.clamp(-half.z, half.z),
        );
        let core_dist = (l - cl).length();
        assert!(
            core_dist > 0.04,
            "이 배치의 코어 거리는 0이 아니다: {core_dist}"
        );

        let m = sphere_box(rs, cs, half, 0.0, &pb, 1.0, true).unwrap();
        assert!(
            (m.points()[0].separation - (core_dist - rs)).abs() < 1e-4,
            "해석 경로는 독립 계산과 일치해야 한다"
        );

        // gjk는 여기서 0을 준다 — 고쳐지면 이 단언이 깨져 알려준다.
        let ss = Shape::Sphere { radius: rs };
        let sbx = Shape::Box { half, radius: 0.0 };
        let ps = Pose {
            pos: cs,
            rot: Quat::IDENTITY,
        };
        let GjkResult::Separated { distance, .. } = gjk(&ss, &ps, &sbx, &pb) else {
            panic!("코어가 분리돼 있으므로 Separated여야 한다");
        };
        assert!(
            (distance - core_dist).abs() < 1e-4,
            "gjk 코어 거리 {distance} 가 독립 계산 {core_dist} 와 다르다"
        );
    }

    /// 부호·크기 자기일관성: 구를 법선 반대로 |sep|만큼 밀면 접촉이 정확히 풀려야 한다.
    /// 관통(중심이 코어 안쪽인 경우 포함) 구간까지 덮는다.
    #[test]
    fn sphere_box_separation_resolves_overlap() {
        let mut rng = Rng(0xB07E5);
        let mut deep = 0;
        for _ in 0..4000 {
            let half = Vec3::new(
                0.3 + rng.f() * 1.5,
                0.3 + rng.f() * 1.5,
                0.3 + rng.f() * 1.5,
            );
            let rs = 0.1 + rng.f() * 1.0;
            let pb = Pose {
                pos: Vec3::new(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0)),
                rot: rng.quat(),
            };
            // 박스 근처에 구를 촘촘히 뿌려 관통·내부 케이스를 포함시킨다
            let cs = pb.pos + Vec3::new(rng.sym(2.0), rng.sym(2.0), rng.sym(2.0));
            let Some(m) = sphere_box(rs, cs, half, 0.0, &pb, 0.0, true) else {
                continue;
            };
            let sep = m.points()[0].separation;
            if sep >= 0.0 {
                continue;
            }
            deep += 1;
            // A(구)를 +normal로 |sep|만큼 밀면 접촉이 풀린다 = separation ≈ 0
            let moved = cs + m.normal * (-sep);
            let after = sphere_box(rs, moved, half, 0.0, &pb, 1.0, true)
                .expect("밀어낸 뒤에도 margin 1.0 안이다");
            let s2 = after.points()[0].separation;
            assert!(
                s2.abs() < 3e-3 * (1.0 + half.x.max(half.y).max(half.z)),
                "밀어낸 뒤 separation이 0이 아니다: {s2} (원래 {sep})"
            );
        }
        assert!(deep > 200, "관통 표본이 너무 적다: {deep}");
    }

    /// 인자 순서 대칭: (구, 박스)와 (박스, 구)는 같은 접촉을 **법선만 뒤집어** 낸다.
    #[test]
    fn sphere_box_argument_order_is_symmetric() {
        let mut rng = Rng(0x5133);
        for _ in 0..2000 {
            let half = Vec3::new(0.3 + rng.f(), 0.3 + rng.f(), 0.3 + rng.f());
            let rs = 0.2 + rng.f();
            let ss = Shape::Sphere { radius: rs };
            let sbx = Shape::Box { half, radius: 0.0 };
            let pb = Pose {
                pos: Vec3::ZERO,
                rot: rng.quat(),
            };
            let ps = Pose {
                pos: Vec3::new(rng.sym(3.0), rng.sym(3.0), rng.sym(3.0)),
                rot: Quat::IDENTITY,
            };
            let a = contact(&ss, &ps, &sbx, &pb, 0.05);
            let b = contact(&sbx, &pb, &ss, &ps, 0.05);
            match (a, b) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert!(
                        (a.normal + b.normal).length() < 1e-5,
                        "순서를 바꾸면 법선이 정확히 반대여야 한다"
                    );
                    assert!(
                        (a.points()[0].separation - b.points()[0].separation).abs() < 1e-5,
                        "separation은 순서에 무관해야 한다"
                    );
                    assert!(
                        (a.points()[0].position - b.points()[0].position).length() < 1e-5,
                        "접점 위치는 순서에 무관해야 한다"
                    );
                }
                _ => panic!("한쪽만 접촉을 냈다"),
            }
        }
    }
}
