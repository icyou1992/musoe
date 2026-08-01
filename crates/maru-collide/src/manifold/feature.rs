//! 일반 볼록 면 클리퍼의 feature ID 코덱 (설계 §4.2 확장, 검수 rev_ae6d97b1 확정안).
//!
//! 박스-박스 전용 클리퍼(`clip.rs`)의 인코딩은 **정점 4개·측면 4개를 비트마스크로** 담아
//! 일반 볼록에 못 쓴다(hull 면은 정점이 최대 32개). 여기서 쓰는 배치는 계보를
//! **인덱스**로 담는다:
//!
//! ```text
//! 31      : ref_is_a      — reference 면이 A 소속인가
//! 25..30  : ref_face  (6) — reference 면 코드
//! 19..24  : inc_face  (6) — incident 면 코드
//! 17..18  : kind      (2) — 접점 계보 종류
//! 16      : ab 비트 9     ┐ 논리 필드 ab = a | (b << 5) 를
//! 9..15   : (자식 네임스페이스 — 여기서 절대 쓰지 않는다)
//! 0..8    : ab 비트 0..8  ┘ 두 구간에 나눠 담는다
//! ```
//!
//! **비연속 배치인 이유**: 복합 자식 네임스페이스가 이미 비트 9..15를 점유한다
//! (`FEATURE_CHILD_SHIFT`). 남는 자리는 `0..8`(9비트)과 `16..31`(16비트)로 갈라져 있고
//! 필요한 폭은 정확히 25비트라 `a`·`b` 10비트가 연속 구간을 못 잡는다. 그래서 `a`와 `b`를
//! **하나의 논리 필드 `ab`**로 합친 뒤 9+1로 쪼갠다 — 쪼개는 곳이 이 파일 한 군데뿐이라
//! 호출자는 비연속을 볼 일이 없다.
//!
//! **센티널 예약**: 면 코드 60..63은 쓰지 않는다. `FEATURE_GENERIC`(0xFFFF_0000)과
//! `FEATURE_ANALYTIC`(0xFFFE_0000)이 이 배치에서 `ref_face = inc_face = 63`으로 읽히므로,
//! 실제 면 코드를 59 이하로 제한하면 센티널과 절대 겹치지 않는다. hull 면 개수 상한이
//! `2 * MAX_HULL_POINTS - 4 = 60`(인덱스 0..=59)이라 이 예약이 딱 들어맞는다 —
//! 여유가 없으므로 범위를 벗어나면 **조용히 자르지 않고 `None`을 준다**(호출자는
//! `FEATURE_GENERIC` 폴백).

use super::FeatureId;

/// 면 코드 상한(포함). 60..63은 센티널 예약.
pub const FACE_CODE_MAX: u32 = 59;
/// 정점·에지 인덱스 상한(포함). `MAX_HULL_POINTS = 32`에 맞춘 5비트.
pub const FEATURE_INDEX_MAX: u32 = 31;

const REF_IS_A_SHIFT: u32 = 31;
const REF_FACE_SHIFT: u32 = 25;
const INC_FACE_SHIFT: u32 = 19;
const KIND_SHIFT: u32 = 17;
const AB_HI_SHIFT: u32 = 16;
const FACE_MASK: u32 = 0x3F;
const KIND_MASK: u32 = 0x3;
/// `ab`의 하위 9비트가 사는 곳 (비트 0..8).
const AB_LO_MASK: u32 = 0x1FF;

/// kind 값 0은 **이 코덱이 아니라는 표식**으로 예약한다 (검수 rev_65fca6a1 MAJOR).
///
/// 판별자가 없으면 박스 전용 클립 ID가 새 디코더를 그냥 통과한다 — 예컨대 박스가 만든
/// `0x0000_0011`은 `IncidentVertex { a: 17 }`로 "성공"해서, 뒤에 오는 코드가
/// `unpack(id).is_some()`으로 계보 유무를 물으면 박스 접점을 일반면 계보로 오인한다.
/// 박스 인코딩은 비트 31·28..30·25..27·0..7만 쓰므로 **kind 비트(17..18)가 항상 0**이다.
/// 그래서 0을 비워 두는 것만으로 완전한 판별자가 된다.
const KIND_NOT_THIS_CODEC: u32 = 0;

/// 클리핑 접점의 계보 종류.
///
/// 순번이나 면 쌍만 남기면 서로 다른 접점이 같은 ID로 충돌해 warm start가 끊긴다
/// (검수 rev_ae6d97b1) — 그래서 "무엇에서 나온 점인가"를 명시한다.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClipKind {
    /// incident 면의 원본 정점 `a`가 그대로 살아남았다 (`b`는 반드시 0).
    IncidentVertex = 1,
    /// incident 에지 `a`가 reference 측면 `b`와 만나 생긴 교점.
    EdgeCross = 2,
    /// reference 면의 원본 정점 `a`가 incident 다각형 안에 들었다 (`b`는 반드시 0).
    ReferenceVertex = 3,
}

/// 어떤 종류도 예약값을 쓰지 않는다 — 누가 kind를 0으로 되돌리면 **컴파일이 깨진다**.
/// 주석으로만 적어 두면 판별자가 조용히 무력해진다.
const _: () = assert!(
    ClipKind::IncidentVertex as u32 != KIND_NOT_THIS_CODEC
        && ClipKind::EdgeCross as u32 != KIND_NOT_THIS_CODEC
        && ClipKind::ReferenceVertex as u32 != KIND_NOT_THIS_CODEC,
    "kind 값 0은 '이 코덱이 아님' 판별자로 예약돼 있다"
);

impl ClipKind {
    /// 이 종류가 `b`를 쓰는가. 안 쓰는 종류는 `b == 0`만 캐노니컬이다.
    fn uses_b(self) -> bool {
        matches!(self, Self::EdgeCross)
    }
    fn from_bits(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::IncidentVertex),
            2 => Some(Self::EdgeCross),
            3 => Some(Self::ReferenceVertex),
            _ => None, // KIND_NOT_THIS_CODEC 포함
        }
    }
}

/// 디코드된 면 접점 계보.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceFeature {
    pub ref_is_a: bool,
    pub ref_face: u32,
    pub inc_face: u32,
    pub kind: ClipKind,
    pub a: u32,
    pub b: u32,
}

/// 계보를 feature ID로 인코딩한다. 자식 네임스페이스 비트(9..15)는 항상 0으로 둔다 —
/// `Manifold::namespace_features`가 나중에 접는다.
///
/// 범위를 벗어나면 `None`. 자르거나 감싸지 않는다: 면 코드를 60 이상으로 자르면
/// 센티널과 같은 값이 되고, 인덱스를 감싸면 서로 다른 접점이 같은 ID를 갖는다.
/// 면 코드가 모자라는 셰이프가 생기면 여기서 `None`이 나므로 억지로 우겨넣지 말고
/// 별도 형식으로 분기해야 한다(검수 rev_65fca6a1).
///
/// **비에지 종류는 `b != 0`을 거절한다** — 같은 물리 계보가 `b` 쓰레기값 때문에 여러 ID를
/// 갖게 되면, 한 프레임은 `b=0`, 다음 프레임은 `b=1`이 되어 직전 임펄스를 못 찾는다
/// (검수 rev_65fca6a1 MAJOR). 코덱이 캐노니컬 표현을 강제한다.
pub fn pack(f: FaceFeature) -> Option<FeatureId> {
    if f.ref_face > FACE_CODE_MAX
        || f.inc_face > FACE_CODE_MAX
        || f.a > FEATURE_INDEX_MAX
        || f.b > FEATURE_INDEX_MAX
    {
        return None;
    }
    if !f.kind.uses_b() && f.b != 0 {
        return None;
    }
    let ab = f.a | (f.b << 5);
    Some(FeatureId(
        ((f.ref_is_a as u32) << REF_IS_A_SHIFT)
            | (f.ref_face << REF_FACE_SHIFT)
            | (f.inc_face << INC_FACE_SHIFT)
            | ((f.kind as u32) << KIND_SHIFT)
            | ((ab >> 9) << AB_HI_SHIFT)
            | (ab & AB_LO_MASK),
    ))
}

/// feature ID를 계보로 디코딩한다. 자식 네임스페이스 비트는 무시한다.
///
/// 이 코덱이 만든 값이 아니면 `None`이며, 그 판정은 세 겹이다:
/// **kind 비트 0**(박스 전용 인코딩 — 그쪽은 17..18을 절대 안 쓴다),
/// **예약 면 코드 60..63**(센티널 GENERIC/ANALYTIC), **비에지 종류의 `b != 0`**
/// (캐노니컬 아님). 셋 다 `pack()`이 만들 수 없는 값이므로 왕복이 유지된다.
pub fn unpack(id: FeatureId) -> Option<FaceFeature> {
    let v = id.0;
    let ref_face = (v >> REF_FACE_SHIFT) & FACE_MASK;
    let inc_face = (v >> INC_FACE_SHIFT) & FACE_MASK;
    if ref_face > FACE_CODE_MAX || inc_face > FACE_CODE_MAX {
        return None;
    }
    let kind = ClipKind::from_bits((v >> KIND_SHIFT) & KIND_MASK)?;
    let ab = (v & AB_LO_MASK) | (((v >> AB_HI_SHIFT) & 1) << 9);
    let b = (ab >> 5) & 0x1F;
    if !kind.uses_b() && b != 0 {
        return None;
    }
    Some(FaceFeature {
        ref_is_a: (v >> REF_IS_A_SHIFT) & 1 == 1,
        ref_face,
        inc_face,
        kind,
        a: ab & 0x1F,
        b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{FEATURE_ANALYTIC, FEATURE_CHILD_MASK, FEATURE_CHILD_SHIFT};
    use std::collections::HashSet;

    const KINDS: [ClipKind; 3] = [
        ClipKind::IncidentVertex,
        ClipKind::EdgeCross,
        ClipKind::ReferenceVertex,
    ];

    fn grid() -> Vec<FaceFeature> {
        let mut out = Vec::new();
        for ref_is_a in [false, true] {
            for ref_face in [0, 1, 7, 58, FACE_CODE_MAX] {
                for inc_face in [0, 3, 59] {
                    for kind in KINDS {
                        // 비에지 종류는 b=0만 캐노니컬이다 — 코덱이 그것을 강제한다.
                        let bs: &[u32] = if kind == ClipKind::EdgeCross {
                            &[0, 2, 16, FEATURE_INDEX_MAX]
                        } else {
                            &[0]
                        };
                        for a in [0, 1, 15, 16, FEATURE_INDEX_MAX] {
                            for b in bs {
                                out.push(FaceFeature {
                                    ref_is_a,
                                    ref_face,
                                    inc_face,
                                    kind,
                                    a,
                                    b: *b,
                                });
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn pack_unpack_roundtrips_over_the_full_range() {
        for f in grid() {
            let id = pack(f).expect("범위 안인데 거절됐다");
            assert_eq!(unpack(id), Some(f), "왕복이 깨졌다: {f:?} -> {id:?}");
        }
    }

    /// 서로 다른 계보가 같은 ID를 받으면 warm start가 엉뚱한 접점에 붙는다.
    #[test]
    fn distinct_lineages_never_share_an_id() {
        let g = grid();
        let ids: HashSet<u32> = g.iter().map(|f| pack(*f).unwrap().0).collect();
        assert_eq!(
            ids.len(),
            g.len(),
            "서로 다른 계보가 같은 feature ID를 받았다"
        );
    }

    /// 코덱은 자식 네임스페이스 비트를 절대 쓰지 않는다 — `namespace_features`의
    /// debug_assert가 이 불변식을 판정한다.
    #[test]
    fn packing_leaves_the_child_namespace_bits_clear() {
        for f in grid() {
            let id = pack(f).unwrap();
            assert_eq!(
                id.0 & FEATURE_CHILD_MASK,
                0,
                "네임스페이스 비트(9..15)를 침범했다: {f:?}"
            );
        }
    }

    /// 자식 키를 접은 뒤에도 계보가 그대로 읽혀야 복합 셰이프에서 warm start가 산다.
    #[test]
    fn child_namespace_key_does_not_disturb_the_lineage() {
        for f in grid() {
            let id = pack(f).unwrap();
            for key in [1u32, 37, 64] {
                let namespaced = FeatureId(id.0 | (key << FEATURE_CHILD_SHIFT));
                assert_eq!(unpack(namespaced), Some(f), "키 {key}가 계보를 흐렸다");
            }
        }
    }

    /// 센티널은 면 코드 63으로 읽혀 예약 구간에 들어간다 — 실제 면 접점과 겹칠 수 없다.
    #[test]
    fn sentinels_decode_into_the_reserved_face_range() {
        for s in [super::super::FEATURE_GENERIC, FEATURE_ANALYTIC] {
            assert_eq!(
                unpack(FeatureId(s)),
                None,
                "센티널이 면 접점으로 읽혔다: {s:#x}"
            );
            assert_eq!((s >> REF_FACE_SHIFT) & FACE_MASK, 63);
            assert_eq!((s >> INC_FACE_SHIFT) & FACE_MASK, 63);
        }
        // 자식 키를 접은 센티널도 마찬가지 (복합 폴백 경로가 만든다).
        for key in [1u32, 64] {
            let s = super::super::FEATURE_GENERIC | (key << FEATURE_CHILD_SHIFT);
            assert_eq!(unpack(FeatureId(s)), None);
        }
    }

    /// 예약 면 코드는 **kind가 정상이어도** 거절돼야 한다.
    ///
    /// 현재 센티널 둘은 kind 비트가 3(미지의 kind)이라 `from_bits`에서 먼저 걸린다 —
    /// 즉 위 테스트만으로는 면 코드 예약 가드가 관측되지 않는다(뮤테이션 실측: 가드를
    /// 지워도 전부 통과). 앞단을 통과하는 반례로 가드 자체를 겨눈다. 이 구간을 새 값이
    /// 열어 쓰면 실제 면 접점과 별칭이 되므로 계약이지 방어가 아니다.
    #[test]
    fn reserved_face_codes_are_rejected_even_with_a_valid_kind() {
        for face in 60u32..=63 {
            for kind in 1u32..=3 {
                let ref_reserved = (face << REF_FACE_SHIFT) | (kind << KIND_SHIFT) | 7;
                assert_eq!(
                    unpack(FeatureId(ref_reserved)),
                    None,
                    "예약 ref_face {face}(kind {kind})가 면 접점으로 읽혔다"
                );
                let inc_reserved =
                    (3 << REF_FACE_SHIFT) | (face << INC_FACE_SHIFT) | (kind << KIND_SHIFT);
                assert_eq!(
                    unpack(FeatureId(inc_reserved)),
                    None,
                    "예약 inc_face {face}(kind {kind})가 면 접점으로 읽혔다"
                );
            }
        }
        // 바로 안쪽(59)은 통과해야 한다 — 가드가 무조건 거절로 바뀌면 여기서 걸린다.
        let ok = (FACE_CODE_MAX << REF_FACE_SHIFT)
            | (FACE_CODE_MAX << INC_FACE_SHIFT)
            | ((ClipKind::IncidentVertex as u32) << KIND_SHIFT);
        assert!(
            unpack(FeatureId(ok)).is_some(),
            "경계 안쪽 면 코드가 거절됐다"
        );
    }

    /// **실제 박스-박스 접점**의 feature ID는 이 코덱으로 읽히면 안 된다.
    /// 생산자를 직접 돌려서 판정한다 — 손으로 만든 값은 실제 인코딩과 어긋날 수 있다.
    #[test]
    fn real_box_contact_features_are_not_read_as_face_lineage() {
        use crate::{Pose, Shape};
        use maru_math::{Quat, Vec3};
        let a = Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        };
        let mut seen = 0usize;
        for (dx, dy, angle) in [
            (0.0f32, 0.94f32, 0.0f32),
            (0.13, 0.95, 0.0),
            (0.0, 0.9, 0.37),
            (0.21, 0.88, 0.61),
        ] {
            let pa = Pose {
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            };
            let pb = Pose {
                pos: Vec3::new(dx, dy, 0.0),
                rot: Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), angle),
            };
            let m = crate::manifold::dispatch::contact(&a, &pa, &a, &pb, 0.02)
                .expect("박스 둘이 접촉해야 한다");
            for p in m.points() {
                seen += 1;
                assert_eq!(
                    unpack(p.feature),
                    None,
                    "박스 접점 {:#010x}이 일반면 계보로 읽혔다",
                    p.feature.0
                );
            }
        }
        assert!(seen >= 8, "박스 접점을 충분히 못 봤다: {seen}");
    }

    /// 비에지 종류에서 `b != 0`은 만들 수도, 읽을 수도 없어야 한다 —
    /// 같은 계보가 여러 ID를 갖게 되면 warm start가 프레임마다 끊긴다.
    #[test]
    fn non_edge_kinds_reject_a_nonzero_b_on_both_sides() {
        for kind in [ClipKind::IncidentVertex, ClipKind::ReferenceVertex] {
            let canonical = FaceFeature {
                ref_is_a: false,
                ref_face: 5,
                inc_face: 9,
                kind,
                a: 7,
                b: 0,
            };
            let id = pack(canonical).expect("캐노니컬이 거절됐다");
            assert_eq!(unpack(id), Some(canonical));
            for b in [1u32, 16, FEATURE_INDEX_MAX] {
                assert_eq!(
                    pack(FaceFeature { b, ..canonical }),
                    None,
                    "{kind:?}에서 b={b}가 통과했다"
                );
                // 디코드 쪽도 막는다: b 비트를 직접 세운 값은 이 코덱의 값이 아니다.
                // `ab`는 물리적으로 0..8 + 비트16에 흩어져 있으므로 그 배치대로 세운다
                // (`id.0 | (b << 5)`로 쓰면 자식 네임스페이스 비트를 건드린다).
                let ab = 7u32 | (b << 5);
                let forged = FeatureId(
                    (id.0 & !(AB_LO_MASK | (1 << AB_HI_SHIFT)))
                        | (ab & AB_LO_MASK)
                        | ((ab >> 9) << AB_HI_SHIFT),
                );
                assert_eq!(unpack(forged), None, "{kind:?}에서 위조 b={b}가 읽혔다");
            }
        }
        // EdgeCross는 b를 쓴다 — 무조건 거절로 바뀌면 여기서 걸린다.
        let edge = FaceFeature {
            ref_is_a: false,
            ref_face: 5,
            inc_face: 9,
            kind: ClipKind::EdgeCross,
            a: 7,
            b: 19,
        };
        assert_eq!(
            unpack(pack(edge).expect("에지 계보가 거절됐다")),
            Some(edge)
        );
    }

    /// 어떤 정상 계보도 센티널 값과 같아질 수 없다.
    #[test]
    fn no_packed_lineage_collides_with_a_sentinel() {
        for f in grid() {
            let id = pack(f).unwrap().0;
            for key in [0u32, 1, 64] {
                let n = id | (key << FEATURE_CHILD_SHIFT);
                assert_ne!(n, super::super::FEATURE_GENERIC, "{f:?}");
                assert_ne!(n, FEATURE_ANALYTIC, "{f:?}");
            }
        }
    }

    /// 경계 바로 안쪽은 통과하고 바로 밖은 거절돼야 한다 — 자르기(clamp)로 바뀌면
    /// 면 코드 60이 센티널 구간으로 새고, 인덱스는 감싸며 충돌한다.
    #[test]
    fn out_of_range_fields_are_rejected_not_clamped() {
        let base = FaceFeature {
            ref_is_a: true,
            ref_face: FACE_CODE_MAX,
            inc_face: FACE_CODE_MAX,
            kind: ClipKind::EdgeCross,
            a: FEATURE_INDEX_MAX,
            b: FEATURE_INDEX_MAX,
        };
        assert!(pack(base).is_some(), "경계값이 거절됐다");
        assert_eq!(
            pack(FaceFeature {
                ref_face: FACE_CODE_MAX + 1,
                ..base
            }),
            None
        );
        assert_eq!(
            pack(FaceFeature {
                inc_face: FACE_CODE_MAX + 1,
                ..base
            }),
            None
        );
        assert_eq!(
            pack(FaceFeature {
                a: FEATURE_INDEX_MAX + 1,
                ..base
            }),
            None
        );
        assert_eq!(
            pack(FaceFeature {
                b: FEATURE_INDEX_MAX + 1,
                ..base
            }),
            None
        );
    }

    /// hull 면 개수 상한이 예약 구간을 넘지 않는다 — 넘으면 마지막 면이 센티널이 된다.
    #[test]
    fn hull_face_cap_fits_below_the_reserved_codes() {
        let max_faces = 2 * crate::MAX_HULL_POINTS - 4;
        assert!(
            max_faces as u32 - 1 <= FACE_CODE_MAX,
            "hull 면 인덱스 최대 {}가 면 코드 상한 {FACE_CODE_MAX}를 넘는다",
            max_faces - 1
        );
    }
}
