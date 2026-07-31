//! 참조면이 정해진 뒤의 **접점 생성** — 면 클리핑과 4점 축약.
//!
//! Sutherland–Hodgman으로 입사면을 참조면 옆면들로 자르고, 남은 점을 4개로 줄인다.

use super::*;

/// 면 f = axis*2 + (양의 방향이면 1). 월드 법선.
/// 면 f의 바깥 법선 = 회전행렬 **열의 ±**. 회전을 새로 돌리지 않는다.
///
/// 예전 판은 `rot.rotate(±e_axis)`로 면마다 쿼터니언 회전을 돌렸다. 열의 부호를 뒤집는 것은
/// 그것과 **비트 동일이 아니다** — `rotate(-e_x)`의 0 성분은 `+0.0`인데 `-rotate(e_x)`는
/// `-0.0`이고, 그 차이는 내적이 통째로 0일 때 부호로 살아남는다. 여기서는 그 차이를 감수한다:
/// 클리퍼 전체가 이미 행렬-벡터 곱으로 옮겨져 값이 마지막 비트에서 달라지고, 그 대가는
/// 봉투 재검증으로 치렀다(README '매니폴드 생성' 절).
#[inline]
pub(super) fn box_face_normal(f: usize, m: &Mat3) -> Vec3 {
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
pub(super) fn box_face_verts(f: usize, half: Vec3, radius: f32) -> [Vec3; 4] {
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
pub(super) fn best_face(m: &Mat3, want: Vec3) -> (usize, f32) {
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
pub(super) fn clip_box_box(
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
pub(super) fn reduce_to_4(pts: &[ContactPoint], n: Vec3) -> [ContactPoint; 4] {
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
