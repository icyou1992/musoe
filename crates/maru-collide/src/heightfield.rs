//! 하이트필드 — 격자 높이맵 지형 (설계 §4.2 "메시·하이트필드, 내부 에지 억제").
//!
//! **왜 삼각형 수프가 아니라 전용 셰이프인가.** 이미 `World::try_add_terrain`이 삼각형마다
//! static 바디 하나를 만드는 경로를 갖고 있다. 그건 임의 메시에는 맞지만 격자에는 낭비다:
//! 256×256 격자면 바디 13만 개, 브로드페이즈 리프 13만 개, 스냅샷도 그만큼이다. 격자는
//! **암묵 구조**라 저장할 것이 높이 배열 하나뿐이고, 어떤 셀이 겹치는지는 나눗셈 두 번으로
//! 나온다 — 그래서 바디 하나로 두고 내로우페이즈가 필요한 셀만 펼친다.
//!
//! **내부 에지 억제가 공짜로 된다.** 삼각형 수프는 어떤 삼각형이 이웃인지 알아내려고
//! 에지 사전을 만들어야 했지만(월드의 지형 등록부), 격자에서는 이웃이 **좌표로 유도**된다.
//! 공면 판정 임계는 지형 등록부와 **같은 값**을 쓴다 — 두 경로가 다른 각도로 용접하면
//! 같은 지형을 두 방식으로 만들었을 때 물리가 달라진다.
//!
//! 로컬 좌표: `x = ix·cell`, `z = iz·cell`, 높이는 `heights[iz·nx + ix]`. 원점은 격자의
//! (0,0) 모서리다(중심이 아니라) — 호스트가 타일을 이어 붙일 때 오프셋 계산이 단순해진다.

use crate::{Aabb, Pose};
use maru_math::Vec3;

/// 한 페어가 만들 수 있는 접촉 매니폴드 상한.
///
/// 하이트필드는 바디 하나가 삼각형 수천 개를 품으므로, 상한이 없으면 한 페어가 스텝
/// 스크래치 예산을 통째로 먹는다(제약 회계가 `parts(a)×parts(b)`로 페어당 상한을 잡는데
/// 그 값이 곧 이 상수다). 넘치면 **뒤를 버리고** 진단으로 보고한다 — 조용한 누락이 아니라
/// 관측 가능한 포화다(CCD 히트 큐와 같은 규약).
pub const MAX_HEIGHTFIELD_MANIFOLDS: usize = 64;

/// 격자 샘플 수 상한 (nx·nz). 스냅샷·검증 예산의 근거.
pub const MAX_HEIGHTFIELD_SAMPLES: usize = 1 << 20;

/// 질의 한 번이 훑을 수 있는 셀 **예산**.
///
/// 넘으면 **자르지 않는다**(자르면 진짜 접촉을 놓쳐 관통한다) — 캐스트는 보수적으로
/// 이동을 막고, 공개 질의(`World::sweep_shape`)는 `ResourceLimit`으로 거절한다.
/// 1024×1024 격자를 통째로 덮는 스윕이 200만 회 캐스트가 되는 것을 이 값이 막는다.
///
/// (아래 옛 설명은 이 상수의 역사다 — 처음에는 절단에 썼고 그게 결함이었다.)
///
/// 인위적 상한은 긴 지형에서 진짜 히트를 조용히 미스로 만들었다(검수 rev_1e81559e BLOCKER:
/// 8194×2 격자를 가로지르는 캐스트가 5,000번째 셀의 접촉을 놓쳤다). 지금은 작업량을
/// **격자 자체**가 묶는다 — 레이 DDA는 한 방향 전진이라 ncx+ncz+2를 못 넘고, 선형 캐스트는
/// 스윕 봉투가 덮는 셀만 본다. 상수는 호스트가 격자 크기를 정할 때 쓰라고 남겨 둔다.
pub const MAX_HEIGHTFIELD_CELLS_PER_QUERY: usize = 4096;

/// 공면 판정 임계 — 지형 등록부(`World::try_add_terrain`)와 **같은 값**이어야 한다.
/// Jolt ActiveEdges 기본과 동급(~5.7°).
pub const COS_COPLANAR: f32 = 0.995;

/// 타일 한 변의 셀 수 — 높이 범위 계층의 해상도.
///
/// 8×8이면 1024×1024 격자의 타일이 128×128 = 16k개다(최소/최대 f32 두 개 = 128KB).
/// 이 정도면 캐시에 들어가고, 스윕 하나가 훑을 타일 수는 셀 수의 1/64이다.
pub const HEIGHTFIELD_TILE: u32 = 8;

/// 격자 하이트필드의 기하 — `Shape::Heightfield`의 계산을 모아 둔 뷰.
///
/// 셰이프가 소유한 데이터를 빌려 쓰기만 한다(복사 없음).
#[derive(Clone, Copy)]
pub struct HeightfieldRef<'a> {
    pub heights: &'a [f32],
    /// 타일별 (최소, 최대) 높이 — **질의가 셀을 안 보고 통째로 건너뛰는 근거**.
    ///
    /// 이게 없으면 총알이 평평한 통로를 50m 지나가는 것만으로 5,000셀을 검사해야 한다
    /// (확인검수 rev_d390e73d BLOCKER: 그 비용을 피하려고 넣은 "예산 초과 = 가짜 히트"가
    /// 정상 장면을 영구 동결시켰다). 타일의 높이 범위가 질의의 Y 밴드와 안 겹치면
    /// 그 타일의 셀 64개는 볼 필요가 **없다** — 근사가 아니라 정확한 가지치기다.
    pub tiles: &'a [(f32, f32)],
    pub nx: u32,
    pub nz: u32,
    pub cell: f32,
    pub radius: f32,
}

impl<'a> HeightfieldRef<'a> {
    #[inline]
    pub fn height(&self, ix: u32, iz: u32) -> f32 {
        self.heights[(iz * self.nx + ix) as usize]
    }

    /// 타일 격자 크기 (x, z).
    #[inline]
    pub fn tile_dims(&self) -> (u32, u32) {
        let (ncx, ncz) = self.cells();
        (
            ncx.div_ceil(HEIGHTFIELD_TILE),
            ncz.div_ceil(HEIGHTFIELD_TILE),
        )
    }

    /// 타일 `(tx, tz)`의 (최소, 최대) 높이. 범위 밖이면 빈 구간.
    #[inline]
    pub fn tile_range(&self, tx: u32, tz: u32) -> (f32, f32) {
        let (ntx, ntz) = self.tile_dims();
        if tx >= ntx || tz >= ntz {
            return (f32::INFINITY, f32::NEG_INFINITY);
        }
        let i = (tz * ntx + tx) as usize;
        self.tiles
            .get(i)
            .copied()
            .unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
    }

    /// 셀 개수 (x, z) — 정점 수보다 하나씩 적다.
    #[inline]
    pub fn cells(&self) -> (u32, u32) {
        (self.nx - 1, self.nz - 1)
    }

    #[inline]
    pub fn vertex(&self, ix: u32, iz: u32) -> Vec3 {
        Vec3::new(
            ix as f32 * self.cell,
            self.height(ix, iz),
            iz as f32 * self.cell,
        )
    }

    /// 로컬 AABB (겉면 = 코어 + radius).
    pub fn local_aabb(&self) -> Aabb {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &h in self.heights {
            lo = lo.min(h);
            hi = hi.max(h);
        }
        let r = self.radius;
        Aabb::new(
            Vec3::new(-r, lo - r, -r),
            Vec3::new(
                (self.nx - 1) as f32 * self.cell + r,
                hi + r,
                (self.nz - 1) as f32 * self.cell + r,
            ),
        )
    }

    /// 셀 `(cx, cz)`의 삼각형 `t`(0|1). 대각선은 (0,0)-(1,1)로 고정한다 —
    /// 분할 방향이 셀마다 달라지면 같은 지형이 다른 물리를 낸다(결정론).
    ///
    /// 와인딩은 **+Y가 바깥**이다(위에서 봤을 때 CCW). 평평한 격자에서 두 삼각형의
    /// 법선이 모두 +Y인 것을 테스트가 잠근다.
    pub fn triangle(&self, cx: u32, cz: u32, t: u8) -> [Vec3; 3] {
        let v00 = self.vertex(cx, cz);
        let v10 = self.vertex(cx + 1, cz);
        let v01 = self.vertex(cx, cz + 1);
        let v11 = self.vertex(cx + 1, cz + 1);
        if t == 0 {
            [v00, v11, v10]
        } else {
            [v00, v01, v11]
        }
    }

    /// 삼각형의 단위 법선. 퇴화(면적 0)면 `None`.
    pub fn triangle_normal(&self, cx: u32, cz: u32, t: u8) -> Option<Vec3> {
        let tri = self.triangle(cx, cz, t);
        let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
        let scale = (tri[1] - tri[0])
            .length()
            .max((tri[2] - tri[0]).length())
            .max((tri[2] - tri[1]).length());
        let len = n.length();
        (len > crate::tolerance::area_eps(scale)).then(|| n * (1.0 / len))
    }

    /// 에지 `e`(0..3) 건너편 이웃 삼각형 — 격자 구조에서 **좌표로 유도**된다.
    ///
    /// 삼각형 0 = (v00, v11, v10): e0=대각(같은 셀 t1), e1=x+ 경계, e2=z− 경계.
    /// 삼각형 1 = (v00, v01, v11): e0=x− 경계, e1=z+ 경계, e2=대각(같은 셀 t0).
    /// 격자 밖이면 `None`(진짜 외곽 경계 — 용접하지 않는다).
    fn neighbor(&self, cx: u32, cz: u32, t: u8, e: u8) -> Option<(u32, u32, u8)> {
        let (ncx, ncz) = self.cells();
        match (t, e) {
            (0, 0) => Some((cx, cz, 1)),
            (0, 1) => (cx + 1 < ncx).then(|| (cx + 1, cz, 1)),
            (0, 2) => (cz > 0).then(|| (cx, cz - 1, 1)),
            (1, 0) => (cx > 0).then(|| (cx - 1, cz, 0)),
            (1, 1) => (cz + 1 < ncz).then(|| (cx, cz + 1, 0)),
            (1, 2) => Some((cx, cz, 0)),
            _ => None,
        }
    }

    /// 내부 에지 마스크 — 이웃이 있고 **공면**인 에지만 1.
    ///
    /// 능선·절벽은 남긴다: 거기서는 에지 법선이 물리적으로 옳다(공을 굴리면 실제로 튄다).
    /// 판정은 지형 등록부와 같은 각도 임계다 — 스케일 불변·대칭이라 스키니 삼각형에서도
    /// 임계가 부풀지 않는다(검수 rev_3398738c에서 배운 것).
    pub fn weld_mask(&self, cx: u32, cz: u32, t: u8) -> u8 {
        let Some(n) = self.triangle_normal(cx, cz, t) else {
            return 0;
        };
        let mut mask = 0u8;
        for e in 0..3u8 {
            let Some((nx, nz, nt)) = self.neighbor(cx, cz, t, e) else {
                continue;
            };
            let Some(n2) = self.triangle_normal(nx, nz, nt) else {
                continue;
            };
            // 격자는 다양체이고 와인딩이 전부 +Y 기준으로 일관되므로 법선 뒤집기가 없다.
            if n.dot(n2) >= COS_COPLANAR {
                mask |= 1 << e;
            }
        }
        mask
    }

    /// 월드 AABB를 로컬 격자 셀 범위로 — 겹치지 않으면 `None`.
    ///
    /// 반환은 `(cx0, cz0, cx1, cz1)` 포함 구간이다. 격자 밖은 잘라낸다(clamp) —
    /// 잘라낸 결과가 비면 겹침이 없다.
    pub fn cell_range(&self, local: Aabb) -> Option<(u32, u32, u32, u32)> {
        let (ncx, ncz) = self.cells();
        if ncx == 0 || ncz == 0 {
            return None;
        }
        let inv = 1.0 / self.cell;
        let lo_x = ((local.min.x - self.radius) * inv).floor();
        let hi_x = ((local.max.x + self.radius) * inv).floor();
        let lo_z = ((local.min.z - self.radius) * inv).floor();
        let hi_z = ((local.max.z + self.radius) * inv).floor();
        if hi_x < 0.0 || hi_z < 0.0 {
            return None;
        }
        let cx0 = lo_x.max(0.0) as u32;
        let cz0 = lo_z.max(0.0) as u32;
        if cx0 >= ncx || cz0 >= ncz {
            return None;
        }
        let cx1 = (hi_x.max(0.0) as u32).min(ncx - 1);
        let cz1 = (hi_z.max(0.0) as u32).min(ncz - 1);
        if cx1 < cx0 || cz1 < cz0 {
            return None;
        }
        Some((cx0, cz0, cx1, cz1))
    }

    /// 로컬 좌표 점 아래(또는 위)의 표면 높이 — 격자 밖이면 `None`.
    /// 레이·캐스트가 아니라 진단·테스트용 편의 함수다.
    pub fn height_at(&self, x: f32, z: f32) -> Option<f32> {
        let (ncx, ncz) = self.cells();
        let inv = 1.0 / self.cell;
        let fx = x * inv;
        let fz = z * inv;
        // **상한도 검사한다.** clamp만 하고 넘기면 격자에서 수십 미터 떨어진 점이
        // 마지막 셀의 외삽으로 "지형 안"이 된다(검수 rev_1e81559e MAJOR:
        // `query_point`가 98m 밖 점을 지형 내부로 보고했다).
        if !(fx >= 0.0 && fz >= 0.0 && fx <= ncx as f32 && fz <= ncz as f32) {
            return None;
        }
        let cx = (fx.floor() as u32).min(ncx.saturating_sub(1));
        let cz = (fz.floor() as u32).min(ncz.saturating_sub(1));
        if cx >= ncx || cz >= ncz {
            return None;
        }
        let (u, v) = (fx - cx as f32, fz - cz as f32);
        // 대각선 (0,0)-(1,1) 기준: u >= v면 삼각형 0, 아니면 1.
        let (h00, h10, h01, h11) = (
            self.height(cx, cz),
            self.height(cx + 1, cz),
            self.height(cx, cz + 1),
            self.height(cx + 1, cz + 1),
        );
        // 무게중심 좌표를 그대로 편 식이다.
        // 삼각형 0 (u ≥ v, 정점 v00·v11·v10): a=1−u, b=v, c=u−v
        //   → h = h00 + u·(h10−h00) + v·(h11−h10)
        // 삼각형 1 (v ≥ u, 정점 v00·v01·v11): a=1−v, b=v−u, c=u
        //   → h = h00 + v·(h01−h00) + u·(h11−h01)
        Some(if u >= v {
            h00 + (h10 - h00) * u + (h11 - h10) * v
        } else {
            h00 + (h01 - h00) * v + (h11 - h01) * u
        })
    }
}

/// 레이 vs 삼각형 (Möller–Trumbore). 반환 = (toi, 단위 법선 — 레이 반대쪽을 향한다).
/// 뒷면도 맞힌다: 지형 아래에서 위로 쏘는 질의가 유효하기 때문이다.
pub fn ray_triangle(o: Vec3, d: Vec3, tri: [Vec3; 3], max_toi: f32) -> Option<(f32, Vec3)> {
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let p = d.cross(e2);
    let det = e1.dot(p);
    // 퇴화·평행은 없음 처리 — 임계는 면적 차원(edge×edge = L²)에서 유도한다.
    let scale = e1.length().max(e2.length()).max((tri[2] - tri[1]).length());
    if det.abs() <= crate::tolerance::area_eps(scale) * d.length().max(1.0) {
        return None;
    }
    let inv = 1.0 / det;
    let t = o - tri[0];
    let u = t.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = t.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let toi = e2.dot(q) * inv;
    if !(0.0..=max_toi).contains(&toi) {
        return None;
    }
    let n = e1.cross(e2);
    let len = n.length();
    if len <= 0.0 {
        return None;
    }
    let n = n * (1.0 / len);
    Some((toi, if n.dot(d) > 0.0 { -n } else { n }))
}

/// 레이 vs 하이트필드 — 격자 위를 **셀 단위로 전진**(2D DDA)하며 삼각형만 시험한다.
///
/// 전체 삼각형을 훑으면 256×256에서 13만 번이다. 레이가 지나는 셀만 보면 O(경로 길이)다.
/// 셀은 레이 순서로 방문하므로 **첫 히트가 최근접**이고 거기서 끝낸다.
pub fn ray_heightfield(
    hf: &HeightfieldRef<'_>,
    o: Vec3,
    d: Vec3,
    max_toi: f32,
) -> Option<(f32, Vec3)> {
    let (ncx, ncz) = hf.cells();
    if ncx == 0 || ncz == 0 {
        return None;
    }
    let (w, depth) = (ncx as f32 * hf.cell, ncz as f32 * hf.cell);
    // ① XZ 상자에 대한 슬랩 클립 — 격자에 들어오는 구간 [t_in, t_out].
    let mut t_in = 0.0f32;
    let mut t_out = max_toi;
    for (oc, dc, hi) in [(o.x, d.x, w), (o.z, d.z, depth)] {
        if dc.abs() <= f32::MIN_POSITIVE {
            if oc < 0.0 || oc > hi {
                return None; // 축에 평행하면서 범위 밖 — 영원히 안 만난다
            }
            continue;
        }
        let inv = 1.0 / dc;
        let (mut a, mut b) = ((0.0 - oc) * inv, (hi - oc) * inv);
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        t_in = t_in.max(a);
        t_out = t_out.min(b);
    }
    if t_in > t_out {
        return None;
    }
    // ② 진입 셀
    let p = o + d * t_in;
    let inv_cell = 1.0 / hf.cell;
    let mut cx = ((p.x * inv_cell).floor().max(0.0) as u32).min(ncx - 1);
    let mut cz = ((p.z * inv_cell).floor().max(0.0) as u32).min(ncz - 1);
    // ③ DDA 준비: 다음 경계까지의 t와 셀 하나를 건너는 t.
    let setup = |oc: f32, dc: f32, ci: u32| -> (i32, f32, f32) {
        if dc > 0.0 {
            let next = (ci + 1) as f32 * hf.cell;
            (1, (next - oc) / dc, hf.cell / dc)
        } else if dc < 0.0 {
            let next = ci as f32 * hf.cell;
            (-1, (next - oc) / dc, hf.cell / -dc)
        } else {
            (0, f32::INFINITY, f32::INFINITY)
        }
    };
    let (step_x, mut t_next_x, dt_x) = setup(o.x, d.x, cx);
    let (step_z, mut t_next_z, dt_z) = setup(o.z, d.z, cz);
    // 상한을 두지 않는다: DDA는 격자를 **한 방향으로** 지나므로 방문 셀이
    // ncx+ncz+2를 넘을 수 없다(경계에 닿으면 루프가 끝난다). 인위적 상한은 긴 지형에서
    // 진짜 히트를 조용히 미스로 만들었다(검수 rev_1e81559e BLOCKER).
    let mut visited = 0usize;
    let visit_bound = (ncx as usize + ncz as usize).saturating_add(2);
    loop {
        for t in 0..2u8 {
            if let Some(hit) = ray_triangle(o, d, hf.triangle(cx, cz, t), max_toi) {
                // 셀 안에서 두 삼각형 모두 맞을 수 있다 — 가까운 쪽.
                let other = ray_triangle(o, d, hf.triangle(cx, cz, t ^ 1), max_toi);
                return Some(match other {
                    Some(h2) if h2.0 < hit.0 => h2,
                    _ => hit,
                });
            }
        }
        visited += 1;
        debug_assert!(
            visited <= visit_bound,
            "DDA가 격자 경계를 못 벗어났다 — 전진 규칙이 깨졌다"
        );
        if visited > visit_bound {
            return None; // 도달 불가(방어) — 무한 루프 대신 종료
        }
        // 다음 셀로
        if t_next_x < t_next_z {
            if t_next_x > t_out {
                return None;
            }
            let nx_i = cx as i64 + step_x as i64;
            if nx_i < 0 || nx_i >= ncx as i64 {
                return None;
            }
            cx = nx_i as u32;
            t_next_x += dt_x;
        } else {
            if t_next_z > t_out {
                return None;
            }
            let nz_i = cz as i64 + step_z as i64;
            if nz_i < 0 || nz_i >= ncz as i64 {
                return None;
            }
            cz = nz_i as u32;
            t_next_z += dt_z;
        }
    }
}

/// 높이 배열에서 타일별 (최소, 최대)를 만든다.
///
/// 셰이프 생성·복원에서 **한 번** 계산해 두는 파생 데이터다(캐노니컬 상태가 아니다 —
/// 높이 배열의 함수이므로 스냅샷에 넣지 않고 복원 때 다시 만든다).
pub fn build_tiles(heights: &[f32], nx: u32, nz: u32) -> Vec<(f32, f32)> {
    // **잘못된 입력에는 빈 목록**을 돌려준다 — 검증 전에 불릴 수 있고(생성 경계),
    // 여기서 패닉하면 "실패는 오류지 trap이 아니다" 계약이 깨진다.
    // 빈 목록은 `validate`가 타일 길이 불일치로 거부한다.
    //
    // **곱셈은 checked로** — wasm32의 usize는 32비트라 `nx*nz`가 래핑한다. nx=nz=65537,
    // 높이 131073개인 악성 스냅샷(0.5MiB)이 길이 검사를 통과한 뒤 타일 6,700만 개(512MiB)를
    // 잡으려 해서 `validate`에 닿기 전에 OOM trap이 났다(확인검수 rev_e50670ad BLOCKER).
    let Some(samples) = (nx as usize).checked_mul(nz as usize) else {
        return Vec::new();
    };
    if nx < 2 || nz < 2 || samples != heights.len() || samples > MAX_HEIGHTFIELD_SAMPLES {
        return Vec::new();
    }
    let (ncx, ncz) = (nx - 1, nz - 1);
    let (ntx, ntz) = (
        ncx.div_ceil(HEIGHTFIELD_TILE),
        ncz.div_ceil(HEIGHTFIELD_TILE),
    );
    let Some(tile_count) = (ntx as usize).checked_mul(ntz as usize) else {
        return Vec::new();
    };
    let mut tiles = Vec::new();
    if tiles.try_reserve_exact(tile_count).is_err() {
        return Vec::new();
    }
    tiles.resize(tile_count, (f32::INFINITY, f32::NEG_INFINITY));
    for tz in 0..ntz {
        for tx in 0..ntx {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            // 셀의 네 꼭짓점을 다 봐야 한다 — 타일 경계 셀은 옆 타일의 정점도 쓴다.
            let x1 = ((tx + 1) * HEIGHTFIELD_TILE).min(ncx);
            let z1 = ((tz + 1) * HEIGHTFIELD_TILE).min(ncz);
            for iz in (tz * HEIGHTFIELD_TILE)..=z1 {
                for ix in (tx * HEIGHTFIELD_TILE)..=x1 {
                    let h = heights[(iz * nx + ix) as usize];
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            tiles[(tz * ntx + tx) as usize] = (lo, hi);
        }
    }
    tiles
}

/// 셀 범위 안에서 **Y 밴드와 겹치는 타일의 셀만** 훑는 반복자 헬퍼.
///
/// 셀을 다 돌면서 타일 범위를 매번 조회하면 가지치기가 **일은 줄여도 반복은 그대로**다 —
/// 1024×1024 격자에서 후보가 0개여도 100만 번 돈다(확인검수 rev_e50670ad MAJOR).
/// 타일을 바깥 루프로 두면 겹치지 않는 타일의 셀 64개는 **아예 방문하지 않는다**.
///
/// `f`가 `false`를 돌려주면 즉시 중단하고 `false`를 반환한다(예산 초과·조기 종료).
pub fn for_each_overlapping_cell(
    hf: &HeightfieldRef<'_>,
    range: (u32, u32, u32, u32),
    y_lo: f32,
    y_hi: f32,
    mut f: impl FnMut(u32, u32) -> bool,
) -> bool {
    let (cx0, cz0, cx1, cz1) = range;
    for tz in (cz0 / HEIGHTFIELD_TILE)..=(cz1 / HEIGHTFIELD_TILE) {
        for tx in (cx0 / HEIGHTFIELD_TILE)..=(cx1 / HEIGHTFIELD_TILE) {
            let (lo, hi) = hf.tile_range(tx, tz);
            if hi < y_lo || lo > y_hi {
                continue;
            }
            let x0 = cx0.max(tx * HEIGHTFIELD_TILE);
            let z0 = cz0.max(tz * HEIGHTFIELD_TILE);
            let x1 = cx1.min((tx + 1) * HEIGHTFIELD_TILE - 1);
            let z1 = cz1.min((tz + 1) * HEIGHTFIELD_TILE - 1);
            for cz in z0..=z1 {
                for cx in x0..=x1 {
                    if !f(cx, cz) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Y 밴드와 겹치는 타일이 덮는 **셀 수** — 공개 질의의 사전 예산 판정용.
///
/// raw 셀 수를 세면 "멀리 있는 봉우리 하나가 지형 AABB를 높여" 실제 후보가 0개인 질의도
/// 거절한다(확인검수 rev_e50670ad MAJOR). 실제로 볼 셀만 센다.
pub fn overlapping_cell_count(
    hf: &HeightfieldRef<'_>,
    range: (u32, u32, u32, u32),
    y_lo: f32,
    y_hi: f32,
) -> usize {
    let mut n = 0usize;
    for_each_overlapping_cell(hf, range, y_lo, y_hi, |_, _| {
        n += 1;
        true
    });
    n
}

/// 타일 배열이 **정말 이 높이 배열에서 유도된 것인가** — 값까지 확인한다.
///
/// 길이만 보면 안 된다: `Shape::Heightfield`는 공개 필드라 호스트가 직접 만들 수 있고,
/// 길이만 맞는 엉뚱한 타일(예: 평지에 `[(100,100)]`)을 넣으면 가지치기가 **모든 실제 지형을
/// 건너뛴다** — 스윕이 `None`을 돌려주고 물체가 지형을 통과한다.
/// 더 나쁜 것은 스냅샷이다: 타일은 직렬화하지 않으므로 복원 월드는 **올바른** 타일을 갖는다.
/// 즉 같은 바이트·같은 상태 해시인데 다음 CCD 결과가 달라진다 — L0 결정론과
/// 입력/복원 대칭이 동시에 깨진다(확인검수 rev_e50670ad BLOCKER).
///
/// 할당 없이(재계산해 비교) 검사한다 — `Body::validate`가 커맨드 flush에서도 불린다.
/// 비교는 **비트 단위**다: 파생값이 조금이라도 다르면 그건 다른 파생이다.
pub fn tiles_match(heights: &[f32], nx: u32, nz: u32, tiles: &[(f32, f32)]) -> bool {
    let Some(samples) = (nx as usize).checked_mul(nz as usize) else {
        return false;
    };
    if nx < 2 || nz < 2 || samples != heights.len() {
        return false;
    }
    let (ncx, ncz) = (nx - 1, nz - 1);
    let (ntx, ntz) = (
        ncx.div_ceil(HEIGHTFIELD_TILE),
        ncz.div_ceil(HEIGHTFIELD_TILE),
    );
    let Some(tile_count) = (ntx as usize).checked_mul(ntz as usize) else {
        return false;
    };
    if tiles.len() != tile_count {
        return false;
    }
    for tz in 0..ntz {
        for tx in 0..ntx {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            let x1 = ((tx + 1) * HEIGHTFIELD_TILE).min(ncx);
            let z1 = ((tz + 1) * HEIGHTFIELD_TILE).min(ncz);
            for iz in (tz * HEIGHTFIELD_TILE)..=z1 {
                for ix in (tx * HEIGHTFIELD_TILE)..=x1 {
                    let h = heights[(iz * nx + ix) as usize];
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            let got = tiles[(tz * ntx + tx) as usize];
            if got.0.to_bits() != lo.to_bits() || got.1.to_bits() != hi.to_bits() {
                return false;
            }
        }
    }
    true
}

/// 로컬 공간 AABB를 포즈로 되돌린 것 — 월드 AABB를 하이트필드 로컬로 옮긴다.
pub fn world_aabb_to_local(aabb: Aabb, pose: &Pose) -> Aabb {
    // 회전의 역변환은 켤레. 축 정렬 상자를 회전시키면 다시 축 정렬 상자로 감싸야 한다.
    let inv_rot = pose.rot.conjugate();
    let c = (aabb.min + aabb.max) * 0.5;
    let e = (aabb.max - aabb.min) * 0.5;
    let lc = inv_rot.rotate(c - pose.pos);
    // 회전된 상자를 다시 축 정렬로 감싼다 — |R|·e (열 벡터 기준).
    let m = maru_math::Mat3::from_quat(inv_rot);
    let le = Vec3::new(
        m.c[0].x.abs() * e.x + m.c[1].x.abs() * e.y + m.c[2].x.abs() * e.z,
        m.c[0].y.abs() * e.x + m.c[1].y.abs() * e.y + m.c[2].y.abs() * e.z,
        m.c[0].z.abs() * e.x + m.c[1].z.abs() * e.y + m.c[2].z.abs() * e.z,
    );
    Aabb::new(lc - le, lc + le)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static TILES: RefCell<Vec<(f32, f32)>> = const { RefCell::new(Vec::new()) };
    }

    fn flat(nx: u32, nz: u32, _cell: f32) -> Vec<f32> {
        vec![0.0; (nx * nz) as usize]
    }

    fn hf(heights: &[f32], nx: u32, nz: u32, cell: f32) -> HeightfieldRef<'_> {
        // 테스트는 타일을 매번 새로 만든다 — 누수 없이 'static으로 두려고 Box::leak을
        // 쓰지 않고, 호출자가 소유하도록 별도 헬퍼를 둔다.
        HeightfieldRef {
            tiles: TILES.with(|t| {
                let v = build_tiles(heights, nx, nz);
                *t.borrow_mut() = v;
                // SAFETY 대신: 테스트 전용으로 thread_local 버퍼를 빌려 준다.
                unsafe { core::slice::from_raw_parts(t.borrow().as_ptr(), t.borrow().len()) }
            }),
            heights,
            nx,
            nz,
            cell,
            radius: 0.0,
        }
    }

    /// 평평한 격자의 두 삼각형은 모두 위를 본다 — 와인딩 계약.
    #[test]
    fn flat_field_triangles_face_up() {
        let h = flat(4, 4, 1.0);
        let f = hf(&h, 4, 4, 1.0);
        for cz in 0..3 {
            for cx in 0..3 {
                for t in 0..2u8 {
                    let n = f.triangle_normal(cx, cz, t).expect("퇴화 아님");
                    assert!(
                        n.y > 0.99,
                        "셀({cx},{cz}) 삼각형{t} 법선이 아래를 본다: {n:?}"
                    );
                }
            }
        }
    }

    /// 평평한 격자에서는 **모든 내부 에지가 용접**되고 외곽 경계만 남는다.
    #[test]
    fn flat_field_welds_every_interior_edge() {
        let h = flat(3, 3, 1.0);
        let f = hf(&h, 3, 3, 1.0);
        // 셀 (0,0) 삼각형 0의 에지: e0=대각(항상 이웃), e1=x+ 이웃 있음, e2=z− 없음(경계)
        assert_eq!(f.weld_mask(0, 0, 0), 0b011);
        // 삼각형 1의 에지: e0=x− 없음, e1=z+ 있음, e2=대각
        assert_eq!(f.weld_mask(0, 0, 1), 0b110);
        // 셀 (1,1) 삼각형 0: e0=대각, e1=x+ 격자 밖(경계), e2=z− 이웃 있음
        assert_eq!(f.weld_mask(1, 1, 0), 0b101);
        // 셀 (1,0) 삼각형 1: e0=x− 이웃, e1=z+ 이웃, e2=대각 — 전부 내부
        assert_eq!(f.weld_mask(1, 0, 1), 0b111);
    }

    /// 절벽(능선)은 용접하지 않는다 — 거기서는 에지 법선이 물리적으로 옳다.
    #[test]
    fn ridge_edges_are_not_welded() {
        // 한 줄만 높이 5 → 급경사
        let mut h = flat(3, 3, 1.0);
        h[3] = 5.0; // (0,1)
        h[4] = 5.0; // (1,1)
        h[5] = 5.0; // (2,1)
        let f = hf(&h, 3, 3, 1.0);
        // 셀(0,0) 삼각형1의 z+ 에지는 셀(0,1)과 만나는데 경사가 급해 공면이 아니다.
        let m = f.weld_mask(0, 0, 1);
        assert_eq!(m & 0b010, 0, "능선 에지가 용접됐다: {m:#b}");
    }

    /// 셀 범위 계산 — 격자 밖·부분 겹침·완전 포함.
    #[test]
    fn cell_range_clamps_to_grid() {
        let h = flat(5, 5, 2.0); // 4×4 셀, 0..8 m
        let f = hf(&h, 5, 5, 2.0);
        assert!(f
            .cell_range(Aabb::new(
                Vec3::new(-10.0, -1.0, -10.0),
                Vec3::new(-5.0, 1.0, -5.0)
            ))
            .is_none());
        let r = f
            .cell_range(Aabb::new(
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(3.0, 1.0, 3.0),
            ))
            .expect("겹친다");
        assert_eq!(r, (0, 0, 1, 1));
        let r = f
            .cell_range(Aabb::new(
                Vec3::new(-100.0, -1.0, -100.0),
                Vec3::new(100.0, 1.0, 100.0),
            ))
            .expect("전체 포함");
        assert_eq!(r, (0, 0, 3, 3));
    }

    /// **타일 범위는 그 타일 셀들의 모든 정점을 덮어야 한다.**
    ///
    /// 이게 깨지면 가지치기가 진짜 접촉을 건너뛴다 — 성능 최적화가 조용한 관통이 된다.
    /// 타일 경계 셀은 **옆 타일의 정점**도 쓰므로(셀 (7,z)는 정점 x=8을 쓴다) 한 줄만
    /// 빠뜨려도 깨진다. 전수로 확인한다.
    #[test]
    fn tile_ranges_cover_every_cell_vertex() {
        // 타일 경계를 넘나드는 크기 + 불규칙한 높이
        let (nx, nz) = (19u32, 13u32);
        let mut h = vec![0.0f32; (nx * nz) as usize];
        for iz in 0..nz {
            for ix in 0..nx {
                // 결정적 의사난수 — 인접 정점이 크게 다르게
                let v = ((ix * 7 + iz * 13) % 11) as f32 * 0.37 - 2.0;
                h[(iz * nx + ix) as usize] = v;
            }
        }
        let tiles = build_tiles(&h, nx, nz);
        let f = HeightfieldRef {
            heights: &h,
            tiles: &tiles,
            nx,
            nz,
            cell: 1.0,
            radius: 0.0,
        };
        let (ncx, ncz) = f.cells();
        for cz in 0..ncz {
            for cx in 0..ncx {
                let (lo, hi) = f.tile_range(cx / HEIGHTFIELD_TILE, cz / HEIGHTFIELD_TILE);
                for t in 0..2u8 {
                    for v in f.triangle(cx, cz, t) {
                        assert!(
                            v.y >= lo && v.y <= hi,
                            "셀({cx},{cz}) 삼각형{t} 정점 y={} 이 타일 범위 [{lo},{hi}] 밖이다\
                             — 가지치기가 이 접촉을 건너뛴다",
                            v.y
                        );
                    }
                }
            }
        }
    }

    /// 잘못된 격자에는 빈 타일 — 생성 경계에서 패닉하지 않는다(검증이 거부한다).
    #[test]
    fn build_tiles_is_defensive_about_bad_input() {
        assert!(build_tiles(&[0.0; 8], 3, 3).is_empty(), "샘플 부족");
        assert!(build_tiles(&[0.0; 1], 1, 1).is_empty(), "격자 미달");
        assert!(!build_tiles(&[0.0; 9], 3, 3).is_empty());
    }

    /// 레이가 격자를 맞히고, 빗나가고, 아래에서 위로도 맞힌다.
    #[test]
    fn ray_hits_the_grid_surface() {
        let mut h = flat(5, 5, 1.0);
        h[12] = 1.0; // 가운데 정점(2,2)만 1m
        let f = hf(&h, 5, 5, 1.0);
        // 위에서 아래로 — 평평한 부분
        let (toi, n) = ray_heightfield(
            &f,
            Vec3::new(0.5, 5.0, 0.5),
            Vec3::new(0.0, -1.0, 0.0),
            100.0,
        )
        .expect("맞아야 한다");
        assert!((toi - 5.0).abs() < 1e-4, "toi {toi}");
        assert!(n.y > 0.99, "법선이 위를 봐야 한다: {n:?}");
        // 솟은 정점 위 — 더 일찍 맞는다
        let (toi2, _) = ray_heightfield(
            &f,
            Vec3::new(2.0, 5.0, 2.0),
            Vec3::new(0.0, -1.0, 0.0),
            100.0,
        )
        .expect("맞아야 한다");
        assert!((toi2 - 4.0).abs() < 1e-3, "toi {toi2}");
        // 격자 밖 — 빗나간다
        assert!(ray_heightfield(
            &f,
            Vec3::new(-3.0, 5.0, -3.0),
            Vec3::new(0.0, -1.0, 0.0),
            100.0
        )
        .is_none());
        // 아래에서 위로 — 뒷면도 맞히고 법선은 레이 반대쪽
        let (_, nb) = ray_heightfield(
            &f,
            Vec3::new(0.5, -5.0, 0.5),
            Vec3::new(0.0, 1.0, 0.0),
            100.0,
        )
        .expect("뒷면 히트");
        assert!(nb.y < -0.99, "뒷면 법선: {nb:?}");
    }

    /// 비스듬한 레이는 여러 셀을 지나 **가장 가까운** 표면을 맞힌다.
    #[test]
    fn oblique_ray_walks_cells_and_takes_the_nearest_hit() {
        // 계단: z가 커질수록 높아진다
        let mut h = flat(6, 6, 1.0);
        for iz in 0..6u32 {
            for ix in 0..6u32 {
                h[(iz * 6 + ix) as usize] = iz as f32 * 0.5;
            }
        }
        let f = hf(&h, 6, 6, 1.0);
        let o = Vec3::new(2.5, 4.0, 0.2);
        let d = Vec3::new(0.0, -1.0, 1.0).normalize_or_zero();
        let (toi, n) = ray_heightfield(&f, o, d, 100.0).expect("계단을 맞힌다");
        let p = o + d * toi;
        let surface = f.height_at(p.x, p.z).expect("격자 안");
        assert!(
            (p.y - surface).abs() < 1e-2,
            "히트가 표면 위가 아니다: y={} surface={surface}",
            p.y
        );
        assert!(n.dot(d) < 0.0, "법선이 레이를 마주봐야 한다");
    }

    /// 보간 높이는 삼각형 평면 위에 있다 — 정점에서 정확히 일치.
    #[test]
    fn height_at_matches_vertices() {
        let mut h = flat(3, 3, 1.0);
        h[4] = 2.0; // 가운데 정점만 올림
        let f = hf(&h, 3, 3, 1.0);
        assert_eq!(f.height_at(1.0, 1.0), Some(2.0));
        assert_eq!(f.height_at(0.0, 0.0), Some(0.0));
        assert_eq!(f.height_at(2.0, 0.0), Some(0.0));
        // 대각선 위 중점은 두 삼각형이 공유 — 어느 쪽으로 풀어도 같은 값
        let mid = f.height_at(0.5, 0.5).expect("안쪽");
        assert!((mid - 1.0).abs() < 1e-6, "대각 중점 {mid}");
    }
}
