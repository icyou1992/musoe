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

/// 캐스트·레이가 훑을 수 있는 셀 수 상한 — 무한 루프 방지.
pub const MAX_HEIGHTFIELD_CELLS_PER_QUERY: usize = 4096;

/// 공면 판정 임계 — 지형 등록부(`World::try_add_terrain`)와 **같은 값**이어야 한다.
/// Jolt ActiveEdges 기본과 동급(~5.7°).
pub const COS_COPLANAR: f32 = 0.995;

/// 격자 하이트필드의 기하 — `Shape::Heightfield`의 계산을 모아 둔 뷰.
///
/// 셰이프가 소유한 데이터를 빌려 쓰기만 한다(복사 없음).
#[derive(Clone, Copy)]
pub struct HeightfieldRef<'a> {
    pub heights: &'a [f32],
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
        if !(fx >= 0.0 && fz >= 0.0) {
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
    let mut visited = 0usize;
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
        if visited >= MAX_HEIGHTFIELD_CELLS_PER_QUERY {
            return None; // 상한 도달 — 조용한 무한 루프 대신 미탐지
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

    fn flat(nx: u32, nz: u32, _cell: f32) -> Vec<f32> {
        vec![0.0; (nx * nz) as usize]
    }

    fn hf(heights: &[f32], nx: u32, nz: u32, cell: f32) -> HeightfieldRef<'_> {
        HeightfieldRef {
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
