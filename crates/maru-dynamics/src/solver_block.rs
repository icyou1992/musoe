//! 접촉 제약의 **SoA 블록** — 게더/스캐터를 스텝당 1회로 밀어낸다 (설계 §3 SoA 핫패스).
//!
//! `solver::solve_x4`는 4-wide로 풀지만 데이터는 여전히 AoS다: 제약 4개에서 접점 `k`의
//! 필드를 매번 `f32x4::from([a,b,c,d])`로 **모아 실어야** 한다. 이 게더는 서브스텝마다
//! warm start·bias·relax 세 번, 스텝당 12번 반복된다.
//!
//! 실측이 그 비용을 지목했다 (2026-07-29, 40단 피라미드):
//! - 각 야코비안 선계산으로 solve의 **산술**을 ~40% 걷어냈는데 시간은 7.6%만 줄었다.
//!   같은 변경이 refresh 비용까지 정직하게 물면 **순손해**였다 (7.40 → 8.52 ms).
//!   → 이 루프는 flop-bound가 아니다.
//! - 블록 1개·패스 1회의 메모리 연산: 바디 게더 ~136 + 접점 게더/스캐터 4×72 ≈ 424.
//!   접점을 SoA로 두면 후자가 0이 된다 (바디는 제약끼리 공유하므로 게더가 남는다).
//!
//! 그래서 접점 데이터를 **처음부터 레인 배치로** 만들어 둔다. 블록 조립은 스텝당 1회,
//! 임펄스 되쓰기(PCM 갱신용)도 스텝당 1회다.
//!
//! **비트 동일 규율**은 `solve_x4`와 같다: 레인당 연산 순서를 스칼라와 동일하게 유지하고,
//! 게이트는 산술 마스킹이 아니라 blend로 한다. 블록 경로와 기존 x4 경로가 같은 상태를
//! 내는 것은 아래 테스트가 잠근다.

use crate::solver::{
    solve as solve_scalar, warm_start as warm_start_scalar, BatchSolveParams, ContactConstraint,
    SolverState, COLOR_OVERFLOW,
};
use maru_math::{Quatx4, Vec3, Vec3x4};
use wide::{f32x4, CmpGt};

/// 블록 하나의 접점 슬롯 (4레인 × 접점 1개).
#[derive(Clone, Copy)]
pub struct BlockPoint {
    /// 질량중심 기준 로컬 앵커 — 매 서브스텝 현재 자세로 재회전한다.
    pub anchor_a: Vec3x4,
    pub anchor_b: Vec3x4,
    /// 이번 서브스텝의 월드 레버암.
    pub ra: Vec3x4,
    pub rb: Vec3x4,
    pub base_separation: f32x4,
    pub normal_mass: f32x4,
    pub tangent_mass: [f32x4; 2],
    pub normal_impulse: f32x4,
    pub tangent_impulse: [f32x4; 2],
    pub max_normal_impulse: f32x4,
    /// 점 존재 마스크 (레인마다 접점 수가 다르다).
    pub has: f32x4,
}

impl Default for BlockPoint {
    fn default() -> Self {
        let z = f32x4::splat(0.0);
        Self {
            anchor_a: Vec3x4::splat(Vec3::ZERO),
            anchor_b: Vec3x4::splat(Vec3::ZERO),
            ra: Vec3x4::splat(Vec3::ZERO),
            rb: Vec3x4::splat(Vec3::ZERO),
            base_separation: z,
            normal_mass: z,
            tangent_mass: [z; 2],
            normal_impulse: z,
            tangent_impulse: [z; 2],
            max_normal_impulse: z,
            has: z.cmp_gt(f32x4::splat(1.0)), // all-false
        }
    }
}

/// 같은 색의 제약 4개를 레인에 실은 SoA 블록. 접점은 매니폴드 상한(4)만큼 슬롯을 갖는다.
#[derive(Clone)]
pub struct ContactBlock {
    pub a: [usize; 4],
    pub b: [usize; 4],
    pub normal: Vec3x4,
    pub tangents: [Vec3x4; 2],
    pub friction: f32x4,
    /// 레인별 접점 수의 최대 — 슬롯 순회 상한.
    pub n_points: usize,
    pub points: [BlockPoint; 4],
}

/// 한 매니폴드가 가질 수 있는 접점 상한 (클리퍼 `reduce_to_4`).
pub const MAX_BLOCK_POINTS: usize = 4;

impl Default for ContactBlock {
    fn default() -> Self {
        Self {
            a: [0; 4],
            b: [0; 4],
            normal: Vec3x4::splat(Vec3::ZERO),
            tangents: [Vec3x4::splat(Vec3::ZERO); 2],
            friction: f32x4::splat(0.0),
            n_points: 0,
            points: [BlockPoint::default(); MAX_BLOCK_POINTS],
        }
    }
}

impl ContactBlock {
    /// 제약 4개를 **제자리에** 싣는다. 접점 수가 다르면 `has` 마스크로 걸러 원본을 보존한다.
    ///
    /// 블록은 1.5KB짜리 구조체라 스텝마다 새로 만들어 push하면 그 이동·할당이 조립 이득을
    /// 갉아먹는다 — 월드가 버퍼를 보관하고 여기서 덮어쓴다.
    pub fn assemble_into(&mut self, cs: [&ContactConstraint; 4]) {
        let n_points = cs.iter().map(|c| c.points.len()).max().unwrap_or(0);
        debug_assert!(
            n_points <= MAX_BLOCK_POINTS,
            "매니폴드 접점이 블록 슬롯을 넘었다 — 클리퍼 상한과 어긋난다"
        );
        let points = &mut self.points;
        for (k, slot) in points.iter_mut().enumerate().take(n_points) {
            let q = [
                cs[0].points.get(k),
                cs[1].points.get(k),
                cs[2].points.get(k),
                cs[3].points.get(k),
            ];
            macro_rules! v3 {
                ($f:ident) => {
                    Vec3x4::from_array([
                        q[0].map_or(Vec3::ZERO, |p| p.$f),
                        q[1].map_or(Vec3::ZERO, |p| p.$f),
                        q[2].map_or(Vec3::ZERO, |p| p.$f),
                        q[3].map_or(Vec3::ZERO, |p| p.$f),
                    ])
                };
            }
            macro_rules! sc {
                ($e:expr) => {
                    f32x4::from([
                        q[0].map_or(0.0, $e),
                        q[1].map_or(0.0, $e),
                        q[2].map_or(0.0, $e),
                        q[3].map_or(0.0, $e),
                    ])
                };
            }
            *slot = BlockPoint {
                anchor_a: v3!(anchor_a),
                anchor_b: v3!(anchor_b),
                ra: v3!(ra),
                rb: v3!(rb),
                base_separation: sc!(|p| p.base_separation),
                normal_mass: sc!(|p| p.normal_mass),
                tangent_mass: [sc!(|p| p.tangent_mass[0]), sc!(|p| p.tangent_mass[1])],
                normal_impulse: sc!(|p| p.normal_impulse),
                tangent_impulse: [sc!(|p| p.tangent_impulse[0]), sc!(|p| p.tangent_impulse[1])],
                max_normal_impulse: sc!(|p| p.max_normal_impulse),
                has: f32x4::from([
                    q[0].is_some() as u32 as f32,
                    q[1].is_some() as u32 as f32,
                    q[2].is_some() as u32 as f32,
                    q[3].is_some() as u32 as f32,
                ])
                .cmp_gt(f32x4::splat(0.5)),
            };
        }
        self.a = [cs[0].a, cs[1].a, cs[2].a, cs[3].a];
        self.b = [cs[0].b, cs[1].b, cs[2].b, cs[3].b];
        self.normal = Vec3x4::from_array([cs[0].normal, cs[1].normal, cs[2].normal, cs[3].normal]);
        self.tangents = [
            Vec3x4::from_array([
                cs[0].tangents[0],
                cs[1].tangents[0],
                cs[2].tangents[0],
                cs[3].tangents[0],
            ]),
            Vec3x4::from_array([
                cs[0].tangents[1],
                cs[1].tangents[1],
                cs[2].tangents[1],
                cs[3].tangents[1],
            ]),
        ];
        self.friction = f32x4::from([
            cs[0].friction,
            cs[1].friction,
            cs[2].friction,
            cs[3].friction,
        ]);
        self.n_points = n_points;
    }

    /// 누적 임펄스를 원본 제약으로 되쓴다 — PCM 캐시 갱신이 스칼라 제약을 읽기 때문이다.
    /// 스텝당 1회(서브스텝 루프 종료 후)면 충분하다.
    pub fn scatter_impulses(&self, cs: [&mut ContactConstraint; 4]) {
        let mut cs = cs;
        for k in 0..self.n_points {
            let p = &self.points[k];
            let (ni, ti0, ti1, mni) = (
                p.normal_impulse.to_array(),
                p.tangent_impulse[0].to_array(),
                p.tangent_impulse[1].to_array(),
                p.max_normal_impulse.to_array(),
            );
            for (lane, c) in cs.iter_mut().enumerate() {
                if let Some(pt) = c.points.get_mut(k) {
                    pt.normal_impulse = ni[lane];
                    pt.tangent_impulse = [ti0[lane], ti1[lane]];
                    pt.max_normal_impulse = mni[lane];
                }
            }
        }
    }
}

/// 4레인 바디 상태 (블록 경로 전용). `inv_i`까지 싣는다 — 임펄스의 각 성분이 쓴다.
#[derive(Clone, Copy)]
struct BodyRegs {
    lin: Vec3x4,
    ang: Vec3x4,
    inv_m: f32x4,
    inv_i: maru_math::Mat3x4,
    dyn_mask: f32x4,
}

impl BodyRegs {
    #[inline]
    fn load(st: &SolverState, idx: [usize; 4]) -> Self {
        Self {
            lin: Vec3x4::from_array([
                st.lin[idx[0]],
                st.lin[idx[1]],
                st.lin[idx[2]],
                st.lin[idx[3]],
            ]),
            ang: Vec3x4::from_array([
                st.ang[idx[0]],
                st.ang[idx[1]],
                st.ang[idx[2]],
                st.ang[idx[3]],
            ]),
            inv_m: f32x4::from([
                st.inv_m[idx[0]],
                st.inv_m[idx[1]],
                st.inv_m[idx[2]],
                st.inv_m[idx[3]],
            ]),
            inv_i: maru_math::Mat3x4::from_array([
                st.inv_i[idx[0]],
                st.inv_i[idx[1]],
                st.inv_i[idx[2]],
                st.inv_i[idx[3]],
            ]),
            dyn_mask: f32x4::from([
                st.dynamic[idx[0]] as u32 as f32,
                st.dynamic[idx[1]] as u32 as f32,
                st.dynamic[idx[2]] as u32 as f32,
                st.dynamic[idx[3]] as u32 as f32,
            ])
            .cmp_gt(f32x4::splat(0.5)),
        }
    }

    #[inline]
    fn store(&self, st: &mut SolverState, idx: [usize; 4]) {
        let (lin, ang) = (lanes(self.lin), lanes(self.ang));
        for k in 0..4 {
            if st.dynamic[idx[k]] {
                st.lin[idx[k]] = lin[k];
                st.ang[idx[k]] = ang[k];
            }
        }
    }

    #[inline]
    fn vel_at(&self, r: Vec3x4) -> Vec3x4 {
        self.lin + self.ang.cross(r)
    }

    #[inline]
    fn apply(&mut self, imp: Vec3x4, r: Vec3x4, active: f32x4) {
        let gate = self.dyn_mask & active;
        let new_lin = self.lin + imp.scale(self.inv_m);
        let new_ang = self.ang + self.inv_i.mul_vec(r.cross(imp));
        self.lin = blend(gate, new_lin, self.lin);
        self.ang = blend(gate, new_ang, self.ang);
    }
}

#[inline]
fn lanes(v: Vec3x4) -> [Vec3; 4] {
    let (x, y, z) = (v.x.to_array(), v.y.to_array(), v.z.to_array());
    [
        Vec3::new(x[0], y[0], z[0]),
        Vec3::new(x[1], y[1], z[1]),
        Vec3::new(x[2], y[2], z[2]),
        Vec3::new(x[3], y[3], z[3]),
    ]
}

#[inline]
fn neg(v: Vec3x4) -> Vec3x4 {
    Vec3x4 {
        x: -v.x,
        y: -v.y,
        z: -v.z,
    }
}

#[inline]
fn blend(gate: f32x4, t: Vec3x4, f: Vec3x4) -> Vec3x4 {
    Vec3x4 {
        x: gate.blend(t.x, f.x),
        y: gate.blend(t.y, f.y),
        z: gate.blend(t.z, f.z),
    }
}

/// 앵커 재회전 — 4-wide. 레인당 자세가 달라도 회전식은 레인별이라 셔플이 없다.
pub fn refresh_anchors_block(st: &SolverState, blk: &mut ContactBlock) {
    let qa = Quatx4::from_array([
        st.rot[blk.a[0]],
        st.rot[blk.a[1]],
        st.rot[blk.a[2]],
        st.rot[blk.a[3]],
    ]);
    let qb = Quatx4::from_array([
        st.rot[blk.b[0]],
        st.rot[blk.b[1]],
        st.rot[blk.b[2]],
        st.rot[blk.b[3]],
    ]);
    for p in blk.points.iter_mut().take(blk.n_points) {
        p.ra = qa.rotate(p.anchor_a);
        p.rb = qb.rotate(p.anchor_b);
    }
}

/// warm start — 블록판. 레인당 스칼라 `warm_start`와 비트 동일.
pub fn warm_start_block(st: &mut SolverState, blk: &ContactBlock) {
    let mut ba = BodyRegs::load(st, blk.a);
    let mut bb = BodyRegs::load(st, blk.b);
    for p in blk.points.iter().take(blk.n_points) {
        let imp = blk.normal.scale(p.normal_impulse)
            + blk.tangents[0].scale(p.tangent_impulse[0])
            + blk.tangents[1].scale(p.tangent_impulse[1]);
        ba.apply(imp, p.ra, p.has);
        bb.apply(neg(imp), p.rb, p.has);
    }
    ba.store(st, blk.a);
    bb.store(st, blk.b);
}

/// solve — 블록판. `solver::solve_x4`와 **같은 산술·같은 순서**이고 게더만 없다.
pub fn solve_block(st: &mut SolverState, blk: &mut ContactBlock, p: &BatchSolveParams) {
    let mut ba = BodyRegs::load(st, blk.a);
    let mut bb = BodyRegs::load(st, blk.b);
    let pos_a = Vec3x4::from_array([
        st.pos[blk.a[0]],
        st.pos[blk.a[1]],
        st.pos[blk.a[2]],
        st.pos[blk.a[3]],
    ]);
    let pos_b = Vec3x4::from_array([
        st.pos[blk.b[0]],
        st.pos[blk.b[1]],
        st.pos[blk.b[2]],
        st.pos[blk.b[3]],
    ]);
    let (zero, one) = (f32x4::splat(0.0), f32x4::splat(1.0));
    let both_dyn = ba.dyn_mask & bb.dyn_mask;
    let sf_bias = both_dyn.blend(
        f32x4::splat(p.soft.bias_rate),
        f32x4::splat(p.static_soft.bias_rate),
    );
    let sf_mass = both_dyn.blend(
        f32x4::splat(p.soft.mass_scale),
        f32x4::splat(p.static_soft.mass_scale),
    );
    let sf_imp = both_dyn.blend(
        f32x4::splat(p.soft.impulse_scale),
        f32x4::splat(p.static_soft.impulse_scale),
    );
    let inv_h = f32x4::splat(p.inv_h);
    let neg_max_bias = f32x4::splat(-p.max_bias);
    let (normal, t0, t1) = (blk.normal, blk.tangents[0], blk.tangents[1]);
    let friction = blk.friction;
    for pt in blk.points.iter_mut().take(blk.n_points) {
        let (ra, rb, has) = (pt.ra, pt.rb, pt.has);
        // s = normal·(pa − pb) + base
        let s = normal.dot((pos_a + ra) - (pos_b + rb)) + pt.base_separation;
        // --- 법선 ---
        let vn = normal.dot(ba.vel_at(ra) - bb.vel_at(rb));
        let s_pos = s.cmp_gt(zero);
        let (bias_else, mass_else, imp_else) = if p.relax {
            (zero, one, zero)
        } else {
            ((sf_bias * s).max(neg_max_bias), sf_mass, sf_imp)
        };
        let bias = s_pos.blend(s * inv_h, bias_else);
        let mass_scale = s_pos.blend(one, mass_else);
        let impulse_scale = s_pos.blend(zero, imp_else);
        let impulse =
            -pt.normal_mass * mass_scale * (vn + bias) - impulse_scale * pt.normal_impulse;
        let new_total = (pt.normal_impulse + impulse).max(zero);
        let applied = new_total - pt.normal_impulse;
        pt.normal_impulse = new_total;
        pt.max_normal_impulse = pt.max_normal_impulse.max(new_total);
        let imp = normal.scale(applied);
        ba.apply(imp, ra, has);
        bb.apply(neg(imp), rb, has);
        // --- 마찰: 축별 순차 GS 후 Coulomb 원판 ---
        let max_f = friction * pt.normal_impulse;
        for axis in 0..2 {
            let t = if axis == 0 { t0 } else { t1 };
            let vt = t.dot(ba.vel_at(ra) - bb.vel_at(rb));
            let delta = -pt.tangent_mass[axis] * vt;
            pt.tangent_impulse[axis] += delta;
            let imp = t.scale(delta);
            ba.apply(imp, ra, has);
            bb.apply(neg(imp), rb, has);
        }
        let mag2 = pt.tangent_impulse[0] * pt.tangent_impulse[0]
            + pt.tangent_impulse[1] * pt.tangent_impulse[1];
        let cond = mag2.cmp_gt(max_f * max_f) & mag2.cmp_gt(zero);
        let scale = max_f / mag2.sqrt();
        let corr0 = pt.tangent_impulse[0] * (scale - one);
        let corr1 = pt.tangent_impulse[1] * (scale - one);
        pt.tangent_impulse[0] = cond.blend(pt.tangent_impulse[0] + corr0, pt.tangent_impulse[0]);
        pt.tangent_impulse[1] = cond.blend(pt.tangent_impulse[1] + corr1, pt.tangent_impulse[1]);
        let imp = t0.scale(corr0) + t1.scale(corr1);
        ba.apply(imp, ra, has & cond);
        bb.apply(neg(imp), rb, has & cond);
    }
    ba.store(st, blk.a);
    bb.store(st, blk.b);
}

/// 서브스텝 루프가 훑는 **작업 순서** 한 칸.
///
/// 블록과 스칼라 잔여를 따로 모아 "블록 전부 → 스칼라 전부"로 돌면 **색 사이 순서가 바뀐다**.
/// 같은 색 안은 바디가 겹치지 않아 순서 무관이지만, 다른 색끼리는 바디를 공유하므로
/// 순서가 결과를 바꾼다(Gauss-Seidel). 그래서 `solve_batched`가 내던 순서 —
/// 색 오름차순으로 4제약 청크, 색 끝의 꼬리 — 를 그대로 보존한다.
#[derive(Clone, Copy)]
pub enum Work {
    /// `blocks[i]`
    Block(u32),
    /// `constraints[start..end]` (스칼라)
    Scalar(u32, u32),
}

/// 색 범위를 **블록 + 스칼라 잔여**의 순서 있는 작업 목록으로 나눈다.
///
/// 분할 규칙은 `solver::solve_batched`의 `chunks_mut(4)`와 정확히 같다: 색마다 앞에서부터
/// 4개씩 블록, 남은 꼬리는 스칼라, 오버플로 색(공유 바디 가능)은 전부 스칼라.
/// 반환한 `Block(i)`의 `i`는 조립 순서대로 매겨진 블록 인덱스이고,
/// `starts[i]`가 그 블록의 `constraints` 시작 인덱스다(임펄스 되쓰기용).
pub fn plan(
    constraints: &[ContactConstraint],
    color_ranges: &[(u32, u32)],
    work: &mut Vec<Work>,
    starts: &mut Vec<usize>,
) {
    let _ = constraints;
    for (color, &(start, end)) in color_ranges.iter().enumerate() {
        let (start, end) = (start as usize, end as usize);
        if color as u32 == COLOR_OVERFLOW {
            if start < end {
                work.push(Work::Scalar(start as u32, end as u32));
            }
            continue;
        }
        let mut i = start;
        while i + 4 <= end {
            work.push(Work::Block(starts.len() as u32));
            starts.push(i);
            i += 4;
        }
        if i < end {
            work.push(Work::Scalar(i as u32, end as u32));
        }
    }
}

/// 앵커 갱신 — 블록은 4-wide, 스칼라 잔여는 기존 경로. 순서는 무관하다(읽기만 한다).
pub fn refresh_anchors_all(
    st: &SolverState,
    blocks: &mut [ContactBlock],
    constraints: &mut [ContactConstraint],
    work: &[Work],
) {
    for w in work {
        match *w {
            Work::Block(i) => refresh_anchors_block(st, &mut blocks[i as usize]),
            Work::Scalar(s, e) => {
                for c in &mut constraints[s as usize..e as usize] {
                    crate::solver::refresh_anchors(st, c);
                }
            }
        }
    }
}

/// warm start — `solve_batched`와 같은 순서로 블록/스칼라를 훑는다.
pub fn warm_start_all(
    st: &mut SolverState,
    blocks: &[ContactBlock],
    constraints: &[ContactConstraint],
    work: &[Work],
) {
    for w in work {
        match *w {
            Work::Block(i) => warm_start_block(st, &blocks[i as usize]),
            Work::Scalar(s, e) => {
                for c in &constraints[s as usize..e as usize] {
                    warm_start_scalar(st, c);
                }
            }
        }
    }
}

/// solve — 같은 순서 보존.
pub fn solve_all(
    st: &mut SolverState,
    blocks: &mut [ContactBlock],
    constraints: &mut [ContactConstraint],
    work: &[Work],
    p: &BatchSolveParams,
) {
    for w in work {
        match *w {
            Work::Block(i) => solve_block(st, &mut blocks[i as usize], p),
            Work::Scalar(s, e) => {
                for c in &mut constraints[s as usize..e as usize] {
                    // softness 선택은 `solve_batched`의 스칼라 분기와 동일하다.
                    let soft = if p.relax {
                        None
                    } else if st.dynamic[c.a] && st.dynamic[c.b] {
                        Some((&p.soft, p.max_bias))
                    } else {
                        Some((&p.static_soft, p.max_bias))
                    };
                    solve_scalar(st, c, p.inv_h, soft);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{
        solve_batched, warm_start_batched, ContactPointConstraint, Softness, COLOR_OVERFLOW,
    };
    use maru_collide::{FeatureId, FEATURE_GENERIC};
    use maru_math::{Mat3, Quat};

    /// 블록 경로는 **기존 x4 경로의 재배치일 뿐**이라는 계약을 잠근다.
    ///
    /// SoA 조립은 성능 변경이지 물리 변경이 아니다 — 그러려면 색 처리 순서(색 오름차순,
    /// 색 안에서 4개 청크 → 꼬리)까지 같아야 한다. 순서가 어긋나면 다른 색끼리 바디를
    /// 공유하는 곳에서 Gauss-Seidel 결과가 갈린다(실제로 처음 구현이 그렇게 틀렸고,
    /// 증분 브로드페이즈 궤적 일치 테스트가 그것을 잡았다).
    #[allow(clippy::field_reassign_with_default)]
    fn mk_state(n: usize) -> SolverState {
        let mut st = SolverState::default();
        st.lin = (0..n)
            .map(|i| Vec3::new(i as f32 * 0.13, -(i as f32) * 0.07, 0.31))
            .collect();
        st.ang = (0..n)
            .map(|i| Vec3::new(0.021 * i as f32, -0.04, -0.011 * i as f32))
            .collect();
        st.pos = (0..n)
            .map(|i| Vec3::new(0.5 * i as f32, 1.0 + 0.25 * i as f32, -0.3))
            .collect();
        st.rot = (0..n)
            .map(|i| Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 0.11 * i as f32))
            .collect();
        st.inv_m = (0..n).map(|i| 0.5 + 0.1 * i as f32).collect();
        st.inv_i = (0..n)
            .map(|i| {
                Mat3::from_diagonal(Vec3::new(1.0 + 0.1 * i as f32, 2.0, 0.5 + 0.05 * i as f32))
            })
            .collect();
        // 일부는 static — dynamic 게이트가 블록/스칼라 양쪽에서 같게 걸리는지 본다.
        st.dynamic = (0..n).map(|i| i % 7 != 3).collect();
        st
    }

    fn mk_point(seed: f32) -> ContactPointConstraint {
        ContactPointConstraint {
            feature: FeatureId(FEATURE_GENERIC),
            position: Vec3::new(seed, 0.2, -0.1),
            anchor_a: Vec3::new(0.2 + 0.1 * seed, -0.13, 0.05 * seed),
            anchor_b: Vec3::new(-0.15, 0.3 * seed, 0.2),
            base_separation: -0.004 * seed,
            normal_mass: 0.7 + 0.05 * seed,
            tangent_mass: [0.6 + 0.03 * seed, 0.55],
            normal_impulse: 0.5 + 0.2 * seed,
            tangent_impulse: [0.1 * seed, -0.2 * seed],
            max_normal_impulse: 0.0,
            ra: Vec3::ZERO,
            rb: Vec3::ZERO,
            approach_vn: -0.3,
        }
    }

    /// 색 구성: 4의 배수가 아닌 꼬리 · 접점 수가 레인마다 다름 · 오버플로 색 포함.
    fn mk_constraints() -> (Vec<ContactConstraint>, Vec<(u32, u32)>) {
        let mut cs = Vec::new();
        // 색 0: 6개(블록 1 + 꼬리 2), 색 1: 4개(블록 1), 색 63(오버플로): 3개 스칼라
        let counts = [6usize, 4, 3];
        let mut k = 0usize;
        for (ci, &cnt) in counts.iter().enumerate() {
            for j in 0..cnt {
                let a = k % 11;
                let b = (k * 3 + 5) % 11;
                let npts = 1 + (k % 4);
                cs.push(ContactConstraint {
                    a,
                    b: if b == a { (b + 1) % 11 } else { b },
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    tangents: [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)],
                    friction: 0.4 + 0.05 * ci as f32,
                    restitution: 0.0,
                    points: (0..npts)
                        .map(|p| mk_point(k as f32 + 0.3 * p as f32))
                        .collect(),
                    warm_matched: 0,
                });
                k += 1;
                let _ = j;
            }
        }
        // 색 범위: 0..6, 6..10, 그리고 오버플로 색 인덱스까지 빈 색으로 채운다.
        let mut ranges = vec![(0u32, 6u32), (6, 10)];
        while ranges.len() < COLOR_OVERFLOW as usize {
            let e = ranges.last().unwrap().1;
            ranges.push((e, e));
        }
        ranges.push((10, 13)); // 색 63 = 오버플로
        (cs, ranges)
    }

    fn params(relax: bool) -> BatchSolveParams {
        BatchSolveParams {
            inv_h: 240.0,
            relax,
            soft: Softness::new(30.0, 10.0, 1.0 / 240.0),
            static_soft: Softness::new(60.0, 10.0, 1.0 / 240.0),
            max_bias: 4.0,
        }
    }

    #[test]
    fn block_path_matches_x4_path_bit_for_bit() {
        let (mut cs_ref, ranges) = mk_constraints();
        let mut cs_blk = cs_ref.clone();
        let mut st_ref = mk_state(11);
        let mut st_blk = mk_state(11);

        // 기준: 기존 x4 배치 경로 (색 순서대로 4청크 → 꼬리)
        for _ in 0..3 {
            for c in cs_ref.iter_mut() {
                crate::solver::refresh_anchors(&st_ref, c);
            }
            warm_start_batched(&mut st_ref, &cs_ref, &ranges);
            solve_batched(&mut st_ref, &mut cs_ref, &ranges, &params(false));
            solve_batched(&mut st_ref, &mut cs_ref, &ranges, &params(true));
        }

        // 대상: SoA 블록 경로
        let mut work = Vec::new();
        let mut starts = Vec::new();
        plan(&cs_blk, &ranges, &mut work, &mut starts);
        let mut blocks = vec![ContactBlock::default(); starts.len()];
        for (bi, &s) in starts.iter().enumerate() {
            blocks[bi].assemble_into([&cs_blk[s], &cs_blk[s + 1], &cs_blk[s + 2], &cs_blk[s + 3]]);
        }
        assert!(
            !blocks.is_empty(),
            "블록이 안 생기면 이 테스트는 무의미하다"
        );
        assert!(
            work.iter().any(|w| matches!(w, Work::Scalar(..))),
            "스칼라 꼬리가 없으면 혼합 순서를 검증하지 못한다"
        );
        for _ in 0..3 {
            refresh_anchors_all(&st_blk, &mut blocks, &mut cs_blk, &work);
            warm_start_all(&mut st_blk, &blocks, &cs_blk, &work);
            solve_all(&mut st_blk, &mut blocks, &mut cs_blk, &work, &params(false));
            solve_all(&mut st_blk, &mut blocks, &mut cs_blk, &work, &params(true));
        }
        for (bi, &s) in starts.iter().enumerate() {
            let (l, r) = cs_blk.split_at_mut(s + 1);
            let (c1, c2) = r.split_at_mut(1);
            let (c3, c4) = c2.split_at_mut(1);
            blocks[bi].scatter_impulses([&mut l[s], &mut c1[0], &mut c3[0], &mut c4[0]]);
        }

        for i in 0..11 {
            assert_eq!(
                st_ref.lin[i].to_bits_triple(),
                st_blk.lin[i].to_bits_triple(),
                "바디 {i} 선속도가 비트 동일하지 않다"
            );
            assert_eq!(
                st_ref.ang[i].to_bits_triple(),
                st_blk.ang[i].to_bits_triple(),
                "바디 {i} 각속도가 비트 동일하지 않다"
            );
        }
        for (ci, (a, b)) in cs_ref.iter().zip(cs_blk.iter()).enumerate() {
            for (pi, (pa, pb)) in a.points.iter().zip(b.points.iter()).enumerate() {
                assert_eq!(
                    pa.normal_impulse.to_bits(),
                    pb.normal_impulse.to_bits(),
                    "제약 {ci} 접점 {pi} 법선 임펄스 불일치"
                );
                assert_eq!(
                    pa.tangent_impulse[0].to_bits(),
                    pb.tangent_impulse[0].to_bits(),
                    "제약 {ci} 접점 {pi} 접선0 임펄스 불일치"
                );
                assert_eq!(
                    pa.tangent_impulse[1].to_bits(),
                    pb.tangent_impulse[1].to_bits(),
                    "제약 {ci} 접점 {pi} 접선1 임펄스 불일치"
                );
                assert_eq!(
                    pa.max_normal_impulse.to_bits(),
                    pb.max_normal_impulse.to_bits(),
                    "제약 {ci} 접점 {pi} max 임펄스 불일치"
                );
            }
        }
    }

    /// `Quatx4::rotate`가 스칼라 `Quat::rotate`와 레인당 비트 동일한지 — 앵커 갱신의 전제.
    #[test]
    fn quatx4_rotate_matches_scalar_bit_for_bit() {
        let qs = [
            Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 0.7),
            Quat::from_axis_angle(Vec3::new(-1.0, 0.4, 0.9), -1.3),
            Quat::IDENTITY,
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 2.9),
        ];
        let vs = [
            Vec3::new(0.31, -0.72, 1.4),
            Vec3::new(-2.0, 0.05, 0.9),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(11.3, -4.1, 0.02),
        ];
        let out = Quatx4::from_array(qs).rotate(Vec3x4::from_array(vs));
        let lanes_out = lanes(out);
        for k in 0..4 {
            let want = qs[k].rotate(vs[k]);
            assert_eq!(
                want.to_bits_triple(),
                lanes_out[k].to_bits_triple(),
                "레인 {k} 회전이 스칼라와 다르다"
            );
        }
    }

    trait Bits {
        fn to_bits_triple(&self) -> (u32, u32, u32);
    }
    impl Bits for Vec3 {
        fn to_bits_triple(&self) -> (u32, u32, u32) {
            (self.x.to_bits(), self.y.to_bits(), self.z.to_bits())
        }
    }
}
