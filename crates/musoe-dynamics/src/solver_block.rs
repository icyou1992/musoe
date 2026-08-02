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
    SolveView, COLOR_OVERFLOW,
};
use crate::threads::{DisjointSlice, Pool};
use musoe_math::{Quatx4, Vec3, Vec3x4};
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
    inv_i: musoe_math::Mat3x4,
    dyn_mask: f32x4,
}

impl BodyRegs {
    #[inline]
    fn load(v: &SolveView, idx: [usize; 4]) -> Self {
        Self {
            lin: Vec3x4::from_array([v.lin(idx[0]), v.lin(idx[1]), v.lin(idx[2]), v.lin(idx[3])]),
            ang: Vec3x4::from_array([v.ang(idx[0]), v.ang(idx[1]), v.ang(idx[2]), v.ang(idx[3])]),
            inv_m: f32x4::from([
                v.inv_m(idx[0]),
                v.inv_m(idx[1]),
                v.inv_m(idx[2]),
                v.inv_m(idx[3]),
            ]),
            inv_i: musoe_math::Mat3x4::from_array([
                v.inv_i(idx[0]),
                v.inv_i(idx[1]),
                v.inv_i(idx[2]),
                v.inv_i(idx[3]),
            ]),
            dyn_mask: f32x4::from([
                v.dynamic(idx[0]) as u32 as f32,
                v.dynamic(idx[1]) as u32 as f32,
                v.dynamic(idx[2]) as u32 as f32,
                v.dynamic(idx[3]) as u32 as f32,
            ])
            .cmp_gt(f32x4::splat(0.5)),
        }
    }

    #[inline]
    fn store(&self, v: &SolveView, idx: [usize; 4]) {
        let (lin, ang) = (lanes(self.lin), lanes(self.ang));
        for k in 0..4 {
            if v.dynamic(idx[k]) {
                v.set_lin(idx[k], lin[k]);
                v.set_ang(idx[k], ang[k]);
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
pub fn refresh_anchors_block(v: &SolveView, blk: &mut ContactBlock) {
    let qa = Quatx4::from_array([
        v.rot(blk.a[0]),
        v.rot(blk.a[1]),
        v.rot(blk.a[2]),
        v.rot(blk.a[3]),
    ]);
    let qb = Quatx4::from_array([
        v.rot(blk.b[0]),
        v.rot(blk.b[1]),
        v.rot(blk.b[2]),
        v.rot(blk.b[3]),
    ]);
    for p in blk.points.iter_mut().take(blk.n_points) {
        p.ra = qa.rotate(p.anchor_a);
        p.rb = qb.rotate(p.anchor_b);
    }
}

/// warm start — 블록판. 레인당 스칼라 `warm_start`와 비트 동일.
pub fn warm_start_block(v: &SolveView, blk: &ContactBlock) {
    let mut ba = BodyRegs::load(v, blk.a);
    let mut bb = BodyRegs::load(v, blk.b);
    for p in blk.points.iter().take(blk.n_points) {
        let imp = blk.normal.scale(p.normal_impulse)
            + blk.tangents[0].scale(p.tangent_impulse[0])
            + blk.tangents[1].scale(p.tangent_impulse[1]);
        ba.apply(imp, p.ra, p.has);
        bb.apply(neg(imp), p.rb, p.has);
    }
    ba.store(v, blk.a);
    bb.store(v, blk.b);
}

/// solve — 블록판. `solver::solve_x4`와 **같은 산술·같은 순서**이고 게더만 없다.
/// **게더 미스 비용 계측 노브** (`MARU_SOLVE_PROBE`, `solve-probe` 피처 뒤 — 기본 off).
///
/// 재려는 것: `bias`/`relax` 루프가 (a) 흩어진 바디 레지스터 게더의 **캐시 미스** 지배인지
/// (b) 산술 지배인지. 답에 따라 유효한 수단이 갈린다 — 미스 지배면 제약별 바디 인덱스
/// 재정렬(지역성), 산술 지배면 반복 횟수·수렴 쪽이다.
///
/// 설계: 진짜 게더를 **한 번 더** 하되 결과를 버린다. 두 팔의 차이는 그 추가 게더가
/// 어디를 읽느냐뿐이다.
/// - `1`: **같은 인덱스**(방금 읽어 L1에 있다) → 추가 비용 ≈ 순수 명령어 비용
/// - `2`: **어긋난 인덱스**(다른 캐시 라인) → 추가 비용 ≈ 명령어 + 미스
///
/// `2 − 1`이 곧 **흩어진 바디 게더 1회의 미스 비용**이다. 결과를 버리므로 **물리는 세 팔
/// 모두 진짜 그대로**다(PCM 프로브와 달리 궤적을 망가뜨리지 않는다) — 그래서 대조군 걱정 없이
/// 실제 장면에서 잰다. 실제 `solve_block`은 게더를 2회(a·b) 하므로 상한은 `2 × (2−1)`이다.
///
/// **실측 (2026-07-31, 피라미드 40단·수면 OFF·1스레드, 각 팔 12회 교차실행):**
/// `2 − 1` = `bias+relax`의 **1.50%**, 쌍대 우세 139/144.
///
/// **이 숫자를 "미스 비용"이라고 부르지 마라 — 상한이다** (교차검수 rev_be09acdc MAJOR).
/// 팔 1의 추가 게더는 방금 읽은 것과 같은 인덱스라 LLVM이 CSE로 **통째로 지울 수 있다**.
/// 그러면 `2 − 1`에는 미스뿐 아니라 게더 명령어 비용 전체가 섞인다. 방향은 안전하다 —
/// 섞여 있다는 것은 진짜 미스 비용이 1.50%**보다 작다**는 뜻이다. 마찬가지로
/// "블록당 2회이므로 2배"도 **상한이 아니다**: 실제 `blk.b`는 지면 같은 반복 바디를 물어
/// 훨씬 뜨거울 수도, 더 찰 수도 있고 두 게더가 OoO로 겹치기도 한다. 대조군
/// (`warm+intv`)도 완전히 독립적이지 않다 — `warm_start_block`이 같은 `BodyRegs`를 읽는다.
/// 이 머신(WSL2)에는 `perf`가 없어 PMU 카운터로 가를 수 없었다.
///
/// **그래서 배치 문제의 판정은 이 프로브가 아니라 `pyramid100`의
/// `MARU_SHUFFLE_BODIES` A/B가 한다** — 삽입 순서를 통째로 섞어 `pos`/`rot` 게더·
/// lin/ang 스토어·`warm`·`refresh`까지 **배치의 영향을 받는 전부**를 한 번에 잰다.
/// 그 결과가 정본이다:
/// - 40단(821바디, SoA 작업집합 ~75 KB): 셔플 대 자연 **중앙값 −0.35%, 쌍대 47/100** —
///   측정 불가. 최악 배치조차 공짜다.
/// - 150단(11,326바디, ~1,029 KB ≈ L2 1 MiB 경계): 셔플이 **+15.2%, 쌍대 36/36**.
///
/// 즉 배치는 **작업집합이 L2를 넘을 때만** 문제다. 표적인 40단은 한참 안쪽이므로
/// **인덱스 재정렬은 이 장면에서 기각**이고, 반대로 만 단위 바디 장면에서는 실재하는
/// 레버다(그때 다시 꺼낼 것).
///
/// 규모 스윕도 같은 얘기를 한다 — 단 정규화는 `work`가 아니라 **접점 수**여야 한다
/// (`Work::Block`은 제약 4개에 접점 1~4개, `Work::Scalar`는 가변이라 `work`는 동질 단위가
/// 아니다 — 같은 검수 MAJOR). 접점당 `bias`(ns):
/// `3KB 122 · 10KB 118 · 19KB 114 · 37KB 119 · 54KB 115 · 75KB 118 · 166KB 124 · 372KB 132`
/// — 75 KB까지 평평하고 그 뒤로 완만히 오른다. 셔플 A/B의 경계와 일치한다.
///
/// 남는 결론: **40단의 `bias`/`relax`는 배치가 아니라 산술·의존 사슬 지배**다.
#[cfg(feature = "solve-probe")]
fn solve_probe_mode() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(u8::MAX);
    let c = CACHED.load(Ordering::Relaxed);
    if c != u8::MAX {
        return c;
    }
    let raw = std::env::var("MARU_SOLVE_PROBE").unwrap_or_else(|_| "0".into());
    let m = match raw.trim().parse::<u8>() {
        Ok(m @ 0..=4) => m,
        _ => panic!("MARU_SOLVE_PROBE={raw:?} — 지원하는 값은 0·1·2·3·4뿐이다"),
    };
    if m != 0 {
        eprintln!("!! MARU_SOLVE_PROBE={m} — 계측 전용 팔이다(추가 게더 1회, 결과는 버린다).");
    }
    CACHED.store(m, Ordering::Relaxed);
    m
}

pub fn solve_block(v: &SolveView, blk: &mut ContactBlock, p: &BatchSolveParams) {
    let mut ba = BodyRegs::load(v, blk.a);
    let mut bb = BodyRegs::load(v, blk.b);
    #[cfg(feature = "solve-probe")]
    {
        // 팔 1·2는 **결과를 버린다** — 물리는 손대지 않는다.
        let m = solve_probe_mode();
        if m == 1 || m == 2 {
            let n = v.len();
            // 어긋난 인덱스는 **같은 배열 안**에 있어야 한다(다른 배열을 읽으면 무엇을
            // 재는지 흐려진다). n/2 만큼 밀어 다른 캐시 라인으로 보낸다.
            let idx = if m == 1 {
                blk.a
            } else {
                [
                    (blk.a[0] + n / 2) % n,
                    (blk.a[1] + n / 2) % n,
                    (blk.a[2] + n / 2) % n,
                    (blk.a[3] + n / 2) % n,
                ]
            };
            core::hint::black_box(BodyRegs::load(v, idx));
        }
    }
    let pos_a = Vec3x4::from_array([
        v.pos(blk.a[0]),
        v.pos(blk.a[1]),
        v.pos(blk.a[2]),
        v.pos(blk.a[3]),
    ]);
    let pos_b = Vec3x4::from_array([
        v.pos(blk.b[0]),
        v.pos(blk.b[1]),
        v.pos(blk.b[2]),
        v.pos(blk.b[3]),
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
        // 팔 `3` — **apply 한 쌍의 비용**을 잰다. 임펄스가 0이라 물리는 그대로다
        // (`lin + 0*inv_m = lin`, `ang + inv_i·(r×0) = ang`). `black_box`로 0을 불투명하게
        // 만들어야 옵티마이저가 통째로 지우지 않는다.
        //
        // 왜 이걸 재나: 접점마다 apply가 **4쌍** 돈다 — 법선 1 · 마찰 2축 각 1 · Coulomb 보정 1.
        // 마찰 두 축을 2x2 블록으로 동시에 풀고 Coulomb 클램프를 같은 apply에 접으면 2쌍이
        // 사라진다. 그 구조 변경의 상한이 `2 × (팔4 − 팔3)`이다 — 짓기 전에 재려고 둔다.
        //
        // **실측 (2026-07-31, 피라미드 40단·수면 OFF·1스레드, 각 팔 12회 교차실행,
        //   대조군 = narrow):** apply 1쌍의 한계비용 = `bias+relax`의 **8.04%**,
        //   쌍대 144/144(완전 분리). → 2쌍 제거의 상한 ≈ 16% of bias+relax ≈ **스텝의 5.2%**.
        //
        // 다른 두 수치를 상한/하한으로 함께 남긴다(어느 쪽도 순수하지 않다):
        // - 진짜 `ba`/`bb`에 제자리로 걸었을 때 6.04% — **과소**. 옵티마이저가 일부를 접을 수
        //   있고, 무엇보다 `-0.0 + 0.0 = +0.0`이라 비트 항등이 아니었다(검수 rev_fa8f70b2).
        // - 복사본을 만들어 한 쌍만 걸었을 때 18.32% — **과대**. 복사본을 살려 두는 레지스터
        //   압박이 일회성으로 섞인다. 한계비용은 그 일회성을 양쪽에서 상쇄한다.
        #[cfg(feature = "solve-probe")]
        if solve_probe_mode() >= 3 {
            // **진짜 `ba`/`bb`가 아니라 복사본에 건다**(교차검수 rev_fa8f70b2 MAJOR).
            // 0 임펄스는 수학적으로 항등이지만 **비트 항등이 아니다**: `apply`는
            // `lin + imp*inv_m`을 계산해 store하므로 `-0.0` 성분이 `+0.0`이 된다.
            // 그러면 vmax·top_y는 같아도 상태 해시가 달라져 "물리 불변" 전제가 깨진다.
            // `BodyRegs`는 `Copy`라 복사는 레지스터 안에서 끝나고, 복사본을 통째로
            // `black_box`로 소비해 계산이 지워지지 않게 한다.
            let z = core::hint::black_box(Vec3x4::from_array([musoe_math::Vec3::ZERO; 4]));
            //
            // 팔 `4`는 **같은 복사본에 한 쌍을 더** 건다. `4 − 3`이 복사·스필 같은 일회성
            // 비용을 뺀 **apply 한 쌍의 한계비용**이다 — `3 − 0`은 복사본을 살려 두는
            // 레지스터 압박까지 물어 과대평가한다(6.04% → 18.32%로 뛴 것이 그 증거다).
            // 이 작업은 ALU·레지스터라 캐시 지배가 아니므로 한계비용이 옳은 추정자다.
            let (mut pa, mut pb) = (ba, bb);
            pa.apply(z, ra, has);
            pb.apply(neg(z), rb, has);
            if solve_probe_mode() == 4 {
                pa.apply(z, ra, has);
                pb.apply(neg(z), rb, has);
            }
            core::hint::black_box((pa, pb));
        }
        // --- 마찰: 접선 2x2 블록 해 + Coulomb 원판 투영을 **한 번의 적용**에 접는다 ---
        // 스칼라 `solver::solve`와 **연산 순서까지 같은 식**이다(비트 동일 계약).
        // 클램프는 분기가 아니라 blend다 — 레인마다 조건이 다르다.
        let max_f = friction * pt.normal_impulse;
        let dv = ba.vel_at(ra) - bb.vel_at(rb);
        let (vt0, vt1) = (t0.dot(dv), t1.dot(dv));
        // 핵심 산술은 `lane::friction_disc` 하나다 — 스칼라 경로와 **같은 코드**를 쓴다.
        let (acc, d) =
            crate::lane::friction_disc(pt.tangent_impulse, pt.tangent_mass, [vt0, vt1], max_f);
        let (d0, d1) = (d[0], d[1]);
        pt.tangent_impulse = acc;
        let imp = t0.scale(d0) + t1.scale(d1);
        ba.apply(imp, ra, has);
        bb.apply(neg(imp), rb, has);
    }
    ba.store(v, blk.a);
    bb.store(v, blk.b);
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
    color_work: &mut Vec<(u32, u32)>,
) {
    let _ = constraints;
    for (color, &(start, end)) in color_ranges.iter().enumerate() {
        let (start, end) = (start as usize, end as usize);
        // **빈 색은 색 목록에 넣지 않는다.** color_work 항목 하나가 배리어 하나이므로 빈 색은
        // "아무 일도 없는 동기화"다. 색 상한을 쓰면(`COLOR_CAP_DEFAULT` 주석) 오버플로 색이
        // 63번이라 그 사이 색들이 통째로 비는데, 그것들을 넣으면 배리어가 색 상한이 아니라
        // **63개**로 늘어 상한을 도입한 목적이 정확히 반대로 뒤집힌다.
        if start >= end {
            continue;
        }
        // 색마다 **자기 work 구간**을 남긴다 — 병렬 솔브가 색 경계에서 배리어를 치려면
        // "이 색의 일" 목록이 필요하다. 색 안은 순서 무관(공유 dynamic 바디 없음),
        // 색 사이는 순서 필수(Gauss-Seidel)라는 비대칭이 이 자료구조의 존재 이유다.
        let w0 = work.len() as u32;
        if color as u32 == COLOR_OVERFLOW {
            if start < end {
                // 오버플로 색은 바디를 공유할 수 있으므로 **쪼개지 않는다** — 항목 1개로
                // 남겨 한 참여자가 통째로 순서대로 푼다.
                work.push(Work::Scalar(start as u32, end as u32));
            }
            color_work.push((w0, work.len() as u32));
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
        color_work.push((w0, work.len() as u32));
    }
}

/// 병렬로 돌릴 만한 일감인가. 문턱 아래면 dispatch 비용이 이득보다 크다 —
/// 어느 쪽이든 **결과는 같다**(문턱도 스위치도 성능 노브일 뿐 시맨틱이 아니다.
/// 그 사실을 `parallel_color_solve_matches_serial_bit_for_bit`가 잠근다).
#[inline]
fn parallel_worth_it(pool: &Pool, work: &[Work], enabled: bool) -> bool {
    enabled && pool.threads() > 1 && work.len() >= SOLVE_PARALLEL_MIN_WORK
}

/// 색 병렬을 켜는 work 항목 수 문턱.
pub const SOLVE_PARALLEL_MIN_WORK: usize = 64;

/// 이번 스텝 솔브 패스를 실제로 나눠 돌 참여자 수 (1 = 직렬). 진단 보고용 —
/// **판정식은 실행 경로와 같은 정본**을 쓴다(따로 적으면 보고와 실제가 갈린다).
pub fn solve_participants(pool: &Pool, parallel: bool, work: &[Work]) -> usize {
    if parallel_worth_it(pool, work, parallel) {
        pool.threads()
    } else {
        1
    }
}

/// 색 단위 병렬 실행 — 이 모듈의 병렬 구조 **단일 정본**이다 (설계 §5).
///
/// 각 참여자가 색 안의 work 항목을 스트라이드로 나눠 갖고, 색이 끝날 때마다 배리어에서
/// 만난다. 색 안이 순서 무관인 근거는 컬러링 계약(같은 색 = dynamic 바디 비공유)이고,
/// 색 사이 순서가 지켜지는 근거는 배리어다. 그래서 결과는 참여자 수와 무관하다.
///
/// 배리어는 **모든 참여자가 같은 횟수** 쳐야 한다 — 자기 몫이 0개인 색에서도 루프를
/// 돌기 때문에 자연히 지켜진다.
#[inline]
fn run_colors<F: Fn(usize) + Sync>(pool: &Pool, work_len: usize, color_work: &[(u32, u32)], f: F) {
    debug_assert_eq!(
        color_work
            .iter()
            .map(|&(s, e)| (e - s) as usize)
            .sum::<usize>(),
        work_len,
        "색별 work 구간이 work 전체를 분할하지 않는다"
    );
    let _ = work_len;
    pool.dispatch(|ctx| {
        for &(ws, we) in color_work {
            let mut i = ws as usize + ctx.index();
            while i < we as usize {
                f(i);
                i += ctx.threads();
            }
            ctx.barrier();
        }
    });
}

/// 앵커 갱신 — 블록은 4-wide, 스칼라 잔여는 기존 경로. 순서는 무관하다(읽기만 한다).
pub fn refresh_anchors_all(
    pool: &Pool,
    parallel: bool,
    v: &SolveView,
    blocks: &mut [ContactBlock],
    constraints: &mut [ContactConstraint],
    work: &[Work],
    color_work: &[(u32, u32)],
) {
    let blocks = DisjointSlice::new(blocks);
    let constraints = DisjointSlice::new(constraints);
    // work 항목 하나가 만지는 블록/제약 구간은 다른 항목과 겹치지 않는다(plan의 분할).
    let one = |i: usize| match work[i] {
        // SAFETY: 블록 인덱스는 work 항목마다 유일하다(plan이 조립 순서대로 부여).
        Work::Block(b) => refresh_anchors_block(v, unsafe { blocks.get(b as usize) }),
        // SAFETY: 스칼라 구간은 색 안에서 서로 겹치지 않는 [s,e)다.
        Work::Scalar(s, e) => {
            for c in unsafe { constraints.range(s as usize, e as usize) } {
                crate::solver::refresh_anchors(v, c);
            }
        }
    };
    if parallel_worth_it(pool, work, parallel) {
        run_colors(pool, work.len(), color_work, one);
    } else {
        for i in 0..work.len() {
            one(i);
        }
    }
}

/// warm start — `solve_batched`와 같은 순서로 블록/스칼라를 훑는다.
pub fn warm_start_all(
    pool: &Pool,
    parallel: bool,
    v: &SolveView,
    blocks: &[ContactBlock],
    constraints: &[ContactConstraint],
    work: &[Work],
    color_work: &[(u32, u32)],
) {
    // 여기서 쓰이는 것은 속도(`v`)뿐이고 블록·제약은 읽기만 한다 — 공유 참조로 충분하다.
    let one = |i: usize| match work[i] {
        Work::Block(b) => warm_start_block(v, &blocks[b as usize]),
        Work::Scalar(s, e) => {
            for c in &constraints[s as usize..e as usize] {
                warm_start_scalar(v, c);
            }
        }
    };
    if parallel_worth_it(pool, work, parallel) {
        run_colors(pool, work.len(), color_work, one);
    } else {
        for i in 0..work.len() {
            one(i);
        }
    }
}

/// solve — 같은 순서 보존.
#[allow(clippy::too_many_arguments)] // 패스 하나가 받는 것: 실행 자원(pool·스위치) +
                                     // 상태 뷰 + 데이터 3종(blocks·constraints·work) + 색 경계 + 솔브 파라미터. 묶을 만한
                                     // 응집 단위가 없어(수명·가변성이 제각각) 구조체화는 이름만 늘린다.
pub fn solve_all(
    pool: &Pool,
    parallel: bool,
    v: &SolveView,
    blocks: &mut [ContactBlock],
    constraints: &mut [ContactConstraint],
    work: &[Work],
    color_work: &[(u32, u32)],
    p: &BatchSolveParams,
) {
    let blocks = DisjointSlice::new(blocks);
    let constraints = DisjointSlice::new(constraints);
    let one = |i: usize| match work[i] {
        // SAFETY: 블록 인덱스는 work 항목마다 유일하다.
        Work::Block(b) => solve_block(v, unsafe { blocks.get(b as usize) }, p),
        // SAFETY: 스칼라 구간은 서로 겹치지 않는다.
        Work::Scalar(s, e) => {
            for c in unsafe { constraints.range(s as usize, e as usize) } {
                // softness 선택은 `solve_batched`의 스칼라 분기와 동일하다.
                let soft = if p.relax {
                    None
                } else if v.dynamic(c.a) && v.dynamic(c.b) {
                    Some((&p.soft, p.max_bias))
                } else {
                    Some((&p.static_soft, p.max_bias))
                };
                solve_scalar(v, c, p.inv_h, soft);
            }
        }
    };
    if parallel_worth_it(pool, work, parallel) {
        run_colors(pool, work.len(), color_work, one);
    } else {
        for i in 0..work.len() {
            one(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{ContactPointConstraint, Softness, SolverState, COLOR_OVERFLOW};
    use musoe_collide::{FeatureId, FEATURE_GENERIC};
    use musoe_math::{Mat3, Quat};

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
                // **색 안에서는 바디가 겹치지 않아야 한다** — 그것이 컬러링 계약이고,
                // 그래야 색 안의 처리 순서가 결과를 안 바꾼다(SIMD 4-wide는 그 전제에서만
                // 스칼라와 같다: 겹치면 4-wide는 Jacobi, 스칼라는 Gauss-Seidel이 되어 갈린다).
                // 예전 픽스처는 `a = k % 11`로 색 안에서 바디를 공유해 **프로덕션에 없는
                // 상황**을 시험하고 있었다. 색마다 바디 풀을 갈라 준다.
                let a = 2 * j;
                let b = 2 * j + 1;
                let npts = 1 + (k % 4);
                cs.push(ContactConstraint {
                    a,
                    b,
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

    /// **스칼라 직접 참조** — 색 순서대로, 색 안에서는 인덱스 순서대로 하나씩 푼다.
    /// 색 안의 제약은 동적 바디가 겹치지 않으므로(컬러링 계약) 처리 순서가 결과를 바꾸지
    /// 않는다 — 그래서 4청크로 묶든 하나씩 풀든 비트 동일이다.
    fn serial_solve(
        st: &mut SolverState,
        cs: &mut [ContactConstraint],
        ranges: &[(u32, u32)],
        p: &BatchSolveParams,
    ) {
        for &(start, end) in ranges {
            for c in cs[start as usize..end as usize].iter_mut() {
                let soft = if p.relax {
                    None
                } else if st.dynamic[c.a] && st.dynamic[c.b] {
                    Some((&p.soft, p.max_bias))
                } else {
                    Some((&p.static_soft, p.max_bias))
                };
                let v = SolveView::new(st);
                solve_scalar(&v, c, p.inv_h, soft);
            }
        }
    }

    fn serial_warm_start(st: &mut SolverState, cs: &[ContactConstraint], ranges: &[(u32, u32)]) {
        for &(start, end) in ranges {
            for c in cs[start as usize..end as usize].iter() {
                let v = SolveView::new(st);
                warm_start_scalar(&v, c);
            }
        }
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

    /// **병렬 색 솔브가 직렬과 비트 동일한지** — 유닛 수준 게이트.
    ///
    /// 월드 수준 결정론 테스트(`thread_determinism_*`)는 파이프라인 전체를 보므로 갈림의
    /// 원인을 못 좁힌다. 여기서는 솔버 패스만 떼어 같은 입력에 참여자 수만 바꿔 돌린다.
    /// 씬은 병렬 문턱(work 64개)을 **넘도록** 만든다 — 안 넘기면 두 번 다 직렬로 돌아
    /// 통과가 아무 의미 없다(그래서 아래에서 work 길이를 assert한다).
    fn colored_scene(colors: usize, per_color: usize) -> (Vec<ContactConstraint>, Vec<(u32, u32)>) {
        let nb = 2 * per_color;
        let mut cs = Vec::new();
        let mut ranges = Vec::new();
        for c in 0..colors {
            let start = cs.len() as u32;
            for k in 0..per_color {
                // 같은 색 안에서는 바디가 겹치지 않고(2k+c, 2k+1+c), 색이 다르면 겹친다 —
                // 색 사이 순서가 결과를 바꾸는 진짜 Gauss-Seidel 커플링을 만든다.
                let a = (2 * k + c) % nb;
                let b = (2 * k + 1 + c) % nb;
                let npts = 1 + (k % 4);
                cs.push(ContactConstraint {
                    a,
                    b,
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    tangents: [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)],
                    friction: 0.4,
                    restitution: 0.0,
                    points: (0..npts)
                        .map(|q| mk_point((k + q + c) as f32 * 0.37))
                        .collect(),
                    warm_matched: 0,
                });
            }
            ranges.push((start, cs.len() as u32));
        }
        (cs, ranges)
    }

    /// 계측용(무시 기본): 색 병렬 솔브가 **패스 자체로는** 스케일하는지.
    ///
    /// 파이프라인 안에서는 dispatch 지연·직렬 구간 경합이 섞여 있어 "안 빨라진다"의
    /// 원인을 못 가른다. 여기서는 워커가 계속 뜨거운 상태로 같은 패스만 반복해
    /// **커널의 병렬 효율 상한**을 잰다.
    /// 실행: cargo test --release -p musoe-dynamics bench_parallel_solve -- --ignored --nocapture
    #[test]
    #[ignore = "계측용"]
    fn bench_parallel_solve_scaling() {
        let (colors, per_color) = (7usize, 300usize);
        let (cs0, ranges) = colored_scene(colors, per_color);
        let st0 = mk_state(2 * per_color);
        for threads in [1usize, 2, 3, 4] {
            let pool = Pool::new(threads);
            let mut st = st0.clone();
            let mut cs = cs0.clone();
            let mut work = Vec::new();
            let mut starts = Vec::new();
            let mut color_work = Vec::new();
            plan(&cs, &ranges, &mut work, &mut starts, &mut color_work);
            let mut blocks = vec![ContactBlock::default(); starts.len()];
            for (bi, &s) in starts.iter().enumerate() {
                blocks[bi].assemble_into([&cs[s], &cs[s + 1], &cs[s + 2], &cs[s + 3]]);
            }
            let reps = 200;
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    solve_all(
                        &pool,
                        true,
                        &SolveView::new(&mut st),
                        &mut blocks,
                        &mut cs,
                        &work,
                        &color_work,
                        &params(false),
                    );
                }
                best = best.min(t.elapsed().as_secs_f64() * 1e6 / reps as f64);
            }
            println!(
                "threads={} (actual {}): solve pass {:.1} us  (work {} items, {} colors)",
                threads,
                pool.threads(),
                best,
                work.len(),
                color_work.len()
            );
        }
    }

    #[test]
    fn parallel_color_solve_matches_serial_bit_for_bit() {
        let (colors, per_color) = (4usize, 64usize);
        let (cs0, ranges) = colored_scene(colors, per_color);
        let st0 = mk_state(2 * per_color);

        let run = |threads: usize| {
            let pool = Pool::new(threads);
            let mut st = st0.clone();
            let mut cs = cs0.clone();
            let mut work = Vec::new();
            let mut starts = Vec::new();
            let mut color_work = Vec::new();
            plan(&cs, &ranges, &mut work, &mut starts, &mut color_work);
            let mut blocks = vec![ContactBlock::default(); starts.len()];
            for (bi, &s) in starts.iter().enumerate() {
                blocks[bi].assemble_into([&cs[s], &cs[s + 1], &cs[s + 2], &cs[s + 3]]);
            }
            for _ in 0..3 {
                refresh_anchors_all(
                    &pool,
                    true,
                    &SolveView::new(&mut st),
                    &mut blocks,
                    &mut cs,
                    &work,
                    &color_work,
                );
                warm_start_all(
                    &pool,
                    true,
                    &SolveView::new(&mut st),
                    &blocks,
                    &cs,
                    &work,
                    &color_work,
                );
                for relax in [false, true] {
                    solve_all(
                        &pool,
                        true,
                        &SolveView::new(&mut st),
                        &mut blocks,
                        &mut cs,
                        &work,
                        &color_work,
                        &params(relax),
                    );
                }
            }
            (st, work.len(), pool.threads())
        };

        let (st_serial, nwork, _) = run(1);
        assert!(
            nwork >= SOLVE_PARALLEL_MIN_WORK,
            "work {nwork}개는 병렬 문턱({SOLVE_PARALLEL_MIN_WORK}) 아래다 — 게이트가 공회전한다"
        );
        for threads in [2usize, 4, 8] {
            let (st_par, _, actual) = run(threads);
            if actual == 1 {
                continue; // 스레드를 못 띄우는 환경
            }
            for i in 0..st_serial.lin.len() {
                assert_eq!(
                    st_serial.lin[i].to_bits_triple(),
                    st_par.lin[i].to_bits_triple(),
                    "threads={threads}: 바디 {i} 선속도가 갈렸다"
                );
                assert_eq!(
                    st_serial.ang[i].to_bits_triple(),
                    st_par.ang[i].to_bits_triple(),
                    "threads={threads}: 바디 {i} 각속도가 갈렸다"
                );
            }
        }
    }

    #[test]
    fn block_path_matches_scalar_bit_for_bit() {
        let (mut cs_ref, ranges) = mk_constraints();
        let mut cs_blk = cs_ref.clone();
        let mut st_ref = mk_state(12);
        let mut st_blk = mk_state(12);

        // 기준: **스칼라 직접 참조** (색 순서대로 하나씩)
        for _ in 0..3 {
            {
                let v = SolveView::new(&mut st_ref);
                for c in cs_ref.iter_mut() {
                    crate::solver::refresh_anchors(&v, c);
                }
            }
            serial_warm_start(&mut st_ref, &cs_ref, &ranges);
            serial_solve(&mut st_ref, &mut cs_ref, &ranges, &params(false));
            serial_solve(&mut st_ref, &mut cs_ref, &ranges, &params(true));
        }

        // 대상: SoA 블록 경로
        let mut work = Vec::new();
        let mut starts = Vec::new();
        let mut color_work = Vec::new();
        plan(&cs_blk, &ranges, &mut work, &mut starts, &mut color_work);
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
        let pool = Pool::new(1);
        for _ in 0..3 {
            refresh_anchors_all(
                &pool,
                true,
                &SolveView::new(&mut st_blk),
                &mut blocks,
                &mut cs_blk,
                &work,
                &color_work,
            );
            warm_start_all(
                &pool,
                true,
                &SolveView::new(&mut st_blk),
                &blocks,
                &cs_blk,
                &work,
                &color_work,
            );
            for relax in [false, true] {
                solve_all(
                    &pool,
                    true,
                    &SolveView::new(&mut st_blk),
                    &mut blocks,
                    &mut cs_blk,
                    &work,
                    &color_work,
                    &params(relax),
                );
            }
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
