//! SoA 4-wide 배치 타입 — 핫패스 정본 레이아웃 (설계 §3).
//!
//! `Vec3x4` = 벡터 4개를 성분별 f32x4로: 모든 연산이 셔플 없는 순수 레인별 SIMD.
//! WASM SIMD128(4×f32)과 정확히 동형 — 설계 §4.1 브로드페이즈 4-ary와 같은 이유.

use crate::{Mat3, Quat, Vec3};
use core::ops::{Add, Sub};
use wide::{f32x4, CmpGe, CmpLe};

#[derive(Clone, Copy, Debug)]
pub struct Vec3x4 {
    pub x: f32x4,
    pub y: f32x4,
    pub z: f32x4,
}

impl Vec3x4 {
    #[inline]
    pub fn splat(v: Vec3) -> Self {
        Self {
            x: f32x4::splat(v.x),
            y: f32x4::splat(v.y),
            z: f32x4::splat(v.z),
        }
    }

    /// AoS 4개 → SoA 1개 (gather는 스칼라 — WASM에 gather 명령 없음, 설계 §3 정직한 범위).
    #[inline]
    pub fn from_array(v: [Vec3; 4]) -> Self {
        Self {
            x: f32x4::from([v[0].x, v[1].x, v[2].x, v[3].x]),
            y: f32x4::from([v[0].y, v[1].y, v[2].y, v[3].y]),
            z: f32x4::from([v[0].z, v[1].z, v[2].z, v[3].z]),
        }
    }

    #[inline]
    pub fn scale(self, s: f32x4) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// 레인별 a + b·s (FMA 비의존 — mul 후 add, 설계 §2).
    #[inline]
    pub fn mul_add(self, b: Self, s: f32x4) -> Self {
        Self {
            x: self.x + b.x * s,
            y: self.y + b.y * s,
            z: self.z + b.z * s,
        }
    }

    #[inline]
    pub fn dot(self, b: Self) -> f32x4 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }

    #[inline]
    pub fn cross(self, b: Self) -> Self {
        Self {
            x: self.y * b.z - self.z * b.y,
            y: self.z * b.x - self.x * b.z,
            z: self.x * b.y - self.y * b.x,
        }
    }

    #[inline]
    pub fn length_sq(self) -> f32x4 {
        self.dot(self)
    }
}

impl Add for Vec3x4 {
    type Output = Self;
    #[inline]
    fn add(self, b: Self) -> Self {
        Self {
            x: self.x + b.x,
            y: self.y + b.y,
            z: self.z + b.z,
        }
    }
}

impl Sub for Vec3x4 {
    type Output = Self;
    #[inline]
    fn sub(self, b: Self) -> Self {
        Self {
            x: self.x - b.x,
            y: self.y - b.y,
            z: self.z - b.z,
        }
    }
}

/// AABB 4개 (SoA) — 브로드페이즈 4-ary 노드의 바운드 레이아웃과 동형 (설계 §4.1).
/// 쿼터니언 4개를 레인에 실은 SoA — 회전을 4-wide로 돌린다.
///
/// 접촉 앵커 갱신(`refresh_anchors`)은 서브스텝마다 접점 전량을 두 번 회전시킨다.
/// 레인마다 자세가 달라도 회전 식은 성분별 곱·덧셈뿐이라 셔플 없이 4-wide가 된다.
/// 레인당 연산 순서가 스칼라 `Quat::rotate`와 같으므로 결과는 **비트 동일**하다.
#[derive(Clone, Copy, Debug)]
pub struct Quatx4 {
    pub x: f32x4,
    pub y: f32x4,
    pub z: f32x4,
    pub w: f32x4,
}

impl Quatx4 {
    #[inline]
    pub fn splat(q: Quat) -> Self {
        Self {
            x: f32x4::splat(q.x),
            y: f32x4::splat(q.y),
            z: f32x4::splat(q.z),
            w: f32x4::splat(q.w),
        }
    }

    #[inline]
    pub fn from_array(q: [Quat; 4]) -> Self {
        Self {
            x: f32x4::from([q[0].x, q[1].x, q[2].x, q[3].x]),
            y: f32x4::from([q[0].y, q[1].y, q[2].y, q[3].y]),
            z: f32x4::from([q[0].z, q[1].z, q[2].z, q[3].z]),
            w: f32x4::from([q[0].w, q[1].w, q[2].w, q[3].w]),
        }
    }

    /// `v + 2w(u×v) + u×(2(u×v))` — 스칼라 `Quat::rotate`와 **같은 순서**.
    #[inline]
    pub fn rotate(self, v: Vec3x4) -> Vec3x4 {
        let u = Vec3x4 {
            x: self.x,
            y: self.y,
            z: self.z,
        };
        let t = u.cross(v).scale(f32x4::splat(2.0));
        v + t.scale(self.w) + u.cross(t)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Aabbx4 {
    pub min: Vec3x4,
    pub max: Vec3x4,
}

impl Aabbx4 {
    /// 질의 AABB [qmin,qmax] 와 4개 박스의 겹침 마스크(레인당 all-1s/0).
    /// 브로드페이즈 순회 1스텝 = 이 호출 1회.
    #[inline]
    pub fn overlaps(self, qmin: Vec3x4, qmax: Vec3x4) -> f32x4 {
        self.min.x.cmp_le(qmax.x)
            & self.min.y.cmp_le(qmax.y)
            & self.min.z.cmp_le(qmax.z)
            & self.max.x.cmp_ge(qmin.x)
            & self.max.y.cmp_ge(qmin.y)
            & self.max.z.cmp_ge(qmin.z)
    }
}

/// Mat3 4개 (SoA) — 열 우선, 각 열이 `Vec3x4`. 솔버 배칭에서 4바디의 월드 역관성을
/// 레인에 실어 `inv_i.mul_vec`를 4-wide로 돈다 (설계 §3·§5 SIMD 제약 배칭).
#[derive(Clone, Copy, Debug)]
pub struct Mat3x4 {
    /// 열 벡터 3개 (각 4-wide).
    pub c: [Vec3x4; 3],
}

impl Mat3x4 {
    /// 같은 행렬을 네 레인에 — **레인이 방향(또는 접점)일 때** 쓰는 형태다.
    /// `from_array([m; 4])`와 값은 같지만 그쪽은 레인마다 스칼라 gather를 돌린다;
    /// 이쪽은 성분마다 splat 하나다(제약 준비의 페어당 1회 셋업 경로).
    #[inline]
    pub fn splat(m: Mat3) -> Self {
        Self {
            c: [
                Vec3x4::splat(m.c[0]),
                Vec3x4::splat(m.c[1]),
                Vec3x4::splat(m.c[2]),
            ],
        }
    }

    /// AoS 4개 → SoA 1개 (gather는 스칼라 — WASM에 gather 명령 없음, 설계 §3 정직한 범위).
    #[inline]
    pub fn from_array(m: [Mat3; 4]) -> Self {
        Self {
            c: [
                Vec3x4::from_array([m[0].c[0], m[1].c[0], m[2].c[0], m[3].c[0]]),
                Vec3x4::from_array([m[0].c[1], m[1].c[1], m[2].c[1], m[3].c[1]]),
                Vec3x4::from_array([m[0].c[2], m[1].c[2], m[2].c[2], m[3].c[2]]),
            ],
        }
    }

    /// 레인별 M·v. 스칼라 `Mat3::mul_vec`(`c0·vx + c1·vy + c2·vz`)와 **연산 순서가 동일**하므로
    /// 레인당 결과가 스칼라와 비트 동일하다(비FMA·IEEE 단정밀 — SIMD 솔브가 콜러드-스칼라와
    /// 같은 L0 해시를 내게 하는 근거, 설계 §8).
    #[inline]
    pub fn mul_vec(self, v: Vec3x4) -> Vec3x4 {
        self.c[0].scale(v.x) + self.c[1].scale(v.y) + self.c[2].scale(v.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soa_cross_matches_scalar() {
        let a = [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-1.0, 0.5, 2.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(4.0, -2.0, 0.25),
        ];
        let b = [
            Vec3::new(0.5, -1.0, 2.0),
            Vec3::new(3.0, 3.0, -3.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-0.5, 0.5, 8.0),
        ];
        let c = Vec3x4::from_array(a).cross(Vec3x4::from_array(b));
        let (cx, cy, cz) = (c.x.to_array(), c.y.to_array(), c.z.to_array());
        for i in 0..4 {
            let s = a[i].cross(b[i]);
            assert!((cx[i] - s.x).abs() < 1e-6);
            assert!((cy[i] - s.y).abs() < 1e-6);
            assert!((cz[i] - s.z).abs() < 1e-6);
        }
    }

    #[test]
    fn soa_dot_matches_scalar() {
        let a = [Vec3::new(1.0, 2.0, 3.0); 4];
        let b = [Vec3::new(-2.0, 0.5, 4.0); 4];
        let d = Vec3x4::from_array(a).dot(Vec3x4::from_array(b)).to_array();
        for i in 0..4 {
            assert!((d[i] - a[i].dot(b[i])).abs() < 1e-6);
        }
    }

    /// 계약: `Mat3x4::mul_vec`는 레인당 스칼라 `Mat3::mul_vec`과 **비트 동일**하다
    /// (epsilon 아님 — 연산 순서 동일 + 비FMA). 이게 깨지면 SIMD 솔브의 L0 해시가 갈린다.
    #[test]
    fn mat3x4_mul_vec_is_bit_identical_to_scalar() {
        use crate::Quat;
        let mats = [
            Mat3::rotate_inertia(
                Vec3::new(0.5, 2.0, 1.3),
                Quat::from_axis_angle(Vec3::new(1.0, 0.2, -0.4), 0.7),
            ),
            Mat3::from_diagonal(Vec3::new(1.0, 3.0, 0.25)),
            Mat3::rotate_inertia(
                Vec3::new(2.0, 0.1, 5.0),
                Quat::from_axis_angle(Vec3::new(-0.3, 1.0, 0.5), 1.9),
            ),
            Mat3::IDENTITY,
        ];
        let vs = [
            Vec3::new(1.0, -2.0, 3.5),
            Vec3::new(0.0, 4.0, -1.0),
            Vec3::new(-3.3, 0.7, 2.2),
            Vec3::new(9.9, -9.9, 0.001),
        ];
        let out = Mat3x4::from_array(mats).mul_vec(Vec3x4::from_array(vs));
        let (ox, oy, oz) = (out.x.to_array(), out.y.to_array(), out.z.to_array());
        for i in 0..4 {
            let s = mats[i].mul_vec(vs[i]);
            // **비트 비교** — `f32` `==`는 +0.0==-0.0이라 state hash가 구분하는 부호 0 차이를
            // 놓친다(교차검수 rev_21070c63 MINOR). to_bits로 부호 0·NaN 페이로드까지 잠근다.
            assert_eq!(ox[i].to_bits(), s.x.to_bits(), "lane {i} x");
            assert_eq!(oy[i].to_bits(), s.y.to_bits(), "lane {i} y");
            assert_eq!(oz[i].to_bits(), s.z.to_bits(), "lane {i} z");
        }
    }

    #[test]
    fn aabb_overlap_mask() {
        let boxes = Aabbx4 {
            min: Vec3x4::from_array([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(5.0, 5.0, 5.0),
                Vec3::new(-2.0, -2.0, -2.0),
                Vec3::new(10.0, 0.0, 0.0),
            ]),
            max: Vec3x4::from_array([
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(6.0, 6.0, 6.0),
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(11.0, 1.0, 1.0),
            ]),
        };
        // 질의: [0.5,0.5,0.5] ~ [1.5,1.5,1.5] → 박스 0만 겹침
        let qmin = Vec3x4::splat(Vec3::new(0.5, 0.5, 0.5));
        let qmax = Vec3x4::splat(Vec3::new(1.5, 1.5, 1.5));
        let m = boxes.overlaps(qmin, qmax).to_array();
        assert!(
            m[0] != 0.0 && m[1] == 0.0 && m[2] == 0.0 && m[3] == 0.0,
            "{m:?}"
        );
    }
}
