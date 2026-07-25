//! SoA 4-wide 배치 타입 — 핫패스 정본 레이아웃 (설계 §3).
//!
//! `Vec3x4` = 벡터 4개를 성분별 f32x4로: 모든 연산이 셔플 없는 순수 레인별 SIMD.
//! WASM SIMD128(4×f32)과 정확히 동형 — 설계 §4.1 브로드페이즈 4-ary와 같은 이유.

use crate::Vec3;
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
