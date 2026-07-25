//! 스칼라 Vec3/Quat — API 표면용. 핫패스 배치는 soa 모듈이 정본.

use core::ops::{Add, AddAssign, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, b: Self) -> f32 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }

    #[inline]
    pub fn cross(self, b: Self) -> Self {
        Self::new(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x,
        )
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// 유한·비영 벡터만 정규화. **무한대 성분은 l2=∞ → ∞·0 = NaN**이므로 유한성까지 본다.
    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let l2 = self.length_sq();
        if l2 > 0.0 && l2.is_finite() {
            let v = self * (1.0 / l2.sqrt());
            if v.is_finite() {
                return Some(v);
            }
        }
        None
    }

    /// 정규화 불가(0·NaN·Inf)면 0 (설계 §8 NaN 위생).
    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        self.try_normalize().unwrap_or(Self::ZERO)
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, b: Self) -> Self {
        Self::new(self.x + b.x, self.y + b.y, self.z + b.z)
    }
}
impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, b: Self) {
        *self = *self + b;
    }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, b: Self) -> Self {
        Self::new(self.x - b.x, self.y - b.y, self.z - b.z)
    }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// 단위 쿼터니언(회전). 저장 순서 (x, y, z, w).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// 축이 정규화 불가하거나 각이 비유한이면 **단위 쿼터니언**을 반환한다
    /// (0 축에 대해 비단위 쿼터니언을 만들면 이후 회전이 조용히 붕괴한다).
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let Some(a) = axis.try_normalize() else {
            return Self::IDENTITY;
        };
        if !angle.is_finite() {
            return Self::IDENTITY;
        }
        let (s, c) = (angle * 0.5).sin_cos();
        Self {
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
            w: c,
        }
    }

    /// v' = q v q⁻¹ — 2×cross 전개(표준 최적화, FMA 비의존).
    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let t = u.cross(v) * 2.0;
        v + t * self.w + u.cross(t)
    }

    /// 켤레 = 단위 쿼터니언의 역회전. 월드→로컬 방향 변환에 사용.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    /// 유한·비영일 때만 정규화 (0/NaN/Inf 무방비 나눗셈 차단).
    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let l2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        if l2 > 0.0 && l2.is_finite() {
            let inv = 1.0 / l2.sqrt();
            let q = Self {
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
                w: self.w * inv,
            };
            if q.is_finite() {
                return Some(q);
            }
        }
        None
    }

    /// 정규화 불가면 단위 쿼터니언 — 자세가 NaN으로 전파되는 것을 여기서 끊는다.
    #[inline]
    pub fn normalize(self) -> Self {
        self.try_normalize().unwrap_or(Self::IDENTITY)
    }

    /// 각속도 ω로 dt만큼 자세 적분(Symplectic Euler 짝 — 설계 §4.4).
    /// q' = normalize(q + 0.5·dt·ω_quat∘q)
    #[inline]
    pub fn integrate(self, omega: Vec3, dt: f32) -> Self {
        let h = 0.5 * dt;
        let dq = Quat {
            x: omega.x,
            y: omega.y,
            z: omega.z,
            w: 0.0,
        } * self;
        Quat {
            x: self.x + dq.x * h,
            y: self.y + dq.y * h,
            z: self.z + dq.z * h,
            w: self.w + dq.w * h,
        }
        .normalize()
    }
}

/// Hamilton product: self ∘ rhs (rhs 먼저 적용).
impl Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, r: Self) -> Self {
        Self {
            x: self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            y: self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            z: self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
            w: self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }
    fn vclose(a: Vec3, b: Vec3) -> bool {
        close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
    }

    #[test]
    fn cross_right_handed() {
        let c = Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0));
        assert!(vclose(c, Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn normalize_zero_is_zero_not_nan() {
        let n = Vec3::ZERO.normalize_or_zero();
        assert_eq!(n, Vec3::ZERO);
    }

    /// 리뷰 회귀: 무한/NaN 입력이 정규화를 통해 NaN으로 새어나가면 안 된다.
    #[test]
    fn nonfinite_inputs_never_produce_nan() {
        for v in [
            Vec3::new(f32::INFINITY, 0.0, 0.0),
            Vec3::new(f32::NAN, 1.0, 0.0),
            Vec3::new(f32::MAX, f32::MAX, f32::MAX),
        ] {
            assert!(v.normalize_or_zero().is_finite(), "{v:?}");
            assert!(v.try_normalize().is_none() || v.try_normalize().unwrap().is_finite());
        }
        for q in [
            Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
            Quat {
                x: f32::NAN,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            Quat {
                x: f32::INFINITY,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
        ] {
            assert!(q.normalize().is_finite(), "{q:?}");
        }
        // 0 축은 단위 쿼터니언
        let q = Quat::from_axis_angle(Vec3::ZERO, 1.0);
        assert_eq!(q, Quat::IDENTITY);
    }

    #[test]
    fn quat_rotate_90deg_z() {
        let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), core::f32::consts::FRAC_PI_2);
        let v = q.rotate(Vec3::new(1.0, 0.0, 0.0));
        assert!(vclose(v, Vec3::new(0.0, 1.0, 0.0)), "{v:?}");
    }

    #[test]
    fn quat_mul_composes() {
        let a = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.7);
        let b = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), -0.3);
        let v = Vec3::new(0.3, -1.2, 2.0);
        assert!(vclose((a * b).rotate(v), a.rotate(b.rotate(v))));
    }

    #[test]
    fn integrate_small_step_matches_axis_angle() {
        let omega = Vec3::new(0.0, 0.0, 2.0);
        let dt = 1.0e-3;
        let q = Quat::IDENTITY.integrate(omega, dt);
        let expect = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 2.0 * dt);
        assert!(close(q.x, expect.x) && close(q.w, expect.w));
    }
}
