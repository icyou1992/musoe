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

    /// 축·각에서 쿼터니언.
    ///
    /// 두 실패 모드를 **다르게** 취급한다 (확인검수 rev_1a31d83f MAJOR):
    /// - **축을 정규화할 수 없으면 항등** — 0 축은 "회전축이 없다 = 회전이 없다"는 뜻으로
    ///   데이터에서 자연히 나온다. 여기서 비단위 쿼터니언을 만들면 이후 회전이 조용히
    ///   붕괴하므로 항등이 맞다(`Quat::normalize`와 같은 규약).
    /// - **각이 비유한이거나 `MAX_AXIS_ANGLE`을 넘으면 패닉** — 이건 데이터가 아니라
    ///   호출자의 버그다(도·라디안 혼동, 발산한 각속도). 예전에는 조용히 항등을 돌려줘서
    ///   "1e11 rad 회전"이 "회전 안 함"으로 뭉개졌고, 물체가 도는 걸 멈춘 이유를 어디서도
    ///   알 수 없었다. `try_` 접두사 없는 생성자가 표현 불가 입력에 패닉하는 것은
    ///   Rust 관례이며, 조용히 틀리는 것보다 낫다.
    ///
    /// 표현 불가 각을 값으로 다뤄야 하면 `try_from_axis_angle`을 쓴다(패닉 없음).
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        assert!(
            angle.is_finite() && angle.abs() <= Self::MAX_AXIS_ANGLE,
            "Quat::from_axis_angle: 표현 불가한 각 {angle} (유한하고 |angle| <= {} 여야 한다; 값으로 처리하려면 try_from_axis_angle)",
            Self::MAX_AXIS_ANGLE
        );
        Self::try_from_axis_angle(axis, angle).unwrap_or(Self::IDENTITY)
    }

    /// 결정적으로 표현 가능한 최대 각(rad) — 이보다 큰 각은 `try_from_axis_angle`이 거부한다.
    ///
    /// 회전은 2π 주기이므로 큰 각도 **의미가 있다**. 다만 그 각을 f32/f64 산술로 정확히
    /// 접으려면 Payne–Hanek 비트 테이블이 필요하고(우리는 안 넣었다), 그 없이 접으면
    /// 결과가 조용히 틀린다 — 그래서 **거부**한다. 물리는 각속도 상한(100 rad/s) 때문에
    /// 스텝당 2 rad을 넘지 않아 이 경계에 닿지 않는다.
    pub const MAX_AXIS_ANGLE: f32 = 1.0e10;

    /// 축·각에서 쿼터니언 (오류 반환형).
    ///
    /// `None`인 경우: 축을 정규화할 수 없음 · 각이 비유한 · **각이 `MAX_AXIS_ANGLE` 초과**.
    /// 마지막 항목이 이 함수가 존재하는 이유다: 예전에는 거대한 유한 각을 조용히 항등
    /// 회전으로 뭉갰다(확인검수 rev_e50670ad MAJOR — 문서는 "비유한만 항등"이라 적혀 있었다).
    pub fn try_from_axis_angle(axis: Vec3, angle: f32) -> Option<Self> {
        let a = axis.try_normalize()?;
        if !angle.is_finite() || angle.abs() > Self::MAX_AXIS_ANGLE {
            return None;
        }
        // **결정적 sin/cos** — 회전 CCD가 매 반복 이 함수를 부른다. 플랫폼 sin_cos의
        // 마지막 비트가 갈리면 TOI가 갈리고 L1 계약이 깨진다(검수 rev_1e81559e BLOCKER).
        let (s, c) = crate::det::sin_cos(angle * 0.5);
        Some(Self {
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
            w: c,
        })
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

    /// **경계 캐노니컬화** — 이미 (f32 오차 범위에서) 단위면 **비트 그대로 둔다**.
    ///
    /// `normalize`는 f32에서 **비트 멱등이 아니다**: `q/√(q·q)`를 두 번 적용하면 값이
    /// 1 ULP 왔다 갔다 하는 **주기 2 진동**이 생긴다(실측: 무작위 표본 20만 개 중 약 5%,
    /// 예 `(-0.03805745, 0.05848086, -0.5699489, 0.788194)`가 두 값을 오간다).
    ///
    /// 그래서 정규화를 여러 경계(생성자·삽입·스냅샷 리더)에서 반복하면 **적용 횟수의
    /// 홀짝에 따라 값이 갈린다** — 삽입 때 2회, 복원 때 3회가 적용되어 같은 월드의
    /// 상태 해시가 왕복만으로 달라졌다(교차검수 rev_2001abc9 NEW_DEFECT).
    ///
    /// 여기서는 `|q·q − 1| ≤ 1e-5`면 그대로 돌려준다. 1회 정규화 후 잔차는 실측 최대
    /// **3.6e-7**(≈3 ULP)이라 이 문턱 안에 확실히 들어오므로, `canonicalize`는
    /// **적용 횟수와 무관한 고정점**이 된다. 문턱은 스냅샷 리더의 수용 범위(0.9~1.1)보다
    /// 다섯 자리 좁아 "단위가 아닌 값을 눈감아 주는" 쪽으로 새지 않는다.
    #[inline]
    pub fn try_canonicalize(self) -> Option<Self> {
        let l2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        // NaN은 어느 비교도 통과하지 못하므로 양성 조건으로 쓴다(부정 비교 금지 규약).
        if l2.is_finite() && l2 > 0.0 && (l2 - 1.0).abs() <= 1.0e-5 {
            return Some(self);
        }
        self.try_normalize()
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

    /// 리뷰 지적: 구 버전은 z축 회전인데 x·w만 검사해 **z 성분의 부호·크기가 틀려도 통과**했다.
    /// 전 성분 + 오차 차수(dt를 반으로 줄이면 오차가 ~4배 준다 = 2차)까지 본다.
    #[test]
    fn integrate_matches_axis_angle_all_components() {
        let omega = Vec3::new(0.0, 0.0, 2.0);
        let err_at = |dt: f32| {
            let q = Quat::IDENTITY.integrate(omega, dt);
            let e = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 2.0 * dt);
            // 전 성분 비교 (z가 핵심 — 회전축 성분)
            ((q.x - e.x).abs() + (q.y - e.y).abs() + (q.z - e.z).abs() + (q.w - e.w).abs())
                .max(1.0e-12)
        };
        let dt = 1.0e-2f32;
        let e1 = err_at(dt);
        let e2 = err_at(dt * 0.5);
        assert!(e1 < 1.0e-4, "절대 오차 과다: {e1}");
        // 2차 정확도면 비율 ≈ 4 (1차면 ≈ 2) — 부호 오류는 여기서 비율이 무너진다
        let ratio = e1 / e2;
        assert!(ratio > 2.5, "오차 차수 저하: e1={e1} e2={e2} ratio={ratio}");
        // 부호: +z 각속도는 +z 성분을 만든다
        let q = Quat::IDENTITY.integrate(omega, dt);
        assert!(q.z > 0.0, "회전 부호 반전: {q:?}");
    }
}
