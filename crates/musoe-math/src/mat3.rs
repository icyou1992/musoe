//! 3×3 행렬 — 관성 텐서 회전용 최소 API (열 우선).

use crate::{Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    /// 열 벡터 3개.
    pub c: [Vec3; 3],
}

impl Mat3 {
    pub const IDENTITY: Self = Self {
        c: [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
    };

    #[inline]
    pub fn from_diagonal(d: Vec3) -> Self {
        Self {
            c: [
                Vec3::new(d.x, 0.0, 0.0),
                Vec3::new(0.0, d.y, 0.0),
                Vec3::new(0.0, 0.0, d.z),
            ],
        }
    }

    #[inline]
    pub fn from_quat(q: Quat) -> Self {
        Self {
            c: [
                q.rotate(Vec3::new(1.0, 0.0, 0.0)),
                q.rotate(Vec3::new(0.0, 1.0, 0.0)),
                q.rotate(Vec3::new(0.0, 0.0, 1.0)),
            ],
        }
    }

    #[inline]
    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        self.c[0] * v.x + self.c[1] * v.y + self.c[2] * v.z
    }

    #[inline]
    pub fn mul_mat(&self, b: &Mat3) -> Mat3 {
        Mat3 {
            c: [
                self.mul_vec(b.c[0]),
                self.mul_vec(b.c[1]),
                self.mul_vec(b.c[2]),
            ],
        }
    }

    #[inline]
    pub fn transpose(&self) -> Mat3 {
        Mat3 {
            c: [
                Vec3::new(self.c[0].x, self.c[1].x, self.c[2].x),
                Vec3::new(self.c[0].y, self.c[1].y, self.c[2].y),
                Vec3::new(self.c[0].z, self.c[1].z, self.c[2].z),
            ],
        }
    }

    /// 3×3 역행렬 (특이면 0 행렬 — NaN 생산 금지, 설계 §8).
    /// **상대 판정**: 절대 임계는 질량 스케일에 취약하다 — 2만 kg 바디의 K=(1/m)I는
    /// det≈1.2e-13이라 절대 1e-12 임계에서 특이로 오판되어 조인트가 죽는다(리뷰 지적).
    pub fn inverse(&self) -> Mat3 {
        let scale = [
            self.c[0].x,
            self.c[0].y,
            self.c[0].z,
            self.c[1].x,
            self.c[1].y,
            self.c[1].z,
            self.c[2].x,
            self.c[2].y,
            self.c[2].z,
        ]
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()));
        if scale == 0.0 || !scale.is_finite() {
            return Mat3::from_diagonal(Vec3::ZERO);
        }
        let inv_scale = 1.0 / scale;
        let a = self.c[0] * inv_scale;
        let b = self.c[1] * inv_scale;
        let c = self.c[2] * inv_scale;
        let r0 = b.cross(c);
        let r1 = c.cross(a);
        let r2 = a.cross(b);
        let det = a.dot(r0);
        if !det.is_finite() || det.abs() < 1.0e-6 {
            return Mat3::from_diagonal(Vec3::ZERO);
        }
        // M = scale·N → M⁻¹ = N⁻¹/scale
        let inv = inv_scale / det;
        Mat3 {
            c: [
                Vec3::new(r0.x, r1.x, r2.x) * inv,
                Vec3::new(r0.y, r1.y, r2.y) * inv,
                Vec3::new(r0.z, r1.z, r2.z) * inv,
            ],
        }
    }

    /// 월드 역관성 = R · I⁻¹_local(대각) · Rᵀ
    #[inline]
    pub fn rotate_inertia(inv_inertia_diag: Vec3, q: Quat) -> Mat3 {
        let r = Mat3::from_quat(q);
        let ri = Mat3 {
            c: [
                r.c[0] * inv_inertia_diag.x,
                r.c[1] * inv_inertia_diag.y,
                r.c[2] * inv_inertia_diag.z,
            ],
        };
        // (R·D)·Rᵀ
        ri.mul_mat(&r.transpose())
    }
}

#[cfg(test)]
mod tests {

    /// 회전행렬의 **열 = 회전된 단위축**이고 순서가 x,y,z다 — 비트 단위로.
    ///
    /// `musoe-collide`의 박스-박스 경로가 `axes_a = Mat3::from_quat(rot).c`로 쿼터니언 회전
    /// 3회를 대체한다. 그 대체가 성립하는 근거가 이 성질이고, **열 순서가 뒤바뀌어도 대칭적인
    /// 씬에서는 상위 테스트가 통과할 수 있다** — 그래서 여기서 직접 잠근다(가장 조용한 실패 모드).
    #[test]
    fn from_quat_columns_are_rotated_unit_axes_bit_for_bit() {
        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 40) as f32 / (1u64 << 23) as f32) * 2.0 - 1.0
        };
        let mut quats = vec![Quat::IDENTITY];
        for _ in 0..200 {
            quats.push(Quat::from_axis_angle(
                Vec3::new(next(), next(), next()).normalize_or_zero() + Vec3::new(0.0, 1e-9, 0.0),
                next() * 3.2,
            ));
        }
        for q in quats {
            let m = Mat3::from_quat(q);
            let axes = [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ];
            for (i, e) in axes.into_iter().enumerate() {
                let r = q.rotate(e);
                assert_eq!(m.c[i].x.to_bits(), r.x.to_bits(), "열 {i}.x (q={q:?})");
                assert_eq!(m.c[i].y.to_bits(), r.y.to_bits(), "열 {i}.y (q={q:?})");
                assert_eq!(m.c[i].z.to_bits(), r.z.to_bits(), "열 {i}.z (q={q:?})");
            }
        }
    }

    use super::*;

    #[test]
    fn identity_rotation_keeps_diagonal() {
        let m = Mat3::rotate_inertia(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        assert!((m.c[0].x - 1.0).abs() < 1e-6);
        assert!((m.c[1].y - 2.0).abs() < 1e-6);
        assert!((m.c[2].z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn inverse_roundtrip() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, 0.2, -0.4), 0.7);
        let m = Mat3::rotate_inertia(Vec3::new(0.5, 2.0, 1.3), q);
        let mi = m.inverse();
        let id = m.mul_mat(&mi);
        for i in 0..3 {
            for (j, want) in [(0, i == 0), (1, i == 1), (2, i == 2)] {
                let v = match j {
                    0 => id.c[i].x,
                    1 => id.c[i].y,
                    _ => id.c[i].z,
                };
                let expect = if want { 1.0 } else { 0.0 };
                assert!((v - expect).abs() < 1e-4, "[{i}][{j}]={v}");
            }
        }
    }

    /// 리뷰 회귀: 큰 질량(작은 역질량)의 K 행렬이 특이로 오판되면 조인트가 죽는다.
    #[test]
    fn inverse_handles_small_scale_matrix() {
        let inv_m = 1.0 / 20000.0; // 20t 바디
        let k = Mat3::from_diagonal(Vec3::new(inv_m, inv_m, inv_m));
        let ki = k.inverse();
        assert!((ki.c[0].x - 20000.0).abs() < 1.0, "{:?}", ki.c[0]);
    }

    #[test]
    fn rotated_inertia_symmetric_and_consistent() {
        let q = Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.5), 0.9);
        let m = Mat3::rotate_inertia(Vec3::new(1.0, 2.0, 3.0), q);
        // 대칭성
        assert!((m.c[0].y - m.c[1].x).abs() < 1e-5);
        assert!((m.c[0].z - m.c[2].x).abs() < 1e-5);
        // 로컬 축을 돌리면 로컬 대각값이 나와야 함: M · (R·e1) = d1 · (R·e1)
        let axis = q.rotate(Vec3::new(0.0, 1.0, 0.0));
        let out = m.mul_vec(axis);
        assert!((out - axis * 2.0).length() < 1e-5, "{out:?}");
    }
}

impl Mat3 {
    /// **대칭** 3×3의 고유분해 (Jacobi 회전) — `A = R · diag(λ) · Rᵀ`, R의 열이 고유벡터.
    ///
    /// 복합 바디의 합성 관성 텐서를 주축으로 돌리는 데 쓴다: 평행축 이동은 관성곱
    /// (비대각 항)을 만들고, 엔진은 로컬 관성을 대각(Vec3)으로 저장하므로 생성 경계에서
    /// 주축을 pose로 접어야 한다. 입력이 대칭이 아니면 결과는 의미가 없다(호출자 책임 —
    /// 관성 텐서는 구성상 대칭이다).
    ///
    /// Jacobi는 대칭 행렬에서 무조건 수렴하고 분기 없는 회전만 쓰므로 결정론 계약(§8)에
    /// 안전하다. 반복 상한 24 — 3×3에서는 6~10회면 기계 정밀도에 닿는다.
    pub fn eigen_symmetric(&self) -> (Vec3, Mat3) {
        let mut a = *self;
        let mut v = Mat3::IDENTITY;
        for _ in 0..24 {
            // 최대 비대각 원소 (p,q) 선택
            let od = [
                (a.c[1].x.abs(), 0usize, 1usize), // a01
                (a.c[2].x.abs(), 0, 2),           // a02
                (a.c[2].y.abs(), 1, 2),           // a12
            ];
            let &(m, p, q) = od
                .iter()
                .max_by(|x, y| x.0.total_cmp(&y.0))
                .expect("3원소 배열");
            // 대각합 대비 무시 가능하면 종료 (스케일 상대 판정)
            let diag_scale = a.c[0].x.abs() + a.c[1].y.abs() + a.c[2].z.abs();
            if m <= f32::EPSILON * diag_scale.max(f32::MIN_POSITIVE) {
                break;
            }
            let apq = get(&a, p, q);
            let app = get(&a, p, p);
            let aqq = get(&a, q, q);
            // 고전 Jacobi 각: tan(2θ) = 2·apq / (app − aqq)
            // **결정적 atan2**(`det` 모듈) — 플랫폼 libm은 마지막 ulp가 갈려 L1 계약을 깬다.
            let theta = 0.5 * crate::det::atan2(2.0 * apq, app - aqq);
            let (s, c) = crate::det::sin_cos(theta);
            // G = 평면 (p,q) 회전. A ← Gᵀ A G, V ← V G
            let mut g = Mat3::IDENTITY;
            set(&mut g, p, p, c);
            set(&mut g, q, q, c);
            set(&mut g, p, q, -s);
            set(&mut g, q, p, s);
            a = g.transpose().mul_mat(&a).mul_mat(&g);
            v = v.mul_mat(&g);
        }
        (Vec3::new(a.c[0].x, a.c[1].y, a.c[2].z), v)
    }
}

impl Quat {
    /// 회전 행렬(정규직교, det=+1) → 쿼터니언 (Shepperd). 주축 베이킹이 고유벡터 행렬을
    /// pose에 접을 때 쓴다. det=−1(반사)은 호출자가 열 반전으로 먼저 고쳐야 한다.
    pub fn from_mat3(m: &Mat3) -> Quat {
        let (m00, m11, m22) = (m.c[0].x, m.c[1].y, m.c[2].z);
        let tr = m00 + m11 + m22;
        let q = if tr > 0.0 {
            let s = (tr + 1.0).sqrt() * 2.0;
            Quat {
                w: 0.25 * s,
                x: (m.c[1].z - m.c[2].y) / s,
                y: (m.c[2].x - m.c[0].z) / s,
                z: (m.c[0].y - m.c[1].x) / s,
            }
        } else if m00 >= m11 && m00 >= m22 {
            let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
            Quat {
                w: (m.c[1].z - m.c[2].y) / s,
                x: 0.25 * s,
                y: (m.c[1].x + m.c[0].y) / s,
                z: (m.c[2].x + m.c[0].z) / s,
            }
        } else if m11 >= m22 {
            let s = (1.0 - m00 + m11 - m22).sqrt() * 2.0;
            Quat {
                w: (m.c[2].x - m.c[0].z) / s,
                x: (m.c[1].x + m.c[0].y) / s,
                y: 0.25 * s,
                z: (m.c[2].y + m.c[1].z) / s,
            }
        } else {
            let s = (1.0 - m00 - m11 + m22).sqrt() * 2.0;
            Quat {
                w: (m.c[0].y - m.c[1].x) / s,
                x: (m.c[2].x + m.c[0].z) / s,
                y: (m.c[2].y + m.c[1].z) / s,
                z: 0.25 * s,
            }
        };
        q.normalize()
    }
}

#[inline]
fn get(m: &Mat3, r: usize, c: usize) -> f32 {
    let col = m.c[c];
    match r {
        0 => col.x,
        1 => col.y,
        _ => col.z,
    }
}

#[inline]
fn set(m: &mut Mat3, r: usize, c: usize, v: f32) {
    let col = &mut m.c[c];
    match r {
        0 => col.x = v,
        1 => col.y = v,
        _ => col.z = v,
    }
}

#[cfg(test)]
mod eigen_tests {
    use super::*;

    fn sym(a00: f32, a01: f32, a02: f32, a11: f32, a12: f32, a22: f32) -> Mat3 {
        Mat3 {
            c: [
                Vec3::new(a00, a01, a02),
                Vec3::new(a01, a11, a12),
                Vec3::new(a02, a12, a22),
            ],
        }
    }

    /// 재구성 검증: R diag(λ) Rᵀ == A (허용 오차 = 스케일 상대).
    #[test]
    fn eigen_reconstructs_symmetric_matrix() {
        for m in [
            sym(2.0, 0.5, -0.3, 1.5, 0.7, 3.0),
            sym(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),       // 등방 (축퇴)
            sym(5.0, 4.9, 0.0, 5.0, 0.0, 0.1),       // 거의 특이
            sym(1e-3, 2e-4, 0.0, 4e-3, -1e-4, 9e-4), // 작은 스케일
        ] {
            let (l, r) = m.eigen_symmetric();
            let rec = r.mul_mat(&Mat3::from_diagonal(l)).mul_mat(&r.transpose());
            let scale: f32 = l.x.abs().max(l.y.abs()).max(l.z.abs()).max(1e-30);
            for c in 0..3 {
                let d = rec.c[c] - m.c[c];
                assert!(
                    d.length() <= 1e-5 * scale.max(1.0) + 1e-6,
                    "재구성 오차: {m:?} → λ={l:?} d={d:?}"
                );
            }
            // R은 회전(정규직교): RᵀR ≈ I
            let i = r.transpose().mul_mat(&r);
            assert!((i.c[0].x - 1.0).abs() < 1e-5 && i.c[1].x.abs() < 1e-5);
        }
    }

    /// from_mat3 ∘ from_quat == ±항등 (쿼터니언 부호 모호성 허용).
    #[test]
    fn quat_mat_roundtrip() {
        for q in [
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.7),
            Quat::from_axis_angle(Vec3::new(1.0, 2.0, -0.5).normalize_or_zero(), 2.9),
            Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::PI),
            Quat::IDENTITY,
        ] {
            let m = Mat3::from_quat(q);
            let r = Quat::from_mat3(&m);
            let dot = (q.x * r.x + q.y * r.y + q.z * r.z + q.w * r.w).abs();
            assert!(dot > 1.0 - 1e-5, "왕복 실패: {q:?} → {r:?} (|dot|={dot})");
        }
    }
}
