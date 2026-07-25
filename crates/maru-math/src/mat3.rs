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
    pub fn inverse(&self) -> Mat3 {
        let a = self.c[0];
        let b = self.c[1];
        let c = self.c[2];
        let r0 = b.cross(c);
        let r1 = c.cross(a);
        let r2 = a.cross(b);
        let det = a.dot(r0);
        if det.abs() < 1.0e-12 {
            return Mat3::from_diagonal(Vec3::ZERO);
        }
        let inv = 1.0 / det;
        // 역행렬 = adj/det, adj 행 = cofactor → 열로 전치
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
