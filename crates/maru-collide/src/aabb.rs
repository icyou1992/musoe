//! 스칼라 AABB — 브로드페이즈 정밀 판정(설계 §4.1 "페어 집합은 트리 비의존")의 단위.

use maru_math::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// 중심±half-extent 생성.
    #[inline]
    pub fn from_center(center: Vec3, half: Vec3) -> Self {
        Self {
            min: center - half,
            max: center + half,
        }
    }

    #[inline]
    pub fn overlaps(self, b: Self) -> bool {
        self.min.x <= b.max.x
            && self.min.y <= b.max.y
            && self.min.z <= b.max.z
            && self.max.x >= b.min.x
            && self.max.y >= b.min.y
            && self.max.z >= b.min.z
    }

    #[inline]
    pub fn merge(self, b: Self) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.min(b.min.x),
                self.min.y.min(b.min.y),
                self.min.z.min(b.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(b.max.x),
                self.max.y.max(b.max.y),
                self.max.z.max(b.max.z),
            ),
        }
    }

    /// speculative margin(설계 §4.5): 속도×dt만큼 이동 방향으로 확장 + 대칭 여유.
    #[inline]
    pub fn expand_sweep(self, vel_dt: Vec3, margin: f32) -> Self {
        let m = Vec3::new(margin, margin, margin);
        Self {
            min: Vec3::new(
                self.min.x + vel_dt.x.min(0.0),
                self.min.y + vel_dt.y.min(0.0),
                self.min.z + vel_dt.z.min(0.0),
            ) - m,
            max: Vec3::new(
                self.max.x + vel_dt.x.max(0.0),
                self.max.y + vel_dt.y.max(0.0),
                self.max.z + vel_dt.z.max(0.0),
            ) + m,
        }
    }

    #[inline]
    pub fn surface_area(self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_and_touch() {
        let a = Aabb::from_center(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::from_center(Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(a.overlaps(b)); // 면 접촉(경계 포함)
        let c = Aabb::from_center(Vec3::new(2.1, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(!a.overlaps(c));
    }

    #[test]
    fn sweep_expansion_directional() {
        let a = Aabb::from_center(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let e = a.expand_sweep(Vec3::new(3.0, 0.0, 0.0), 0.1);
        assert!((e.max.x - 4.1).abs() < 1e-6); // +x로만 스윕
        assert!((e.min.x + 1.1).abs() < 1e-6); // -x는 margin만
    }
}
