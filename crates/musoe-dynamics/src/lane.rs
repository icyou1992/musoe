//! 스칼라와 4-wide SIMD가 **같은 산술을 한 번만 쓰게** 하는 얇은 추상.
//!
//! 솔버는 경로가 둘이다: 스칼라(`solver::solve` — 색 꼬리·오버플로 담당)와 SoA 블록
//! (`solver_block::solve_block` — 프로덕션 SIMD). 둘은 **레인당 비트 동일**이어야 하는데,
//! 예전에는 같은 식을 두 곳에 손으로 맞춰 넣었다. 그러면 한쪽만 고치는 순간 조용히 갈리고,
//! 그 어긋남은 차등 테스트가 돌 때까지 안 보인다(2026-07-31 마찰 변경이 실제로 그랬다).
//!
//! 여기서는 **갈릴 수 있는 부분만** 제네릭으로 뽑는다: 분기(`if`)와 blend가 다를 뿐 나머지
//! 산술은 같으므로, 조건 선택을 `select`로 통일하면 식이 하나가 된다. 벡터 내적·임펄스 적용
//! 같은 배관은 타입마다 모양이 달라 각 경로에 남긴다 — 억지로 묶으면 트레이트만 커진다.
//!
//! **비트 동일 규율**: 연산 순서를 바꾸지 않는다. `select`는 산술 마스킹이 아니라 진짜
//! 선택이어야 한다(`x + 0.0`은 `-0.0`을 `+0.0`으로 뒤집어 상태 해시를 가른다).

use core::ops::{Add, Div, Mul, Sub};
use wide::{f32x4, CmpGt};

/// 스칼라 `f32`와 `f32x4`를 같은 코드로 다루기 위한 최소 연산 집합.
pub trait Lane:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    /// 레인별 비교 결과. 스칼라는 `bool`, SIMD는 마스크 벡터다.
    type Mask: Copy;

    fn splat(x: f32) -> Self;
    fn sqrt(self) -> Self;
    fn gt(self, other: Self) -> Self::Mask;
    fn mask_and(a: Self::Mask, b: Self::Mask) -> Self::Mask;
    /// 참인 레인은 `t`, 거짓인 레인은 `f`를 **비트 그대로** 고른다.
    fn select(m: Self::Mask, t: Self, f: Self) -> Self;
}

impl Lane for f32 {
    type Mask = bool;
    #[inline(always)]
    fn splat(x: f32) -> Self {
        x
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline(always)]
    fn gt(self, other: Self) -> bool {
        self > other
    }
    #[inline(always)]
    fn mask_and(a: bool, b: bool) -> bool {
        a && b
    }
    #[inline(always)]
    fn select(m: bool, t: Self, f: Self) -> Self {
        if m {
            t
        } else {
            f
        }
    }
}

impl Lane for f32x4 {
    type Mask = f32x4;
    #[inline(always)]
    fn splat(x: f32) -> Self {
        f32x4::splat(x)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32x4::sqrt(self)
    }
    #[inline(always)]
    fn gt(self, other: Self) -> f32x4 {
        self.cmp_gt(other)
    }
    #[inline(always)]
    fn mask_and(a: f32x4, b: f32x4) -> f32x4 {
        a & b
    }
    #[inline(always)]
    fn select(m: f32x4, t: Self, f: Self) -> Self {
        m.blend(t, f)
    }
}

/// 접선 마찰 한 접점: **두 축을 동시에 풀고 Coulomb 원판에 투영한 뒤 그 차이만 돌려준다.**
///
/// 반환 `(누적, 델타)` — 호출자는 누적을 저장하고 델타로 임펄스를 만든다. 임펄스 적용은
/// 경로마다 모양이 달라(스칼라는 마스크가 없고 블록은 `has`가 있다) 여기서 하지 않는다.
///
/// 클램프를 **적용 앞에** 두는 것이 이 함수의 핵심이다: 축별 임펄스를 클램프 없이 먼저
/// 적용하고 나중에 보정하면 중간 상태가 Coulomb 한계를 넘고 그 초과분이 접점마다 겹쳐
/// 발산한다(예전 Jacobi가 낙하 체인에서 그랬다). 누적을 먼저 원판에 가두면 어떤 중간
/// 상태도 한계를 넘지 않는다.
///
/// 분모를 `select`로 1에 묶는 이유는 클램프가 필요 없는 레인에서 `0/0`을 실행하지 않기
/// 위해서다 — `blend`는 값의 NaN만 가리고 **FP 예외 플래그는 이미 오른다**.
#[inline(always)]
pub fn friction_disc<L: Lane>(ti: [L; 2], tm: [L; 2], vt: [L; 2], max_f: L) -> ([L; 2], [L; 2]) {
    let one = L::splat(1.0);
    let zero = L::splat(0.0);
    let n0 = ti[0] - tm[0] * vt[0];
    let n1 = ti[1] - tm[1] * vt[1];
    let mag2 = n0 * n0 + n1 * n1;
    let cond = L::mask_and(mag2.gt(max_f * max_f), mag2.gt(zero));
    let scale = max_f / L::select(cond, mag2, one).sqrt();
    let n0 = L::select(cond, n0 * scale, n0);
    let n1 = L::select(cond, n1 * scale, n1);
    ([n0, n1], [n0 - ti[0], n1 - ti[1]])
}
