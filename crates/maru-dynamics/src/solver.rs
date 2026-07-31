//! TGS-soft 접촉 솔버 (설계 §4.4 — Box2D v3 "Soft Step" 정식화).
//!
//! 서브스텝마다: 속도 적분 → warm start → soft bias 임펄스 해석 → 위치 적분 → relax(무bias).
//! 전 서브스텝 후 restitution 패스 1회 — **임팩트 게이트**(설계 §4.4): 접촉 생성 시점의
//! 접근 속도(approach_vn)가 임계 이상일 때만, 실제 임펄스가 발생한 접점에만 발화.
//! [근거: Solver2D 실증 + Box2D v3 releasing-3.0]

use crate::body::Body;
use maru_collide::Manifold;
use maru_math::{Mat3, Mat3x4, Quat, Vec3, Vec3x4};
// 솔브 패스의 프로덕션 SIMD는 `solver_block`이다. 이 모듈이 SIMD를 쓰는 곳은
// **제약 준비의 유효질량**(`effective_mass_x3`) 하나이고, 나머지 x4 코드는
// 참조 구현(cfg(test))이다.
use wide::{f32x4, CmpGt};

#[derive(Clone)]
pub struct SolverParams {
    pub contact_hertz: f32,
    pub contact_damping: f32,
    /// 침투 해소 최대 속도 (m/s) — 폭발 방지.
    pub max_bias_speed: f32,
    pub restitution_threshold: f32,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            contact_hertz: 30.0,
            contact_damping: 10.0,
            max_bias_speed: 3.0,
            restitution_threshold: 1.0,
        }
    }
}

/// soft constraint 계수 (Box2D v3 공식).
#[derive(Clone, Copy)]
pub struct Softness {
    pub bias_rate: f32,
    pub mass_scale: f32,
    pub impulse_scale: f32,
}

impl Softness {
    pub fn new(hertz: f32, zeta: f32, h: f32) -> Self {
        if hertz <= 0.0 {
            return Self {
                bias_rate: 0.0,
                mass_scale: 1.0,
                impulse_scale: 0.0,
            };
        }
        let omega = 2.0 * core::f32::consts::PI * hertz;
        let a1 = 2.0 * zeta + h * omega;
        let a2 = h * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Self {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }
}

#[derive(Clone)]
pub struct ContactPointConstraint {
    /// PCM 캐시 매칭 키 (설계 §4.2 canonical feature ID — M1 클리핑 계보).
    pub feature: maru_collide::FeatureId,
    /// prepare 시점 접점 월드 위치 (generic 접점 근접 매칭용).
    pub position: Vec3,
    /// 질량중심 기준 로컬 앵커 (서브스텝마다 현재 자세로 재회전 — TGS의 위치 추적).
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    /// prepare 시점 separation (앵커 동일 월드점 기준 오프셋).
    pub base_separation: f32,
    pub normal_mass: f32,
    /// 접선 두 축의 유효질량 `[1/k00, 1/k11]`.
    ///
    /// 두 축은 **동시에** 풀고 누적 임펄스를 Coulomb 원판에 투영한 뒤 그 차이만 한 번
    /// 적용한다(`solve` 참조). 결합항 `k01`을 넣은 완전한 2x2 해도 만들어 재 봤는데
    /// **실측으로 열등해 기각했다** — 아래 `solve`의 주석에 수치가 있다.
    pub tangent_mass: [f32; 2],
    pub normal_impulse: f32,
    pub tangent_impulse: [f32; 2],
    pub max_normal_impulse: f32,
    /// 이번 서브스텝의 월드 레버암 — `refresh_anchors`가 채운다.
    /// warm start와 bias 솔브는 같은 자세를 보므로 회전을 한 번만 한다
    /// (전에는 warm_start·solve·relax가 각각 다시 돌려 서브스텝당 3중 계산이었다).
    pub ra: Vec3,
    pub rb: Vec3,
    /// 접촉 생성 시점 법선 접근 속도 (restitution 게이트 입력).
    pub approach_vn: f32,
}

#[derive(Clone)]
pub struct ContactConstraint {
    pub a: usize,
    pub b: usize,
    /// A를 +normal로 밀면 분리 (maru-collide 관례).
    pub normal: Vec3,
    pub tangents: [Vec3; 2],
    pub friction: f32,
    pub restitution: f32,
    /// **각 야코비안을 접점에 선계산해 두지 말 것** (2026-07-29 실측으로 기각).
    /// 임펄스 적용마다 도는 `cross(r, imp)` + `Mat3::mul_vec`(24 flops)을
    /// `g = I⁻¹(r×axis)`를 미리 두고 `ang += g·λ`(6 flops)로 바꾸는, rapier
    /// `ii_torque_dir`과 같은 구조를 시도했다. 결과:
    /// - 산술은 실제로 ~40% 빠졌는데 **시간은 7.6%만** 줄었다 → 이 루프는 flop-bound가 아니다.
    /// - `g`를 스텝당 1회(prepare)만 만들면 회전이 빠른 장면이 **깨진다**:
    ///   `maru-bench --bin spin`(50 rad/s 이방성 막대)에서 600스텝 잔류 에너지비가
    ///   0.0014 → 0.0503, 각속도가 0 → 21.9 rad/s로 남았다. 크기(K⁻¹)가 낡은 것과
    ///   **적용 방향이 낡은 것은 다르다** — 방향이 틀리면 감쇠 자체가 성립하지 않는다.
    /// - 정확한 주기(서브스텝당 2회, 적용과 같은 자세)로 갱신하면 재계산 288 flops가
    ///   절약 324 flops를 먹어 **순손해**였다: 40단 7.40 → **8.52 ms**.
    ///
    /// 대신 이긴 것은 데이터 배치였다 — `solver_block`(SoA 블록) 참조.
    ///
    /// **인라인 배열로 바꾸지 말 것** (2026-07-26 실측으로 기각).
    /// "제약 배열을 스텝당 40번 훑으니 포인터 추적이 비쌀 것"이라는 가설로
    /// `[ContactPointConstraint; 4] + count`(≈400B)로 바꿔봤다. 결과:
    /// solve는 **변화 없음**(15.1 → 15.2 ms — 포인터 추적은 비용이 아니었다),
    /// narrow는 **40% 악화**(3.52 → 4.93 ms — `prepare`가 큰 구조체를 값으로 돌려주고
    /// `constraints.push`가 그걸 복사한다). `Manifold`의 같은 변경이 중립이었던 것과 달리
    /// 여기서는 순손해다.
    pub points: Vec<ContactPointConstraint>,
    /// 이번 스텝에 **실제로 warm load된** 접점 수 (진단 전용, 캐노니컬 상태 아님).
    pub warm_matched: u32,
}

impl ContactConstraint {
    /// 접점 버퍼를 **비운 채로 꺼낸다** — 풀에 되돌려 다음 스텝이 용량을 재사용한다.
    /// (`std::mem::take`로 꺼내면 제약은 빈 벡터를 들고 드롭된다 = 할당 없음.)
    pub(crate) fn into_points_buf(mut self) -> Vec<ContactPointConstraint> {
        self.points.clear();
        core::mem::take(&mut self.points)
    }
}

/// 법선에서 결정적 접선 기저.
fn tangent_basis(n: Vec3) -> [Vec3; 2] {
    let t1 = if n.x.abs() > 0.707 {
        Vec3::new(n.y, -n.x, 0.0).normalize_or_zero()
    } else {
        Vec3::new(0.0, n.z, -n.y).normalize_or_zero()
    };
    [t1, n.cross(t1)]
}

fn effective_mass(
    inv_m_a: f32,
    inv_i_a: &Mat3,
    ra: Vec3,
    inv_m_b: f32,
    inv_i_b: &Mat3,
    rb: Vec3,
    dir: Vec3,
) -> f32 {
    let rna = ra.cross(dir);
    let rnb = rb.cross(dir);
    let k = inv_m_a + inv_m_b + rna.dot(inv_i_a.mul_vec(rna)) + rnb.dot(inv_i_b.mul_vec(rnb));
    if k > 0.0 {
        1.0 / k
    } else {
        0.0
    }
}

/// 유효질량 3개(법선 + 접선 2)를 **한 번의 4-wide 패스**로 만든다 — 레인이 방향이다.
///
/// 스칼라 `effective_mass`를 방향마다 세 번 부르면 같은 `ra`·`rb`에 대해 외적 2회·관성텐서
/// 곱 2회·내적 2회를 세 번 되풀이한다. 방향을 레인에 실으면 그 세 번이 한 번이 된다
/// (넷째 레인은 법선을 다시 실어 버린다 — 3방향뿐이라 25%는 낭비지만, 남는 레인을 채우려
/// 접점을 섞으면 매니폴드마다 접점 수가 1~4로 달라 셔플이 이득을 먹는다).
///
/// 관성텐서는 **페어당 상수**다. 그래서 `Mat3x4::splat` 셋업은 호출자가 페어당 1회 하고
/// 이 함수는 접점마다 `ra`·`rb` splat만 한다 — 셋업을 접점마다 하면 3회 스칼라보다 느려진다.
///
/// **결과는 스칼라 경로와 비트 동일하다.** 레인당 연산 순서가 같기 때문이다
/// (`Vec3x4::cross`·`Mat3x4::mul_vec`·`Vec3x4::dot`이 스칼라판과 같은 결합 순서이고
/// FMA에 의존하지 않는다 — 설계 §8 L0). 두 곳만 스칼라와 표현이 다르고, 둘 다 값이 같다:
/// - `if k > 0.0` 분기 → `cmp_gt` + `blend`. NaN에서 둘 다 거짓이라 0.0으로 같다.
/// - 나눗셈 전에 k를 1.0으로 **치환**한다.
///
/// 치환에 대한 정직한 진술: 이것은 **값으로는 항등**이다. `one / k`로 두어 k≤0에서 inf가
/// 나와도 blend가 그 레인을 0.0으로 덮으므로 결과 비트는 같고, 그래서 **값 기반 테스트로는
/// 이 줄을 지킬 수 없다**(뮤테이션으로 확인했다 — 지우면 테스트가 통과한다). 남겨 두는 이유는
/// 0 나눗셈이 스칼라 경로가 만들지 않는 DZ 상태 플래그를 세우기 때문이고, 설계 §8의 FP 계약이
/// 규율하는 것은 FTZ/DAZ·라운딩이지 **예외 마스킹이 아니라서** 예외를 언마스크한 호스트에서는
/// 그 차이가 트랩으로 관측될 수 있다. 값이 같으니 "단순화"하고 싶어지는 줄이라 근거를 여기 남긴다.
#[inline]
fn effective_mass_x3(
    inv_m_sum: f32x4,
    inv_i_a: &Mat3x4,
    inv_i_b: &Mat3x4,
    ra: Vec3,
    rb: Vec3,
    dirs: Vec3x4,
) -> [f32; 4] {
    let rna = Vec3x4::splat(ra).cross(dirs);
    let rnb = Vec3x4::splat(rb).cross(dirs);
    // 결합 순서는 스칼라와 같다: ((inv_m_a + inv_m_b) + dotA) + dotB.
    let k = inv_m_sum + rna.dot(inv_i_a.mul_vec(rna)) + rnb.dot(inv_i_b.mul_vec(rnb));
    let zero = f32x4::splat(0.0);
    let one = f32x4::splat(1.0);
    let positive = k.cmp_gt(zero);
    positive
        .blend(one / positive.blend(k, one), zero)
        .to_array()
}

/// 솔버 핫패스의 **압축 상태**(설계 §3: SoA 핫패스).
///
/// 솔버가 `Body` 배열을 직접 훑으면 셰이프·재질까지 딸린 뚱뚱한 구조체를 제약 순서대로
/// 무작위 접근하게 된다. 실제로 점당 73ns가 나왔는데 산술은 40 flops 남짓이었다 —
/// 비용은 계산이 아니라 **메모리**였다. 솔버가 실제로 읽고 쓰는 6개 배열만 뽑아
/// 연속 메모리에 두고, 서브스텝 루프가 끝난 뒤 한 번만 바디로 되쓴다.
#[derive(Default, Clone)]
pub struct SolverState {
    pub lin: Vec<Vec3>,
    pub ang: Vec<Vec3>,
    pub pos: Vec<Vec3>,
    pub rot: Vec<Quat>,
    pub inv_m: Vec<f32>,
    pub inv_i: Vec<Mat3>,
    /// 동적 바디만 임펄스를 받는다 (static/kinematic은 inv_m=0이지만 명시 게이트가 더 싸다).
    pub dynamic: Vec<bool>,
}

impl SolverState {
    /// 바디에서 상태를 **제자리로** 채운다. 매 스텝 새 Vec을 만들면 할당 비용이
    /// 솔버 이득을 먹는다(실측: SoA 전환이 되레 1.3ms 느려졌다) — 버퍼는 월드가 보관한다.
    /// 용량만 미리 확보한다 — **캐노니컬 변이 전에** 호출해 실패를 여기서 끝낸다.
    pub fn reserve_for(&mut self, n: usize) -> Result<(), crate::world::ApiError> {
        // **`try_reserve`의 인자는 `len`에 더할 여분이다.** 목표에서 `capacity`를 빼면
        // `len < capacity < 목표`일 때 모자란 예약이 통과하고(예: len=100·cap=128·목표=1000
        // → 보장 972), 그 뒤 `fill_from`의 `resize`가 **wake 폐쇄 뒤에** 재할당한다
        // (확인검수 rev_071f1cd9 MAJOR — `NarrowOut::reserve`와 같은 계열의 잔존).
        macro_rules! res {
            ($v:expr) => {
                $v.try_reserve(n.saturating_sub($v.len()))
                    .map_err(|_| crate::world::ApiError::ResourceLimit)?;
            };
        }
        res!(self.lin);
        res!(self.ang);
        res!(self.pos);
        res!(self.rot);
        res!(self.inv_m);
        res!(self.inv_i);
        res!(self.dynamic);
        Ok(())
    }

    /// 목표 길이 `n`을 넘겨 **쥐고만 있는** 용량의 바이트. 회계는 현재 길이 기준인데
    /// `Vec` 용량은 저절로 줄지 않으므로, 큰 월드를 한 번 돌린 뒤 작은 월드가 되면
    /// 그 차이가 회계 밖에 남는다(확인검수 rev_21c1a97f MAJOR).
    pub fn excess_capacity_bytes(&self, n: usize) -> usize {
        let mut bytes = 0usize;
        macro_rules! add {
            ($v:expr, $t:ty) => {
                bytes = bytes.saturating_add(
                    $v.capacity()
                        .saturating_sub(n)
                        .saturating_mul(core::mem::size_of::<$t>()),
                );
            };
        }
        add!(self.lin, Vec3);
        add!(self.ang, Vec3);
        add!(self.pos, Vec3);
        add!(self.rot, Quat);
        add!(self.inv_m, f32);
        add!(self.inv_i, Mat3);
        add!(self.dynamic, bool);
        bytes
    }

    /// 부푼 용량을 **해제**한다 — 축소가 아니라 통째로 버린다.
    ///
    /// `Vec::shrink_to`는 더 작은 할당으로 **복사**하느라 재할당을 시도할 수 있고,
    /// 실패하면 fallible이 아니라 abort다. 이 함수가 불리는 시점은 스크래치가 상한을
    /// 넘어 자원이 빠듯한 바로 그 순간이라, 거기서 abort하면 `ResourceLimit`을 돌려주려던
    /// 보호 경로가 프로세스를 죽인다(확인검수 rev_de34bbb5 MAJOR).
    /// `Vec::new()` 대입은 **할당 없이** 옛 버퍼를 반납한다. 용량은 뒤따르는 통합
    /// fallible 블록(`reserve_for`)이 wake 폐쇄 이전에 다시 잡는다.
    pub fn release(&mut self) {
        self.lin = Vec::new();
        self.ang = Vec::new();
        self.pos = Vec::new();
        self.rot = Vec::new();
        self.inv_m = Vec::new();
        self.inv_i = Vec::new();
        self.dynamic = Vec::new();
    }

    /// 테스트 전용 — 예약이 목표에 닿았는지 보기 위한 창.
    #[cfg(test)]
    pub fn lin_capacity(&self) -> usize {
        self.lin.capacity()
    }

    /// 보유 용량(위생 판정용).
    pub fn lin_capacity_pub(&self) -> usize {
        self.lin.capacity()
    }

    /// 용량이 확보된 상태에서 값을 채운다 (실패하지 않는다).
    pub fn fill_from(&mut self, bodies: &[Body], alive: &[bool], oob: &[bool]) {
        let n = bodies.len();
        macro_rules! fit {
            ($v:expr, $d:expr) => {
                $v.clear();
                $v.resize(n, $d);
            };
        }
        fit!(self.lin, Vec3::ZERO);
        fit!(self.ang, Vec3::ZERO);
        fit!(self.pos, Vec3::ZERO);
        fit!(self.rot, Quat::IDENTITY);
        fit!(self.inv_m, 0.0);
        fit!(self.inv_i, Mat3::from_diagonal(Vec3::ZERO));
        fit!(self.dynamic, false);
        for i in 0..n {
            let b = &bodies[i];
            // **OOB는 솔버에서도 비활성**이다(리뷰 BLOCKER: `dynamic`을 타입만 보고 정해서
            // OOB에 임펄스가 들어가고 그 속도가 바디로 되쓰였다). 적분 게이트만으로는 부족하다.
            let active = alive[i] && !oob[i];
            self.lin[i] = if active { b.linvel } else { Vec3::ZERO };
            self.ang[i] = if active { b.angvel } else { Vec3::ZERO };
            self.pos[i] = b.pose.pos;
            self.rot[i] = b.pose.rot;
            self.inv_m[i] = if active { b.inv_mass } else { 0.0 };
            self.dynamic[i] = active && b.is_dynamic();
        }
    }

    /// 솔버 결과를 바디로 되쓴다 (스텝당 1회 — 핫패스 밖).
    pub fn write_back(&self, bodies: &mut [Body], alive: &[bool]) {
        for i in 0..bodies.len() {
            if alive[i] {
                bodies[i].linvel = self.lin[i];
                bodies[i].angvel = self.ang[i];
                bodies[i].pose.pos = self.pos[i];
                bodies[i].pose.rot = self.rot[i];
            }
        }
    }

    #[inline]
    fn vel_at(&self, i: usize, r: Vec3) -> Vec3 {
        self.lin[i] + self.ang[i].cross(r)
    }
    #[inline]
    fn apply(&mut self, i: usize, imp: Vec3, r: Vec3) {
        if self.dynamic[i] {
            self.lin[i] += imp * self.inv_m[i];
            self.ang[i] += self.inv_i[i].mul_vec(r.cross(imp));
        }
    }
    #[inline]
    fn apply_ang(&mut self, i: usize, ang: Vec3) {
        if self.dynamic[i] {
            self.ang[i] += self.inv_i[i].mul_vec(ang);
        }
    }
    #[inline]
    pub fn vel_at_pub(&self, i: usize, r: Vec3) -> Vec3 {
        self.vel_at(i, r)
    }
    #[inline]
    pub fn apply_pub(&mut self, i: usize, imp: Vec3, r: Vec3) {
        self.apply(i, imp, r)
    }
    #[inline]
    pub fn apply_ang_pub(&mut self, i: usize, ang: Vec3) {
        self.apply_ang(i, ang)
    }
    /// **활성 바디만** 갱신한다 — 잠든 바디는 자세가 변하지 않아 값이 그대로다.
    pub fn refresh_inv_i_active(&mut self, local: &[Vec3], active: &[u32]) {
        for &i in active {
            let i = i as usize;
            if self.dynamic[i] {
                self.inv_i[i] = Mat3::rotate_inertia(local[i], self.rot[i]);
            }
        }
    }
}

/// 캐시된 접점 임펄스 (스텝 간 warm start — 설계 §3 PCM).
/// generic witness 접점의 warm start 매칭 규칙 (리뷰 PARTIAL MAJOR).
///
/// 위치 근접 **하나만** 보면 양방향으로 틀린다: 기하 척도만 쓰면 반두께 1cm 판의 허용치가
/// 2.5mm라 60Hz·1m/s 정상 슬라이딩(프레임당 16.7mm)에서 매 프레임 warm start가 끊기고,
/// 반대로 큰 반지름 곡면에서는 같은 비율이 다른 feature로의 점프를 통과시킨다.
/// 그래서 세 가지를 함께 쓴다:
/// - **기하 바닥** = 최소 특징 크기의 이 비율 (정지 접촉의 수치 흔들림 흡수)
const WARM_MATCH_GEO: f32 = 0.25;
/// - **운동 항** = 접점이 이번 스텝에 **정당하게 움직일 수 있는 상한** × 여유.
///   상대 슬립 속도를 쓰면 안 된다: 구가 구르면 슬립은 0인데 기하 접점은 구의 선속도만큼
///   월드에서 이동한다(실측: 슬립 기준으로는 계승률 0.01). 상한은 두 바디의 선속도와
///   회전이 접점에 만드는 이동량의 합이다.
const WARM_MATCH_MOTION: f32 = 1.0;
/// - **법선 게이트** = 캐시된 법선과의 정렬. 위치가 가까워도 접촉 면이 바뀌었으면
///   다른 접촉이다 — 곡면에서 feature가 점프해도 법선은 크게 돈다.
const WARM_MATCH_COS: f32 = 0.9;

#[derive(Clone, Copy, Default)]
pub struct CachedImpulse {
    pub normal: f32,
    /// 이전 기저 성분 (진단·직렬화용).
    pub tangent: [f32; 2],
    /// **월드 공간 마찰 임펄스**. 접선 기저는 법선에서 유도되므로(0.707 분기) 법선이 회전하면
    /// 같은 성분이 다른 월드 방향으로 재적용된다 — 월드 벡터를 저장해 새 접평면에 투영한다.
    pub tangent_world: Vec3,
    /// 접점의 월드 위치 (generic witness 접점의 근접 매칭용).
    pub position: Vec3,
    /// 캐시 시점의 접촉 **법선 방향**(`normal`은 법선 임펄스 크기라 이름을 구분한다).
    /// 위치가 가까워도 면이 바뀌었으면 다른 접촉이다.
    pub contact_normal: Vec3,
}

/// 매니폴드 → 제약 준비. `cache`는 전 스텝 (feature → 임펄스) — feature ID 일치 접점만 warm load.
/// 접점 Vec은 narrowphase(=wake 이후)에서 불가피하게 할당되므로 **fallible**하다 — wasm OOM을
/// trap이 아니라 `ResourceLimit`으로 돌려준다(교차검수 rev_2ade4bf3 MAJOR). 접점 수 ≤4라 실할당은
/// 미미하지만 trap 경로는 닫는다.
/// **접점 버퍼는 호출자가 준다** (2026-07-30). 예전에는 여기서 `Vec::new()`로 매번 새로 잡아
/// 스텝당 수천 회 할당이 났다 — 설계 §6의 "스텝 중 할당 0" 위반이었고 실측으로 확인했다
/// (구 더미 3000: 스텝당 6,136회). 호출자(`NarrowOut`)가 풀에서 꺼내 주면 두 번째 스텝부터
/// 할당이 사라진다.
///
/// **반환형을 키우지 않는다** — `[ContactPointConstraint; 4]` 인라인은 2026-07-26에
/// narrow +40%로 이미 기각됐다(`prepare`가 큰 구조체를 값으로 돌려주고 push가 복사한다).
/// 버퍼를 받아 채워 돌려주면 그 문제 없이 할당만 없앤다.
/// 풀 없이 부르는 편의판 — **테스트·단발 호출 전용**이다. 스텝 핫패스는 `prepare_into`로
/// 버퍼를 재사용해야 한다(그게 이 분리의 이유다).
#[cfg(test)]
pub fn prepare(
    bodies: &[Body],
    a: usize,
    b: usize,
    m: &Manifold,
    cache: &[(maru_collide::FeatureId, CachedImpulse)],
    dt: f32,
) -> Result<ContactConstraint, crate::world::ApiError> {
    prepare_into(bodies, a, b, m, cache, dt, Vec::new())
}

/// 유효질량 경로 A/B 노브 (**측정 전용**, 기본 = SIMD).
///
/// `MARU_SCALAR_EFFMASS=1`이면 스칼라 3회 경로로 돈다. 이 노브가 있는 이유는 A/B를
/// **같은 바이너리·같은 의존성 그래프**에서 번갈아 돌리기 위해서다 — 빌드를 갈아 A/B하면
/// 그래프가 달라져 가짜 회귀(+80% 실측 사례)가 난다. 두 경로는 비트 동일하므로
/// 이 노브는 결과를 바꾸지 않는다(결정론 테스트가 두 팔 모두를 지난다).
///
/// 읽기는 페어당 원자 로드 1회다(~1ns, 페어당 비용 수백 ns 대비 무해). 환경변수 파싱은
/// 프로세스당 1회다 — 핫패스에서 `env::var`를 부르면 페어마다 할당이 난다.
fn scalar_effective_mass_path() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(u8::MAX);
    let v = CACHED.load(Ordering::Relaxed);
    if v != u8::MAX {
        return v == 1;
    }
    let on = std::env::var("MARU_SCALAR_EFFMASS")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);
    CACHED.store(on as u8, Ordering::Relaxed);
    on
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_into(
    bodies: &[Body],
    a: usize,
    b: usize,
    m: &Manifold,
    cache: &[(maru_collide::FeatureId, CachedImpulse)],
    dt: f32,
    mut points: Vec<ContactPointConstraint>,
) -> Result<ContactConstraint, crate::world::ApiError> {
    // 기하 바닥은 형상 크기에 비례해야 한다(절대 상수 금지 — 톨러런스 정책과 동일 원칙).
    let geo_tol =
        WARM_MATCH_GEO * maru_collide::min_feature_size(&bodies[a].shape, &bodies[b].shape);
    let (ba, bb) = (&bodies[a], &bodies[b]);
    let n = m.normal;
    let tangents = tangent_basis(n);
    // **실제로 warm load된 접점 수**. 진단이 feature ID 일치만 세면 generic 접점의 위치·법선
    // 게이트가 얼마나 자주 매칭을 끊는지 보이지 않는다 — 지표가 거짓이면 회귀 테스트도 거짓이다.
    let mut warm_matched = 0u32;
    let (ia, ib) = (ba.inv_inertia_world(), bb.inv_inertia_world());
    let scalar_mass = scalar_effective_mass_path();
    points.clear();
    if points.try_reserve(m.points().len()).is_err() {
        return Err(crate::world::ApiError::ResourceLimit);
    }
    // 유효질량 4-wide 셋업 — **페어당 1회**(접점마다 같은 값이라 접점 루프 밖이어야 한다).
    //
    // **조건부인 것이 A/B의 정확성 조건이다** (교차검수 rev_4976e200 BLOCKER). 무조건 셋업하면
    // 스칼라 팔이 *옛 경로 + 쓰지도 않는 셋업*이 되어 기준선이 부풀고, SIMD 순이득을 그만큼
    // 과대평가한다 — 컴파일러가 이 순수 계산을 분기 안으로 sink해 주기를 **기대하면 안 된다**
    // (클로저 경계를 넘고, 분기 조건이 런타임 값이다). 실제로 1차 측정이 그 상태였고,
    // 보고한 이득이 1~2%뿐이라 결론이 뒤집힐 수 있는 크기였다.
    // 접점 0개 매니폴드에서 셋업하지 않는 것도 같은 이유다(할 일이 없는데 값을 치른다).
    let simd_setup = if scalar_mass || m.is_empty() {
        None
    } else {
        Some((
            f32x4::splat(ba.inv_mass + bb.inv_mass),
            Mat3x4::splat(ia),
            Mat3x4::splat(ib),
            Vec3x4::from_array([n, tangents[0], tangents[1], n]),
        ))
    };
    points.extend(m.points().iter().map(|p| {
        let ra = p.position - ba.pose.pos;
        let rb = p.position - bb.pose.pos;
        let vn = n.dot(ba.velocity_at(ra) - bb.velocity_at(rb));
        // warm start 매칭 규칙 (리뷰 MAJOR):
        // - 클리핑 접점: feature ID가 계보를 담아 안정적 → ID 일치
        // - generic witness 접점: pair 안에서 ID를 공유하므로(구-박스 접점이 면→모서리로
        //   이동해도 같은 ID) **위치 근접**까지 요구한다. 점프하면 임펄스를 물려주지 않는다.
        // **자식 네임스페이스 비트는 벗기고 판정한다** — 복합 자식의 generic 접점은
        // 0xFFFF_0000 | (자식쌍<<9)라 정확 일치로는 generic으로 안 보이고, 그러면 위치
        // 근접 게이트 없이 ID만으로 warm-match돼 점프한 접점이 낡은 임펄스를 승계한다
        // (이 게이트가 애초에 막던 결함의 부활 — 자체 검수 발견).
        let generic =
            (p.feature.0 & !maru_collide::FEATURE_CHILD_MASK) == maru_collide::FEATURE_GENERIC;
        // 허용치 = 기하 바닥 + 접점의 정당한 이동 상한. 정지 접촉은 기하 바닥만 쓴다.
        let drift = ba.linvel.length()
            + ba.angvel.length() * ra.length()
            + bb.linvel.length()
            + bb.angvel.length() * rb.length();
        let match_tol = geo_tol + WARM_MATCH_MOTION * drift * dt;
        let match_tol2 = match_tol * match_tol;
        let hit = cache
            .iter()
            .find(|(f, c)| {
                *f == p.feature
                    && (!generic
                        || ((c.position - p.position).length_sq() <= match_tol2
                                // 면이 바뀌었으면 다른 접촉 (곡면 feature 점프 차단)
                                && c.contact_normal.dot(n) >= WARM_MATCH_COS))
            })
            .map(|(_, c)| *c);
        if hit.is_some() {
            warm_matched += 1;
        }
        let warm = hit.unwrap_or_default();
        // 월드 마찰 임펄스를 현재 접평면에 재투영 (기저 회전 대응)
        let warm_t = [
            warm.tangent_world.dot(tangents[0]),
            warm.tangent_world.dot(tangents[1]),
        ];
        // 유효질량 = [법선, 접선0, 접선1]. 두 경로는 **비트 동일**이고(위 주석 + 단위 테스트),
        // 스위치는 같은 바이너리로 A/B를 돌리기 위한 측정 노브다.
        let mass = if scalar_mass {
            [
                effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, n),
                effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, tangents[0]),
                effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, tangents[1]),
                0.0,
            ]
        } else {
            let (inv_m_sum, ia4, ib4, dirs) = simd_setup
                .as_ref()
                .expect("접점이 있는 SIMD 경로에는 셋업이 있다");
            effective_mass_x3(*inv_m_sum, ia4, ib4, ra, rb, *dirs)
        };
        ContactPointConstraint {
            feature: p.feature,
            position: p.position,
            anchor_a: ba.pose.rot.conjugate().rotate(ra),
            anchor_b: bb.pose.rot.conjugate().rotate(rb),
            base_separation: p.separation,
            normal_mass: mass[0],
            tangent_mass: [mass[1], mass[2]],
            normal_impulse: warm.normal,
            tangent_impulse: warm_t,
            max_normal_impulse: 0.0,
            ra,
            rb,
            approach_vn: vn,
        }
    }));
    Ok(ContactConstraint {
        warm_matched,
        a,
        b,
        normal: n,
        tangents,
        friction: (ba.friction * bb.friction).sqrt(),
        restitution: ba.restitution.max(bb.restitution),
        points,
    })
}

/// 현재 자세로 레버암을 갱신한다 (서브스텝당 1회, 위치 적분 뒤 1회).
///
/// **유효질량은 여기서 다시 구하지 않는다 — 알려진 불일치이고, 측정으로 평가한 트레이드다.**
/// `prepare`는 프레임 시작 자세의 `inv_inertia_world()`로 `normal_mass`·`tangent_mass`를
/// 한 번 계산하는데, 월드는 서브스텝마다 `refresh_inv_i_active`로 `st.inv_i`를 갱신한다.
/// 그래서 솔버는 낡은 K⁻¹로 임펄스 크기를 정하고 그 임펄스를 새 K로 적용한다
/// (GPT 검수 rev_29a3ec4b가 MAJOR로 지목했고, 3D 이방성 관성에서만 나는 계열이다).
///
/// 여기서 재계산하도록 고쳐 실측한 결과(2026-07-26):
/// - 서브스텝 절벽(5=폭발/6=안정) **불변** — 검수의 예측(과보정 발산)은 재현되지 않았다
/// - 봉투 **불변** (40·60단 통과, 65단 실패)
/// - 이 결함을 최대로 노출시키려 만든 장면(`maru-bench --bin spin`: 각속도 90 rad/s로
///   도는 이방성 막대 24개의 접촉)에서도 에너지 차이 8%뿐이고 양쪽 다 발산하지 않는다
/// - 비용 **+10%** (40단 nosleep 200스텝, 교차 4쌍)
///
/// 반대 방향(서브스텝 `inv_i` 갱신 제거)도 재봤는데 그건 회전 장면을 크게 바꾼다
/// (E1/E0 0.20 → 0.30, angmax 57 → 88) — 그러니 `inv_i` 갱신은 유지한다.
/// 결론: 이론상 불일치는 실재하지만 관측되는 대가가 없고, 10%는 관측된다.
pub fn refresh_anchors(v: &SolveView, c: &mut ContactConstraint) {
    let (qa, qb) = (v.rot(c.a), v.rot(c.b));
    for p in &mut c.points {
        p.ra = qa.rotate(p.anchor_a);
        p.rb = qb.rotate(p.anchor_b);
    }
}

/// 솔브 패스가 보는 상태 — **속도만 쓰고 나머지는 읽는다**.
///
/// 병렬 색 구간에서 참여자들이 공유하는 유일한 상태이고, 그래서 이 타입이 병렬 솔버의
/// 안전 계약이 사는 곳이다.
///
/// - `lin`/`ang`: **원소별 raw 접근**. 참여자들이 같은 배열의 서로 다른 원소를 쓰므로
///   슬라이스 참조를 만들면(공유든 가변이든) 그 자체로 규칙 위반이다 — 원소 하나짜리
///   접근만 한다. 서로 다른 원소만 만진다는 보장은 **그래프 컬러링**에서 온다:
///   같은 색의 두 제약은 dynamic 바디를 공유하지 않고(`color_constraints`), 솔브가
///   write하는 집합(`dynamic`)은 컬러링이 conflict로 본 집합과 같은 술어다.
/// - 나머지(`pos`/`rot`/`inv_m`/`inv_i`/`dynamic`): 패스 **동안 불변**이라 평범한
///   공유 슬라이스다. 서브스텝 사이의 위치 적분·`inv_i` 갱신은 직렬 구간에서 일어난다.
///
/// `SolverState`를 통째로 넘기지 않는 이유가 이것이다: 넘기면 병렬 커널이 실수로
/// `st.lin[i]`를 읽을 수 있고, 그 순간 동시 변이 중인 배열에 공유 참조가 생긴다.
/// 뷰는 그 경로를 **타입에서** 없앤다.
#[derive(Clone, Copy)]
pub struct SolveView<'a> {
    lin: *mut Vec3,
    ang: *mut Vec3,
    pos: &'a [Vec3],
    rot: &'a [Quat],
    inv_m: &'a [f32],
    inv_i: &'a [Mat3],
    dynamic: &'a [bool],
}

/// SAFETY: 원소 비겹침은 호출자(컬러링) 계약이고, 뷰 자체는 `&mut SolverState`에서만
/// 만들어지므로 뷰가 사는 동안 다른 접근 경로가 없다.
unsafe impl Send for SolveView<'_> {}
unsafe impl Sync for SolveView<'_> {}

impl<'a> SolveView<'a> {
    pub fn new(st: &'a mut SolverState) -> Self {
        Self {
            lin: st.lin.as_mut_ptr(),
            ang: st.ang.as_mut_ptr(),
            pos: &st.pos,
            rot: &st.rot,
            inv_m: &st.inv_m,
            inv_i: &st.inv_i,
            dynamic: &st.dynamic,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pos.len()
    }
    #[inline]
    pub fn lin(&self, i: usize) -> Vec3 {
        debug_assert!(i < self.len());
        // SAFETY: 인덱스 유효 + 이 원소를 쓰는 참여자는 자기 하나(컬러링 계약).
        unsafe { *self.lin.add(i) }
    }
    #[inline]
    pub fn ang(&self, i: usize) -> Vec3 {
        debug_assert!(i < self.len());
        unsafe { *self.ang.add(i) }
    }
    #[inline]
    pub fn set_lin(&self, i: usize, v: Vec3) {
        debug_assert!(i < self.len());
        unsafe { *self.lin.add(i) = v }
    }
    #[inline]
    pub fn set_ang(&self, i: usize, v: Vec3) {
        debug_assert!(i < self.len());
        unsafe { *self.ang.add(i) = v }
    }
    #[inline]
    pub fn pos(&self, i: usize) -> Vec3 {
        self.pos[i]
    }
    #[inline]
    pub fn rot(&self, i: usize) -> Quat {
        self.rot[i]
    }
    #[inline]
    pub fn inv_m(&self, i: usize) -> f32 {
        self.inv_m[i]
    }
    #[inline]
    pub fn inv_i(&self, i: usize) -> Mat3 {
        self.inv_i[i]
    }
    #[inline]
    pub fn dynamic(&self, i: usize) -> bool {
        self.dynamic[i]
    }
}

/// 제약 하나를 푸는 동안 레지스터에 올려두는 바디 상태 (Box2D v3 `b2BodyState` 동형).
///
/// `st.apply`/`st.vel_at`을 접점마다 부르면 접점 1개당 `lin·ang·inv_m·inv_i·dynamic`
/// 5개 배열을 바디 인덱스로 무작위 접근하고, 그걸 법선·마찰 2축·원판투영까지 4번 반복한다.
/// 산술은 그대로 두고 **왕복만** 없앤다 — 제약 진입 시 1회 로드, 종료 시 1회 스토어.
/// 갱신 순서가 동일하므로 결과는 비트 동일하다(스냅샷 해시 회귀가 이를 잠근다).
#[derive(Clone, Copy)]
struct BodyReg {
    lin: Vec3,
    ang: Vec3,
    inv_m: f32,
    inv_i: Mat3,
    dynamic: bool,
}

impl BodyReg {
    #[inline]
    fn load(v: &SolveView, i: usize) -> Self {
        Self {
            lin: v.lin(i),
            ang: v.ang(i),
            inv_m: v.inv_m(i),
            inv_i: v.inv_i(i),
            dynamic: v.dynamic(i),
        }
    }
    #[inline]
    fn store(&self, v: &SolveView, i: usize) {
        if self.dynamic {
            v.set_lin(i, self.lin);
            v.set_ang(i, self.ang);
        }
    }
    #[inline]
    fn vel_at(&self, r: Vec3) -> Vec3 {
        self.lin + self.ang.cross(r)
    }
    #[inline]
    fn apply(&mut self, imp: Vec3, r: Vec3) {
        if self.dynamic {
            self.lin += imp * self.inv_m;
            self.ang += self.inv_i.mul_vec(r.cross(imp));
        }
    }
}

/// warm start: 누적 임펄스를 서브스텝 시작 추정으로 재적용 (Box2D v3 동형).
pub fn warm_start(v: &SolveView, c: &ContactConstraint) {
    let (mut ba, mut bb) = (BodyReg::load(v, c.a), BodyReg::load(v, c.b));
    for p in &c.points {
        let (ra, rb) = (p.ra, p.rb);
        let imp = c.normal * p.normal_impulse
            + c.tangents[0] * p.tangent_impulse[0]
            + c.tangents[1] * p.tangent_impulse[1];
        ba.apply(imp, ra);
        bb.apply(-imp, rb);
    }
    ba.store(v, c.a);
    bb.store(v, c.b);
}

// ===== SIMD 배치 (Unit 3 — 설계 §5) — 지금은 **참조 구현**이다 =====
//
// 프로덕션 경로는 `solver_block`(SoA 블록)이다. 아래 AoS-게더 x4 경로는 지우지 않고
// **차등 테스트의 오라클**로 남긴다: 블록 경로가 이것과 레인당 비트 동일하다는 것이
// "SoA 전환은 성능 변경이지 물리 변경이 아니다"의 증거이고, 그 증거는 구현이 바뀔 때마다
// 다시 돌아야 한다(`solver_block::tests::block_path_matches_x4_path_bit_for_bit`).
// 그래서 `cfg(test)`다 — 릴리스 바이너리에는 들어가지 않는다.
//
// 같은 색의 4제약을 레인에 실어 warm_start/solve를 4-wide로 돈다. 같은 색은 공유 dynamic 바디가
// 없으므로 레인이 서로 독립 — 스캐터 충돌이 없다(static/kinematic만 겹칠 수 있고 그건 write 대상
// 아님). 게더/스캐터는 스칼라(WASM gather 명령 없음, 설계 §3 정직한 범위).
//
// **비트 동일 규율**: 레인당 연산 순서를 스칼라와 동일하게 유지하고(비FMA), 게이트(dynamic·
// 점 존재)는 산술 마스킹이 아니라 **blend**로 한다 — `x + 0.0`이 `-0.0`을 `+0.0`으로 뒤집어
// state hash를 가르기 때문. blend는 게이트 off 레인의 원본 값을 비트 그대로 보존한다.

/// Vec3x4 → 레인별 Vec3 4개 (스캐터용).
#[inline]
#[cfg(test)]
fn vec3x4_lanes(v: Vec3x4) -> [Vec3; 4] {
    let (x, y, z) = (v.x.to_array(), v.y.to_array(), v.z.to_array());
    [
        Vec3::new(x[0], y[0], z[0]),
        Vec3::new(x[1], y[1], z[1]),
        Vec3::new(x[2], y[2], z[2]),
        Vec3::new(x[3], y[3], z[3]),
    ]
}

#[inline]
#[cfg(test)]
fn negx4(v: Vec3x4) -> Vec3x4 {
    Vec3x4 {
        x: -v.x,
        y: -v.y,
        z: -v.z,
    }
}

/// 레인별 blend: gate 마스크가 참인 레인은 `t`, 아니면 `f`(원본 보존).
#[inline]
#[cfg(test)]
fn blend_vec(gate: f32x4, t: Vec3x4, f: Vec3x4) -> Vec3x4 {
    Vec3x4 {
        x: gate.blend(t.x, f.x),
        y: gate.blend(t.y, f.y),
        z: gate.blend(t.z, f.z),
    }
}

/// 4제약 배치의 바디 상태 (SoA 레인) — 스칼라 `BodyReg`의 4-wide 대응.
#[derive(Clone, Copy)]
#[cfg(test)]
struct BodyRegx4 {
    lin: Vec3x4,
    ang: Vec3x4,
    inv_m: f32x4,
    inv_i: Mat3x4,
    /// dynamic 게이트 마스크 (레인당 all-1s/0). 비-dynamic 레인은 write 안 됨.
    dyn_mask: f32x4,
}

#[cfg(test)]
impl BodyRegx4 {
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
            inv_i: Mat3x4::from_array([
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

    /// dynamic·active 레인에만 임펄스 적용 (스칼라 `BodyReg::apply`의 4-wide, blend 게이트).
    /// `active` = 점 존재 마스크. 순서·연산은 스칼라와 동일 → 레인당 비트 동일.
    #[inline]
    fn apply(&mut self, imp: Vec3x4, r: Vec3x4, active: f32x4) {
        let gate = self.dyn_mask & active;
        let new_lin = self.lin + imp.scale(self.inv_m);
        let new_ang = self.ang + self.inv_i.mul_vec(r.cross(imp));
        self.lin = blend_vec(gate, new_lin, self.lin);
        self.ang = blend_vec(gate, new_ang, self.ang);
    }

    /// dynamic 레인만 바디로 되쓴다 (스칼라 `store`와 동일 게이트 — 비-dynamic은 불변).
    #[inline]
    fn store(&self, st: &mut SolverState, idx: [usize; 4]) {
        let (lin, ang) = (vec3x4_lanes(self.lin), vec3x4_lanes(self.ang));
        for k in 0..4 {
            if st.dynamic[idx[k]] {
                st.lin[idx[k]] = lin[k];
                st.ang[idx[k]] = ang[k];
            }
        }
    }

    /// 레인별 접점 속도 v + ω×r (스칼라 `BodyReg::vel_at`의 4-wide).
    #[inline]
    fn vel_at(&self, r: Vec3x4) -> Vec3x4 {
        self.lin + self.ang.cross(r)
    }
}

/// warm start의 4제약 배치판. 레인당 결과는 스칼라 `warm_start`와 **비트 동일**하다
/// (연산 순서 동일·비FMA·게이트 blend). 점 수가 다른 레인은 없는 점을 0/ZERO로 채우되,
/// blend `active` 마스크로 걸러 그 레인 값을 건드리지 않는다.
#[cfg(test)]
pub fn warm_start_x4(st: &mut SolverState, cs: [&ContactConstraint; 4]) {
    let a_idx = [cs[0].a, cs[1].a, cs[2].a, cs[3].a];
    let b_idx = [cs[0].b, cs[1].b, cs[2].b, cs[3].b];
    let mut ba = BodyRegx4::load(st, a_idx);
    let mut bb = BodyRegx4::load(st, b_idx);
    let normal = Vec3x4::from_array([cs[0].normal, cs[1].normal, cs[2].normal, cs[3].normal]);
    let t0 = Vec3x4::from_array([
        cs[0].tangents[0],
        cs[1].tangents[0],
        cs[2].tangents[0],
        cs[3].tangents[0],
    ]);
    let t1 = Vec3x4::from_array([
        cs[0].tangents[1],
        cs[1].tangents[1],
        cs[2].tangents[1],
        cs[3].tangents[1],
    ]);
    let maxpts = cs.iter().map(|c| c.points.len()).max().unwrap_or(0);
    for k in 0..maxpts {
        let p = [
            cs[0].points.get(k),
            cs[1].points.get(k),
            cs[2].points.get(k),
            cs[3].points.get(k),
        ];
        let ni = f32x4::from([
            p[0].map_or(0.0, |q| q.normal_impulse),
            p[1].map_or(0.0, |q| q.normal_impulse),
            p[2].map_or(0.0, |q| q.normal_impulse),
            p[3].map_or(0.0, |q| q.normal_impulse),
        ]);
        let ti0 = f32x4::from([
            p[0].map_or(0.0, |q| q.tangent_impulse[0]),
            p[1].map_or(0.0, |q| q.tangent_impulse[0]),
            p[2].map_or(0.0, |q| q.tangent_impulse[0]),
            p[3].map_or(0.0, |q| q.tangent_impulse[0]),
        ]);
        let ti1 = f32x4::from([
            p[0].map_or(0.0, |q| q.tangent_impulse[1]),
            p[1].map_or(0.0, |q| q.tangent_impulse[1]),
            p[2].map_or(0.0, |q| q.tangent_impulse[1]),
            p[3].map_or(0.0, |q| q.tangent_impulse[1]),
        ]);
        let ra = Vec3x4::from_array([
            p[0].map_or(Vec3::ZERO, |q| q.ra),
            p[1].map_or(Vec3::ZERO, |q| q.ra),
            p[2].map_or(Vec3::ZERO, |q| q.ra),
            p[3].map_or(Vec3::ZERO, |q| q.ra),
        ]);
        let rb = Vec3x4::from_array([
            p[0].map_or(Vec3::ZERO, |q| q.rb),
            p[1].map_or(Vec3::ZERO, |q| q.rb),
            p[2].map_or(Vec3::ZERO, |q| q.rb),
            p[3].map_or(Vec3::ZERO, |q| q.rb),
        ]);
        let has = f32x4::from([
            p[0].is_some() as u32 as f32,
            p[1].is_some() as u32 as f32,
            p[2].is_some() as u32 as f32,
            p[3].is_some() as u32 as f32,
        ])
        .cmp_gt(f32x4::splat(0.5));
        // imp = normal*ni + t0*ti0 + t1*ti1 (스칼라와 동일 좌→우 순서)
        let imp = normal.scale(ni) + t0.scale(ti0) + t1.scale(ti1);
        ba.apply(imp, ra, has);
        bb.apply(negx4(imp), rb, has);
    }
    ba.store(st, a_idx);
    bb.store(st, b_idx);
}

/// warm_start를 **색 범위 단위로 배치 디스패치** (Unit 3b). 같은 색 4제약은 SIMD(공유 dynamic
/// 바디 없음 → 레인 독립), 나머지(<4)와 오버플로 색(63, 공유 바디 가능)은 스칼라. 색 안의 제약은
/// dynamic 바디가 서로 겹치지 않아 처리 순서와 무관하고, 색 그룹 순서는 콜러드-스칼라와 동일하므로
/// 결과는 **비트 동일**하다(L0 해시 불변 — 기존 state_hash 테스트가 이를 잠근다).
#[cfg(test)]
pub fn warm_start_batched(
    st: &mut SolverState,
    constraints: &[ContactConstraint],
    color_ranges: &[(u32, u32)],
) {
    for (color, &(start, end)) in color_ranges.iter().enumerate() {
        let (start, end) = (start as usize, end as usize);
        // 색 63은 오버플로 버킷일 수 있어(공유 바디) 항상 스칼라 — 정상 색 63의 손실은 무시 가능.
        if color as u32 == COLOR_OVERFLOW {
            let v = SolveView::new(st);
            for c in &constraints[start..end] {
                warm_start(&v, c);
            }
            continue;
        }
        let mut i = start;
        while i + 4 <= end {
            warm_start_x4(
                st,
                [
                    &constraints[i],
                    &constraints[i + 1],
                    &constraints[i + 2],
                    &constraints[i + 3],
                ],
            );
            i += 4;
        }
        {
            let v = SolveView::new(st);
            for c in &constraints[i..end] {
                warm_start(&v, c);
            }
        }
    }
}

/// 배치 solve 파라미터 (Unit 3c). bias 페이즈면 `relax=false`+per-lane softness 선택, relax면
/// `relax=true`(bias 제거). soft=동적-동적, static_soft=동적-정적 — 레인마다 both-dynamic로 고른다.
pub struct BatchSolveParams {
    pub inv_h: f32,
    pub relax: bool,
    pub soft: Softness,
    pub static_soft: Softness,
    pub max_bias: f32,
}

/// solve의 4제약 배치판 (Unit 3c). 레인당 스칼라 `solve`와 **비트 동일**하다: 연산 순서 동일·
/// 비FMA, 분기(s>0 speculative·Coulomb 원판)는 blend, 없는 점·비cond 레인은 마스크로 원본 보존
/// (x+0.0 부호0 뒤집힘·0/0 NaN 오염 차단 — NaN은 select 밖으로 못 샌다). 점 변이(normal/tangent
/// impulse·max)는 레인별로 각 제약 점에 스캐터. 같은 색이라 dynamic 바디 disjoint → 레인 독립.
#[cfg(test)]
pub fn solve_x4(st: &mut SolverState, mut cs: [&mut ContactConstraint; 4], p: &BatchSolveParams) {
    let a_idx = [cs[0].a, cs[1].a, cs[2].a, cs[3].a];
    let b_idx = [cs[0].b, cs[1].b, cs[2].b, cs[3].b];
    let mut ba = BodyRegx4::load(st, a_idx);
    let mut bb = BodyRegx4::load(st, b_idx);
    let pos_a = Vec3x4::from_array([
        st.pos[a_idx[0]],
        st.pos[a_idx[1]],
        st.pos[a_idx[2]],
        st.pos[a_idx[3]],
    ]);
    let pos_b = Vec3x4::from_array([
        st.pos[b_idx[0]],
        st.pos[b_idx[1]],
        st.pos[b_idx[2]],
        st.pos[b_idx[3]],
    ]);
    let normal = Vec3x4::from_array([cs[0].normal, cs[1].normal, cs[2].normal, cs[3].normal]);
    let t0 = Vec3x4::from_array([
        cs[0].tangents[0],
        cs[1].tangents[0],
        cs[2].tangents[0],
        cs[3].tangents[0],
    ]);
    let t1 = Vec3x4::from_array([
        cs[0].tangents[1],
        cs[1].tangents[1],
        cs[2].tangents[1],
        cs[3].tangents[1],
    ]);
    let friction = f32x4::from([
        cs[0].friction,
        cs[1].friction,
        cs[2].friction,
        cs[3].friction,
    ]);
    let (zero, one) = (f32x4::splat(0.0), f32x4::splat(1.0));
    // per-lane softness (bias 페이즈): both-dynamic면 soft, 아니면 static_soft.
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
    let maxpts = cs.iter().map(|c| c.points.len()).max().unwrap_or(0);
    for k in 0..maxpts {
        let q = [
            cs[0].points.get(k),
            cs[1].points.get(k),
            cs[2].points.get(k),
            cs[3].points.get(k),
        ];
        let has = f32x4::from([
            q[0].is_some() as u32 as f32,
            q[1].is_some() as u32 as f32,
            q[2].is_some() as u32 as f32,
            q[3].is_some() as u32 as f32,
        ])
        .cmp_gt(f32x4::splat(0.5));
        let ra = Vec3x4::from_array([
            q[0].map_or(Vec3::ZERO, |p| p.ra),
            q[1].map_or(Vec3::ZERO, |p| p.ra),
            q[2].map_or(Vec3::ZERO, |p| p.ra),
            q[3].map_or(Vec3::ZERO, |p| p.ra),
        ]);
        let rb = Vec3x4::from_array([
            q[0].map_or(Vec3::ZERO, |p| p.rb),
            q[1].map_or(Vec3::ZERO, |p| p.rb),
            q[2].map_or(Vec3::ZERO, |p| p.rb),
            q[3].map_or(Vec3::ZERO, |p| p.rb),
        ]);
        let base_sep = f32x4::from([
            q[0].map_or(0.0, |p| p.base_separation),
            q[1].map_or(0.0, |p| p.base_separation),
            q[2].map_or(0.0, |p| p.base_separation),
            q[3].map_or(0.0, |p| p.base_separation),
        ]);
        let normal_mass = f32x4::from([
            q[0].map_or(0.0, |p| p.normal_mass),
            q[1].map_or(0.0, |p| p.normal_mass),
            q[2].map_or(0.0, |p| p.normal_mass),
            q[3].map_or(0.0, |p| p.normal_mass),
        ]);
        let tmi = |j: usize| {
            f32x4::from([
                q[0].map_or(0.0, |p| p.tangent_mass[j]),
                q[1].map_or(0.0, |p| p.tangent_mass[j]),
                q[2].map_or(0.0, |p| p.tangent_mass[j]),
                q[3].map_or(0.0, |p| p.tangent_mass[j]),
            ])
        };
        let (tm0, tm1) = (tmi(0), tmi(1));
        let mut ni = f32x4::from([
            q[0].map_or(0.0, |p| p.normal_impulse),
            q[1].map_or(0.0, |p| p.normal_impulse),
            q[2].map_or(0.0, |p| p.normal_impulse),
            q[3].map_or(0.0, |p| p.normal_impulse),
        ]);
        let mut ti0 = f32x4::from([
            q[0].map_or(0.0, |p| p.tangent_impulse[0]),
            q[1].map_or(0.0, |p| p.tangent_impulse[0]),
            q[2].map_or(0.0, |p| p.tangent_impulse[0]),
            q[3].map_or(0.0, |p| p.tangent_impulse[0]),
        ]);
        let mut ti1 = f32x4::from([
            q[0].map_or(0.0, |p| p.tangent_impulse[1]),
            q[1].map_or(0.0, |p| p.tangent_impulse[1]),
            q[2].map_or(0.0, |p| p.tangent_impulse[1]),
            q[3].map_or(0.0, |p| p.tangent_impulse[1]),
        ]);
        let mut mni = f32x4::from([
            q[0].map_or(0.0, |p| p.max_normal_impulse),
            q[1].map_or(0.0, |p| p.max_normal_impulse),
            q[2].map_or(0.0, |p| p.max_normal_impulse),
            q[3].map_or(0.0, |p| p.max_normal_impulse),
        ]);

        // s = normal·(pa − pb) + base
        let s = normal.dot((pos_a + ra) - (pos_b + rb)) + base_sep;
        // --- 법선 ---
        let vn = normal.dot(ba.vel_at(ra) - bb.vel_at(rb));
        let s_pos = s.cmp_gt(zero);
        // else 분기: relax면 (0,1,0), bias면 ((bias_rate·s).max(-max_bias), mass_scale, impulse_scale)
        let (bias_else, mass_else, imp_else) = if p.relax {
            (zero, one, zero)
        } else {
            ((sf_bias * s).max(neg_max_bias), sf_mass, sf_imp)
        };
        let bias = s_pos.blend(s * inv_h, bias_else);
        let mass_scale = s_pos.blend(one, mass_else);
        let impulse_scale = s_pos.blend(zero, imp_else);
        let impulse = -normal_mass * mass_scale * (vn + bias) - impulse_scale * ni;
        let new_total = (ni + impulse).max(zero);
        let applied = new_total - ni;
        ni = new_total;
        mni = mni.max(new_total);
        let imp = normal.scale(applied);
        ba.apply(imp, ra, has);
        bb.apply(negx4(imp), rb, has);
        // --- 마찰: 접선 2x2 블록 해 + Coulomb 원판 투영을 한 번의 적용에 접는다 ---
        // 스칼라 `solve`·블록 `solve_block`과 **같은 식·같은 순서**다(세 경로 비트 동일 계약).
        let max_f = friction * ni;
        let dv = ba.vel_at(ra) - bb.vel_at(rb);
        let (vt0, vt1) = (t0.dot(dv), t1.dot(dv));
        let n0 = ti0 - tm0 * vt0;
        let n1 = ti1 - tm1 * vt1;
        let mag2 = n0 * n0 + n1 * n1;
        // **0/0을 실행하지 않는다** (교차검수 rev_87748fe6 MAJOR). 정지 접촉(누적 임펄스 0,
        // 접선 속도 0)이나 블록의 빈 레인은 `max_f = 0`·`mag2 = 0`이 흔한데, `scale`을 조건과
        // 무관하게 먼저 계산하면 매 반복 `0/0` invalid 연산이 난다. `blend`는 값의 NaN만 가리고
        // **FP 예외 플래그는 이미 오른다** — 이 저장소의 `fp_verify()`는 FTZ/DAZ와 반올림만
        // 보므로 예외를 언마스크한 호스트에서는 트랩하고, 기본 환경에서도 sticky 플래그가 오염된다.
        // 클램프가 필요 없는 레인은 분모를 1로 갈아 끼운다(그 레인의 `scale`은 어차피 버려진다).
        let cond = mag2.cmp_gt(max_f * max_f) & mag2.cmp_gt(zero);
        let scale = max_f / cond.blend(mag2, one).sqrt();
        let n0 = cond.blend(n0 * scale, n0);
        let n1 = cond.blend(n1 * scale, n1);
        let (d0, d1) = (n0 - ti0, n1 - ti1);
        ti0 = n0;
        ti1 = n1;
        let imp = t0.scale(d0) + t1.scale(d1);
        ba.apply(imp, ra, has);
        bb.apply(negx4(imp), rb, has);

        // 점 변이 스캐터 (레인별, 존재하는 점만)
        let (ni_l, ti0_l, ti1_l, mni_l) = (
            ni.to_array(),
            ti0.to_array(),
            ti1.to_array(),
            mni.to_array(),
        );
        for (lane, c) in cs.iter_mut().enumerate() {
            if let Some(pt) = c.points.get_mut(k) {
                pt.normal_impulse = ni_l[lane];
                pt.tangent_impulse = [ti0_l[lane], ti1_l[lane]];
                pt.max_normal_impulse = mni_l[lane];
            }
        }
    }
    ba.store(st, a_idx);
    bb.store(st, b_idx);
}

/// solve를 색 범위 단위로 배치 디스패치 (Unit 3c). 색 내 4제약 SIMD, 나머지·오버플로 스칼라.
#[cfg(test)]
pub fn solve_batched(
    st: &mut SolverState,
    constraints: &mut [ContactConstraint],
    color_ranges: &[(u32, u32)],
    p: &BatchSolveParams,
) {
    for (color, &(start, end)) in color_ranges.iter().enumerate() {
        let (start, end) = (start as usize, end as usize);
        let overflow = color as u32 == COLOR_OVERFLOW;
        for chunk in constraints[start..end].chunks_mut(4) {
            if chunk.len() == 4 && !overflow {
                if let [c0, c1, c2, c3] = chunk {
                    solve_x4(st, [c0, c1, c2, c3], p);
                    continue;
                }
            }
            for c in chunk.iter_mut() {
                let soft = if p.relax {
                    None
                } else if st.dynamic[c.a] && st.dynamic[c.b] {
                    Some((&p.soft, p.max_bias))
                } else {
                    Some((&p.static_soft, p.max_bias))
                };
                let v = SolveView::new(st);
                solve(&v, c, p.inv_h, soft);
            }
        }
    }
}

/// 속도 임펄스 해석. `soft`=bias 페이즈, `relax`(soft=None)=bias 제거 페이즈.
pub fn solve(
    v: &SolveView,
    c: &mut ContactConstraint,
    inv_h: f32,
    soft: Option<(&Softness, f32)>, // (계수, max_bias_speed)
) {
    let (mut ba, mut bb) = (BodyReg::load(v, c.a), BodyReg::load(v, c.b));
    let (pos_a, pos_b) = (v.pos(c.a), v.pos(c.b));
    for p in &mut c.points {
        let (ra, rb) = (p.ra, p.rb);
        // TGS: 현재 자세로 separation 추적. A가 +n로 이동하면 s 증가해야 하므로
        // s = n·(pa − pb) + base (prepare 시점엔 앵커가 같은 월드점 → s = base).
        let pa = pos_a + ra;
        let pb = pos_b + rb;
        let s = c.normal.dot(pa - pb) + p.base_separation;

        // --- 법선 ---
        let vn = c.normal.dot(ba.vel_at(ra) - bb.vel_at(rb));
        let (bias, mass_scale, impulse_scale) = match soft {
            Some((sf, max_bias)) => {
                if s > 0.0 {
                    // speculative: 이번 서브스텝에 닿을 만큼만 감속 (임팩트 게이트 유지)
                    (s * inv_h, 1.0, 0.0)
                } else {
                    (
                        (sf.bias_rate * s).max(-max_bias),
                        sf.mass_scale,
                        sf.impulse_scale,
                    )
                }
            }
            None => {
                if s > 0.0 {
                    (s * inv_h, 1.0, 0.0)
                } else {
                    (0.0, 1.0, 0.0)
                }
            }
        };
        // 부호 계약: vn = n·(va−vb) = ds/dt — 분리=양수, 접근=음수.
        // s<0(침투): bias<0 → impulse>0 (밀어냄). s>0(speculative): bias=s/h → 초과 접근분만 감속.
        let impulse = -p.normal_mass * mass_scale * (vn + bias) - impulse_scale * p.normal_impulse;
        let new_total = (p.normal_impulse + impulse).max(0.0);
        let applied = new_total - p.normal_impulse;
        p.normal_impulse = new_total;
        p.max_normal_impulse = p.max_normal_impulse.max(new_total);
        let imp = c.normal * applied;
        ba.apply(imp, ra);
        bb.apply(-imp, rb);

        // --- 마찰: 접선 **2x2 블록 해** + Coulomb **원판** 투영을 한 번의 적용에 접는다 ---
        //
        // 사각형 클램프는 대각 슬라이드에서 √2·μN까지 허용해 41% 과도 감속하고 기저 의존적이라
        // 쓰지 않는다(원판 유지). 예전에는 축0 → 축1 **순차**로 풀고 원판 투영을 **또 한 번**
        // 적용했다 — 접점당 마찰 `apply`가 3쌍이었고, 먼저 푸는 축이 유리해 접선면 비등방이
        // 6.85% 있었다.
        //
        // 지금은 두 축을 **동시에** 풀고, 누적 임펄스를 원판에 투영한 **뒤** 그 차이만 한 번
        // 적용한다 → 마찰 `apply` 1쌍. 속도도 한 번만 읽는다(순차 갱신은 축마다 다시 읽었다).
        //
        // **예전 주석은 "동시에 풀면 낙하 체인에서 발산한다"고 기각하고 있었다.** 그때와 다른
        // 점은 **클램프를 적용 앞으로 옮긴 것**이다: 예전 Jacobi는 축별 임펄스를 *클램프 없이*
        // 적용한 뒤 원판 보정을 따로 걸어, 중간 상태가 Coulomb 한계를 넘고 그 초과분이 접점마다
        // 겹쳤다. 여기서는 누적 임펄스를 먼저 원판에 가둔 뒤 **차이만** 적용하므로 어떤 중간
        // 상태도 한계를 넘지 않는다. 낙하 체인 게이트가 이 구분을 지킨다(통과 확인).
        //
        // **결합항 `k01`까지 넣은 완전한 2x2 해는 재 보고 기각했다** (2026-07-31 실측).
        // 이론상 더 정확하지만 Coulomb 클램프와 겹치면 오히려 나빠진다:
        //   미끄럼 가속도 오차(이론 대비)  37도 +24.9% · 40도 +15.9%   ← 결합항 포함
        //                                 37도  +0.1% · 40도  +0.0%   ← 결합항 없음(현재)
        //   접선 비등방                    6.85%                       ← 결합항 포함
        //                                 ~0.01%                      ← 결합항 없음(현재)
        // 결합은 상자 모서리 접촉에서 실재한다(`c = -rx·rz/I` 꼴로 0이 아니다). 그런데 그
        // 결합을 정확히 반영한 스텝은 클램프가 걸리는 순간 더 이상 최적이 아니고, 증폭
        // (`1/(1−ab·c²)` ≥ 1)이 과잉 보정으로 돌아온다.
        let max_f = c.friction * p.normal_impulse;
        let dv = ba.vel_at(ra) - bb.vel_at(rb);
        let (vt0, vt1) = (c.tangents[0].dot(dv), c.tangents[1].dot(dv));
        let n0 = p.tangent_impulse[0] - p.tangent_mass[0] * vt0;
        let n1 = p.tangent_impulse[1] - p.tangent_mass[1] * vt1;
        let mag2 = n0 * n0 + n1 * n1;
        let (n0, n1) = if mag2 > max_f * max_f && mag2 > 0.0 {
            let scale = max_f / mag2.sqrt();
            (n0 * scale, n1 * scale)
        } else {
            (n0, n1)
        };
        let d0 = n0 - p.tangent_impulse[0];
        let d1 = n1 - p.tangent_impulse[1];
        p.tangent_impulse = [n0, n1];
        let imp = c.tangents[0] * d0 + c.tangents[1] * d1;
        ba.apply(imp, ra);
        bb.apply(-imp, rb);
    }
    ba.store(v, c.a);
    bb.store(v, c.b);
    let _ = inv_h;
}

/// restitution 패스 — 전 서브스텝 종료 후 1회 (설계 §4.4 임팩트 게이트).
pub fn apply_restitution(st: &mut SolverState, c: &mut ContactConstraint, threshold: f32) {
    if c.restitution == 0.0 {
        return;
    }
    for p in &mut c.points {
        // 게이트(설계 §4.4): 실제 임펄스가 발생했고, 접근 속도(음수)가 임계보다 빨랐던 접점만
        if p.approach_vn > -threshold || p.max_normal_impulse == 0.0 {
            continue;
        }
        let ra = st.rot[c.a].rotate(p.anchor_a);
        let rb = st.rot[c.b].rotate(p.anchor_b);
        let vn = c.normal.dot(st.vel_at(c.a, ra) - st.vel_at(c.b, rb));
        // 목표: vn_out = −e·approach_vn (approach_vn<0 → 양수 반발)
        let impulse = -p.normal_mass * (vn + c.restitution * p.approach_vn);
        let new_total = (p.normal_impulse + impulse).max(0.0);
        let applied = new_total - p.normal_impulse;
        p.normal_impulse = new_total;
        let imp = c.normal * applied;
        st.apply(c.a, imp, ra);
        st.apply(c.b, -imp, rb);
    }
}

/// 오버플로 색 — 64색을 다 쓴 제약이 몰리는 버킷. 이 색은 같은 색 안에 공유 dynamic 바디가
/// 있을 수 있으므로 **SIMD 배칭에서 제외**하고 스칼라로 처리한다. 정착 씬의 접촉 그래프
/// 최대 차수는 작아(박스가 이웃 ~4개) 실무에선 거의 발화하지 않는다.
pub const COLOR_OVERFLOW: u32 = 63;

/// 제약을 결정적으로 그래프 컬러링한다 (설계 §5 — SIMD 배칭 전제).
///
/// 입력 `endpoints`는 **canonical 순서**(중력깊이+pair-key 전순서)로 들어오고, 그 순서 기반
/// 그리디로 색을 매긴다: 각 제약에 두 바디가 아직 안 쓴 **최소 색**. 같은 색 안에는 공유
/// conflict 바디가 없으므로 그 색의 제약들은 서로 독립 — 레인 병렬(4-wide)로 풀어도 결과가 같다.
/// 색 배정이 정렬 순서에만 의존하므로 결과는 결정적이다(설계 §5:439-441 방어선 ②).
///
/// **static/kinematic 바디는 conflict가 아니다**: 솔버 `apply`/`store`는 `dynamic`일 때만
/// 바디에 쓰므로(위 `BodyReg`) 정적 바닥 하나에 붙은 접촉 수천 개가 한 색에 안전하게 들어간다.
/// 이걸 놓치면 바닥 접촉마다 색이 폭발한다(Box2D v3도 static은 컬러링 conflict에서 제외).
///
/// `conflict[i]`=바디 i가 솔버 write 대상(dynamic&active)인가. `body_color`=스크래치
/// (len=바디수, 0으로 채워 전달, 각 바디의 배정색 비트마스크). `out`에 제약별 색을 채운다.
/// 색 상한의 기본값 — 64색 전부 허용(= 오버플로는 실질적으로 안 생긴다).
///
/// **왜 상한이 파라미터인가.** 색 수는 **패스당 배리어 수**다: 병렬 솔브는 색 경계마다
/// 배리어를 쳐야 한다(다른 색은 바디를 공유하므로 Gauss-Seidel 순서가 결과를 바꾼다).
/// 40단 피라미드는 색이 7개라 스텝당 배리어가 7 × 5패스 × 4서브스텝 = 140회이고,
/// 색당 참여자 몫이 수십 us뿐이라 동기화가 이득을 먹었다(README '스레드 모드' 실측).
/// 상한을 P로 두면 배리어가 P+1로 유계가 되고, 들어가지 못한 제약은 오버플로 버킷으로
/// 밀려 **한 참여자가 순서대로** 푼다 — Jolt의 large island splitter와 같은 거래다
/// (병렬 폭을 유계로 사고 그 대가를 직렬 잔여로 지불한다).
///
/// **상한은 스레드 수와 무관해야 한다.** 상한이 바뀌면 제약 순서가 바뀌고 궤적이 바뀐다 —
/// 스레드 수에 따라 정하면 결정론(설계 §8 L0)이 깨진다.
pub const COLOR_CAP_DEFAULT: u32 = 64;

pub fn color_constraints(
    endpoints: &[(u32, u32)],
    conflict: &[bool],
    body_color: &mut [u64],
    out: &mut Vec<u32>,
    color_cap: u32,
) {
    out.clear();
    // 상한 0은 의미가 없다(전부 오버플로) — 1 이상으로 죈다. 64는 u64 비트폭 상한이다.
    let cap = color_cap.clamp(1, 64);
    for &(a, b) in endpoints {
        let (a, b) = (a as usize, b as usize);
        let mut forbidden = 0u64;
        if conflict[a] {
            forbidden |= body_color[a];
        }
        if conflict[b] {
            forbidden |= body_color[b];
        }
        // 최소 free 색 = forbidden의 최하위 0비트. 상한을 넘으면 오버플로 버킷.
        let free = (!forbidden).trailing_zeros();
        let color = if free >= cap { COLOR_OVERFLOW } else { free };
        let bit = 1u64 << color;
        if conflict[a] {
            body_color[a] |= bit;
        }
        if conflict[b] {
            body_color[b] |= bit;
        }
        out.push(color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Body;
    use maru_collide::{ContactPoint, FeatureId, Manifold, Pose, Shape, FEATURE_GENERIC};
    use maru_math::Quat;

    fn world_bodies() -> Vec<Body> {
        let pose = |y: f32| Pose {
            pos: Vec3::new(0.0, y, 0.0),
            rot: Quat::IDENTITY,
        };
        vec![
            Body::new_static(
                Shape::Box {
                    half: Vec3::new(5.0, 0.5, 5.0),
                    radius: 0.0,
                },
                pose(0.0),
            ),
            Body::new_dynamic(Shape::Sphere { radius: 0.5 }, pose(1.0), 1.0),
        ]
    }

    fn manifold_at(pos: Vec3) -> Manifold {
        Manifold::from_points(
            Vec3::new(0.0, 1.0, 0.0),
            &[ContactPoint {
                position: pos,
                separation: -0.001,
                feature: FeatureId(FEATURE_GENERIC),
            }],
        )
    }

    /// 유효질량 4-wide 경로가 스칼라 3회와 **비트 동일**함을 잠근다.
    ///
    /// 이 테스트가 A/B 노브(`MARU_SCALAR_EFFMASS`)를 정당화한다 — 두 팔이 비트 동일하지
    /// 않으면 성능 비교가 애초에 성립하지 않는다(다른 물리를 재는 것이 된다).
    /// 근사 비교(`abs_diff < eps`)를 쓰지 않는 이유: 유효질량은 임펄스 크기를 정하므로
    /// 마지막 ulp 차이가 서브스텝을 타고 궤적을 갈라 결정론(L0)을 깬다.
    #[test]
    fn effective_mass_x3_matches_scalar_bit_for_bit() {
        // 이방성 관성 + 비축정렬 자세 + 퇴화 경우(양쪽 static = k 0)를 함께 덮는다.
        let cases: &[(f32, Vec3, f32, Vec3, Vec3, Vec3, Vec3)] = &[
            // (inv_m_a, inv_i_diag_a, inv_m_b, inv_i_diag_b, ra, rb, normal)
            (
                1.0,
                Vec3::new(6.0, 6.0, 6.0),
                0.0,
                Vec3::ZERO,
                Vec3::new(0.3, -0.5, 0.2),
                Vec3::new(0.3, 0.5, 0.2),
                Vec3::new(0.0, 1.0, 0.0),
            ),
            (
                0.5,
                Vec3::new(11.3, 2.7, 7.9),
                2.0,
                Vec3::new(0.4, 19.1, 3.3),
                Vec3::new(-1.7, 0.9, 2.4),
                Vec3::new(0.8, -2.2, 0.1),
                Vec3::new(0.577, 0.577, 0.577),
            ),
            // 양쪽 무한질량·무한관성 → k = 0 → 유효질량 0 (분기/blend 경계)
            (
                0.0,
                Vec3::ZERO,
                0.0,
                Vec3::ZERO,
                Vec3::new(1.0, 2.0, 3.0),
                Vec3::new(-1.0, 0.5, 0.25),
                Vec3::new(1.0, 0.0, 0.0),
            ),
        ];
        let rot = Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 0.7);
        for (inv_m_a, ii_a, inv_m_b, ii_b, ra, rb, n) in cases.iter().copied() {
            let n = n.normalize_or_zero();
            let ia = Mat3::rotate_inertia(ii_a, rot);
            let ib = Mat3::rotate_inertia(ii_b, rot);
            let tangents = tangent_basis(n);
            let scalar = [
                effective_mass(inv_m_a, &ia, ra, inv_m_b, &ib, rb, n),
                effective_mass(inv_m_a, &ia, ra, inv_m_b, &ib, rb, tangents[0]),
                effective_mass(inv_m_a, &ia, ra, inv_m_b, &ib, rb, tangents[1]),
            ];
            let simd = effective_mass_x3(
                f32x4::splat(inv_m_a + inv_m_b),
                &Mat3x4::splat(ia),
                &Mat3x4::splat(ib),
                ra,
                rb,
                Vec3x4::from_array([n, tangents[0], tangents[1], n]),
            );
            for i in 0..3 {
                assert_eq!(
                    scalar[i].to_bits(),
                    simd[i].to_bits(),
                    "레인 {i}: 스칼라 {} vs SIMD {}",
                    scalar[i],
                    simd[i]
                );
            }
            // 넷째 레인은 법선을 다시 실은 것이므로 0번과 같아야 한다(셋업이 어긋나면 여기가 깨진다).
            assert_eq!(simd[3].to_bits(), scalar[0].to_bits());
        }
    }

    /// `if k > 0.0` 분기와 안전 분모가 **모든 경계 입력**에서 스칼라와 같은지.
    ///
    /// 위 테스트는 k>0 두 건과 k=0 한 건뿐이라 `k<0`·NaN·비정규수·±inf·−0.0을 하나도 밟지
    /// 않았다(교차검수 rev_4976e200 MAJOR). 그 상태면 NaN 마스크를 반대로 고르거나 음수
    /// 레인에서 원래 k로 나누는 회귀가 들어와도 전부 통과한다 — 즉 **실패할 수 없는 테스트**였다.
    /// 관성을 0으로 두면 k = inv_m_a가 되어 분기 입력을 직접 지정할 수 있다.
    #[test]
    fn effective_mass_x3_matches_scalar_at_branch_edges() {
        let zero_i = Mat3::from_diagonal(Vec3::ZERO);
        let zero_i4 = Mat3x4::splat(zero_i);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let dirs = Vec3x4::from_array([dir; 4]);

        for inv_m_a in [
            -1.0,
            f32::NEG_INFINITY,
            f32::from_bits(0x8000_0001), // 음의 최소 비정규수
            -0.0,
            0.0,
            f32::from_bits(0x0000_0001), // 양의 최소 비정규수
            f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::from_bits(0x7fc0_1234), // qNaN
        ] {
            let scalar =
                effective_mass(inv_m_a, &zero_i, Vec3::ZERO, 0.0, &zero_i, Vec3::ZERO, dir);
            let simd = effective_mass_x3(
                f32x4::splat(inv_m_a + 0.0),
                &zero_i4,
                &zero_i4,
                Vec3::ZERO,
                Vec3::ZERO,
                dirs,
            );
            for (lane, got) in simd.iter().enumerate() {
                assert_eq!(
                    scalar.to_bits(),
                    got.to_bits(),
                    "k={inv_m_a:?} lane={lane}: 스칼라 {scalar:?} vs SIMD {got:?}"
                );
            }
        }

        // 합산이 실제로 **좌결합**인지 — 재결합하면 값이 달라지는 입력으로 잠근다.
        // ((2^24 + −2^24) + 1) = 1 이지만 2^24 + (−2^24 + 1)도 1이라 그 짝은 판별력이 없다;
        // 판별력은 dotA가 흡수되는 쪽에서 나온다: (2^24 + 1) + (−2^24) = 0 ≠ 1.
        let ia = Mat3::from_diagonal(Vec3::new(0.0, 0.0, 1.0));
        let ra = Vec3::new(0.0, 1.0, 0.0);
        let (inv_m_a, inv_m_b) = (16_777_216.0f32, -16_777_216.0f32);
        let scalar = effective_mass(inv_m_a, &ia, ra, inv_m_b, &zero_i, Vec3::ZERO, dir);
        assert_eq!(
            scalar.to_bits(),
            1.0f32.to_bits(),
            "좌결합이면 k=1이라 유효질량도 1이다"
        );
        let simd = effective_mass_x3(
            f32x4::splat(inv_m_a + inv_m_b),
            &Mat3x4::splat(ia),
            &zero_i4,
            ra,
            Vec3::ZERO,
            dirs,
        );
        for (lane, got) in simd.iter().enumerate() {
            assert_eq!(scalar.to_bits(), got.to_bits(), "좌결합 lane={lane}");
        }
    }

    /// 회귀(자체 검수): **복합 자식의 generic 접점**(네임스페이스 비트 포함)도 위치 근접
    /// 게이트를 통과해야 한다 — 정확 일치 판정이면 네임스페이스된 generic이 안정 feature로
    /// 오분류돼 점프한 접점이 임펄스를 승계한다.
    #[test]
    fn namespaced_generic_contacts_still_require_proximity() {
        let bodies = world_bodies();
        let ns = maru_collide::FEATURE_GENERIC | (5u32 << maru_collide::FEATURE_CHILD_SHIFT);
        let cache = vec![(
            FeatureId(ns),
            CachedImpulse {
                normal: 7.0,
                tangent: [0.0, 0.0],
                tangent_world: Vec3::ZERO,
                position: Vec3::new(0.0, 0.5, 0.0),
                contact_normal: Vec3::new(0.0, 1.0, 0.0),
            },
        )];
        // 접점이 0.4m 점프 — ID는 같아도 승계 금지
        let far = Manifold::from_points(
            Vec3::new(0.0, 1.0, 0.0),
            &[ContactPoint {
                position: Vec3::new(0.4, 0.5, 0.0),
                separation: -0.001,
                feature: FeatureId(ns),
            }],
        );
        let c = prepare(&bodies, 1, 0, &far, &cache, 1.0 / 60.0).unwrap();
        assert_eq!(
            c.points[0].normal_impulse, 0.0,
            "네임스페이스된 generic이 위치 게이트 없이 warm-match됐다"
        );
    }

    /// 계약(리뷰 MAJOR): 계보 없는 witness 접점은 ID를 공유하므로 **위치 근접**까지 맞을 때만
    /// 이전 임펄스를 물려받는다. 접점이 점프하면(면→모서리) 물려받지 않는다.
    #[test]
    fn generic_contacts_warm_start_only_when_close() {
        let bodies = world_bodies();
        let cached_at = Vec3::new(0.0, 0.5, 0.0);
        let cache = vec![(
            FeatureId(FEATURE_GENERIC),
            CachedImpulse {
                normal: 7.0,
                tangent: [0.0, 0.0],
                tangent_world: Vec3::ZERO,
                position: cached_at,
                contact_normal: Vec3::new(0.0, 1.0, 0.0),
            },
        )];
        // 정지 상태이므로 허용치 = 기하 바닥만 (최소 특징 크기 = 구 반지름 0.5 → 0.125)
        const DT: f32 = 1.0 / 60.0;
        // ① 근접(1cm 이동): 계승
        let near = prepare(
            &bodies,
            1,
            0,
            &manifold_at(cached_at + Vec3::new(0.01, 0.0, 0.0)),
            &cache,
            DT,
        )
        .unwrap();
        assert_eq!(
            near.points[0].normal_impulse, 7.0,
            "근접 접점은 warm start 계승"
        );
        // ② 점프(1m 이동): 차단
        let far = prepare(
            &bodies,
            1,
            0,
            &manifold_at(cached_at + Vec3::new(1.0, 0.0, 0.0)),
            &cache,
            DT,
        )
        .unwrap();
        assert_eq!(
            far.points[0].normal_impulse, 0.0,
            "점프한 접점에 이전 임펄스가 적용됨"
        );
        // ③ 법선이 뒤집힌 접점: 위치가 같아도 다른 접촉이다 (곡면 feature 점프 차단)
        let flipped = vec![(
            FeatureId(FEATURE_GENERIC),
            CachedImpulse {
                normal: 7.0,
                tangent: [0.0, 0.0],
                tangent_world: Vec3::ZERO,
                position: cached_at,
                contact_normal: Vec3::new(1.0, 0.0, 0.0),
            },
        )];
        let turned = prepare(&bodies, 1, 0, &manifold_at(cached_at), &flipped, DT).unwrap();
        assert_eq!(
            turned.points[0].normal_impulse, 0.0,
            "법선이 바뀐 접촉에 이전 임펄스가 계승됨"
        );
    }

    /// 대조: 클리핑 접점(계보 있는 ID)은 위치와 무관하게 ID로 매칭된다.
    #[test]
    fn clipped_contacts_match_by_feature_id() {
        let bodies = world_bodies();
        let fid = FeatureId(0x8123_4567);
        let cache = vec![(
            fid,
            CachedImpulse {
                normal: 3.0,
                tangent: [0.0, 0.0],
                tangent_world: Vec3::ZERO,
                position: Vec3::new(0.0, 0.5, 0.0),
                contact_normal: Vec3::new(0.0, 1.0, 0.0),
            },
        )];
        let mut m = manifold_at(Vec3::new(2.0, 0.5, 0.0));
        m.points_mut()[0].feature = fid;
        let c = prepare(&bodies, 1, 0, &m, &cache, 1.0 / 60.0).unwrap();
        assert_eq!(c.points[0].normal_impulse, 3.0);
    }

    /// 컬러링 계약: 수직 체인은 인접이 바디를 공유하므로 색이 교대하고, 결과는 결정적이다.
    #[test]
    fn coloring_alternates_on_a_dynamic_chain_and_is_deterministic() {
        // (0,1)(1,2)(2,3) 전부 dynamic.
        let endpoints = [(0u32, 1u32), (1, 2), (2, 3)];
        let conflict = [true, true, true, true];
        let mut bc = vec![0u64; 4];
        let mut out = Vec::new();
        color_constraints(&endpoints, &conflict, &mut bc, &mut out, COLOR_CAP_DEFAULT);
        // (0,1)=0 → (1,2)는 색0 금지 → 1 → (2,3)은 색1만 금지 → 0
        assert_eq!(out, vec![0, 1, 0]);
        // 결정성: 같은 입력 → 같은 색
        let mut bc2 = vec![0u64; 4];
        let mut out2 = Vec::new();
        color_constraints(
            &endpoints,
            &conflict,
            &mut bc2,
            &mut out2,
            COLOR_CAP_DEFAULT,
        );
        assert_eq!(out, out2);
        // 색 내 conflict-free 불변식: 같은 색 제약들은 conflict 바디를 공유하지 않는다.
        assert_conflict_free(&endpoints, &conflict, &out);
    }

    /// static 바디(바닥)는 conflict가 아니므로 여러 접촉이 한 색에 들어간다 — 색 폭발 방지.
    #[test]
    fn static_bodies_do_not_force_new_colors() {
        // body0 = static(바닥), box 1·2·3이 각각 바닥에 접촉.
        let endpoints = [(0u32, 1u32), (0, 2), (0, 3)];
        let conflict = [false, true, true, true];
        let mut bc = vec![0u64; 4];
        let mut out = Vec::new();
        color_constraints(&endpoints, &conflict, &mut bc, &mut out, COLOR_CAP_DEFAULT);
        assert_eq!(out, vec![0, 0, 0]);
        assert_conflict_free(&endpoints, &conflict, &out);
    }

    /// 오버플로 색(63)은 conflict-free가 **아닌** 스칼라 버킷임을 잠근다 (Unit 3 SIMD 배칭이
    /// 색 63을 제외해야 하는 안전 전제 — 교차검수 rev_0bb4745e MINOR). 중심 0에 65개 동적
    /// 바디가 연결되면 안전 색 0..62 소진 후 64·65번째가 모두 색 63으로 몰려 중심 바디가 중복된다.
    #[test]
    fn overflow_color_is_explicitly_scalar_and_may_conflict() {
        let endpoints: Vec<(u32, u32)> = (1..=65).map(|i| (0, i)).collect();
        let conflict = vec![true; 66];
        let mut bc = vec![0u64; 66];
        let mut out = Vec::new();
        color_constraints(&endpoints, &conflict, &mut bc, &mut out, COLOR_CAP_DEFAULT);
        // 첫 64개는 색 0..63, 65번째는 오버플로 색 63으로 몰린다.
        assert_eq!(&out[..64], (0u32..64).collect::<Vec<_>>().as_slice());
        assert_eq!(out[64], COLOR_OVERFLOW);
        assert_eq!(out.iter().filter(|&&c| c == COLOR_OVERFLOW).count(), 2);
        // 오버플로를 스킵하므로 합법 충돌을 오판하지 않는다.
        assert_conflict_free(&endpoints, &conflict, &out);
    }

    /// 색 상한(= 배리어 상한)의 계약: 상한 아래 색은 여전히 conflict-free이고,
    /// 들어가지 못한 제약은 **오버플로 버킷**으로 간다(그 버킷만 순서대로 직렬 처리된다).
    ///
    /// 이 테스트가 없으면 상한을 낮췄을 때 "같은 색 = 바디 비공유" 불변식이 조용히 깨져도
    /// 모른다 — 그 불변식이 4-wide 레인 독립과 병렬 안전의 **유일한** 근거다.
    #[test]
    fn color_cap_bounds_colors_and_spills_the_rest() {
        // 중심 바디 0에 12개가 매달린 부채꼴: 상한이 없으면 색 0..11이 필요하다.
        let endpoints: Vec<(u32, u32)> = (1..=12).map(|i| (0, i)).collect();
        let conflict = vec![true; 13];

        // 상한 64(기본) — 색이 12개 나온다.
        let mut bc = vec![0u64; 13];
        let mut out = Vec::new();
        color_constraints(&endpoints, &conflict, &mut bc, &mut out, 64);
        assert_eq!(out, (0u32..12).collect::<Vec<_>>());
        assert_conflict_free(&endpoints, &conflict, &out);

        // 상한 4 — 앞 4개만 색을 받고 나머지 8개는 오버플로로 몰린다.
        let mut bc = vec![0u64; 13];
        let mut out = Vec::new();
        color_constraints(&endpoints, &conflict, &mut bc, &mut out, 4);
        assert_eq!(&out[..4], &[0, 1, 2, 3]);
        assert!(
            out[4..].iter().all(|&c| c == COLOR_OVERFLOW),
            "상한 초과분은 전부 오버플로여야 한다: {out:?}"
        );
        // 상한 아래 색들은 여전히 바디를 공유하지 않는다(오버플로는 판정에서 제외된다).
        assert_conflict_free(&endpoints, &conflict, &out);
        // 실제로 나온 **비어 있지 않은** 색 수 = 상한 + 오버플로 1개.
        let mut used: Vec<u32> = out.clone();
        used.sort_unstable();
        used.dedup();
        assert_eq!(used.len(), 5, "색 4개 + 오버플로 1개여야 한다: {used:?}");

        // 상한 1 — 색 0 하나만, 나머지 전부 오버플로(퇴화 경계).
        let mut bc = vec![0u64; 13];
        let mut out = Vec::new();
        color_constraints(&endpoints, &conflict, &mut bc, &mut out, 1);
        assert_eq!(out[0], 0);
        assert!(out[1..].iter().all(|&c| c == COLOR_OVERFLOW));
        // 상한 0은 1로 죄어진다 — 전부 오버플로가 되면 병렬 폭이 0이라 무의미하다.
        let mut bc0 = vec![0u64; 13];
        let mut out0 = Vec::new();
        color_constraints(&endpoints, &conflict, &mut bc0, &mut out0, 0);
        assert_eq!(out0, out, "상한 0은 상한 1과 같아야 한다(클램프)");
    }

    // --- Unit 3: SIMD warm_start 비트 동일성 ---
    #[allow(clippy::field_reassign_with_default)]
    fn mk_state(n: usize) -> SolverState {
        let mut st = SolverState::default();
        st.lin = (0..n)
            .map(|i| Vec3::new(i as f32 * 0.13, -(i as f32) * 0.07, 0.31))
            .collect();
        st.ang = (0..n)
            .map(|i| Vec3::new(0.021 * i as f32, -0.04, -0.011 * i as f32))
            .collect();
        st.pos = vec![Vec3::ZERO; n];
        st.rot = vec![Quat::IDENTITY; n];
        st.inv_m = (0..n).map(|i| 0.5 + 0.1 * i as f32).collect();
        st.inv_i = (0..n)
            .map(|i| {
                Mat3::from_diagonal(Vec3::new(1.0 + 0.1 * i as f32, 2.0, 0.5 + 0.05 * i as f32))
            })
            .collect();
        st.dynamic = vec![true; n];
        st
    }

    fn mk_point(seed: f32) -> ContactPointConstraint {
        ContactPointConstraint {
            feature: FeatureId(FEATURE_GENERIC),
            position: Vec3::ZERO,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            base_separation: 0.0,
            normal_mass: 1.0,
            tangent_mass: [1.0, 1.0],
            normal_impulse: 0.5 + seed,
            tangent_impulse: [0.1 * seed, -0.2 * seed],
            max_normal_impulse: 0.0,
            ra: Vec3::new(0.2 + seed, -0.1, 0.05 * seed),
            rb: Vec3::new(-0.15, 0.3 * seed, 0.2),
            approach_vn: 0.0,
        }
    }

    fn mk_constraint(a: usize, b: usize, npts: usize) -> ContactConstraint {
        ContactConstraint {
            a,
            b,
            normal: Vec3::new(0.0, 1.0, 0.0),
            tangents: [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)],
            friction: 0.5,
            restitution: 0.0,
            points: (0..npts)
                .map(|k| mk_point(a as f32 + 0.3 * k as f32))
                .collect(),
            warm_matched: 0,
        }
    }

    fn assert_states_bit_identical(a: &SolverState, b: &SolverState, n: usize) {
        for i in 0..n {
            for (u, v, tag) in [(a.lin[i], b.lin[i], "lin"), (a.ang[i], b.ang[i], "ang")] {
                assert_eq!(u.x.to_bits(), v.x.to_bits(), "{tag}[{i}].x");
                assert_eq!(u.y.to_bits(), v.y.to_bits(), "{tag}[{i}].y");
                assert_eq!(u.z.to_bits(), v.z.to_bits(), "{tag}[{i}].z");
            }
        }
    }

    /// 계약: `warm_start_x4`(4-wide)는 4번의 스칼라 `warm_start`와 레인당 **비트 동일**하다.
    /// 점 수가 1·2·3·4로 다른 4제약 — 없는 점의 no-op가 blend로 걸러져 부호0 뒤집힘이 없다.
    #[test]
    fn warm_start_x4_matches_scalar_bit_for_bit() {
        let n = 8;
        let st = mk_state(n);
        let cs = [
            mk_constraint(0, 1, 1),
            mk_constraint(2, 3, 2),
            mk_constraint(4, 5, 3),
            mk_constraint(6, 7, 4),
        ];
        let mut st_s = st.clone();
        {
            let v = SolveView::new(&mut st_s);
            for c in &cs {
                warm_start(&v, c);
            }
        }
        let mut st_v = st.clone();
        warm_start_x4(&mut st_v, [&cs[0], &cs[1], &cs[2], &cs[3]]);
        assert_states_bit_identical(&st_s, &st_v, n);
    }

    /// 계약: 공유 static 바디(레인 간 겹침 허용)는 write되지 않고, dynamic 게이트가 스칼라와
    /// 일치한다 — 4제약이 같은 static B를 공유해도 SIMD 결과가 스칼라와 비트 동일.
    #[test]
    fn warm_start_x4_respects_dynamic_gate_and_shared_static() {
        let n = 9;
        let mut st = mk_state(n);
        st.inv_m[8] = 0.0;
        st.dynamic[8] = false; // body 8 = static, 네 제약의 공유 B
        let cs = [
            mk_constraint(0, 8, 2),
            mk_constraint(1, 8, 1),
            mk_constraint(2, 8, 3),
            mk_constraint(3, 8, 2),
        ];
        let mut st_s = st.clone();
        {
            let v = SolveView::new(&mut st_s);
            for c in &cs {
                warm_start(&v, c);
            }
        }
        let mut st_v = st.clone();
        warm_start_x4(&mut st_v, [&cs[0], &cs[1], &cs[2], &cs[3]]);
        assert_states_bit_identical(&st_s, &st_v, n);
        // static 바디(8)는 양쪽 다 불변 — 원본과 비트 동일
        assert_eq!(st_v.lin[8].x.to_bits(), st.lin[8].x.to_bits());
        assert_eq!(st_v.ang[8].z.to_bits(), st.ang[8].z.to_bits());
    }

    /// 계약: `solve_x4`(bias·relax)는 4번의 스칼라 `solve`와 레인당 **비트 동일**하다 —
    /// s>0 speculative 분기, per-lane softness(dyn-dyn vs dyn-static), 마찰 2축, Coulomb 원판
    /// 클램프, 점 변이 스캐터 전부. 혼합 dynamic/static + Coulomb 트리거 + 가변 점수를 모두 친다.
    #[test]
    fn solve_x4_matches_scalar_bit_for_bit() {
        let soft = Softness::new(30.0, 10.0, 1.0 / 240.0);
        let static_soft = Softness::new(60.0, 10.0, 1.0 / 240.0);
        let inv_h = 240.0;
        let max_bias = 3.0;
        let build = || -> Vec<ContactConstraint> {
            let mut v = vec![
                mk_constraint(0, 1, 1),
                mk_constraint(2, 3, 2),
                mk_constraint(4, 5, 3),
                mk_constraint(6, 7, 4),
            ];
            // Coulomb 클램프 트리거: 제약0 점0에 큰 접선 임펄스 + 낮은 마찰.
            v[0].friction = 0.05;
            v[0].points[0].tangent_impulse = [5.0, -4.0];
            v[0].points[0].normal_impulse = 0.3;
            v
        };
        for relax in [false, true] {
            let n = 8;
            let mut st = mk_state(n);
            st.dynamic[7] = false; // static → dyn-static softness 경로
            st.inv_m[7] = 0.0;
            st.pos = (0..n)
                .map(|i| Vec3::new(0.1 * i as f32, if i % 2 == 0 { 0.03 } else { -0.02 }, 0.0))
                .collect();
            let p = BatchSolveParams {
                inv_h,
                relax,
                soft,
                static_soft,
                max_bias,
            };
            // 스칼라 (per-constraint sf 선택 — world.rs와 동일)
            let mut st_s = st.clone();
            let mut cs_s = build();
            for c in &mut cs_s {
                let opt = if relax {
                    None
                } else if st_s.dynamic[c.a] && st_s.dynamic[c.b] {
                    Some((&soft, max_bias))
                } else {
                    Some((&static_soft, max_bias))
                };
                let v = SolveView::new(&mut st_s);
                solve(&v, c, inv_h, opt);
            }
            // SIMD
            let mut st_v = st.clone();
            let mut cs_v = build();
            {
                let [c0, c1, c2, c3] = &mut cs_v[..] else {
                    unreachable!()
                };
                solve_x4(&mut st_v, [c0, c1, c2, c3], &p);
            }
            assert_states_bit_identical(&st_s, &st_v, n);
            for (a, b) in cs_s.iter().zip(&cs_v) {
                for (pa, pb) in a.points.iter().zip(&b.points) {
                    assert_eq!(
                        pa.normal_impulse.to_bits(),
                        pb.normal_impulse.to_bits(),
                        "ni relax={relax}"
                    );
                    assert_eq!(
                        pa.tangent_impulse[0].to_bits(),
                        pb.tangent_impulse[0].to_bits(),
                        "ti0 relax={relax}"
                    );
                    assert_eq!(
                        pa.tangent_impulse[1].to_bits(),
                        pb.tangent_impulse[1].to_bits(),
                        "ti1 relax={relax}"
                    );
                    assert_eq!(
                        pa.max_normal_impulse.to_bits(),
                        pb.max_normal_impulse.to_bits(),
                        "mni relax={relax}"
                    );
                }
            }
        }
    }

    /// 같은 색 안에 공유 conflict 바디가 없음을 전수 검사한다 (오버플로 색 63 제외).
    fn assert_conflict_free(endpoints: &[(u32, u32)], conflict: &[bool], color_of: &[u32]) {
        use std::collections::HashSet;
        let ncol = color_of.iter().copied().max().unwrap_or(0) as usize + 1;
        let mut used: Vec<HashSet<u32>> = vec![HashSet::new(); ncol];
        for (k, &(a, b)) in endpoints.iter().enumerate() {
            let col = color_of[k] as usize;
            // 오버플로 색(63)은 설계상 스칼라 버킷 — conflict-free가 아니다(SIMD 배칭 제외).
            if color_of[k] == COLOR_OVERFLOW {
                continue;
            }
            for body in [a, b] {
                if conflict[body as usize] {
                    assert!(
                        used[col].insert(body),
                        "색 {col}에 conflict 바디 {body} 중복 — 레인 병렬 시 write 유실"
                    );
                }
            }
        }
    }
}
