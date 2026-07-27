//! TGS-soft 접촉 솔버 (설계 §4.4 — Box2D v3 "Soft Step" 정식화).
//!
//! 서브스텝마다: 속도 적분 → warm start → soft bias 임펄스 해석 → 위치 적분 → relax(무bias).
//! 전 서브스텝 후 restitution 패스 1회 — **임팩트 게이트**(설계 §4.4): 접촉 생성 시점의
//! 접근 속도(approach_vn)가 임계 이상일 때만, 실제 임펄스가 발생한 접점에만 발화.
//! [근거: Solver2D 실증 + Box2D v3 releasing-3.0]

use crate::body::Body;
use maru_collide::Manifold;
use maru_math::{Mat3, Mat3x4, Quat, Vec3, Vec3x4};
use wide::{f32x4, CmpGt};

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

pub struct ContactConstraint {
    pub a: usize,
    pub b: usize,
    /// A를 +normal로 밀면 분리 (maru-collide 관례).
    pub normal: Vec3,
    pub tangents: [Vec3; 2],
    pub friction: f32,
    pub restitution: f32,
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
        macro_rules! res {
            ($v:expr) => {
                $v.try_reserve(n.saturating_sub($v.capacity()))
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
pub fn prepare(
    bodies: &[Body],
    a: usize,
    b: usize,
    m: &Manifold,
    cache: &[(maru_collide::FeatureId, CachedImpulse)],
    dt: f32,
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
    let mut points = Vec::new();
    if points.try_reserve(m.points().len()).is_err() {
        return Err(crate::world::ApiError::ResourceLimit);
    }
    points.extend(m.points().iter().map(|p| {
        let ra = p.position - ba.pose.pos;
        let rb = p.position - bb.pose.pos;
        let vn = n.dot(ba.velocity_at(ra) - bb.velocity_at(rb));
        // warm start 매칭 규칙 (리뷰 MAJOR):
        // - 클리핑 접점: feature ID가 계보를 담아 안정적 → ID 일치
        // - generic witness 접점: pair 안에서 ID를 공유하므로(구-박스 접점이 면→모서리로
        //   이동해도 같은 ID) **위치 근접**까지 요구한다. 점프하면 임펄스를 물려주지 않는다.
        let generic = p.feature.0 == maru_collide::FEATURE_GENERIC;
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
        ContactPointConstraint {
            feature: p.feature,
            position: p.position,
            anchor_a: ba.pose.rot.conjugate().rotate(ra),
            anchor_b: bb.pose.rot.conjugate().rotate(rb),
            base_separation: p.separation,
            normal_mass: effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, n),
            tangent_mass: [
                effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, tangents[0]),
                effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, tangents[1]),
            ],
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
pub fn refresh_anchors(st: &SolverState, c: &mut ContactConstraint) {
    let (qa, qb) = (st.rot[c.a], st.rot[c.b]);
    for p in &mut c.points {
        p.ra = qa.rotate(p.anchor_a);
        p.rb = qb.rotate(p.anchor_b);
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
    fn load(st: &SolverState, i: usize) -> Self {
        Self {
            lin: st.lin[i],
            ang: st.ang[i],
            inv_m: st.inv_m[i],
            inv_i: st.inv_i[i],
            dynamic: st.dynamic[i],
        }
    }
    #[inline]
    fn store(&self, st: &mut SolverState, i: usize) {
        if self.dynamic {
            st.lin[i] = self.lin;
            st.ang[i] = self.ang;
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
pub fn warm_start(st: &mut SolverState, c: &ContactConstraint) {
    let (mut ba, mut bb) = (BodyReg::load(st, c.a), BodyReg::load(st, c.b));
    for p in &c.points {
        let (ra, rb) = (p.ra, p.rb);
        let imp = c.normal * p.normal_impulse
            + c.tangents[0] * p.tangent_impulse[0]
            + c.tangents[1] * p.tangent_impulse[1];
        ba.apply(imp, ra);
        bb.apply(-imp, rb);
    }
    ba.store(st, c.a);
    bb.store(st, c.b);
}

// ===== SIMD 배치 (Unit 3 — 설계 §5) =====
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
#[allow(dead_code)] // 슬라이스 3b(서브스텝 루프 배선) 전까지 스테이징 — 격리 테스트로 검증됨
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
#[allow(dead_code)]
fn negx4(v: Vec3x4) -> Vec3x4 {
    Vec3x4 {
        x: -v.x,
        y: -v.y,
        z: -v.z,
    }
}

/// 레인별 blend: gate 마스크가 참인 레인은 `t`, 아니면 `f`(원본 보존).
#[inline]
#[allow(dead_code)]
fn blend_vec(gate: f32x4, t: Vec3x4, f: Vec3x4) -> Vec3x4 {
    Vec3x4 {
        x: gate.blend(t.x, f.x),
        y: gate.blend(t.y, f.y),
        z: gate.blend(t.z, f.z),
    }
}

/// 4제약 배치의 바디 상태 (SoA 레인) — 스칼라 `BodyReg`의 4-wide 대응.
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct BodyRegx4 {
    lin: Vec3x4,
    ang: Vec3x4,
    inv_m: f32x4,
    inv_i: Mat3x4,
    /// dynamic 게이트 마스크 (레인당 all-1s/0). 비-dynamic 레인은 write 안 됨.
    dyn_mask: f32x4,
}

#[allow(dead_code)]
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
}

/// warm start의 4제약 배치판. 레인당 결과는 스칼라 `warm_start`와 **비트 동일**하다
/// (연산 순서 동일·비FMA·게이트 blend). 점 수가 다른 레인은 없는 점을 0/ZERO로 채우되,
/// blend `active` 마스크로 걸러 그 레인 값을 건드리지 않는다.
#[allow(dead_code)] // 슬라이스 3b에서 서브스텝 루프에 색 배치로 배선 예정
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

/// 속도 임펄스 해석. `soft`=bias 페이즈, `relax`(soft=None)=bias 제거 페이즈.
pub fn solve(
    st: &mut SolverState,
    c: &mut ContactConstraint,
    inv_h: f32,
    soft: Option<(&Softness, f32)>, // (계수, max_bias_speed)
) {
    let (mut ba, mut bb) = (BodyReg::load(st, c.a), BodyReg::load(st, c.b));
    let (pos_a, pos_b) = (st.pos[c.a], st.pos[c.b]);
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

        // --- 마찰: 축별 **순차** 해석(GS) 후 누적 2D 임펄스를 Coulomb **원판**에 투영 ---
        // 사각형 클램프는 대각 슬라이드에서 √2·μN까지 허용해 41% 과도 감속하고 기저 의존적이다.
        // 단, 두 축을 동시(Jacobi)로 풀면 유효질량이 큰 접점에서 과잉 보정이 겹쳐 발산한다
        // (낙하 체인 실증) — 순차 갱신을 유지한 채 마지막에 원판으로 되돌린다.
        let max_f = c.friction * p.normal_impulse;
        for axis in 0..2 {
            let t = c.tangents[axis];
            let vt = t.dot(ba.vel_at(ra) - bb.vel_at(rb));
            let delta = -p.tangent_mass[axis] * vt;
            p.tangent_impulse[axis] += delta;
            let imp = t * delta;
            ba.apply(imp, ra);
            bb.apply(-imp, rb);
        }
        let mag2 = p.tangent_impulse[0] * p.tangent_impulse[0]
            + p.tangent_impulse[1] * p.tangent_impulse[1];
        if mag2 > max_f * max_f && mag2 > 0.0 {
            let scale = max_f / mag2.sqrt();
            let corr0 = p.tangent_impulse[0] * (scale - 1.0);
            let corr1 = p.tangent_impulse[1] * (scale - 1.0);
            p.tangent_impulse[0] += corr0;
            p.tangent_impulse[1] += corr1;
            let imp = c.tangents[0] * corr0 + c.tangents[1] * corr1;
            ba.apply(imp, ra);
            bb.apply(-imp, rb);
        }
    }
    ba.store(st, c.a);
    bb.store(st, c.b);
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
pub fn color_constraints(
    endpoints: &[(u32, u32)],
    conflict: &[bool],
    body_color: &mut [u64],
    out: &mut Vec<u32>,
) {
    out.clear();
    for &(a, b) in endpoints {
        let (a, b) = (a as usize, b as usize);
        let mut forbidden = 0u64;
        if conflict[a] {
            forbidden |= body_color[a];
        }
        if conflict[b] {
            forbidden |= body_color[b];
        }
        // 최소 free 색 = forbidden의 최하위 0비트. 64색 초과면 오버플로 버킷.
        let color = (!forbidden).trailing_zeros().min(COLOR_OVERFLOW);
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
        color_constraints(&endpoints, &conflict, &mut bc, &mut out);
        // (0,1)=0 → (1,2)는 색0 금지 → 1 → (2,3)은 색1만 금지 → 0
        assert_eq!(out, vec![0, 1, 0]);
        // 결정성: 같은 입력 → 같은 색
        let mut bc2 = vec![0u64; 4];
        let mut out2 = Vec::new();
        color_constraints(&endpoints, &conflict, &mut bc2, &mut out2);
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
        color_constraints(&endpoints, &conflict, &mut bc, &mut out);
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
        color_constraints(&endpoints, &conflict, &mut bc, &mut out);
        // 첫 64개는 색 0..63, 65번째는 오버플로 색 63으로 몰린다.
        assert_eq!(&out[..64], (0u32..64).collect::<Vec<_>>().as_slice());
        assert_eq!(out[64], COLOR_OVERFLOW);
        assert_eq!(out.iter().filter(|&&c| c == COLOR_OVERFLOW).count(), 2);
        // 오버플로를 스킵하므로 합법 충돌을 오판하지 않는다.
        assert_conflict_free(&endpoints, &conflict, &out);
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
        for c in &cs {
            warm_start(&mut st_s, c);
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
        for c in &cs {
            warm_start(&mut st_s, c);
        }
        let mut st_v = st.clone();
        warm_start_x4(&mut st_v, [&cs[0], &cs[1], &cs[2], &cs[3]]);
        assert_states_bit_identical(&st_s, &st_v, n);
        // static 바디(8)는 양쪽 다 불변 — 원본과 비트 동일
        assert_eq!(st_v.lin[8].x.to_bits(), st.lin[8].x.to_bits());
        assert_eq!(st_v.ang[8].z.to_bits(), st.ang[8].z.to_bits());
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
