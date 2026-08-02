//! `CharacterVirtual` — **솔버 밖**에서 사는 캐릭터 컨트롤러 (설계 §9, M5).
//!
//! **왜 강체가 아닌가.** 플레이어를 강체로 만들면 게임이 원하지 않는 것을 전부 얻는다:
//! 경사에서 미끄러지고, 밀리면 넘어지고, 계단에서 걸리고, 마찰을 조정하면 다른 접촉이
//! 망가진다. 그래서 상용 엔진은 캐릭터를 **질의 기반**으로 둔다(Jolt CharacterVirtual,
//! Unity CharacterController) — 스윕으로 갈 수 있는 만큼 가고, 막히면 표면을 따라 미끄러지고,
//! 지면·경사·계단은 규칙으로 정한다. 물리와 게임 규칙의 경계를 여기서 긋는 것이다.
//!
//! **캐노니컬 상태가 아니다.** 캐릭터는 월드 스냅샷에 들어가지 않는다 — 위치·속도는 호스트가
//! 들고 저장한다(`position`·`velocity`는 그냥 필드다). 리플레이·롤백을 하는 호스트는
//! 월드 스냅샷과 **함께** 이 두 값을 저장해야 한다. 엔진이 대신 정해 주지 않는 이유는
//! 게임마다 캐릭터 상태(체력·애니메이션·입력 버퍼)가 달라서, 일부만 저장해 주면 오히려
//! "저장했다"는 착각을 만들기 때문이다.
//!
//! 결정론: 모든 판정이 `World::sweep_shape`(슬롯 오름차순·최소 TOI·동률은 낮은 슬롯)와
//! 산술만으로 이뤄진다 — 같은 월드·같은 입력이면 같은 결과다.

use crate::world::{ApiError, QueryFilter, SweepHit, World};
use musoe_collide::{Pose, Shape};
use musoe_math::{Quat, Vec3};

/// 한 번의 이동에서 미끄러짐을 시도할 최대 횟수.
///
/// 구석(두 벽 사이)에서는 한 번 미끄러진 방향이 다시 막힌다. 3회면 3면 구석까지 처리되고,
/// 그 이상은 사실상 갇힌 상태다 — 무한 루프 대신 **남은 이동을 돌려준다**(호스트가 안다).
const MAX_SLIDES: u32 = 4;

/// 질의 기반 캐릭터.
#[derive(Clone)]
pub struct CharacterVirtual {
    /// 몸통 형상 — 캡슐이 표준이다(모서리에 안 걸린다).
    pub shape: Shape,
    /// 형상 원점의 월드 위치.
    pub position: Vec3,
    /// 호스트가 관리하는 속도(중력·점프는 호스트 규칙이다 — 엔진은 이동만 해석한다).
    pub velocity: Vec3,
    /// "위" 방향(단위). 중력과 반대인 것이 보통이지만 벽타기 같은 규칙도 가능하다.
    pub up: Vec3,
    /// 걸어 오를 수 있는 최대 경사의 **코사인**. 0.7 ≈ 45°.
    pub max_slope_cos: f32,
    /// 표면에서 유지할 여유. 0이면 다음 스윕이 즉시 접촉으로 시작해 진행이 멈춘다.
    pub skin: f32,
    /// 자동으로 올라설 수 있는 턱 높이(계단).
    pub step_height: f32,
}

/// 이동 결과.
#[derive(Clone, Copy, Debug)]
pub struct MoveResult {
    /// 실제로 도달한 위치.
    pub position: Vec3,
    /// 적용하지 못하고 남은 이동 — 0이 아니면 끼었다는 뜻이다.
    pub remaining: Vec3,
    /// 발 밑이 걸을 수 있는 지면인가(경사 한계 안).
    pub on_ground: bool,
    /// 지면 법선(`on_ground`일 때만 의미 있다).
    pub ground_normal: Vec3,
    /// 미끄러짐 반복 횟수 — 구석에 낀 정도의 진단.
    pub slides: u32,
    /// 계단을 올라갔는가.
    pub stepped_up: bool,
    /// **이동 전에 이미 무언가에 파고들어 있었다** — 이동을 적용하지 않았다.
    ///
    /// 스폰·텔레포트·움직이는 지형 때문에 생긴다. 이 상태에서 스윕은 "장애물 없음"(`None`)을
    /// 돌려주므로 그대로 이동하면 벽을 통과한다 — 그래서 **멈추고 알린다**. 호스트는 밀어내기
    /// (depenetration)를 하거나 위치를 되돌린 뒤 다시 부른다.
    pub started_overlapped: bool,
}

impl CharacterVirtual {
    /// 기본값: 캡슐, 45° 경사, 2cm 여유, 30cm 턱.
    pub fn new(shape: Shape, position: Vec3) -> Self {
        Self {
            shape,
            position,
            velocity: Vec3::ZERO,
            up: Vec3::new(0.0, 1.0, 0.0),
            max_slope_cos: 0.7,
            skin: 0.02,
            step_height: 0.3,
        }
    }

    fn pose_at(&self, p: Vec3) -> Pose {
        Pose {
            pos: p,
            rot: Quat::IDENTITY,
        }
    }

    /// 한 번의 스윕 — 맞으면 `(안전 위치, 히트)`.
    ///
    /// 안전 위치는 접촉 직전에서 `skin`만큼 물러난 곳이다. 물러나지 않으면 다음 스윕이
    /// 침투 상태로 시작해 TOI 0이 나오고, 캐릭터가 벽에 붙어 영원히 못 움직인다.
    fn sweep(
        &self,
        world: &World,
        from: Vec3,
        motion: Vec3,
        filter: QueryFilter,
    ) -> Result<(Vec3, Option<SweepHit>), ApiError> {
        let len = motion.length();
        if len <= 0.0 {
            return Ok((from, None));
        }
        let hit = world.sweep_shape(&self.shape, self.pose_at(from), motion, filter)?;
        match hit {
            None => Ok((from + motion, None)),
            Some(h) => {
                // toi를 거리로 환산해 skin만큼 뺀다(음수면 제자리).
                let travel = (h.toi * len - self.skin).max(0.0);
                Ok((from + motion * (travel / len), Some(h)))
            }
        }
    }

    /// **이동 + 미끄러짐**. 막히면 표면 접선으로 남은 이동을 접어 다시 시도한다.
    ///
    /// 계단(`step_height`)은 정면이 막혔을 때만 시도한다: 위로 들어 올려 → 앞으로 → 아래로
    /// 내려놓기. 세 스윕이 모두 진전을 만들 때만 채택하고, 아니면 원래 결과를 쓴다 —
    /// 즉 계단 처리가 **없던 진전을 만들어내지는 않는다**.
    pub fn move_and_slide(
        &mut self,
        world: &World,
        displacement: Vec3,
        filter: QueryFilter,
    ) -> Result<MoveResult, ApiError> {
        // **설정 필드가 전부 public이므로 매 호출 검증한다** (검수 rev_1e81559e BLOCKER).
        // 예: `skin`이 음수면 접촉에서 물러나는 대신 **더 나아가** 벽 반대편으로 순간이동한다
        // (`travel = t·len − skin`). 한 번 잘못 설정하면 조용히 물리를 깨는 종류의 값이라,
        // 생성자에서만 막아서는 부족하다(호스트가 나중에 대입할 수 있다).
        let up2 = self.up.length_sq();
        if !musoe_collide::vec_in_range(displacement)
            || !musoe_collide::vec_in_range(self.position)
            || self.shape.validate().is_err()
            // 스윕 주체는 볼록이어야 한다 — 비볼록을 넣으면 GJK가 trap이다.
            // **비볼록 변종을 여기 빠뜨리면 안 된다.** 메시는 `validate`·`is_castable`을
            // 통과하는 정상 셰이프라(그건 "대상으로 유효"라는 뜻이다) 셰이프 검증이 이 책임을
            // 대신 져 주지 않는다. TriMesh가 빠져 있었고, 지금은 다른 경로가 우연히 먼저
            // 거부하지만 그건 계약이 아니다 — 같은 계열 누락 셋을 교차검수 rev_953115e8과
            // 그 후속 전수 점검에서 잡았다(`manifold::is_non_convex`·`Body::validate`·여기).
            || matches!(
                self.shape,
                Shape::Heightfield { .. } | Shape::TriMesh { .. } | Shape::Compound { .. }
            )
            || !(0.0..=1.0).contains(&self.max_slope_cos)
            || !(self.skin.is_finite() && self.skin >= 0.0)
            || !(self.step_height.is_finite() && self.step_height >= 0.0)
            || !(0.99..=1.01).contains(&up2)
        {
            return Err(ApiError::InvalidValue);
        }
        // **시작 침투를 먼저 본다** (확인검수 rev_7efe4e35 BLOCKER).
        if world.shape_penetrates(&self.shape, self.pose_at(self.position), filter)? {
            return Ok(MoveResult {
                position: self.position,
                remaining: displacement,
                on_ground: false,
                ground_normal: self.up,
                slides: 0,
                stepped_up: false,
                started_overlapped: true,
            });
        }
        let mut pos = self.position;
        let mut remaining = displacement;
        let mut slides = 0u32;
        let mut stepped_up = false;

        while slides < MAX_SLIDES && remaining.length() > 0.0 {
            let (moved_to, hit) = self.sweep(world, pos, remaining, filter)?;
            let progressed = moved_to - pos;
            pos = moved_to;
            let Some(h) = hit else {
                remaining = Vec3::ZERO;
                break;
            };
            // 남은 이동에서 이미 간 만큼을 뺀다.
            let left = remaining - progressed;
            // 계단: 수평으로 막혔고 턱이 낮으면 올라선다.
            let horizontal = left - self.up * left.dot(self.up);
            if !stepped_up
                && self.step_height > 0.0
                && horizontal.length() > 0.0
                && h.normal.dot(self.up).abs() < self.max_slope_cos
            {
                if let Some(p) = self.try_step_up(world, pos, horizontal, filter)? {
                    pos = p;
                    stepped_up = true;
                    remaining = Vec3::ZERO;
                    break;
                }
            }
            // 미끄러짐: 남은 이동을 접촉면 접선으로 사영한다.
            let n = h.normal;
            remaining = left - n * left.dot(n);
            slides += 1;
            // 진전도 없고 사영도 0이면 더 해봐야 같은 자리다.
            if progressed.length() <= 0.0 && remaining.length() <= 0.0 {
                break;
            }
        }

        // 지면 판정: 아래로 짧게 쏴 본다(스냅이 아니라 **판정**만 — 스냅은 호스트 규칙).
        let probe = self.up * -(self.skin * 4.0 + 1e-4);
        let (_, ground) = self.sweep(world, pos, probe, filter)?;
        let (on_ground, ground_normal) = match ground {
            Some(g) if g.normal.dot(self.up) >= self.max_slope_cos => (true, g.normal),
            Some(g) => (false, g.normal),
            None => (false, self.up),
        };

        self.position = pos;
        Ok(MoveResult {
            position: pos,
            remaining,
            on_ground,
            ground_normal,
            slides,
            stepped_up,
            started_overlapped: false,
        })
    }

    /// 계단 시도 — 위로 → 앞으로 → 아래로. 전부 성공하고 **앞으로 실제 진전이 있을 때만**
    /// 새 위치를 돌려준다.
    fn try_step_up(
        &self,
        world: &World,
        from: Vec3,
        horizontal: Vec3,
        filter: QueryFilter,
    ) -> Result<Option<Vec3>, ApiError> {
        let up_motion = self.up * self.step_height;
        let (lifted, _) = self.sweep(world, from, up_motion, filter)?;
        // 충분히 못 올라갔으면 천장이 낮은 것 — 계단이 아니다.
        if (lifted - from).length() < self.step_height * 0.5 {
            return Ok(None);
        }
        let (forward, _) = self.sweep(world, lifted, horizontal, filter)?;
        let gained = (forward - lifted).length();
        if gained <= self.skin {
            return Ok(None); // 들어 올려도 여전히 막혔다
        }
        // 내려놓기: 올라간 만큼 + 턱 여유까지 내려간다. 착지 못하면 허공이므로 취소.
        let down = self.up * -((lifted - from).length() + self.step_height);
        let (landed, hit) = self.sweep(world, forward, down, filter)?;
        match hit {
            Some(h) if h.normal.dot(self.up) >= self.max_slope_cos => Ok(Some(landed)),
            _ => Ok(None),
        }
    }
}
