//! **점 투영** — 임의의 점에서 셰이프 겉면 위의 최근접점 (rapier `project_point` 대응).
//!
//! 레이 캐스트가 "이 방향으로 쏘면 어디에 닿나"라면 이건 "이 점에서 가장 가까운 겉면이
//! 어디냐"다. 호스트는 이걸로 스냅·집기·자석 배치·최단거리 판정을 만든다.
//!
//! **반환은 항상 겉면 위 점이다** — 질의점이 셰이프 안에 있어도 그렇다. rapier의
//! `solid=true`(안에 있으면 점 자신) 의미가 필요하면 호출자가
//! `if proj.inside { p } else { proj.point }`로 만들면 된다. 반대 방향(겉면 점에서
//! solid 의미를 복원하는 것)은 불가능하므로 **더 많은 정보를 주는 쪽**을 원시 연산으로 둔다.
//!
//! 안팎 판정(`inside`)은 `contains_point`와 **같은 답**이어야 한다 — 두 질의가 갈리면
//! 호스트가 "안에 있다는데 투영은 밖을 가리킨다"를 보게 된다. 테스트가 셰이프별로 잠근다.

use crate::gjk::{gjk, GjkResult};
use crate::ray::QueryReject;
use crate::{Aabb, Pose, Shape};
use maru_math::{Quat, Vec3};

/// 방향이 없을 때 쓰는 **결정적** 축 — 임의로 고른 값이 아니라 고정 규약이다.
/// (구의 중심·박스 코어 안·평면 위 점처럼 "가장 가까운 겉면 방향"이 유일하지 않은 자리.)
const TIE_X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
const TIE_Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);

/// 한 번의 투영 질의(월드 질의는 물체 전체)가 들여다볼 수 있는 **후보 수** 상한.
///
/// 단위는 "후보 하나를 들여다본 횟수"다 — 메시 삼각형 하나, hull 지지 내적 하나가 각 1이다.
/// 32점 hull 코어 **안쪽** 질의 하나가 최악 약 16만이므로, 한 물체는 넉넉히 통과하고
/// 그런 물체가 여럿 겹친 질의는 막힌다(검수 rev_c30d32fd MAJOR).
pub const MAX_PROJECT_WORK: usize = 1 << 18;

/// GJK 한 번이 부를 수 있는 support 횟수 상한 (`gjk::MAX_ITER`와 같은 값).
///
/// 예산 선청구용 보수 상수다. 실제보다 크게 물지만, 예산의 목적은 정산이 아니라
/// **한 질의가 프레임을 삼키지 못하게 하는 것**이다.
const GJK_MAX_SUPPORTS: usize = 64;

#[inline]
fn splat(v: f32) -> Vec3 {
    Vec3::new(v, v, v)
}

/// 점 투영 결과.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointProjection {
    /// 셰이프 **겉면**(convex radius 포함) 위의 최근접점 — 월드 좌표.
    pub point: Vec3,
    /// 질의점이 셰이프 안에 있었는가 (경계 위는 안으로 친다 — `contains_point`와 동일).
    pub inside: bool,
}

/// 공개 진입 — 형상·산술 입력을 검사하고 자세를 정규화한 뒤 푼다.
///
/// `try_cast_ray`·`try_contains_point`와 같은 계약이다: 잘못된 입력은 "먼 점"으로
/// 위장하지 않고 오류로 나온다.
pub fn try_project_point(
    shape: &Shape,
    pose: &Pose,
    p: Vec3,
) -> Result<PointProjection, QueryReject> {
    let mut work = MAX_PROJECT_WORK;
    try_project_point_with_budget(shape, pose, p, &mut work)
}

/// **작업 예산을 공유하는** 투영 — 여러 물체를 훑는 호출자용.
///
/// `budget`은 "후보 하나를 들여다본 횟수"의 잔량이고 호출 사이에 **누적**된다. 소진되면
/// 답을 지어내지 않고 `ResourceLimit`이다. 한 물체가 아무리 비싸도 상한이 있지만, 진짜
/// 위험은 **여러 물체가 겹친 한 번의 질의**다: hull 코어 안쪽 질의는 물체마다 지지 함수를
/// 전수로 돌리므로 겹친 hull 수만큼 곱해진다(검수 rev_c30d32fd MAJOR). 예산을 호출자가
/// 들고 다니면 그 곱셈이 한 프레임을 삼키지 못한다.
pub fn try_project_point_with_budget(
    shape: &Shape,
    pose: &Pose,
    p: Vec3,
    budget: &mut usize,
) -> Result<PointProjection, QueryReject> {
    if !shape.is_castable() || !crate::vec_in_range(p) || !crate::vec_in_range(pose.pos) {
        return Err(QueryReject::InvalidInput);
    }
    let pose = Pose {
        pos: pose.pos,
        rot: pose.rot.try_normalize().ok_or(QueryReject::InvalidInput)?,
    };
    let local = pose.rot.conjugate().rotate(p - pose.pos);
    let proj = project_local(shape, local, budget)?;
    Ok(PointProjection {
        point: pose.pos + pose.rot.rotate(proj.point),
        inside: proj.inside,
    })
}

/// 로컬 좌표 투영 — **검증된 입력 전용**(크레이트 내부·월드 경계 통과분).
pub(crate) fn project_local(
    shape: &Shape,
    p: Vec3,
    budget: &mut usize,
) -> Result<PointProjection, QueryReject> {
    let r = shape.radius();
    match *shape {
        Shape::Sphere { radius } => Ok(inflate(Vec3::ZERO, p, radius, TIE_X)),
        Shape::Capsule {
            half_height,
            radius,
        } => {
            let c = Vec3::new(0.0, p.y.clamp(-half_height, half_height), 0.0);
            Ok(inflate(c, p, radius, TIE_X))
        }
        Shape::Box { half, radius } => Ok(project_box(half, radius, p)),
        Shape::Triangle { points, radius } => {
            let q = closest_on_triangle(p, points);
            // fallback은 삼각형 **평면의 법선**이다. 외적이 정확히 0(세 점이 한 직선)일
            // 때만 그 직선에 수직인 방향으로 간다 — 고정축으로 마감하면 퇴화한 선분과
            // 나란할 수 있다(검수 rev_ef5d1868 MAJOR: (-1,0,0)·(1,0,0)·(0,0,0)에
            // radius 0.1, 원점 질의가 (0.1,0,0)을 냈다 — 여전히 코어 선분 안이다).
            // **임계로 랭크를 판정하지 않는다** — 얇은 삼각형도 평면은 잘 정의된다.
            let fallback = (points[1] - points[0])
                .cross(points[2] - points[0])
                .try_normalize()
                .unwrap_or_else(|| collinear_exit_dir(&points));
            let mut proj = inflate(q, p, radius, fallback);
            // 삼각형은 **부피가 없다** — radius가 0이면 어떤 점도 안이 아니다
            // (`contains_point`의 삼각형 분기와 같은 계약).
            proj.inside &= radius > 0.0;
            Ok(proj)
        }
        Shape::Hull { .. } => project_convex_core(shape, p, r, budget),
        Shape::Compound { ref children } => project_compound(children, p, budget),
        Shape::TriMesh(ref mesh) => project_trimesh(mesh, p, r, budget),
        Shape::Heightfield { .. } => {
            let hf = shape.heightfield_ref().expect("Heightfield 변종");
            project_heightfield(&hf, p, r, budget)
        }
    }
}

/// 코어 위 최근접점 `q`에서 **겉면**으로 반지름만큼 밀어낸다.
///
/// `p == q`(코어 안 또는 코어 표면 위)면 방향이 없다 — 그때만 `fallback`을 쓴다.
/// 방향이 없다고 0 벡터를 내보내면 "겉면 위 점"이라는 계약이 깨진다.
#[inline]
fn inflate(q: Vec3, p: Vec3, radius: f32, fallback: Vec3) -> PointProjection {
    let v = p - q;
    let l2 = v.length_sq();
    if l2 > 0.0 {
        PointProjection {
            point: q + v * (radius / l2.sqrt()),
            inside: l2 <= radius * radius,
        }
    } else {
        PointProjection {
            point: q + fallback * radius,
            inside: true,
        }
    }
}

/// 박스 — 코어(겉면 half에서 radius를 뺀 상자)에 클램프한 뒤 부풀린다.
///
/// 코어 **안쪽** 점은 클램프가 자기 자신을 돌려주므로 방향이 없다. 그때는 축별로 코어
/// 면까지의 여유를 재서 **가장 가까운 면**으로 내보낸다(동률은 낮은 축 번호 — 결정론).
fn project_box(half: Vec3, radius: f32, p: Vec3) -> PointProjection {
    let core = Vec3::new(
        (half.x - radius).max(0.0),
        (half.y - radius).max(0.0),
        (half.z - radius).max(0.0),
    );
    let q = Vec3::new(
        p.x.clamp(-core.x, core.x),
        p.y.clamp(-core.y, core.y),
        p.z.clamp(-core.z, core.z),
    );
    if (p - q).length_sq() > 0.0 {
        return inflate(q, p, radius, TIE_X);
    }
    let pa = [p.x, p.y, p.z];
    let ca = [core.x, core.y, core.z];
    let mut axis = 0usize;
    let mut best = ca[0] - pa[0].abs();
    for (i, (c, v)) in ca.iter().zip(pa.iter()).enumerate().skip(1) {
        let gap = c - v.abs();
        if gap < best {
            best = gap;
            axis = i;
        }
    }
    // 0은 +로 민다 — 부호가 없는 축에서 결정적으로 한 쪽을 고른다.
    let sign = if pa[axis] < 0.0 { -1.0 } else { 1.0 };
    let mut out = pa;
    out[axis] = sign * (ca[axis] + radius);
    PointProjection {
        point: Vec3::new(out[0], out[1], out[2]),
        inside: true,
    }
}

/// 일반 convex 코어(hull) — 검증된 GJK/EPA를 재사용한다.
///
/// 점을 반지름 0 구로 모델링하면 점-볼록체 거리는 셰이프-셰이프 거리의 특수한 경우다
/// (`ray::closest_core_dist`와 같은 기법 — 여기서는 witness 점까지 받는다).
/// 코어 **안쪽** 점은 GJK가 교차를 주므로 EPA로 최소 이탈 방향을 구한다: 최소 관통
/// 깊이의 반대 방향이 곧 최근접 겉면 방향이다.
fn project_convex_core(
    shape: &Shape,
    p: Vec3,
    radius: f32,
    budget: &mut usize,
) -> Result<PointProjection, QueryReject> {
    let point = Shape::Sphere { radius: 0.0 };
    let at = Pose {
        pos: p,
        rot: Quat::IDENTITY,
    };
    let ident = Pose {
        pos: Vec3::ZERO,
        rot: Quat::IDENTITY,
    };
    // **GJK 비용을 먼저 청구한다.** GJK도 support마다 hull 점을 전수 내적하고 반복 상한이
    // 있으므로, 이걸 안 세면 "코어 밖 hull이 수천 개"인 질의가 예산을 한 톨도 안 쓰고
    // 수백만 내적을 돈다(검수 rev_8a258ed3 MAJOR). 보수적으로 상한(점수 × 반복상한)을
    // 선청구한다 — 실제보다 크게 물지만, 예산은 "못 넘게 하는 것"이지 정산이 아니다.
    if let Shape::Hull { ref points, .. } = *shape {
        let cost = points.len().saturating_mul(GJK_MAX_SUPPORTS);
        if *budget < cost {
            return Err(QueryReject::ResourceLimit);
        }
        *budget -= cost;
    }
    match gjk(&point, &at, shape, &ident) {
        GjkResult::Separated { on_b, .. } => {
            // **반지름이 0이어도 같은 경로를 탄다.** 예전에는 "두께가 없으면 코어가 곧
            // 겉면"이라며 GJK `on_b`를 그대로 돌려줬는데, 아래에서 보듯 GJK는 코어 안의
            // 점에도 양의 거리를 준다 — 그러면 코어 **안쪽** 점을 겉면이라고 내놓는다
            // (검수 rev_b16c028d BLOCKER). radius가 0이면 아래 식이 자연히
            // `p + n·gap`(코어 표면)과 `on_b`(밖일 때)로 떨어진다.
            //
            // **코어 AABB 밖이면 확실히 밖이다** — 전수 열거 없이 witness 방향을 쓴다.
            // (아래 지지 최소화는 C(m,3)·m이라 비싸다. 이 가지치기는 순수 비교라 안전하다.)
            let bb = shape.local_aabb();
            let outside_aabb = p.x < bb.min.x
                || p.x > bb.max.x
                || p.y < bb.min.y
                || p.y > bb.max.y
                || p.z < bb.min.z
                || p.z > bb.max.z;
            if !outside_aabb {
                // **안팎 판정은 지지 여유로 한다 — GJK의 `distance`를 믿지 않는다.**
                //
                // GJK는 코어 **안**의 점에도 양의 거리를 돌려줄 수 있다: 같은 support가
                // 다시 나오면 조기 종료하는데, 그때 심플렉스의 최근접 벡터가 0이 아닐 수
                // 있다. 실측(검수 rev_230afeaf BLOCKER): 반크기 (1000,1,1) hull의 내부점
                // (999.999, 0.25, 0)에서 `distance = 0.1768`이 나왔다 — 그걸 믿으면
                // "밖"으로 분기해 코어 **안쪽** 점을 겉면이라고 돌려준다.
                //
                // 지지 여유 `gap = min_n max_p n·(pt − p)`는 그런 착시가 없다: 모든 지지
                // 반공간 안이면 `gap >= 0`이고, 이 사례에서도 +0.000977로 정확히 안이라고
                // 답한다. `gap == 0`(지지 평면 위)일 때만 "평면 안이지만 다각형 밖"을
                // 가려야 해서 `distance`를 보는데, 그 경우는 GJK가 0을 정확히 준다.
                // 안팎은 `core_exit`이 끝낸다 — GJK 거리는 여기서 쓰지 않는다.
                let (n, gap, inside_core) = core_exit(shape, p, budget)?;
                if inside_core {
                    return Ok(PointProjection {
                        point: p + n * (gap + radius),
                        // 코어 안이면 겉면(코어 ⊕ 반지름 공) 안이기도 하다.
                        inside: true,
                    });
                }
            }
            // 코어 밖 — 최근접점에서 질의점 방향으로 radius.
            let v = p - on_b;
            let l = v.length();
            if l > 0.0 {
                return Ok(PointProjection {
                    point: on_b + v * (radius / l),
                    inside: l <= radius,
                });
            }
            // 방향이 없다(witness가 질의점과 같다) — 지지 방향으로 마감한다.
            let (n, gap, _) = core_exit(shape, p, budget)?;
            Ok(PointProjection {
                point: p + n * (gap + radius),
                inside: true,
            })
        }
        GjkResult::Intersecting(_) => {
            // 코어 안이다. **EPA를 쓰지 않는다** — EPA는 GJK가 부피 있는 사면체 시드를
            // 만들 수 있을 때만 돌고, 그 판정은 최대 에지 길이에 **상대적**이라 얇고 긴
            // 코어(예: 반크기 (1000, 0.002, 0.002))에서는 전체 차원인데도 시드를 못 만든다.
            // 지지 함수 경로는 그 문턱이 없다.
            let (n, gap, _) = core_exit(shape, p, budget)?;
            Ok(PointProjection {
                point: p + n * (gap + radius),
                inside: true,
            })
        }
    }
}

/// 점이 hull **코어 안**인가 — GJK의 `distance`를 믿지 않는 판정.
///
/// GJK는 코어 안의 점에도 양의 거리를 돌려줄 수 있다(같은 support 재등장 시 조기 종료).
/// 그래서 포함 질의와 투영이 **같은 판정**을 써야 한다 — 안 그러면 호스트가 "안에 있다는데
/// 투영은 밖"을 본다(검수 rev_230afeaf BLOCKER).
///
/// AABB 밖은 즉시 거짓이라 비싼 전수 열거는 애매한 경우에만 돈다.
pub(crate) fn core_contains(
    shape: &Shape,
    p: Vec3,
    budget: &mut usize,
) -> Result<bool, QueryReject> {
    // **GJK 비용을 AABB 검사보다 먼저 청구한다.** 호출자는 이미 GJK를 돌린 뒤 들어오므로,
    // AABB 밖에서 조기 반환하면 그 GJK가 통째로 예산 밖이 된다 — "멀리 있는 hull 수만 개"인
    // 질의가 수천만 내적을 돌면서 예산이 한 톨도 안 줄었다(검수 rev_6beb627b MAJOR).
    if let Shape::Hull { ref points, .. } = *shape {
        let cost = points.len().saturating_mul(GJK_MAX_SUPPORTS);
        if *budget < cost {
            return Err(QueryReject::ResourceLimit);
        }
        *budget -= cost;
    }
    let bb = shape.local_aabb();
    if p.x < bb.min.x
        || p.x > bb.max.x
        || p.y < bb.min.y
        || p.y > bb.max.y
        || p.z < bb.min.z
        || p.z > bb.max.z
    {
        return Ok(false);
    }
    // 예산은 **호출자가 들고 다닌다** — 물체마다 새로 주면 "질의점을 품는 hull 수천 개"인
    // 월드가 한 번의 질의에서 수억 번 내적을 돈다. 소진은 거짓으로 뭉개지 않고 오류다.
    let (_, _, inside) = core_exit(shape, p, budget)?;
    Ok(inside)
}

/// 코어 **위/안**의 점 `q`에서 나가는 방향과, 그 방향으로 코어 표면까지의 여유.
///
/// 겉면 위 점은 `q + dir * (gap + radius)`다.
///
/// 왜 GJK/EPA를 쓰지 않나: 둘 다 **상대 임계**에 걸린다. 코어가 평면·선이면 법선이
/// 코어 안 방향으로 나오고(검수 rev_6429a223), 전체 차원이어도 종횡비가 크면 EPA 시드
/// 조건(`부피 > 1e-6·최대에지³`)을 못 넘겨 관통 경로 자체가 안 열린다(rev_ef5d1868).
/// 두 실패 모두 "그럴듯하게 틀린" 값이라 0-검사로는 안 걸린다.
///
/// 대신 **지지 함수**로 푼다. 볼록체 안의 점에서 표면까지의 거리는
/// `min_n ( max_p n·(p − q) )` 이고, 최솟값을 주는 `n`이 곧 가장 가까운 면의 바깥 법선이다.
/// 임의의 `n`에 대해 `max_p n·(p − q)`는 그 방향 지지 평면까지의 거리이므로 **항상 답 이상**
/// 이다 — 후보에 진짜 면 법선만 들어 있으면 최솟값이 정확한 답이 된다. 그리고 볼록 껍질의
/// 모든 면은 껍질 점 3개가 이룬다. 임계도 분기도 없다.
///
/// 비용은 점 수의 4제곱(`C(m,3)·m`)이지만 상한이 32개로 잠겨 있고, 이 경로는 질의점이
/// 코어 **안**일 때만 돈다(밖이면 위치 차이가 방향을 준다).
fn core_exit(shape: &Shape, q: Vec3, budget: &mut usize) -> Result<(Vec3, f32, bool), QueryReject> {
    let Shape::Hull { ref points, .. } = *shape else {
        // 다른 셰이프는 해석 경로가 따로 있어 여기 오지 않는다.
        return Ok((TIE_X, 0.0, false));
    };
    // **지지 최소화를 먼저 돌린다.** 예전에는 아핀 랭크를 먼저 재서 "저차원이면 여유 0"으로
    // 갈랐는데, 그 판정이 상대 임계라 **얇지만 3차원인** 코어를 평면으로 오판했다. 그러면
    // 그 방향의 진짜 여유(반두께)를 0으로 버려 반환점이 코어 안에 남는다
    // (검수 rev_c30d32fd BLOCKER: 반크기 (1000, 7.5e-4, 7.5e-4) + radius 1e-6).
    //
    // 순서를 뒤집으면 임계가 아예 없어진다: 평면 코어는 후보 법선이 그 평면의 법선뿐이고
    // 모든 점이 평면 위라 여유가 **자연히 0**으로 나오며, 얇은 3차원 코어는 반두께가
    // 그대로 나온다. 같은 식이 두 경우를 다 맞힌다.
    let mut best: Option<(f32, Vec3)> = None;
    let mut exhausted = false;
    // **코어가 3차원이라는 증거를 봤는가.** 어떤 후보 법선 방향으로든 점들이 두께를 가지면
    // (`hi > lo`) 3차원이다. 저차원(평면·선)이면 모든 후보에서 두께가 0이라 증거가 없다.
    //
    // 왜 필요한가: 3차원 코어에서는 `gap >= 0`이 곧 "코어 안/경계"의 증거다. 저차원이면
    // 아핀 껍질 위 **어디서나** `gap == 0`이라 "다각형 안"과 "평면 위 다각형 밖"이 안 갈려
    // 다른 근거가 더 필요하다. 둘을 같은 규칙으로 다루면 3차원 코어의 **면 위** 점이
    // 밖으로 오판된다(검수 rev_bb1b7abe BLOCKER).
    let mut full_dim = false;
    'outer: for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            for k in (j + 1)..points.len() {
                // **외적·정규화 앞에서 청구한다.** 뒤에 두면 공선 삼중조합이 `continue`로
                // 빠져나가 `C(m,3)`번의 검사가 통째로 예산 밖이 된다 — 전부 공선인 hull은
                // 점 순회를 한 번도 안 돌아 예산이 0만큼 깎인다(검수 rev_bb1b7abe MAJOR).
                if *budget == 0 {
                    exhausted = true;
                    break 'outer;
                }
                *budget -= 1;
                let Some(n) = (points[j] - points[i])
                    .cross(points[k] - points[i])
                    .try_normalize()
                else {
                    continue; // 세 점이 한 직선 위 — 면을 이루지 못한다
                };
                // **한 번의 순회로 두 방향을 다 본다**: +n의 지지 거리는 max, −n의 지지
                // 거리는 −min이다. 그리고 둘 다 현재 최선을 넘긴 순간 이 법선은 답이 될 수
                // 없으므로 끊는다(정확성 불변 — 최소를 찾는 중이다).
                let bound = best.map_or(f32::INFINITY, |(h, _)| h);
                let (mut hi, mut lo) = (f32::NEG_INFINITY, f32::INFINITY);
                for pt in points {
                    if *budget == 0 {
                        // **중간에 끊긴 최소값은 답이 아니다.** 아직 못 본 면이 더 가까울 수
                        // 있고, 그러면 반환점이 코어 안에 남는다 — 조용한 오답이다.
                        exhausted = true;
                        break 'outer;
                    }
                    *budget -= 1;
                    let d = n.dot(*pt - q);
                    hi = hi.max(d);
                    lo = lo.min(d);
                    if hi >= bound && -lo >= bound {
                        break;
                    }
                }
                // 부분 순회여도 `hi > lo`는 3차원의 **증거**다(반증은 못 된다).
                full_dim |= hi > lo;
                // 동률은 먼저 본 후보가 이긴다 — 순회가 결정적이므로 답도 결정적이다.
                for (h, cand) in [(hi, n), (-lo, n * -1.0)] {
                    if h.is_finite() && best.is_none_or(|(bh, _)| h < bh) {
                        best = Some((h, cand));
                    }
                }
            }
        }
    }
    if exhausted {
        return Err(QueryReject::ResourceLimit);
    }
    // **안팎 판정은 여기서 끝낸다** — 호출자가 GJK 거리로 보정하지 않게.
    //
    // 3차원 코어: `gap >= 0`이면 모든 지지 반공간 안이므로 코어 안/경계다.
    // 저차원 코어: 아핀 껍질 위 어디서나 `gap == 0`이라 그것만으로는 "다각형 안"인지
    //   모른다. 한 차원 낮춰 같은 열거를 돌린다 — 평면 코어의 "면"은 **에지**이고 그
    //   바깥 법선은 `cross(에지, 평면법선)`이다. GJK를 안 쓰므로 중복-support 결함이 없다
    //   (검수 rev_6beb627b BLOCKER: 평면 삼각형의 내부점에서 GJK가 0.0147을 줘 밖으로 갈렸다).
    Ok(match best {
        // **부호를 살린다.** 이 함수는 "코어에 아주 가까운 점"까지 받는데, 그 점이 코어
        // 바깥이면 지지 여유가 음수다. 0으로 자르면 그만큼 겉면을 지나친다
        // (검수 rev_8a258ed3 BLOCKER: 반크기 (1000,1,1) hull·radius 0.1에 p=(1000.001,0,0)
        //  → 겉면 x=1000.1인데 x=1000.101이 나왔다). 부호를 살리면 같은 식
        //  `p + n·(gap + radius)`가 안팎 모두에서 정확한 표면을 준다.
        Some((h, n)) => {
            let inside = if full_dim {
                h >= 0.0
            } else if h == 0.0 {
                // 평면 위다 — 다각형 안인지 에지 열거로 가린다.
                in_plane_contains(points, q, n, budget)?
            } else {
                false // 평면 밖
            };
            (n, h, inside)
        }
        // 면 후보가 하나도 없다 = 모든 점이 한 직선 위(또는 3점 미만). 그 직선에는 지지
        // 평면이 무한히 많아 위 최소화가 방향을 못 고르므로 수직 방향을 준다.
        // 면 후보가 하나도 없다 = 모든 점이 한 직선 위(또는 3점 미만).
        None => (
            collinear_exit_dir(points),
            0.0,
            on_core_segment(points, q, budget)?,
        ),
    })
}

/// 평면 코어에서 `q`가 **다각형 안**인가 — 에지 바깥 법선으로 지지 열거를 한 차원 낮춰 돈다.
///
/// 평면 법선 `n`에 수직인 에지 법선 `cross(에지, n)`에 대해 모든 점이 `q` 쪽에 있으면
/// 그 에지는 지지선이고, 모든 지지선 안쪽이면 다각형 안이다. 3차원 면 열거와 같은 논증이다.
fn in_plane_contains(
    points: &[Vec3],
    q: Vec3,
    n: Vec3,
    budget: &mut usize,
) -> Result<bool, QueryReject> {
    let mut worst = f32::INFINITY;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            // 차감은 정규화 **앞**이다 — 뒤에 두면 퇴화 에지가 공짜로 검사된다.
            if *budget == 0 {
                return Err(QueryReject::ResourceLimit);
            }
            *budget -= 1;
            let Some(e) = (points[j] - points[i]).cross(n).try_normalize() else {
                continue; // 에지가 평면 법선과 나란(퇴화) — 지지선을 못 만든다
            };
            let mut hi = f32::NEG_INFINITY;
            let mut lo = f32::INFINITY;
            for pt in points {
                if *budget == 0 {
                    return Err(QueryReject::ResourceLimit);
                }
                *budget -= 1;
                let d = e.dot(*pt - q);
                hi = hi.max(d);
                lo = lo.min(d);
            }
            // `+e`가 바깥이면 여유는 hi, `-e`가 바깥이면 −lo다. 둘 중 작은 쪽이 이 에지의 여유.
            worst = worst.min(hi.min(-lo));
        }
    }
    // 지지선이 하나도 없으면(모든 에지 퇴화) 판정할 근거가 없다 — 안이라고 우기지 않는다.
    Ok(worst.is_finite() && worst >= 0.0)
}

/// 모든 점이 한 직선 위일 때 `q`가 **그 선분 위**인가.
fn on_core_segment(points: &[Vec3], q: Vec3, budget: &mut usize) -> Result<bool, QueryReject> {
    let Some(&p0) = points.first() else {
        return Ok(false);
    };
    let mut dir = None;
    for pt in points {
        if let Some(d) = (*pt - p0).try_normalize() {
            dir = Some(d);
            break;
        }
    }
    let Some(d) = dir else {
        return Ok(q == p0); // 모든 점이 한 자리 — 코어는 점이다
    };
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for pt in points {
        if *budget == 0 {
            return Err(QueryReject::ResourceLimit);
        }
        *budget -= 1;
        let t = d.dot(*pt - p0);
        lo = lo.min(t);
        hi = hi.max(t);
    }
    let t = d.dot(q - p0).clamp(lo, hi);
    Ok(q == p0 + d * t)
}

/// 모든 점이 **한 직선**(또는 한 자리)에 있을 때의 이탈 방향 — 그 직선에 수직인 벡터.
///
/// 임계로 랭크를 재지 않는다. 호출자는 "면 후보가 하나도 없었다"는 사실로 이미 공선을
/// 알고 있고, 여기서 임계를 또 쓰면 얇은 형상을 오판하는 자리가 다시 생긴다.
fn collinear_exit_dir(points: &[Vec3]) -> Vec3 {
    let Some(&p0) = points.first() else {
        return TIE_X;
    };
    for q in points {
        if let Some(d) = (*q - p0).try_normalize() {
            return perpendicular_to(d);
        }
    }
    TIE_X // 모든 점이 한 자리 — 코어는 점이라 어느 방향이든 겉면에 닿는다
}

/// `d`에 수직인 결정적 단위 벡터 — 가장 덜 정렬된 축과 외적한다(퇴화 없음).
fn perpendicular_to(d: Vec3) -> Vec3 {
    let a = if d.x.abs() <= d.y.abs() && d.x.abs() <= d.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if d.y.abs() <= d.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    d.cross(a).try_normalize().unwrap_or(TIE_X)
}

/// 복합 — **자식 겉면들의 합집합** 위 최근접점.
///
/// 자식이 서로 겹치지 않으면 이것이 곧 복합 전체의 겉면이다(정상적인 복합 셰이프).
/// 겹치는 자식을 준 경우 반환점이 다른 자식 안쪽일 수 있다 — 그건 "합집합의 경계"가
/// 아니라 "경계들의 합집합"이며, rapier도 같은 정의다. 계약으로 명시하고 숨기지 않는다.
/// (진짜 합집합 경계는 자식 쌍마다 곡면 교선을 풀어야 한다.)
fn project_compound(
    children: &[(Pose, Shape)],
    p: Vec3,
    budget: &mut usize,
) -> Result<PointProjection, QueryReject> {
    // 빈 복합은 답이 없다 — "무한히 먼 점"으로 위장하면 호스트가 NaN을 먹는다.
    // (`validate`가 1..=8을 강제하지만 이 함수는 그걸 가정하지 않는다.)
    let mut best: Option<(f32, Vec3)> = None;
    let mut inside = false;
    for (cp, cs) in children {
        // **자식 회전을 정규화한다** — `contains_point`·`cast_ray`가 쓰는 것과 같은
        // 변환이어야 한다. 형상 검증은 단위에서 길이제곱 1e-3까지 벗어난 자식 자세를
        // 통과시키므로, 정규화를 빼면 같은 합법 형상에서 안팎 판정이 다른 질의와 갈린다
        // (검수 rev_6429a223 MAJOR).
        let rot = cp.rot.try_normalize().ok_or(QueryReject::InvalidInput)?;
        let local = rot.conjugate().rotate(p - cp.pos);
        let sub = project_local(cs, local, budget)?;
        let world = cp.pos + rot.rotate(sub.point);
        inside |= sub.inside;
        let d2 = (world - p).length_sq();
        // 동률은 **먼저 온 자식**이 이긴다(엄격 부등호) — 순서만이 결정한다.
        if best.is_none_or(|(bd, _)| d2 < bd) {
            best = Some((d2, world));
        }
    }
    match best {
        Some((_, point)) => Ok(PointProjection { point, inside }),
        None => Err(QueryReject::InvalidInput),
    }
}

/// 삼각형 다발(메시·하이트필드) 공통 — **반경을 넓혀 가며** 후보를 훑는다.
///
/// 전수 순회는 메시가 커지면 못 쓴다. 그렇다고 반경 하나를 찍으면 그 안에 삼각형이 없을
/// 때 답이 없다. 그래서 `p`에서 셰이프 AABB까지의 거리(하한)에서 시작해 **답을 찾고 그
/// 답이 현재 반경 안에 들어올 때까지** 배로 넓힌다 — 넓힌 반경 밖에 더 가까운 삼각형은
/// 있을 수 없으므로 그 시점의 답이 정확하다. AABB 전체를 덮으면 반드시 끝난다.
struct NearestTri {
    best_d2: f32,
    best: Vec3,
}

impl NearestTri {
    fn new() -> Self {
        Self {
            best_d2: f32::INFINITY,
            best: Vec3::ZERO,
        }
    }
    #[inline]
    fn feed(&mut self, p: Vec3, tri: [Vec3; 3]) {
        let q = closest_on_triangle(p, tri);
        let d2 = (q - p).length_sq();
        // 동률은 먼저 본 삼각형이 이긴다 — 순회 순서가 결정론이므로 답도 결정적이다.
        if d2 < self.best_d2 {
            self.best_d2 = d2;
            self.best = q;
        }
    }
}

/// 확장 검색 **전체**의 후보 예산 — 라운드마다가 아니라 누적이다.
///
/// 없으면 합법적인 단일 질의 하나가 백만 삼각형을 훑는다: 평평한 지형에 아주 높은 점을
/// 질의하면 첫 반경이 이미 AABB 전체를 덮어 모든 셀이 후보가 된다(검수 rev_6429a223 MAJOR).
/// BVH 스택 포화는 **깊이**만 막지 후보 수를 막지 않는다. 캐스트·매니폴드 경로와 같은
/// 상수를 써서 엔진 전체가 한 예산 정책을 쓰게 한다.
const PROJECT_TRI_BUDGET: usize = crate::MAX_HEIGHTFIELD_CELLS_PER_QUERY;

/// 확장 검색의 반경 수열: AABB까지의 거리에서 시작해 배로 늘리고, 전체를 덮으면 멈춘다.
fn search_radii(p: Vec3, bounds: Aabb) -> (f32, f32) {
    let clamped = Vec3::new(
        p.x.clamp(bounds.min.x, bounds.max.x),
        p.y.clamp(bounds.min.y, bounds.max.y),
        p.z.clamp(bounds.min.z, bounds.max.z),
    );
    let d0 = (clamped - p).length();
    let ext = bounds.max - bounds.min;
    let full = d0 + ext.x.max(ext.y).max(ext.z);
    // 첫 반경은 0이면 안 된다(0 AABB 질의는 경계 삼각형을 놓칠 수 있다).
    let seed = (full * 0.0625).max(f32::MIN_POSITIVE);
    (d0.max(seed), full.max(seed))
}

fn project_trimesh(
    mesh: &crate::trimesh::TriMeshData,
    p: Vec3,
    radius: f32,
    budget: &mut usize,
) -> Result<PointProjection, QueryReject> {
    let bounds = mesh.local_aabb();
    let (mut r, full) = search_radii(p, bounds);
    let mut acc;
    let mut tri_budget = PROJECT_TRI_BUDGET;
    loop {
        acc = NearestTri::new();
        let q = Aabb::new(p - splat(r), p + splat(r));
        let mut over = false;
        // 순회 스택 포화도 예산 소진도 **후보 누락**이다 — 그대로 답을 내면 "더 먼 겉면"을
        // 최근접이라 우기게 된다. 오답 대신 자원 오류로 드러낸다.
        let complete = mesh.for_each_candidate(q, |i| {
            if tri_budget == 0 || *budget == 0 {
                over = true;
                return false;
            }
            tri_budget -= 1;
            *budget -= 1;
            acc.feed(p, mesh.triangle(i));
            true
        });
        if over || !complete {
            return Err(QueryReject::ResourceLimit);
        }
        if acc.best_d2.sqrt() <= r || r >= full {
            break;
        }
        r = (r * 2.0).min(full);
    }
    if !acc.best_d2.is_finite() {
        // 삼각형이 하나도 없는 메시 — `TriMeshData::new`가 막지만 가정하지 않는다.
        return Err(QueryReject::InvalidInput);
    }
    // 메시는 표면이지 부피가 아니다 — `contains_point`와 같이 항상 밖이다.
    let mut proj = inflate(acc.best, p, radius, TIE_X);
    proj.inside = false;
    Ok(proj)
}

fn project_heightfield(
    hf: &crate::heightfield::HeightfieldRef<'_>,
    p: Vec3,
    radius: f32,
    budget: &mut usize,
) -> Result<PointProjection, QueryReject> {
    let bounds = hf.local_aabb();
    let (mut r, full) = search_radii(p, bounds);
    let mut acc;
    let mut cell_budget = PROJECT_TRI_BUDGET;
    loop {
        acc = NearestTri::new();
        let q = Aabb::new(p - splat(r), p + splat(r));
        let mut over = false;
        if let Some(range) = hf.cell_range(q) {
            crate::heightfield::for_each_overlapping_cell(hf, range, p.y - r, p.y + r, |cx, cz| {
                if cell_budget == 0 || *budget < 2 {
                    over = true;
                    return false;
                }
                cell_budget -= 1;
                *budget -= 2;
                acc.feed(p, hf.triangle(cx, cz, 0));
                acc.feed(p, hf.triangle(cx, cz, 1));
                true
            });
        }
        if over {
            return Err(QueryReject::ResourceLimit);
        }
        if acc.best_d2.sqrt() <= r || r >= full {
            break;
        }
        r = (r * 2.0).min(full);
    }
    if !acc.best_d2.is_finite() {
        return Err(QueryReject::InvalidInput);
    }
    let mut proj = inflate(acc.best, p, radius, TIE_Y);
    // 하이트필드는 **표면 아래를 속이 찬 것으로** 본다 — `contains_point`와 같은 계약이라
    // 반지름 근방 판정이 아니라 높이 비교가 답이다.
    proj.inside = hf.height_at(p.x, p.z).is_some_and(|h| p.y <= h);
    Ok(proj)
}

/// 점 → 삼각형 최근접점 (보로노이 영역 분해).
///
/// 정점 3 · 에지 3 · 내부 1의 일곱 영역을 순서대로 배제한다. 분모가 0이 되는 퇴화
/// 삼각형에서도 정점을 돌려주므로 NaN이 나가지 않는다.
#[inline]
pub(crate) fn closest_on_triangle(p: Vec3, tri: [Vec3; 3]) -> Vec3 {
    let (a, b, c) = (tri[0], tri[1], tri[2]);
    let ab = b - a;
    let ac = c - a;
    let d1 = ab.dot(p - a);
    let d2 = ac.dot(p - a);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let d3 = ab.dot(p - b);
    let d4 = ac.dot(p - b);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        return if denom > 0.0 {
            a + ab * (d1 / denom)
        } else {
            a
        };
    }
    let d5 = ab.dot(p - c);
    let d6 = ac.dot(p - c);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        return if denom > 0.0 {
            a + ac * (d2 / denom)
        } else {
            a
        };
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        return if denom > 0.0 {
            b + (c - b) * ((d4 - d3) / denom)
        } else {
            b
        };
    }
    let sum = va + vb + vc;
    if sum > 0.0 {
        let inv = 1.0 / sum;
        a + ab * (vb * inv) + ac * (vc * inv)
    } else {
        a // 퇴화(면적 0·NaN) — 내부 영역이 없다. 정점으로 마감해 NaN을 내보내지 않는다.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray::try_contains_point;

    fn ident() -> Pose {
        Pose {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        }
    }

    /// 회전+평행이동이 섞인 자세 — 로컬↔월드 왕복이 빠졌는지 잡는다.
    fn tilted() -> Pose {
        Pose {
            pos: Vec3::new(3.0, -2.0, 7.0),
            rot: Quat::from_axis_angle(Vec3::new(1.0, 2.0, -0.5), 0.9),
        }
    }

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1u32 << 24) as f32
        }
        fn point(&mut self, span: f32) -> Vec3 {
            v(
                (self.next() - 0.5) * span,
                (self.next() - 0.5) * span,
                (self.next() - 0.5) * span,
            )
        }
    }

    fn shapes() -> Vec<(&'static str, Shape)> {
        let hull_pts = vec![
            v(-0.8, -0.5, -0.4),
            v(0.9, -0.5, -0.4),
            v(0.0, 1.1, -0.4),
            v(0.0, 0.0, 0.7),
        ];
        vec![
            ("구", Shape::Sphere { radius: 1.3 }),
            (
                "캡슐",
                Shape::Capsule {
                    half_height: 0.7,
                    radius: 0.4,
                },
            ),
            (
                "각진 박스",
                Shape::Box {
                    half: v(0.6, 1.0, 0.35),
                    radius: 0.0,
                },
            ),
            (
                "둥근 박스",
                Shape::Box {
                    half: v(0.9, 0.7, 0.5),
                    radius: 0.25,
                },
            ),
            (
                "hull(각)",
                Shape::Hull {
                    points: hull_pts.clone(),
                    radius: 0.0,
                },
            ),
            (
                "hull(둥근)",
                Shape::Hull {
                    points: hull_pts,
                    radius: 0.2,
                },
            ),
            (
                "hull(평면 코어+두께)",
                Shape::Hull {
                    points: vec![
                        v(-0.9, 0.0, -0.6),
                        v(0.9, 0.0, -0.6),
                        v(0.9, 0.0, 0.6),
                        v(-0.9, 0.0, 0.6),
                    ],
                    radius: 0.18,
                },
            ),
            (
                "삼각형(얇음)",
                Shape::Triangle {
                    points: [v(-1.0, 0.0, -0.6), v(1.2, 0.0, -0.6), v(0.0, 0.0, 0.9)],
                    radius: 0.15,
                },
            ),
            (
                "복합",
                Shape::Compound {
                    children: vec![
                        (ident(), Shape::Sphere { radius: 0.5 }),
                        (
                            Pose {
                                pos: v(1.4, 0.0, 0.0),
                                rot: Quat::IDENTITY,
                            },
                            Shape::Box {
                                half: v(0.4, 0.4, 0.4),
                                radius: 0.0,
                            },
                        ),
                    ],
                },
            ),
        ]
    }

    // ---------- 해석적 대조 ----------

    #[test]
    fn sphere_projection_is_analytic() {
        let s = Shape::Sphere { radius: 2.0 };
        let pose = tilted();
        // 밖: 중심에서 같은 방향, 거리 정확히 radius.
        let p = pose.pos + v(0.0, 5.0, 0.0);
        let pr = try_project_point(&s, &pose, p).unwrap();
        assert!(!pr.inside);
        assert!((pr.point - pose.pos).length() - 2.0 < 1e-5);
        assert!((pr.point - (pose.pos + v(0.0, 2.0, 0.0))).length() < 1e-4);
        // 안: 여전히 겉면 위.
        let p = pose.pos + v(0.5, 0.0, 0.0);
        let pr = try_project_point(&s, &pose, p).unwrap();
        assert!(pr.inside);
        assert!(((pr.point - pose.pos).length() - 2.0).abs() < 1e-4);
        // 중심: 방향이 없다 — 겉면 위 결정적 한 점.
        let pr = try_project_point(&s, &pose, pose.pos).unwrap();
        assert!(pr.inside);
        assert!(((pr.point - pose.pos).length() - 2.0).abs() < 1e-4);
    }

    #[test]
    fn capsule_projection_is_analytic() {
        let s = Shape::Capsule {
            half_height: 1.0,
            radius: 0.5,
        };
        // 옆면: 축에서 수직으로 radius.
        let pr = try_project_point(&s, &ident(), v(3.0, 0.5, 0.0)).unwrap();
        assert!(!pr.inside);
        assert!((pr.point - v(0.5, 0.5, 0.0)).length() < 1e-5);
        // 반구 끝: 축 끝점에서 radius.
        let pr = try_project_point(&s, &ident(), v(0.0, 4.0, 0.0)).unwrap();
        assert!((pr.point - v(0.0, 1.5, 0.0)).length() < 1e-5);
        // 축 위(방향 없음): 겉면까지 정확히 radius.
        let pr = try_project_point(&s, &ident(), v(0.0, 0.3, 0.0)).unwrap();
        assert!(pr.inside);
        assert!(((pr.point - v(0.0, 0.3, 0.0)).length() - 0.5).abs() < 1e-5);
    }

    /// 둥근 박스의 **모서리는 깎여 있다** — half 상자로 재면 안이라고 우기는 자리다.
    /// `radius == half`면 그 박스는 구다.
    #[test]
    fn rounded_box_corner_is_cut_not_square() {
        let s = Shape::Box {
            half: v(1.0, 1.0, 1.0),
            radius: 1.0,
        };
        let corner = v(1.0, 1.0, 1.0); // 중심에서 √3 — 반지름 1인 구의 밖
        let pr = try_project_point(&s, &ident(), corner).unwrap();
        assert!(!pr.inside, "깎인 모서리를 안이라고 판정했다");
        assert!(
            (pr.point.length() - 1.0).abs() < 1e-5,
            "겉면(반지름 1)이 아니다: {}",
            pr.point.length()
        );
        // 안팎 질의도 같은 답이어야 한다.
        assert!(!try_contains_point(&s, &ident(), corner).unwrap());
        // 면 중앙은 여전히 안이다.
        assert!(try_contains_point(&s, &ident(), v(0.99, 0.0, 0.0)).unwrap());
    }

    #[test]
    fn box_interior_projects_to_nearest_face() {
        let s = Shape::Box {
            half: v(2.0, 0.5, 3.0),
            radius: 0.0,
        };
        // Y가 가장 얇다 — 안쪽 점은 Y면으로 나간다.
        let pr = try_project_point(&s, &ident(), v(0.1, 0.2, -0.3)).unwrap();
        assert!(pr.inside);
        assert!(
            (pr.point - v(0.1, 0.5, -0.3)).length() < 1e-6,
            "{:?}",
            pr.point
        );
        // 아래쪽이면 −Y로.
        let pr = try_project_point(&s, &ident(), v(0.1, -0.2, -0.3)).unwrap();
        assert!((pr.point - v(0.1, -0.5, -0.3)).length() < 1e-6);
        // X가 가장 가까운 자리.
        let pr = try_project_point(&s, &ident(), v(1.9, 0.0, 0.0)).unwrap();
        assert!(
            (pr.point - v(2.0, 0.0, 0.0)).length() < 1e-6,
            "{:?}",
            pr.point
        );
    }

    /// hull은 GJK/EPA 경로다 — 같은 기하를 박스로 준 해석 경로와 답이 같아야 한다.
    #[test]
    fn hull_matches_analytic_box() {
        let h = v(0.7, 0.5, 0.9);
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(h.x * sx, h.y * sy, h.z * sz));
                }
            }
        }
        let hull = Shape::Hull {
            points: pts,
            radius: 0.0,
        };
        let boxy = Shape::Box {
            half: h,
            radius: 0.0,
        };
        let mut rng = Lcg(7);
        for _ in 0..200 {
            let p = rng.point(4.0);
            let a = try_project_point(&hull, &ident(), p).unwrap();
            let b = try_project_point(&boxy, &ident(), p).unwrap();
            assert_eq!(a.inside, b.inside, "안팎이 갈렸다 p={p:?}");
            // 내부 점은 등거리 면이 여러 개일 수 있으므로 **거리**로 비교한다.
            let da = (a.point - p).length();
            let db = (b.point - p).length();
            assert!(
                (da - db).abs() < 2e-3,
                "hull {da} vs box {db} (p={p:?}, inside={})",
                a.inside
            );
        }
    }

    #[test]
    fn triangle_without_radius_is_never_inside() {
        let tri = Shape::Triangle {
            points: [v(-1.0, 0.0, -1.0), v(1.0, 0.0, -1.0), v(0.0, 0.0, 1.0)],
            radius: 0.0,
        };
        // 삼각형 위 점 — 겉면 위지만 부피가 없으므로 안이 아니다.
        let pr = try_project_point(&tri, &ident(), v(0.0, 0.0, 0.0)).unwrap();
        assert!(!pr.inside);
        assert!((pr.point - v(0.0, 0.0, 0.0)).length() < 1e-6);
        assert_eq!(
            pr.inside,
            try_contains_point(&tri, &ident(), Vec3::ZERO).unwrap()
        );
        // 정점 밖은 그 정점으로.
        let pr = try_project_point(&tri, &ident(), v(5.0, 0.0, -5.0)).unwrap();
        assert!(
            (pr.point - v(1.0, 0.0, -1.0)).length() < 1e-5,
            "{:?}",
            pr.point
        );
    }

    #[test]
    fn compound_picks_nearest_child_surface() {
        let s = Shape::Compound {
            children: vec![
                (ident(), Shape::Sphere { radius: 0.5 }),
                (
                    Pose {
                        pos: v(5.0, 0.0, 0.0),
                        rot: Quat::IDENTITY,
                    },
                    Shape::Sphere { radius: 0.5 },
                ),
            ],
        };
        let pr = try_project_point(&s, &ident(), v(-3.0, 0.0, 0.0)).unwrap();
        assert!(
            (pr.point - v(-0.5, 0.0, 0.0)).length() < 1e-5,
            "{:?}",
            pr.point
        );
        let pr = try_project_point(&s, &ident(), v(8.0, 0.0, 0.0)).unwrap();
        assert!(
            (pr.point - v(5.5, 0.0, 0.0)).length() < 1e-5,
            "{:?}",
            pr.point
        );
        // 둘째 자식 안 → inside
        let pr = try_project_point(&s, &ident(), v(5.1, 0.0, 0.0)).unwrap();
        assert!(pr.inside);
    }

    // ---------- 불변식 (전 셰이프) ----------

    /// **안팎 판정은 `contains_point`와 같은 답이어야 한다.**
    /// 두 질의가 갈리면 호스트가 "안에 있다는데 투영은 밖"을 보게 된다.
    #[test]
    fn inside_agrees_with_contains_point() {
        let mut rng = Lcg(31);
        for (name, s) in shapes() {
            for pose in [ident(), tilted()] {
                for _ in 0..300 {
                    let p = pose.pos + rng.point(5.0);
                    let a = try_project_point(&s, &pose, p).unwrap().inside;
                    let b = try_contains_point(&s, &pose, p).unwrap();
                    assert_eq!(a, b, "{name}: 안팎이 갈렸다 p={p:?}");
                }
            }
        }
    }

    /// **겉면 위 점을 다시 투영하면 자기 자신이다** — 반환점이 진짜 겉면 위라는 증명이다.
    /// (반환점이 코어 위거나 안쪽이면 두 번째 투영이 그걸 밖으로 밀어내 어긋난다.)
    #[test]
    fn projection_is_idempotent_on_surface() {
        let mut rng = Lcg(1009);
        for (name, s) in shapes() {
            for _ in 0..300 {
                let p = rng.point(6.0);
                let once = try_project_point(&s, &ident(), p).unwrap();
                let twice = try_project_point(&s, &ident(), once.point).unwrap();
                let drift = (twice.point - once.point).length();
                assert!(
                    drift < 3e-3,
                    "{name}: 겉면 위가 아니다 (드리프트 {drift}, p={p:?})"
                );
            }
        }
    }

    /// **더 가까운 겉면 점은 없다** — 겉면을 조밀히 표본해 반환값이 최소인지 확인한다.
    /// 표본은 상한이므로 "반환 거리 ≤ 표본 최소 거리 + 여유"만 요구한다.
    #[test]
    fn no_sampled_surface_point_is_closer() {
        let mut rng = Lcg(4242);
        for (name, s) in shapes() {
            // 겉면 표본: 사방에서 투영해 모은다(투영이 겉면 위라는 건 위 테스트가 잠근다).
            let mut surface = Vec::new();
            let mut s_rng = Lcg(77);
            for _ in 0..2000 {
                let far = s_rng.point(1.0);
                if far.length_sq() <= 0.0 {
                    continue;
                }
                let far = far * (12.0 / far.length());
                surface.push(try_project_point(&s, &ident(), far).unwrap().point);
            }
            for _ in 0..60 {
                let p = rng.point(7.0);
                let got = try_project_point(&s, &ident(), p).unwrap();
                let mine = (got.point - p).length();
                let best = surface
                    .iter()
                    .map(|q| (*q - p).length())
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    mine <= best + 1e-3,
                    "{name}: 표본이 더 가깝다 ({mine} > {best}, p={p:?})"
                );
            }
        }
    }

    // ---------- 메시·하이트필드 ----------

    fn brute_force_nearest(tris: &[[Vec3; 3]], p: Vec3) -> Vec3 {
        let mut best = (f32::INFINITY, Vec3::ZERO);
        for t in tris {
            let q = closest_on_triangle(p, *t);
            let d2 = (q - p).length_sq();
            if d2 < best.0 {
                best = (d2, q);
            }
        }
        best.1
    }

    #[test]
    fn trimesh_matches_brute_force() {
        // 계단형 리본 — BVH가 여러 갈래로 갈리도록.
        let mut verts = Vec::new();
        let mut tris = Vec::new();
        for i in 0..12u32 {
            let x = i as f32 * 0.9;
            let y = ((i % 3) as f32) * 0.6;
            verts.push(v(x, y, -1.0));
            verts.push(v(x, y, 1.0));
        }
        for i in 0..11u32 {
            let (a, b, c, d) = (i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 3);
            tris.push([a, b, c]);
            tris.push([b, d, c]);
        }
        let mesh = Shape::trimesh(verts.clone(), tris.clone()).expect("메시");
        let flat: Vec<[Vec3; 3]> = tris
            .iter()
            .map(|t| {
                [
                    verts[t[0] as usize],
                    verts[t[1] as usize],
                    verts[t[2] as usize],
                ]
            })
            .collect();
        let mut rng = Lcg(555);
        for _ in 0..400 {
            let p = v(
                rng.next() * 14.0 - 2.0,
                rng.next() * 6.0 - 3.0,
                rng.next() * 6.0 - 3.0,
            );
            let got = try_project_point(&mesh, &ident(), p).unwrap();
            let want = brute_force_nearest(&flat, p);
            assert!(!got.inside, "메시는 표면이지 부피가 아니다");
            let (dg, dw) = ((got.point - p).length(), (want - p).length());
            assert!(
                (dg - dw).abs() < 1e-4,
                "확장 검색이 더 가까운 삼각형을 놓쳤다 {dg} vs {dw} (p={p:?})"
            );
        }
    }

    #[test]
    fn heightfield_matches_brute_force_and_contains() {
        let (nx, nz, cell) = (9u32, 7u32, 0.8f32);
        let heights: Vec<f32> = (0..(nx * nz))
            .map(|i| ((i % 5) as f32) * 0.3 - 0.4)
            .collect();
        let hf = Shape::heightfield(heights.clone(), nx, nz, cell).expect("하이트필드");
        let href = hf.heightfield_ref().unwrap();
        let mut flat = Vec::new();
        let (ncx, ncz) = href.cells();
        for cz in 0..ncz {
            for cx in 0..ncx {
                flat.push(href.triangle(cx, cz, 0));
                flat.push(href.triangle(cx, cz, 1));
            }
        }
        let mut rng = Lcg(9001);
        for _ in 0..400 {
            let p = v(
                rng.next() * 10.0 - 2.0,
                rng.next() * 4.0 - 2.0,
                rng.next() * 8.0 - 2.0,
            );
            let got = try_project_point(&hf, &ident(), p).unwrap();
            let want = brute_force_nearest(&flat, p);
            let (dg, dw) = ((got.point - p).length(), (want - p).length());
            assert!(
                (dg - dw).abs() < 1e-4,
                "확장 검색 누락 {dg} vs {dw} (p={p:?})"
            );
            assert_eq!(
                got.inside,
                try_contains_point(&hf, &ident(), p).unwrap(),
                "하이트필드 안팎이 갈렸다 p={p:?}"
            );
        }
    }

    // ---------- 입력 검증 ----------

    // ---------- 검수 rev_6429a223 회귀 ----------

    /// **저차원 코어 + 두께**: z=0 평면 hull에 convex radius를 주면 겉면은 z=±r인데,
    /// GJK 법선을 그대로 쓰면 반올림이 만든 **평면 안** 방향으로 밀어 물체 안에 머문다.
    #[test]
    fn flat_hull_with_radius_escapes_the_plane() {
        let s = Shape::Hull {
            points: vec![
                v(-1.0, -1.0, 0.0),
                v(1.0, -1.0, 0.0),
                v(1.0, 1.0, 0.0),
                v(-1.0, 1.0, 0.0),
            ],
            radius: 0.1,
        };
        assert!(s.validate().is_ok(), "합법 셰이프라는 전제가 깨졌다");
        for p in [v(0.0, 0.0, 0.0), v(0.5, -0.3, 0.0), v(-0.9, 0.9, 0.0)] {
            let pr = try_project_point(&s, &ident(), p).unwrap();
            assert!(pr.inside, "코어 안인데 밖이라고 했다 p={p:?}");
            assert!(
                (pr.point.z.abs() - 0.1).abs() < 1e-4,
                "겉면(z=±0.1)이 아니다: {:?} (p={p:?})",
                pr.point
            );
            assert!(
                ((pr.point - p).length() - 0.1).abs() < 1e-4,
                "겉면까지 거리가 radius가 아니다: {} (p={p:?})",
                (pr.point - p).length()
            );
        }
        // 선형 코어(두 점)도 같은 계약이다 — 여공간이 2차원이라 방향 선택이 결정적이어야 한다.
        let seg = Shape::Hull {
            points: vec![v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
            radius: 0.3,
        };
        let pr = try_project_point(&seg, &ident(), v(0.2, 0.0, 0.0)).unwrap();
        assert!(pr.inside);
        assert!(
            ((pr.point - v(0.2, 0.0, 0.0)).length() - 0.3).abs() < 1e-4,
            "{:?}",
            pr.point
        );
    }

    /// **전체 차원이지만 얇고 긴 코어**: EPA 시드 조건(`부피 > 1e-6·최대에지³`)이
    /// 최대 에지에 상대적이라, 종횡비가 크면 관통 경로가 아예 안 열린다. 지지 함수
    /// 경로는 그 문턱이 없어야 한다(검수 rev_ef5d1868 BLOCKER).
    #[test]
    fn sliver_hull_exits_through_the_thin_axis() {
        let h = v(1000.0, 0.002, 0.002);
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(h.x * sx, h.y * sy, h.z * sz));
                }
            }
        }
        let s = Shape::Hull {
            points: pts,
            radius: 0.1,
        };
        assert!(s.validate().is_ok(), "합법 셰이프라는 전제가 깨졌다");
        let pr = try_project_point(&s, &ident(), Vec3::ZERO).unwrap();
        assert!(pr.inside);
        // 얇은 축(±Y·±Z)으로 나가야 한다 — 긴 축으로 나가면 코어 안에 그대로 남는다.
        assert!(
            pr.point.x.abs() < 1.0e-4,
            "긴 축으로 나갔다(코어 안에 남는다): {:?}",
            pr.point
        );
        let d = pr.point.length();
        assert!(
            (d - 0.102).abs() < 1.0e-4,
            "겉면까지 거리가 반두께+radius가 아니다: {d}"
        );
    }

    /// **얇지만 3차원인 코어**: 아핀 랭크를 임계로 먼저 재면 이걸 평면으로 오판하고
    /// 그 방향의 진짜 여유(반두께)를 0으로 버린다(검수 rev_c30d32fd BLOCKER).
    /// 지지 최소화를 먼저 돌리면 같은 식이 평면 코어와 얇은 3차원 코어를 다 맞힌다.
    #[test]
    fn thin_but_three_dimensional_core_keeps_its_half_thickness() {
        let h = v(1000.0, 7.5e-4, 7.5e-4);
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(h.x * sx, h.y * sy, h.z * sz));
                }
            }
        }
        let radius = 1.0e-6;
        let s = Shape::Hull {
            points: pts,
            radius,
        };
        assert!(s.validate().is_ok(), "합법 셰이프라는 전제가 깨졌다");
        let pr = try_project_point(&s, &ident(), Vec3::ZERO).unwrap();
        assert!(pr.inside);
        // 겉면까지 거리 = 반두께 + radius. 여유를 버리면 radius(1e-6)만 움직인다.
        let d = pr.point.length();
        assert!(
            (d - (7.5e-4 + radius)).abs() < 1.0e-6,
            "반두께를 버렸다(코어 안에 남는다): 거리 {d}, 점 {:?}",
            pr.point
        );
        assert!(
            pr.point.x.abs() < 1.0e-6,
            "긴 축으로 나갔다: {:?}",
            pr.point
        );
    }

    /// 코어 **바깥** 질의도 예산을 먹어야 한다 — GJK도 support마다 점을 전수 내적한다.
    /// 안 세면 "밖에 있는 hull이 수천 개"인 질의가 예산을 한 톨도 안 쓰고 돈다
    /// (검수 rev_8a258ed3 MAJOR).
    #[test]
    fn budget_counts_gjk_for_points_outside_the_core() {
        // 원점을 품지 않는 tetra: 모든 꼭짓점이 x+y+z >= 1 위에 있다.
        let s = Shape::Hull {
            points: vec![
                v(1.0, 1.0, 1.0),
                v(-1.0, 1.0, 1.0),
                v(1.0, -1.0, 1.0),
                v(1.0, 1.0, -1.0),
            ],
            radius: 0.05,
        };
        // 전제: 질의점이 코어 **밖**이라 지지 함수 경로(비싼 쪽)를 안 탄다.
        let pr = try_project_point(&s, &ident(), Vec3::ZERO).unwrap();
        assert!(!pr.inside, "밖이라는 전제가 깨졌다");
        let mut budget = super::MAX_PROJECT_WORK;
        let mut ok = 0usize;
        let mut limited = false;
        for _ in 0..8192 {
            match try_project_point_with_budget(&s, &ident(), Vec3::ZERO, &mut budget) {
                Ok(_) => ok += 1,
                Err(QueryReject::ResourceLimit) => {
                    limited = true;
                    break;
                }
                Err(e) => panic!("예상 밖 오류 {e:?}"),
            }
        }
        assert!(ok > 0, "첫 질의부터 막히면 선청구가 너무 크다");
        assert!(
            limited,
            "코어 밖 질의가 예산을 전혀 쓰지 않았다 (통과 {ok}회)"
        );
    }

    /// **GJK의 `distance`는 코어 안에서도 양수일 수 있다.** 같은 support가 다시 나와
    /// 조기 종료할 때 심플렉스 최근접 벡터가 0이 아닌 경우다 — 그걸 안팎 판정에 쓰면
    /// 코어 안쪽 점을 "겉면"이라고 돌려준다(검수 rev_230afeaf BLOCKER).
    #[test]
    fn interior_point_is_inside_even_when_gjk_reports_distance() {
        let h = v(1000.0, 1.0, 1.0);
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(h.x * sx, h.y * sy, h.z * sz));
                }
            }
        }
        let radius = 0.1;
        let s = Shape::Hull {
            points: pts,
            radius,
        };
        let p = v(999.999, 0.25, 0.0);
        // 전제: 이 점은 코어 안이다(모든 축에서 반크기 안).
        assert!(p.x.abs() <= h.x && p.y.abs() <= h.y && p.z.abs() <= h.z);
        // **반지름 0 hull도 같은 계약이다** — 두께가 없다고 GJK 답을 그대로 쓰면
        // 코어 안쪽 점을 겉면이라고 내놓는다(검수 rev_b16c028d BLOCKER).
        {
            let thin = Shape::Hull {
                points: match &s {
                    Shape::Hull { points, .. } => points.clone(),
                    _ => unreachable!(),
                },
                radius: 0.0,
            };
            let pr0 = try_project_point(&thin, &ident(), p).unwrap();
            assert!(pr0.inside, "radius 0: 코어 안인데 밖이라고 했다: {pr0:?}");
            assert!(
                pr0.point.x >= h.x - 1.0e-3,
                "radius 0: 코어 표면이 아니라 안쪽 점을 돌려줬다: {:?}",
                pr0.point
            );
            assert_eq!(
                pr0.inside,
                try_contains_point(&thin, &ident(), p).unwrap(),
                "radius 0: 안팎이 갈렸다"
            );
        }
        let pr = try_project_point(&s, &ident(), p).unwrap();
        assert!(pr.inside, "코어 안인데 밖이라고 했다: {pr:?}");
        // 겉면은 코어 표면에서 radius만큼 밖이다. 가장 가까운 면은 +X(여유 0.001)이므로
        // 답은 x ≈ 1000 + radius 근처여야 하고, **코어 안(|x| < 1000)이면 안 된다**.
        assert!(
            pr.point.x > h.x,
            "겉면이 아니라 코어 안쪽 점을 돌려줬다: {:?}",
            pr.point
        );
        assert!(
            ((pr.point - p).length() - (0.001 + radius)).abs() < 1.0e-3,
            "겉면까지 거리가 여유+radius가 아니다: {} (점 {:?})",
            (pr.point - p).length(),
            pr.point
        );
        // 안팎 판정은 `contains_point`와도 같아야 한다 — 두 질의가 갈리면 호스트가
        // "안에 있다는데 투영은 밖"을 보게 된다.
        assert_eq!(
            pr.inside,
            try_contains_point(&s, &ident(), p).unwrap(),
            "안팎이 갈렸다"
        );
    }

    /// **평면 코어의 다각형 내부** 점도 GJK 거리로 가리면 안 된다 — 같은 중복-support
    /// 결함이 평면에서도 양의 거리를 준다(검수 rev_6beb627b BLOCKER).
    /// 한 차원 낮춘 에지 열거로 풀어야 한다.
    #[test]
    fn planar_polygon_interior_is_inside_without_gjk() {
        let radius = 0.0005;
        let s = Shape::Hull {
            points: vec![
                v(-4.0, -913.0, 0.0),
                v(93.0, -692.0, 0.0),
                v(487.0, 366.0, 0.0),
            ],
            radius,
        };
        assert!(s.validate().is_ok(), "합법 셰이프라는 전제가 깨졌다");
        // 무게중심 근방 — 명백한 다각형 내부.
        let p = v(270.6, -207.20001, 0.0);
        let pr = try_project_point(&s, &ident(), p).unwrap();
        assert!(pr.inside, "다각형 안인데 밖이라고 했다: {pr:?}");
        // 겉면은 평면에서 radius만큼 떨어진 자리다.
        assert!(
            (pr.point.z.abs() - radius).abs() < 1.0e-5,
            "겉면(z=±{radius})이 아니다: {:?}",
            pr.point
        );
        assert_eq!(
            pr.inside,
            try_contains_point(&s, &ident(), p).unwrap(),
            "안팎이 갈렸다"
        );
        // 평면 위지만 다각형 **밖**인 점은 여전히 밖이다.
        let out = v(-500.0, 500.0, 0.0);
        let pr_out = try_project_point(&s, &ident(), out).unwrap();
        assert!(!pr_out.inside, "다각형 밖인데 안이라고 했다: {pr_out:?}");
    }

    /// **코어 면 위** 점은 3차원 코어에서 `gap == 0`이다 — 그때 GJK 거리를 추가 근거로
    /// 요구하면 같은 중복-support 결함에 다시 걸려 경계를 밖으로 오판한다
    /// (검수 rev_bb1b7abe BLOCKER). 3차원이 확인되면 `gap >= 0`만으로 충분하다.
    #[test]
    fn point_on_a_core_face_is_inside() {
        let h = v(1000.0, 1.0, 1.0);
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(h.x * sx, h.y * sy, h.z * sz));
                }
            }
        }
        for radius in [0.0f32, 0.1] {
            let s = Shape::Hull {
                points: pts.clone(),
                radius,
            };
            // +X 면 **위**의 점(코어 경계).
            let p = v(1000.0, 0.25, 0.0);
            let pr = try_project_point(&s, &ident(), p).unwrap();
            assert!(pr.inside, "radius {radius}: 경계인데 밖이라고 했다: {pr:?}");
            // 겉면은 면에서 radius만큼 밖이다.
            assert!(
                (pr.point.x - (h.x + radius)).abs() < 1.0e-3,
                "radius {radius}: 겉면이 아니다: {:?}",
                pr.point
            );
            assert_eq!(
                pr.inside,
                try_contains_point(&s, &ident(), p).unwrap(),
                "radius {radius}: 안팎이 갈렸다"
            );
        }
    }

    /// 전부 공선인 hull은 유효 법선이 하나도 안 나오는데, 그 삼중조합 검사 자체가 일이다.
    /// 예산이 점 순회에서만 깎이면 그 작업이 **공짜로** 계산돼 상한이 무력화된다
    /// (검수 rev_bb1b7abe MAJOR).
    #[test]
    fn collinear_hull_triples_consume_budget() {
        // x축 위 32점 — 모든 삼중조합이 공선이라 법선이 안 나온다.
        let pts: Vec<Vec3> = (0..32)
            .map(|i| v(i as f32 * 0.05 - 0.8, 0.0, 0.0))
            .collect();
        let s = Shape::Hull {
            points: pts,
            radius: 0.001,
        };
        assert!(s.validate().is_ok(), "합법 셰이프라는 전제가 깨졌다");
        // 선분 위 점은 안, 벗어난 점은 밖이어야 한다 — 저차원 코어의 포함 판정이
        // "항상 안"으로 무너지면 여기서 걸린다.
        assert!(
            try_project_point(&s, &ident(), Vec3::ZERO).unwrap().inside,
            "선분 위인데 밖이라고 했다"
        );
        // AABB **안**이면서 선분 밖인 점 — 이것만이 저차원 포함 판정에 도달한다
        // (AABB 밖은 그 전에 잘려 판정을 안 탄다).
        for outside in [v(2.0, 0.0, 0.0), v(0.0, 0.5, 0.0), v(0.7509, 0.0009, 0.0)] {
            let pr = try_project_point(&s, &ident(), outside).unwrap();
            assert!(
                !pr.inside,
                "선분 밖인데 안이라고 했다: {outside:?} → {pr:?}"
            );
            assert_eq!(
                pr.inside,
                try_contains_point(&s, &ident(), outside).unwrap(),
                "안팎이 갈렸다 {outside:?}"
            );
        }
        // 한 번의 질의가 **삼중조합 수만큼은** 먹어야 한다. GJK 선청구(32×64=2048)만
        // 세면 이 값이 그 아래로 떨어진다 — 그래서 횟수가 아니라 **소모량**을 잰다.
        let mut probe = usize::MAX;
        try_project_point_with_budget(&s, &ident(), Vec3::ZERO, &mut probe).unwrap();
        let used = usize::MAX - probe;
        let triples = 32 * 31 * 30 / 6; // C(32,3) = 4960
        assert!(
            used >= triples,
            "공선 삼중조합 검사가 예산을 안 먹었다: {used} < {triples}"
        );
        // 그리고 반복하면 실제로 막혀야 한다.
        let mut budget = super::MAX_PROJECT_WORK;
        let mut ok = 0usize;
        let mut limited = false;
        for _ in 0..4096 {
            match try_project_point_with_budget(&s, &ident(), Vec3::ZERO, &mut budget) {
                Ok(_) => ok += 1,
                Err(QueryReject::ResourceLimit) => {
                    limited = true;
                    break;
                }
                Err(e) => panic!("예상 밖 오류 {e:?}"),
            }
        }
        assert!(ok > 0, "첫 질의부터 막히면 예산이 너무 빡빡하다");
        assert!(limited, "반복 질의가 예산에 안 걸렸다 (통과 {ok}회)");
    }

    /// **모서리 바로 바깥** 점은 면 법선 후보로 풀 수 없다 — 실제 최근접 방향이 모서리에서
    /// 질의점으로 가는 방향이라 어떤 면 법선도 아니다(검수 rev_2da962df BLOCKER).
    /// 코어 밖이면 거리가 아무리 작아도 GJK witness 방향을 써야 한다.
    #[test]
    fn point_just_outside_an_edge_uses_the_witness_direction() {
        let h = v(1000.0, 1.0, 1.0);
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(h.x * sx, h.y * sy, h.z * sz));
                }
            }
        }
        let radius = 0.001;
        let s = Shape::Hull {
            points: pts,
            radius,
        };
        assert!(s.validate().is_ok(), "합법 셰이프라는 전제가 깨졌다");
        // 코어 모서리 (1000, 1, 0)에서 대각으로 살짝 벗어난 점.
        let p = v(1000.0019, 1.0006, 0.0);
        let edge = v(1000.0, 1.0, 0.0);
        let d = (p - edge).length();
        assert!(
            d < 0.002 && d > radius,
            "전제: 모서리에서 아주 가깝지만 radius보다는 멀다 (d={d})"
        );
        let pr = try_project_point(&s, &ident(), p).unwrap();
        assert!(!pr.inside, "밖인데 안이라고 했다");
        // 겉면 위 점은 코어(모서리)에서 정확히 radius만큼 떨어져 있어야 한다.
        let from_edge = (pr.point - edge).length();
        // 허용치는 **GJK witness 정밀도**다. 좌표 크기가 1000이면 f32 간격이 6e-5라
        // witness가 그 정도 흔들린다(실측 2.2e-5). 면 법선 후보로 푸는 옛 경로는 오차가
        // 1.7e-4라 이 허용치로도 갈린다 — 뮤테이션으로 확인했다.
        const TOL: f32 = 6.0e-5;
        assert!(
            (from_edge - radius).abs() < TOL,
            "겉면이 아니다: 모서리에서 {from_edge} (기대 {radius}), 점 {:?}",
            pr.point
        );
        // 그리고 질의점과 모서리를 잇는 선 위에 있어야 한다.
        let want = edge + (p - edge) * (radius / d);
        assert!(
            (pr.point - want).length() < TOL,
            "{:?} vs {want:?}",
            pr.point
        );
    }

    /// 코어 **바로 바깥**(반올림 임계 안쪽) 점도 지지 함수 경로로 가는데, 그때 여유는
    /// 음수다. 0으로 자르면 그만큼 겉면을 지나친다(검수 rev_8a258ed3 BLOCKER).
    #[test]
    fn point_just_outside_a_long_core_lands_exactly_on_the_surface() {
        let h = v(1000.0, 1.0, 1.0);
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(h.x * sx, h.y * sy, h.z * sz));
                }
            }
        }
        let s = Shape::Hull {
            points: pts,
            radius: 0.1,
        };
        assert!(s.validate().is_ok());
        // 코어 밖 0.001 — 특성 길이가 커서 "위치 차이" 임계 안쪽으로 들어온다.
        let p = v(1000.001, 0.0, 0.0);
        // 전제: 이 구성이 정말 "위치 차이" 임계 안쪽으로 들어온다(= 지지 함수 경로).
        // 특성 길이 ≈ 2000이라 eps = 1e-6·2000 = 0.002 > 0.001이다. 이게 깨지면 아래
        // 단언이 다른 경로를 재게 되므로 테스트가 무의미해진다.
        let eps = crate::length_eps(crate::characteristic_length(&s, &s));
        assert!(
            eps > 0.001,
            "임계가 좁아 지지 함수 경로에 안 들어간다: {eps}"
        );
        let pr = try_project_point(&s, &ident(), p).unwrap();
        assert!(
            (pr.point.x - 1000.1).abs() < 3.0e-4,
            "겉면(x=1000.1)을 지나쳤다: {:?}",
            pr.point
        );
        // 겉면 위 점이므로 재투영이 제자리여야 한다.
        let again = try_project_point(&s, &ident(), pr.point).unwrap();
        assert!(
            (again.point - pr.point).length() < 3.0e-4,
            "드리프트 {}",
            (again.point - pr.point).length()
        );
    }

    /// 예산은 **호출 사이에 누적**된다 — 겹친 hull이 많은 한 번의 질의가 프레임을 삼키지
    /// 못하게 하는 것이 목적이다(검수 rev_c30d32fd MAJOR).
    #[test]
    fn shared_budget_stops_repeated_interior_hull_queries() {
        // 32점 hull(구 표면 샘플) — 코어 안쪽 질의가 지지 함수를 전수로 돈다.
        let mut pts = Vec::new();
        let mut rng = Lcg(4321);
        while pts.len() < 32 {
            let d = rng.point(2.0);
            if let Some(u) = d.try_normalize() {
                pts.push(u);
            }
        }
        let s = Shape::Hull {
            points: pts,
            radius: 0.05,
        };
        // 예산 없이 한 번은 통과한다.
        assert!(try_project_point(&s, &ident(), Vec3::ZERO).is_ok());
        // 공유 예산으로 반복하면 언젠가 막힌다 — 그리고 "먼 점"이 아니라 오류다.
        // 소모량을 재서 예산 상한이 "한 물체는 통과, 여럿 겹치면 차단"인지 확인한다.
        let mut probe = usize::MAX;
        try_project_point_with_budget(&s, &ident(), Vec3::ZERO, &mut probe).unwrap();
        let per_query = usize::MAX - probe;
        assert!(
            per_query < super::MAX_PROJECT_WORK,
            "한 물체가 예산을 다 먹는다: {per_query}"
        );
        let mut budget = super::MAX_PROJECT_WORK;
        let mut ok = 0usize;
        let mut limited = false;
        for _ in 0..4096 {
            match try_project_point_with_budget(&s, &ident(), Vec3::ZERO, &mut budget) {
                Ok(_) => ok += 1,
                Err(QueryReject::ResourceLimit) => {
                    limited = true;
                    break;
                }
                Err(e) => panic!("예상 밖 오류 {e:?}"),
            }
        }
        assert!(ok > 0, "첫 질의부터 막히면 예산이 너무 빡빡하다");
        assert!(
            limited,
            "예산이 누적되지 않아 무한히 통과했다 (통과 {ok}회)"
        );
    }

    /// 면적 0 삼각형(사실상 선분)도 `is_castable`을 통과한다 — fallback이 코어와
    /// **나란한** 축이면 반환점이 코어 안에 남는다(검수 rev_ef5d1868 MAJOR).
    #[test]
    fn degenerate_triangle_exits_perpendicular_to_its_core() {
        let s = Shape::Triangle {
            points: [v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 0.0, 0.0)],
            radius: 0.1,
        };
        assert!(
            s.is_castable_pub_for_test(),
            "질의 입력으로 수용된다는 전제"
        );
        for q in [Vec3::ZERO, v(0.5, 0.0, 0.0), v(-1.0, 0.0, 0.0)] {
            let pr = try_project_point(&s, &ident(), q).unwrap();
            let d = pr.point - q;
            assert!(
                (d.length() - 0.1).abs() < 1.0e-5,
                "겉면까지 거리가 radius가 아니다: {} (q={q:?})",
                d.length()
            );
            assert!(
                d.x.abs() < 1.0e-5,
                "코어 선분과 나란한 방향으로 나갔다: {d:?} (q={q:?})"
            );
            // 겉면 위 점이므로 재투영이 제자리여야 한다.
            let again = try_project_point(&s, &ident(), pr.point).unwrap();
            assert!((again.point - pr.point).length() < 1.0e-5, "q={q:?}");
        }
    }

    /// 복합 자식 회전은 **다른 질의와 같은 정규화**를 거쳐야 한다 — 검증이 허용하는
    /// 비단위 자세에서 안팎 판정이 갈리면 안 된다.
    #[test]
    fn compound_child_rotation_is_normalized_like_other_queries() {
        let k = (1.0f32 + 9.0e-4).sqrt();
        let q = Quat::from_axis_angle(v(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_4);
        let scaled = Quat {
            x: q.x * k,
            y: q.y * k,
            z: q.z * k,
            w: q.w * k,
        };
        let s = Shape::Compound {
            children: vec![(
                Pose {
                    pos: Vec3::ZERO,
                    rot: scaled,
                },
                Shape::Box {
                    half: v(900.0, 1.0, 1.0),
                    radius: 0.0,
                },
            )],
        };
        assert!(s.validate().is_ok(), "합법 셰이프라는 전제가 깨졌다");
        let mut rng = Lcg(2024);
        for _ in 0..400 {
            // 상자 표면 근처를 훑는다 — 정규화 차이가 드러나는 자리다.
            let local = v(
                (rng.next() - 0.5) * 1900.0,
                (rng.next() - 0.5) * 2.4,
                (rng.next() - 0.5) * 2.4,
            );
            let p = q.rotate(local);
            let a = try_project_point(&s, &ident(), p).unwrap();
            let b = try_contains_point(&s, &ident(), p).unwrap();
            assert_eq!(a.inside, b, "안팎이 갈렸다 local={local:?}");
        }
    }

    /// 확장 검색은 **누적 예산**을 넘으면 답을 지어내지 않고 자원 오류를 낸다.
    #[test]
    fn oversized_mesh_query_reports_resource_limit() {
        // 예산(4096)을 넘는 삼각형을 한 반경 안에 두고, 멀리서 질의해 전부 후보가 되게 한다.
        let (nx, nz) = (64u32, 64u32);
        let mut verts = Vec::new();
        for iz in 0..nz {
            for ix in 0..nx {
                verts.push(v(ix as f32, 0.0, iz as f32));
            }
        }
        let mut tris = Vec::new();
        for iz in 0..(nz - 1) {
            for ix in 0..(nx - 1) {
                let a = iz * nx + ix;
                tris.push([a, a + 1, a + nx]);
                tris.push([a + 1, a + nx + 1, a + nx]);
            }
        }
        assert!(
            tris.len() > super::PROJECT_TRI_BUDGET,
            "전제: 예산 초과 규모"
        );
        let mesh = Shape::trimesh(verts, tris).expect("메시");
        // 격자 한참 위 — 첫 반경이 이미 전체를 덮는다.
        let r = try_project_point(&mesh, &ident(), v(32.0, 5000.0, 32.0));
        assert_eq!(r, Err(QueryReject::ResourceLimit));
    }

    #[test]
    fn invalid_input_is_an_error_not_a_far_point() {
        let s = Shape::Sphere { radius: 1.0 };
        assert_eq!(
            try_project_point(&s, &ident(), v(f32::NAN, 0.0, 0.0)),
            Err(QueryReject::InvalidInput)
        );
        let bad_pose = Pose {
            pos: Vec3::ZERO,
            rot: Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
        };
        assert_eq!(
            try_project_point(&s, &bad_pose, Vec3::ZERO),
            Err(QueryReject::InvalidInput)
        );
        // 빈 hull은 GJK를 죽인다 — 질의 경계에서 막혀야 한다.
        let empty = Shape::Hull {
            points: vec![],
            radius: 0.0,
        };
        assert_eq!(
            try_project_point(&empty, &ident(), Vec3::ZERO),
            Err(QueryReject::InvalidInput)
        );
    }
}
