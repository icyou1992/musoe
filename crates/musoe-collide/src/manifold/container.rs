//! **비볼록 컨테이너 셰이프**의 접촉 — 트라이메시·하이트필드.
//!
//! 이 둘은 바디 하나가 내부에 삼각형 격자를 들고 접촉 순간에만 펼친다. 그래서 디스패처가
//! 곧장 볼록 알고리즘으로 내려보내면 안 되고(`support_core`가 `unreachable!`이다),
//! 여기서 후보 삼각형을 골라 볼록 대 삼각형으로 나눠 부른다.

use super::*;

/// 메시 접촉 — BVH로 겹치는 삼각형만 펼친다 (`heightfield_contacts`의 동형).
///
/// 격자와 다른 점은 후보를 얻는 방법뿐이다: 셀 범위 순회 대신 **내부 BVH 질의**.
/// 나머지 계약은 그대로다 —
/// - `flip`이면 A/B 역할이 반대라 매니폴드를 되돌린다.
/// - 예산은 **호출자 소유**다(질의 단위). 페어마다 새로 주면 여러 바디를 도는 질의의
///   총 작업량이 유계가 아니다.
/// - 상한(`MAX_HEIGHTFIELD_MANIFOLDS`)에서 끊고 그 사실을 반환값으로 낸다(조용한 누락 금지).
/// - **feature ID는 전부 generic으로 덮는다.** 클리핑이 삼각형마다 주는 ID는 삼각형이 다르면
///   같은 값이 나올 수 있어(같은 페어 안에서 충돌) warm start가 남의 임펄스를 물려받는다.
///   generic은 위치 근접으로 매칭되므로 삼각형이 달라도 옳게 갈린다.
///
/// **BVH 순회가 포화되면 예산 초과로 올린다.** 후보를 조용히 빠뜨리면 물체가 메시를
/// 통과하는데, 그건 "접촉 없음"과 구분되지 않아 디버깅이 불가능해진다.
#[allow(clippy::too_many_arguments)]
pub(super) fn trimesh_contacts(
    sa: &Shape,
    pa: &Pose,
    mesh_shape: &Shape,
    mesh_pose: &Pose,
    margin: f32,
    flip: bool,
    budget: &mut usize,
    f: &mut dyn FnMut(u8, u8, Manifold),
) -> ContactWalk {
    let mesh = mesh_shape.trimesh_ref().expect("TriMesh 변종");
    // 볼록 쪽 월드 AABB를 메시 로컬로 옮긴다(margin만큼 넓힌다).
    let mut world = crate::transformed_aabb(sa.local_aabb(), pa);
    let pad = Vec3::new(margin, margin, margin);
    world = crate::Aabb::new(world.min - pad, world.max + pad);
    let local = crate::heightfield::world_aabb_to_local(world, mesh_pose);
    let mut emitted = 0usize;
    let mut saturated = false;
    let mut over_budget = false;
    // 복합 자식도 함께 펼친다 — 자식 키는 그대로 콜백에 넘겨 네임스페이싱을 유지한다.
    let children: &[(Pose, Shape)] = match sa {
        Shape::Compound { children } => children,
        _ => &[],
    };
    let complete = mesh.for_each_candidate(local, |ti| {
        if *budget == 0 {
            over_budget = true;
            return false;
        }
        *budget -= 1;
        let tri_local = mesh.triangle(ti);
        let tri_shape = Shape::Triangle {
            points: tri_local,
            radius: 0.0,
        };
        let mask = mesh.weld_mask(ti);
        let mut emit = |ci: u8, cs: &Shape, cp: &Pose| -> bool {
            let Some(mut m) = contact(cs, cp, &tri_shape, mesh_pose, margin) else {
                return false;
            };
            if m.is_empty() {
                return false;
            }
            // 내부 에지 억제 — 월드 좌표 삼각형이 필요하다. 밴드 폭 = 허용오차 + 침투 깊이
            // (근거는 하이트필드 쪽 같은 자리 주석: 깊이 파고들수록 최근접점이 에지에서 멀어진다).
            if mask != 0 {
                let tw = [
                    mesh_pose.pos + mesh_pose.rot.rotate(tri_local[0]),
                    mesh_pose.pos + mesh_pose.rot.rotate(tri_local[1]),
                    mesh_pose.pos + mesh_pose.rot.rotate(tri_local[2]),
                ];
                let pen = m.points().iter().fold(0.0f32, |a, p| a.max(-p.separation));
                crate::weld_internal_edges(&mut m, tw, margin + pen.max(0.0), mask);
            }
            for i in 0..m.count as usize {
                m.points[i].feature = FeatureId(FEATURE_GENERIC);
            }
            if flip {
                m.flip();
                f(0, ci, m);
            } else {
                f(ci, 0, m);
            }
            true
        };
        let hit = if children.is_empty() {
            emit(0, sa, pa)
        } else {
            let mut any = false;
            for (i, (cp, cs)) in children.iter().enumerate() {
                let wp = Pose {
                    pos: pa.pos + pa.rot.rotate(cp.pos),
                    rot: pa.rot * cp.rot.normalize(),
                };
                if emit(i as u8, cs, &wp) {
                    any = true;
                    emitted += 1;
                    if emitted >= crate::MAX_HEIGHTFIELD_MANIFOLDS {
                        saturated = true;
                        return false;
                    }
                }
            }
            any
        };
        if children.is_empty() && hit {
            emitted += 1;
            if emitted >= crate::MAX_HEIGHTFIELD_MANIFOLDS {
                saturated = true;
                return false;
            }
        }
        true
    });
    if !complete {
        // BVH 순회 스택 포화 — 후보를 다 못 봤다. 예산 초과와 같은 계열로 올린다.
        return ContactWalk::BudgetExceeded;
    }
    if over_budget {
        return ContactWalk::BudgetExceeded;
    }
    if saturated {
        return ContactWalk::Saturated;
    }
    ContactWalk::Complete
}

/// 하이트필드 접촉 — 겹치는 셀만 삼각형으로 펼친다.
///
/// `flip`이면 A/B 역할이 원래와 반대라 매니폴드 법선·접점 소유를 되돌린다.
/// 반환 = 상한(`MAX_HEIGHTFIELD_MANIFOLDS`)에서 끊었는가.
///
/// **feature ID는 전부 generic으로 덮는다.** 클리핑이 삼각형마다 부여하는 ID는 삼각형이
/// 다르면 같은 값이 나올 수 있어(같은 페어 안에서 충돌) warm start가 남의 임펄스를
/// 물려받는다. generic은 위치 근접으로 매칭되므로 삼각형이 달라도 옳게 갈린다 —
/// 계보 대신 위치를 쓰는 대가는 슬라이딩 시 계승률 하락뿐이다.
#[allow(clippy::too_many_arguments)]
pub(super) fn heightfield_contacts(
    sa: &Shape,
    pa: &Pose,
    hf_shape: &Shape,
    hf_pose: &Pose,
    margin: f32,
    flip: bool,
    budget: &mut usize,
    f: &mut dyn FnMut(u8, u8, Manifold),
) -> ContactWalk {
    let hf = hf_shape.heightfield_ref().expect("Heightfield 변종");
    // 볼록 쪽 월드 AABB를 격자 로컬로 옮겨 셀 범위를 얻는다(margin만큼 넓힌다).
    let mut world = crate::transformed_aabb(sa.local_aabb(), pa);
    #[allow(unused_assignments)]
    let pad = Vec3::new(margin, margin, margin);
    world = crate::Aabb::new(world.min - pad, world.max + pad);
    let local = crate::heightfield::world_aabb_to_local(world, hf_pose);
    let Some(range) = hf.cell_range(local) else {
        return ContactWalk::Complete;
    };
    let (y_lo, y_hi) = (local.min.y, local.max.y);
    let mut emitted = 0usize;
    let mut visited = 0usize;
    let mut saturated = false;
    let mut over_budget = false;
    // 복합 자식도 함께 펼친다 — 자식 키는 그대로 콜백에 넘겨 네임스페이싱을 유지한다.
    let children: &[(Pose, Shape)] = match sa {
        Shape::Compound { children } => children,
        _ => &[],
    };
    // **타일 → 셀** 순서로 걷는다. 겹치지 않는 타일의 셀은 방문조차 하지 않고,
    // 방문 수가 예산을 넘으면 순회를 포기해 호출자가 자원 오류로 처리하게 한다.
    crate::heightfield::for_each_overlapping_cell(&hf, range, y_lo, y_hi, |cx, cz| {
        // 예산은 **호출자 소유**다(질의 단위). 페어마다 새로 4096이면 여러 바디를 도는
        // 질의의 총 작업량이 유계가 아니다.
        if *budget == 0 {
            over_budget = true;
            return false;
        }
        *budget -= 1;
        visited += 1;
        if visited > crate::MAX_HEIGHTFIELD_CELLS_PER_QUERY {
            over_budget = true;
            return false;
        }
        {
            for t in 0..2u8 {
                let tri_local = hf.triangle(cx, cz, t);
                let tri_shape = Shape::Triangle {
                    points: tri_local,
                    radius: 0.0,
                };
                let mask = hf.weld_mask(cx, cz, t);
                let mut emit = |ci: u8, cs: &Shape, cp: &Pose| -> bool {
                    let Some(mut m) = contact(cs, cp, &tri_shape, hf_pose, margin) else {
                        return false;
                    };
                    if m.is_empty() {
                        return false;
                    }
                    // 내부 에지 억제 — 월드 좌표 삼각형이 필요하다.
                    if mask != 0 {
                        let tw = [
                            hf_pose.pos + hf_pose.rot.rotate(tri_local[0]),
                            hf_pose.pos + hf_pose.rot.rotate(tri_local[1]),
                            hf_pose.pos + hf_pose.rot.rotate(tri_local[2]),
                        ];
                        // 밴드 폭 = **접촉 허용오차 + 침투 깊이**.
                        //
                        // 하이트필드 삼각형은 두께가 0이라 radius를 밴드로 쓸 수 없다(지형
                        // 등록부는 삼각형 radius가 그 역할을 했다). 그리고 접점은 에지에서
                        // 정확히 0만큼 떨어져 있지 않다 — 깊이 파고들수록 최근접점이 에지에서
                        // 멀어진다(실측: 18cm 침투에서 5cm). 침투를 밴드에 넣지 않으면
                        // **정작 유령 에지가 세게 차는 깊은 침투에서만 용접이 꺼진다**.
                        let pen = m.points().iter().fold(0.0f32, |a, p| a.max(-p.separation));
                        crate::weld_internal_edges(&mut m, tw, margin + pen.max(0.0), mask);
                    }
                    for i in 0..m.count as usize {
                        m.points[i].feature = FeatureId(FEATURE_GENERIC);
                    }
                    if flip {
                        m.flip();
                    }
                    if flip {
                        f(0, ci, m);
                    } else {
                        f(ci, 0, m);
                    }
                    true
                };
                let hit = if children.is_empty() {
                    emit(0, sa, pa)
                } else {
                    let mut any = false;
                    for (i, (cp, cs)) in children.iter().enumerate() {
                        let wp = Pose {
                            pos: pa.pos + pa.rot.rotate(cp.pos),
                            // 자식 회전 정규화 — 근거는 `cast.rs`의 compose 주석에.
                            rot: pa.rot * cp.rot.normalize(),
                        };
                        if emit(i as u8, cs, &wp) {
                            any = true;
                            emitted += 1;
                            if emitted >= crate::MAX_HEIGHTFIELD_MANIFOLDS {
                                saturated = true;
                                return false;
                            }
                        }
                    }
                    any
                };
                if children.is_empty() && hit {
                    emitted += 1;
                    if emitted >= crate::MAX_HEIGHTFIELD_MANIFOLDS {
                        saturated = true;
                        return false;
                    }
                }
            }
        }
        true
    });
    if over_budget {
        return ContactWalk::BudgetExceeded;
    }
    if saturated {
        return ContactWalk::Saturated;
    }
    ContactWalk::Complete
}
