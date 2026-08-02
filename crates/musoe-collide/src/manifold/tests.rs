//! `manifold` 테스트.

use super::sat::{box_box_axis, sphere_box, SatResult};
use super::*;

#[cfg(test)]
mod trimesh_dispatch_tests {
    use super::*;
    use crate::{FEATURE_GENERIC, MAX_INPUT};
    use musoe_math::Quat;

    fn pose(x: f32, y: f32, z: f32) -> Pose {
        Pose {
            pos: Vec3::new(x, y, z),
            rot: Quat::IDENTITY,
        }
    }

    /// 평평한 2×2 사각형 메시 (대각을 공유하는 삼각형 2개), y=0.
    fn quad_mesh() -> Shape {
        Shape::trimesh(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 2.0),
                Vec3::new(2.0, 0.0, 2.0),
            ],
            vec![[0, 1, 2], [1, 3, 2]],
        )
        .expect("정상 메시")
    }

    fn manifolds(sa: &Shape, pa: &Pose, mesh: &Shape, pm: &Pose) -> Vec<Manifold> {
        let mut out = Vec::new();
        for_each_contact(sa, pa, mesh, pm, 0.02, &mut |_, _, m| out.push(m));
        out
    }

    /// 평평한 메시 위의 접촉은 **면 법선**을 낸다(기울지 않는다).
    ///
    /// **정직한 한계**: 이 테스트는 용접(`mask != 0` 분기)을 덮지 못한다. 뮤테이션으로
    /// 확인했다 — 용접을 통째로 지워도 통과한다. 평평하고 동일 평면인 메시에서는 두 삼각형
    /// 모두 이미 면 법선을 내므로 용접이 바꿀 것이 없기 때문이다. 용접이 실제로 값을 하는
    /// 기하(에지를 옆에서 때리는 배치)는 여기서 만들지 못했다. 용접 **로직 자체**는
    /// `weld_tests`가 덮고 있고, 덮이지 않은 것은 이 경로의 **배선**이다.
    #[test]
    fn flat_mesh_contacts_use_the_face_normal() {
        let mesh = quad_mesh();
        let ball = Shape::Sphere { radius: 0.5 };
        // 대각선 (2,0,0)–(0,0,2) 위의 점 (1,·,1)에 중심을 두고 살짝 띄운다.
        let ms = manifolds(&ball, &pose(1.0, 0.48, 1.0), &mesh, &pose(0.0, 0.0, 0.0));
        assert!(!ms.is_empty(), "접촉이 아예 없다 — 씬이 틀렸다");
        for (i, m) in ms.iter().enumerate() {
            let ny = m.normal.y.abs();
            assert!(
                ny > 0.99,
                "매니폴드 {i}의 법선이 면에서 기울었다: {:?} (|ny|={ny}) — 유령 에지",
                m.normal
            );
        }
    }

    /// **접점 feature는 전부 generic이다** — warm start가 삼각형을 건너 섞이지 않는 근거.
    ///
    /// **정직한 한계**: 이 불변식은 참이지만, 지금 경로에서는 `feature = FEATURE_GENERIC`
    /// 덮어쓰기를 지워도 유지된다(뮤테이션 확인). 삼각형 접촉이 클리핑이 아니라 witness
    /// 경로를 타서 **이미** generic으로 나오기 때문이다. 덮어쓰기는 하이트필드와 같은
    /// 방어선이고(그쪽도 같은 줄이 있다), 나중에 삼각형 전용 해석 경로가 생기면 그때 값을 한다.
    /// 이 테스트가 잠그는 것은 **불변식**이지 그 줄이 아니다.
    #[test]
    fn mesh_contacts_are_tagged_generic() {
        let mesh = quad_mesh();
        let box_shape = Shape::Box {
            half: Vec3::new(0.6, 0.3, 0.6),
            radius: 0.0,
        };
        // 대각을 걸치도록 놓아 삼각형 둘 다에서 매니폴드가 나오게 한다.
        let ms = manifolds(
            &box_shape,
            &pose(1.0, 0.28, 1.0),
            &mesh,
            &pose(0.0, 0.0, 0.0),
        );
        assert!(!ms.is_empty(), "접촉이 아예 없다 — 씬이 틀렸다");
        let mut points = 0usize;
        for m in &ms {
            for p in m.points() {
                assert_eq!(
                    p.feature.0, FEATURE_GENERIC,
                    "메시 접점이 generic이 아니다: {:#x}",
                    p.feature.0
                );
                points += 1;
            }
        }
        assert!(points > 0, "접점이 하나도 없다 — 단언이 아무것도 안 봤다");
    }

    /// 메시가 A쪽이어도(질의 경로) 같은 답이 나오고 법선이 뒤집힌다.
    #[test]
    fn mesh_on_either_side_agrees() {
        let mesh = quad_mesh();
        let ball = Shape::Sphere { radius: 0.5 };
        let (pb, pm) = (pose(1.0, 0.45, 1.0), pose(0.0, 0.0, 0.0));
        let ab = manifolds(&ball, &pb, &mesh, &pm);
        let mut ba = Vec::new();
        for_each_contact(&mesh, &pm, &ball, &pb, 0.02, &mut |_, _, m| ba.push(m));
        assert_eq!(ab.len(), ba.len(), "역할을 바꾸니 매니폴드 수가 달라졌다");
        for (x, y) in ab.iter().zip(&ba) {
            let d = (x.normal + y.normal).length();
            assert!(
                d < 1.0e-3,
                "법선이 반대가 아니다: {:?} vs {:?}",
                x.normal,
                y.normal
            );
        }
    }

    /// **메시를 볼록 인자 자리에 넣어도 trap하지 않는다** (교차검수 rev_953115e8 BLOCKER).
    ///
    /// `contact`의 비볼록 가드가 TriMesh를 빠뜨리면 GJK로 내려가 `support_core`의
    /// `unreachable!`에서 **프로세스가 죽는다**. 공개 API라 호스트가 한 줄로 재현할 수 있고,
    /// wasm에서는 trap이다. 메시-메시 질의도 같은 경로다.
    #[test]
    fn mesh_in_a_convex_slot_does_not_trap() {
        let mesh = quad_mesh();
        let ball = Shape::Sphere { radius: 0.5 };
        // 겹치는 배치 — 가드가 없으면 GJK로 내려간다.
        let _ = manifolds(&mesh, &pose(1.0, 0.2, 1.0), &ball, &pose(1.0, 0.45, 1.0));
        // 메시 대 메시 — 양쪽 모두 비볼록이다.
        let _ = manifolds(&mesh, &pose(0.0, 0.0, 0.0), &mesh, &pose(0.5, 0.01, 0.5));
        // 여기 도달하면 trap이 없었다는 뜻이다.
    }

    /// 크기 정책 밖 메시는 만들어지지 않는다(생성자 경계) — 질의 경로가 볼 일이 없다.
    #[test]
    fn oversized_mesh_is_rejected_at_construction() {
        assert!(Shape::trimesh(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(MAX_INPUT * 4.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2]],
        )
        .is_none());
    }
}

#[cfg(test)]
mod weld_tests {
    use super::*;
    use musoe_math::Quat;

    // 바닥 평면의 삼각형: (−1,0,−1)·(1,0,−1)·(0,0,1), 에지 0 = 앞쪽 (−1..1, z=−1).
    fn tri() -> [Vec3; 3] {
        [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
    }

    fn edge_manifold(tilt: Vec3) -> Manifold {
        let mut m = Manifold {
            normal: (Vec3::new(0.0, 1.0, 0.0) + tilt).normalize_or_zero(),
            ..Default::default()
        };
        m.push(ContactPoint {
            position: Vec3::new(0.0, 0.0, -1.0), // 에지 0 위
            separation: -0.01,
            feature: FeatureId(3),
        });
        m
    }

    /// 내부 에지 근방의 기운 법선은 면 법선으로 스냅된다.
    #[test]
    fn tilted_normal_on_internal_edge_snaps_to_face() {
        let mut m = edge_manifold(Vec3::new(0.0, 0.0, -0.6)); // 에지 0 밖으로 기움
        weld_internal_edges(&mut m, tri(), 0.05, 0b001);
        assert!((m.normal - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
    }

    /// 아래에서 온 접촉은 아래쪽 면 법선으로 — 부호는 현재 법선 쪽으로 정렬된다.
    #[test]
    fn snap_keeps_the_side_the_contact_came_from() {
        let mut m = edge_manifold(Vec3::new(0.0, 0.0, -0.6));
        m.normal = Vec3::new(0.0, -1.0, 0.3).normalize_or_zero();
        weld_internal_edges(&mut m, tri(), 0.05, 0b001);
        assert!((m.normal - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-6);
    }

    /// 마스크가 0이거나(용접 없음) 접점이 외부 에지에 있으면 무변경.
    #[test]
    fn external_edges_and_unmasked_triangles_are_untouched() {
        let before = edge_manifold(Vec3::new(0.0, 0.0, -0.6));
        let mut m = before;
        weld_internal_edges(&mut m, tri(), 0.05, 0);
        assert_eq!(m.normal, before.normal);
        // 에지 1·2만 내부라고 표시 — 접점은 에지 0 위라 밴드 밖.
        let mut m = before;
        weld_internal_edges(&mut m, tri(), 0.05, 0b110);
        assert_eq!(m.normal, before.normal);
    }

    /// 면 접촉(법선이 이미 원뿔 안)은 내부 에지 위 접점이라도 무변경 — 보정은 방향만.
    #[test]
    fn face_contacts_are_left_alone() {
        let mut m = edge_manifold(Vec3::ZERO);
        let before = m.normal;
        weld_internal_edges(&mut m, tri(), 0.05, 0b111);
        assert_eq!(m.normal, before);
    }

    /// 실제 협폭 경로로도 성립: 구가 내부 에지의 둥근 밴드에 접촉 → 기운 법선 → 용접.
    #[test]
    fn real_sphere_edge_contact_gets_welded() {
        let hull = Shape::Hull {
            data: crate::HullData::new(tri().to_vec()),
            radius: 0.05,
        };
        let sphere = Shape::Sphere { radius: 0.3 };
        let id = Pose {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        };
        // 에지 0(z=−1) 바로 바깥 위 — 에지 밴드 접촉이라 법선이 −z로 기운다.
        let pa = Pose {
            pos: Vec3::new(0.0, 0.32, -1.12),
            rot: Quat::IDENTITY,
        };
        let m0 = contact(&sphere, &pa, &hull, &id, 0.1).expect("접촉이 있어야 한다");
        assert!(
            m0.normal.z < -0.2,
            "전제: 에지 접촉 법선이 기울어야 한다 (n={:?})",
            m0.normal
        );
        let mut m = m0;
        weld_internal_edges(&mut m, tri(), 0.05, 0b001);
        assert!(
            (m.normal - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6,
            "용접 후 면 법선이어야 한다 (n={:?})",
            m.normal
        );
    }
}

#[cfg(test)]
mod heightfield_contact_tests {
    use super::*;
    use musoe_math::Quat;

    fn pose(x: f32, y: f32, z: f32) -> Pose {
        Pose {
            pos: Vec3::new(x, y, z),
            rot: Quat::IDENTITY,
        }
    }

    /// **내부 에지 억제의 직접 증거**: 셀 경계 바로 위에 놓인 구의 접촉 법선이
    /// 에지 법선이 아니라 **면 법선**이어야 한다.
    ///
    /// 완만한 경사(≈4°, 용접 원뿔 안)를 주면 에지에서의 최근접 방향이 두 면 법선 어느
    /// 쪽과도 다르다 — 용접이 없으면 구가 이음매를 지날 때마다 그 방향으로 차인다.
    /// 용접을 끄면(mask=0) 이 테스트가 실패하는 것으로 기제를 잠근다.
    #[test]
    fn welding_snaps_seam_contact_normal_to_the_face() {
        // z 방향으로 셀마다 7cm 오르는 격자(1m 셀 → ≈4°)
        let (nx, nz) = (4u32, 4u32);
        let mut heights = vec![0.0f32; (nx * nz) as usize];
        for iz in 0..nz {
            for ix in 0..nx {
                heights[(iz * nx + ix) as usize] = iz as f32 * 0.07;
            }
        }
        let hf_shape = Shape::heightfield(heights, nx, nz, 1.0).expect("격자");
        let hf = hf_shape.heightfield_ref().unwrap();
        // 이음매(z=2)를 **넘어선 쪽**에 중심을 두고 살짝 파고들게 놓는다.
        // 이러면 뒤 셀(1,1) 삼각형에서 최근접 특징이 **공유 에지**가 되고, 에지에서
        // 중심으로 향하는 법선은 면 법선과 크게 어긋난다 — 용접이 잡아야 하는 바로 그 경우다.
        // (구를 에지 바로 위에 얹으면 에지 법선이 이미 면 법선의 5° 원뿔 안이라 아무 일도
        //  일어나지 않는다 — 그건 용접이 필요 없는 상황이다.)
        let surf = hf.height_at(1.5, 2.15).unwrap();
        let sphere = Shape::Sphere { radius: 0.3 };
        let sp = pose(1.5, surf + 0.12, 2.15);

        let mut normals = Vec::new();
        for_each_contact(
            &sphere,
            &sp,
            &hf_shape,
            &pose(0.0, 0.0, 0.0),
            0.02,
            &mut |_, _, m| {
                if !m.is_empty() {
                    normals.push(m.normal);
                }
            },
        );
        assert!(
            !normals.is_empty(),
            "이음매에서 접촉이 없다 — 테스트가 공회전"
        );
        // 면 법선(두 이웃 삼각형 중 하나) 과 거의 같아야 한다.
        let faces: Vec<Vec3> = [(1u32, 1u32, 1u8), (1, 2, 0)]
            .iter()
            .map(|&(cx, cz, t)| hf.triangle_normal(cx, cz, t).unwrap())
            .collect();
        for n in &normals {
            let best = faces.iter().map(|f| f.dot(*n).abs()).fold(0.0f32, f32::max);
            assert!(
                best > 0.999,
                "이음매 법선이 면 법선과 다르다(용접 실패): n={n:?} faces={faces:?}"
            );
        }
    }
}

// 볼록 대 볼록(박스·구·캡슐) 매니폴드 — SAT·클리퍼·4점 축약의 통합 검증.
#[cfg(test)]
mod convex_manifold_tests {
    use super::*;

    fn pose(x: f32, y: f32, z: f32) -> Pose {
        Pose {
            pos: Vec3::new(x, y, z),
            rot: Quat::IDENTITY,
        }
    }

    /// M1 완료 기준: box 면 접촉이 4점 매니폴드를 낸다.
    #[test]
    fn box_on_box_face_gives_4_points() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        // top이 ground 위에 딱 얹힘 (y: 1+1=2)
        let m = contact(
            &top,
            &pose(0.0, 2.0, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .expect("contact expected");
        assert_eq!(m.points().len(), 4, "{:?}", m.points);
        for p in m.points() {
            assert!(p.separation.abs() < 1e-4, "{:?}", p);
        }
        assert!(
            (m.normal.y - 1.0).abs() < 1e-4,
            "top을 +y로 밀면 분리: {:?}",
            m.normal
        );
        // 4점의 feature는 서로 달라야 함
        let mut feats: Vec<_> = m.points().iter().map(|p| p.feature).collect();
        feats.sort_unstable();
        feats.dedup();
        assert_eq!(feats.len(), 4);
    }

    /// M1 완료 기준: feature ID 안정성 — 미세 섭동에도 같은 feature 집합.
    #[test]
    fn feature_ids_stable_under_perturbation() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let base = contact(
            &top,
            &pose(0.0, 1.999, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        let perturbed = contact(
            &top,
            &pose(1.0e-4, 1.999, -2.0e-4),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        let f0: Vec<_> = base.points.iter().map(|p| p.feature).collect();
        let f1: Vec<_> = perturbed.points.iter().map(|p| p.feature).collect();
        assert_eq!(f0, f1, "feature 집합이 섭동에 흔들림");
    }

    /// 회전 슬라이드 케이스: yaw 회전한 박스가 얹혀도 ≤4점 + 전부 접촉권.
    #[test]
    fn rotated_box_on_ground_clipped() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.6);
        let pa = Pose {
            pos: Vec3::new(0.0, 1.995, 0.0),
            rot,
        };
        let m = contact(&top, &pa, &ground, &pose(0.0, 0.0, 0.0), 0.05).unwrap();
        assert!(
            !m.points.is_empty() && m.points().len() <= 4,
            "{:?}",
            m.points().len()
        );
        for p in m.points() {
            assert!(p.separation < 0.05);
        }
    }

    /// 침투 케이스: EPA 경로에서도 클리핑 4점 + 음의 separation.
    #[test]
    fn penetrating_face_contact() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let m = contact(
            &top,
            &pose(0.0, 1.9, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        assert_eq!(m.points().len(), 4);
        for p in m.points() {
            assert!((p.separation + 0.1).abs() < 1e-3, "sep={}", p.separation);
        }
    }

    /// 확인 라운드 MAJOR 회귀: 경계 정점이 클리핑 평면 위에 정확히 놓여도 중복 접점 금지.
    #[test]
    fn boundary_vertex_no_duplicate_points() {
        // incident 꼭짓점이 reference 측면 평면(x=1) 위에 정확히 놓이는 45° 구성
        let ground = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let top = Shape::Box {
            half: Vec3::new(
                core::f32::consts::FRAC_1_SQRT_2,
                1.0,
                core::f32::consts::FRAC_1_SQRT_2,
            ),
            radius: 0.0,
        };
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_4);
        let pa = Pose {
            pos: Vec3::new(1.0, 2.0, 0.0),
            rot,
        };
        let m = contact(&top, &pa, &ground, &pose(0.0, 0.0, 0.0), 0.05).unwrap();
        assert!(!m.points.is_empty() && m.points().len() <= 4);
        for (i, p) in m.points().iter().enumerate() {
            for q in m.points().iter().skip(i + 1) {
                let d = (p.position - q.position).length_sq();
                assert!(d > 1.0e-8, "중복 접점: {p:?} vs {q:?}");
            }
        }
    }

    /// 고종횡비 형상의 **국소 접촉 패치**가 중복 판정에 삼켜지지 않는지.
    ///
    /// 교차검수 rev_7a05abc1 BLOCKER: 중복 임계를 `characteristic_length`(작은 쪽 AABB 대각)로
    /// 잡으면 얇고 큰 판에서 임계가 접촉 패치보다 수천 배 커진다. 아래 판 두 장의 겹침은
    /// 약 1mm×1mm인데 대각은 2.8km라 임계가 2.8mm — 네 모서리가 전부 "같은 점"이 되어
    /// 면 접촉이 점 접촉으로 붕괴하고 회전 저항이 사라진다.
    #[test]
    fn thin_large_boxes_keep_distinct_contact_patch_points() {
        let plate = Shape::Box {
            half: Vec3::new(1000.0, 0.001, 1000.0),
            radius: 0.0,
        };
        let m = contact(
            &plate,
            &pose(0.0, 0.0, 0.0),
            &plate,
            &pose(1999.999, 0.0019, 1999.999),
            0.001,
        )
        .expect("1mm×1mm 면 접촉이 존재한다");
        assert_eq!(
            m.points().len(),
            4,
            "중복 임계가 얇은 접촉 패치를 합쳤다: {:?}",
            m.points()
        );
    }

    /// 리뷰 요구: **metamorphic 스케일 테스트**. 같은 씬을 1e-3~1e2배로 변환해도
    /// 접촉 판정·법선·상대 침투가 유지되어야 한다(절대 임계가 남아 있으면 여기서 깨진다).
    #[test]
    fn sat_normal_agrees_with_gjk_epa() {
        // 정지 스택에서 실제로 나타나는 배치(축정렬 + 미세 회전)를 결정적으로 흔들며
        // SAT 법선과 GJK/EPA 법선의 각도차를 잰다. 두 경로가 다른 축을 고르면
        // 참조면이 바뀌어 feature ID가 요동치고 warm start가 끊긴다.
        let half = Vec3::new(0.5, 0.5, 0.5);
        let mut rng: u32 = 0x1234_5678;
        let mut next = || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5 // [-0.5, 0.5)
        };
        let (mut worst_deg, mut disagree, mut total) = (0.0f32, 0, 0);
        for case in 0..600 {
            // 수직 적층 / 수평 인접 / 대각 이웃을 번갈아
            let base = match case % 3 {
                0 => Vec3::new(0.0, 1.0, 0.0),
                1 => Vec3::new(1.02, 0.0, 0.0),
                _ => Vec3::new(0.51, 1.0, 0.0),
            };
            let jitter = Vec3::new(next(), next(), next()) * 0.02;
            let mut tilt = |a: f32, b: f32, c: f32| {
                Quat::from_axis_angle(Vec3::new(a, b, c).normalize_or_zero(), 0.01 * next())
            };
            let pa = Pose {
                pos: Vec3::ZERO,
                rot: tilt(1.0, 0.0, 0.0),
            };
            let pb = Pose {
                pos: base + jitter,
                rot: tilt(0.0, 0.0, 1.0),
            };
            let (ma, mb) = (Mat3::from_quat(pa.rot), Mat3::from_quat(pb.rot));
            let sat = match box_box_axis(half, 0.0, &pa, &ma, half, 0.0, &pb, &mb, 0.05) {
                SatResult::Axis(n) => n,
                _ => continue,
            };
            let sa = Shape::Box { half, radius: 0.0 };
            let reference = match gjk(&sa, &pa, &sa, &pb) {
                GjkResult::Separated { normal, .. } => normal,
                GjkResult::Intersecting(seed) => epa(seed, &sa, &pa, &sa, &pb).normal,
            };
            total += 1;
            let cos = sat.dot(reference).clamp(-1.0, 1.0);
            let deg = cos.acos().to_degrees();
            if deg > worst_deg {
                worst_deg = deg;
            }
            if deg > 1.0 {
                disagree += 1;
            }
        }
        println!("SAT vs GJK/EPA: 표본 {total}, 1° 초과 불일치 {disagree}, 최대 {worst_deg:.2}°");
        assert!(
            disagree == 0,
            "SAT 법선이 {disagree}/{total} 케이스에서 기존 경로와 다르다 (최대 {worst_deg:.2}°)"
        );
    }

    #[test]
    fn contact_is_scale_invariant() {
        for &k in &[1.0e-3f32, 1.0e-1, 1.0, 1.0e1, 1.0e2] {
            let top = Shape::Box {
                half: Vec3::new(k, k, k),
                radius: 0.0,
            };
            let ground = Shape::Box {
                half: Vec3::new(4.0 * k, k, 4.0 * k),
                radius: 0.0,
            };
            // 상대 침투 1% 로 동일하게 구성
            let m = contact(
                &top,
                &pose(0.0, 1.99 * k, 0.0),
                &ground,
                &pose(0.0, 0.0, 0.0),
                0.05 * k,
            )
            .unwrap_or_else(|| panic!("scale {k}: 접촉 소실"));
            assert_eq!(
                m.points().len(),
                4,
                "scale {k}: 매니폴드 점수 {}",
                m.points().len()
            );
            assert!(
                (m.normal.y - 1.0).abs() < 1.0e-3,
                "scale {k}: 법선 {:?}",
                m.normal
            );
            for p in m.points() {
                let rel = p.separation / k;
                assert!(
                    (rel + 0.01).abs() < 0.05,
                    "scale {k}: 상대 침투 {rel} (절대 {})",
                    p.separation
                );
            }
        }
    }

    /// 리뷰 MAJOR 회귀: rounded box의 접점은 평면 패치(접선 ±(half−radius)) 안에 있어야 한다.
    #[test]
    fn rounded_box_face_points_stay_on_flat_patch() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.25,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let m = contact(
            &top,
            &pose(0.0, 2.0, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        assert_eq!(m.points().len(), 4);
        for p in m.points() {
            assert!(p.position.x.abs() <= 0.7501, "point={p:?}");
            assert!(p.position.z.abs() <= 0.7501, "point={p:?}");
            assert!(p.separation.abs() < 1.0e-4, "point={p:?}");
        }
    }

    /// 구-박스: witness 단일점 + radius 재팽창 separation.
    #[test]
    fn sphere_box_single_point() {
        let ball = Shape::Sphere { radius: 0.5 };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let m = contact(
            &ball,
            &pose(0.0, 1.52, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.1,
        )
        .unwrap();
        assert_eq!(m.points().len(), 1);
        assert!(
            (m.points()[0].separation - 0.02).abs() < 1e-3,
            "{:?}",
            m.points()[0]
        );
        assert!((m.normal.y - 1.0).abs() < 1e-3);
    }

    /// speculative: margin 밖이면 None, 안이면 양의 separation 접촉.
    #[test]
    fn speculative_margin_gate() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        assert!(contact(&s, &pose(0.0, 2.3, 0.0), &s, &pose(0.0, 0.0, 0.0), 0.05).is_none());
        let m = contact(&s, &pose(0.0, 2.3, 0.0), &s, &pose(0.0, 0.0, 0.0), 0.5).unwrap();
        assert!(m.points().iter().all(|p| (p.separation - 0.3).abs() < 1e-3));
    }
}

#[cfg(test)]
mod sphere_box_tests {
    use super::*;

    /// 고정 시드 xorshift — 재현 가능한 무작위 배치.
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn sym(&mut self, s: f32) -> f32 {
            (self.f() - 0.5) * 2.0 * s
        }
        fn quat(&mut self) -> Quat {
            Quat::from_axis_angle(
                Vec3::new(self.sym(1.0), self.sym(1.0), self.sym(1.0)),
                self.sym(3.2),
            )
        }
    }

    /// 독립 계산과의 대조. GJK를 기준으로 쓰지 않는 이유는 실측 때문이다 — 중심이
    /// 박스 밖 0.05에 있는 배치에서 `gjk`가 `Separated { distance: 0 }`을 돌려줬다.
    /// 그래서 기준을 **회전축 내적으로 구한 점-OBB 거리**로 둔다. 해석 경로가 쓰는
    /// 쿼터니언 켤레 변환과 다른 경로라, 켤레 방향을 뒤집는 류의 오류를 잡는다.
    #[test]
    fn sphere_box_matches_independent_point_obb_distance() {
        let mut rng = Rng(0x5B07);
        let mut checked = 0;
        for _ in 0..12000 {
            let half = Vec3::new(
                0.2 + rng.f() * 2.0,
                0.2 + rng.f() * 2.0,
                0.2 + rng.f() * 2.0,
            );
            let rs = 0.1 + rng.f() * 1.5;
            let pb = Pose {
                pos: Vec3::new(rng.sym(2.0), rng.sym(2.0), rng.sym(2.0)),
                rot: rng.quat(),
            };
            let cs = Vec3::new(rng.sym(6.0), rng.sym(6.0), rng.sym(6.0));

            // 독립 기준: 박스의 월드 축 3개에 내적해 로컬 좌표를 얻고 클램프한다.
            let ax = pb.rot.rotate(Vec3::new(1.0, 0.0, 0.0));
            let ay = pb.rot.rotate(Vec3::new(0.0, 1.0, 0.0));
            let az = pb.rot.rotate(Vec3::new(0.0, 0.0, 1.0));
            let dv = cs - pb.pos;
            let l = Vec3::new(dv.dot(ax), dv.dot(ay), dv.dot(az));
            let cl = Vec3::new(
                l.x.clamp(-half.x, half.x),
                l.y.clamp(-half.y, half.y),
                l.z.clamp(-half.z, half.z),
            );
            let outside = (l - cl).length();
            let inside_depth = if outside > 0.0 {
                0.0
            } else {
                (half.x - l.x.abs())
                    .min(half.y - l.y.abs())
                    .min(half.z - l.z.abs())
            };
            let want_sep = if outside > 0.0 {
                outside - rs
            } else {
                -(inside_depth + rs)
            };
            if want_sep > 1.0 {
                continue; // margin 밖 — 매니폴드가 없는 게 정답이다
            }
            let m = sphere_box(rs, cs, half, 0.0, &pb, 1.0, true)
                .expect("margin 1.0 안이면 매니폴드가 나와야 한다");
            let got = m.points()[0].separation;
            let scale = 1.0 + half.x.max(half.y).max(half.z) + rs;
            assert!(
                (got - want_sep).abs() < 1e-4 * scale,
                "separation 불일치: 해석 {got} vs 독립 {want_sep}  half={half:?} rs={rs}"
            );
            assert!(
                (m.normal.length() - 1.0).abs() < 1e-4,
                "법선이 단위벡터가 아니다"
            );
            // 법선은 박스 → 구 방향이어야 한다 (A=구를 +normal로 밀면 분리)
            if outside > 1e-3 {
                let e = l - cl;
                let want_n = (ax * e.x + ay * e.y + az * e.z) * (1.0 / outside);
                assert!(
                    m.normal.dot(want_n) > 0.999,
                    "법선 방향 불일치: dot={}",
                    m.normal.dot(want_n)
                );
            }
            checked += 1;
        }
        assert!(checked > 1000, "표본이 너무 적다: {checked}");
    }

    /// **GJK 거리 과소평가 회귀 (2026-07-26 수정 완료).**
    ///
    /// 이 배치에서 `gjk`가 `Separated { distance: 0 }`을 돌려줬다 — 구 중심이 박스 코어
    /// 밖 0.0516에 있는데도. 원인은 둘이었다: ① 진전 판정 임계가 f32 상쇄 오차보다
    /// 엄격해 종료하지 못하고 같은 support를 반복 삽입 ② 그렇게 생긴 영부피 사면체의
    /// 복구 경로가 직전 최근접점을 버리고 거리 0을 **강제**(`Vec3::ZERO` 전달).
    /// 이제 gjk도 독립 점-OBB 계산과 일치해야 한다 — 회귀가 다시 열리면 여기서 잡힌다.
    #[test]
    fn gjk_core_distance_matches_independent_calculation() {
        let half = Vec3::new(0.4904285, 2.0158985, 1.071742);
        let rs = 1.4453597f32;
        let pb = Pose {
            pos: Vec3::new(1.4413323, -1.2419696, 1.250838),
            rot: Quat {
                x: -0.047053058,
                y: -0.12782943,
                z: 0.15615694,
                w: 0.9782948,
            },
        };
        let cs = Vec3::new(1.368325, 0.080718756, 2.0537288);

        // 독립 계산 (회전축 내적)
        let ax = pb.rot.rotate(Vec3::new(1.0, 0.0, 0.0));
        let ay = pb.rot.rotate(Vec3::new(0.0, 1.0, 0.0));
        let az = pb.rot.rotate(Vec3::new(0.0, 0.0, 1.0));
        let dv = cs - pb.pos;
        let l = Vec3::new(dv.dot(ax), dv.dot(ay), dv.dot(az));
        let cl = Vec3::new(
            l.x.clamp(-half.x, half.x),
            l.y.clamp(-half.y, half.y),
            l.z.clamp(-half.z, half.z),
        );
        let core_dist = (l - cl).length();
        assert!(
            core_dist > 0.04,
            "이 배치의 코어 거리는 0이 아니다: {core_dist}"
        );

        let m = sphere_box(rs, cs, half, 0.0, &pb, 1.0, true).unwrap();
        assert!(
            (m.points()[0].separation - (core_dist - rs)).abs() < 1e-4,
            "해석 경로는 독립 계산과 일치해야 한다"
        );

        // gjk는 여기서 0을 준다 — 고쳐지면 이 단언이 깨져 알려준다.
        let ss = Shape::Sphere { radius: rs };
        let sbx = Shape::Box { half, radius: 0.0 };
        let ps = Pose {
            pos: cs,
            rot: Quat::IDENTITY,
        };
        let GjkResult::Separated { distance, .. } = gjk(&ss, &ps, &sbx, &pb) else {
            panic!("코어가 분리돼 있으므로 Separated여야 한다");
        };
        assert!(
            (distance - core_dist).abs() < 1e-4,
            "gjk 코어 거리 {distance} 가 독립 계산 {core_dist} 와 다르다"
        );
    }

    /// 부호·크기 자기일관성: 구를 법선 반대로 |sep|만큼 밀면 접촉이 정확히 풀려야 한다.
    /// 관통(중심이 코어 안쪽인 경우 포함) 구간까지 덮는다.
    #[test]
    fn sphere_box_separation_resolves_overlap() {
        let mut rng = Rng(0xB07E5);
        let mut deep = 0;
        for _ in 0..4000 {
            let half = Vec3::new(
                0.3 + rng.f() * 1.5,
                0.3 + rng.f() * 1.5,
                0.3 + rng.f() * 1.5,
            );
            let rs = 0.1 + rng.f() * 1.0;
            let pb = Pose {
                pos: Vec3::new(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0)),
                rot: rng.quat(),
            };
            // 박스 근처에 구를 촘촘히 뿌려 관통·내부 케이스를 포함시킨다
            let cs = pb.pos + Vec3::new(rng.sym(2.0), rng.sym(2.0), rng.sym(2.0));
            let Some(m) = sphere_box(rs, cs, half, 0.0, &pb, 0.0, true) else {
                continue;
            };
            let sep = m.points()[0].separation;
            if sep >= 0.0 {
                continue;
            }
            deep += 1;
            // A(구)를 +normal로 |sep|만큼 밀면 접촉이 풀린다 = separation ≈ 0
            let moved = cs + m.normal * (-sep);
            let after = sphere_box(rs, moved, half, 0.0, &pb, 1.0, true)
                .expect("밀어낸 뒤에도 margin 1.0 안이다");
            let s2 = after.points()[0].separation;
            assert!(
                s2.abs() < 3e-3 * (1.0 + half.x.max(half.y).max(half.z)),
                "밀어낸 뒤 separation이 0이 아니다: {s2} (원래 {sep})"
            );
        }
        assert!(deep > 200, "관통 표본이 너무 적다: {deep}");
    }

    /// 인자 순서 대칭: (구, 박스)와 (박스, 구)는 같은 접촉을 **법선만 뒤집어** 낸다.
    #[test]
    fn sphere_box_argument_order_is_symmetric() {
        let mut rng = Rng(0x5133);
        for _ in 0..2000 {
            let half = Vec3::new(0.3 + rng.f(), 0.3 + rng.f(), 0.3 + rng.f());
            let rs = 0.2 + rng.f();
            let ss = Shape::Sphere { radius: rs };
            let sbx = Shape::Box { half, radius: 0.0 };
            let pb = Pose {
                pos: Vec3::ZERO,
                rot: rng.quat(),
            };
            let ps = Pose {
                pos: Vec3::new(rng.sym(3.0), rng.sym(3.0), rng.sym(3.0)),
                rot: Quat::IDENTITY,
            };
            let a = contact(&ss, &ps, &sbx, &pb, 0.05);
            let b = contact(&sbx, &pb, &ss, &ps, 0.05);
            match (a, b) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert!(
                        (a.normal + b.normal).length() < 1e-5,
                        "순서를 바꾸면 법선이 정확히 반대여야 한다"
                    );
                    assert!(
                        (a.points()[0].separation - b.points()[0].separation).abs() < 1e-5,
                        "separation은 순서에 무관해야 한다"
                    );
                    assert!(
                        (a.points()[0].position - b.points()[0].position).length() < 1e-5,
                        "접점 위치는 순서에 무관해야 한다"
                    );
                }
                _ => panic!("한쪽만 접촉을 냈다"),
            }
        }
    }
}
