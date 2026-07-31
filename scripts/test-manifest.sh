#!/usr/bin/env bash
# 테스트 **매니페스트** — 이름 + 속성(ignore·cfg_attr·should_panic)을 소스에서 뽑는다.
#
# 왜 필요한가: 테스트를 파일 사이로 옮기는 리팩터에서 `cargo test -- --list`의 이름만
# 대조하면 **속성 유실을 못 잡는다**. `#[ignore]`가 떨어지거나 `should_panic`이 사라져도
# 이름은 그대로라 diff가 비어 있다(교차검수 rev_661c32d5 지적).
# 그리고 모듈 경로가 바뀌면 `--list`의 완전 경로는 어차피 달라져 그대로는 비교가 안 된다.
#
# 그래서 **leaf 이름 + 정렬된 속성 집합**을 뽑는다 — 파일이나 모듈이 어디로 가든 불변이다.
#
# 사용:
#   scripts/test-manifest.sh > /tmp/before.txt
#   ... 리팩터 ...
#   scripts/test-manifest.sh > /tmp/after.txt && diff /tmp/before.txt /tmp/after.txt
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - "$@" <<'PY'
import re, sys, pathlib

rows = []
for p in sorted(pathlib.Path("crates").rglob("*.rs")):
    lines = p.read_text(encoding="utf-8").splitlines()
    attrs = []
    for line in lines:
        s = line.strip()
        if s.startswith("#["):
            # 속성은 여러 줄일 수 있으나 이 저장소는 한 줄이다 — 한 줄만 다룬다.
            attrs.append(re.sub(r"\s+", "", s))
            continue
        m = re.match(r"^(?:pub(?:\(crate\))? )?fn ([A-Za-z0-9_]+)", s)
        if m and any(a == "#[test]" for a in attrs):
            # `#[test]` 자체는 빼고 나머지를 정렬해 붙인다(선언 순서에 무관하게).
            rest = sorted(a for a in attrs if a != "#[test]")
            rows.append(f"{m.group(1)}\t{'|'.join(rest)}")
            attrs = []
            continue
        if s and not s.startswith("//"):
            attrs = []

rows.sort()
print(f"# 테스트 {len(rows)}개 (leaf 이름 + 속성)")
print("\n".join(rows))
PY
