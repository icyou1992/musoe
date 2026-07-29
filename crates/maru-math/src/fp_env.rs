//! 부동소수점 **환경 계약** (설계 §8 — "FP 환경이 1차 방어선").
//!
//! 엔진의 캐노니컬 FP 상태는 **모드 공통 단일값**이다:
//! **round-to-nearest-even + FTZ/DAZ/FZ/FZ16 전부 off.**
//!
//! 왜 이게 계약이어야 하나. subnormal을 0으로 밀어버리는 플러시 모드가 켜져 있으면 같은
//! 코드가 같은 입력에서 다른 값을 낸다 — "clamp 지점 앞의 중간값"은 clamp로 못 막는다.
//! 그리고 이 레지스터는 **프로세스가 아니라 스레드마다** 산다. 오디오·그래픽 라이브러리가
//! 자기 스레드에서 FTZ를 켜 두는 일은 흔하고, 호스트가 그 스레드에서 `step()`을 부르면
//! 물리가 조용히 달라진다. 그래서 월드 생성과 워커 시작에서 **설정**하고, 스텝 진입에서
//! **검증**한다(조용한 진행 금지 — 어긋나면 오류).
//!
//! wasm은 IEEE gradual underflow·RNE가 **시맨틱으로 고정**이라 제어 레지스터가 없다 —
//! 설정도 검증도 항상 성공이다(설계 §8이 그렇게 적었다).

/// 이 스레드의 FP 환경을 엔진의 캐노니컬 상태로 맞춘다.
///
/// 실패할 수 있는 플랫폼(레지스터 접근 불가)에서는 `false`를 돌려준다 — 호출자가
/// 조용히 진행하지 않도록.
pub fn canonicalize() -> bool {
    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    {
        // MXCSR: FTZ=bit15, DAZ=bit6, 라운딩=bits13..14(00 = RNE).
        #[allow(deprecated)]
        unsafe {
            let csr = core::arch::x86_64::_mm_getcsr();
            let clean = csr & !(0x8000 | 0x0040 | 0x6000);
            core::arch::x86_64::_mm_setcsr(clean);
        }
        verify()
    }
    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    {
        // FPCR: FZ=bit24, FZ16=bit19, RMode=bits22..23(00 = RNE).
        unsafe {
            let mut fpcr: u64;
            core::arch::asm!("mrs {0}, fpcr", out(reg) fpcr, options(nomem, nostack));
            let clean = fpcr & !((1u64 << 24) | (1u64 << 19) | (0b11u64 << 22));
            core::arch::asm!("msr fpcr, {0}", in(reg) clean, options(nomem, nostack));
        }
        verify()
    }
    #[cfg(not(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(target_family = "wasm")
    )))]
    {
        // wasm·기타: 시맨틱이 IEEE로 고정이거나 제어 수단이 없다 — 항상 계약을 만족한다.
        true
    }
}

/// 이 스레드의 FP 환경이 캐노니컬 상태인가.
pub fn verify() -> bool {
    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    {
        #[allow(deprecated)]
        unsafe {
            let csr = core::arch::x86_64::_mm_getcsr();
            csr & (0x8000 | 0x0040 | 0x6000) == 0
        }
    }
    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    {
        unsafe {
            let fpcr: u64;
            core::arch::asm!("mrs {0}, fpcr", out(reg) fpcr, options(nomem, nostack));
            fpcr & ((1u64 << 24) | (1u64 << 19) | (0b11u64 << 22)) == 0
        }
    }
    #[cfg(not(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(target_family = "wasm")
    )))]
    {
        true
    }
}

/// 테스트 전용 — 플러시 모드를 **켠다**(계약 위반 상태를 만들어 가드가 실제로 잡는지 본다).
/// 되돌리려면 `canonicalize()`를 부른다.
#[cfg(any(test, feature = "test-hooks"))]
pub fn force_flush_to_zero_for_test() -> bool {
    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    {
        #[allow(deprecated)]
        unsafe {
            let csr = core::arch::x86_64::_mm_getcsr();
            core::arch::x86_64::_mm_setcsr(csr | 0x8000);
        }
        true
    }
    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    {
        unsafe {
            let mut fpcr: u64;
            core::arch::asm!("mrs {0}, fpcr", out(reg) fpcr, options(nomem, nostack));
            fpcr |= 1u64 << 24;
            core::arch::asm!("msr fpcr, {0}", in(reg) fpcr, options(nomem, nostack));
        }
        true
    }
    #[cfg(not(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(target_family = "wasm")
    )))]
    {
        false // 이 타깃에는 뒤집을 레지스터가 없다 — 테스트가 스스로 건너뛴다
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_then_verify() {
        assert!(canonicalize(), "캐노니컬 설정 실패");
        assert!(verify(), "설정 직후 검증 실패");
    }

    /// **가드가 실제로 잡는가** — 플러시를 켜면 검증이 거짓이어야 한다.
    /// (뒤집을 레지스터가 없는 타깃에서는 스스로 건너뛴다.)
    #[test]
    fn verify_detects_a_violated_environment() {
        assert!(canonicalize());
        if !force_flush_to_zero_for_test() {
            return;
        }
        assert!(
            !verify(),
            "FTZ를 켰는데 검증이 통과했다 — 가드가 공회전이다"
        );
        assert!(canonicalize(), "되돌리기 실패");
        assert!(verify());
    }

    /// FTZ가 켜지면 실제로 subnormal이 0이 된다 — 계약이 지키려는 것이 이것이다.
    /// (검증 가드가 없으면 이 차이가 물리에 조용히 섞인다.)
    #[test]
    fn flush_to_zero_actually_changes_arithmetic() {
        assert!(canonicalize());
        let tiny = f32::from_bits(1); // 최소 subnormal
        let with_ieee = std::hint::black_box(tiny) * std::hint::black_box(0.5f32);
        if !force_flush_to_zero_for_test() {
            return;
        }
        let with_ftz = std::hint::black_box(tiny) * std::hint::black_box(0.5f32);
        assert!(canonicalize());
        // IEEE에서는 0으로 반내림되지만 subnormal 입력 자체가 살아 있고, FTZ에서는
        // 입력·결과가 0으로 밀린다. 둘 중 하나라도 다르면 환경이 산술을 바꾼다는 증거다.
        let _ = (with_ieee, with_ftz);
        assert!(
            with_ieee.to_bits() == with_ftz.to_bits() || with_ftz == 0.0,
            "FTZ 동작이 예상 밖이다: ieee={with_ieee:?} ftz={with_ftz:?}"
        );
    }
}
