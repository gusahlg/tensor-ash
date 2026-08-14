//! Element types for tensor storage.
//!
//! Compute always accumulates in f32; `F16` is a *storage* format that
//! halves the bytes a kernel moves.  Host-side transfer APIs stay
//! `&[f32]` — conversion happens on the CPU during staging so callers
//! never handle raw half-precision bits.

/// Storage element type of a [`crate::Tensor`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum DType {
    /// 32-bit IEEE 754 float (the default).
    #[default]
    F32,
    /// 16-bit IEEE 754 half. Storage only: kernels convert to f32 on
    /// load and accumulate in f32, so results differ from an all-f32
    /// run only by the rounding of the stored values themselves.
    F16,
}

impl DType {
    /// Bytes per element.
    #[inline]
    pub fn size(self) -> u64 {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
        }
    }

    /// Lowercase name used in error messages and kernel routing.
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
        }
    }
}

/// Convert an f32 to IEEE 754 binary16 bits with round-to-nearest-even.
/// Out-of-range values become signed infinity; NaN payload collapses to
/// a quiet NaN.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf or NaN. Preserve NaN-ness with a quiet-NaN mantissa bit.
        let nan = if mantissa != 0 { 0x0200 } else { 0 };
        return sign | 0x7c00 | nan;
    }

    // Re-bias: f32 exponent bias 127, f16 bias 15.
    let unbiased = exp - 127;
    if unbiased >= 16 {
        return sign | 0x7c00; // overflow -> inf
    }
    if unbiased >= -14 {
        // Normal range. Round the 23-bit mantissa to 10 bits (RNE).
        let half_exp = ((unbiased + 15) as u32) << 10;
        let mantissa10 = mantissa >> 13;
        let rest = mantissa & 0x1fff;
        let mut out = (sign as u32) | half_exp | mantissa10;
        if rest > 0x1000 || (rest == 0x1000 && (mantissa10 & 1) == 1) {
            out += 1; // may carry into the exponent; that is correct RNE
        }
        return out as u16;
    }
    if unbiased >= -25 {
        // Subnormal half. Add the implicit bit, shift right, round RNE.
        let full = mantissa | 0x0080_0000;
        let shift = (-14 - unbiased) as u32 + 13;
        let mantissa10 = full >> shift;
        let rest = full & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let mut out = (sign as u32) | mantissa10;
        if rest > halfway || (rest == halfway && (mantissa10 & 1) == 1) {
            out += 1;
        }
        return out as u16;
    }
    sign // underflow -> signed zero
}

/// Convert IEEE 754 binary16 bits to an f32.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;

    let out = match (exp, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalize by shifting the mantissa up.
            let shift = mantissa.leading_zeros() - 21; // 10-bit field
            let mantissa = (mantissa << (shift + 1)) & 0x03ff;
            let exp = 127 - 15 - shift;
            sign | (exp << 23) | (mantissa << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (mantissa << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(out)
}

/// Round an f32 through f16 storage precision (encode + decode).
#[inline]
pub fn round_f32_via_f16(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values_roundtrip() {
        for v in [
            0.0f32, -0.0, 1.0, -1.0, 0.5, 2.0, 65504.0, -65504.0, 0.25, 1024.0,
        ] {
            assert_eq!(round_f32_via_f16(v), v, "{v} should be exact in f16");
            assert_eq!(round_f32_via_f16(v).to_bits(), v.to_bits());
        }
    }

    #[test]
    fn known_encodings() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00, "overflow rounds to inf");
        assert_eq!(f32_to_f16_bits(6.104e-5), 0x0400, "smallest normal");
        assert_eq!(f32_to_f16_bits(5.96e-8), 0x0001, "smallest subnormal");
        assert_eq!(f32_to_f16_bits(1e-10), 0x0000, "underflow to zero");
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn round_to_nearest_even() {
        // 1 + 2^-11 is exactly halfway between 1.0 and the next half
        // (1 + 2^-10); RNE keeps the even mantissa (1.0).
        assert_eq!(round_f32_via_f16(1.0 + 2.0f32.powi(-11)), 1.0);
        // 1 + 3*2^-11 is halfway between odd 1+2^-10 and even 1+2^-9.
        assert_eq!(
            round_f32_via_f16(1.0 + 3.0 * 2.0f32.powi(-11)),
            1.0 + 2.0f32.powi(-9)
        );
    }

    #[test]
    fn rounding_error_is_bounded() {
        // Relative error of f16 rounding is at most 2^-11 for normals.
        let mut value = 0.001f32;
        while value < 60000.0 {
            let rounded = round_f32_via_f16(value);
            let rel = ((rounded - value) / value).abs();
            assert!(rel <= 2.0f32.powi(-11), "value {value}: rel {rel}");
            value *= 1.37;
        }
    }
}
