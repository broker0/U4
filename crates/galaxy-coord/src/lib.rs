//! Galactic coordinate system.
//!
//! Absolute positions across the galaxy are stored as [`GalacticCoord`], built
//! on the [`Fixed`] newtype over `i128`. The current scale is **1 unit = 1
//! meter** with no fractional part (`SCALE = 1`). This is deliberate:
//!
//! * `i128` covers +/-1.7e38 meters ~= +/-1.8e22 light years, dwarfing any
//!   galaxy while leaving arithmetic overflow-free for addition/subtraction.
//! * Sub-meter precision near objects is recovered through the *floating
//!   origin* technique: [`GalacticCoord::relative_f64`] subtracts a nearby
//!   origin first, then converts the small delta to `f64`.
//!
//! The `SCALE` constant and the [`Fixed`] wrapper make it cheap to migrate to a
//! fixed-point format (e.g. Q96.32) later without touching call sites.

use glam::DVec3;

/// Number of light years per meter helper (meters per light year).
pub const METERS_PER_LIGHT_YEAR: f64 = 9.460_730_472_580_8e15;
/// Meters per parsec.
pub const METERS_PER_PARSEC: f64 = 3.085_677_581_491_367e16;
/// Meters per astronomical unit.
pub const METERS_PER_AU: f64 = 1.495_978_707e11;

/// Fixed-point scalar backing a single axis of a galactic coordinate.
///
/// Currently `SCALE == 1`, so the stored `i128` is a whole number of meters.
/// Kept as a newtype so the representation can change centrally.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Fixed(i128);

impl Fixed {
    /// Sub-units per whole unit. `1` means the raw value counts meters.
    pub const SCALE: i128 = 1;

    /// Zero value.
    pub const ZERO: Fixed = Fixed(0);

    /// Wrap a raw backing value (in sub-units) directly.
    #[inline]
    pub const fn from_raw(raw: i128) -> Self {
        Fixed(raw)
    }

    /// The raw backing integer (sub-units).
    #[inline]
    pub const fn raw(self) -> i128 {
        self.0
    }

    /// Construct from a whole number of meters.
    #[inline]
    pub const fn from_meters_i128(m: i128) -> Self {
        Fixed(m * Self::SCALE)
    }

    /// Construct from a (possibly fractional) meter value.
    ///
    /// Rounds to the nearest sub-unit. For very large magnitudes `f64` loses
    /// precision; that is acceptable because absolute placement of stars only
    /// needs coarse global resolution (fine detail comes from floating origin).
    #[inline]
    pub fn from_meters_f64(m: f64) -> Self {
        Fixed((m * Self::SCALE as f64).round() as i128)
    }

    /// Construct from a light-year value.
    #[inline]
    pub fn from_light_years(ly: f64) -> Self {
        Self::from_meters_f64(ly * METERS_PER_LIGHT_YEAR)
    }

    /// Value expressed in meters as `f64` (may lose precision when huge).
    #[inline]
    pub fn to_meters_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    /// Difference `self - other` expressed in meters as `f64`.
    ///
    /// This is the precision-preserving operation: the subtraction happens in
    /// exact `i128` space, so the result is accurate to a sub-unit as long as
    /// the magnitude of the *difference* fits in `f64`'s 52-bit mantissa
    /// (~9e15, i.e. ~1 light year of sub-meter-accurate range).
    #[inline]
    pub fn delta_meters_f64(self, other: Fixed) -> f64 {
        (self.0 - other.0) as f64 / Self::SCALE as f64
    }
}

impl core::ops::Add for Fixed {
    type Output = Fixed;
    #[inline]
    fn add(self, rhs: Fixed) -> Fixed {
        Fixed(self.0 + rhs.0)
    }
}

impl core::ops::Sub for Fixed {
    type Output = Fixed;
    #[inline]
    fn sub(self, rhs: Fixed) -> Fixed {
        Fixed(self.0 - rhs.0)
    }
}

impl core::ops::Neg for Fixed {
    type Output = Fixed;
    #[inline]
    fn neg(self) -> Fixed {
        Fixed(-self.0)
    }
}

impl core::ops::Mul<i128> for Fixed {
    type Output = Fixed;
    #[inline]
    fn mul(self, rhs: i128) -> Fixed {
        Fixed(self.0 * rhs)
    }
}

impl core::fmt::Debug for Fixed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Fixed({} m)", self.0 / Self::SCALE)
    }
}

/// An absolute position in the galaxy.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct GalacticCoord {
    pub x: Fixed,
    pub y: Fixed,
    pub z: Fixed,
}

impl GalacticCoord {
    pub const ORIGIN: GalacticCoord = GalacticCoord {
        x: Fixed::ZERO,
        y: Fixed::ZERO,
        z: Fixed::ZERO,
    };

    #[inline]
    pub const fn new(x: Fixed, y: Fixed, z: Fixed) -> Self {
        GalacticCoord { x, y, z }
    }

    /// Build from whole-meter integer components.
    #[inline]
    pub const fn from_meters_i128(x: i128, y: i128, z: i128) -> Self {
        GalacticCoord {
            x: Fixed::from_meters_i128(x),
            y: Fixed::from_meters_i128(y),
            z: Fixed::from_meters_i128(z),
        }
    }

    /// Build from a floating meter position (coarse global placement).
    #[inline]
    pub fn from_meters_f64(v: DVec3) -> Self {
        GalacticCoord {
            x: Fixed::from_meters_f64(v.x),
            y: Fixed::from_meters_f64(v.y),
            z: Fixed::from_meters_f64(v.z),
        }
    }

    /// Position relative to `origin`, in meters, as a small `f64` vector.
    ///
    /// This is the heart of the floating-origin renderer: it removes the huge
    /// absolute magnitude before any `f64`/`f32` work, avoiding jitter.
    #[inline]
    pub fn relative_f64(self, origin: GalacticCoord) -> DVec3 {
        DVec3::new(
            self.x.delta_meters_f64(origin.x),
            self.y.delta_meters_f64(origin.y),
            self.z.delta_meters_f64(origin.z),
        )
    }

    /// Add a small `f64` meter offset to an absolute position.
    #[inline]
    pub fn offset_f64(self, delta: DVec3) -> GalacticCoord {
        GalacticCoord {
            x: self.x + Fixed::from_meters_f64(delta.x),
            y: self.y + Fixed::from_meters_f64(delta.y),
            z: self.z + Fixed::from_meters_f64(delta.z),
        }
    }

    /// Squared distance in meters as `f64`. Uses exact `i128` subtraction, so
    /// the result is accurate provided the true distance fits in `f64`.
    #[inline]
    pub fn distance_sq_meters(self, other: GalacticCoord) -> f64 {
        let d = self.relative_f64(other);
        d.length_squared()
    }

    /// Distance in meters as `f64`.
    #[inline]
    pub fn distance_meters(self, other: GalacticCoord) -> f64 {
        self.distance_sq_meters(other).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn galaxy_scale_fits_without_overflow() {
        // Radius of the Milky Way ~ 50_000 ly. Place two stars on opposite
        // edges and ensure add/sub stay well inside i128.
        let r = Fixed::from_light_years(50_000.0);
        let a = GalacticCoord::new(r, r, r);
        let b = GalacticCoord::new(-r, -r, -r);
        // Should not panic (no overflow) and be positive.
        let d = a.distance_meters(b);
        assert!(d > 0.0);
        // ~ sqrt(3) * 100_000 ly in meters.
        let expected = (3.0_f64).sqrt() * 100_000.0 * METERS_PER_LIGHT_YEAR;
        assert!((d - expected).abs() / expected < 1e-6);
    }

    #[test]
    fn floating_origin_preserves_submeter_near_origin() {
        // Two points ~1 meter apart, but a galactic distance from absolute (0).
        let base = Fixed::from_light_years(10_000.0);
        let p1 = GalacticCoord::new(base, Fixed::ZERO, Fixed::ZERO);
        let p2 = GalacticCoord::new(base + Fixed::from_meters_i128(1), Fixed::ZERO, Fixed::ZERO);
        // Relative to p1, p2 must be exactly (1, 0, 0) meters, no jitter.
        let rel = p2.relative_f64(p1);
        assert_eq!(rel, DVec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn relative_is_antisymmetric() {
        let a = GalacticCoord::from_meters_i128(1_000, -2_000, 3_000);
        let b = GalacticCoord::from_meters_i128(-4_000, 5_000, -6_000);
        let ab = a.relative_f64(b);
        let ba = b.relative_f64(a);
        assert_eq!(ab, -ba);
    }

    #[test]
    fn offset_roundtrip() {
        let p = GalacticCoord::from_light_years_vec(1_000.0, 2_000.0, 3_000.0);
        let moved = p.offset_f64(DVec3::new(10.0, -20.0, 30.0));
        let back = moved.relative_f64(p);
        assert_eq!(back, DVec3::new(10.0, -20.0, 30.0));
    }

    #[test]
    fn from_meters_f64_rounds() {
        assert_eq!(Fixed::from_meters_f64(2.4).raw(), 2);
        assert_eq!(Fixed::from_meters_f64(2.6).raw(), 3);
        assert_eq!(Fixed::from_meters_f64(-2.6).raw(), -3);
    }
}

impl GalacticCoord {
    /// Convenience for tests/setup: build from light-year components.
    pub fn from_light_years_vec(x: f64, y: f64, z: f64) -> Self {
        GalacticCoord {
            x: Fixed::from_light_years(x),
            y: Fixed::from_light_years(y),
            z: Fixed::from_light_years(z),
        }
    }
}
