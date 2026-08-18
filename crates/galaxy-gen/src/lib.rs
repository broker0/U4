//! Procedural star generation.
//!
//! Content is *computed on demand* from a root seed and a node key; nothing is
//! stored. The design ties octree depth to a **brightness layer**: shallow
//! nodes (huge volumes) emit only intrinsically bright stars visible from far
//! away, deeper nodes add progressively dimmer stars. Each star belongs to
//! exactly one depth (the one whose magnitude band contains it), so descending
//! the tree never produces duplicates.
//!
//! Star *counts* follow a realistic field luminosity function (H-R diagram
//! shape): the whole galaxy's `total_stars` is distributed across the absolute-
//! magnitude bands by how common such stars really are (red dwarfs dominate,
//! giants are rare), and each node's budget is its band's share divided by the
//! `8^depth` nodes partitioning the cube. So the resolved star density per
//! volume matches the real solar neighbourhood, rather than giving every band
//! an equal count.

use galaxy_coord::{Fixed, GalacticCoord, METERS_PER_LIGHT_YEAR};
use glam::DVec3;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Identifies a cubic octree node by explicit level indices.
///
/// The universe cube spans `[0, 2^depth)` cells per axis at a given `depth`;
/// `depth == 0` is the single root cell covering the whole modeled volume.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeKey {
    pub depth: u8,
    pub ix: u64,
    pub iy: u64,
    pub iz: u64,
}

impl NodeKey {
    pub const ROOT: NodeKey = NodeKey {
        depth: 0,
        ix: 0,
        iy: 0,
        iz: 0,
    };

    #[inline]
    pub const fn new(depth: u8, ix: u64, iy: u64, iz: u64) -> Self {
        NodeKey { depth, ix, iy, iz }
    }

    /// The eight children of this node.
    pub fn children(self) -> [NodeKey; 8] {
        let d = self.depth + 1;
        let (bx, by, bz) = (self.ix * 2, self.iy * 2, self.iz * 2);
        let mut out = [NodeKey::new(d, 0, 0, 0); 8];
        let mut i = 0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    out[i] = NodeKey::new(d, bx + dx, by + dy, bz + dz);
                    i += 1;
                }
            }
        }
        out
    }
}

/// A generated star (a point light with color and intrinsic brightness).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Star {
    /// Absolute galactic position.
    pub pos: GalacticCoord,
    /// Absolute magnitude (smaller = intrinsically brighter).
    pub abs_mag: f32,
    /// Linear RGB color derived from spectral class.
    pub color: [f32; 3],
    /// Display radius hint in meters (stellar radius, coarse).
    pub radius: f32,
}

/// Parameters controlling the shape of the generated galaxy.
#[derive(Clone, Copy, Debug)]
pub struct GalaxyParams {
    /// Root random seed.
    pub root_seed: u64,
    /// Edge length of the whole modeled cube, in meters (root node span).
    ///
    /// The default is `2^17 = 131_072` light years, so every node edge is a
    /// clean power-of-two number of light years (`2^(17 - depth)`).
    pub universe_size_m: f64,
    /// Total number of stars the galaxy would contain if fully resolved to
    /// `max_depth`. The per-node budget is derived from this and from a
    /// realistic luminosity function (see [`GalaxyParams::expected_stars`]):
    /// each magnitude band's star count is proportional to how common stars of
    /// that absolute magnitude actually are, so the generated population
    /// follows the mass-weighted main-sequence (H-R) distribution instead of
    /// giving equal counts to every band.
    pub total_stars: f64,
    /// Deepest octree level whose (faintest) magnitude band is populated.
    pub max_depth: u8,
}

impl Default for GalaxyParams {
    fn default() -> Self {
        GalaxyParams {
            root_seed: 0x5347_414C_4158_5921,
            universe_size_m: 131_072.0 * METERS_PER_LIGHT_YEAR,
            total_stars: 2.0e11,
            max_depth: 24,
        }
    }
}

impl GalaxyParams {
    /// Edge length in meters of a node at `depth`.
    #[inline]
    pub fn node_size_m(&self, depth: u8) -> f64 {
        self.universe_size_m / (1u64 << depth) as f64
    }

    /// Absolute-magnitude upper bound (inclusive) shown *first* at `depth`.
    ///
    /// Depth 0 shows only the very brightest (small magnitude); each deeper
    /// level raises the limit, revealing dimmer stars. The band owned by
    /// `depth` is `(mag_limit(depth-1), mag_limit(depth)]`.
    #[inline]
    pub fn mag_limit(&self, depth: u8) -> f32 {
        // Roughly one magnitude of fainter stars unlocked per level, starting
        // near supergiants (abs mag ~ -8) down to faint dwarfs (~ +16).
        -9.0 + depth as f32 * 1.05
    }

    /// The half-open absolute-magnitude band owned exclusively by `depth`.
    #[inline]
    pub fn mag_band(&self, depth: u8) -> (f32, f32) {
        let hi = self.mag_limit(depth);
        let lo = if depth == 0 {
            f32::NEG_INFINITY
        } else {
            self.mag_limit(depth - 1)
        };
        (lo, hi)
    }

    /// Midpoint of the absolute-magnitude band owned by `depth`.
    ///
    /// Depth 0's lower bound is `-inf`; use the clamped sampling limit instead.
    #[inline]
    pub fn band_mid(&self, depth: u8) -> f32 {
        if depth == 0 {
            -10.5
        } else {
            (self.mag_limit(depth - 1) + self.mag_limit(depth)) * 0.5
        }
    }

    /// The number of stars the whole galaxy should contain in `depth`'s
    /// magnitude band, as a fraction of [`GalaxyParams::total_stars`].
    ///
    /// Uses a field luminosity function (stars per cubic parsec per magnitude,
    /// from the local solar-neighbourhood census): stellar density rises steeply
    /// from the bright end to a red-dwarf peak near `M ~ +11.5` and then falls
    /// off for the faintest dwarfs. Bands whose midpoint falls outside the
    /// table clamp to the nearest tabulated value.
    pub fn band_weight(&self, depth: u8) -> f64 {
        let total: f64 = (0..=self.max_depth)
            .map(|d| luminosity_weight(self.band_mid(d)))
            .sum();
        if total <= 0.0 {
            0.0
        } else {
            luminosity_weight(self.band_mid(depth)) / total
        }
    }

    /// Expected number of stars emitted by a single node at `depth`, given the
    /// band's share of the galaxy-wide [`GalaxyParams::total_stars`] spread
    /// uniformly over all `8^depth` nodes at that level.
    ///
    /// This is the key difference from a constant per-node budget: because the
    /// galaxy-wide total of a band stays *fixed* regardless of resolution, the
    /// per-node count falls off as `1/8^depth`. Equivalently, the star density
    /// per unit volume follows the luminosity function (dense near shell nodes
    /// are full of faint red dwarfs, huge shallow nodes carry only the rare
    /// bright giants).
    #[inline]
    pub fn expected_stars(&self, depth: u8) -> f64 {
        let d = depth.min(self.max_depth);
        self.total_stars * self.band_weight(d) / (8.0f64).powi(d as i32)
    }
}

/// Field luminosity function: relative density of stars per cubic parsec per
/// absolute-magnitude bin, `(M_V, log10(relative weight))`.
///
/// Shape approximates the solar-neighbourhood census: a steep falloff for
/// luminous giants, a slow rise through the main sequence, a red-dwarf peak
/// near `M ~ +11.5`, then decline into the faintest dwarfs. Weights are
/// relative (normalized later), so only the *shape*, not the scale, matters
/// here.
static LUMINOSITY_FUNCTION: &[(f32, f64)] = &[
    (-10.5, -11.00),
    (-9.5, -9.80),
    (-8.5, -8.70),
    (-7.5, -7.60),
    (-6.5, -6.60),
    (-5.5, -5.70),
    (-4.5, -4.90),
    (-3.5, -4.20),
    (-2.5, -3.70),
    (-1.5, -3.30),
    (-0.5, -3.00),
    (0.5, -2.80),
    (1.5, -2.60),
    (2.5, -2.45),
    (3.5, -2.35),
    (4.5, -2.30),
    (5.5, -2.30),
    (6.5, -2.25),
    (7.5, -2.20),
    (8.5, -2.15),
    (9.5, -2.05),
    (10.5, -1.95),
    (11.5, -1.90),
    (12.5, -1.95),
    (13.5, -2.05),
    (14.5, -2.20),
    (15.5, -2.45),
    (16.5, -2.80),
    (18.0, -3.50),
];

/// Relative luminosity-function weight at an absolute magnitude, interpolated
/// linearly in log space between tabulated entries.
#[inline]
fn luminosity_weight(abs_mag: f32) -> f64 {
    let first = LUMINOSITY_FUNCTION[0];
    let last = *LUMINOSITY_FUNCTION.last().unwrap();
    if abs_mag <= first.0 {
        return 10f64.powf(first.1);
    }
    if abs_mag >= last.0 {
        return 10f64.powf(last.1);
    }
    for seg in LUMINOSITY_FUNCTION.windows(2) {
        let (m0, l0) = seg[0];
        let (m1, l1) = seg[1];
        if (m0..=m1).contains(&abs_mag) {
            if m1 <= m0 {
                return 10f64.powf(l0);
            }
            let t = ((abs_mag - m0) / (m1 - m0)) as f64;
            return 10f64.powf(l0 + (l1 - l0) * t);
        }
    }
    0.0
}

/// Deterministic per-node seed derived from the root seed and node key.
///
/// Uses a SplitMix64-style avalanche so neighboring nodes decorrelate well.
#[inline]
pub fn node_seed(root_seed: u64, key: NodeKey) -> u64 {
    let mut h = root_seed;
    h = mix(h ^ 0x9E37_79B9_7F4A_7C15 ^ key.depth as u64);
    h = mix(h ^ key.ix.rotate_left(17));
    h = mix(h ^ key.iy.rotate_left(31));
    h = mix(h ^ key.iz.rotate_left(47));
    h
}

#[inline]
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Generate the stars owned by `key`: those whose absolute magnitude falls in
/// this depth's exclusive band and whose position lies inside the node cube.
pub fn generate_stars(params: &GalaxyParams, key: NodeKey) -> Vec<Star> {
    let seed = node_seed(params.root_seed, key);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let node_size = params.node_size_m(key.depth);
    // Node origin (min corner) in absolute meters. Root cube is centered on
    // the universe origin so the galaxy straddles (0,0,0).
    let half = params.universe_size_m * 0.5;
    let origin = DVec3::new(
        key.ix as f64 * node_size - half,
        key.iy as f64 * node_size - half,
        key.iz as f64 * node_size - half,
    );

    // Per-node budget follows the luminosity function: shallow (galaxy-wide)
    // nodes own only the rare bright giants, deeper nodes carry the common
    // dwarf population, so the resolved star density per volume matches the
    // real solar-neighbourhood (H-R) distribution.
    let count = poisson_ish(&mut rng, params.expected_stars(key.depth));

    let (band_lo, band_hi) = params.mag_band(key.depth);
    // Finite bounds for uniform sampling within the band. Depth 0's lower
    // bound is -inf; clamp to a sane brightest absolute magnitude.
    let lo = if band_lo.is_finite() { band_lo } else { -12.0 };
    let hi = band_hi;

    let mut stars = Vec::with_capacity(count.min(8192));
    for _ in 0..count {
        // Draw an absolute magnitude directly inside this depth's band, biased
        // toward the faint end to mimic a stellar mass function locally.
        let u: f32 = rng.random::<f32>();
        let mut abs_mag = lo + (hi - lo) * u.powf(1.6);
        // The band is half-open `(lo, hi]`: never land exactly on `lo`, which
        // marks another depth's bright edge (float rounding can collide).
        if abs_mag <= lo {
            abs_mag = lo + (hi - lo) * 1.0e-4;
        }

        let lp = DVec3::new(
            rng.random::<f64>() * node_size,
            rng.random::<f64>() * node_size,
            rng.random::<f64>() * node_size,
        );
        let world = origin + lp;

        let temp_k = spectral_temperature(&mut rng);
        let color = blackbody_srgb(temp_k);
        let radius = stellar_radius_m(abs_mag, temp_k);

        stars.push(Star {
            pos: GalacticCoord::new(
                Fixed::from_meters_f64(world.x),
                Fixed::from_meters_f64(world.y),
                Fixed::from_meters_f64(world.z),
            ),
            abs_mag,
            color,
            radius,
        });
    }
    stars
}

/// Approximate Poisson draw good enough for content generation.
fn poisson_ish(rng: &mut ChaCha8Rng, lambda: f64) -> usize {
    if lambda <= 0.0 {
        return 0;
    }
    if lambda < 30.0 {
        // Knuth's algorithm.
        let l = (-lambda).exp();
        let mut k = 0usize;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.random::<f64>();
            if p <= l {
                return k - 1;
            }
        }
    }
    // Normal approximation for large lambda.
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    (lambda + z * lambda.sqrt()).round().max(0.0) as usize
}

/// Sample an effective temperature (Kelvin) correlated loosely with type.
fn spectral_temperature(rng: &mut ChaCha8Rng) -> f32 {
    // Range ~2500 K (M) to ~30000 K (O/B), skewed cool.
    let u: f64 = rng.random::<f64>();
    (2500.0 + 27_500.0 * u.powf(2.2)) as f32
}

/// Very rough stellar radius in meters from magnitude and temperature.
fn stellar_radius_m(abs_mag: f32, temp_k: f32) -> f32 {
    // Luminosity from magnitude (Sun abs mag ~ 4.83, L_sun ~ 3.828e26 W).
    let l_rel = 10f64.powf((4.83 - abs_mag as f64) / 2.5).clamp(1e-4, 1e8);
    let l_w = l_rel * 3.828e26;
    // Stefan-Boltzmann: L = 4 pi R^2 sigma T^4 -> R.
    let sigma = 5.670_374_419e-8;
    let t = temp_k as f64;
    let r = (l_w / (4.0 * std::f64::consts::PI * sigma * t.powi(4))).sqrt();
    r as f32
}

/// Approximate blackbody color -> normalized linear sRGB.
fn blackbody_srgb(temp_k: f32) -> [f32; 3] {
    // Simple piecewise approximation (Tanner Helland style, normalized).
    let t = (temp_k / 100.0) as f64;
    let r;
    let g;
    let b;
    if t <= 66.0 {
        r = 1.0;
        g = (0.39008157 * t.ln() - 0.63184144).clamp(0.0, 1.0);
    } else {
        r = (1.292936 * (t - 60.0).powf(-0.1332047)).clamp(0.0, 1.0);
        g = (1.129890 * (t - 60.0).powf(-0.0755148)).clamp(0.0, 1.0);
    }
    if t >= 66.0 {
        b = 1.0;
    } else if t <= 19.0 {
        b = 0.0;
    } else {
        b = (0.5432067 * (t - 10.0).ln() - 1.196254).clamp(0.0, 1.0);
    }
    [r as f32, g as f32, b as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> GalaxyParams {
        GalaxyParams {
            root_seed: 12345,
            universe_size_m: 1.0e18,
            total_stars: 1.0e13,
            max_depth: 10,
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let p = params();
        let key = NodeKey::new(3, 2, 5, 1);
        let a = generate_stars(&p, key);
        let b = generate_stars(&p, key);
        assert_eq!(a, b);
        assert!(!a.is_empty(), "expected some stars in a mid-depth node");
    }

    #[test]
    fn different_nodes_differ() {
        let p = params();
        let a = generate_stars(&p, NodeKey::new(3, 2, 5, 1));
        let b = generate_stars(&p, NodeKey::new(3, 2, 5, 2));
        assert_ne!(a, b);
    }

    #[test]
    fn stars_respect_their_depth_magnitude_band() {
        let p = params();
        for depth in 0..8u8 {
            let (lo, hi) = p.mag_band(depth);
            // Sample a handful of nodes at this depth.
            for i in 0..4u64 {
                let stars = generate_stars(&p, NodeKey::new(depth, i, i, i));
                for s in &stars {
                    assert!(
                        s.abs_mag > lo && s.abs_mag <= hi,
                        "depth {depth}: mag {} outside band ({lo}, {hi}]",
                        s.abs_mag
                    );
                }
            }
        }
    }

    #[test]
    fn stars_lie_inside_node_cube() {
        let p = params();
        let key = NodeKey::new(4, 3, 7, 2);
        let node_size = p.node_size_m(key.depth);
        let half = p.universe_size_m * 0.5;
        let min = DVec3::new(
            key.ix as f64 * node_size - half,
            key.iy as f64 * node_size - half,
            key.iz as f64 * node_size - half,
        );
        let max = min + DVec3::splat(node_size);
        for s in generate_stars(&p, key) {
            let x = s.pos.x.to_meters_f64();
            let y = s.pos.y.to_meters_f64();
            let z = s.pos.z.to_meters_f64();
            assert!(x >= min.x - 1.0 && x <= max.x + 1.0);
            assert!(y >= min.y - 1.0 && y <= max.y + 1.0);
            assert!(z >= min.z - 1.0 && z <= max.z + 1.0);
        }
    }

    #[test]
    fn children_partition_parent() {
        let parent = NodeKey::new(2, 1, 1, 1);
        let kids = parent.children();
        assert_eq!(kids.len(), 8);
        for k in kids {
            assert_eq!(k.depth, 3);
            assert!(k.ix / 2 == parent.ix && k.iy / 2 == parent.iy && k.iz / 2 == parent.iz);
        }
    }
}
