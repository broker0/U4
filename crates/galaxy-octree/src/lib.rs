//! Octree traversal, level-of-detail selection, physical visibility culling,
//! and an LRU cache of generated node contents.
//!
//! The traversal walks the tree top-down. A node is subdivided when it is both
//! large enough on screen (angular-size heuristic) and shallow enough to still
//! own dimmer stars worth revealing. For every visited node we generate (or
//! fetch from cache) its stars, then keep only those whose *apparent* magnitude
//! — computed from absolute magnitude and true distance to the camera — passes
//! a configurable limit. This is the physically-motivated criterion.

use std::num::NonZeroUsize;

use galaxy_coord::{GalacticCoord, METERS_PER_PARSEC};
use galaxy_gen::{generate_stars, GalaxyParams, NodeKey, Star};
use glam::DVec3;
use lru::LruCache;

/// Apparent magnitude of a star of absolute magnitude `abs_mag` seen from
/// `distance_m` meters. `m = M + 5*(log10(d_pc) - 1)`.
#[inline]
pub fn apparent_magnitude(abs_mag: f32, distance_m: f64) -> f32 {
    let d_pc = (distance_m / METERS_PER_PARSEC).max(1e-6);
    abs_mag + 5.0 * (d_pc.log10() - 1.0) as f32
}

/// Tunable parameters for a single frame's traversal.
#[derive(Clone, Copy, Debug)]
pub struct ViewParams {
    /// Faintest apparent magnitude that is still rendered (larger = more stars).
    pub mag_limit: f32,
    /// Width (in magnitudes) of the fade band just below `mag_limit`. Stars in
    /// `[mag_limit - fade_width, mag_limit]` ramp their brightness from full to
    /// zero, so faint stars appear/disappear smoothly instead of popping in as
    /// hard "walls" when the camera approaches.
    pub fade_width: f32,
    /// A node is subdivided when its angular size (rad) exceeds this.
    pub subdivide_angle: f64,
    /// Hard cap on traversal depth (safety / performance).
    pub max_depth: u8,
    /// Hard cap on nodes visited per frame.
    pub max_nodes: usize,
}

impl Default for ViewParams {
    fn default() -> Self {
        ViewParams {
            mag_limit: 10.0,
            fade_width: 1.5,
            subdivide_angle: 1.0,
            max_depth: 24,
            max_nodes: 5_000,
        }
    }
}

/// A visible star together with a view-dependent fade weight in `[0, 1]` that
/// the renderer multiplies into the star's intensity.
#[derive(Clone, Copy, Debug)]
pub struct VisibleStar {
    pub star: Star,
    pub fade: f32,
}

/// Diagnostics collected during a traversal (surfaced in the debug overlay).
#[derive(Clone, Copy, Debug, Default)]
pub struct TraverseStats {
    pub nodes_visited: usize,
    pub nodes_subdivided: usize,
    pub stars_generated: usize,
    pub stars_visible: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// A node queued for best-first expansion, ordered by angular importance.
struct QueuedNode {
    key: NodeKey,
    /// Angular size = node edge length / nearest AABB distance.
    angular: f64,
}

impl PartialEq for QueuedNode {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for QueuedNode {}
impl PartialOrd for QueuedNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QueuedNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Larger angular size = higher priority (`total_cmp` handles any NaN).
        // Break ties by the node key so ordering is fully deterministic and the
        // truncated tail under `max_nodes` is reproducible regardless of the
        // insertion history (e.g. how subdivide_angle was changed to get here).
        self.angular
            .total_cmp(&other.angular)
            .then_with(|| self.key.depth.cmp(&other.key.depth))
            .then_with(|| self.key.ix.cmp(&other.key.ix))
            .then_with(|| self.key.iy.cmp(&other.key.iy))
            .then_with(|| self.key.iz.cmp(&other.key.iz))
    }
}

/// Holds the galaxy parameters and a cache of generated node contents.
pub struct Galaxy {
    pub params: GalaxyParams,
    cache: LruCache<NodeKey, std::sync::Arc<Vec<Star>>>,
}

impl Galaxy {
    pub fn new(params: GalaxyParams, cache_capacity: usize) -> Self {
        Galaxy {
            params,
            cache: LruCache::new(NonZeroUsize::new(cache_capacity.max(1)).unwrap()),
        }
    }

    /// Min corner of a node in absolute meters (root cube centered on origin).
    fn node_min_m(&self, key: NodeKey) -> DVec3 {
        let size = self.params.node_size_m(key.depth);
        let half = self.params.universe_size_m * 0.5;
        DVec3::new(
            key.ix as f64 * size - half,
            key.iy as f64 * size - half,
            key.iz as f64 * size - half,
        )
    }

    /// Center of a node in absolute galactic coordinates.
    pub fn node_center(&self, key: NodeKey) -> GalacticCoord {
        let size = self.params.node_size_m(key.depth);
        let min = self.node_min_m(key);
        GalacticCoord::from_meters_f64(min + DVec3::splat(size * 0.5))
    }

    /// Min corner of a node in absolute meters plus its edge size (meters).
    ///
    /// Exposed for debug tooling (e.g. the wireframe overlay) that needs the
    /// cube's world bounds of an arbitrary node key.
    pub fn node_bounds(&self, key: NodeKey) -> (DVec3, f64) {
        let size = self.params.node_size_m(key.depth);
        (self.node_min_m(key), size)
    }

    /// Distance in meters from `camera` to the nearest point of a node's cube.
    ///
    /// Using the nearest point (not the center) is essential for correct LOD:
    /// a huge shallow node whose center is far away may still touch the camera,
    /// in which case it must subdivide. The clamp-to-box formula gives 0 when
    /// the camera is inside the node.
    pub fn node_near_dist(&self, key: NodeKey, camera: GalacticCoord) -> f64 {
        let size = self.params.node_size_m(key.depth);
        let min = self.node_min_m(key);
        let max = min + DVec3::splat(size);
        // Camera position relative to the box min corner, in meters. Using the
        // box min as the floating origin keeps the subtraction exact.
        let origin = GalacticCoord::from_meters_f64(min);
        let cam = camera.relative_f64(origin); // camera - min, in meters
        let extent = max - min; // == size on each axis
                                // Clamp camera into [0, extent] per axis; distance to that clamped pt.
        let clamped = cam.clamp(DVec3::ZERO, extent);
        (cam - clamped).length()
    }

    /// Build a queue entry for `key`, computing its angular importance from the
    /// nearest AABB distance.
    fn make_queued(&self, key: NodeKey, camera: GalacticCoord) -> QueuedNode {
        let size = self.params.node_size_m(key.depth);
        let near = self.node_near_dist(key, camera).max(1.0);
        QueuedNode {
            key,
            angular: size / near,
        }
    }

    /// Fetch (or generate and cache) the stars owned by a node.
    fn stars(&mut self, key: NodeKey, stats: &mut TraverseStats) -> std::sync::Arc<Vec<Star>> {
        if let Some(v) = self.cache.get(&key) {
            stats.cache_hits += 1;
            return v.clone();
        }
        stats.cache_misses += 1;
        let v = std::sync::Arc::new(generate_stars(&self.params, key));
        self.cache.put(key, v.clone());
        v
    }

    /// Collect all stars visible from `camera` under `view`.
    ///
    /// Returns visible stars, traversal statistics, and the visited node keys
    /// (in visit order) for optional octree wireframe visualization.
    /// Frustum culling is left to the caller/renderer for now; this does
    /// distance-based LOD and the physical apparent-magnitude cut.
    ///
    /// Traversal is a *best-first* expansion ordered by angular importance
    /// (`size / nearest_distance`). When the `max_nodes` budget is exhausted we
    /// therefore drop the least important (smallest, farthest) nodes rather than
    /// an arbitrary DFS tail. This makes the visible set stable and symmetric as
    /// the camera turns or `subdivide_angle` changes.
    pub fn collect_visible(
        &mut self,
        camera: GalacticCoord,
        view: &ViewParams,
    ) -> (Vec<VisibleStar>, TraverseStats, Vec<NodeKey>) {
        use std::collections::BinaryHeap;

        let mut stats = TraverseStats::default();
        let mut out: Vec<VisibleStar> = Vec::new();
        let mut visited: Vec<NodeKey> = Vec::new();

        let mut heap: BinaryHeap<QueuedNode> = BinaryHeap::new();
        heap.push(self.make_queued(NodeKey::ROOT, camera));

        let fade_w = view.fade_width.max(1e-3);

        while let Some(QueuedNode { key, angular, .. }) = heap.pop() {
            if stats.nodes_visited >= view.max_nodes {
                break;
            }
            stats.nodes_visited += 1;
            visited.push(key);

            // Emit this node's owned stars (its magnitude band). Instead of a
            // hard apparent-magnitude cut, fade brightness over a window just
            // below `mag_limit` so faint stars ramp in smoothly (no "walls").
            let stars = self.stars(key, &mut stats);
            stats.stars_generated += stars.len();
            for s in stars.iter() {
                let d = s.pos.distance_meters(camera);
                let m = apparent_magnitude(s.abs_mag, d);
                // fade = 1 well above the limit, ramps to 0 at the limit.
                let fade = ((view.mag_limit - m) / fade_w).clamp(0.0, 1.0);
                if fade > 0.0 {
                    out.push(VisibleStar { star: *s, fade });
                    stats.stars_visible += 1;
                }
            }

            // Subdivide when the node subtends a large angle (near relative to
            // its size). `angular` uses the nearest AABB point, so large shallow
            // nodes touching the camera expand correctly.
            if key.depth < view.max_depth
                && key.depth < self.params.max_depth
                && angular > view.subdivide_angle
            {
                stats.nodes_subdivided += 1;
                for child in key.children() {
                    heap.push(self.make_queued(child, camera));
                }
            }
        }

        (out, stats, visited)
    }

    /// Find the nearest generated star to `camera`, used to pick a floating
    /// origin. Searches only the leaf-ish nodes containing the camera by
    /// descending to `probe_depth`.
    pub fn nearest_star(&mut self, camera: GalacticCoord, probe_depth: u8) -> Option<Star> {
        let mut stats = TraverseStats::default();
        let mut best: Option<(f64, Star)> = None;
        // Probe the node containing the camera and its neighbors at a few
        // depths, accumulating candidate stars.
        for depth in 0..=probe_depth.min(self.params.max_depth) {
            let size = self.params.node_size_m(depth);
            let half = self.params.universe_size_m * 0.5;
            let rel = camera.relative_f64(GalacticCoord::from_meters_f64(DVec3::splat(-half)));
            let ci = (rel.x / size).floor() as i64;
            let cj = (rel.y / size).floor() as i64;
            let ck = (rel.z / size).floor() as i64;
            let span = 1i64 << depth;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (i, j, k) = (ci + dx, cj + dy, ck + dz);
                        if i < 0 || j < 0 || k < 0 || i >= span || j >= span || k >= span {
                            continue;
                        }
                        let key = NodeKey::new(depth, i as u64, j as u64, k as u64);
                        let stars = self.stars(key, &mut stats);
                        for s in stars.iter() {
                            let d = s.pos.distance_meters(camera);
                            if best.as_ref().map_or(true, |(bd, _)| d < *bd) {
                                best = Some((d, *s));
                            }
                        }
                    }
                }
            }
        }
        best.map(|(_, s)| s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use galaxy_coord::{METERS_PER_LIGHT_YEAR, METERS_PER_PARSEC};

    #[test]
    fn apparent_magnitude_at_10pc_equals_absolute() {
        let m = apparent_magnitude(5.0, 10.0 * METERS_PER_PARSEC);
        assert!((m - 5.0).abs() < 1e-4);
    }

    #[test]
    fn farther_stars_are_fainter() {
        let near = apparent_magnitude(0.0, 10.0 * METERS_PER_PARSEC);
        let far = apparent_magnitude(0.0, 1000.0 * METERS_PER_PARSEC);
        assert!(far > near);
    }

    #[test]
    fn traversal_terminates_and_reports_stats() {
        let params = GalaxyParams {
            root_seed: 7,
            universe_size_m: 131_072.0 * METERS_PER_LIGHT_YEAR,
            total_stars: 2.0e10,
            max_depth: 12,
        };
        let mut g = Galaxy::new(params, 4096);
        let cam = g.node_center(NodeKey::new(6, 30, 30, 30));
        let view = ViewParams {
            mag_limit: 15.0,
            fade_width: 1.5,
            subdivide_angle: 0.2,
            max_depth: 12,
            max_nodes: 20_000,
        };
        let (stars, stats, nodes) = g.collect_visible(cam, &view);
        assert!(stats.nodes_visited > 0);
        assert!(stats.nodes_visited <= view.max_nodes);
        // The visited-node list must mirror the visit count (wireframe source).
        assert_eq!(nodes.len(), stats.nodes_visited);
        // Visible set is a subset of generated.
        assert!(stats.stars_visible <= stats.stars_generated);
        assert_eq!(stars.len(), stats.stars_visible);
    }

    #[test]
    fn cache_produces_hits_on_repeat() {
        let params = GalaxyParams {
            root_seed: 1,
            universe_size_m: 1.0e18,
            total_stars: 2.0e10,
            max_depth: 12,
        };
        let mut g = Galaxy::new(params, 8192);
        let cam = g.node_center(NodeKey::ROOT);
        // Bounded traversal so the whole visited set fits in the cache and is
        // stable between passes.
        let view = ViewParams {
            mag_limit: 12.0,
            fade_width: 1.5,
            subdivide_angle: 0.5,
            max_depth: 6,
            max_nodes: 2_000,
        };
        let (_, s1, _) = g.collect_visible(cam, &view);
        let (_, s2, _) = g.collect_visible(cam, &view);
        assert!(s1.cache_misses > 0);
        assert!(s2.cache_hits > 0, "second pass should hit the cache");
        assert!(
            s2.cache_misses == 0,
            "second identical pass should be fully cached"
        );
    }

    #[test]
    fn near_dist_is_zero_inside_and_positive_outside() {
        let params = GalaxyParams {
            root_seed: 3,
            universe_size_m: 1.0e18,
            total_stars: 2.0e10,
            max_depth: 12,
        };
        let g = Galaxy::new(params, 16);
        let key = NodeKey::new(3, 2, 2, 2);
        // Camera at the node center -> inside -> distance 0.
        let inside = g.node_center(key);
        assert_eq!(g.node_near_dist(key, inside), 0.0);
        // Camera far outside -> positive distance.
        let far = GalacticCoord::from_meters_f64(DVec3::splat(1.0e17));
        assert!(g.node_near_dist(key, far) > 0.0);
    }

    #[test]
    fn shallow_bright_stars_are_stable_across_subdivide_angle() {
        // The whole point of best-first traversal: the brightest (shallow)
        // stars must not be dropped or swapped when subdivide_angle changes,
        // because shallow nodes always have the highest angular priority.
        let params = GalaxyParams {
            root_seed: 42,
            universe_size_m: 1.0e18,
            total_stars: 2.0e10,
            max_depth: 14,
        };
        let cam = GalacticCoord::from_meters_f64(DVec3::splat(1.0e16));

        let collect_shallow = |angle: f64| -> Vec<[i128; 3]> {
            let mut g = Galaxy::new(params, 200_000);
            let view = ViewParams {
                mag_limit: 20.0,
                fade_width: 1.5,
                subdivide_angle: angle,
                max_depth: 14,
                // Deliberately small budget to force truncation of the deep,
                // unimportant tail.
                max_nodes: 1_500,
            };
            let (stars, _, _) = g.collect_visible(cam, &view);
            // Keep only intrinsically bright stars (shallow bands: abs_mag<0).
            let mut ids: Vec<[i128; 3]> = stars
                .iter()
                .filter(|v| v.star.abs_mag < 0.0)
                .map(|v| [v.star.pos.x.raw(), v.star.pos.y.raw(), v.star.pos.z.raw()])
                .collect();
            ids.sort();
            ids
        };

        let a = collect_shallow(0.1);
        let b = collect_shallow(0.4);
        // The bright set contained in both must be identical: neither truncation
        // (best-first keeps shallow nodes) nor angle change should teleport
        // bright stars.
        assert!(!a.is_empty(), "expected some bright stars");
        assert_eq!(
            a, b,
            "bright shallow stars changed with subdivide_angle (traversal not stable)"
        );
    }

    #[test]
    fn collect_visible_is_fully_deterministic() {
        // Identical camera + view must yield a byte-identical visible set, even
        // when the node budget truncates the tail. The Ord tie-break by NodeKey
        // guarantees a reproducible truncation regardless of heap history.
        let params = GalaxyParams {
            root_seed: 99,
            universe_size_m: 1.0e18,
            total_stars: 2.0e10,
            max_depth: 14,
        };
        let cam = GalacticCoord::from_meters_f64(DVec3::new(3.1e15, -2.2e15, 1.7e15));
        let view = ViewParams {
            mag_limit: 18.0,
            fade_width: 1.5,
            subdivide_angle: 0.13,
            max_depth: 14,
            max_nodes: 2_000, // force truncation
        };
        let run = || {
            let mut g = Galaxy::new(params, 300_000);
            let (stars, _, _) = g.collect_visible(cam, &view);
            stars
                .iter()
                .map(|v| {
                    (
                        [v.star.pos.x.raw(), v.star.pos.y.raw(), v.star.pos.z.raw()],
                        v.star.abs_mag.to_bits(),
                        v.fade.to_bits(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "collect_visible not deterministic");
    }
}
