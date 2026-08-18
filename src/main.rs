//! Desktop entry point. The web build uses `lib::start` instead (see lib.rs).

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("info,wgpu_core=error,wgpu_hal=error,naga=warn"),
    )
    .init();
    u4::run_native()
}

// On wasm the binary target is unused; the cdylib `start` export is the entry.
#[cfg(target_arch = "wasm32")]
fn main() {}
