//! Build helper tasks for U4.
//!
//! `cargo xtask build-wasm` compiles the `u4` cdylib to `wasm32-unknown-unknown`,
//! runs `wasm-bindgen` to emit the JS glue, and drops everything (plus the
//! `index.html`) into the top-level `wasm/` directory ready to be served by any
//! static HTTP server.

use std::env;
use std::path::{Path, PathBuf};

use xshell::{cmd, Shell};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("help");

    match command {
        "build-wasm" => build_wasm(args.iter().any(|a| a == "--release"))?,
        "serve" => serve()?,
        _ => print_help(),
    }

    Ok(())
}

fn build_wasm(release: bool) -> anyhow::Result<()> {
    let sh = Shell::new()?;
    let root = project_root();
    sh.change_dir(&root);

    let profile = if release { "release" } else { "debug" };
    println!("Building u4 cdylib for wasm32 ({profile})...");
    if release {
        cmd!(
            sh,
            "cargo build -p u4 --lib --target wasm32-unknown-unknown --release"
        )
        .run()?;
    } else {
        cmd!(sh, "cargo build -p u4 --lib --target wasm32-unknown-unknown").run()?;
    }

    let wasm_path = format!("target/wasm32-unknown-unknown/{profile}/u4.wasm");
    let dist_dir = "wasm";

    println!("Running wasm-bindgen...");
    if cmd!(sh, "wasm-bindgen --version").quiet().run().is_err() {
        anyhow::bail!(
            "wasm-bindgen-cli is not installed or not on PATH.\n\
             Install a version matching the `wasm-bindgen` crate in Cargo.lock:\n\
             cargo install wasm-bindgen-cli --version <X.Y.Z>"
        );
    }

    cmd!(
        sh,
        "wasm-bindgen --target web --out-dir {dist_dir} --no-typescript {wasm_path}"
    )
    .run()?;

    println!("\nSuccess! WASM output is in `{dist_dir}/`.");
    println!("Serve it with a static server, e.g.:  cargo xtask serve");
    Ok(())
}

fn serve() -> anyhow::Result<()> {
    let sh = Shell::new()?;
    sh.change_dir(project_root().join("wasm"));
    println!("Serving ./wasm on http://localhost:8080 (Ctrl+C to stop)");
    // Prefer python if available; it ships almost everywhere.
    if cmd!(sh, "python --version").quiet().run().is_ok() {
        cmd!(sh, "python -m http.server 8080").run()?;
    } else {
        anyhow::bail!("python not found; run any static file server in ./wasm");
    }
    Ok(())
}

fn print_help() {
    println!("U4 xtask commands:");
    println!("  build-wasm [--release]  Compile to wasm and run wasm-bindgen into ./wasm");
    println!("  serve                   Serve ./wasm on http://localhost:8080");
}

fn project_root() -> PathBuf {
    Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()))
        .ancestors()
        .nth(1)
        .unwrap()
        .to_path_buf()
}
