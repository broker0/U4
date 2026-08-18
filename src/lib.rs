//! U4 — full-scale galaxy viewer.
//!
//! Built on eframe (egui) with a custom wgpu star renderer embedded via a paint
//! callback. The same [`GalaxyApp`] runs natively and on the web; the only
//! difference is the bootstrap (`run_native` vs `WebRunner`).

pub mod app;

pub use app::GalaxyApp;

/// Native options shared by the desktop entry point.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("U4 — galaxy")
            .with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "U4 — galaxy",
        native_options,
        Box::new(|cc| {
            GalaxyApp::new(cc)
                .map(|a| Box::new(a) as Box<dyn eframe::App>)
                .ok_or_else(|| "wgpu render state unavailable".into())
        }),
    )
}

/// Web entry point, called from `index.html` after the wasm module loads.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), wasm_bindgen::JsValue> {
    use eframe::wasm_bindgen::JsCast as _;

    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    let document = web_sys::window()
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("no window"))?
        .document()
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("no document"))?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("canvas element not found"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| wasm_bindgen::JsValue::from_str("element is not a canvas"))?;

    let web_options = eframe::WebOptions::default();
    eframe::WebRunner::new()
        .start(
            canvas,
            web_options,
            Box::new(|cc| {
                GalaxyApp::new(cc)
                    .map(|a| Box::new(a) as Box<dyn eframe::App>)
                    .ok_or_else(|| "wgpu render state unavailable".into())
            }),
        )
        .await
}
