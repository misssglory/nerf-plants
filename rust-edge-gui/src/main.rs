use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use eframe::egui;
use image::{DynamicImage, GrayImage, RgbaImage};
use imageproc::{edges::canny, filter::gaussian_blur_f32};

const MAX_CANNY_THRESHOLD: f32 = 1140.0;

fn main() -> eframe::Result {
    let initial_path = std::env::args_os().nth(1).map(PathBuf::from);

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([820.0, 560.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "Rust Canny Edge Viewer",
        native_options,
        Box::new(move |creation_context| {
            Ok(Box::new(EdgeApp::new(
                creation_context,
                initial_path.clone(),
            )))
        }),
    )
}

struct EdgeApp {
    image_path: Option<PathBuf>,
    original_rgba: Option<RgbaImage>,
    original_gray: Option<GrayImage>,
    edges: Option<GrayImage>,
    original_texture: Option<egui::TextureHandle>,
    edge_texture: Option<egui::TextureHandle>,

    low_threshold: f32,
    high_threshold: f32,
    preblur_sigma: f32,
    invert_edges: bool,
    update_while_dragging: bool,
    dirty: bool,

    edge_pixels: usize,
    status: String,
    error: Option<String>,
}

impl EdgeApp {
    fn new(cc: &eframe::CreationContext<'_>, initial_path: Option<PathBuf>) -> Self {
        let mut app = Self {
            image_path: None,
            original_rgba: None,
            original_gray: None,
            edges: None,
            original_texture: None,
            edge_texture: None,
            low_threshold: 50.0,
            high_threshold: 150.0,
            preblur_sigma: 0.0,
            invert_edges: false,
            update_while_dragging: true,
            dirty: false,
            edge_pixels: 0,
            status: "Open an image or drop one into the window.".to_owned(),
            error: None,
        };

        if let Some(path) = initial_path {
            app.load_image(&path, &cc.egui_ctx);
        }

        app
    }

    fn load_image(&mut self, path: &Path, ctx: &egui::Context) {
        match load_image_data(path) {
            Ok((rgba, gray)) => {
                let original_color = rgba_to_color_image(&rgba);
                self.original_texture = Some(ctx.load_texture(
                    "original-image",
                    original_color,
                    egui::TextureOptions::LINEAR,
                ));

                self.image_path = Some(path.to_owned());
                self.original_rgba = Some(rgba);
                self.original_gray = Some(gray);
                self.error = None;
                self.status = format!("Loaded {}", path.display());
                self.dirty = true;
                self.recompute_edges(ctx);
            }
            Err(error) => {
                self.error = Some(format!("Failed to load {}: {error:#}", path.display()));
            }
        }
    }

    fn recompute_edges(&mut self, ctx: &egui::Context) {
        let Some(gray) = self.original_gray.as_ref() else {
            return;
        };

        let low = self.low_threshold.min(self.high_threshold);
        let high = self.high_threshold.max(self.low_threshold);

        let blurred;
        let input = if self.preblur_sigma > 0.01 {
            blurred = gaussian_blur_f32(gray, self.preblur_sigma);
            &blurred
        } else {
            gray
        };

        let mut edges = canny(input, low, high);

        if self.invert_edges {
            for pixel in edges.pixels_mut() {
                pixel.0[0] = 255 - pixel.0[0];
            }
        }

        self.edge_pixels = if self.invert_edges {
            edges.pixels().filter(|pixel| pixel.0[0] == 0).count()
        } else {
            edges.pixels().filter(|pixel| pixel.0[0] != 0).count()
        };

        let color_image = gray_to_color_image(&edges);
        match self.edge_texture.as_mut() {
            Some(texture) => texture.set(color_image, egui::TextureOptions::NEAREST),
            None => {
                self.edge_texture = Some(ctx.load_texture(
                    "edge-image",
                    color_image,
                    egui::TextureOptions::NEAREST,
                ));
            }
        }

        let total_pixels = edges.width() as usize * edges.height() as usize;
        let percentage = if total_pixels == 0 {
            0.0
        } else {
            self.edge_pixels as f64 * 100.0 / total_pixels as f64
        };

        self.status = format!(
            "Canny complete: {} edge pixels ({percentage:.2}%)",
            self.edge_pixels
        );
        self.edges = Some(edges);
        self.dirty = false;
    }

    fn save_edges(&mut self) {
        let Some(edges) = self.edges.as_ref() else {
            self.error = Some("There is no processed image to save.".to_owned());
            return;
        };

        let suggested_name = self
            .image_path
            .as_deref()
            .and_then(Path::file_stem)
            .and_then(|name| name.to_str())
            .map(|name| format!("{name}_edges.png"))
            .unwrap_or_else(|| "edges.png".to_owned());

        let Some(path) = rfd::FileDialog::new()
            .set_title("Save edge image")
            .set_file_name(suggested_name)
            .add_filter("PNG image", &["png"])
            .save_file()
        else {
            return;
        };

        let path = ensure_png_extension(path);
        match DynamicImage::ImageLuma8(edges.clone()).save(&path) {
            Ok(()) => {
                self.status = format!("Saved {}", path.display());
                self.error = None;
            }
            Err(error) => {
                self.error = Some(format!("Failed to save {}: {error}", path.display()));
            }
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped_paths = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect::<Vec<_>>()
        });

        if let Some(path) = dropped_paths.first() {
            self.load_image(path, ctx);
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Canny controls");
        ui.add_space(4.0);

        if ui.button("Open image…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .set_title("Open image")
                .add_filter("Image", &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
                .pick_file()
            {
                self.load_image(&path, ctx);
            }
        }

        if ui
            .add_enabled(self.edges.is_some(), egui::Button::new("Save edges…"))
            .clicked()
        {
            self.save_edges();
        }

        ui.separator();

        let mut parameters_changed = false;
        parameters_changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.low_threshold,
                    0.0..=self.high_threshold.max(0.0),
                )
                .text("Low threshold")
                .clamping(egui::SliderClamping::Always),
            )
            .changed();

        parameters_changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.high_threshold,
                    self.low_threshold..=MAX_CANNY_THRESHOLD,
                )
                .text("High threshold")
                .clamping(egui::SliderClamping::Always),
            )
            .changed();

        parameters_changed |= ui
            .add(
                egui::Slider::new(&mut self.preblur_sigma, 0.0..=10.0)
                    .text("Extra blur σ")
                    .fixed_decimals(2),
            )
            .changed();

        parameters_changed |= ui.checkbox(&mut self.invert_edges, "Invert output").changed();

        ui.checkbox(
            &mut self.update_while_dragging,
            "Update while dragging sliders",
        );

        if parameters_changed {
            self.dirty = true;
        }

        let pointer_down = ui.input(|input| input.pointer.primary_down());
        if self.dirty && (self.update_while_dragging || !pointer_down) {
            self.recompute_edges(ctx);
        }

        if ui
            .add_enabled(self.dirty, egui::Button::new("Apply parameters"))
            .clicked()
        {
            self.recompute_edges(ctx);
        }

        ui.separator();

        if let Some(path) = self.image_path.as_ref() {
            ui.label("Input");
            ui.monospace(path.display().to_string());
        }

        if let Some(gray) = self.original_gray.as_ref() {
            ui.label(format!("Resolution: {} × {}", gray.width(), gray.height()));
            ui.label(format!("Edge pixels: {}", self.edge_pixels));
        }

        ui.separator();
        ui.small(
            "Tip: lower thresholds detect more weak detail. Extra blur suppresses leaf texture and image noise before Canny.",
        );
        ui.small("You can also drag and drop an image into the window.");
    }

    fn previews(&self, ui: &mut egui::Ui) {
        ui.columns(2, |columns| {
            preview_panel(&mut columns[0], "Original", self.original_texture.as_ref());
            preview_panel(&mut columns[1], "Edges", self.edge_texture.as_ref());
        });
    }
}

impl eframe::App for EdgeApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // egui 0.35 moved the main application entry point from Context to Ui.
        // Clone the cheap context handle so image loading and texture updates can
        // still use it without borrowing the root Ui for the whole frame.
        let ctx = ui.ctx().clone();

        self.handle_dropped_files(&ctx);

        egui::Panel::bottom("status-bar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if let Some(error) = self.error.as_ref() {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                } else {
                    ui.label(&self.status);
                }
            });
        });

        egui::Panel::left("controls")
            .resizable(false)
            .default_size(260.0)
            .show(ui, |ui| {
                self.controls(ui, &ctx);
            });

        egui::CentralPanel::default().show(ui, |ui| {
            self.previews(ui);
        });
    }
}

fn load_image_data(path: &Path) -> Result<(RgbaImage, GrayImage)> {
    let decoded = image::open(path)
        .with_context(|| format!("unable to decode image {}", path.display()))?;
    Ok((decoded.to_rgba8(), decoded.to_luma8()))
}

fn rgba_to_color_image(image: &RgbaImage) -> egui::ColorImage {
    egui::ColorImage::from_rgba_unmultiplied(
        [image.width() as usize, image.height() as usize],
        image.as_raw(),
    )
}

fn gray_to_color_image(image: &GrayImage) -> egui::ColorImage {
    egui::ColorImage::from_gray(
        [image.width() as usize, image.height() as usize],
        image.as_raw(),
    )
}

fn preview_panel(
    ui: &mut egui::Ui,
    title: &str,
    texture: Option<&egui::TextureHandle>,
) {
    ui.heading(title);
    ui.separator();

    let Some(texture) = texture else {
        ui.centered_and_justified(|ui| {
            ui.label("No image loaded");
        });
        return;
    };

    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let texture_size = texture.size_vec2();
            let available_width = ui.available_width().max(1.0);
            let scale = (available_width / texture_size.x).min(1.0);
            let display_size = texture_size * scale;

            ui.add(
                egui::Image::new(texture)
                    .fit_to_exact_size(display_size)
                    .maintain_aspect_ratio(true),
            );
        });
}

fn ensure_png_extension(mut path: PathBuf) -> PathBuf {
    if path.extension().is_none() {
        path.set_extension("png");
    }
    path
}
