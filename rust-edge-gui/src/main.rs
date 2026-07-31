use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use eframe::egui;
use image::{DynamicImage, GrayImage, Rgba, RgbaImage};
use imageproc::{edges::canny, filter::gaussian_blur_f32};

const MAX_CANNY_THRESHOLD: f32 = 1140.0;
const MIN_PREVIEW_SCALE: f32 = 0.10;
const MAX_PREVIEW_SCALE: f32 = 8.00;

fn main() -> eframe::Result {
    let initial_path = std::env::args_os().nth(1).map(PathBuf::from);

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([900.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "Rust Multi-layer Edge Viewer",
        native_options,
        Box::new(move |creation_context| {
            Ok(Box::new(EdgeApp::new(
                creation_context,
                initial_path.clone(),
            )))
        }),
    )
}

#[derive(Clone)]
struct EdgeLayer {
    id: u64,
    enabled: bool,
    low_threshold: f32,
    high_threshold: f32,
    color: egui::Color32,
    edge_pixels: usize,
}

impl EdgeLayer {
    fn new(id: u64, low_threshold: f32, high_threshold: f32, color: egui::Color32) -> Self {
        Self {
            id,
            enabled: true,
            low_threshold,
            high_threshold,
            color,
            edge_pixels: 0,
        }
    }
}

struct EdgeApp {
    image_path: Option<PathBuf>,
    original_rgba: Option<RgbaImage>,
    original_gray: Option<GrayImage>,
    composite_edges: Option<RgbaImage>,
    original_texture: Option<egui::TextureHandle>,
    edge_texture: Option<egui::TextureHandle>,

    layers: Vec<EdgeLayer>,
    next_layer_id: u64,
    preblur_sigma: f32,
    white_background: bool,
    update_while_dragging: bool,
    dirty: bool,

    original_preview_scale: f32,
    edge_preview_scale: f32,

    composite_edge_pixels: usize,
    status: String,
    error: Option<String>,
}

impl EdgeApp {
    fn new(cc: &eframe::CreationContext<'_>, initial_path: Option<PathBuf>) -> Self {
        let mut app = Self {
            image_path: None,
            original_rgba: None,
            original_gray: None,
            composite_edges: None,
            original_texture: None,
            edge_texture: None,
            layers: vec![EdgeLayer::new(
                1,
                50.0,
                150.0,
                egui::Color32::from_rgb(0, 220, 255),
            )],
            next_layer_id: 2,
            preblur_sigma: 0.0,
            white_background: false,
            update_while_dragging: true,
            dirty: false,
            original_preview_scale: 1.0,
            edge_preview_scale: 1.0,
            composite_edge_pixels: 0,
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

    fn add_layer(&mut self) {
        let (low, high) = self
            .layers
            .last()
            .map(|layer| (layer.low_threshold, layer.high_threshold))
            .unwrap_or((50.0, 150.0));

        let color = layer_palette(self.layers.len());
        let id = self.next_layer_id;
        self.next_layer_id += 1;
        self.layers.push(EdgeLayer::new(id, low, high, color));
        self.dirty = true;
    }

    fn recompute_edges(&mut self, ctx: &egui::Context) {
        let Some(gray) = self.original_gray.as_ref() else {
            return;
        };

        let blurred;
        let input = if self.preblur_sigma > 0.01 {
            blurred = gaussian_blur_f32(gray, self.preblur_sigma);
            &blurred
        } else {
            gray
        };

        let width = input.width();
        let height = input.height();
        let background = if self.white_background {
            Rgba([255, 255, 255, 255])
        } else {
            Rgba([0, 0, 0, 255])
        };

        let mut composite = RgbaImage::from_pixel(width, height, background);
        let mut covered = vec![false; width as usize * height as usize];

        for layer in &mut self.layers {
            layer.edge_pixels = 0;
            if !layer.enabled {
                continue;
            }

            let low = layer.low_threshold.min(layer.high_threshold);
            let high = layer.high_threshold.max(layer.low_threshold);
            let mask = canny(input, low, high);
            let layer_color = layer.color.to_srgba_unmultiplied();

            for (index, (mask_pixel, output_pixel)) in
                mask.pixels().zip(composite.pixels_mut()).enumerate()
            {
                if mask_pixel.0[0] == 0 {
                    continue;
                }

                layer.edge_pixels += 1;
                covered[index] = true;
                blend_srgba_over(output_pixel, layer_color);
            }
        }

        self.composite_edge_pixels = covered.into_iter().filter(|covered| *covered).count();

        let color_image = rgba_to_color_image(&composite);
        match self.edge_texture.as_mut() {
            Some(texture) => texture.set(color_image, egui::TextureOptions::NEAREST),
            None => {
                self.edge_texture = Some(ctx.load_texture(
                    "edge-composite",
                    color_image,
                    egui::TextureOptions::NEAREST,
                ));
            }
        }

        let total_pixels = width as usize * height as usize;
        let percentage = if total_pixels == 0 {
            0.0
        } else {
            self.composite_edge_pixels as f64 * 100.0 / total_pixels as f64
        };
        let enabled_layers = self.layers.iter().filter(|layer| layer.enabled).count();

        self.status = format!(
            "Rendered {enabled_layers} layer(s): {} unique edge pixels ({percentage:.2}%)",
            self.composite_edge_pixels
        );
        self.composite_edges = Some(composite);
        self.dirty = false;
        self.error = None;
    }

    fn save_edges(&mut self) {
        let Some(edges) = self.composite_edges.as_ref() else {
            self.error = Some("There is no processed image to save.".to_owned());
            return;
        };

        let suggested_name = self
            .image_path
            .as_deref()
            .and_then(Path::file_stem)
            .and_then(|name| name.to_str())
            .map(|name| format!("{name}_edge_layers.png"))
            .unwrap_or_else(|| "edge_layers.png".to_owned());

        let Some(path) = rfd::FileDialog::new()
            .set_title("Save layered edge image")
            .set_file_name(suggested_name)
            .add_filter("PNG image", &["png"])
            .save_file()
        else {
            return;
        };

        let path = ensure_png_extension(path);
        match DynamicImage::ImageRgba8(edges.clone()).save(&path) {
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
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.heading("Edge compositor");
                ui.add_space(4.0);

                if ui.button("Open image…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_title("Open image")
                        .add_filter(
                            "Image",
                            &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
                        )
                        .pick_file()
                    {
                        self.load_image(&path, ctx);
                    }
                }

                if ui
                    .add_enabled(
                        self.composite_edges.is_some(),
                        egui::Button::new("Save layered edges…"),
                    )
                    .clicked()
                {
                    self.save_edges();
                }

                ui.separator();
                ui.strong("Preview scale");

                ui.horizontal(|ui| {
                    ui.label("Original");
                    ui.add(
                        egui::Slider::new(
                            &mut self.original_preview_scale,
                            MIN_PREVIEW_SCALE..=MAX_PREVIEW_SCALE,
                        )
                        .logarithmic(true)
                        .suffix("×"),
                    );
                    if ui.small_button("Reset").clicked() {
                        self.original_preview_scale = 1.0;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Edges");
                    ui.add(
                        egui::Slider::new(
                            &mut self.edge_preview_scale,
                            MIN_PREVIEW_SCALE..=MAX_PREVIEW_SCALE,
                        )
                        .logarithmic(true)
                        .suffix("×"),
                    );
                    if ui.small_button("Reset").clicked() {
                        self.edge_preview_scale = 1.0;
                    }
                });

                if ui.button("Use same scale for both").clicked() {
                    self.edge_preview_scale = self.original_preview_scale;
                }

                ui.small("1× fits a large image to its preview column; larger values enable scrolling.");

                ui.separator();
                ui.strong("Shared processing");

                let mut parameters_changed = ui
                    .add(
                        egui::Slider::new(&mut self.preblur_sigma, 0.0..=10.0)
                            .text("Pre-blur σ")
                            .fixed_decimals(2),
                    )
                    .changed();

                parameters_changed |= ui
                    .checkbox(&mut self.white_background, "White background")
                    .changed();

                ui.checkbox(
                    &mut self.update_while_dragging,
                    "Update while dragging sliders",
                );

                ui.separator();
                ui.horizontal(|ui| {
                    ui.strong(format!("Layers ({})", self.layers.len()));
                    if ui.button("＋ Add layer").clicked() {
                        self.add_layer();
                    }
                });

                let mut layer_to_remove = None;

                for (index, layer) in self.layers.iter_mut().enumerate() {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            parameters_changed |= ui.checkbox(&mut layer.enabled, "").changed();
                            ui.strong(format!("Layer {}", index + 1));
                            ui.add_space(4.0);
                            parameters_changed |= ui.color_edit_button_srgba(&mut layer.color).changed();

                            if ui.small_button("Delete").clicked() {
                                layer_to_remove = Some(layer.id);
                            }
                        });

                        parameters_changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut layer.low_threshold,
                                    0.0..=layer.high_threshold.max(0.0),
                                )
                                .text("Low")
                                .clamping(egui::SliderClamping::Always),
                            )
                            .changed();

                        parameters_changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut layer.high_threshold,
                                    layer.low_threshold..=MAX_CANNY_THRESHOLD,
                                )
                                .text("High")
                                .clamping(egui::SliderClamping::Always),
                            )
                            .changed();

                        ui.small(format!("Detected pixels: {}", layer.edge_pixels));
                    });
                    ui.add_space(4.0);
                }

                if let Some(id) = layer_to_remove {
                    self.layers.retain(|layer| layer.id != id);
                    parameters_changed = true;
                }

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
                    ui.label(format!(
                        "Unique composite edge pixels: {}",
                        self.composite_edge_pixels
                    ));
                }

                ui.separator();
                ui.small(
                    "Each enabled layer runs Canny with its own thresholds. Edge colors are composited in layer order; later layers are drawn over earlier ones.",
                );
                ui.small("You can also drag and drop an image into the window.");
            });
    }

    fn previews(&self, ui: &mut egui::Ui) {
        ui.columns(2, |columns| {
            preview_panel(
                &mut columns[0],
                "Original",
                self.original_texture.as_ref(),
                self.original_preview_scale,
            );
            preview_panel(
                &mut columns[1],
                "Layered edges",
                self.edge_texture.as_ref(),
                self.edge_preview_scale,
            );
        });
    }
}

impl eframe::App for EdgeApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
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
            .resizable(true)
            .default_size(360.0)
            .min_size(300.0)
            .max_size(560.0)
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

fn blend_srgba_over(destination: &mut Rgba<u8>, source: [u8; 4]) {
    let source_alpha = source[3] as f32 / 255.0;
    if source_alpha <= 0.0 {
        return;
    }

    let destination_alpha = destination.0[3] as f32 / 255.0;
    let output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha);

    if output_alpha <= f32::EPSILON {
        destination.0 = [0, 0, 0, 0];
        return;
    }

    for channel in 0..3 {
        let source_value = source[channel] as f32 / 255.0;
        let destination_value = destination.0[channel] as f32 / 255.0;
        let output_value = (source_value * source_alpha
            + destination_value * destination_alpha * (1.0 - source_alpha))
            / output_alpha;
        destination.0[channel] = (output_value * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    destination.0[3] = (output_alpha * 255.0).round().clamp(0.0, 255.0) as u8;
}

fn layer_palette(index: usize) -> egui::Color32 {
    const COLORS: [[u8; 3]; 8] = [
        [0, 220, 255],
        [255, 70, 170],
        [255, 210, 40],
        [110, 255, 100],
        [170, 100, 255],
        [255, 120, 40],
        [80, 150, 255],
        [255, 255, 255],
    ];

    let [r, g, b] = COLORS[index % COLORS.len()];
    egui::Color32::from_rgb(r, g, b)
}

fn preview_panel(
    ui: &mut egui::Ui,
    title: &str,
    texture: Option<&egui::TextureHandle>,
    preview_scale: f32,
) {
    ui.horizontal(|ui| {
        ui.heading(title);
        ui.weak(format!("{preview_scale:.2}×"));
    });
    ui.separator();

    let Some(texture) = texture else {
        ui.centered_and_justified(|ui| {
            ui.label("No image loaded");
        });
        return;
    };

    let available_width = ui.available_width().max(1.0);

    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let texture_size = texture.size_vec2();
            let fit_scale = (available_width / texture_size.x).min(1.0);
            let display_size = texture_size * fit_scale * preview_scale;

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
