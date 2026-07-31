use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use eframe::egui;
use image::{DynamicImage, GrayImage, Rgba, RgbaImage};

const MAX_CANNY_THRESHOLD: f32 = 1140.0;
const MIN_SCALE: f32 = 0.10;
const MAX_SCALE: f32 = 8.0;

fn main() -> eframe::Result {
    let initial_path = std::env::args_os().nth(1).map(PathBuf::from);

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1480.0, 940.0])
            .with_min_inner_size([980.0, 700.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "Rust Green Shape + Edge Composer",
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
    enabled: bool,
    name: String,
    low_threshold: f32,
    high_threshold: f32,
    reduction_strength: f32,
    reduction_radius: u32,
    color: egui::Color32,
    edge_pixels: usize,
}

impl EdgeLayer {
    fn new(index: usize) -> Self {
        let palette = [
            egui::Color32::from_rgb(0, 255, 255),
            egui::Color32::from_rgb(255, 0, 255),
            egui::Color32::from_rgb(255, 210, 0),
            egui::Color32::from_rgb(0, 220, 120),
            egui::Color32::from_rgb(255, 120, 80),
            egui::Color32::from_rgb(130, 180, 255),
        ];
        let color = palette[index % palette.len()];
        Self {
            enabled: true,
            name: format!("Edge layer {}", index + 1),
            low_threshold: 50.0 + (index as f32 * 20.0),
            high_threshold: 150.0 + (index as f32 * 40.0),
            reduction_strength: 0.35,
            reduction_radius: 30,
            color,
            edge_pixels: 0,
        }
    }
}

#[derive(Clone)]
struct GreenShapeLayer {
    enabled: bool,
    name: String,
    green_excess_threshold: f32,
    green_ratio_threshold: f32,
    mask_grow_radius: u32,
    color: egui::Color32,
    fill_alpha: u8,
    show_holes: bool,
}

impl Default for GreenShapeLayer {
    fn default() -> Self {
        Self {
            enabled: true,
            name: "Green closed shape".to_owned(),
            green_excess_threshold: 28.0,
            green_ratio_threshold: 0.38,
            mask_grow_radius: 1,
            color: egui::Color32::from_rgb(80, 255, 140),
            fill_alpha: 54,
            show_holes: true,
        }
    }
}

#[derive(Default, Clone)]
struct GreenShapeResult {
    width: u32,
    height: u32,
    found: bool,
    mask: Vec<bool>,
    boundary: Vec<bool>,
    hole_mask: Vec<bool>,
    hole_boundary: Vec<bool>,
    distance_map: Vec<f32>,
    area_pixels: usize,
    hole_pixels: usize,
    hole_count: usize,
    weighted_center: Option<(f32, f32)>,
}

impl GreenShapeResult {
    fn area_percent(&self) -> f32 {
        let total = (self.width as usize).saturating_mul(self.height as usize);
        if total == 0 {
            0.0
        } else {
            self.area_pixels as f32 * 100.0 / total as f32
        }
    }

    fn hole_percent(&self) -> f32 {
        let total = (self.width as usize).saturating_mul(self.height as usize);
        if total == 0 {
            0.0
        } else {
            self.hole_pixels as f32 * 100.0 / total as f32
        }
    }
}

struct EdgeApp {
    image_path: Option<PathBuf>,
    original_rgba: Option<RgbaImage>,
    original_gray: Option<GrayImage>,
    original_texture: Option<egui::TextureHandle>,
    composite_texture: Option<egui::TextureHandle>,

    shape_layer: GreenShapeLayer,
    green_shape: Option<GreenShapeResult>,
    edge_layers: Vec<EdgeLayer>,

    preblur_sigma: f32,
    dimness: f32,
    black_background: bool,
    update_while_dragging: bool,
    dirty: bool,

    original_scale: f32,
    composite_scale: f32,

    composite_rgba: Option<RgbaImage>,
    unique_edge_pixels: usize,
    status: String,
    error: Option<String>,
}

impl EdgeApp {
    fn new(cc: &eframe::CreationContext<'_>, initial_path: Option<PathBuf>) -> Self {
        configure_black_visuals(&cc.egui_ctx);

        let mut app = Self {
            image_path: None,
            original_rgba: None,
            original_gray: None,
            original_texture: None,
            composite_texture: None,
            shape_layer: GreenShapeLayer::default(),
            green_shape: None,
            edge_layers: vec![EdgeLayer::new(0), EdgeLayer::new(1)],
            preblur_sigma: 0.7,
            dimness: 0.55,
            black_background: true,
            update_while_dragging: true,
            dirty: false,
            original_scale: 1.0,
            composite_scale: 1.0,
            composite_rgba: None,
            unique_edge_pixels: 0,
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
                self.original_texture = Some(ctx.load_texture(
                    "original-image",
                    rgba_to_color_image(&rgba),
                    egui::TextureOptions::LINEAR,
                ));
                self.image_path = Some(path.to_owned());
                self.original_rgba = Some(rgba);
                self.original_gray = Some(gray);
                self.original_scale = 1.0;
                self.composite_scale = 1.0;
                self.error = None;
                self.status = format!("Loaded {}", path.display());
                self.dirty = true;
                self.recompute(ctx);
            }
            Err(error) => {
                self.error = Some(format!("Failed to load {}: {error:#}", path.display()));
            }
        }
    }

    fn add_layer(&mut self) {
        let index = self.edge_layers.len();
        self.edge_layers.push(EdgeLayer::new(index));
        self.dirty = true;
    }

    fn remove_layer(&mut self, index: usize) {
        if index < self.edge_layers.len() {
            self.edge_layers.remove(index);
            self.dirty = true;
        }
    }

    fn recompute(&mut self, ctx: &egui::Context) {
        let (Some(original), Some(gray)) = (self.original_rgba.as_ref(), self.original_gray.as_ref())
        else {
            return;
        };

        let width = gray.width();
        let height = gray.height();
        let intensities = grayscale_values(gray);
        let blurred = if self.preblur_sigma > 0.01 {
            gaussian_blur_naive(&intensities, width, height, self.preblur_sigma)
        } else {
            intensities.clone()
        };

        let shape = detect_green_shape(original, &self.shape_layer);
        self.green_shape = Some(shape.clone());

        let mut composite = dimmed_original(original, self.dimness, self.black_background);
        let mut union_mask = vec![false; (width * height) as usize];

        if self.shape_layer.enabled && shape.found {
            alpha_fill_mask(&mut composite, &shape.mask, self.shape_layer.color, self.shape_layer.fill_alpha);
            paint_mask(&mut composite, &shape.boundary, self.shape_layer.color);
            merge_union(&mut union_mask, &shape.boundary);

            if self.shape_layer.show_holes && shape.hole_count > 0 {
                let hole_color = brighten(self.shape_layer.color, 0.7);
                paint_mask(&mut composite, &shape.hole_boundary, hole_color);
                merge_union(&mut union_mask, &shape.hole_boundary);
            }

            if let Some((cx, cy)) = shape.weighted_center {
                draw_cross(&mut composite, cx.round() as i32, cy.round() as i32, 7, egui::Color32::WHITE);
            }
        }

        for layer in &mut self.edge_layers {
            layer.edge_pixels = 0;
            if !layer.enabled {
                continue;
            }

            let edge_mask = adaptive_edge_mask(
                &blurred,
                width,
                height,
                layer.low_threshold.min(layer.high_threshold),
                layer.high_threshold.max(layer.low_threshold),
                layer.reduction_strength,
                layer.reduction_radius,
                shape.distance_map.as_slice(),
            );

            layer.edge_pixels = edge_mask.iter().filter(|&&value| value).count();
            paint_mask(&mut composite, &edge_mask, layer.color);
            merge_union(&mut union_mask, &edge_mask);
        }

        self.unique_edge_pixels = union_mask.iter().filter(|&&v| v).count();
        self.composite_rgba = Some(composite.clone());

        let color_image = rgba_to_color_image(&composite);
        match self.composite_texture.as_mut() {
            Some(texture) => texture.set(color_image, egui::TextureOptions::LINEAR),
            None => {
                self.composite_texture = Some(ctx.load_texture(
                    "composite-image",
                    color_image,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }

        self.error = None;
        self.status = if let Some(shape) = self.green_shape.as_ref() {
            if shape.found {
                format!(
                    "Green shape: {} px ({:.2}%), holes: {} px in {} hole(s), total edge/outline pixels: {}",
                    shape.area_pixels,
                    shape.area_percent(),
                    shape.hole_pixels,
                    shape.hole_count,
                    self.unique_edge_pixels,
                )
            } else {
                format!(
                    "No green closed shape found. Overlay edge pixels: {}",
                    self.unique_edge_pixels
                )
            }
        } else {
            format!("Overlay edge pixels: {}", self.unique_edge_pixels)
        };
        self.dirty = false;
    }

    fn save_composite(&mut self) {
        let Some(composite) = self.composite_rgba.as_ref() else {
            self.error = Some("There is no processed image to save.".to_owned());
            return;
        };

        let suggested_name = self
            .image_path
            .as_deref()
            .and_then(Path::file_stem)
            .and_then(|name| name.to_str())
            .map(|name| format!("{name}_green_shape_edges.png"))
            .unwrap_or_else(|| "green_shape_edges.png".to_owned());

        let Some(path) = rfd::FileDialog::new()
            .set_title("Save composite image")
            .set_file_name(suggested_name)
            .add_filter("PNG image", &["png"])
            .save_file()
        else {
            return;
        };

        let path = ensure_png_extension(path);
        match DynamicImage::ImageRgba8(composite.clone()).save(&path) {
            Ok(()) => {
                self.error = None;
                self.status = format!("Saved {}", path.display());
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
        ui.heading("Green shape + edge composer");
        ui.add_space(6.0);

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
            .add_enabled(self.composite_rgba.is_some(), egui::Button::new("Save composite…"))
            .clicked()
        {
            self.save_composite();
        }

        if ui.button("Sync both scales").clicked() {
            let scale = self.original_scale.max(self.composite_scale).clamp(MIN_SCALE, MAX_SCALE);
            self.original_scale = scale;
            self.composite_scale = scale;
        }

        ui.separator();

        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.preblur_sigma, 0.0..=8.0)
                    .text("Shared blur σ")
                    .fixed_decimals(2),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.dimness, 0.0..=1.0)
                    .text("Edge preview dimness")
                    .fixed_decimals(2),
            )
            .changed();
        changed |= ui.checkbox(&mut self.black_background, "Black edge-preview background").changed();
        ui.checkbox(&mut self.update_while_dragging, "Update while dragging sliders");

        ui.separator();
        ui.collapsing("Preview scale", |ui| {
            ui.small("Use the mouse wheel while hovering a preview to zoom that image.");
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.original_scale, MIN_SCALE..=MAX_SCALE)
                        .text("Original scale")
                        .fixed_decimals(2),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.composite_scale, MIN_SCALE..=MAX_SCALE)
                        .text("Overlay scale")
                        .fixed_decimals(2),
                )
                .changed();
        });

        ui.separator();
        ui.collapsing("Layer 0 — closed green shape (locked role)", |ui| {
            changed |= ui.checkbox(&mut self.shape_layer.enabled, "Enabled").changed();
            ui.small("This first layer always finds the closed green leaf/object whose center is the weighted center of green pixels.");
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.shape_layer.green_excess_threshold, 0.0..=150.0)
                        .text("Green excess threshold")
                        .fixed_decimals(1),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.shape_layer.green_ratio_threshold, 0.0..=1.0)
                        .text("Green ratio threshold")
                        .fixed_decimals(2),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.shape_layer.mask_grow_radius, 0..=8)
                        .text("Mask grow radius"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.shape_layer.fill_alpha, 0..=180)
                        .text("Shape fill alpha"),
                )
                .changed();
            changed |= ui.checkbox(&mut self.shape_layer.show_holes, "Detect and show holes").changed();
            ui.horizontal(|ui| {
                ui.label("Shape color");
                changed |= ui.color_edit_button_srgba(&mut self.shape_layer.color).changed();
            });

            if let Some(shape) = self.green_shape.as_ref() {
                if shape.found {
                    ui.colored_label(
                        self.shape_layer.color,
                        format!(
                            "Area: {} px ({:.2}%) | Holes: {} px ({:.2}%) in {} hole(s)",
                            shape.area_pixels,
                            shape.area_percent(),
                            shape.hole_pixels,
                            shape.hole_percent(),
                            shape.hole_count
                        ),
                    );
                    if let Some((cx, cy)) = shape.weighted_center {
                        ui.label(format!("Weighted green center: ({cx:.1}, {cy:.1})"));
                    }
                } else {
                    ui.colored_label(egui::Color32::LIGHT_RED, "No closed green shape found.");
                }
            }
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.heading("Extra edge layers");
            if ui.button("Add layer").clicked() {
                self.add_layer();
                changed = true;
            }
        });
        ui.small("Each extra layer has its own color, thresholds, and optional threshold reduction near the green shape.");

        let mut remove_index = None;
        for (index, layer) in self.edge_layers.iter_mut().enumerate() {
            ui.separator();
            ui.collapsing(format!("{}", layer.name), |ui| {
                ui.horizontal(|ui| {
                    changed |= ui.checkbox(&mut layer.enabled, "Enabled").changed();
                    if ui.button("Delete layer").clicked() {
                        remove_index = Some(index);
                    }
                });
                changed |= ui
                    .add(
                        egui::Slider::new(&mut layer.low_threshold, 0.0..=MAX_CANNY_THRESHOLD)
                            .text("Low threshold")
                            .fixed_decimals(1),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut layer.high_threshold, 0.0..=MAX_CANNY_THRESHOLD)
                            .text("High threshold")
                            .fixed_decimals(1),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut layer.reduction_strength, 0.0..=0.95)
                            .text("Threshold reduction near green")
                            .fixed_decimals(2),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut layer.reduction_radius, 0..=100)
                            .text("Reduction radius (px)"),
                    )
                    .changed();
                ui.horizontal(|ui| {
                    ui.label("Layer color");
                    changed |= ui.color_edit_button_srgba(&mut layer.color).changed();
                });
                ui.colored_label(
                    layer.color,
                    format!("Active edge pixels: {}", layer.edge_pixels),
                );
            });
        }

        if let Some(index) = remove_index {
            self.remove_layer(index);
            changed = true;
        }

        if changed {
            self.dirty = true;
        }

        let pointer_down = ui.input(|input| input.pointer.primary_down());
        if self.dirty && (self.update_while_dragging || !pointer_down) {
            self.recompute(ctx);
        }

        if ui
            .add_enabled(self.dirty, egui::Button::new("Apply parameters"))
            .clicked()
        {
            self.recompute(ctx);
        }

        ui.separator();
        if let Some(path) = self.image_path.as_ref() {
            ui.label("Input file");
            ui.monospace(path.display().to_string());
        }
        if let Some(gray) = self.original_gray.as_ref() {
            ui.label(format!("Resolution: {} × {}", gray.width(), gray.height()));
        }
        ui.label(format!("Overlay outline pixels: {}", self.unique_edge_pixels));
        ui.small("Tip: if leaf edges are weak, increase the green reduction radius and reduction strength in the extra layers, or lower the green-shape thresholds to lock the main shape first.");
    }

    fn previews(&mut self, ui: &mut egui::Ui) {
        ui.columns(2, |columns| {
            show_zoomable_texture(
                &mut columns[0],
                "Original",
                self.original_texture.as_ref(),
                &mut self.original_scale,
            );
            show_zoomable_texture(
                &mut columns[1],
                "Overlay / edges",
                self.composite_texture.as_ref(),
                &mut self.composite_scale,
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
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.controls(ui, &ctx);
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            self.previews(ui);
        });
    }
}

fn configure_black_visuals(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);

    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = egui::Color32::BLACK;
    style.visuals.window_fill = egui::Color32::BLACK;
    style.visuals.extreme_bg_color = egui::Color32::BLACK;
    style.visuals.code_bg_color = egui::Color32::from_rgb(10, 10, 10);
    style.visuals.faint_bg_color = egui::Color32::from_rgb(12, 12, 12);
    ctx.set_style_of(egui::Theme::Dark, style);
}

fn show_zoomable_texture(
    ui: &mut egui::Ui,
    title: &str,
    texture: Option<&egui::TextureHandle>,
    scale: &mut f32,
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
            let display_size = texture_size * *scale;
            let response = ui.add(
                egui::Image::new(texture)
                    .fit_to_exact_size(display_size)
                    .maintain_aspect_ratio(true),
            );

            if response.hovered() {
                let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                if scroll.abs() > f32::EPSILON {
                    let factor: f32 = if scroll > 0.0 { 1.10 } else { 1.0 / 1.10 };
                    let steps = (scroll.abs() / 32.0).max(1.0);
                    *scale = (*scale * factor.powf(steps)).clamp(MIN_SCALE, MAX_SCALE);
                }
            }
        });

    ui.small(format!("Scale: {:.2}×", *scale));
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

fn ensure_png_extension(mut path: PathBuf) -> PathBuf {
    if path.extension().is_none() {
        path.set_extension("png");
    }
    path
}

fn grayscale_values(gray: &GrayImage) -> Vec<f32> {
    gray.pixels().map(|pixel| pixel.0[0] as f32).collect()
}

fn gaussian_blur_naive(input: &[f32], width: u32, height: u32, sigma: f32) -> Vec<f32> {
    if sigma <= 0.01 {
        return input.to_vec();
    }

    let radius = (sigma * 3.0).ceil() as i32;
    let kernel = gaussian_kernel_1d(radius, sigma);
    let w = width as usize;
    let h = height as usize;

    let mut temp = vec![0.0; input.len()];
    let mut out = vec![0.0; input.len()];

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in -radius..=radius {
                let xx = (x as i32 + k).clamp(0, width as i32 - 1) as usize;
                sum += input[y * w + xx] * kernel[(k + radius) as usize];
            }
            temp[y * w + x] = sum;
        }
    }

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in -radius..=radius {
                let yy = (y as i32 + k).clamp(0, height as i32 - 1) as usize;
                sum += temp[yy * w + x] * kernel[(k + radius) as usize];
            }
            out[y * w + x] = sum;
        }
    }

    out
}

fn gaussian_kernel_1d(radius: i32, sigma: f32) -> Vec<f32> {
    let sigma2 = 2.0 * sigma * sigma;
    let mut kernel = Vec::with_capacity((radius * 2 + 1) as usize);
    let mut sum = 0.0;
    for x in -radius..=radius {
        let value = (-(x * x) as f32 / sigma2).exp();
        kernel.push(value);
        sum += value;
    }
    if sum > 0.0 {
        for value in &mut kernel {
            *value /= sum;
        }
    }
    kernel
}

fn adaptive_edge_mask(
    image: &[f32],
    width: u32,
    height: u32,
    low_threshold: f32,
    high_threshold: f32,
    reduction_strength: f32,
    reduction_radius: u32,
    green_distances: &[f32],
) -> Vec<bool> {
    let w = width as usize;
    let h = height as usize;
    let mut strong = vec![false; w * h];
    let mut weak = vec![false; w * h];
    let mut magnitude = vec![0.0; w * h];

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let idx = y * w + x;
            let gx = -image[(y - 1) * w + (x - 1)]
                + image[(y - 1) * w + (x + 1)]
                - 2.0 * image[y * w + (x - 1)]
                + 2.0 * image[y * w + (x + 1)]
                - image[(y + 1) * w + (x - 1)]
                + image[(y + 1) * w + (x + 1)];
            let gy = image[(y - 1) * w + (x - 1)]
                + 2.0 * image[(y - 1) * w + x]
                + image[(y - 1) * w + (x + 1)]
                - image[(y + 1) * w + (x - 1)]
                - 2.0 * image[(y + 1) * w + x]
                - image[(y + 1) * w + (x + 1)];
            let mag = (gx * gx + gy * gy).sqrt();
            magnitude[idx] = mag;

            let proximity = if reduction_radius == 0 || green_distances.len() != magnitude.len() {
                0.0
            } else {
                let distance = green_distances[idx];
                if !distance.is_finite() || distance > reduction_radius as f32 {
                    0.0
                } else {
                    1.0 - (distance / reduction_radius as f32).clamp(0.0, 1.0)
                }
            };
            let reduction = 1.0 - reduction_strength.clamp(0.0, 0.95) * proximity;
            let local_low = low_threshold * reduction;
            let local_high = high_threshold * reduction;

            if mag >= local_high {
                strong[idx] = true;
            } else if mag >= local_low {
                weak[idx] = true;
            }
        }
    }

    let mut final_edges = vec![false; w * h];
    let mut queue = VecDeque::new();

    for (idx, &is_strong) in strong.iter().enumerate() {
        if is_strong {
            final_edges[idx] = true;
            queue.push_back(idx);
        }
    }

    while let Some(idx) = queue.pop_front() {
        let x = idx % w;
        let y = idx / w;
        for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                let nidx = ny * w + nx;
                if !final_edges[nidx] && weak[nidx] {
                    final_edges[nidx] = true;
                    queue.push_back(nidx);
                }
            }
        }
    }

    final_edges
}

fn detect_green_shape(image: &RgbaImage, settings: &GreenShapeLayer) -> GreenShapeResult {
    let width = image.width();
    let height = image.height();
    let total = (width * height) as usize;

    let mut raw_mask = vec![false; total];
    let mut weighted_sum = 0.0f32;
    let mut weighted_x = 0.0f32;
    let mut weighted_y = 0.0f32;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let pixel = image.get_pixel(x, y).0;
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let green_excess = (g - r.max(b)).max(0.0);
            let green_ratio = g / (r + g + b + 1.0);
            let weight = green_excess * green_ratio.max(0.0);
            if weight > 0.0 {
                weighted_sum += weight;
                weighted_x += x as f32 * weight;
                weighted_y += y as f32 * weight;
            }

            if green_excess >= settings.green_excess_threshold
                && green_ratio >= settings.green_ratio_threshold
            {
                raw_mask[idx] = true;
            }
        }
    }

    if settings.mask_grow_radius > 0 {
        raw_mask = dilate_mask(&raw_mask, width, height, settings.mask_grow_radius);
    }

    let weighted_center = if weighted_sum > 0.0 {
        Some((weighted_x / weighted_sum, weighted_y / weighted_sum))
    } else {
        None
    };

    let mut result = GreenShapeResult {
        width,
        height,
        weighted_center,
        distance_map: vec![f32::INFINITY; total],
        ..Default::default()
    };

    let Some(center) = weighted_center else {
        return result;
    };

    let components = find_components(&raw_mask, width, height);
    if components.is_empty() {
        return result;
    }

    let cx = center.0.round().clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let cy = center.1.round().clamp(0.0, height.saturating_sub(1) as f32) as u32;
    let center_idx = (cy * width + cx) as usize;

    let mut selected = 0usize;
    if raw_mask[center_idx] {
        for (index, component) in components.iter().enumerate() {
            if component[center_idx] {
                selected = index;
                break;
            }
        }
    } else {
        let mut best_distance = f32::INFINITY;
        for (index, component) in components.iter().enumerate() {
            let mut sx = 0.0f32;
            let mut sy = 0.0f32;
            let mut count = 0.0f32;
            for (idx, &value) in component.iter().enumerate() {
                if value {
                    let x = (idx % width as usize) as f32;
                    let y = (idx / width as usize) as f32;
                    sx += x;
                    sy += y;
                    count += 1.0;
                }
            }
            if count > 0.0 {
                let mx = sx / count;
                let my = sy / count;
                let dx = mx - center.0;
                let dy = my - center.1;
                let dist = dx * dx + dy * dy;
                if dist < best_distance {
                    best_distance = dist;
                    selected = index;
                }
            }
        }
    }

    let selected_mask = components[selected].clone();
    let boundary = mask_boundary(&selected_mask, width, height);
    let (hole_mask, hole_boundary, hole_count, hole_pixels) = detect_holes(&selected_mask, width, height);

    result.found = true;
    result.area_pixels = selected_mask.iter().filter(|&&v| v).count();
    result.hole_pixels = hole_pixels;
    result.hole_count = hole_count;
    result.mask = selected_mask.clone();
    result.boundary = boundary;
    result.hole_mask = hole_mask;
    result.hole_boundary = hole_boundary;
    result.distance_map = distance_map(&selected_mask, width, height, 100);
    result
}

fn find_components(mask: &[bool], width: u32, height: u32) -> Vec<Vec<bool>> {
    let w = width as usize;
    let h = height as usize;
    let mut visited = vec![false; mask.len()];
    let mut components = Vec::new();

    for idx in 0..mask.len() {
        if !mask[idx] || visited[idx] {
            continue;
        }
        let mut component = vec![false; mask.len()];
        let mut queue = VecDeque::new();
        queue.push_back(idx);
        visited[idx] = true;
        component[idx] = true;

        while let Some(current) = queue.pop_front() {
            let x = current % w;
            let y = current / w;
            for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let nidx = ny * w + nx;
                    if mask[nidx] && !visited[nidx] {
                        visited[nidx] = true;
                        component[nidx] = true;
                        queue.push_back(nidx);
                    }
                }
            }
        }
        components.push(component);
    }

    components.sort_by_key(|component| std::cmp::Reverse(component.iter().filter(|&&v| v).count()));
    components
}

fn dilate_mask(mask: &[bool], width: u32, height: u32, radius: u32) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    let w = width as usize;
    let h = height as usize;
    let r = radius as i32;
    let r2 = r * r;
    let mut out = vec![false; mask.len()];
    for y in 0..h {
        for x in 0..w {
            let mut hit = false;
            'outer: for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy > r2 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && nx < width as i32 && ny < height as i32 {
                        let nidx = ny as usize * w + nx as usize;
                        if mask[nidx] {
                            hit = true;
                            break 'outer;
                        }
                    }
                }
            }
            out[y * w + x] = hit;
        }
    }
    out
}

fn mask_boundary(mask: &[bool], width: u32, height: u32) -> Vec<bool> {
    let w = width as usize;
    let h = height as usize;
    let mut boundary = vec![false; mask.len()];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !mask[idx] {
                continue;
            }
            let mut is_boundary = x == 0 || y == 0 || x + 1 == w || y + 1 == h;
            if !is_boundary {
                for ny in y - 1..=y + 1 {
                    for nx in x - 1..=x + 1 {
                        let nidx = ny * w + nx;
                        if !mask[nidx] {
                            is_boundary = true;
                            break;
                        }
                    }
                    if is_boundary {
                        break;
                    }
                }
            }
            boundary[idx] = is_boundary;
        }
    }
    boundary
}

fn detect_holes(mask: &[bool], width: u32, height: u32) -> (Vec<bool>, Vec<bool>, usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut has_pixels = false;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if mask[idx] {
                has_pixels = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }

    if !has_pixels || min_x >= max_x || min_y >= max_y {
        return (vec![false; mask.len()], vec![false; mask.len()], 0, 0);
    }

    let mut visited = vec![false; mask.len()];
    let mut outside = vec![false; mask.len()];
    let mut queue = VecDeque::new();

    for x in min_x..=max_x {
        for &y in &[min_y, max_y] {
            let idx = y * w + x;
            if !mask[idx] && !visited[idx] {
                visited[idx] = true;
                outside[idx] = true;
                queue.push_back(idx);
            }
        }
    }
    for y in min_y..=max_y {
        for &x in &[min_x, max_x] {
            let idx = y * w + x;
            if !mask[idx] && !visited[idx] {
                visited[idx] = true;
                outside[idx] = true;
                queue.push_back(idx);
            }
        }
    }

    while let Some(current) = queue.pop_front() {
        let x = current % w;
        let y = current / w;
        for ny in y.saturating_sub(1)..=(y + 1).min(max_y) {
            for nx in x.saturating_sub(1)..=(x + 1).min(max_x) {
                if nx < min_x || ny < min_y {
                    continue;
                }
                let nidx = ny * w + nx;
                if !mask[nidx] && !visited[nidx] {
                    visited[nidx] = true;
                    outside[nidx] = true;
                    queue.push_back(nidx);
                }
            }
        }
    }

    let mut hole_mask = vec![false; mask.len()];
    let mut hole_count = 0usize;
    let mut hole_pixels = 0usize;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let idx = y * w + x;
            if mask[idx] || outside[idx] || hole_mask[idx] {
                continue;
            }

            hole_count += 1;
            let mut local_queue = VecDeque::new();
            local_queue.push_back(idx);
            hole_mask[idx] = true;
            hole_pixels += 1;

            while let Some(current) = local_queue.pop_front() {
                let cx = current % w;
                let cy = current / w;
                for ny in cy.saturating_sub(1)..=(cy + 1).min(max_y) {
                    for nx in cx.saturating_sub(1)..=(cx + 1).min(max_x) {
                        if nx < min_x || ny < min_y {
                            continue;
                        }
                        let nidx = ny * w + nx;
                        if !mask[nidx] && !outside[nidx] && !hole_mask[nidx] {
                            hole_mask[nidx] = true;
                            hole_pixels += 1;
                            local_queue.push_back(nidx);
                        }
                    }
                }
            }
        }
    }

    let hole_boundary = mask_boundary(&hole_mask, width, height);
    (hole_mask, hole_boundary, hole_count, hole_pixels)
}

fn distance_map(mask: &[bool], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let total = (width * height) as usize;
    if radius == 0 || !mask.iter().any(|&v| v) {
        return vec![f32::INFINITY; total];
    }

    let w = width as usize;
    let h = height as usize;
    let r = radius as i32;
    let r2 = (radius * radius) as i32;
    let mut proximity = vec![f32::INFINITY; total];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if mask[idx] {
                proximity[idx] = 0.0;
                continue;
            }
            let mut best = r2 + 1;
            for dy in -r..=r {
                for dx in -r..=r {
                    let dist2 = dx * dx + dy * dy;
                    if dist2 > r2 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && nx < width as i32 && ny < height as i32 {
                        let nidx = ny as usize * w + nx as usize;
                        if mask[nidx] {
                            best = best.min(dist2);
                        }
                    }
                }
            }
            if best <= r2 {
                let distance = (best as f32).sqrt();
                proximity[idx] = distance;
            }
        }
    }

    proximity
}

fn dimmed_original(image: &RgbaImage, dimness: f32, black_background: bool) -> RgbaImage {
    let factor = (1.0 - dimness).clamp(0.0, 1.0);
    let mut out = image.clone();
    for pixel in out.pixels_mut() {
        let [r, g, b, a] = pixel.0;
        let nr = (r as f32 * factor).round().clamp(0.0, 255.0) as u8;
        let ng = (g as f32 * factor).round().clamp(0.0, 255.0) as u8;
        let nb = (b as f32 * factor).round().clamp(0.0, 255.0) as u8;
        *pixel = if black_background {
            Rgba([nr, ng, nb, a])
        } else {
            let lift = ((1.0 - factor) * 255.0 * 0.15) as u8;
            Rgba([
                nr.saturating_add(lift),
                ng.saturating_add(lift),
                nb.saturating_add(lift),
                a,
            ])
        };
    }
    out
}

fn alpha_fill_mask(image: &mut RgbaImage, mask: &[bool], color: egui::Color32, alpha: u8) {
    let fill = Rgba([color.r(), color.g(), color.b(), alpha]);
    for (idx, pixel) in image.pixels_mut().enumerate() {
        if mask.get(idx).copied().unwrap_or(false) {
            *pixel = alpha_blend(*pixel, fill);
        }
    }
}

fn paint_mask(image: &mut RgbaImage, mask: &[bool], color: egui::Color32) {
    let solid = Rgba([color.r(), color.g(), color.b(), 255]);
    for (idx, pixel) in image.pixels_mut().enumerate() {
        if mask.get(idx).copied().unwrap_or(false) {
            *pixel = solid;
        }
    }
}

fn merge_union(union: &mut [bool], mask: &[bool]) {
    for (dst, &src) in union.iter_mut().zip(mask.iter()) {
        *dst |= src;
    }
}

fn brighten(color: egui::Color32, factor: f32) -> egui::Color32 {
    let factor = factor.max(0.0);
    egui::Color32::from_rgba_unmultiplied(
        ((color.r() as f32) + (255.0 - color.r() as f32) * factor)
            .round()
            .clamp(0.0, 255.0) as u8,
        ((color.g() as f32) + (255.0 - color.g() as f32) * factor)
            .round()
            .clamp(0.0, 255.0) as u8,
        ((color.b() as f32) + (255.0 - color.b() as f32) * factor)
            .round()
            .clamp(0.0, 255.0) as u8,
        color.a(),
    )
}

fn draw_cross(image: &mut RgbaImage, cx: i32, cy: i32, radius: i32, color: egui::Color32) {
    let pixel = Rgba([color.r(), color.g(), color.b(), 255]);
    for dx in -radius..=radius {
        let x = cx + dx;
        if x >= 0 && cy >= 0 && x < image.width() as i32 && cy < image.height() as i32 {
            image.put_pixel(x as u32, cy as u32, pixel);
        }
    }
    for dy in -radius..=radius {
        let y = cy + dy;
        if cx >= 0 && y >= 0 && cx < image.width() as i32 && y < image.height() as i32 {
            image.put_pixel(cx as u32, y as u32, pixel);
        }
    }
}

fn alpha_blend(dst: Rgba<u8>, src: Rgba<u8>) -> Rgba<u8> {
    let alpha = src[3] as f32 / 255.0;
    let inv_alpha = 1.0 - alpha;
    let r = (src[0] as f32 * alpha + dst[0] as f32 * inv_alpha)
        .round()
        .clamp(0.0, 255.0) as u8;
    let g = (src[1] as f32 * alpha + dst[1] as f32 * inv_alpha)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (src[2] as f32 * alpha + dst[2] as f32 * inv_alpha)
        .round()
        .clamp(0.0, 255.0) as u8;
    Rgba([r, g, b, 255])
}
