mod viewer;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Context as _, Result};
use eframe::egui;
use image::{DynamicImage, GrayImage, Rgba, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const APP_TITLE: &str = "Rust Egui Viewer — Green Shapes";
const MAX_HISTORY: usize = 20;

fn main() -> eframe::Result {
    let initial_source = std::env::args().nth(1);

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 940.0])
            .with_min_inner_size([980.0, 700.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        APP_TITLE,
        native_options,
        Box::new(move |cc| Ok(Box::new(GreenViewerApp::new(cc, initial_source.clone())))),
    )
}

#[derive(Clone, Debug)]
struct GreenSettings {
    green_excess_threshold: f32,
    green_ratio_threshold: f32,
    min_component_area: usize,
    grow_radius: u32,
    fill_color: egui::Color32,
    fill_opacity: u8,
    outline_color: egui::Color32,
    outline_opacity: u8,
    dimness: f32,
    edge_enabled: bool,
    edge_threshold: f32,
    edge_color: egui::Color32,
    edge_opacity: u8,
}

impl Default for GreenSettings {
    fn default() -> Self {
        Self {
            green_excess_threshold: 28.0,
            green_ratio_threshold: 0.38,
            min_component_area: 80,
            grow_radius: 1,
            fill_color: egui::Color32::from_rgb(35, 255, 105),
            fill_opacity: 64,
            outline_color: egui::Color32::from_rgb(80, 255, 140),
            outline_opacity: 255,
            dimness: 0.55,
            edge_enabled: false,
            edge_threshold: 120.0,
            edge_color: egui::Color32::from_rgb(0, 220, 255),
            edge_opacity: 220,
        }
    }
}

#[derive(Clone)]
struct ProcessingRequest {
    id: u64,
    rgba: Arc<RgbaImage>,
    gray: Arc<GrayImage>,
    settings: GreenSettings,
}

struct ProcessingResult {
    processed: RgbaImage,
    shape_count: usize,
    green_pixels: usize,
    boundary_pixels: usize,
}

enum ProcessingMessage {
    Progress {
        id: u64,
        value: f32,
        stage: &'static str,
    },
    Finished {
        id: u64,
        result: ProcessingResult,
    },
    Failed {
        id: u64,
        error: String,
    },
}

struct ProcessingWorker {
    job_tx: mpsc::Sender<ProcessingRequest>,
    message_rx: mpsc::Receiver<ProcessingMessage>,
    latest_id: Arc<AtomicU64>,
    _thread: thread::JoinHandle<()>,
}

impl ProcessingWorker {
    fn spawn(repaint_ctx: egui::Context) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<ProcessingRequest>();
        let (message_tx, message_rx) = mpsc::channel::<ProcessingMessage>();
        let latest_id = Arc::new(AtomicU64::new(0));
        let worker_latest = Arc::clone(&latest_id);

        let worker = thread::Builder::new()
            .name("green-processing-worker".to_owned())
            .spawn(move || processing_loop(job_rx, message_tx, worker_latest, repaint_ctx))
            .expect("failed to spawn processing worker");

        Self {
            job_tx,
            message_rx,
            latest_id,
            _thread: worker,
        }
    }
}

#[derive(Clone)]
enum SourceRequest {
    File(PathBuf),
    Url(String),
}

struct LoadedImage {
    label: String,
    rgba: RgbaImage,
    gray: GrayImage,
}

enum SourceMessage {
    Loaded { id: u64, image: LoadedImage },
    Failed { id: u64, error: String },
}

struct SourceWorker {
    tx: mpsc::Sender<(u64, SourceRequest)>,
    rx: mpsc::Receiver<SourceMessage>,
    latest_id: Arc<AtomicU64>,
    _thread: thread::JoinHandle<()>,
}

impl SourceWorker {
    fn spawn(repaint_ctx: egui::Context) -> Self {
        let (tx, job_rx) = mpsc::channel::<(u64, SourceRequest)>();
        let (message_tx, rx) = mpsc::channel::<SourceMessage>();
        let latest_id = Arc::new(AtomicU64::new(0));
        let worker_latest = Arc::clone(&latest_id);

        let worker = thread::Builder::new()
            .name("image-source-worker".to_owned())
            .spawn(move || source_loop(job_rx, message_tx, worker_latest, repaint_ctx))
            .expect("failed to spawn source worker");

        Self {
            tx,
            rx,
            latest_id,
            _thread: worker,
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
struct PersistedState {
    #[serde(default)]
    source_history: Vec<String>,
}

struct GreenViewerApp {
    original_rgba: Option<Arc<RgbaImage>>,
    original_gray: Option<Arc<GrayImage>>,
    original_texture: Option<egui::TextureHandle>,
    processed_rgba: Option<RgbaImage>,
    processed_texture: Option<egui::TextureHandle>,

    source_label: String,
    url_input: String,
    source_history: Vec<String>,
    source_worker: SourceWorker,
    next_source_id: u64,
    active_source_id: u64,
    source_loading: bool,

    settings: GreenSettings,
    dirty: bool,
    update_while_dragging: bool,

    processing_worker: ProcessingWorker,
    next_job_id: u64,
    active_job_id: u64,
    processing: bool,
    progress: f32,
    progress_stage: String,

    shape_count: usize,
    green_pixels: usize,
    boundary_pixels: usize,
    status: String,
    error: Option<String>,

    original_view: viewer::ImageViewState,
    processed_view: viewer::ImageViewState,
    link_views: bool,
}

impl GreenViewerApp {
    fn new(cc: &eframe::CreationContext<'_>, initial_source: Option<String>) -> Self {
        configure_dark_ui(&cc.egui_ctx);
        let persisted = load_persisted_state();

        let mut app = Self {
            original_rgba: None,
            original_gray: None,
            original_texture: None,
            processed_rgba: None,
            processed_texture: None,
            source_label: "No image".to_owned(),
            url_input: String::new(),
            source_history: persisted.source_history,
            source_worker: SourceWorker::spawn(cc.egui_ctx.clone()),
            next_source_id: 0,
            active_source_id: 0,
            source_loading: false,
            settings: GreenSettings::default(),
            dirty: false,
            update_while_dragging: true,
            processing_worker: ProcessingWorker::spawn(cc.egui_ctx.clone()),
            next_job_id: 0,
            active_job_id: 0,
            processing: false,
            progress: 0.0,
            progress_stage: "Idle".to_owned(),
            shape_count: 0,
            green_pixels: 0,
            boundary_pixels: 0,
            status: "Open an image from disk or enter an image URL.".to_owned(),
            error: None,
            original_view: viewer::ImageViewState::default(),
            processed_view: viewer::ImageViewState::default(),
            link_views: false,
        };

        if let Some(source) = initial_source {
            if source.starts_with("http://") || source.starts_with("https://") {
                app.url_input = source.clone();
                app.queue_source(SourceRequest::Url(source));
            } else {
                app.queue_source(SourceRequest::File(PathBuf::from(source)));
            }
        }

        app
    }

    fn queue_source(&mut self, request: SourceRequest) {
        self.next_source_id = self.next_source_id.wrapping_add(1).max(1);
        self.active_source_id = self.next_source_id;
        self.source_worker
            .latest_id
            .store(self.active_source_id, Ordering::Release);

        match self.source_worker.tx.send((self.active_source_id, request)) {
            Ok(()) => {
                self.source_loading = true;
                self.error = None;
                self.status = "Loading image…".to_owned();
            }
            Err(error) => {
                self.source_loading = false;
                self.error = Some(format!("Image loader stopped: {error}"));
            }
        }
    }

    fn poll_source_worker(&mut self, ctx: &egui::Context) {
        while let Ok(message) = self.source_worker.rx.try_recv() {
            match message {
                SourceMessage::Loaded { id, image } if id == self.active_source_id => {
                    self.source_loading = false;
                    self.install_loaded_image(image, ctx);
                }
                SourceMessage::Failed { id, error } if id == self.active_source_id => {
                    self.source_loading = false;
                    self.error = Some(error);
                    self.status = "Load failed".to_owned();
                }
                _ => {}
            }
        }
    }

    fn install_loaded_image(&mut self, image: LoadedImage, ctx: &egui::Context) {
        let rgba = Arc::new(image.rgba);
        let gray = Arc::new(image.gray);
        self.original_texture = Some(ctx.load_texture(
            "original-image",
            rgba_to_color_image(rgba.as_ref()),
            egui::TextureOptions::LINEAR,
        ));
        self.original_rgba = Some(rgba);
        self.original_gray = Some(gray);
        self.source_label = image.label.clone();
        self.original_view.reset_fit();
        self.processed_view.reset_fit();
        self.status = format!("Loaded {}", image.label);
        self.error = None;
        self.remember_source(image.label);
        self.schedule_processing();
    }

    fn remember_source(&mut self, source: String) {
        self.source_history.retain(|entry| entry != &source);
        self.source_history.insert(0, source);
        self.source_history.truncate(MAX_HISTORY);
        save_persisted_state(&PersistedState {
            source_history: self.source_history.clone(),
        });
    }

    fn schedule_processing(&mut self) {
        let (Some(rgba), Some(gray)) = (self.original_rgba.as_ref(), self.original_gray.as_ref())
        else {
            return;
        };

        self.next_job_id = self.next_job_id.wrapping_add(1).max(1);
        self.active_job_id = self.next_job_id;
        self.processing_worker
            .latest_id
            .store(self.active_job_id, Ordering::Release);

        let request = ProcessingRequest {
            id: self.active_job_id,
            rgba: Arc::clone(rgba),
            gray: Arc::clone(gray),
            settings: self.settings.clone(),
        };

        match self.processing_worker.job_tx.send(request) {
            Ok(()) => {
                self.processing = true;
                self.progress = 0.0;
                self.progress_stage = "Queued".to_owned();
                self.dirty = false;
                self.error = None;
            }
            Err(error) => {
                self.processing = false;
                self.error = Some(format!("Processing worker stopped: {error}"));
            }
        }
    }

    fn poll_processing_worker(&mut self, ctx: &egui::Context) {
        while let Ok(message) = self.processing_worker.message_rx.try_recv() {
            match message {
                ProcessingMessage::Progress { id, value, stage }
                    if id == self.active_job_id =>
                {
                    self.processing = true;
                    self.progress = value.clamp(0.0, 1.0);
                    self.progress_stage = stage.to_owned();
                }
                ProcessingMessage::Finished { id, result } if id == self.active_job_id => {
                    self.processing = false;
                    self.progress = 1.0;
                    self.progress_stage = "Complete".to_owned();
                    self.shape_count = result.shape_count;
                    self.green_pixels = result.green_pixels;
                    self.boundary_pixels = result.boundary_pixels;
                    self.processed_rgba = Some(result.processed.clone());

                    let color_image = rgba_to_color_image(&result.processed);
                    if let Some(texture) = self.processed_texture.as_mut() {
                        texture.set(color_image, egui::TextureOptions::LINEAR);
                    } else {
                        self.processed_texture = Some(ctx.load_texture(
                            "processed-image",
                            color_image,
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                    self.status = format!(
                        "Detected {} green shape(s), {} green px, {} boundary px",
                        self.shape_count, self.green_pixels, self.boundary_pixels
                    );
                    self.error = None;
                }
                ProcessingMessage::Failed { id, error } if id == self.active_job_id => {
                    self.processing = false;
                    self.progress_stage = "Failed".to_owned();
                    self.error = Some(error);
                }
                _ => {}
            }
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect::<Vec<_>>()
        });
        if let Some(path) = dropped.first() {
            self.queue_source(SourceRequest::File(path.clone()));
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Green-shape image viewer");
        ui.small("Disk + HTTP input, asynchronous green detection, idle-safe repainting.");
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Open image…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Open image")
                    .add_filter(
                        "Image",
                        &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
                    )
                    .pick_file()
                {
                    self.queue_source(SourceRequest::File(path));
                }
            }

            if ui
                .add_enabled(self.processed_rgba.is_some(), egui::Button::new("Save processed…"))
                .clicked()
            {
                self.save_processed();
            }
        });

        ui.add_space(4.0);
        ui.label("Network image URL");
        let url_response = ui.text_edit_singleline(&mut self.url_input);
        let enter = url_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if (ui.button("Load URL").clicked() || enter) && !self.url_input.trim().is_empty() {
            self.queue_source(SourceRequest::Url(self.url_input.trim().to_owned()));
        }
        ui.small("The URL must return image bytes directly (JPEG/PNG/WebP/etc.).");

        if !self.source_history.is_empty() {
            ui.collapsing("Source history", |ui| {
                let history = self.source_history.clone();
                for source in history {
                    if ui.selectable_label(false, &source).clicked() {
                        if source.starts_with("http://") || source.starts_with("https://") {
                            self.url_input = source.clone();
                            self.queue_source(SourceRequest::Url(source));
                        } else {
                            self.queue_source(SourceRequest::File(PathBuf::from(source)));
                        }
                    }
                }
            });
        }

        ui.separator();
        ui.collapsing("Image windows", |ui| {
            ui.checkbox(&mut self.original_view.open, "Show original");
            ui.checkbox(&mut self.processed_view.open, "Show processed");
            ui.checkbox(&mut self.link_views, "Link pan + zoom");
            ui.horizontal(|ui| {
                if ui.button("Fit both").clicked() {
                    self.original_view.reset_fit();
                    self.processed_view.reset_fit();
                }
                if ui.button("Processed ← Original").clicked() {
                    self.processed_view.copy_transform_from(&self.original_view);
                }
            });
        });

        ui.separator();
        ui.heading("Green detection");
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.green_excess_threshold, 0.0..=180.0)
                    .text("Green excess")
                    .fixed_decimals(1),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.green_ratio_threshold, 0.0..=1.0)
                    .text("Green ratio")
                    .fixed_decimals(2),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.min_component_area, 1..=100_000)
                    .logarithmic(true)
                    .text("Min shape area"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.settings.grow_radius, 0..=12).text("Mask grow"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.dimness, 0.0..=1.0)
                    .text("Background brightness")
                    .fixed_decimals(2),
            )
            .changed();

        ui.horizontal(|ui| {
            ui.label("Fill");
            changed |= ui
                .color_edit_button_srgba(&mut self.settings.fill_color)
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.settings.fill_opacity, 0..=255).text("α"))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Outline");
            changed |= ui
                .color_edit_button_srgba(&mut self.settings.outline_color)
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.settings.outline_opacity, 0..=255).text("α"))
                .changed();
        });

        ui.separator();
        ui.collapsing("Additional edge overlay", |ui| {
            changed |= ui
                .checkbox(&mut self.settings.edge_enabled, "Enabled")
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.edge_threshold, 0.0..=900.0)
                        .text("Sobel threshold")
                        .fixed_decimals(1),
                )
                .changed();
            ui.horizontal(|ui| {
                ui.label("Color");
                changed |= ui
                    .color_edit_button_srgba(&mut self.settings.edge_color)
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.settings.edge_opacity, 0..=255).text("α"))
                    .changed();
            });
        });

        ui.checkbox(
            &mut self.update_while_dragging,
            "Reprocess while dragging sliders",
        );

        if changed {
            self.dirty = true;
        }
        let pointer_down = ui.input(|input| input.pointer.primary_down());
        if self.dirty && (self.update_while_dragging || !pointer_down) {
            self.schedule_processing();
        }
        if ui
            .add_enabled(self.dirty, egui::Button::new("Apply parameters"))
            .clicked()
        {
            self.schedule_processing();
        }

        ui.separator();
        ui.label(format!("Source: {}", self.source_label));
        if let Some(gray) = self.original_gray.as_ref() {
            ui.label(format!("Resolution: {} × {}", gray.width(), gray.height()));
        }
        ui.label(format!("Green shapes: {}", self.shape_count));
        ui.label(format!("Green pixels: {}", self.green_pixels));
        ui.label(format!("Boundary pixels: {}", self.boundary_pixels));
        ui.small("CPU stays idle after work: both worker threads block on recv(), and the UI uses worker-triggered repaint instead of a repaint timer.");
    }

    fn previews(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.strong("Floating image windows");
            ui.checkbox(&mut self.original_view.open, "Original");
            ui.checkbox(&mut self.processed_view.open, "Processed");
            ui.checkbox(&mut self.link_views, "Link views");
        });
        ui.small("Resize each window. Fit follows its borders. Drag the image to pan; pinch/Ctrl+wheel zooms around the cursor.");
        ui.allocate_space(ui.available_size());

        let ctx = ui.ctx().clone();
        let original_interaction = viewer::show_floating_image_window(
            &ctx,
            "Original image",
            self.original_texture.as_ref(),
            &mut self.original_view,
            egui::pos2(390.0, 80.0),
        );
        if self.link_views && original_interaction.transform_changed {
            self.processed_view.copy_transform_from(&self.original_view);
        }

        let processed_interaction = viewer::show_floating_image_window(
            &ctx,
            "Processed — green shapes",
            self.processed_texture.as_ref(),
            &mut self.processed_view,
            egui::pos2(840.0, 120.0),
        );
        if self.link_views
            && processed_interaction.transform_changed
            && !original_interaction.transform_changed
        {
            self.original_view.copy_transform_from(&self.processed_view);
        }
    }

    fn save_processed(&mut self) {
        let Some(image) = self.processed_rgba.as_ref() else {
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save processed image")
            .set_file_name("green_shapes.png")
            .add_filter("PNG", &["png"])
            .save_file()
        else {
            return;
        };

        let mut path = path;
        if path.extension().is_none() {
            path.set_extension("png");
        }
        match DynamicImage::ImageRgba8(image.clone()).save(&path) {
            Ok(()) => {
                self.status = format!("Saved {}", path.display());
                self.error = None;
            }
            Err(error) => {
                self.error = Some(format!("Failed to save {}: {error}", path.display()));
            }
        }
    }
}

impl eframe::App for GreenViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_dropped_files(&ctx);
        self.poll_source_worker(&ctx);
        self.poll_processing_worker(&ctx);

        egui::Panel::bottom("status-bar").show(ui, |ui| {
            if self.source_loading {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Loading image source…");
                });
            } else if self.processing {
                ui.horizontal(|ui| {
                    ui.label(&self.progress_stage);
                    ui.add(
                        egui::ProgressBar::new(self.progress)
                            .show_percentage()
                            .desired_width(260.0),
                    );
                });
            } else if let Some(error) = self.error.as_ref() {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            } else {
                ui.label(&self.status);
            }
        });

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(380.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });

        egui::CentralPanel::default().show(ui, |ui| self.previews(ui));
    }
}

fn source_loop(
    job_rx: mpsc::Receiver<(u64, SourceRequest)>,
    message_tx: mpsc::Sender<SourceMessage>,
    latest_id: Arc<AtomicU64>,
    repaint_ctx: egui::Context,
) {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(15))
        .user_agent("rust-egui-viewer/0.3")
        .build()
        .expect("failed to build HTTP client");

    while let Ok((mut id, mut request)) = job_rx.recv() {
        while let Ok((new_id, newer)) = job_rx.try_recv() {
            id = new_id;
            request = newer;
        }

        if latest_id.load(Ordering::Acquire) != id {
            continue;
        }

        let result = match request {
            SourceRequest::File(path) => load_file_source(&path),
            SourceRequest::Url(url) => load_url_source(&client, &url),
        };

        if latest_id.load(Ordering::Acquire) != id {
            continue;
        }

        let message = match result {
            Ok(image) => SourceMessage::Loaded { id, image },
            Err(error) => SourceMessage::Failed {
                id,
                error: format!("Failed to load image: {error:#}"),
            },
        };
        let _ = message_tx.send(message);
        repaint_ctx.request_repaint();
    }
}

fn load_file_source(path: &Path) -> Result<LoadedImage> {
    let decoded = image::open(path)
        .with_context(|| format!("unable to decode {}", path.display()))?;
    Ok(LoadedImage {
        label: path.display().to_string(),
        rgba: decoded.to_rgba8(),
        gray: decoded.to_luma8(),
    })
}

fn load_url_source(client: &reqwest::blocking::Client, url: &str) -> Result<LoadedImage> {
    let response = client
        .get(url)
        .send()
        .with_context(|| format!("request failed for {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;

    let bytes = response.bytes().context("failed reading HTTP response body")?;
    if bytes.is_empty() {
        return Err(anyhow!("server returned an empty response"));
    }
    let decoded = image::load_from_memory(&bytes)
        .with_context(|| "response is not a supported image; enter the direct image endpoint")?;
    Ok(LoadedImage {
        label: url.to_owned(),
        rgba: decoded.to_rgba8(),
        gray: decoded.to_luma8(),
    })
}

fn processing_loop(
    job_rx: mpsc::Receiver<ProcessingRequest>,
    message_tx: mpsc::Sender<ProcessingMessage>,
    latest_id: Arc<AtomicU64>,
    repaint_ctx: egui::Context,
) {
    while let Ok(mut request) = job_rx.recv() {
        while let Ok(newer) = job_rx.try_recv() {
            request = newer;
        }

        let id = request.id;
        match process_image(&request, &latest_id, &message_tx, &repaint_ctx) {
            Ok(Some(result)) => {
                let _ = message_tx.send(ProcessingMessage::Finished { id, result });
                repaint_ctx.request_repaint();
            }
            Ok(None) => {}
            Err(error) => {
                let _ = message_tx.send(ProcessingMessage::Failed {
                    id,
                    error: format!("Processing failed: {error:#}"),
                });
                repaint_ctx.request_repaint();
            }
        }
    }
}

fn process_image(
    request: &ProcessingRequest,
    latest_id: &AtomicU64,
    progress_tx: &mpsc::Sender<ProcessingMessage>,
    repaint_ctx: &egui::Context,
) -> Result<Option<ProcessingResult>> {
    let id = request.id;
    let cancelled = || latest_id.load(Ordering::Acquire) != id;

    progress(progress_tx, repaint_ctx, id, 0.05, "Finding green pixels");
    let width = request.rgba.width();
    let height = request.rgba.height();
    if width == 0 || height == 0 {
        return Err(anyhow!("image has zero size"));
    }

    let settings = &request.settings;
    let mut raw_mask: Vec<bool> = request
        .rgba
        .as_raw()
        .par_chunks_exact(4)
        .map(|pixel| {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let green_excess = g - r.max(b);
            let green_ratio = g / (r + g + b + 1.0);
            green_excess >= settings.green_excess_threshold
                && green_ratio >= settings.green_ratio_threshold
        })
        .collect();

    if cancelled() {
        return Ok(None);
    }

    if settings.grow_radius > 0 {
        progress(progress_tx, repaint_ctx, id, 0.22, "Growing green mask");
        raw_mask = dilate_mask(
            &raw_mask,
            width as usize,
            height as usize,
            settings.grow_radius as usize,
        );
    }

    if cancelled() {
        return Ok(None);
    }

    progress(progress_tx, repaint_ctx, id, 0.35, "Finding green shapes");
    let (shape_mask, shape_count) = keep_components(
        &raw_mask,
        width as usize,
        height as usize,
        settings.min_component_area,
        latest_id,
        id,
    );
    let Some(shape_mask) = shape_mask else {
        return Ok(None);
    };

    progress(progress_tx, repaint_ctx, id, 0.58, "Building outlines");
    let boundary = boundary_mask(&shape_mask, width as usize, height as usize);

    if cancelled() {
        return Ok(None);
    }

    progress(progress_tx, repaint_ctx, id, 0.70, "Compositing result");
    let mut processed = dim_image(&request.rgba, settings.dimness);
    alpha_paint_mask(
        &mut processed,
        &shape_mask,
        settings.fill_color,
        settings.fill_opacity,
    );
    alpha_paint_mask(
        &mut processed,
        &boundary,
        settings.outline_color,
        settings.outline_opacity,
    );

    if settings.edge_enabled {
        progress(progress_tx, repaint_ctx, id, 0.82, "Detecting additional edges");
        let edge_mask = sobel_edges(&request.gray, settings.edge_threshold, latest_id, id);
        let Some(edge_mask) = edge_mask else {
            return Ok(None);
        };
        alpha_paint_mask(
            &mut processed,
            &edge_mask,
            settings.edge_color,
            settings.edge_opacity,
        );
    }

    if cancelled() {
        return Ok(None);
    }

    let green_pixels = shape_mask.iter().filter(|&&v| v).count();
    let boundary_pixels = boundary.iter().filter(|&&v| v).count();
    progress(progress_tx, repaint_ctx, id, 1.0, "Complete");

    Ok(Some(ProcessingResult {
        processed,
        shape_count,
        green_pixels,
        boundary_pixels,
    }))
}

fn progress(
    tx: &mpsc::Sender<ProcessingMessage>,
    repaint_ctx: &egui::Context,
    id: u64,
    value: f32,
    stage: &'static str,
) {
    let _ = tx.send(ProcessingMessage::Progress { id, value, stage });
    repaint_ctx.request_repaint();
}

fn keep_components(
    mask: &[bool],
    width: usize,
    height: usize,
    min_area: usize,
    latest_id: &AtomicU64,
    job_id: u64,
) -> (Option<Vec<bool>>, usize) {
    let mut visited = vec![false; mask.len()];
    let mut output = vec![false; mask.len()];
    let mut shape_count = 0;
    let mut queue = VecDeque::new();
    let mut component = Vec::new();

    for start in 0..mask.len() {
        if start % 65_536 == 0 && latest_id.load(Ordering::Relaxed) != job_id {
            return (None, 0);
        }
        if !mask[start] || visited[start] {
            continue;
        }

        visited[start] = true;
        queue.push_back(start);
        component.clear();

        while let Some(idx) = queue.pop_front() {
            component.push(idx);
            let x = idx % width;
            let y = idx / width;

            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);

            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let n = ny * width + nx;
                    if mask[n] && !visited[n] {
                        visited[n] = true;
                        queue.push_back(n);
                    }
                }
            }
        }

        if component.len() >= min_area {
            shape_count += 1;
            for &idx in &component {
                output[idx] = true;
            }
        }
    }

    (Some(output), shape_count)
}

fn dilate_mask(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    (0..mask.len())
        .into_par_iter()
        .map(|idx| {
            let x = idx % width;
            let y = idx / width;
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(width - 1);
            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius).min(height - 1);
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    if mask[ny * width + nx] {
                        return true;
                    }
                }
            }
            false
        })
        .collect()
}

fn boundary_mask(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    (0..mask.len())
        .into_par_iter()
        .map(|idx| {
            if !mask[idx] {
                return false;
            }
            let x = idx % width;
            let y = idx / width;
            if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
                return true;
            }
            !mask[idx - 1] || !mask[idx + 1] || !mask[idx - width] || !mask[idx + width]
        })
        .collect()
}

fn sobel_edges(
    gray: &GrayImage,
    threshold: f32,
    latest_id: &AtomicU64,
    job_id: u64,
) -> Option<Vec<bool>> {
    let width = gray.width() as usize;
    let height = gray.height() as usize;
    if width < 3 || height < 3 {
        return Some(vec![false; width * height]);
    }
    let values = gray.as_raw();
    let rows: Vec<Vec<bool>> = (0..height)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![false; width];
            if y == 0 || y + 1 == height || latest_id.load(Ordering::Relaxed) != job_id {
                return row;
            }
            for (x, value) in row.iter_mut().enumerate().take(width - 1).skip(1) {
                let p = |xx: usize, yy: usize| values[yy * width + xx] as f32;
                let gx = -p(x - 1, y - 1)
                    + p(x + 1, y - 1)
                    - 2.0 * p(x - 1, y)
                    + 2.0 * p(x + 1, y)
                    - p(x - 1, y + 1)
                    + p(x + 1, y + 1);
                let gy = -p(x - 1, y - 1)
                    - 2.0 * p(x, y - 1)
                    - p(x + 1, y - 1)
                    + p(x - 1, y + 1)
                    + 2.0 * p(x, y + 1)
                    + p(x + 1, y + 1);
                *value = (gx * gx + gy * gy).sqrt() >= threshold;
            }
            row
        })
        .collect();

    if latest_id.load(Ordering::Acquire) != job_id {
        return None;
    }
    Some(rows.into_iter().flatten().collect())
}

fn dim_image(image: &RgbaImage, brightness: f32) -> RgbaImage {
    let brightness = brightness.clamp(0.0, 1.0);
    let mut output = image.clone();
    output
        .as_mut()
        .par_chunks_exact_mut(4)
        .for_each(|pixel| {
            pixel[0] = (pixel[0] as f32 * brightness).round() as u8;
            pixel[1] = (pixel[1] as f32 * brightness).round() as u8;
            pixel[2] = (pixel[2] as f32 * brightness).round() as u8;
        });
    output
}

fn alpha_paint_mask(image: &mut RgbaImage, mask: &[bool], color: egui::Color32, opacity: u8) {
    let alpha = opacity as f32 / 255.0;
    let [r, g, b, _] = color.to_array();
    image
        .as_mut()
        .par_chunks_exact_mut(4)
        .zip(mask.par_iter())
        .for_each(|(pixel, &paint)| {
            if !paint {
                return;
            }
            pixel[0] = blend(pixel[0], r, alpha);
            pixel[1] = blend(pixel[1], g, alpha);
            pixel[2] = blend(pixel[2], b, alpha);
            pixel[3] = pixel[3].max(opacity);
        });
}

fn blend(base: u8, overlay: u8, alpha: f32) -> u8 {
    (base as f32 * (1.0 - alpha) + overlay as f32 * alpha)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn rgba_to_color_image(image: &RgbaImage) -> egui::ColorImage {
    egui::ColorImage::from_rgba_unmultiplied(
        [image.width() as usize, image.height() as usize],
        image.as_raw(),
    )
}

fn configure_dark_ui(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = egui::Color32::BLACK;
    style.visuals.window_fill = egui::Color32::from_rgb(8, 8, 8);
    style.visuals.extreme_bg_color = egui::Color32::BLACK;
    ctx.set_style_of(egui::Theme::Dark, style);
}

fn state_file_path() -> Option<PathBuf> {
    dirs::config_dir().map(|dir| dir.join("rust-egui-viewer").join("state.json"))
}

fn load_persisted_state() -> PersistedState {
    let Some(path) = state_file_path() else {
        return PersistedState::default();
    };
    let Ok(bytes) = std::fs::read(path) else {
        return PersistedState::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

fn save_persisted_state(state: &PersistedState) {
    let Some(path) = state_file_path() else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    if let Ok(data) = serde_json::to_vec_pretty(state) {
        let _ = std::fs::write(path, data);
    }
}
