use std::collections::VecDeque;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use eframe::egui;
use image::{DynamicImage, GrayImage, Rgba, RgbaImage};
use rayon::prelude::*;

const MAX_EDGE_THRESHOLD: f32 = 1140.0;
const MIN_SCALE: f32 = 0.10;
const MAX_SCALE: f32 = 8.0;
const SOURCE_HISTORY_LIMIT: usize = 40;
const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_PROFILE_NAME: &str = "default";
const PROFILE_AUTOSAVE_DELAY: Duration = Duration::from_millis(500);
const PROFILE_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CameraResolution {
    label: &'static str,
    framesize: i32,
}

const CAMERA_RESOLUTIONS: &[CameraResolution] = &[
    CameraResolution { label: "Keep camera setting", framesize: -1 },
    CameraResolution { label: "96 × 96", framesize: 0 },
    CameraResolution { label: "160 × 120 (QQVGA)", framesize: 1 },
    CameraResolution { label: "128 × 128", framesize: 2 },
    CameraResolution { label: "176 × 144 (QCIF)", framesize: 3 },
    CameraResolution { label: "240 × 176 (HQVGA)", framesize: 4 },
    CameraResolution { label: "240 × 240", framesize: 5 },
    CameraResolution { label: "320 × 240 (QVGA)", framesize: 6 },
    CameraResolution { label: "320 × 320", framesize: 7 },
    CameraResolution { label: "400 × 296 (CIF)", framesize: 8 },
    CameraResolution { label: "480 × 320 (HVGA)", framesize: 9 },
    CameraResolution { label: "640 × 480 (VGA)", framesize: 10 },
    CameraResolution { label: "800 × 600 (SVGA)", framesize: 11 },
    CameraResolution { label: "1024 × 768 (XGA)", framesize: 12 },
    CameraResolution { label: "1280 × 720 (HD)", framesize: 13 },
    CameraResolution { label: "1280 × 1024 (SXGA)", framesize: 14 },
    CameraResolution { label: "1600 × 1200 (UXGA)", framesize: 15 },
];

fn camera_resolution_by_framesize(framesize: i32) -> CameraResolution {
    CAMERA_RESOLUTIONS
        .iter()
        .copied()
        .find(|entry| entry.framesize == framesize)
        .unwrap_or(CAMERA_RESOLUTIONS[0])
}

fn main() -> eframe::Result {
    let initial_path = std::env::args_os().nth(1).map(PathBuf::from);

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 940.0])
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
    opacity: u8,
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

        Self {
            enabled: true,
            name: format!("Edge layer {}", index + 1),
            low_threshold: 50.0 + index as f32 * 20.0,
            high_threshold: 150.0 + index as f32 * 40.0,
            reduction_strength: 0.35,
            reduction_radius: 30,
            color: palette[index % palette.len()],
            opacity: 255,
            edge_pixels: 0,
        }
    }
}

#[derive(Clone)]
struct GreenShapeLayer {
    enabled: bool,
    green_excess_threshold: f32,
    green_ratio_threshold: f32,
    mask_grow_radius: u32,
    outline_color: egui::Color32,
    outline_opacity: u8,
    show_holes: bool,
    show_center: bool,
}

impl Default for GreenShapeLayer {
    fn default() -> Self {
        Self {
            enabled: true,
            green_excess_threshold: 28.0,
            green_ratio_threshold: 0.38,
            mask_grow_radius: 1,
            outline_color: egui::Color32::from_rgb(80, 255, 140),
            outline_opacity: 255,
            show_holes: true,
            show_center: true,
        }
    }
}

#[derive(Clone)]
struct GreenAreaOverlay {
    enabled: bool,
    color: egui::Color32,
    opacity: u8,
    output_alpha_enabled: bool,
    output_alpha: u8,
}

impl Default for GreenAreaOverlay {
    fn default() -> Self {
        Self {
            enabled: true,
            color: egui::Color32::from_rgb(35, 255, 105),
            opacity: 64,
            output_alpha_enabled: false,
            output_alpha: 96,
        }
    }
}

#[derive(Clone)]
struct ProfileSnapshot {
    source_input: String,
    source_history: VecDeque<String>,
    camera_framesize: i32,
    shape_layer: GreenShapeLayer,
    green_overlay: GreenAreaOverlay,
    edge_layers: Vec<EdgeLayer>,
    preblur_sigma: f32,
    dimness: f32,
    black_background: bool,
    update_while_dragging: bool,
    original_scale: f32,
    composite_scale: f32,
}

impl Default for ProfileSnapshot {
    fn default() -> Self {
        Self {
            source_input: String::new(),
            source_history: VecDeque::new(),
            camera_framesize: -1,
            shape_layer: GreenShapeLayer::default(),
            green_overlay: GreenAreaOverlay::default(),
            edge_layers: vec![EdgeLayer::new(0), EdgeLayer::new(1)],
            preblur_sigma: 0.7,
            dimness: 0.55,
            black_background: true,
            update_while_dragging: true,
            original_scale: 1.0,
            composite_scale: 1.0,
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
        percentage(self.area_pixels, self.width, self.height)
    }

    fn hole_percent(&self) -> f32 {
        percentage(self.hole_pixels, self.width, self.height)
    }
}

#[derive(Clone)]
struct ProcessingSettings {
    shape_layer: GreenShapeLayer,
    green_overlay: GreenAreaOverlay,
    edge_layers: Vec<EdgeLayer>,
    preblur_sigma: f32,
    dimness: f32,
    black_background: bool,
}

struct ProcessingRequest {
    id: u64,
    original_rgba: Arc<RgbaImage>,
    original_gray: Arc<GrayImage>,
    settings: ProcessingSettings,
}

struct ProcessingResult {
    composite: RgbaImage,
    green_shape: GreenShapeResult,
    edge_counts: Vec<usize>,
    unique_edge_pixels: usize,
}

enum WorkerMessage {
    Progress {
        id: u64,
        value: f32,
        stage: String,
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
    message_rx: mpsc::Receiver<WorkerMessage>,
    latest_id: Arc<AtomicU64>,
    _thread: thread::JoinHandle<()>,
}

impl ProcessingWorker {
    fn spawn() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<ProcessingRequest>();
        let (message_tx, message_rx) = mpsc::channel::<WorkerMessage>();
        let latest_id = Arc::new(AtomicU64::new(0));
        let worker_latest_id = Arc::clone(&latest_id);

        let worker_thread = thread::Builder::new()
            .name("green-edge-worker".to_owned())
            .spawn(move || worker_loop(job_rx, message_tx, worker_latest_id))
            .expect("failed to start background image worker");

        Self {
            job_tx,
            message_rx,
            latest_id,
            _thread: worker_thread,
        }
    }
}


struct SourceRequest {
    id: u64,
    source: String,
    camera_framesize: i32,
}

struct SourceLoaded {
    id: u64,
    source: String,
    resolved_source: String,
    image_path: Option<PathBuf>,
    rgba: RgbaImage,
    gray: GrayImage,
}

enum SourceMessage {
    Loaded(SourceLoaded),
    Failed {
        id: u64,
        source: String,
        error: String,
    },
}

struct SourceWorker {
    job_tx: mpsc::Sender<SourceRequest>,
    message_rx: mpsc::Receiver<SourceMessage>,
    _thread: thread::JoinHandle<()>,
}

impl SourceWorker {
    fn spawn() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<SourceRequest>();
        let (message_tx, message_rx) = mpsc::channel::<SourceMessage>();
        let worker_thread = thread::Builder::new()
            .name("image-source-worker".to_owned())
            .spawn(move || source_worker_loop(job_rx, message_tx))
            .expect("failed to start image source worker");

        Self {
            job_tx,
            message_rx,
            _thread: worker_thread,
        }
    }
}

struct EdgeApp {
    image_path: Option<PathBuf>,
    image_source: Option<String>,
    source_input: String,
    source_history: VecDeque<String>,
    camera_framesize: i32,

    current_profile: String,
    profile_name_input: String,
    available_profiles: Vec<String>,
    config_dir: PathBuf,
    profile_dirty: bool,
    last_profile_save: Instant,

    source_worker: SourceWorker,
    next_source_id: u64,
    active_source_id: u64,
    source_loading: bool,
    original_rgba: Option<Arc<RgbaImage>>,
    original_gray: Option<Arc<GrayImage>>,
    original_texture: Option<egui::TextureHandle>,
    composite_texture: Option<egui::TextureHandle>,
    composite_rgba: Option<RgbaImage>,

    shape_layer: GreenShapeLayer,
    green_overlay: GreenAreaOverlay,
    green_shape: Option<GreenShapeResult>,
    edge_layers: Vec<EdgeLayer>,

    preblur_sigma: f32,
    dimness: f32,
    black_background: bool,
    update_while_dragging: bool,
    dirty: bool,

    original_scale: f32,
    composite_scale: f32,

    unique_edge_pixels: usize,
    status: String,
    error: Option<String>,

    worker: ProcessingWorker,
    next_job_id: u64,
    active_job_id: u64,
    processing: bool,
    progress: f32,
    progress_stage: String,
}

impl EdgeApp {
    fn new(cc: &eframe::CreationContext<'_>, initial_path: Option<PathBuf>) -> Self {
        configure_black_visuals(&cc.egui_ctx);

        let config_dir = app_config_dir();
        let current_profile = load_active_profile_name(&config_dir)
            .unwrap_or_else(|| DEFAULT_PROFILE_NAME.to_owned());
        let current_profile_path = profile_file_path(&config_dir, &current_profile);
        let mut initial_profile = load_profile(&current_profile_path).unwrap_or_else(|| {
            let mut profile = ProfileSnapshot::default();
            // One-time compatibility with v0.3.x global history.
            profile.source_history = load_source_history(&legacy_source_history_path());
            profile
        });
        if initial_profile.edge_layers.is_empty() {
            initial_profile.edge_layers.push(EdgeLayer::new(0));
        }
        let mut available_profiles = list_profile_names(&config_dir);
        if !available_profiles.iter().any(|name| name == &current_profile) {
            available_profiles.push(current_profile.clone());
        }
        available_profiles.sort();
        available_profiles.dedup();

        let mut app = Self {
            image_path: None,
            image_source: None,
            source_input: initial_profile.source_input.clone(),
            source_history: initial_profile.source_history.clone(),
            camera_framesize: initial_profile.camera_framesize,
            current_profile: current_profile.clone(),
            profile_name_input: current_profile,
            available_profiles,
            config_dir,
            profile_dirty: false,
            last_profile_save: Instant::now(),
            source_worker: SourceWorker::spawn(),
            next_source_id: 0,
            active_source_id: 0,
            source_loading: false,
            original_rgba: None,
            original_gray: None,
            original_texture: None,
            composite_texture: None,
            composite_rgba: None,
            shape_layer: initial_profile.shape_layer,
            green_overlay: initial_profile.green_overlay,
            green_shape: None,
            edge_layers: initial_profile.edge_layers,
            preblur_sigma: initial_profile.preblur_sigma,
            dimness: initial_profile.dimness,
            black_background: initial_profile.black_background,
            update_while_dragging: initial_profile.update_while_dragging,
            dirty: false,
            original_scale: initial_profile.original_scale,
            composite_scale: initial_profile.composite_scale,
            unique_edge_pixels: 0,
            status: "Open a local image, drop one here, or enter an ESP32 camera address.".to_owned(),
            error: None,
            worker: ProcessingWorker::spawn(),
            next_job_id: 0,
            active_job_id: 0,
            processing: false,
            progress: 0.0,
            progress_stage: "Idle".to_owned(),
        };

        // Ensure even the implicit default profile exists on disk and remember it as active.
        app.profile_dirty = true;
        app.flush_profile_save();

        if let Some(path) = initial_path {
            app.source_input = path.display().to_string();
            app.load_source(app.source_input.clone());
        }

        app
    }

    fn load_source(&mut self, source: String) {
        let source = source.trim().to_owned();
        if source.is_empty() {
            self.error = Some("Enter an image path or controller address first.".to_owned());
            return;
        }

        self.source_input = source.clone();
        self.next_source_id = self.next_source_id.wrapping_add(1).max(1);
        self.active_source_id = self.next_source_id;
        self.source_loading = true;
        self.error = None;
        self.status = format!("Loading {source} …");

        if let Err(error) = self.source_worker.job_tx.send(SourceRequest {
            id: self.active_source_id,
            source,
            camera_framesize: self.camera_framesize,
        }) {
            self.source_loading = false;
            self.error = Some(format!("Image source worker stopped: {error}"));
        }
    }

    fn finish_loaded_source(&mut self, loaded: SourceLoaded, ctx: &egui::Context) {
        self.original_texture = Some(ctx.load_texture(
            "original-image",
            rgba_to_color_image(&loaded.rgba),
            egui::TextureOptions::LINEAR,
        ));
        self.image_path = loaded.image_path;
        self.image_source = Some(loaded.source.clone());
        self.original_rgba = Some(Arc::new(loaded.rgba));
        self.original_gray = Some(Arc::new(loaded.gray));
        self.original_scale = 1.0;
        self.composite_scale = 1.0;
        self.error = None;
        self.source_loading = false;
        self.status = if loaded.source == loaded.resolved_source {
            format!("Loaded {}", loaded.source)
        } else {
            format!("Loaded {} via {}", loaded.source, loaded.resolved_source)
        };
        self.remember_source(loaded.source);
        self.dirty = true;
        self.schedule_recompute();
    }

    fn remember_source(&mut self, source: String) {
        self.source_history.retain(|entry| entry != &source);
        self.source_history.push_front(source.clone());
        self.source_history.truncate(SOURCE_HISTORY_LIMIT);
        self.source_input = source;
        self.mark_profile_dirty();
    }

    fn profile_snapshot(&self) -> ProfileSnapshot {
        ProfileSnapshot {
            source_input: self.source_input.clone(),
            source_history: self.source_history.clone(),
            camera_framesize: self.camera_framesize,
            shape_layer: self.shape_layer.clone(),
            green_overlay: self.green_overlay.clone(),
            edge_layers: self.edge_layers.clone(),
            preblur_sigma: self.preblur_sigma,
            dimness: self.dimness,
            black_background: self.black_background,
            update_while_dragging: self.update_while_dragging,
            original_scale: self.original_scale,
            composite_scale: self.composite_scale,
        }
    }

    fn mark_profile_dirty(&mut self) {
        self.profile_dirty = true;
    }

    fn flush_profile_save(&mut self) {
        if !self.profile_dirty {
            return;
        }
        let path = profile_file_path(&self.config_dir, &self.current_profile);
        match save_profile(&path, &self.profile_snapshot()) {
            Ok(()) => {
                let _ = save_active_profile_name(&self.config_dir, &self.current_profile);
                self.profile_dirty = false;
                self.last_profile_save = Instant::now();
            }
            Err(error) => {
                self.error = Some(format!("Failed to save profile '{}': {error:#}", self.current_profile));
            }
        }
    }

    fn maybe_autosave_profile(&mut self) {
        if self.profile_dirty && self.last_profile_save.elapsed() >= PROFILE_AUTOSAVE_DELAY {
            self.flush_profile_save();
        }
    }

    fn apply_profile(&mut self, name: String) {
        if name == self.current_profile {
            return;
        }
        self.flush_profile_save();
        let path = profile_file_path(&self.config_dir, &name);
        let Some(profile) = load_profile(&path) else {
            self.error = Some(format!("Unable to load profile '{name}'"));
            return;
        };

        self.current_profile = name.clone();
        self.profile_name_input = name.clone();
        self.source_input = profile.source_input;
        self.source_history = profile.source_history;
        self.camera_framesize = profile.camera_framesize;
        self.shape_layer = profile.shape_layer;
        self.green_overlay = profile.green_overlay;
        self.edge_layers = if profile.edge_layers.is_empty() {
            vec![EdgeLayer::new(0)]
        } else {
            profile.edge_layers
        };
        self.preblur_sigma = profile.preblur_sigma;
        self.dimness = profile.dimness;
        self.black_background = profile.black_background;
        self.update_while_dragging = profile.update_while_dragging;
        self.original_scale = profile.original_scale;
        self.composite_scale = profile.composite_scale;
        self.profile_dirty = false;
        self.last_profile_save = Instant::now();
        let _ = save_active_profile_name(&self.config_dir, &self.current_profile);
        self.error = None;
        self.status = format!("Loaded profile '{}'", self.current_profile);
        self.dirty = self.original_rgba.is_some();
        if self.dirty {
            self.schedule_recompute();
        }
    }

    fn create_or_overwrite_profile(&mut self, requested_name: String) {
        let name = requested_name.trim();
        if name.is_empty() {
            self.error = Some("Enter a profile name first.".to_owned());
            return;
        }
        let name = name.to_owned();
        let path = profile_file_path(&self.config_dir, &name);
        if let Err(error) = save_profile(&path, &self.profile_snapshot()) {
            self.error = Some(format!("Failed to save profile '{name}': {error:#}"));
            return;
        }
        self.current_profile = name.clone();
        self.profile_name_input = name.clone();
        if !self.available_profiles.iter().any(|entry| entry == &name) {
            self.available_profiles.push(name.clone());
            self.available_profiles.sort();
        }
        self.profile_dirty = false;
        self.last_profile_save = Instant::now();
        let _ = save_active_profile_name(&self.config_dir, &name);
        self.error = None;
        self.status = format!("Saved profile '{name}'");
    }

    fn delete_current_profile(&mut self) {
        if self.available_profiles.len() <= 1 {
            self.error = Some("At least one profile must remain.".to_owned());
            return;
        }
        let deleting = self.current_profile.clone();
        let path = profile_file_path(&self.config_dir, &deleting);
        if let Err(error) = fs::remove_file(&path) {
            if error.kind() != std::io::ErrorKind::NotFound {
                self.error = Some(format!("Failed to delete profile '{deleting}': {error}"));
                return;
            }
        }
        self.available_profiles.retain(|name| name != &deleting);
        let next = self.available_profiles.first().cloned().unwrap_or_else(|| DEFAULT_PROFILE_NAME.to_owned());
        self.profile_dirty = false;
        self.current_profile.clear();
        self.apply_profile(next);
    }

    fn poll_source_worker(&mut self, ctx: &egui::Context) {
        while let Ok(message) = self.source_worker.message_rx.try_recv() {
            match message {
                SourceMessage::Loaded(loaded) if loaded.id == self.active_source_id => {
                    self.finish_loaded_source(loaded, ctx);
                }
                SourceMessage::Failed { id, source, error } if id == self.active_source_id => {
                    self.source_loading = false;
                    self.error = Some(format!("Failed to load {source}: {error}"));
                }
                _ => {}
            }
        }

        if self.source_loading {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }

    fn processing_settings(&self) -> ProcessingSettings {
        ProcessingSettings {
            shape_layer: self.shape_layer.clone(),
            green_overlay: self.green_overlay.clone(),
            edge_layers: self.edge_layers.clone(),
            preblur_sigma: self.preblur_sigma,
            dimness: self.dimness,
            black_background: self.black_background,
        }
    }

    fn schedule_recompute(&mut self) {
        let (Some(original_rgba), Some(original_gray)) =
            (self.original_rgba.as_ref(), self.original_gray.as_ref())
        else {
            return;
        };

        self.next_job_id = self.next_job_id.wrapping_add(1).max(1);
        self.active_job_id = self.next_job_id;
        self.worker
            .latest_id
            .store(self.active_job_id, Ordering::Release);

        let request = ProcessingRequest {
            id: self.active_job_id,
            original_rgba: Arc::clone(original_rgba),
            original_gray: Arc::clone(original_gray),
            settings: self.processing_settings(),
        };

        match self.worker.job_tx.send(request) {
            Ok(()) => {
                self.processing = true;
                self.progress = 0.0;
                self.progress_stage = "Queued".to_owned();
                self.error = None;
                self.dirty = false;
            }
            Err(error) => {
                self.processing = false;
                self.error = Some(format!("Background worker stopped: {error}"));
            }
        }
    }

    fn poll_worker(&mut self, ctx: &egui::Context) {
        while let Ok(message) = self.worker.message_rx.try_recv() {
            match message {
                WorkerMessage::Progress { id, value, stage } if id == self.active_job_id => {
                    self.processing = true;
                    self.progress = value.clamp(0.0, 1.0);
                    self.progress_stage = stage;
                }
                WorkerMessage::Finished { id, result } if id == self.active_job_id => {
                    self.processing = false;
                    self.progress = 1.0;
                    self.progress_stage = "Complete".to_owned();
                    self.composite_rgba = Some(result.composite.clone());
                    self.green_shape = Some(result.green_shape.clone());
                    self.unique_edge_pixels = result.unique_edge_pixels;

                    for (layer, count) in self.edge_layers.iter_mut().zip(result.edge_counts) {
                        layer.edge_pixels = count;
                    }

                    let color_image = rgba_to_color_image(&result.composite);
                    match self.composite_texture.as_mut() {
                        Some(texture) => {
                            texture.set(color_image, egui::TextureOptions::LINEAR);
                        }
                        None => {
                            self.composite_texture = Some(ctx.load_texture(
                                "composite-image",
                                color_image,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                    }

                    self.status = completion_status(
                        self.green_shape.as_ref(),
                        self.unique_edge_pixels,
                    );
                    self.error = None;
                }
                WorkerMessage::Failed { id, error } if id == self.active_job_id => {
                    self.processing = false;
                    self.progress_stage = "Failed".to_owned();
                    self.error = Some(error);
                }
                _ => {
                    // Stale jobs are deliberately ignored. Latest slider state wins.
                }
            }
        }

        if self.processing {
            ctx.request_repaint_after(Duration::from_millis(33));
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
            self.load_source(path.display().to_string());
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Green shape + edge composer");
        ui.add_space(6.0);

        ui.label("Profile");
        ui.small("Each profile persists camera resolution, layer settings, preview settings, last source, and its own source history.");
        let mut switch_to_profile: Option<String> = None;
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("profile-selector")
                .selected_text(&self.current_profile)
                .show_ui(ui, |ui| {
                    for name in self.available_profiles.clone() {
                        if ui
                            .selectable_label(name == self.current_profile, &name)
                            .clicked()
                        {
                            switch_to_profile = Some(name);
                            ui.close();
                        }
                    }
                });
            if ui.button("Save now").clicked() {
                self.mark_profile_dirty();
                self.flush_profile_save();
            }
            if ui
                .add_enabled(self.available_profiles.len() > 1, egui::Button::new("Delete"))
                .clicked()
            {
                self.delete_current_profile();
            }
        });
        if let Some(name) = switch_to_profile {
            self.apply_profile(name);
        }
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.profile_name_input)
                    .hint_text("new profile name")
                    .desired_width(190.0),
            );
            if ui.button("Save as / copy").clicked() {
                self.create_or_overwrite_profile(self.profile_name_input.clone());
            }
        });

        ui.separator();
        ui.label("Image source");
        ui.small("Local path, controller IP/hostname, or full http:// URL. Bare host -> /capture; a full URL is used exactly as typed.");

        let source_response = ui.add(
            egui::TextEdit::singleline(&mut self.source_input)
                .hint_text("192.168.1.42  |  esp32cam.local  |  /path/image.jpg")
                .desired_width(f32::INFINITY),
        );
        let enter_pressed = source_response.lost_focus()
            && ui.input(|input| input.key_pressed(egui::Key::Enter));

        ui.horizontal(|ui| {
            ui.label("Camera resolution");
            let current_resolution = camera_resolution_by_framesize(self.camera_framesize);
            let mut selected_framesize = self.camera_framesize;
            egui::ComboBox::from_id_salt("camera-resolution")
                .selected_text(current_resolution.label)
                .show_ui(ui, |ui| {
                    for resolution in CAMERA_RESOLUTIONS {
                        ui.selectable_value(
                            &mut selected_framesize,
                            resolution.framesize,
                            resolution.label,
                        );
                    }
                });
            if selected_framesize != self.camera_framesize {
                self.camera_framesize = selected_framesize;
                self.mark_profile_dirty();
            }
        });
        ui.small("For CameraWebServer this sends /control?var=framesize&val=N before /capture. Local files are not resized.");

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.source_loading, egui::Button::new("Load source"))
                .clicked()
                || enter_pressed
            {
                self.load_source(self.source_input.clone());
            }

            if ui.button("Browse…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Open image")
                    .add_filter(
                        "Image",
                        &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
                    )
                    .pick_file()
                {
                    self.source_input = path.display().to_string();
                    self.load_source(self.source_input.clone());
                }
            }

            let history_label = if self.source_history.is_empty() {
                "History".to_owned()
            } else {
                format!("History ({})", self.source_history.len())
            };
            egui::ComboBox::from_id_salt("source-history")
                .selected_text(history_label)
                .show_ui(ui, |ui| {
                    if self.source_history.is_empty() {
                        ui.label("No previous sources yet");
                    } else {
                        let entries = self.source_history.iter().cloned().collect::<Vec<_>>();
                        for entry in entries {
                            if ui.selectable_label(false, &entry).clicked() {
                                self.source_input = entry.clone();
                                self.load_source(entry);
                                ui.close();
                            }
                        }
                        ui.separator();
                        if ui.button("Clear history").clicked() {
                            self.source_history.clear();
                            self.mark_profile_dirty();
                            ui.close();
                        }
                    }
                });
        });

        if self.source_loading {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.small("Fetching/decoding image…");
            });
        }

        if ui
            .add_enabled(
                self.composite_rgba.is_some(),
                egui::Button::new("Save composite…"),
            )
            .clicked()
        {
            self.save_composite();
        }

        if ui.button("Sync both scales").clicked() {
            let scale = self
                .original_scale
                .max(self.composite_scale)
                .clamp(MIN_SCALE, MAX_SCALE);
            self.original_scale = scale;
            self.composite_scale = scale;
            self.mark_profile_dirty();
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
                    .text("Original dimness")
                    .fixed_decimals(2),
            )
            .changed();
        changed |= ui
            .checkbox(&mut self.black_background, "Black preview base")
            .changed();
        changed |= ui
            .checkbox(
                &mut self.update_while_dragging,
                "Update while dragging sliders",
            )
            .changed();

        ui.separator();
        ui.collapsing("Preview scaling", |ui| {
            ui.small("Hover a preview and rotate the mouse wheel to zoom it.");
            let mut preview_changed = false;
            preview_changed |= ui
                .add(
                    egui::Slider::new(&mut self.original_scale, MIN_SCALE..=MAX_SCALE)
                        .text("Original scale")
                        .fixed_decimals(2),
                )
                .changed();
            preview_changed |= ui
                .add(
                    egui::Slider::new(&mut self.composite_scale, MIN_SCALE..=MAX_SCALE)
                        .text("Overlay scale")
                        .fixed_decimals(2),
                )
                .changed();
            if preview_changed {
                self.mark_profile_dirty();
            }
        });

        ui.separator();
        ui.collapsing("Layer 0 — closed green shape (locked role)", |ui| {
            changed |= ui.checkbox(&mut self.shape_layer.enabled, "Enabled").changed();
            ui.small(
                "This layer always selects the connected green shape containing, or nearest to, the weighted center of green pixels.",
            );
            changed |= ui
                .add(
                    egui::Slider::new(
                        &mut self.shape_layer.green_excess_threshold,
                        0.0..=150.0,
                    )
                    .text("Green excess threshold")
                    .fixed_decimals(1),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(
                        &mut self.shape_layer.green_ratio_threshold,
                        0.0..=1.0,
                    )
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
                .checkbox(&mut self.shape_layer.show_holes, "Detect and outline holes")
                .changed();
            changed |= ui
                .checkbox(&mut self.shape_layer.show_center, "Show weighted center")
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.shape_layer.outline_opacity, 0..=255)
                        .text("Outline opacity"),
                )
                .changed();
            ui.horizontal(|ui| {
                ui.label("Outline color");
                changed |= ui
                    .color_edit_button_srgba(&mut self.shape_layer.outline_color)
                    .changed();
            });

            if let Some(shape) = self.green_shape.as_ref() {
                if shape.found {
                    ui.colored_label(
                        self.shape_layer.outline_color,
                        format!(
                            "Area: {} px ({:.2}%) | holes: {} px ({:.2}%) in {} hole(s)",
                            shape.area_pixels,
                            shape.area_percent(),
                            shape.hole_pixels,
                            shape.hole_percent(),
                            shape.hole_count,
                        ),
                    );
                    if let Some((cx, cy)) = shape.weighted_center {
                        ui.label(format!("Weighted green center: ({cx:.1}, {cy:.1})"));
                    }
                } else {
                    ui.colored_label(
                        egui::Color32::LIGHT_RED,
                        "No closed green shape found.",
                    );
                }
            }
        });

        ui.separator();
        ui.collapsing("Special green-area transparency layer", |ui| {
            ui.small(
                "This layer is generated from Layer 0's selected shape. It paints a translucent area and can also preserve that area as alpha in the saved PNG.",
            );
            changed |= ui
                .checkbox(&mut self.green_overlay.enabled, "Render green-area overlay")
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.green_overlay.opacity, 0..=255)
                        .text("Overlay opacity"),
                )
                .changed();
            ui.horizontal(|ui| {
                ui.label("Overlay color");
                changed |= ui
                    .color_edit_button_srgba(&mut self.green_overlay.color)
                    .changed();
            });
            changed |= ui
                .checkbox(
                    &mut self.green_overlay.output_alpha_enabled,
                    "Use shape as output alpha mask",
                )
                .changed();
            changed |= ui
                .add_enabled(
                    self.green_overlay.output_alpha_enabled,
                    egui::Slider::new(&mut self.green_overlay.output_alpha, 0..=255)
                        .text("Shape output alpha"),
                )
                .changed();
            ui.small("0 = fully transparent shape, 255 = fully opaque. Holes remain holes.");
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.heading("Extra edge layers");
            if ui.button("Add layer").clicked() {
                self.add_layer();
                changed = true;
            }
        });
        ui.small(
            "Thresholds are reduced inside and near the Layer 0 green shape. Each edge layer is processed independently and composited with its own opacity.",
        );

        let mut remove_index = None;
        for (index, layer) in self.edge_layers.iter_mut().enumerate() {
            ui.separator();
            ui.collapsing(layer.name.clone(), |ui| {
                ui.horizontal(|ui| {
                    changed |= ui.checkbox(&mut layer.enabled, "Enabled").changed();
                    if ui.button("Delete layer").clicked() {
                        remove_index = Some(index);
                    }
                });
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut layer.low_threshold,
                            0.0..=MAX_EDGE_THRESHOLD,
                        )
                        .text("Low threshold")
                        .fixed_decimals(1),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut layer.high_threshold,
                            0.0..=MAX_EDGE_THRESHOLD,
                        )
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
                changed |= ui
                    .add(
                        egui::Slider::new(&mut layer.opacity, 0..=255)
                            .text("Layer opacity"),
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
            self.mark_profile_dirty();
        }

        let pointer_down = ui.input(|input| input.pointer.primary_down());
        if self.dirty && (self.update_while_dragging || !pointer_down) {
            self.schedule_recompute();
        }

        if ui
            .add_enabled(self.dirty, egui::Button::new("Apply parameters"))
            .clicked()
        {
            self.schedule_recompute();
        }

        ui.separator();
        if let Some(source) = self.image_source.as_ref() {
            ui.label("Current input");
            ui.monospace(source);
        }
        if let Some(gray) = self.original_gray.as_ref() {
            ui.label(format!("Resolution: {} × {}", gray.width(), gray.height()));
        }
        ui.label(format!("Unique overlay outline pixels: {}", self.unique_edge_pixels));
        ui.small("Renderer: wgpu (Vulkan is preferred on Linux via WGPU_BACKEND=vulkan).");
    }

    fn previews(&mut self, ui: &mut egui::Ui) {
        let mut scale_changed = false;
        ui.columns(2, |columns| {
            scale_changed |= show_zoomable_texture(
                &mut columns[0],
                "Original",
                self.original_texture.as_ref(),
                &mut self.original_scale,
            );
            scale_changed |= show_zoomable_texture(
                &mut columns[1],
                "Overlay / alpha composite",
                self.composite_texture.as_ref(),
                &mut self.composite_scale,
            );
        });
        if scale_changed {
            self.mark_profile_dirty();
        }
    }
}

impl Drop for EdgeApp {
    fn drop(&mut self) {
        self.flush_profile_save();
    }
}

impl eframe::App for EdgeApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_dropped_files(&ctx);
        self.poll_source_worker(&ctx);
        self.poll_worker(&ctx);

        egui::Panel::bottom("status-bar").show(ui, |ui| {
            if self.source_loading {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(&self.status);
                });
            } else if self.processing {
                ui.horizontal(|ui| {
                    ui.label(&self.progress_stage);
                    ui.add(
                        egui::ProgressBar::new(self.progress)
                            .show_percentage()
                            .desired_width(260.0),
                    );
                    ui.label("background worker");
                });
            } else if let Some(error) = self.error.as_ref() {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            } else {
                ui.label(&self.status);
            }
        });

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(370.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.controls(ui);
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            self.previews(ui);
        });

        self.maybe_autosave_profile();
    }
}

fn worker_loop(
    job_rx: mpsc::Receiver<ProcessingRequest>,
    message_tx: mpsc::Sender<WorkerMessage>,
    latest_id: Arc<AtomicU64>,
) {
    while let Ok(mut request) = job_rx.recv() {
        // Coalesce queued slider updates before starting expensive work.
        while let Ok(newer) = job_rx.try_recv() {
            request = newer;
        }

        let id = request.id;
        match process_request(&request, &message_tx, &latest_id) {
            Ok(Some(result)) => {
                let _ = message_tx.send(WorkerMessage::Finished { id, result });
            }
            Ok(None) => {
                // Cancelled because a newer parameter snapshot exists.
            }
            Err(error) => {
                let _ = message_tx.send(WorkerMessage::Failed {
                    id,
                    error: format!("Processing failed: {error:#}"),
                });
            }
        }
    }
}

fn process_request(
    request: &ProcessingRequest,
    progress_tx: &mpsc::Sender<WorkerMessage>,
    latest_id: &AtomicU64,
) -> Result<Option<ProcessingResult>> {
    let id = request.id;
    let cancelled = || latest_id.load(Ordering::Acquire) != id;

    send_progress(progress_tx, id, 0.02, "Preparing image");
    let width = request.original_gray.width();
    let height = request.original_gray.height();
    let intensities = grayscale_values(&request.original_gray);
    if cancelled() {
        return Ok(None);
    }

    send_progress(progress_tx, id, 0.08, "Blurring grayscale image");
    let blurred = if request.settings.preblur_sigma > 0.01 {
        gaussian_blur_parallel(
            &intensities,
            width,
            height,
            request.settings.preblur_sigma,
            latest_id,
            id,
        )
    } else {
        Some(intensities)
    };
    let Some(blurred) = blurred else {
        return Ok(None);
    };

    send_progress(progress_tx, id, 0.20, "Finding weighted green center and shape");
    let green_shape = detect_green_shape(
        &request.original_rgba,
        &request.settings.shape_layer,
        latest_id,
        id,
    );
    let Some(green_shape) = green_shape else {
        return Ok(None);
    };

    send_progress(progress_tx, id, 0.40, "Creating dimmed base and green-area layer");
    let mut composite = dimmed_original_parallel(
        &request.original_rgba,
        request.settings.dimness,
        request.settings.black_background,
    );

    if request.settings.shape_layer.enabled && green_shape.found {
        if request.settings.green_overlay.enabled {
            alpha_fill_mask_parallel(
                &mut composite,
                &green_shape.mask,
                request.settings.green_overlay.color,
                request.settings.green_overlay.opacity,
            );
        }

        paint_mask_parallel(
            &mut composite,
            &green_shape.boundary,
            request.settings.shape_layer.outline_color,
            request.settings.shape_layer.outline_opacity,
        );

        if request.settings.shape_layer.show_holes {
            paint_mask_parallel(
                &mut composite,
                &green_shape.hole_boundary,
                brighten(request.settings.shape_layer.outline_color, 0.70),
                request.settings.shape_layer.outline_opacity,
            );
        }

        if request.settings.shape_layer.show_center {
            if let Some((cx, cy)) = green_shape.weighted_center {
                draw_cross(
                    &mut composite,
                    cx.round() as i32,
                    cy.round() as i32,
                    7,
                    egui::Color32::WHITE,
                    255,
                );
            }
        }
    }

    if cancelled() {
        return Ok(None);
    }

    let enabled_count = request
        .settings
        .edge_layers
        .iter()
        .filter(|layer| layer.enabled)
        .count();
    let completed_layers = AtomicUsize::new(0);

    send_progress(progress_tx, id, 0.50, "Computing edge layers");
    let edge_outputs: Vec<Option<Vec<bool>>> = request
        .settings
        .edge_layers
        .par_iter()
        .map(|layer| {
            if !layer.enabled || latest_id.load(Ordering::Acquire) != id {
                return None;
            }

            let output = adaptive_edge_mask(
                &blurred,
                width,
                height,
                layer.low_threshold.min(layer.high_threshold),
                layer.high_threshold.max(layer.low_threshold),
                layer.reduction_strength,
                layer.reduction_radius,
                &green_shape.distance_map,
                latest_id,
                id,
            );

            let completed = completed_layers.fetch_add(1, Ordering::AcqRel) + 1;
            let fraction = if enabled_count == 0 {
                1.0
            } else {
                completed as f32 / enabled_count as f32
            };
            send_progress(
                progress_tx,
                id,
                0.50 + fraction * 0.35,
                &format!("Edge layers {completed}/{enabled_count}"),
            );
            output
        })
        .collect();

    if cancelled() {
        return Ok(None);
    }

    send_progress(progress_tx, id, 0.88, "Compositing colored edge layers");
    let total_pixels = (width * height) as usize;
    let mut union_mask = vec![false; total_pixels];
    if request.settings.shape_layer.enabled && green_shape.found {
        merge_union(&mut union_mask, &green_shape.boundary);
        if request.settings.shape_layer.show_holes {
            merge_union(&mut union_mask, &green_shape.hole_boundary);
        }
    }

    let mut edge_counts = Vec::with_capacity(request.settings.edge_layers.len());
    for (layer, maybe_mask) in request.settings.edge_layers.iter().zip(edge_outputs.iter()) {
        if let Some(mask) = maybe_mask {
            let count = mask.iter().filter(|&&value| value).count();
            edge_counts.push(count);
            paint_mask_parallel(&mut composite, mask, layer.color, layer.opacity);
            merge_union(&mut union_mask, mask);
        } else {
            edge_counts.push(0);
        }
    }

    if request.settings.green_overlay.output_alpha_enabled && green_shape.found {
        apply_output_alpha_parallel(
            &mut composite,
            &green_shape.mask,
            request.settings.green_overlay.output_alpha,
        );
    }

    let unique_edge_pixels = union_mask.iter().filter(|&&value| value).count();
    send_progress(progress_tx, id, 0.98, "Finalizing RGBA texture");

    if cancelled() {
        return Ok(None);
    }

    send_progress(progress_tx, id, 1.0, "Complete");
    Ok(Some(ProcessingResult {
        composite,
        green_shape,
        edge_counts,
        unique_edge_pixels,
    }))
}

fn send_progress(
    sender: &mpsc::Sender<WorkerMessage>,
    id: u64,
    value: f32,
    stage: &str,
) {
    let _ = sender.send(WorkerMessage::Progress {
        id,
        value,
        stage: stage.to_owned(),
    });
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
) -> bool {
    ui.heading(title);
    ui.separator();

    let Some(texture) = texture else {
        ui.centered_and_justified(|ui| {
            ui.label("No image loaded");
        });
        return false;
    };

    let mut changed = false;
    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let display_size = texture.size_vec2() * *scale;
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
                    changed = true;
                }
            }
        });

    ui.small(format!("Scale: {:.2}×", *scale));
    changed
}


fn source_worker_loop(job_rx: mpsc::Receiver<SourceRequest>, message_tx: mpsc::Sender<SourceMessage>) {
    while let Ok(mut request) = job_rx.recv() {
        while let Ok(newer) = job_rx.try_recv() {
            request = newer;
        }

        let id = request.id;
        let source = request.source.clone();
        match load_source_data(&source, request.camera_framesize) {
            Ok((resolved_source, image_path, rgba, gray)) => {
                let _ = message_tx.send(SourceMessage::Loaded(SourceLoaded {
                    id,
                    source,
                    resolved_source,
                    image_path,
                    rgba,
                    gray,
                }));
            }
            Err(error) => {
                let _ = message_tx.send(SourceMessage::Failed {
                    id,
                    source,
                    error: format!("{error:#}"),
                });
            }
        }
    }
}

fn load_source_data(source: &str, camera_framesize: i32) -> Result<(String, Option<PathBuf>, RgbaImage, GrayImage)> {
    if looks_like_remote_source(source) {
        let url = normalize_camera_url(source)?;
        if url_path(&url).eq_ignore_ascii_case("/stream") {
            anyhow::bail!("/stream is an MJPEG stream; use the controller address or /capture for one frame");
        }
        if camera_framesize >= 0 {
            set_camera_framesize(&url, camera_framesize)?;
        }
        let bytes = fetch_http_bytes(&url)?;
        let decoded = image::load_from_memory(&bytes)
            .with_context(|| format!("unable to decode image returned by {url}"))?;
        Ok((url, None, decoded.to_rgba8(), decoded.to_luma8()))
    } else {
        let path = expand_home_path(source);
        let (rgba, gray) = load_image_data(&path)?;
        Ok((path.display().to_string(), Some(path), rgba, gray))
    }
}

fn looks_like_remote_source(source: &str) -> bool {
    let source = source.trim();
    if source.starts_with("http://") || source.starts_with("https://") {
        return true;
    }
    if Path::new(source).exists() || source.starts_with('/') || source.starts_with("./") || source.starts_with("../") || source.starts_with('~') {
        return false;
    }

    let host_part = source.split('/').next().unwrap_or(source);
    host_part.eq_ignore_ascii_case("localhost")
        || host_part.ends_with(".local")
        || host_part.parse::<std::net::IpAddr>().is_ok()
        || host_part
            .split_once(':')
            .is_some_and(|(host, port)| !host.is_empty() && port.parse::<u16>().is_ok())
        || (!source.contains('/') && Path::new(source).extension().is_none())
}

fn normalize_camera_url(source: &str) -> Result<String> {
    let source = source.trim();
    if source.starts_with("https://") {
        anyhow::bail!("HTTPS is not supported by the built-in ESP32 fetcher; use the controller's http:// address");
    }

    // A full URL is authoritative: use exactly the path the user typed.
    // A bare controller hostname/IP is convenience shorthand for the
    // CameraWebServer still-image endpoint at /capture.
    if source.starts_with("http://") {
        let after_scheme = &source["http://".len()..];
        if after_scheme.is_empty() {
            anyhow::bail!("controller address is empty");
        }
        return Ok(source.to_owned());
    }

    if source.is_empty() {
        anyhow::bail!("controller address is empty");
    }

    let mut url = format!("http://{source}");
    if !source.contains('/') {
        url.push_str("/capture");
    }
    Ok(url)
}

fn url_path(url: &str) -> &str {
    let rest = url.strip_prefix("http://").unwrap_or(url);
    match rest.find('/') {
        Some(index) => &rest[index..],
        None => "/",
    }
}

fn set_camera_framesize(image_url: &str, framesize: i32) -> Result<()> {
    let authority = http_authority(image_url)?;
    let control_url = format!("http://{authority}/control?var=framesize&val={framesize}");
    fetch_http_bytes(&control_url)
        .with_context(|| format!("unable to set camera resolution using {control_url}"))?;
    // Give the OV sensor a brief moment to settle before requesting the new frame.
    thread::sleep(Duration::from_millis(80));
    Ok(())
}

fn http_authority(url: &str) -> Result<&str> {
    let rest = url
        .strip_prefix("http://")
        .context("only plain http:// controller URLs are supported")?;
    let end = rest.find('/').unwrap_or(rest.len());
    let authority = &rest[..end];
    if authority.is_empty() {
        anyhow::bail!("missing controller hostname in {url}");
    }
    Ok(authority)
}

fn fetch_http_bytes(url: &str) -> Result<Vec<u8>> {
    let rest = url
        .strip_prefix("http://")
        .context("only plain http:// controller URLs are supported")?;
    let (authority, path) = match rest.find('/') {
        Some(index) => (&rest[..index], &rest[index..]),
        None => (rest, "/"),
    };
    if authority.is_empty() {
        anyhow::bail!("missing controller hostname in {url}");
    }

    let (host, port) = parse_http_authority(authority)?;
    let socket_addr = (host.as_str(), port)
        .to_socket_addrs()
        .with_context(|| format!("unable to resolve {host}"))?
        .next()
        .with_context(|| format!("no address found for {host}"))?;

    let mut stream = TcpStream::connect_timeout(&socket_addr, HTTP_TIMEOUT)
        .with_context(|| format!("unable to connect to {authority}"))?;
    stream.set_read_timeout(Some(HTTP_TIMEOUT))?;
    stream.set_write_timeout(Some(HTTP_TIMEOUT))?;

    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {authority}\r\nUser-Agent: rust-edge-gui/0.4\r\nAccept: image/*\r\nConnection: close\r\n\r\n"
    )?;
    stream.flush()?;

    let response = read_http_response(&mut stream, url)?;

    let header_end = find_subslice(&response, b"\r\n\r\n")
        .context("controller returned an invalid HTTP response")?;
    let header_bytes = &response[..header_end];
    let body = &response[header_end + 4..];
    let headers = String::from_utf8_lossy(header_bytes);
    let mut lines = headers.lines();
    let status = lines.next().context("HTTP response has no status line")?;
    let status_code = status
        .split_whitespace()
        .nth(1)
        .and_then(|part| part.parse::<u16>().ok())
        .context("unable to parse HTTP status")?;
    if !(200..300).contains(&status_code) {
        anyhow::bail!("controller returned HTTP {status_code} ({status})");
    }

    let chunked = headers.lines().any(|line| {
        line.split_once(':').is_some_and(|(name, value)| {
            name.eq_ignore_ascii_case("transfer-encoding")
                && value.to_ascii_lowercase().contains("chunked")
        })
    });
    if chunked {
        decode_chunked_body(body)
    } else {
        Ok(body.to_vec())
    }
}


fn read_http_response(stream: &mut TcpStream, url: &str) -> Result<Vec<u8>> {
    let mut response = Vec::new();
    let mut buffer = [0_u8; 16 * 1024];

    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => {
                response.extend_from_slice(&buffer[..count]);
                if http_response_is_complete(&response) {
                    break;
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                // Linux commonly reports SO_RCVTIMEO expiry as EAGAIN (os error 11).
                // If we already have an HTTP response, parse what arrived rather than
                // turning that temporary socket condition into a misleading hard error.
                if !response.is_empty() {
                    break;
                }
                return Err(error).with_context(|| format!("timed out while reading {url}"));
            }
            Err(error) => {
                return Err(error).with_context(|| format!("failed while reading {url}"));
            }
        }
    }

    if response.is_empty() {
        anyhow::bail!("controller closed the connection without returning data: {url}");
    }
    Ok(response)
}

fn http_response_is_complete(response: &[u8]) -> bool {
    let Some(header_end) = find_subslice(response, b"\r\n\r\n") else {
        return false;
    };
    let body = &response[header_end + 4..];
    let headers = String::from_utf8_lossy(&response[..header_end]);

    if let Some(content_length) = headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.eq_ignore_ascii_case("content-length") {
            value.trim().parse::<usize>().ok()
        } else {
            None
        }
    }) {
        return body.len() >= content_length;
    }

    let chunked = headers.lines().any(|line| {
        line.split_once(':').is_some_and(|(name, value)| {
            name.eq_ignore_ascii_case("transfer-encoding")
                && value.to_ascii_lowercase().contains("chunked")
        })
    });
    if chunked {
        return chunked_body_is_complete(body);
    }

    false
}

fn chunked_body_is_complete(mut body: &[u8]) -> bool {
    loop {
        let Some(line_end) = find_subslice(body, b"\r\n") else {
            return false;
        };
        let size_text = match std::str::from_utf8(&body[..line_end]) {
            Ok(text) => text.split(';').next().unwrap_or("").trim(),
            Err(_) => return false,
        };
        let size = match usize::from_str_radix(size_text, 16) {
            Ok(size) => size,
            Err(_) => return false,
        };
        body = &body[line_end + 2..];

        if size == 0 {
            // The terminating zero-size chunk is sufficient for the ESP32 responses
            // we consume. Optional trailers, if any, are irrelevant to image decoding.
            return body.len() >= 2;
        }
        if body.len() < size + 2 || &body[size..size + 2] != b"\r\n" {
            return false;
        }
        body = &body[size + 2..];
    }
}

fn parse_http_authority(authority: &str) -> Result<(String, u16)> {
    if authority.starts_with('[') {
        let end = authority.find(']').context("invalid IPv6 controller address")?;
        let host = authority[1..end].to_owned();
        let port = authority[end + 1..]
            .strip_prefix(':')
            .map(|p| p.parse::<u16>())
            .transpose()
            .context("invalid HTTP port")?
            .unwrap_or(80);
        return Ok((host, port));
    }

    if let Some((host, port)) = authority.rsplit_once(':') {
        if !host.contains(':') {
            return Ok((host.to_owned(), port.parse::<u16>().context("invalid HTTP port")?));
        }
    }
    Ok((authority.to_owned(), 80))
}

fn decode_chunked_body(mut input: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let line_end = find_subslice(input, b"\r\n").context("invalid chunked HTTP body")?;
        let size_line = std::str::from_utf8(&input[..line_end]).context("invalid chunk size")?;
        let size_hex = size_line.split(';').next().unwrap_or(size_line).trim();
        let size = usize::from_str_radix(size_hex, 16).context("invalid chunk size")?;
        input = &input[line_end + 2..];
        if size == 0 {
            break;
        }
        if input.len() < size + 2 {
            anyhow::bail!("truncated chunked HTTP body");
        }
        out.extend_from_slice(&input[..size]);
        input = &input[size + 2..];
    }
    Ok(out)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

fn expand_home_path(source: &str) -> PathBuf {
    if source == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    } else if let Some(rest) = source.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(source)
}

fn app_config_dir() -> PathBuf {
    if let Some(config) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(config).join("rust-edge-gui");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".config").join("rust-edge-gui");
    }
    PathBuf::from(".rust-edge-gui")
}

fn legacy_source_history_path() -> PathBuf {
    app_config_dir().join("source-history.txt")
}

fn active_profile_path(config_dir: &Path) -> PathBuf {
    config_dir.join("active-profile.txt")
}

fn profiles_dir(config_dir: &Path) -> PathBuf {
    config_dir.join("profiles")
}

fn profile_file_path(config_dir: &Path, name: &str) -> PathBuf {
    profiles_dir(config_dir).join(format!("{}.profile", hex_encode(name.as_bytes())))
}

fn load_active_profile_name(config_dir: &Path) -> Option<String> {
    fs::read_to_string(active_profile_path(config_dir))
        .ok()
        .map(|text| text.trim().to_owned())
        .filter(|name| !name.is_empty())
}

fn save_active_profile_name(config_dir: &Path, name: &str) -> Result<()> {
    fs::create_dir_all(config_dir)
        .with_context(|| format!("unable to create config directory {}", config_dir.display()))?;
    fs::write(active_profile_path(config_dir), format!("{name}\n"))
        .with_context(|| "unable to save active profile name")?;
    Ok(())
}

fn list_profile_names(config_dir: &Path) -> Vec<String> {
    let mut names = Vec::new();
    let Ok(entries) = fs::read_dir(profiles_dir(config_dir)) else {
        return names;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("profile") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if let Ok(bytes) = hex_decode(stem) {
            if let Ok(name) = String::from_utf8(bytes) {
                if !name.trim().is_empty() {
                    names.push(name);
                }
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

fn load_source_history(path: &Path) -> VecDeque<String> {
    let Ok(text) = fs::read_to_string(path) else {
        return VecDeque::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(SOURCE_HISTORY_LIMIT)
        .map(ToOwned::to_owned)
        .collect()
}

fn save_profile(path: &Path, profile: &ProfileSnapshot) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("unable to create profile directory {}", parent.display()))?;
    }

    let mut lines = Vec::new();
    lines.push(format!("version={PROFILE_FORMAT_VERSION}"));
    lines.push(format!("source_hex={}", hex_encode(profile.source_input.as_bytes())));
    lines.push(format!("camera_framesize={}", profile.camera_framesize));
    lines.push(format!("preblur_sigma={}", profile.preblur_sigma));
    lines.push(format!("dimness={}", profile.dimness));
    lines.push(format!("black_background={}", bool_num(profile.black_background)));
    lines.push(format!("update_while_dragging={}", bool_num(profile.update_while_dragging)));
    lines.push(format!("original_scale={}", profile.original_scale));
    lines.push(format!("composite_scale={}", profile.composite_scale));

    lines.push(format!("shape.enabled={}", bool_num(profile.shape_layer.enabled)));
    lines.push(format!("shape.green_excess_threshold={}", profile.shape_layer.green_excess_threshold));
    lines.push(format!("shape.green_ratio_threshold={}", profile.shape_layer.green_ratio_threshold));
    lines.push(format!("shape.mask_grow_radius={}", profile.shape_layer.mask_grow_radius));
    lines.push(format!("shape.outline_color={}", color_to_string(profile.shape_layer.outline_color)));
    lines.push(format!("shape.outline_opacity={}", profile.shape_layer.outline_opacity));
    lines.push(format!("shape.show_holes={}", bool_num(profile.shape_layer.show_holes)));
    lines.push(format!("shape.show_center={}", bool_num(profile.shape_layer.show_center)));

    lines.push(format!("overlay.enabled={}", bool_num(profile.green_overlay.enabled)));
    lines.push(format!("overlay.color={}", color_to_string(profile.green_overlay.color)));
    lines.push(format!("overlay.opacity={}", profile.green_overlay.opacity));
    lines.push(format!("overlay.output_alpha_enabled={}", bool_num(profile.green_overlay.output_alpha_enabled)));
    lines.push(format!("overlay.output_alpha={}", profile.green_overlay.output_alpha));

    for source in profile.source_history.iter().take(SOURCE_HISTORY_LIMIT) {
        lines.push(format!("history_hex={}", hex_encode(source.as_bytes())));
    }

    lines.push(format!("edge.count={}", profile.edge_layers.len()));
    for (index, layer) in profile.edge_layers.iter().enumerate() {
        let p = format!("edge.{index}");
        lines.push(format!("{p}.enabled={}", bool_num(layer.enabled)));
        lines.push(format!("{p}.name_hex={}", hex_encode(layer.name.as_bytes())));
        lines.push(format!("{p}.low_threshold={}", layer.low_threshold));
        lines.push(format!("{p}.high_threshold={}", layer.high_threshold));
        lines.push(format!("{p}.reduction_strength={}", layer.reduction_strength));
        lines.push(format!("{p}.reduction_radius={}", layer.reduction_radius));
        lines.push(format!("{p}.color={}", color_to_string(layer.color)));
        lines.push(format!("{p}.opacity={}", layer.opacity));
    }

    let mut text = lines.join("\n");
    text.push('\n');
    let tmp = path.with_extension("profile.tmp");
    fs::write(&tmp, text).with_context(|| format!("unable to write {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("unable to replace {}", path.display()))?;
    Ok(())
}

fn load_profile(path: &Path) -> Option<ProfileSnapshot> {
    let text = fs::read_to_string(path).ok()?;
    let mut profile = ProfileSnapshot::default();
    profile.source_history.clear();

    let mut edge_count = 0usize;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "source_hex" => {
                if let Ok(bytes) = hex_decode(value) {
                    if let Ok(decoded) = String::from_utf8(bytes) {
                        profile.source_input = decoded;
                    }
                }
            }
            "camera_framesize" => profile.camera_framesize = parse_or(value, profile.camera_framesize),
            "preblur_sigma" => profile.preblur_sigma = parse_or(value, profile.preblur_sigma),
            "dimness" => profile.dimness = parse_or(value, profile.dimness),
            "black_background" => profile.black_background = parse_bool(value, profile.black_background),
            "update_while_dragging" => profile.update_while_dragging = parse_bool(value, profile.update_while_dragging),
            "original_scale" => profile.original_scale = parse_or(value, profile.original_scale),
            "composite_scale" => profile.composite_scale = parse_or(value, profile.composite_scale),
            "shape.enabled" => profile.shape_layer.enabled = parse_bool(value, profile.shape_layer.enabled),
            "shape.green_excess_threshold" => profile.shape_layer.green_excess_threshold = parse_or(value, profile.shape_layer.green_excess_threshold),
            "shape.green_ratio_threshold" => profile.shape_layer.green_ratio_threshold = parse_or(value, profile.shape_layer.green_ratio_threshold),
            "shape.mask_grow_radius" => profile.shape_layer.mask_grow_radius = parse_or(value, profile.shape_layer.mask_grow_radius),
            "shape.outline_color" => profile.shape_layer.outline_color = parse_color(value).unwrap_or(profile.shape_layer.outline_color),
            "shape.outline_opacity" => profile.shape_layer.outline_opacity = parse_or(value, profile.shape_layer.outline_opacity),
            "shape.show_holes" => profile.shape_layer.show_holes = parse_bool(value, profile.shape_layer.show_holes),
            "shape.show_center" => profile.shape_layer.show_center = parse_bool(value, profile.shape_layer.show_center),
            "overlay.enabled" => profile.green_overlay.enabled = parse_bool(value, profile.green_overlay.enabled),
            "overlay.color" => profile.green_overlay.color = parse_color(value).unwrap_or(profile.green_overlay.color),
            "overlay.opacity" => profile.green_overlay.opacity = parse_or(value, profile.green_overlay.opacity),
            "overlay.output_alpha_enabled" => profile.green_overlay.output_alpha_enabled = parse_bool(value, profile.green_overlay.output_alpha_enabled),
            "overlay.output_alpha" => profile.green_overlay.output_alpha = parse_or(value, profile.green_overlay.output_alpha),
            "history_hex" => {
                if profile.source_history.len() < SOURCE_HISTORY_LIMIT {
                    if let Ok(bytes) = hex_decode(value) {
                        if let Ok(decoded) = String::from_utf8(bytes) {
                            if !decoded.is_empty() {
                                profile.source_history.push_back(decoded);
                            }
                        }
                    }
                }
            }
            "edge.count" => edge_count = parse_or(value, 0usize),
            _ => {}
        }
    }

    profile.edge_layers = (0..edge_count).map(EdgeLayer::new).collect();
    if edge_count > 0 {
        for raw_line in text.lines() {
            let Some((key, value)) = raw_line.trim().split_once('=') else {
                continue;
            };
            let Some(rest) = key.strip_prefix("edge.") else {
                continue;
            };
            let Some((index_text, field)) = rest.split_once('.') else {
                continue;
            };
            let Ok(index) = index_text.parse::<usize>() else {
                continue;
            };
            let Some(layer) = profile.edge_layers.get_mut(index) else {
                continue;
            };
            match field {
                "enabled" => layer.enabled = parse_bool(value, layer.enabled),
                "name_hex" => {
                    if let Ok(bytes) = hex_decode(value) {
                        if let Ok(decoded) = String::from_utf8(bytes) {
                            layer.name = decoded;
                        }
                    }
                }
                "low_threshold" => layer.low_threshold = parse_or(value, layer.low_threshold),
                "high_threshold" => layer.high_threshold = parse_or(value, layer.high_threshold),
                "reduction_strength" => layer.reduction_strength = parse_or(value, layer.reduction_strength),
                "reduction_radius" => layer.reduction_radius = parse_or(value, layer.reduction_radius),
                "color" => layer.color = parse_color(value).unwrap_or(layer.color),
                "opacity" => layer.opacity = parse_or(value, layer.opacity),
                _ => {}
            }
        }
    }

    profile.original_scale = profile.original_scale.clamp(MIN_SCALE, MAX_SCALE);
    profile.composite_scale = profile.composite_scale.clamp(MIN_SCALE, MAX_SCALE);
    if !CAMERA_RESOLUTIONS.iter().any(|entry| entry.framesize == profile.camera_framesize) {
        profile.camera_framesize = -1;
    }
    Some(profile)
}

fn bool_num(value: bool) -> u8 {
    if value { 1 } else { 0 }
}

fn parse_bool(value: &str, default: bool) -> bool {
    match value.trim() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => default,
    }
}

fn parse_or<T>(value: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    value.trim().parse::<T>().unwrap_or(default)
}

fn color_to_string(color: egui::Color32) -> String {
    format!("{},{},{},{}", color.r(), color.g(), color.b(), color.a())
}

fn parse_color(value: &str) -> Option<egui::Color32> {
    let parts = value
        .split(',')
        .map(|part| part.trim().parse::<u8>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()?;
    if parts.len() != 4 {
        return None;
    }
    Some(egui::Color32::from_rgba_unmultiplied(parts[0], parts[1], parts[2], parts[3]))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn hex_decode(text: &str) -> Result<Vec<u8>> {
    if text.len() % 2 != 0 {
        anyhow::bail!("invalid hex string length");
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut index = 0;
    while index < bytes.len() {
        let high = hex_digit(bytes[index]).context("invalid hex digit")?;
        let low = hex_digit(bytes[index + 1]).context("invalid hex digit")?;
        out.push((high << 4) | low);
        index += 2;
    }
    Ok(out)
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
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

fn ensure_png_extension(mut path: PathBuf) -> PathBuf {
    if path.extension().is_none() {
        path.set_extension("png");
    }
    path
}

fn percentage(pixels: usize, width: u32, height: u32) -> f32 {
    let total = (width as usize).saturating_mul(height as usize);
    if total == 0 {
        0.0
    } else {
        pixels as f32 * 100.0 / total as f32
    }
}

fn completion_status(shape: Option<&GreenShapeResult>, unique_edges: usize) -> String {
    match shape {
        Some(shape) if shape.found => format!(
            "Green shape: {} px ({:.2}%), holes: {} px in {} hole(s), overlay outlines: {} px",
            shape.area_pixels,
            shape.area_percent(),
            shape.hole_pixels,
            shape.hole_count,
            unique_edges,
        ),
        _ => format!(
            "No green closed shape found. Overlay outlines: {} px",
            unique_edges
        ),
    }
}

fn grayscale_values(gray: &GrayImage) -> Vec<f32> {
    gray.pixels().map(|pixel| pixel.0[0] as f32).collect()
}

fn gaussian_blur_parallel(
    input: &[f32],
    width: u32,
    height: u32,
    sigma: f32,
    latest_id: &AtomicU64,
    job_id: u64,
) -> Option<Vec<f32>> {
    if sigma <= 0.01 {
        return Some(input.to_vec());
    }

    let radius = (sigma * 3.0).ceil() as i32;
    let kernel = gaussian_kernel_1d(radius, sigma);
    let w = width as usize;
    let h = height as usize;

    let mut temp = vec![0.0; input.len()];
    temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if latest_id.load(Ordering::Acquire) != job_id {
            return;
        }
        for (x, output) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for k in -radius..=radius {
                let xx = (x as i32 + k).clamp(0, width as i32 - 1) as usize;
                sum += input[y * w + xx] * kernel[(k + radius) as usize];
            }
            *output = sum;
        }
    });

    if latest_id.load(Ordering::Acquire) != job_id {
        return None;
    }

    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![0.0; w];
            for x in 0..w {
                let mut sum = 0.0;
                for k in -radius..=radius {
                    let yy = (y as i32 + k).clamp(0, height as i32 - 1) as usize;
                    sum += temp[yy * w + x] * kernel[(k + radius) as usize];
                }
                row[x] = sum;
            }
            row
        })
        .collect();

    if latest_id.load(Ordering::Acquire) != job_id {
        return None;
    }

    Some(rows.into_iter().flatten().collect())
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
    latest_id: &AtomicU64,
    job_id: u64,
) -> Option<Vec<bool>> {
    let w = width as usize;
    let h = height as usize;
    let total = w * h;
    let reduction_strength = reduction_strength.clamp(0.0, 0.95);

    let classifications: Vec<u8> = (0..total)
        .into_par_iter()
        .map(|idx| {
            if latest_id.load(Ordering::Relaxed) != job_id {
                return 0;
            }

            let x = idx % w;
            let y = idx / w;
            if x == 0 || y == 0 || x + 1 >= w || y + 1 >= h {
                return 0;
            }

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
            let magnitude = (gx * gx + gy * gy).sqrt();

            let proximity = if reduction_radius == 0 || green_distances.len() != total {
                0.0
            } else {
                let distance = green_distances[idx];
                if !distance.is_finite() || distance > reduction_radius as f32 {
                    0.0
                } else {
                    1.0 - (distance / reduction_radius as f32).clamp(0.0, 1.0)
                }
            };

            let reduction = 1.0 - reduction_strength * proximity;
            let local_low = low_threshold * reduction;
            let local_high = high_threshold * reduction;

            if magnitude >= local_high {
                2
            } else if magnitude >= local_low {
                1
            } else {
                0
            }
        })
        .collect();

    if latest_id.load(Ordering::Acquire) != job_id {
        return None;
    }

    let mut edges = vec![false; total];
    let mut queue = VecDeque::new();
    for (idx, &classification) in classifications.iter().enumerate() {
        if classification == 2 {
            edges[idx] = true;
            queue.push_back(idx);
        }
    }

    while let Some(idx) = queue.pop_front() {
        if latest_id.load(Ordering::Relaxed) != job_id {
            return None;
        }
        let x = idx % w;
        let y = idx / w;
        for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                let nidx = ny * w + nx;
                if !edges[nidx] && classifications[nidx] == 1 {
                    edges[nidx] = true;
                    queue.push_back(nidx);
                }
            }
        }
    }

    Some(edges)
}

fn detect_green_shape(
    image: &RgbaImage,
    settings: &GreenShapeLayer,
    latest_id: &AtomicU64,
    job_id: u64,
) -> Option<GreenShapeResult> {
    let width = image.width();
    let height = image.height();
    let total = (width * height) as usize;

    let samples: Vec<(bool, f32, f32, f32)> = image
        .as_raw()
        .par_chunks_exact(4)
        .enumerate()
        .map(|(idx, pixel)| {
            if latest_id.load(Ordering::Relaxed) != job_id {
                return (false, 0.0, 0.0, 0.0);
            }

            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let green_excess = (g - r.max(b)).max(0.0);
            let green_ratio = g / (r + g + b + 1.0);
            let weight = green_excess * green_ratio.max(0.0);
            let x = (idx % width as usize) as f32;
            let y = (idx / width as usize) as f32;
            (
                green_excess >= settings.green_excess_threshold
                    && green_ratio >= settings.green_ratio_threshold,
                weight,
                x,
                y,
            )
        })
        .collect();

    if latest_id.load(Ordering::Acquire) != job_id {
        return None;
    }

    let mut raw_mask = Vec::with_capacity(total);
    let mut weighted_sum = 0.0f32;
    let mut weighted_x = 0.0f32;
    let mut weighted_y = 0.0f32;
    for (is_green, weight, x, y) in samples {
        raw_mask.push(is_green);
        weighted_sum += weight;
        weighted_x += x * weight;
        weighted_y += y * weight;
    }

    if settings.mask_grow_radius > 0 {
        raw_mask = dilate_mask(
            &raw_mask,
            width,
            height,
            settings.mask_grow_radius,
            latest_id,
            job_id,
        )?;
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
        return Some(result);
    };

    let components = find_components(&raw_mask, width, height, latest_id, job_id)?;
    if components.is_empty() {
        return Some(result);
    }

    let cx = center.0.round().clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let cy = center.1.round().clamp(0.0, height.saturating_sub(1) as f32) as u32;
    let center_idx = (cy * width + cx) as usize;

    let selected_index = if raw_mask[center_idx] {
        components
            .iter()
            .position(|component| component[center_idx])
            .unwrap_or(0)
    } else {
        components
            .iter()
            .enumerate()
            .filter_map(|(index, component)| {
                let mut sx = 0.0f32;
                let mut sy = 0.0f32;
                let mut count = 0.0f32;
                for (idx, &value) in component.iter().enumerate() {
                    if value {
                        sx += (idx % width as usize) as f32;
                        sy += (idx / width as usize) as f32;
                        count += 1.0;
                    }
                }
                if count == 0.0 {
                    None
                } else {
                    let dx = sx / count - center.0;
                    let dy = sy / count - center.1;
                    Some((index, dx * dx + dy * dy))
                }
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index)
            .unwrap_or(0)
    };

    let selected_mask = components[selected_index].clone();
    let boundary = mask_boundary(&selected_mask, width, height);
    let (hole_mask, hole_boundary, hole_count, hole_pixels) =
        detect_holes(&selected_mask, width, height, latest_id, job_id)?;
    let max_distance = 100;
    let distance_map = distance_map_bfs(
        &selected_mask,
        width,
        height,
        max_distance,
        latest_id,
        job_id,
    )?;

    result.found = true;
    result.area_pixels = selected_mask.iter().filter(|&&value| value).count();
    result.hole_pixels = hole_pixels;
    result.hole_count = hole_count;
    result.mask = selected_mask;
    result.boundary = boundary;
    result.hole_mask = hole_mask;
    result.hole_boundary = hole_boundary;
    result.distance_map = distance_map;
    Some(result)
}

fn find_components(
    mask: &[bool],
    width: u32,
    height: u32,
    latest_id: &AtomicU64,
    job_id: u64,
) -> Option<Vec<Vec<bool>>> {
    let w = width as usize;
    let h = height as usize;
    let mut visited = vec![false; mask.len()];
    let mut components = Vec::new();

    for idx in 0..mask.len() {
        if latest_id.load(Ordering::Relaxed) != job_id {
            return None;
        }
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

    components.sort_by_key(|component| {
        std::cmp::Reverse(component.iter().filter(|&&value| value).count())
    });
    Some(components)
}

fn dilate_mask(
    mask: &[bool],
    width: u32,
    height: u32,
    radius: u32,
    latest_id: &AtomicU64,
    job_id: u64,
) -> Option<Vec<bool>> {
    if radius == 0 {
        return Some(mask.to_vec());
    }

    let w = width as usize;
    let r = radius as i32;
    let r2 = r * r;
    let output: Vec<bool> = (0..mask.len())
        .into_par_iter()
        .map(|idx| {
            if latest_id.load(Ordering::Relaxed) != job_id {
                return false;
            }
            let x = idx % w;
            let y = idx / w;
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy > r2 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && nx < width as i32 && ny < height as i32 {
                        let nidx = ny as usize * w + nx as usize;
                        if mask[nidx] {
                            return true;
                        }
                    }
                }
            }
            false
        })
        .collect();

    if latest_id.load(Ordering::Acquire) != job_id {
        None
    } else {
        Some(output)
    }
}

fn mask_boundary(mask: &[bool], width: u32, height: u32) -> Vec<bool> {
    let w = width as usize;
    let h = height as usize;
    (0..mask.len())
        .into_par_iter()
        .map(|idx| {
            if !mask[idx] {
                return false;
            }
            let x = idx % w;
            let y = idx / w;
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                return true;
            }
            for ny in y - 1..=y + 1 {
                for nx in x - 1..=x + 1 {
                    if !mask[ny * w + nx] {
                        return true;
                    }
                }
            }
            false
        })
        .collect()
}

fn detect_holes(
    mask: &[bool],
    width: u32,
    height: u32,
    latest_id: &AtomicU64,
    job_id: u64,
) -> Option<(Vec<bool>, Vec<bool>, usize, usize)> {
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
        return Some((
            vec![false; mask.len()],
            vec![false; mask.len()],
            0,
            0,
        ));
    }

    let mut outside = vec![false; mask.len()];
    let mut queue = VecDeque::new();

    for x in min_x..=max_x {
        for y in [min_y, max_y] {
            let idx = y * w + x;
            if !mask[idx] && !outside[idx] {
                outside[idx] = true;
                queue.push_back(idx);
            }
        }
    }
    for y in min_y..=max_y {
        for x in [min_x, max_x] {
            let idx = y * w + x;
            if !mask[idx] && !outside[idx] {
                outside[idx] = true;
                queue.push_back(idx);
            }
        }
    }

    while let Some(current) = queue.pop_front() {
        if latest_id.load(Ordering::Relaxed) != job_id {
            return None;
        }
        let x = current % w;
        let y = current / w;
        for ny in y.saturating_sub(1)..=(y + 1).min(max_y) {
            for nx in x.saturating_sub(1)..=(x + 1).min(max_x) {
                if nx < min_x || ny < min_y {
                    continue;
                }
                let nidx = ny * w + nx;
                if !mask[nidx] && !outside[nidx] {
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
    Some((hole_mask, hole_boundary, hole_count, hole_pixels))
}

fn distance_map_bfs(
    mask: &[bool],
    width: u32,
    height: u32,
    max_distance: u32,
    latest_id: &AtomicU64,
    job_id: u64,
) -> Option<Vec<f32>> {
    let w = width as usize;
    let h = height as usize;
    let mut distance = vec![u32::MAX; mask.len()];
    let mut queue = VecDeque::new();

    for (idx, &inside) in mask.iter().enumerate() {
        if inside {
            distance[idx] = 0;
            queue.push_back(idx);
        }
    }

    while let Some(idx) = queue.pop_front() {
        if latest_id.load(Ordering::Relaxed) != job_id {
            return None;
        }
        let current_distance = distance[idx];
        if current_distance >= max_distance {
            continue;
        }
        let x = idx % w;
        let y = idx / w;
        let neighbors = [
            x.checked_sub(1).map(|nx| y * w + nx),
            (x + 1 < w).then_some(y * w + x + 1),
            y.checked_sub(1).map(|ny| ny * w + x),
            (y + 1 < h).then_some((y + 1) * w + x),
        ];
        for nidx in neighbors.into_iter().flatten() {
            if distance[nidx] > current_distance + 1 {
                distance[nidx] = current_distance + 1;
                queue.push_back(nidx);
            }
        }
    }

    Some(
        distance
            .into_iter()
            .map(|value| {
                if value == u32::MAX {
                    f32::INFINITY
                } else {
                    value as f32
                }
            })
            .collect(),
    )
}

fn dimmed_original_parallel(
    image: &RgbaImage,
    dimness: f32,
    black_background: bool,
) -> RgbaImage {
    let factor = (1.0 - dimness).clamp(0.0, 1.0);
    let mut raw = image.as_raw().clone();
    raw.par_chunks_mut(4).for_each(|pixel| {
        let r = (pixel[0] as f32 * factor).round().clamp(0.0, 255.0) as u8;
        let g = (pixel[1] as f32 * factor).round().clamp(0.0, 255.0) as u8;
        let b = (pixel[2] as f32 * factor).round().clamp(0.0, 255.0) as u8;
        if black_background {
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        } else {
            let lift = ((1.0 - factor) * 255.0 * 0.15) as u8;
            pixel[0] = r.saturating_add(lift);
            pixel[1] = g.saturating_add(lift);
            pixel[2] = b.saturating_add(lift);
        }
    });
    RgbaImage::from_raw(image.width(), image.height(), raw)
        .expect("RGBA buffer dimensions must remain valid")
}

fn alpha_fill_mask_parallel(
    image: &mut RgbaImage,
    mask: &[bool],
    color: egui::Color32,
    opacity: u8,
) {
    image
        .as_mut()
        .par_chunks_mut(4)
        .zip(mask.par_iter())
        .for_each(|(pixel, &inside)| {
            if inside {
                blend_raw_pixel(pixel, color, opacity);
            }
        });
}

fn paint_mask_parallel(
    image: &mut RgbaImage,
    mask: &[bool],
    color: egui::Color32,
    opacity: u8,
) {
    image
        .as_mut()
        .par_chunks_mut(4)
        .zip(mask.par_iter())
        .for_each(|(pixel, &paint)| {
            if paint {
                blend_raw_pixel(pixel, color, opacity);
            }
        });
}

fn apply_output_alpha_parallel(image: &mut RgbaImage, mask: &[bool], alpha: u8) {
    image
        .as_mut()
        .par_chunks_mut(4)
        .zip(mask.par_iter())
        .for_each(|(pixel, &inside)| {
            if inside {
                pixel[3] = alpha;
            }
        });
}

fn blend_raw_pixel(pixel: &mut [u8], color: egui::Color32, opacity: u8) {
    let alpha = opacity as f32 / 255.0;
    let inverse = 1.0 - alpha;
    pixel[0] = (color.r() as f32 * alpha + pixel[0] as f32 * inverse)
        .round()
        .clamp(0.0, 255.0) as u8;
    pixel[1] = (color.g() as f32 * alpha + pixel[1] as f32 * inverse)
        .round()
        .clamp(0.0, 255.0) as u8;
    pixel[2] = (color.b() as f32 * alpha + pixel[2] as f32 * inverse)
        .round()
        .clamp(0.0, 255.0) as u8;
}

fn merge_union(union: &mut [bool], mask: &[bool]) {
    union
        .par_iter_mut()
        .zip(mask.par_iter())
        .for_each(|(destination, &source)| *destination |= source);
}

fn brighten(color: egui::Color32, factor: f32) -> egui::Color32 {
    let factor = factor.clamp(0.0, 1.0);
    egui::Color32::from_rgba_unmultiplied(
        (color.r() as f32 + (255.0 - color.r() as f32) * factor)
            .round()
            .clamp(0.0, 255.0) as u8,
        (color.g() as f32 + (255.0 - color.g() as f32) * factor)
            .round()
            .clamp(0.0, 255.0) as u8,
        (color.b() as f32 + (255.0 - color.b() as f32) * factor)
            .round()
            .clamp(0.0, 255.0) as u8,
        color.a(),
    )
}

fn draw_cross(
    image: &mut RgbaImage,
    center_x: i32,
    center_y: i32,
    radius: i32,
    color: egui::Color32,
    opacity: u8,
) {
    for dx in -radius..=radius {
        blend_image_pixel(image, center_x + dx, center_y, color, opacity);
    }
    for dy in -radius..=radius {
        blend_image_pixel(image, center_x, center_y + dy, color, opacity);
    }
}

fn blend_image_pixel(
    image: &mut RgbaImage,
    x: i32,
    y: i32,
    color: egui::Color32,
    opacity: u8,
) {
    if x < 0 || y < 0 || x >= image.width() as i32 || y >= image.height() as i32 {
        return;
    }
    let pixel = image.get_pixel_mut(x as u32, y as u32);
    let mut raw = pixel.0;
    blend_raw_pixel(&mut raw, color, opacity);
    *pixel = Rgba(raw);
}
