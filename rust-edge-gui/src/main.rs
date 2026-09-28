mod viewer;
mod yolo;

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context as _, Result};
use chrono::{DateTime, Local};
use eframe::egui;
use image::{DynamicImage, GrayImage, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use yolo::{
    YoloDevice, YoloMessage, YoloModelInfo, YoloPostSettings, YoloRequest,
    YoloRuntimeSettings, YoloWorker,
};

const APP_TITLE: &str = "Rust Edge GUI v0.7 — Sequence Tracking + YOLO Segmentation";
const MAX_HISTORY: usize = 20;
const MIN_CAPTURE_INTERVAL_SECONDS: f32 = 0.1;
const MAX_SEQUENCE_MASK_CACHE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DetectionMode {
    LegacyGreen,
    PlantIndex,
    Yolo,
    Hybrid,
}

impl DetectionMode {
    const fn label(self) -> &'static str {
        match self {
            Self::LegacyGreen => "Legacy green excess",
            Self::PlantIndex => "RG/B plant index",
            Self::Yolo => "YOLO segmentation",
            Self::Hybrid => "Hybrid: YOLO ∩ color",
        }
    }

    const fn uses_yolo(self) -> bool {
        matches!(self, Self::Yolo | Self::Hybrid)
    }

    const fn persisted(self) -> &'static str {
        match self {
            Self::LegacyGreen => "legacy",
            Self::PlantIndex => "plant-index",
            Self::Yolo => "yolo",
            Self::Hybrid => "hybrid",
        }
    }

    fn from_persisted(value: &str) -> Self {
        match value {
            "legacy" => Self::LegacyGreen,
            "yolo" => Self::Yolo,
            "hybrid" => Self::Hybrid,
            _ => Self::PlantIndex,
        }
    }
}

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
    blue_deficit_threshold: f32,
    plant_index_threshold: f32,
    min_green_red_ratio: f32,
    min_rg_brightness: f32,
    hybrid_color_expand: u32,
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
            blue_deficit_threshold: 28.0,
            plant_index_threshold: 0.28,
            min_green_red_ratio: 0.72,
            min_rg_brightness: 35.0,
            hybrid_color_expand: 3,
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
    detection_mode: DetectionMode,
    ai_mask: Option<Arc<Vec<bool>>>,
    temporal_support_masks: Vec<Arc<Vec<bool>>>,
    temporal_overlap_threshold: f32,
    temporal_required_frames: usize,
}

struct ProcessingResult {
    processed: RgbaImage,
    mask: Vec<bool>,
    width: usize,
    height: usize,
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
    SequenceFrame(PathBuf),
    Url(String),
    CaptureUrl(String),
}

struct LoadedImage {
    label: String,
    rgba: RgbaImage,
    gray: GrayImage,
}

enum SourceMessage {
    Loaded {
        id: u64,
        image: LoadedImage,
        remember_source: bool,
        preserve_view: bool,
    },
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
    #[serde(default)]
    sequence_history: Vec<String>,
    #[serde(default)]
    capture_base_dir: String,
    #[serde(default)]
    capture_interval_secs: f32,
    #[serde(default)]
    detection_mode: String,
    #[serde(default)]
    yolo_model_path: String,
    #[serde(default)]
    yolo_class_ids: String,
    #[serde(default)]
    yolo_confidence: f32,
    #[serde(default)]
    yolo_iou: f32,
    #[serde(default)]
    yolo_mask_threshold: f32,
    #[serde(default)]
    yolo_input_size: usize,
}

#[derive(Clone)]
struct SequenceFrame {
    path: PathBuf,
    timestamp: SystemTime,
}

struct ImageSequence {
    root: PathBuf,
    frames: Vec<SequenceFrame>,
    selected: usize,
}

#[derive(Clone)]
struct CachedSequenceMask {
    width: usize,
    height: usize,
    mask: Arc<Vec<bool>>,
}

#[derive(Clone)]
struct MaskComponent {
    pixels: Vec<u32>,
    area: usize,
}

#[derive(Clone, Debug)]
struct ShapeObservation {
    area: usize,
    overlap: f32,
}

struct ShapeTrack {
    id: u64,
    name: String,
    pivot: (u32, u32),
    reference_frame: usize,
    reference_pixels: Vec<u32>,
    observations: BTreeMap<usize, ShapeObservation>,
    enabled: bool,
}

struct CaptureSession {
    root: PathBuf,
    original_dir: PathBuf,
    processed_dir: PathBuf,
    next_frame_index: u64,
}

struct PendingCapture {
    source_id: u64,
    processing_job_id: Option<u64>,
    file_name: String,
    captured_at: SystemTime,
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
    image_history_input: String,
    sequence_history: Vec<String>,
    sequence_history_input: String,
    active_sequence: Option<ImageSequence>,
    source_worker: SourceWorker,
    next_source_id: u64,
    active_source_id: u64,
    source_loading: bool,

    settings: GreenSettings,
    detection_mode: DetectionMode,
    dirty: bool,
    update_while_dragging: bool,

    processing_worker: ProcessingWorker,
    yolo_worker: YoloWorker,
    next_job_id: u64,
    active_job_id: u64,
    processing: bool,
    progress: f32,
    progress_stage: String,

    shape_count: usize,
    green_pixels: usize,
    boundary_pixels: usize,
    ai_mask: Option<Arc<Vec<bool>>>,
    ai_mask_rgba: Option<RgbaImage>,
    ai_mask_texture: Option<egui::TextureHandle>,
    ai_view: viewer::ImageViewState,
    yolo_model_path_input: String,
    yolo_class_ids_input: String,
    yolo_confidence: f32,
    yolo_iou: f32,
    yolo_mask_threshold: f32,
    yolo_input_size: usize,
    yolo_device: YoloDevice,
    yolo_model_generation: u64,
    yolo_fallback_color: bool,
    yolo_model_info: Option<YoloModelInfo>,
    yolo_instance_count: usize,
    yolo_mask_pixels: usize,
    yolo_elapsed_ms: f64,
    yolo_summary: String,
    status: String,
    error: Option<String>,

    original_view: viewer::ImageViewState,
    processed_view: viewer::ImageViewState,

    current_final_mask: Option<Arc<Vec<bool>>>,
    active_processing_sequence_frame: Option<usize>,
    sequence_mask_cache: BTreeMap<usize, CachedSequenceMask>,
    sequence_mask_cache_order: VecDeque<usize>,
    temporal_filter_enabled: bool,
    temporal_lookahead_frames: usize,
    temporal_required_frames: usize,
    temporal_overlap_threshold: f32,
    track_overlap_threshold: f32,
    shape_tracks: Vec<ShapeTrack>,
    next_track_id: u64,
    shape_plot_open: bool,
    sequence_playing: bool,
    sequence_playback_fps: f32,
    sequence_loop: bool,
    sequence_wait_processing: bool,
    next_sequence_frame_due: Option<Instant>,

    continuous_capture: bool,
    capture_interval_secs: f32,
    capture_base_dir_input: String,
    capture_save_original: bool,
    capture_save_processed: bool,
    capture_url: Option<String>,
    capture_session: Option<CaptureSession>,
    pending_capture: Option<PendingCapture>,
    next_capture_due: Option<Instant>,
}

impl GreenViewerApp {
    fn new(cc: &eframe::CreationContext<'_>, initial_source: Option<String>) -> Self {
        configure_dark_ui(&cc.egui_ctx);
        let persisted = load_persisted_state();
        let image_history_input = persisted.source_history.first().cloned().unwrap_or_default();
        let sequence_history_input = persisted.sequence_history.first().cloned().unwrap_or_default();
        let capture_base_dir_input = if persisted.capture_base_dir.trim().is_empty() {
            default_capture_base_dir().display().to_string()
        } else {
            persisted.capture_base_dir.clone()
        };
        let capture_interval_secs = if persisted.capture_interval_secs >= MIN_CAPTURE_INTERVAL_SECONDS {
            persisted.capture_interval_secs
        } else {
            2.0
        };
        let detection_mode = DetectionMode::from_persisted(&persisted.detection_mode);
        let yolo_model_path_input = if persisted.yolo_model_path.trim().is_empty() {
            "yolo26n-seg.onnx".to_owned()
        } else {
            persisted.yolo_model_path.clone()
        };
        let yolo_class_ids_input = if persisted.yolo_class_ids.trim().is_empty() {
            "58".to_owned()
        } else {
            persisted.yolo_class_ids.clone()
        };
        let yolo_confidence = if (0.01..=1.0).contains(&persisted.yolo_confidence) {
            persisted.yolo_confidence
        } else {
            0.25
        };
        let yolo_iou = if (0.01..=1.0).contains(&persisted.yolo_iou) {
            persisted.yolo_iou
        } else {
            0.70
        };
        let yolo_mask_threshold = if (0.01..=1.0).contains(&persisted.yolo_mask_threshold) {
            persisted.yolo_mask_threshold
        } else {
            0.50
        };
        let yolo_input_size = match persisted.yolo_input_size {
            320 | 512 | 640 | 768 | 1024 | 1280 => persisted.yolo_input_size,
            _ => 640,
        };

        let mut app = Self {
            original_rgba: None,
            original_gray: None,
            original_texture: None,
            processed_rgba: None,
            processed_texture: None,
            source_label: "No image".to_owned(),
            url_input: String::new(),
            source_history: persisted.source_history,
            image_history_input,
            sequence_history: persisted.sequence_history,
            sequence_history_input,
            active_sequence: None,
            source_worker: SourceWorker::spawn(cc.egui_ctx.clone()),
            next_source_id: 0,
            active_source_id: 0,
            source_loading: false,
            settings: GreenSettings::default(),
            detection_mode,
            dirty: false,
            update_while_dragging: true,
            processing_worker: ProcessingWorker::spawn(cc.egui_ctx.clone()),
            yolo_worker: YoloWorker::spawn(cc.egui_ctx.clone()),
            next_job_id: 0,
            active_job_id: 0,
            processing: false,
            progress: 0.0,
            progress_stage: "Idle".to_owned(),
            shape_count: 0,
            green_pixels: 0,
            boundary_pixels: 0,
            ai_mask: None,
            ai_mask_rgba: None,
            ai_mask_texture: None,
            ai_view: viewer::ImageViewState {
                open: false,
                ..Default::default()
            },
            yolo_model_path_input,
            yolo_class_ids_input,
            yolo_confidence,
            yolo_iou,
            yolo_mask_threshold,
            yolo_input_size,
            yolo_device: YoloDevice::Auto,
            yolo_model_generation: 0,
            yolo_fallback_color: true,
            yolo_model_info: None,
            yolo_instance_count: 0,
            yolo_mask_pixels: 0,
            yolo_elapsed_ms: 0.0,
            yolo_summary: "No YOLO inference yet".to_owned(),
            status: "Open an image from disk or enter a camera IP/address.".to_owned(),
            error: None,
            original_view: viewer::ImageViewState::default(),
            processed_view: viewer::ImageViewState::default(),
            current_final_mask: None,
            active_processing_sequence_frame: None,
            sequence_mask_cache: BTreeMap::new(),
            sequence_mask_cache_order: VecDeque::new(),
            temporal_filter_enabled: false,
            temporal_lookahead_frames: 2,
            temporal_required_frames: 1,
            temporal_overlap_threshold: 0.35,
            track_overlap_threshold: 0.30,
            shape_tracks: Vec::new(),
            next_track_id: 0,
            shape_plot_open: false,
            sequence_playing: false,
            sequence_playback_fps: 5.0,
            sequence_loop: true,
            sequence_wait_processing: true,
            next_sequence_frame_due: None,
            continuous_capture: false,
            capture_interval_secs,
            capture_base_dir_input,
            capture_save_original: true,
            capture_save_processed: true,
            capture_url: None,
            capture_session: None,
            pending_capture: None,
            next_capture_due: None,
        };

        if let Some(source) = initial_source {
            if source.starts_with("http://") || source.starts_with("https://") {
                app.url_input = source.clone();
                app.queue_source(SourceRequest::Url(source));
            } else {
                let path = PathBuf::from(&source);
                if path.exists() {
                    app.queue_source(SourceRequest::File(path));
                } else if let Ok(url) = normalize_camera_address(&source) {
                    app.url_input = source;
                    app.queue_source(SourceRequest::Url(url));
                } else {
                    app.queue_source(SourceRequest::File(path));
                }
            }
        }

        app
    }

    fn queue_source(&mut self, request: SourceRequest) {
        let is_capture = matches!(&request, SourceRequest::CaptureUrl(_));
        if matches!(&request, SourceRequest::SequenceFrame(_)) {
            self.current_final_mask = None;
        }
        if matches!(
            &request,
            SourceRequest::File(_) | SourceRequest::Url(_) | SourceRequest::CaptureUrl(_)
        ) {
            self.active_sequence = None;
            self.sequence_playing = false;
            self.next_sequence_frame_due = None;
            self.active_processing_sequence_frame = None;
            self.current_final_mask = None;
        }
        if !is_capture {
            self.pending_capture = None;
        }

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
                SourceMessage::Loaded {
                    id,
                    image,
                    remember_source,
                    preserve_view,
                } if id == self.active_source_id => {
                    self.source_loading = false;
                    let is_pending_capture = self
                        .pending_capture
                        .as_ref()
                        .is_some_and(|pending| pending.source_id == id);
                    if is_pending_capture && self.continuous_capture {
                        self.capture_url = Some(image.label.clone());
                    }
                    self.install_loaded_image(image, remember_source, preserve_view, ctx);
                    if is_pending_capture {
                        self.save_pending_capture_original();
                        if self.processing {
                            if let Some(pending) = self.pending_capture.as_mut() {
                                pending.processing_job_id = Some(self.active_job_id);
                            }
                        } else {
                            self.pending_capture = None;
                        }
                    }
                }
                SourceMessage::Failed { id, error } if id == self.active_source_id => {
                    self.source_loading = false;
                    if self
                        .pending_capture
                        .as_ref()
                        .is_some_and(|pending| pending.source_id == id)
                    {
                        self.pending_capture = None;
                    }
                    self.error = Some(error);
                    self.status = "Load failed".to_owned();
                }
                _ => {}
            }
        }
    }

    fn install_loaded_image(
        &mut self,
        image: LoadedImage,
        remember_source: bool,
        preserve_view: bool,
        ctx: &egui::Context,
    ) {
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
        if !preserve_view {
            self.original_view.reset_fit();
            self.processed_view.reset_fit();
        }
        self.status = format!("Loaded {}; processing…", image.label);
        self.error = None;
        if remember_source {
            self.remember_source(image.label);
        }
        self.schedule_processing();
    }

    fn remember_source(&mut self, source: String) {
        self.source_history.retain(|entry| entry != &source);
        self.source_history.insert(0, source.clone());
        self.source_history.truncate(MAX_HISTORY);
        self.image_history_input = source;
        self.save_preferences();
    }

    fn remember_sequence(&mut self, source: String) {
        self.sequence_history.retain(|entry| entry != &source);
        self.sequence_history.insert(0, source.clone());
        self.sequence_history.truncate(MAX_HISTORY);
        self.sequence_history_input = source;
        self.save_preferences();
    }

    fn save_preferences(&self) {
        save_persisted_state(&PersistedState {
            source_history: self.source_history.clone(),
            sequence_history: self.sequence_history.clone(),
            capture_base_dir: self.capture_base_dir_input.clone(),
            capture_interval_secs: self.capture_interval_secs,
            detection_mode: self.detection_mode.persisted().to_owned(),
            yolo_model_path: self.yolo_model_path_input.clone(),
            yolo_class_ids: self.yolo_class_ids_input.clone(),
            yolo_confidence: self.yolo_confidence,
            yolo_iou: self.yolo_iou,
            yolo_mask_threshold: self.yolo_mask_threshold,
            yolo_input_size: self.yolo_input_size,
        });
    }

    fn open_image_from_history_field(&mut self) {
        let source = self.image_history_input.trim().to_owned();
        if source.is_empty() {
            return;
        }

        if source.starts_with("http://") || source.starts_with("https://") {
            self.url_input = source.clone();
            self.queue_source(SourceRequest::Url(source));
            return;
        }

        let path = PathBuf::from(&source);
        if path.is_file() {
            self.queue_source(SourceRequest::File(path));
        } else if let Ok(url) = normalize_camera_address(&source) {
            self.url_input = source;
            self.queue_source(SourceRequest::Url(url));
        } else {
            self.error = Some(format!("Image path does not exist: {}", path.display()));
        }
    }

    fn open_sequence_from_history_field(&mut self) {
        let value = self.sequence_history_input.trim().to_owned();
        if value.is_empty() {
            return;
        }
        self.open_sequence(PathBuf::from(value));
    }

    fn open_sequence(&mut self, root: PathBuf) {
        match load_image_sequence(&root) {
            Ok(sequence) => {
                let selected_path = sequence.frames[sequence.selected].path.clone();
                let root_label = sequence.root.display().to_string();
                self.active_sequence = Some(sequence);
                self.sequence_playing = false;
                self.next_sequence_frame_due = None;
                self.current_final_mask = None;
                self.active_processing_sequence_frame = None;
                self.sequence_mask_cache.clear();
                self.sequence_mask_cache_order.clear();
                self.shape_tracks.clear();
                self.shape_plot_open = false;
                self.remember_sequence(root_label.clone());
                self.status = format!("Opened image sequence {root_label}");
                self.error = None;
                self.original_view.reset_fit();
                self.processed_view.reset_fit();
                self.ai_view.reset_fit();
                self.queue_source(SourceRequest::SequenceFrame(selected_path));
            }
            Err(error) => {
                self.error = Some(format!("Failed to open image sequence: {error:#}"));
            }
        }
    }

    fn start_continuous_capture(&mut self) {
        if !self.capture_save_original && !self.capture_save_processed {
            self.error = Some("Enable original and/or processed capture saving first.".to_owned());
            return;
        }

        let address = self.url_input.trim().to_owned();
        let url = match normalize_camera_address(&address) {
            Ok(url) => url,
            Err(error) => {
                self.error = Some(format!("Invalid camera address: {error:#}"));
                return;
            }
        };

        self.capture_interval_secs = self
            .capture_interval_secs
            .max(MIN_CAPTURE_INTERVAL_SECONDS);
        let base_dir = PathBuf::from(self.capture_base_dir_input.trim());
        if base_dir.as_os_str().is_empty() {
            self.error = Some("Choose a capture directory first.".to_owned());
            return;
        }

        let stamp = Local::now().format("%Y%m%d_%H%M%S_%3f").to_string();
        let root = base_dir.join(format!("capture_{stamp}"));
        let original_dir = root.join("original");
        let processed_dir = root.join("processed");

        let create_result = (|| -> Result<()> {
            std::fs::create_dir_all(&root)
                .with_context(|| format!("cannot create {}", root.display()))?;
            if self.capture_save_original {
                std::fs::create_dir_all(&original_dir)
                    .with_context(|| format!("cannot create {}", original_dir.display()))?;
            }
            if self.capture_save_processed {
                std::fs::create_dir_all(&processed_dir)
                    .with_context(|| format!("cannot create {}", processed_dir.display()))?;
            }
            Ok(())
        })();

        if let Err(error) = create_result {
            self.error = Some(format!("Failed to start capture: {error:#}"));
            return;
        }

        self.capture_session = Some(CaptureSession {
            root: root.clone(),
            original_dir,
            processed_dir,
            next_frame_index: 1,
        });
        self.capture_url = Some(url);
        self.pending_capture = None;
        self.next_capture_due = Some(Instant::now());
        self.continuous_capture = true;
        self.original_view.reset_fit();
        self.processed_view.reset_fit();
        self.remember_sequence(root.display().to_string());
        self.save_preferences();
        self.status = format!(
            "Continuous capture started: every {:.3}s → {}",
            self.capture_interval_secs,
            root.display()
        );
        self.error = None;
    }

    fn stop_continuous_capture(&mut self) {
        self.continuous_capture = false;
        self.next_capture_due = None;
        let location = self
            .capture_session
            .as_ref()
            .map(|session| session.root.display().to_string());
        self.capture_url = None;
        self.save_preferences();
        self.status = match location {
            Some(location) => format!("Continuous capture stopped. Saved in {location}"),
            None => "Continuous capture stopped.".to_owned(),
        };
    }

    fn tick_continuous_capture(&mut self, ctx: &egui::Context) {
        if !self.continuous_capture {
            return;
        }

        let now = Instant::now();
        let due = self.next_capture_due.unwrap_or(now);
        if now >= due
            && self.pending_capture.is_none()
            && !self.source_loading
            && !self.processing
        {
            let Some(url) = self.capture_url.clone() else {
                self.stop_continuous_capture();
                return;
            };
            let Some(session) = self.capture_session.as_mut() else {
                self.stop_continuous_capture();
                return;
            };

            let captured_at = SystemTime::now();
            let timestamp: DateTime<Local> = captured_at.into();
            let file_name = format!(
                "frame_{:06}_{}.png",
                session.next_frame_index,
                timestamp.format("%Y%m%d_%H%M%S_%3f")
            );
            session.next_frame_index = session.next_frame_index.saturating_add(1);

            self.queue_source(SourceRequest::CaptureUrl(url));
            self.pending_capture = Some(PendingCapture {
                source_id: self.active_source_id,
                processing_job_id: None,
                file_name,
                captured_at,
            });
            self.next_capture_due = Some(
                now + Duration::from_secs_f32(
                    self.capture_interval_secs.max(MIN_CAPTURE_INTERVAL_SECONDS),
                ),
            );
        }

        let delay = match self.next_capture_due {
            Some(next) if next > now => next.duration_since(now),
            _ => Duration::from_millis(100),
        };
        ctx.request_repaint_after(delay.min(Duration::from_secs(1)));
    }

    fn save_pending_capture_original(&mut self) {
        if !self.capture_save_original {
            return;
        }
        let (Some(session), Some(pending), Some(image)) = (
            self.capture_session.as_ref(),
            self.pending_capture.as_ref(),
            self.original_rgba.as_ref(),
        ) else {
            return;
        };
        let path = session.original_dir.join(&pending.file_name);
        if let Err(error) = DynamicImage::ImageRgba8((**image).clone()).save(&path) {
            self.error = Some(format!("Failed to save capture {}: {error}", path.display()));
        }
    }

    fn save_pending_capture_processed(&mut self, completed_job_id: u64) {
        let should_finish = self
            .pending_capture
            .as_ref()
            .and_then(|pending| pending.processing_job_id)
            .is_some_and(|id| id == completed_job_id);
        if !should_finish {
            return;
        }

        if self.capture_save_processed {
            if let (Some(session), Some(pending), Some(image)) = (
                self.capture_session.as_ref(),
                self.pending_capture.as_ref(),
                self.processed_rgba.as_ref(),
            ) {
                let path = session.processed_dir.join(&pending.file_name);
                if let Err(error) = DynamicImage::ImageRgba8(image.clone()).save(&path) {
                    self.error = Some(format!(
                        "Failed to save processed capture {}: {error}",
                        path.display()
                    ));
                } else {
                    let absolute = format_absolute_time(pending.captured_at);
                    self.status = format!("Captured {} ({absolute})", path.display());
                }
            }
        }
        self.pending_capture = None;
    }

    fn schedule_processing(&mut self) {
        if self.original_rgba.is_none() || self.original_gray.is_none() {
            return;
        }

        self.active_processing_sequence_frame = self.active_sequence.as_ref().map(|sequence| sequence.selected);
        self.next_job_id = self.next_job_id.wrapping_add(1).max(1);
        self.active_job_id = self.next_job_id;
        self.processing_worker
            .latest_id
            .store(self.active_job_id, Ordering::Release);
        self.yolo_worker
            .latest_id
            .store(self.active_job_id, Ordering::Release);

        self.processing = true;
        self.progress = 0.0;
        self.dirty = false;
        self.error = None;

        if self.detection_mode.uses_yolo() {
            let class_ids = match parse_class_ids(&self.yolo_class_ids_input) {
                Ok(ids) => ids,
                Err(error) => {
                    self.processing = false;
                    self.error = Some(error.to_string());
                    return;
                }
            };
            let runtime = YoloRuntimeSettings {
                model_path: self.yolo_model_path_input.trim().to_owned(),
                confidence: self.yolo_confidence,
                iou: self.yolo_iou,
                input_size: self.yolo_input_size,
                device: self.yolo_device,
                generation: self.yolo_model_generation,
            };
            let rgba = Arc::clone(self.original_rgba.as_ref().expect("checked above"));
            let request = YoloRequest {
                id: self.active_job_id,
                rgba,
                source_label: self.source_label.clone(),
                runtime,
                post: YoloPostSettings {
                    class_ids,
                    mask_threshold: self.yolo_mask_threshold,
                },
            };
            match self.yolo_worker.tx.send(request) {
                Ok(()) => {
                    self.progress = 0.04;
                    self.progress_stage = "YOLO segmentation".to_owned();
                }
                Err(error) => {
                    self.processing = false;
                    self.error = Some(format!("YOLO worker stopped: {error}"));
                }
            }
        } else {
            // Do not leave an old YOLO preview visible after switching back to a color-only mode.
            self.ai_mask = None;
            self.ai_mask_rgba = None;
            self.ai_mask_texture = None;
            self.yolo_mask_pixels = 0;
            self.yolo_instance_count = 0;
            self.submit_processing_request(self.active_job_id, self.detection_mode, None);
        }
    }

    fn submit_processing_request(
        &mut self,
        id: u64,
        detection_mode: DetectionMode,
        ai_mask: Option<Arc<Vec<bool>>>,
    ) {
        let (Some(rgba), Some(gray)) = (self.original_rgba.as_ref(), self.original_gray.as_ref())
        else {
            self.processing = false;
            return;
        };
        let temporal_support_masks = self.temporal_support_masks_for_active_frame();
        let request = ProcessingRequest {
            id,
            rgba: Arc::clone(rgba),
            gray: Arc::clone(gray),
            settings: self.settings.clone(),
            detection_mode,
            ai_mask,
            temporal_support_masks,
            temporal_overlap_threshold: self.temporal_overlap_threshold,
            temporal_required_frames: self.temporal_required_frames,
        };
        match self.processing_worker.job_tx.send(request) {
            Ok(()) => {
                self.processing = true;
                self.progress = if detection_mode.uses_yolo() { 0.52 } else { 0.0 };
                self.progress_stage = "Building final plant mask".to_owned();
            }
            Err(error) => {
                self.processing = false;
                self.error = Some(format!("Processing worker stopped: {error}"));
            }
        }
    }

    fn poll_yolo_worker(&mut self, ctx: &egui::Context) {
        while let Ok(message) = self.yolo_worker.rx.try_recv() {
            match message {
                YoloMessage::Finished { id, output } if id == self.active_job_id => {
                    let mask = Arc::new(output.mask);
                    self.yolo_model_info = Some(output.model_info);
                    self.yolo_instance_count = output.instance_count;
                    self.yolo_mask_pixels = output.mask_pixels;
                    self.yolo_elapsed_ms = output.elapsed_ms;
                    self.yolo_summary = output.summary;
                    self.ai_mask = Some(Arc::clone(&mask));

                    let preview = mask_to_rgba(&mask, output.width, output.height);
                    let color_image = rgba_to_color_image(&preview);
                    if let Some(texture) = self.ai_mask_texture.as_mut() {
                        texture.set(color_image, egui::TextureOptions::NEAREST);
                    } else {
                        self.ai_mask_texture = Some(ctx.load_texture(
                            "yolo-mask-image",
                            color_image,
                            egui::TextureOptions::NEAREST,
                        ));
                    }
                    self.ai_mask_rgba = Some(preview);
                    self.progress = 0.50;
                    self.progress_stage = "YOLO mask ready; compositing".to_owned();
                    self.submit_processing_request(id, self.detection_mode, Some(mask));
                }
                YoloMessage::Failed { id, error } if id == self.active_job_id => {
                    self.ai_mask = None;
                    self.ai_mask_rgba = None;
                    self.ai_mask_texture = None;
                    self.yolo_model_info = None;
                    self.yolo_instance_count = 0;
                    self.yolo_mask_pixels = 0;
                    self.yolo_elapsed_ms = 0.0;
                    self.yolo_summary = error.clone();
                    if self.yolo_fallback_color {
                        self.status = format!("{error}; falling back to RG/B plant index");
                        self.progress_stage = "YOLO failed; color fallback".to_owned();
                        self.submit_processing_request(id, DetectionMode::PlantIndex, None);
                    } else {
                        self.processing = false;
                        self.progress_stage = "YOLO failed".to_owned();
                        if self
                            .pending_capture
                            .as_ref()
                            .and_then(|pending| pending.processing_job_id)
                            .is_some_and(|job_id| job_id == id)
                        {
                            self.pending_capture = None;
                        }
                        self.error = Some(error);
                    }
                }
                _ => {}
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

                    let frame_mask = Arc::new(result.mask);
                    self.current_final_mask = Some(Arc::clone(&frame_mask));
                    if let Some(frame_index) = self.active_processing_sequence_frame {
                        self.cache_sequence_mask(
                            frame_index,
                            result.width,
                            result.height,
                            Arc::clone(&frame_mask),
                        );
                        self.update_shape_tracks_for_frame(
                            frame_index,
                            result.width,
                            result.height,
                            frame_mask.as_ref(),
                        );
                    }

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
                        "Detected {} mask component(s), {} plant px, {} boundary px",
                        self.shape_count, self.green_pixels, self.boundary_pixels
                    );
                    self.error = None;
                    self.save_pending_capture_processed(id);
                }
                ProcessingMessage::Failed { id, error } if id == self.active_job_id => {
                    self.processing = false;
                    self.progress_stage = "Failed".to_owned();
                    if self
                        .pending_capture
                        .as_ref()
                        .and_then(|pending| pending.processing_job_id)
                        .is_some_and(|job_id| job_id == id)
                    {
                        self.pending_capture = None;
                    }
                    self.error = Some(error);
                }
                _ => {}
            }
        }
    }

    fn temporal_support_masks_for_active_frame(&self) -> Vec<Arc<Vec<bool>>> {
        if !self.temporal_filter_enabled || self.temporal_required_frames == 0 {
            return Vec::new();
        }
        let Some(sequence) = self.active_sequence.as_ref() else {
            return Vec::new();
        };
        let current = sequence.selected;
        let expected_dims = self
            .original_rgba
            .as_ref()
            .map(|image| (image.width() as usize, image.height() as usize));
        let mut support = Vec::new();
        for offset in 1..=self.temporal_lookahead_frames.max(1) {
            let Some(index) = current.checked_add(offset) else {
                break;
            };
            if index >= sequence.frames.len() {
                break;
            }
            if let Some(cached) = self.sequence_mask_cache.get(&index) {
                if expected_dims.is_none_or(|dims| dims == (cached.width, cached.height)) {
                    support.push(Arc::clone(&cached.mask));
                }
            }
        }
        if support.len() >= self.temporal_required_frames {
            support
        } else {
            Vec::new()
        }
    }

    fn cache_sequence_mask(
        &mut self,
        frame_index: usize,
        width: usize,
        height: usize,
        mask: Arc<Vec<bool>>,
    ) {
        self.sequence_mask_cache.insert(
            frame_index,
            CachedSequenceMask {
                width,
                height,
                mask,
            },
        );
        self.sequence_mask_cache_order.retain(|index| *index != frame_index);
        self.sequence_mask_cache_order.push_back(frame_index);
        while self.sequence_mask_cache_order.len() > MAX_SEQUENCE_MASK_CACHE {
            if let Some(oldest) = self.sequence_mask_cache_order.pop_front() {
                self.sequence_mask_cache.remove(&oldest);
            }
        }
    }

    fn update_shape_tracks_for_frame(
        &mut self,
        frame_index: usize,
        width: usize,
        height: usize,
        mask: &[bool],
    ) {
        if self.shape_tracks.is_empty() || mask.len() != width.saturating_mul(height) {
            return;
        }
        let components = extract_mask_components(mask, width, height);
        if components.is_empty() {
            return;
        }
        let threshold = self.track_overlap_threshold.clamp(0.0, 1.0);
        for track in &mut self.shape_tracks {
            if !track.enabled {
                continue;
            }
            let mut best: Option<(&MaskComponent, f32)> = None;
            for component in &components {
                let overlap = component_iou(&track.reference_pixels, &component.pixels);
                if best.is_none_or(|(_, score)| overlap > score) {
                    best = Some((component, overlap));
                }
            }
            let Some((component, overlap)) = best else {
                continue;
            };
            if overlap + f32::EPSILON < threshold {
                continue;
            }
            track.observations.insert(
                frame_index,
                ShapeObservation {
                    area: component.area,
                    overlap,
                },
            );
            track.reference_frame = frame_index;
            track.pivot = component_centroid(component, width);
            track.reference_pixels = component.pixels.clone();
        }
    }

    fn add_shape_track_at_pivot(&mut self, pixel: (u32, u32)) {
        let Some(frame_index) = self.active_sequence.as_ref().map(|sequence| sequence.selected) else {
            self.status = "Open an image sequence before adding a tracked pivot.".to_owned();
            return;
        };
        let Some(mask) = self.current_final_mask.as_ref() else {
            self.status = "Wait until the current mask is processed before pressing P.".to_owned();
            return;
        };
        let Some(image) = self.original_rgba.as_ref() else {
            return;
        };
        let width = image.width() as usize;
        let height = image.height() as usize;
        if pixel.0 as usize >= width || pixel.1 as usize >= height {
            return;
        }
        let components = extract_mask_components(mask.as_ref(), width, height);
        let pixel_index = pixel.1 as usize * width + pixel.0 as usize;
        let Some(component) = components
            .into_iter()
            .find(|component| component.pixels.binary_search(&(pixel_index as u32)).is_ok())
        else {
            self.status = format!(
                "P pivot ({}, {}) is outside a closed mask component.",
                pixel.0, pixel.1
            );
            return;
        };

        self.next_track_id = self.next_track_id.wrapping_add(1).max(1);
        let mut observations = BTreeMap::new();
        observations.insert(
            frame_index,
            ShapeObservation {
                area: component.area,
                overlap: 1.0,
            },
        );
        self.shape_tracks.push(ShapeTrack {
            id: self.next_track_id,
            name: format!("shape-{}", self.next_track_id),
            pivot: pixel,
            reference_frame: frame_index,
            reference_pixels: component.pixels,
            observations,
            enabled: true,
        });
        self.shape_plot_open = true;
        self.status = format!(
            "Tracking shape {} from frame {} at pivot ({}, {}).",
            self.next_track_id,
            frame_index.saturating_add(1),
            pixel.0,
            pixel.1
        );
    }

    fn track_overlays(&self) -> Vec<viewer::OverlayPoint> {
        self.shape_tracks
            .iter()
            .filter(|track| track.enabled)
            .map(|track| viewer::OverlayPoint {
                pixel: track.pivot,
                color: track_color(track.id),
                label: format!("T{}", track.id),
            })
            .collect()
    }

    fn step_sequence(&mut self, delta: isize) {
        let loop_enabled = self.sequence_loop;
        let mut reached_end = false;
        let path = {
            let Some(sequence) = self.active_sequence.as_mut() else {
                return;
            };
            if sequence.frames.is_empty() {
                return;
            }
            let len = sequence.frames.len() as isize;
            let current = sequence.selected as isize;
            let next = if loop_enabled {
                (current + delta).rem_euclid(len)
            } else {
                (current + delta).clamp(0, len - 1)
            } as usize;
            if next == sequence.selected {
                reached_end = !loop_enabled && delta > 0;
                None
            } else {
                sequence.selected = next;
                Some(sequence.frames[next].path.clone())
            }
        };
        if reached_end {
            self.sequence_playing = false;
        }
        if let Some(path) = path {
            self.queue_source(SourceRequest::SequenceFrame(path));
        }
    }

    fn tick_sequence_playback(&mut self, ctx: &egui::Context) {
        if !self.sequence_playing || self.active_sequence.is_none() {
            return;
        }
        let now = Instant::now();
        let due = self.next_sequence_frame_due.unwrap_or(now);
        let busy = self.source_loading || (self.sequence_wait_processing && self.processing);
        if now >= due && !busy {
            self.step_sequence(1);
            let fps = self.sequence_playback_fps.clamp(0.1, 120.0);
            self.next_sequence_frame_due = Some(now + Duration::from_secs_f32(1.0 / fps));
        }
        let delay = self
            .next_sequence_frame_due
            .map(|next| next.saturating_duration_since(now))
            .unwrap_or(Duration::from_millis(16));
        ctx.request_repaint_after(delay.min(Duration::from_millis(250)));
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
            if path.is_dir() {
                self.sequence_history_input = path.display().to_string();
                self.open_sequence(path.clone());
            } else {
                self.image_history_input = path.display().to_string();
                self.queue_source(SourceRequest::File(path.clone()));
            }
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Rust Edge GUI");
        ui.small("Disk or camera IP input → color/YOLO plant segmentation.");

        if self.active_sequence.is_some() {
            ui.separator();
            self.sequence_timeline(ui);
            self.sequence_analysis_controls(ui);
        }

        ui.separator();
        ui.collapsing("Status", |ui| {
            if self.source_loading {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Loading image source…");
                });
            } else if self.processing {
                ui.label(&self.progress_stage);
                ui.add(
                    egui::ProgressBar::new(self.progress)
                        .show_percentage()
                        .desired_width(300.0),
                );
            } else if let Some(error) = self.error.as_ref() {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            } else {
                ui.label(&self.status);
            }
        });
        ui.separator();

        ui.label("Camera IP / address");
        let url_response = ui.add(
            egui::TextEdit::singleline(&mut self.url_input)
                .hint_text("10.87.121.137   or   http://10.87.121.137"),
        );
        let enter = url_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let load_camera = ui
            .add_enabled(!self.source_loading, egui::Button::new("Load camera + process"))
            .clicked()
            || enter;
        if load_camera && !self.url_input.trim().is_empty() {
            let address = self.url_input.trim().to_owned();
            match normalize_camera_address(&address) {
                Ok(url) => {
                    self.url_input = address;
                    self.queue_source(SourceRequest::Url(url));
                }
                Err(error) => {
                    self.error = Some(format!("Invalid camera address: {error:#}"));
                }
            }
        }
        ui.small(
            "A plain IP is enough. The app tries http://IP first; if that is not an image, /capture is tried as a fallback. Every loaded frame is processed automatically.",
        );

        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            if ui.button("Open image…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Open image")
                    .add_filter(
                        "Image",
                        &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
                    )
                    .pick_file()
                {
                    self.image_history_input = path.display().to_string();
                    self.queue_source(SourceRequest::File(path));
                }
            }

            if ui
                .add_enabled(self.original_rgba.is_some(), egui::Button::new("Save original…"))
                .clicked()
            {
                self.save_original();
            }

            if ui
                .add_enabled(self.processed_rgba.is_some(), egui::Button::new("Save processed…"))
                .clicked()
            {
                self.save_processed();
            }
        });

        ui.separator();
        ui.heading("History / sequences");
        ui.label("Image or URL");
        ui.horizontal(|ui| {
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.image_history_input)
                    .hint_text("/path/to/image.png or URL"),
            );
            let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if ui.button("Open").clicked() || enter {
                self.open_image_from_history_field();
            }
        });
        if !self.source_history.is_empty() {
            ui.collapsing("Recent images / sources", |ui| {
                let history = self.source_history.clone();
                for source in history {
                    if ui.selectable_label(false, &source).clicked() {
                        self.image_history_input = source.clone();
                        self.open_image_from_history_field();
                    }
                }
            });
        }

        ui.add_space(4.0);
        ui.label("Image sequence folder");
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.sequence_history_input)
                    .hint_text("/path/to/sequence or capture session"),
            );
            if ui.button("Open sequence").clicked() {
                self.open_sequence_from_history_field();
            }
            if ui.small_button("…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Open image sequence folder")
                    .pick_folder()
                {
                    self.sequence_history_input = path.display().to_string();
                    self.open_sequence(path);
                }
            }
        });
        if !self.sequence_history.is_empty() {
            ui.collapsing("Recent image sequences", |ui| {
                let history = self.sequence_history.clone();
                for source in history {
                    if ui.selectable_label(false, &source).clicked() {
                        self.sequence_history_input = source;
                        self.open_sequence_from_history_field();
                    }
                }
            });
        }

        ui.separator();
        ui.heading("Continuous capture");
        ui.horizontal(|ui| {
            ui.label("Every");
            ui.add(
                egui::DragValue::new(&mut self.capture_interval_secs)
                    .range(MIN_CAPTURE_INTERVAL_SECONDS..=3600.0)
                    .speed(0.1)
                    .suffix(" s"),
            );
        });
        ui.label("Save directory");
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.capture_base_dir_input)
                    .hint_text("capture directory"),
            );
            if ui.small_button("…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Choose capture directory")
                    .pick_folder()
                {
                    self.capture_base_dir_input = path.display().to_string();
                    self.save_preferences();
                }
            }
        });
        ui.add_enabled_ui(!self.continuous_capture, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.capture_save_original, "Original");
                ui.checkbox(&mut self.capture_save_processed, "Processed");
            });
        });
        ui.horizontal(|ui| {
            if !self.continuous_capture {
                if ui.button("Start continuous capture").clicked() {
                    self.start_continuous_capture();
                    ui.ctx().request_repaint();
                }
            } else if ui.button("Stop capture").clicked() {
                self.stop_continuous_capture();
            }
            if self.continuous_capture {
                ui.spinner();
                ui.label("recording");
            }
        });
        if let Some(session) = self.capture_session.as_ref() {
            ui.small(format!("Session: {}", session.root.display()));
        }

        ui.separator();
        ui.collapsing("Image windows", |ui| {
            ui.checkbox(&mut self.original_view.open, "Show original");
            ui.checkbox(&mut self.processed_view.open, "Show processed");
            ui.checkbox(&mut self.ai_view.open, "Show AI mask");
            ui.small("Each window has independent pan/zoom. Trackpad gestures affect only the image under the pointer.");
            if ui.button("Fit all independently").clicked() {
                self.original_view.reset_fit();
                self.processed_view.reset_fit();
                self.ai_view.reset_fit();
            }
        });

        ui.separator();
        ui.heading("Plant detection / mask source");
        let mut changed = false;
        let mut force_run = false;
        let previous_mode = self.detection_mode;
        egui::ComboBox::from_label("Detection mode")
            .selected_text(self.detection_mode.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.detection_mode,
                    DetectionMode::PlantIndex,
                    DetectionMode::PlantIndex.label(),
                );
                ui.selectable_value(
                    &mut self.detection_mode,
                    DetectionMode::Yolo,
                    DetectionMode::Yolo.label(),
                );
                ui.selectable_value(
                    &mut self.detection_mode,
                    DetectionMode::Hybrid,
                    DetectionMode::Hybrid.label(),
                );
                ui.selectable_value(
                    &mut self.detection_mode,
                    DetectionMode::LegacyGreen,
                    DetectionMode::LegacyGreen.label(),
                );
            });
        if self.detection_mode != previous_mode {
            changed = true;
            self.save_preferences();
        }

        match self.detection_mode {
            DetectionMode::LegacyGreen => {
                ui.small("Legacy rule: G - max(R,B) + green ratio. Kept for compatibility.");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.settings.green_excess_threshold, -80.0..=180.0)
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
            }
            DetectionMode::PlantIndex | DetectionMode::Hybrid => {
                ui.small("RG/B rule: high red+green, low blue; designed for top-view plant cameras.");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.settings.blue_deficit_threshold, -20.0..=160.0)
                            .text("Blue deficit: (R+G)/2 - B")
                            .fixed_decimals(1),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.settings.plant_index_threshold, -0.5..=1.0)
                            .text("Plant index")
                            .fixed_decimals(2),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.settings.min_green_red_ratio, 0.0..=1.5)
                            .text("Min G/R")
                            .fixed_decimals(2),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.settings.min_rg_brightness, 0.0..=255.0)
                            .text("Min RG brightness")
                            .fixed_decimals(0),
                    )
                    .changed();
                if self.detection_mode == DetectionMode::Hybrid {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.settings.hybrid_color_expand, 0..=24)
                                .text("Hybrid color gate expand"),
                        )
                        .changed();
                }
            }
            DetectionMode::Yolo => {}
        }

        if self.detection_mode.uses_yolo() {
            ui.add_space(5.0);
            ui.group(|ui| {
                ui.strong("YOLO segmentation (offline ONNX)");
                ui.horizontal(|ui| {
                    let model_response = ui.add(
                        egui::TextEdit::singleline(&mut self.yolo_model_path_input)
                            .hint_text("/path/to/best.onnx"),
                    );
                    if model_response.changed() {
                        changed = true;
                    }
                    if ui.small_button("…").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .set_title("Choose YOLO segmentation ONNX model")
                            .add_filter("ONNX model", &["onnx"])
                            .pick_file()
                        {
                            self.yolo_model_path_input = path.display().to_string();
                            self.yolo_model_generation = self.yolo_model_generation.wrapping_add(1);
                            changed = true;
                            self.save_preferences();
                        }
                    }
                });
                ui.small("Select a local Ultralytics segmentation .onnx file. Inference stays offline after the model is on disk.");

                ui.horizontal(|ui| {
                    ui.label("Plant class IDs");
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut self.yolo_class_ids_input)
                                .desired_width(95.0)
                                .hint_text("0 or 0,1"),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    if ui.small_button("Custom: 0").clicked() {
                        self.yolo_class_ids_input = "0".to_owned();
                        changed = true;
                    }
                    if ui.small_button("COCO plant: 58").clicked() {
                        self.yolo_class_ids_input = "58".to_owned();
                        changed = true;
                    }
                });
                ui.small("Instance-seg: empty = accept every detected class. Semantic-seg: class IDs are required.");

                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.yolo_confidence, 0.01..=1.0)
                            .text("Confidence")
                            .fixed_decimals(2),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.yolo_iou, 0.05..=0.95)
                            .text("NMS IoU")
                            .fixed_decimals(2),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.yolo_mask_threshold, 0.05..=0.95)
                            .text("Mask threshold")
                            .fixed_decimals(2),
                    )
                    .changed();

                ui.horizontal(|ui| {
                    ui.label("Input size");
                    egui::ComboBox::from_id_salt("yolo-imgsz")
                        .selected_text(self.yolo_input_size.to_string())
                        .show_ui(ui, |ui| {
                            for size in [320usize, 512, 640, 768, 1024, 1280] {
                                if ui
                                    .selectable_value(&mut self.yolo_input_size, size, size.to_string())
                                    .changed()
                                {
                                    changed = true;
                                }
                            }
                        });
                    ui.label("Device");
                    egui::ComboBox::from_id_salt("yolo-device")
                        .selected_text(self.yolo_device.label())
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_value(&mut self.yolo_device, YoloDevice::Auto, "Auto")
                                .changed()
                            {
                                changed = true;
                            }
                            if ui
                                .selectable_value(&mut self.yolo_device, YoloDevice::Cpu, "CPU")
                                .changed()
                            {
                                changed = true;
                            }
                        });
                });
                ui.checkbox(&mut self.yolo_fallback_color, "Fallback to RG/B mask if YOLO fails");

                ui.horizontal(|ui| {
                    if ui.button("Reload model + run").clicked() {
                        self.yolo_model_generation = self.yolo_model_generation.wrapping_add(1);
                        force_run = true;
                        self.save_preferences();
                    }
                    if self.ai_mask_rgba.is_some() && ui.button("Save AI mask…").clicked() {
                        self.save_ai_mask();
                    }
                });

                if let Some(info) = self.yolo_model_info.as_ref() {
                    ui.small(format!(
                        "{} · {} · {} classes · {}×{} · {}",
                        info.task,
                        info.provider,
                        info.class_count,
                        info.input_size.1,
                        info.input_size.0,
                        info.model_path
                    ));
                }
                ui.small(format!(
                    "AI: {} · {} px · {:.1} ms · {}",
                    self.yolo_instance_count,
                    self.yolo_mask_pixels,
                    self.yolo_elapsed_ms,
                    self.yolo_summary
                ));
            });
        }

        ui.add_space(5.0);
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.min_component_area, 1..=100_000)
                    .logarithmic(true)
                    .text("Min shape area"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.settings.grow_radius, 0..=12).text("Final mask grow"))
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
            self.sequence_mask_cache.clear();
            self.sequence_mask_cache_order.clear();
            self.save_preferences();
        }
        let pointer_down = ui.input(|input| input.pointer.primary_down());
        let cheap_auto_reprocess = !self.detection_mode.uses_yolo();
        if self.dirty
            && cheap_auto_reprocess
            && (self.update_while_dragging || !pointer_down)
        {
            self.schedule_processing();
        }
        if force_run {
            self.schedule_processing();
        }
        if ui
            .add_enabled(self.dirty, egui::Button::new("Apply parameters"))
            .clicked()
        {
            self.schedule_processing();
        }
        if self.detection_mode.uses_yolo() {
            ui.small("YOLO/Hybrid parameter edits wait for Apply to avoid running inference on every slider tick.");
        }

        ui.separator();
        ui.label(format!("Source: {}", self.source_label));
        if let Some(gray) = self.original_gray.as_ref() {
            ui.label(format!("Resolution: {} × {}", gray.width(), gray.height()));
        }
        ui.label(format!("Mask components: {}", self.shape_count));
        ui.label(format!("Mask pixels: {}", self.green_pixels));
        ui.label(format!("Boundary pixels: {}", self.boundary_pixels));
        ui.small("Workers block on recv() while idle. A repaint timer is only scheduled while continuous capture is active.");
    }

    fn sequence_timeline(&mut self, ui: &mut egui::Ui) {
        if self.active_sequence.is_none() {
            return;
        }

        let mut selected_path = None;
        let mut step_delta = 0isize;

        ui.group(|ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong("Sequence transport");
                if ui.button("Prev").on_hover_text("Previous frame").clicked() {
                    step_delta = -1;
                    self.sequence_playing = false;
                }
                let play_label = if self.sequence_playing { "Pause" } else { "Play" };
                if ui.button(play_label).clicked() {
                    self.sequence_playing = !self.sequence_playing;
                    self.next_sequence_frame_due = Some(Instant::now());
                }
                if ui.button("Next").on_hover_text("Next frame").clicked() {
                    step_delta = 1;
                    self.sequence_playing = false;
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Playback");
                ui.add(
                    egui::DragValue::new(&mut self.sequence_playback_fps)
                        .range(0.1..=120.0)
                        .speed(0.25)
                        .suffix(" fps"),
                );
                ui.checkbox(&mut self.sequence_loop, "Loop");
                ui.checkbox(&mut self.sequence_wait_processing, "Wait for processing");
            });

            if let Some(sequence) = self.active_sequence.as_mut() {
                let frame_count = sequence.frames.len();
                if frame_count > 0 {
                    let previous = sequence.selected;
                    let frame = &sequence.frames[sequence.selected];
                    let absolute = format_absolute_time(frame.timestamp);
                    let relative = frame
                        .timestamp
                        .duration_since(sequence.frames[0].timestamp)
                        .unwrap_or_default();

                    ui.horizontal_wrapped(|ui| {
                        ui.monospace(format!(
                            "FRAME {}/{}",
                            sequence.selected.saturating_add(1),
                            frame_count
                        ));
                        ui.separator();
                        ui.monospace(format!("ABS {absolute}"));
                        ui.separator();
                        ui.monospace(format!("VIDEO {}", format_video_time(relative)));
                    });
                    ui.add(
                        egui::Slider::new(&mut sequence.selected, 0..=frame_count - 1)
                            .show_value(false)
                            .text("timeline"),
                    );
                    ui.small(format!("{}", sequence.root.display()));
                    if sequence.selected != previous {
                        selected_path = Some(sequence.frames[sequence.selected].path.clone());
                        self.sequence_playing = false;
                    }
                }
            }
        });

        if step_delta != 0 {
            self.step_sequence(step_delta);
        } else if let Some(path) = selected_path {
            self.queue_source(SourceRequest::SequenceFrame(path));
        }
    }

    fn sequence_analysis_controls(&mut self, ui: &mut egui::Ui) {
        if self.active_sequence.is_none() {
            return;
        }
        ui.collapsing("Sequence mask tracking / temporal filter", |ui| {
            ui.label("Shape tracking");
            ui.add(
                egui::Slider::new(&mut self.track_overlap_threshold, 0.0..=1.0)
                    .text("Same-shape IoU")
                    .fixed_decimals(2),
            );
            ui.small("Hover a closed component in the Processed window and press P to add a pivot/track. The component is matched between frames by IoU.");

            ui.separator();
            let mut temporal_changed = false;
            temporal_changed |= ui
                .checkbox(
                    &mut self.temporal_filter_enabled,
                    "Filter transient components using subsequent frames",
                )
                .changed();
            temporal_changed |= ui
                .add(
                    egui::Slider::new(&mut self.temporal_lookahead_frames, 1..=12)
                        .text("Look-ahead frames"),
                )
                .changed();
            if self.temporal_required_frames > self.temporal_lookahead_frames {
                self.temporal_required_frames = self.temporal_lookahead_frames;
            }
            temporal_changed |= ui
                .add(
                    egui::Slider::new(
                        &mut self.temporal_required_frames,
                        1..=self.temporal_lookahead_frames.max(1),
                    )
                    .text("Required confirmations"),
                )
                .changed();
            temporal_changed |= ui
                .add(
                    egui::Slider::new(&mut self.temporal_overlap_threshold, 0.0..=1.0)
                        .text("Temporal overlap")
                        .fixed_decimals(2),
                )
                .changed();
            if temporal_changed {
                self.dirty = true;
                self.schedule_processing();
            }

            let future_cached = self
                .active_sequence
                .as_ref()
                .map(|sequence| {
                    (1..=self.temporal_lookahead_frames)
                        .filter_map(|offset| sequence.selected.checked_add(offset))
                        .filter(|index| self.sequence_mask_cache.contains_key(index))
                        .count()
                })
                .unwrap_or(0);
            ui.small(format!(
                "Mask cache: {} frame(s); future support available for current frame: {}/{}.",
                self.sequence_mask_cache.len(),
                future_cached,
                self.temporal_required_frames
            ));
            ui.small("The look-ahead filter is applied when enough subsequent frame masks are already cached. Play/analyze the sequence once, then scrub back for temporally filtered review.");

            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.shape_tracks.is_empty(), egui::Button::new("Open size plot"))
                    .clicked()
                {
                    self.shape_plot_open = true;
                }
                if ui
                    .add_enabled(!self.shape_tracks.is_empty(), egui::Button::new("Clear tracks"))
                    .clicked()
                {
                    self.shape_tracks.clear();
                }
            });
            let mut remove_track = None;
            for (index, track) in self.shape_tracks.iter_mut().enumerate() {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut track.enabled, "");
                    ui.colored_label(track_color(track.id), format!("T{}", track.id));
                    ui.text_edit_singleline(&mut track.name);
                    ui.monospace(format!("{} samples · ref {}", track.observations.len(), track.reference_frame.saturating_add(1)));
                    if ui.small_button("×").clicked() {
                        remove_track = Some(index);
                    }
                });
            }
            if let Some(index) = remove_track {
                self.shape_tracks.remove(index);
            }
        });
    }

    fn previews(&mut self, ui: &mut egui::Ui) {
        ui.allocate_space(ui.available_size());
        let ctx = ui.ctx().clone();
        let no_overlays: [viewer::OverlayPoint; 0] = [];

        viewer::show_floating_image_window(
            &ctx,
            "Original image",
            self.original_texture.as_ref(),
            self.original_rgba.as_deref(),
            &no_overlays,
            false,
            &mut self.original_view,
            egui::pos2(470.0, 60.0),
        );

        let overlays = self.track_overlays();
        let processed_interaction = viewer::show_floating_image_window(
            &ctx,
            "Processed — final plant mask",
            self.processed_texture.as_ref(),
            self.processed_rgba.as_ref(),
            &overlays,
            self.active_sequence.is_some(),
            &mut self.processed_view,
            egui::pos2(900.0, 100.0),
        );
        if let Some(pixel) = processed_interaction.pivot_pixel {
            self.add_shape_track_at_pivot(pixel);
        }

        viewer::show_floating_image_window(
            &ctx,
            "YOLO raw plant mask",
            self.ai_mask_texture.as_ref(),
            self.ai_mask_rgba.as_ref(),
            &no_overlays,
            false,
            &mut self.ai_view,
            egui::pos2(1080.0, 240.0),
        );
    }

    fn shape_size_plot_window(&mut self, ctx: &egui::Context) {
        if !self.shape_plot_open || self.shape_tracks.is_empty() {
            return;
        }
        let mut open = self.shape_plot_open;
        egui::Window::new("Tracked mask size")
            .open(&mut open)
            .default_pos(egui::pos2(720.0, 80.0))
            .default_size(egui::vec2(680.0, 360.0))
            .resizable(true)
            .show(ctx, |ui| {
                ui.small("X = sequence frame · Y = mask area in pixels. Tracks update as frames are processed.");
                let desired = egui::vec2(ui.available_width().max(300.0), ui.available_height().max(220.0));
                let (response, painter) = ui.allocate_painter(desired, egui::Sense::hover());
                let rect = response.rect.shrink2(egui::vec2(48.0, 28.0));
                let border = egui::Stroke::new(1.0, egui::Color32::DARK_GRAY);
                painter.line_segment([rect.left_top(), rect.right_top()], border);
                painter.line_segment([rect.right_top(), rect.right_bottom()], border);
                painter.line_segment([rect.right_bottom(), rect.left_bottom()], border);
                painter.line_segment([rect.left_bottom(), rect.left_top()], border);

                let mut max_frame = 1usize;
                let mut max_area = 1usize;
                for track in self.shape_tracks.iter().filter(|track| track.enabled) {
                    if let Some((&frame, _)) = track.observations.last_key_value() {
                        max_frame = max_frame.max(frame);
                    }
                    max_area = max_area.max(
                        track
                            .observations
                            .values()
                            .map(|observation| observation.area)
                            .max()
                            .unwrap_or(1),
                    );
                }
                if let Some(sequence) = self.active_sequence.as_ref() {
                    max_frame = max_frame.max(sequence.frames.len().saturating_sub(1));
                }

                painter.text(
                    egui::pos2(rect.left(), rect.bottom() + 7.0),
                    egui::Align2::LEFT_TOP,
                    "0",
                    egui::FontId::monospace(11.0),
                    egui::Color32::GRAY,
                );
                painter.text(
                    egui::pos2(rect.right(), rect.bottom() + 7.0),
                    egui::Align2::RIGHT_TOP,
                    format!("{}", max_frame.saturating_add(1)),
                    egui::FontId::monospace(11.0),
                    egui::Color32::GRAY,
                );
                painter.text(
                    egui::pos2(rect.left() - 6.0, rect.top()),
                    egui::Align2::RIGHT_TOP,
                    format!("{} px", max_area),
                    egui::FontId::monospace(11.0),
                    egui::Color32::GRAY,
                );

                for track in self.shape_tracks.iter().filter(|track| track.enabled) {
                    let color = track_color(track.id);
                    let mut points = Vec::new();
                    for (&frame, observation) in &track.observations {
                        let x = rect.left()
                            + frame as f32 / max_frame.max(1) as f32 * rect.width();
                        let y = rect.bottom()
                            - observation.area as f32 / max_area.max(1) as f32 * rect.height();
                        points.push(egui::pos2(x, y));
                    }
                    for pair in points.windows(2) {
                        painter.line_segment([pair[0], pair[1]], egui::Stroke::new(2.0, color));
                    }
                    for point in points {
                        painter.circle_filled(point, 3.0, color);
                    }
                }

                ui.horizontal_wrapped(|ui| {
                    for track in self.shape_tracks.iter().filter(|track| track.enabled) {
                        let last = track.observations.last_key_value();
                        let suffix = last
                            .map(|(_, observation)| format!("{} px · IoU {:.2}", observation.area, observation.overlap))
                            .unwrap_or_else(|| "no sample".to_owned());
                        ui.colored_label(
                            track_color(track.id),
                            format!("T{} {}: {}", track.id, track.name, suffix),
                        );
                    }
                });
            });
        self.shape_plot_open = open;
    }

    fn save_original(&mut self) {
        let Some(image) = self.original_rgba.as_ref() else {
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save original image")
            .set_file_name("original.png")
            .add_filter("PNG", &["png"])
            .save_file()
        else {
            return;
        };

        let mut path = path;
        if path.extension().is_none() {
            path.set_extension("png");
        }
        match DynamicImage::ImageRgba8((**image).clone()).save(&path) {
            Ok(()) => {
                self.status = format!("Saved {}", path.display());
                self.error = None;
            }
            Err(error) => {
                self.error = Some(format!("Failed to save {}: {error}", path.display()));
            }
        }
    }

    fn save_processed(&mut self) {
        let Some(image) = self.processed_rgba.as_ref() else {
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save processed image")
            .set_file_name("plant_mask_processed.png")
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

    fn save_ai_mask(&mut self) {
        let Some(image) = self.ai_mask_rgba.as_ref() else {
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save raw YOLO mask")
            .set_file_name("yolo_plant_mask.png")
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
        self.poll_yolo_worker(&ctx);
        self.poll_processing_worker(&ctx);
        self.tick_continuous_capture(&ctx);
        self.tick_sequence_playback(&ctx);

        egui::CentralPanel::default().show(ui, |ui| self.previews(ui));

        egui::Window::new("Controls / sequence")
            .default_pos(egui::pos2(12.0, 24.0))
            .default_size(egui::vec2(430.0, 850.0))
            .min_size(egui::vec2(330.0, 320.0))
            .resizable(true)
            .vscroll(true)
            .show(&ctx, |ui| self.controls(ui));

        self.shape_size_plot_window(&ctx);
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
        .timeout(Duration::from_secs(30))
        .user_agent("rust-edge-gui/0.7")
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

        let remember_source = matches!(&request, SourceRequest::File(_) | SourceRequest::Url(_));
        let preserve_view = matches!(
            &request,
            SourceRequest::SequenceFrame(_) | SourceRequest::CaptureUrl(_)
        );
        let result = match request {
            SourceRequest::File(path) | SourceRequest::SequenceFrame(path) => load_file_source(&path),
            SourceRequest::Url(url) | SourceRequest::CaptureUrl(url) => {
                load_url_source(&client, &url)
            }
        };

        if latest_id.load(Ordering::Acquire) != id {
            continue;
        }

        let message = match result {
            Ok(image) => SourceMessage::Loaded {
                id,
                image,
                remember_source,
                preserve_view,
            },
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

fn normalize_camera_address(input: &str) -> Result<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("camera address is empty"));
    }

    let candidate = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_owned()
    } else {
        format!("http://{trimmed}")
    };

    let parsed = reqwest::Url::parse(&candidate)
        .with_context(|| format!("cannot parse {candidate}"))?;
    if parsed.host_str().is_none() {
        return Err(anyhow!("camera address has no host"));
    }
    Ok(parsed.to_string())
}

fn camera_url_candidates(url: &str) -> Result<Vec<String>> {
    let parsed = reqwest::Url::parse(url)
        .with_context(|| format!("cannot parse camera URL {url}"))?;
    let mut candidates = vec![parsed.to_string()];

    let path = parsed.path();
    if path.is_empty() || path == "/" {
        let mut capture = parsed.clone();
        capture.set_path("/capture");
        capture.set_query(None);
        let capture = capture.to_string();
        if !candidates.iter().any(|candidate| candidate == &capture) {
            candidates.push(capture);
        }
    }

    Ok(candidates)
}

fn fetch_single_image(client: &reqwest::blocking::Client, url: &str) -> Result<LoadedImage> {
    let response = client
        .get(url)
        .header(reqwest::header::ACCEPT, "image/*,*/*;q=0.8")
        .send()
        .with_context(|| format!("request failed for {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();

    if content_type.starts_with("text/html") {
        return Err(anyhow!("{url} returned an HTML page, not an image"));
    }
    if content_type.starts_with("multipart/") {
        return Err(anyhow!("{url} returned a multipart stream, not a single snapshot"));
    }

    // `bytes()` drains the whole response before decoding. This avoids decoding a
    // partially received high-resolution JPEG as only its upper section.
    let bytes = response
        .bytes()
        .with_context(|| format!("failed while reading the complete image from {url}"))?;
    if bytes.is_empty() {
        return Err(anyhow!("{url} returned an empty response"));
    }

    let decoded = image::load_from_memory(&bytes)
        .with_context(|| format!("response from {url} is not a complete supported image"))?;
    Ok(LoadedImage {
        label: url.to_owned(),
        rgba: decoded.to_rgba8(),
        gray: decoded.to_luma8(),
    })
}

fn load_url_source(client: &reqwest::blocking::Client, url: &str) -> Result<LoadedImage> {
    let candidates = camera_url_candidates(url)?;
    let mut errors = Vec::new();

    for candidate in &candidates {
        match fetch_single_image(client, candidate) {
            Ok(image) => return Ok(image),
            Err(error) => errors.push(format!("{candidate}: {error:#}")),
        }
    }

    Err(anyhow!(
        "camera did not return a decodable image. Tried:\n{}",
        errors.join("\n")
    ))
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

fn pipeline_progress(mode: DetectionMode, value: f32) -> f32 {
    if mode.uses_yolo() {
        0.52 + value.clamp(0.0, 1.0) * 0.48
    } else {
        value.clamp(0.0, 1.0)
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

    progress(
        progress_tx,
        repaint_ctx,
        id,
        pipeline_progress(request.detection_mode, 0.05),
        "Building source mask",
    );
    let width = request.rgba.width();
    let height = request.rgba.height();
    if width == 0 || height == 0 {
        return Err(anyhow!("image has zero size"));
    }

    let settings = &request.settings;
    let color_mask = || plant_index_mask(&request.rgba, settings);
    let mut raw_mask: Vec<bool> = match request.detection_mode {
        DetectionMode::LegacyGreen => legacy_green_mask(&request.rgba, settings),
        DetectionMode::PlantIndex => color_mask(),
        DetectionMode::Yolo => request
            .ai_mask
            .as_ref()
            .filter(|mask| mask.len() == (width * height) as usize)
            .map(|mask| mask.as_ref().clone())
            .ok_or_else(|| anyhow!("YOLO mode has no valid AI mask"))?,
        DetectionMode::Hybrid => {
            let ai = request
                .ai_mask
                .as_ref()
                .filter(|mask| mask.len() == (width * height) as usize)
                .ok_or_else(|| anyhow!("Hybrid mode has no valid AI mask"))?;
            let mut color = color_mask();
            if settings.hybrid_color_expand > 0 {
                color = dilate_mask(
                    &color,
                    width as usize,
                    height as usize,
                    settings.hybrid_color_expand as usize,
                );
            }
            ai.iter()
                .zip(color.iter())
                .map(|(&ai_pixel, &color_pixel)| ai_pixel && color_pixel)
                .collect()
        }
    };

    if cancelled() {
        return Ok(None);
    }

    if settings.grow_radius > 0 {
        progress(
            progress_tx,
            repaint_ctx,
            id,
            pipeline_progress(request.detection_mode, 0.22),
            "Growing plant mask",
        );
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

    progress(
        progress_tx,
        repaint_ctx,
        id,
        pipeline_progress(request.detection_mode, 0.35),
        "Filtering plant components",
    );
    let (shape_mask, shape_count) = keep_components(
        &raw_mask,
        width as usize,
        height as usize,
        settings.min_component_area,
        latest_id,
        id,
    );
    let Some(mut shape_mask) = shape_mask else {
        return Ok(None);
    };
    let mut shape_count = shape_count;

    if request.temporal_required_frames > 0
        && request.temporal_support_masks.len() >= request.temporal_required_frames
    {
        progress(
            progress_tx,
            repaint_ctx,
            id,
            pipeline_progress(request.detection_mode, 0.48),
            "Temporal persistence filter",
        );
        let (filtered, filtered_count) = filter_mask_by_temporal_support(
            &shape_mask,
            width as usize,
            height as usize,
            &request.temporal_support_masks,
            request.temporal_required_frames,
            request.temporal_overlap_threshold,
        );
        shape_mask = filtered;
        shape_count = filtered_count;
    }

    progress(
        progress_tx,
        repaint_ctx,
        id,
        pipeline_progress(request.detection_mode, 0.58),
        "Building outlines",
    );
    let boundary = boundary_mask(&shape_mask, width as usize, height as usize);

    if cancelled() {
        return Ok(None);
    }

    progress(
        progress_tx,
        repaint_ctx,
        id,
        pipeline_progress(request.detection_mode, 0.70),
        "Compositing result",
    );
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
        progress(
            progress_tx,
            repaint_ctx,
            id,
            pipeline_progress(request.detection_mode, 0.82),
            "Detecting additional edges",
        );
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
    progress(
        progress_tx,
        repaint_ctx,
        id,
        pipeline_progress(request.detection_mode, 1.0),
        "Complete",
    );

    Ok(Some(ProcessingResult {
        processed,
        mask: shape_mask,
        width: width as usize,
        height: height as usize,
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

fn legacy_green_mask(image: &RgbaImage, settings: &GreenSettings) -> Vec<bool> {
    image
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
        .collect()
}

fn plant_index_mask(image: &RgbaImage, settings: &GreenSettings) -> Vec<bool> {
    image
        .as_raw()
        .par_chunks_exact(4)
        .map(|pixel| {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let rg_mean = (r + g) * 0.5;
            let blue_deficit = rg_mean - b;
            let plant_index = (r + g - 2.0 * b) / (r + g + 2.0 * b + 1.0);
            let green_red_ratio = g / (r + 1.0);

            blue_deficit >= settings.blue_deficit_threshold
                && plant_index >= settings.plant_index_threshold
                && green_red_ratio >= settings.min_green_red_ratio
                && rg_mean >= settings.min_rg_brightness
        })
        .collect()
}

fn parse_class_ids(value: &str) -> Result<Vec<usize>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let cleaned = trimmed
        .replace('[', "")
        .replace(']', "")
        .replace('(', "")
        .replace(')', "");
    let mut ids = Vec::new();
    for item in cleaned.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let id = item
            .parse::<usize>()
            .with_context(|| format!("invalid YOLO class ID '{item}'"))?;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    Ok(ids)
}

fn mask_to_rgba(mask: &[bool], width: usize, height: usize) -> RgbaImage {
    let mut image = RgbaImage::new(width as u32, height as u32);
    image
        .as_mut()
        .par_chunks_exact_mut(4)
        .zip(mask.par_iter())
        .for_each(|(pixel, &selected)| {
            if selected {
                pixel.copy_from_slice(&[40, 255, 110, 255]);
            } else {
                pixel.copy_from_slice(&[0, 0, 0, 255]);
            }
        });
    image
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

fn extract_mask_components(mask: &[bool], width: usize, height: usize) -> Vec<MaskComponent> {
    if width == 0 || height == 0 || mask.len() != width.saturating_mul(height) {
        return Vec::new();
    }
    let mut visited = vec![false; mask.len()];
    let mut queue = VecDeque::new();
    let mut components = Vec::new();

    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        queue.push_back(start);
        let mut pixels = Vec::new();
        while let Some(idx) = queue.pop_front() {
            pixels.push(idx as u32);
            let x = idx % width;
            let y = idx / width;
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let next = ny * width + nx;
                    if mask[next] && !visited[next] {
                        visited[next] = true;
                        queue.push_back(next);
                    }
                }
            }
        }
        pixels.sort_unstable();
        let area = pixels.len();
        components.push(MaskComponent { pixels, area });
    }
    components
}

fn component_centroid(component: &MaskComponent, width: usize) -> (u32, u32) {
    if component.pixels.is_empty() || width == 0 {
        return (0, 0);
    }
    let mut sum_x = 0u64;
    let mut sum_y = 0u64;
    for &index in &component.pixels {
        let index = index as usize;
        sum_x += (index % width) as u64;
        sum_y += (index / width) as u64;
    }
    let count = component.pixels.len() as u64;
    ((sum_x / count) as u32, (sum_y / count) as u32)
}

fn component_iou(a: &[u32], b: &[u32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut ai = 0usize;
    let mut bi = 0usize;
    let mut intersection = 0usize;
    while ai < a.len() && bi < b.len() {
        match a[ai].cmp(&b[bi]) {
            std::cmp::Ordering::Less => ai += 1,
            std::cmp::Ordering::Greater => bi += 1,
            std::cmp::Ordering::Equal => {
                intersection += 1;
                ai += 1;
                bi += 1;
            }
        }
    }
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn filter_mask_by_temporal_support(
    mask: &[bool],
    width: usize,
    height: usize,
    support_masks: &[Arc<Vec<bool>>],
    required_frames: usize,
    overlap_threshold: f32,
) -> (Vec<bool>, usize) {
    if required_frames == 0 || support_masks.len() < required_frames {
        let count = extract_mask_components(mask, width, height).len();
        return (mask.to_vec(), count);
    }
    let threshold = overlap_threshold.clamp(0.0, 1.0);
    let components = extract_mask_components(mask, width, height);
    let mut output = vec![false; mask.len()];
    let mut kept = 0usize;

    for component in components {
        let mut confirmations = 0usize;
        for support in support_masks {
            if support.len() != mask.len() {
                continue;
            }
            let intersection = component
                .pixels
                .iter()
                .filter(|&&idx| support[idx as usize])
                .count();
            let overlap = intersection as f32 / component.area.max(1) as f32;
            if overlap + f32::EPSILON >= threshold {
                confirmations += 1;
            }
        }
        if confirmations >= required_frames {
            kept += 1;
            for idx in component.pixels {
                output[idx as usize] = true;
            }
        }
    }
    (output, kept)
}

fn track_color(id: u64) -> egui::Color32 {
    let colors = [
        egui::Color32::from_rgb(80, 220, 255),
        egui::Color32::from_rgb(255, 120, 190),
        egui::Color32::from_rgb(160, 255, 120),
        egui::Color32::from_rgb(255, 210, 90),
        egui::Color32::from_rgb(170, 130, 255),
        egui::Color32::from_rgb(255, 150, 80),
        egui::Color32::from_rgb(80, 255, 210),
        egui::Color32::from_rgb(235, 235, 235),
    ];
    colors[(id as usize).saturating_sub(1) % colors.len()]
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

fn default_capture_base_dir() -> PathBuf {
    dirs::picture_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("rust-edge-gui-captures")
}

fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff"
            )
        })
        .unwrap_or(false)
}

fn load_image_sequence(root: &Path) -> Result<ImageSequence> {
    if !root.exists() {
        return Err(anyhow!("{} does not exist", root.display()));
    }
    if !root.is_dir() {
        return Err(anyhow!("{} is not a directory", root.display()));
    }

    let candidate_dirs = [root.join("original"), root.join("processed"), root.to_path_buf()];
    let mut frames = Vec::new();
    let mut scan_dir = root.to_path_buf();

    for candidate in candidate_dirs {
        if !candidate.is_dir() {
            continue;
        }
        let candidate_frames = std::fs::read_dir(&candidate)
            .with_context(|| format!("cannot read {}", candidate.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && is_supported_image_path(path))
            .map(|path| {
                let timestamp = std::fs::metadata(&path)
                    .and_then(|metadata| metadata.modified())
                    .unwrap_or(UNIX_EPOCH);
                SequenceFrame { path, timestamp }
            })
            .collect::<Vec<_>>();
        if !candidate_frames.is_empty() {
            scan_dir = candidate;
            frames = candidate_frames;
            break;
        }
    }

    frames.sort_by(|a, b| a.path.cmp(&b.path));
    if frames.is_empty() {
        return Err(anyhow!(
            "no supported images found in {}",
            scan_dir.display()
        ));
    }

    Ok(ImageSequence {
        root: root.to_path_buf(),
        frames,
        selected: 0,
    })
}

fn format_absolute_time(time: SystemTime) -> String {
    let date_time: DateTime<Local> = time.into();
    date_time.format("%Y-%m-%d %H:%M:%S%.3f").to_string()
}

fn format_video_time(duration: Duration) -> String {
    let total_millis = duration.as_millis();
    let millis = total_millis % 1000;
    let total_seconds = total_millis / 1000;
    let seconds = total_seconds % 60;
    let total_minutes = total_seconds / 60;
    let minutes = total_minutes % 60;
    let hours = total_minutes / 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

fn configure_dark_ui(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = egui::Color32::BLACK;
    style.visuals.window_fill = egui::Color32::from_rgb(8, 8, 8);
    style.visuals.extreme_bg_color = egui::Color32::BLACK;
    style.text_styles.insert(egui::TextStyle::Heading, egui::FontId::monospace(20.0));
    style.text_styles.insert(egui::TextStyle::Body, egui::FontId::monospace(14.0));
    style.text_styles.insert(egui::TextStyle::Monospace, egui::FontId::monospace(14.0));
    style.text_styles.insert(egui::TextStyle::Button, egui::FontId::monospace(14.0));
    style.text_styles.insert(egui::TextStyle::Small, egui::FontId::monospace(11.0));
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
