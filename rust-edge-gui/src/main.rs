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

const APP_TITLE: &str = "Rust Edge GUI v0.7.3 — Anchored Shape Tracking";
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
        persist_window: true,
        viewport: egui::ViewportBuilder::default()
            .with_app_id("rust-edge-gui")
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

#[derive(Clone, Serialize, Deserialize)]
struct PersistedGreenSettings {
    green_excess_threshold: f32,
    green_ratio_threshold: f32,
    blue_deficit_threshold: f32,
    plant_index_threshold: f32,
    min_green_red_ratio: f32,
    min_rg_brightness: f32,
    hybrid_color_expand: u32,
    min_component_area: usize,
    grow_radius: u32,
    fill_color: [u8; 4],
    fill_opacity: u8,
    outline_color: [u8; 4],
    outline_opacity: u8,
    dimness: f32,
    edge_enabled: bool,
    edge_threshold: f32,
    edge_color: [u8; 4],
    edge_opacity: u8,
}

impl PersistedGreenSettings {
    fn from_settings(settings: &GreenSettings) -> Self {
        Self {
            green_excess_threshold: settings.green_excess_threshold,
            green_ratio_threshold: settings.green_ratio_threshold,
            blue_deficit_threshold: settings.blue_deficit_threshold,
            plant_index_threshold: settings.plant_index_threshold,
            min_green_red_ratio: settings.min_green_red_ratio,
            min_rg_brightness: settings.min_rg_brightness,
            hybrid_color_expand: settings.hybrid_color_expand,
            min_component_area: settings.min_component_area,
            grow_radius: settings.grow_radius,
            fill_color: settings.fill_color.to_array(),
            fill_opacity: settings.fill_opacity,
            outline_color: settings.outline_color.to_array(),
            outline_opacity: settings.outline_opacity,
            dimness: settings.dimness,
            edge_enabled: settings.edge_enabled,
            edge_threshold: settings.edge_threshold,
            edge_color: settings.edge_color.to_array(),
            edge_opacity: settings.edge_opacity,
        }
    }

    fn into_settings(self) -> GreenSettings {
        GreenSettings {
            green_excess_threshold: self.green_excess_threshold,
            green_ratio_threshold: self.green_ratio_threshold,
            blue_deficit_threshold: self.blue_deficit_threshold,
            plant_index_threshold: self.plant_index_threshold,
            min_green_red_ratio: self.min_green_red_ratio,
            min_rg_brightness: self.min_rg_brightness,
            hybrid_color_expand: self.hybrid_color_expand,
            min_component_area: self.min_component_area,
            grow_radius: self.grow_radius,
            fill_color: egui::Color32::from_rgba_unmultiplied(
                self.fill_color[0], self.fill_color[1], self.fill_color[2], self.fill_color[3],
            ),
            fill_opacity: self.fill_opacity,
            outline_color: egui::Color32::from_rgba_unmultiplied(
                self.outline_color[0], self.outline_color[1], self.outline_color[2], self.outline_color[3],
            ),
            outline_opacity: self.outline_opacity,
            dimness: self.dimness,
            edge_enabled: self.edge_enabled,
            edge_threshold: self.edge_threshold,
            edge_color: egui::Color32::from_rgba_unmultiplied(
                self.edge_color[0], self.edge_color[1], self.edge_color[2], self.edge_color[3],
            ),
            edge_opacity: self.edge_opacity,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct PersistedViewState {
    open: bool,
    zoom: f32,
    pan: [f32; 2],
    fit_to_window: bool,
    window_pos: Option<[f32; 2]>,
    window_size: Option<[f32; 2]>,
}

impl PersistedViewState {
    fn from_view(view: &viewer::ImageViewState) -> Self {
        Self {
            open: view.open,
            zoom: view.zoom,
            pan: [view.pan.x, view.pan.y],
            fit_to_window: view.fit_to_window,
            window_pos: view.window_pos.map(|p| [p.x, p.y]),
            window_size: view.window_size.map(|v| [v.x, v.y]),
        }
    }

    fn into_view(self) -> viewer::ImageViewState {
        viewer::ImageViewState {
            open: self.open,
            zoom: self.zoom.clamp(viewer::MIN_ZOOM, viewer::MAX_ZOOM),
            pan: egui::vec2(self.pan[0], self.pan[1]),
            fit_to_window: self.fit_to_window,
            window_pos: self.window_pos.map(|p| egui::pos2(p[0], p[1])),
            window_size: self.window_size.map(|v| egui::vec2(v[0].max(300.0), v[1].max(240.0))),
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct PersistedWindowRect {
    pos: [f32; 2],
    size: [f32; 2],
}

impl PersistedWindowRect {
    fn new(pos: egui::Pos2, size: egui::Vec2) -> Self {
        Self { pos: [pos.x, pos.y], size: [size.x, size.y] }
    }
    fn pos(self) -> egui::Pos2 { egui::pos2(self.pos[0], self.pos[1]) }
    fn size(self) -> egui::Vec2 { egui::vec2(self.size[0], self.size[1]) }
}

#[derive(Default, Serialize, Deserialize)]
struct PersistedState {
    #[serde(default)]
    version: u32,
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

    #[serde(default)]
    green_settings: Option<PersistedGreenSettings>,
    #[serde(default)]
    update_while_dragging: bool,
    #[serde(default)]
    yolo_device: String,
    #[serde(default)]
    yolo_fallback_color: bool,
    #[serde(default)]
    temporal_filter_enabled: bool,
    #[serde(default)]
    temporal_window_frames: usize,
    #[serde(default)]
    temporal_required_frames: usize,
    #[serde(default)]
    temporal_overlap_threshold: f32,
    #[serde(default)]
    track_overlap_threshold: f32,
    #[serde(default)]
    sequence_playback_fps: f32,
    #[serde(default)]
    sequence_loop: bool,
    #[serde(default)]
    sequence_wait_processing: bool,
    #[serde(default)]
    capture_save_original: bool,
    #[serde(default)]
    capture_save_processed: bool,
    #[serde(default)]
    original_view: Option<PersistedViewState>,
    #[serde(default)]
    processed_view: Option<PersistedViewState>,
    #[serde(default)]
    ai_view: Option<PersistedViewState>,
    #[serde(default)]
    controls_window: Option<PersistedWindowRect>,
    #[serde(default)]
    shape_plot_window: Option<PersistedWindowRect>,
    #[serde(default)]
    shape_plot_open: bool,
    #[serde(default)]
    url_input: String,
    #[serde(default)]
    image_history_input: String,
    #[serde(default)]
    sequence_history_input: String,
    #[serde(default)]
    sequence_glue_inputs: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct SequenceManifest {
    version: u32,
    frames: Vec<SequenceManifestFrame>,
}

#[derive(Serialize, Deserialize)]
struct SequenceManifestFrame {
    path: String,
    timestamp_ms: u64,
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
    /// Immutable identity anchor: the exact user-picked pixel and frame.
    /// Tracking may adapt to mask growth/shrink, but never rewrites this origin.
    anchor_pivot: (u32, u32),
    anchor_frame: usize,
    anchor_width: usize,
    anchor_height: usize,
    anchor_pixels: Vec<u32>,
    /// Per-frame matched component geometry. Keeping this history lets us match
    /// outward from the anchor in either time direction without reference drift.
    matched_pixels: BTreeMap<usize, Vec<u32>>,
    matched_centroids: BTreeMap<usize, (u32, u32)>,
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
    sequence_frame_index: Option<usize>,
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
    sequence_glue_inputs: Vec<String>,

    controls_window_pos: egui::Pos2,
    controls_window_size: egui::Vec2,
    shape_plot_window_pos: egui::Pos2,
    shape_plot_window_size: egui::Vec2,
    next_preferences_save: Instant,
    preferences_dirty: bool,

    continuous_capture: bool,
    capture_interval_secs: f32,
    capture_base_dir_input: String,
    capture_save_original: bool,
    capture_save_processed: bool,
    capture_url: Option<String>,
    capture_session: Option<CaptureSession>,
    // Source acquisition and image processing are intentionally decoupled so a
    // slow YOLO/temporal pass cannot stretch a 2-minute recording interval to 12 minutes.
    pending_capture: Option<PendingCapture>,
    processing_capture: Option<PendingCapture>,
    deferred_sequence_frame: Option<PathBuf>,
    next_capture_due: Option<Instant>,
}

impl GreenViewerApp {
    fn new(cc: &eframe::CreationContext<'_>, initial_source: Option<String>) -> Self {
        configure_dark_ui(&cc.egui_ctx);
        let persisted = load_persisted_state();
        let persisted_v2 = persisted.version >= 2;
        let image_history_input = if persisted_v2 && !persisted.image_history_input.is_empty() {
            persisted.image_history_input.clone()
        } else {
            persisted.source_history.first().cloned().unwrap_or_default()
        };
        let sequence_history_input = if persisted_v2 && !persisted.sequence_history_input.is_empty() {
            persisted.sequence_history_input.clone()
        } else {
            persisted.sequence_history.first().cloned().unwrap_or_default()
        };
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
        let settings = if persisted_v2 {
            persisted.green_settings.clone().map(PersistedGreenSettings::into_settings).unwrap_or_default()
        } else {
            GreenSettings::default()
        };
        let original_view = if persisted_v2 {
            persisted.original_view.clone().map(PersistedViewState::into_view).unwrap_or_default()
        } else {
            viewer::ImageViewState::default()
        };
        let processed_view = if persisted_v2 {
            persisted.processed_view.clone().map(PersistedViewState::into_view).unwrap_or_default()
        } else {
            viewer::ImageViewState::default()
        };
        let ai_view = if persisted_v2 {
            persisted.ai_view.clone().map(PersistedViewState::into_view).unwrap_or_else(|| viewer::ImageViewState { open: false, ..Default::default() })
        } else {
            viewer::ImageViewState { open: false, ..Default::default() }
        };
        let controls_window = persisted.controls_window.unwrap_or(PersistedWindowRect::new(egui::pos2(12.0, 24.0), egui::vec2(430.0, 850.0)));
        let shape_plot_window = persisted.shape_plot_window.unwrap_or(PersistedWindowRect::new(egui::pos2(720.0, 80.0), egui::vec2(680.0, 360.0)));
        let temporal_window_frames = if persisted_v2 { persisted.temporal_window_frames.clamp(4, 26) } else { 6 };
        let temporal_radius = (temporal_window_frames / 2).saturating_sub(1).max(1);
        let temporal_required_frames = if persisted_v2 {
            persisted.temporal_required_frames.clamp(1, temporal_radius * 2)
        } else { 1 };

        let mut app = Self {
            original_rgba: None,
            original_gray: None,
            original_texture: None,
            processed_rgba: None,
            processed_texture: None,
            source_label: "No image".to_owned(),
            url_input: if persisted_v2 { persisted.url_input.clone() } else { String::new() },
            source_history: persisted.source_history,
            image_history_input,
            sequence_history: persisted.sequence_history,
            sequence_history_input,
            active_sequence: None,
            source_worker: SourceWorker::spawn(cc.egui_ctx.clone()),
            next_source_id: 0,
            active_source_id: 0,
            source_loading: false,
            settings,
            detection_mode,
            dirty: false,
            update_while_dragging: if persisted_v2 { persisted.update_while_dragging } else { true },
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
            ai_view,
            yolo_model_path_input,
            yolo_class_ids_input,
            yolo_confidence,
            yolo_iou,
            yolo_mask_threshold,
            yolo_input_size,
            yolo_device: if persisted_v2 && persisted.yolo_device == "cpu" { YoloDevice::Cpu } else { YoloDevice::Auto },
            yolo_model_generation: 0,
            yolo_fallback_color: if persisted_v2 { persisted.yolo_fallback_color } else { true },
            yolo_model_info: None,
            yolo_instance_count: 0,
            yolo_mask_pixels: 0,
            yolo_elapsed_ms: 0.0,
            yolo_summary: "No YOLO inference yet".to_owned(),
            status: "Open an image from disk or enter a camera IP/address.".to_owned(),
            error: None,
            original_view,
            processed_view,
            current_final_mask: None,
            active_processing_sequence_frame: None,
            sequence_mask_cache: BTreeMap::new(),
            sequence_mask_cache_order: VecDeque::new(),
            temporal_filter_enabled: if persisted_v2 { persisted.temporal_filter_enabled } else { false },
            temporal_lookahead_frames: temporal_window_frames,
            temporal_required_frames,
            temporal_overlap_threshold: if persisted_v2 { persisted.temporal_overlap_threshold.clamp(0.0, 1.0) } else { 0.35 },
            track_overlap_threshold: if persisted_v2 { persisted.track_overlap_threshold.clamp(0.0, 1.0) } else { 0.30 },
            shape_tracks: Vec::new(),
            next_track_id: 0,
            shape_plot_open: if persisted_v2 { persisted.shape_plot_open } else { false },
            sequence_playing: false,
            sequence_playback_fps: if persisted_v2 { persisted.sequence_playback_fps.clamp(0.1, 120.0) } else { 5.0 },
            sequence_loop: if persisted_v2 { persisted.sequence_loop } else { true },
            sequence_wait_processing: if persisted_v2 { persisted.sequence_wait_processing } else { true },
            next_sequence_frame_due: None,
            sequence_glue_inputs: if persisted_v2 { persisted.sequence_glue_inputs.clone() } else { Vec::new() },
            controls_window_pos: controls_window.pos(),
            controls_window_size: controls_window.size(),
            shape_plot_window_pos: shape_plot_window.pos(),
            shape_plot_window_size: shape_plot_window.size(),
            next_preferences_save: Instant::now() + Duration::from_secs(1),
            preferences_dirty: false,
            continuous_capture: false,
            capture_interval_secs,
            capture_base_dir_input,
            capture_save_original: if persisted_v2 { persisted.capture_save_original } else { true },
            capture_save_processed: if persisted_v2 { persisted.capture_save_processed } else { true },
            capture_url: None,
            capture_session: None,
            pending_capture: None,
            processing_capture: None,
            deferred_sequence_frame: None,
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
        if let SourceRequest::SequenceFrame(path) = &request {
            if !path.is_file() {
                if let Some(sequence) = self.active_sequence.as_mut() {
                    sequence.frames.retain(|frame| frame.path.is_file());
                    sequence.selected = sequence.selected.min(sequence.frames.len().saturating_sub(1));
                    if !sequence.frames.is_empty() {
                        let _ = write_sequence_manifest(sequence);
                    }
                }
                self.source_loading = false;
                self.error = Some(format!(
                    "Sequence frame disappeared before it could be opened: {}. The sequence was pruned to existing files.",
                    path.display()
                ));
                return;
            }
        }
        if let SourceRequest::SequenceFrame(path) = &request {
            if self.continuous_capture && self.pending_capture.is_some() {
                self.deferred_sequence_frame = Some(path.clone());
                self.status = "A capture is being fetched/saved; sequence navigation is queued until the raw frame is committed.".to_owned();
                return;
            }
            if let Some(sequence) = self.active_sequence.as_mut() {
                if let Some(index) = sequence.frames.iter().position(|frame| &frame.path == path) {
                    sequence.selected = index;
                }
            }
        }
        let is_capture = matches!(&request, SourceRequest::CaptureUrl(_));
        if matches!(&request, SourceRequest::SequenceFrame(_) | SourceRequest::CaptureUrl(_)) {
            self.current_final_mask = None;
        }
        if matches!(
            &request,
            SourceRequest::File(_) | SourceRequest::Url(_)
        ) {
            self.active_sequence = None;
            self.sequence_playing = false;
            self.next_sequence_frame_due = None;
            self.active_processing_sequence_frame = None;
            self.current_final_mask = None;
        }
        if !is_capture {
            self.pending_capture = None;
            // Loading another image invalidates the current capture-processing job.
            // The original capture is already safely on disk; processed output is best-effort
            // when interactive analysis competes with continuous recording.
            self.processing_capture = None;
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
                    self.install_loaded_image(
                        image,
                        remember_source,
                        preserve_view,
                        ctx,
                        !is_pending_capture,
                    );
                    if is_pending_capture {
                        // Commit the raw frame first, then schedule processing. This
                        // guarantees that tracking/temporal filtering sees a sequence
                        // entry only after its file exists and uses the correct frame index.
                        let committed_index = self.save_pending_capture_original();
                        if self.capture_save_original && committed_index.is_none() {
                            // Original saving was requested but failed. We may still save
                            // a processed-only fallback if processing succeeds.
                            self.status = "Capture loaded, but original save failed; processing fallback…".to_owned();
                        }
                        self.schedule_processing();
                        if let Some(frame_index) = committed_index {
                            self.active_processing_sequence_frame = Some(frame_index);
                        }
                        if self.processing {
                            if let Some(pending) = self.pending_capture.as_mut() {
                                pending.processing_job_id = Some(self.active_job_id);
                            }
                            // Raw acquisition is complete. Move metadata to a separate
                            // processing slot so the next timed capture is not blocked by YOLO.
                            self.processing_capture = self.pending_capture.take();
                        } else {
                            self.pending_capture = None;
                        }
                        if let Some(path) = self.deferred_sequence_frame.take() {
                            self.queue_source(SourceRequest::SequenceFrame(path));
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
                        self.remove_pending_capture_sequence_frame();
                        self.pending_capture = None;
                    }
                    self.error = Some(error);
                    self.status = "Load failed".to_owned();
                    if let Some(path) = self.deferred_sequence_frame.take() {
                        self.queue_source(SourceRequest::SequenceFrame(path));
                    }
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
        schedule_processing: bool,
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
        if schedule_processing {
            self.schedule_processing();
        }
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
            version: 2,
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
            green_settings: Some(PersistedGreenSettings::from_settings(&self.settings)),
            update_while_dragging: self.update_while_dragging,
            yolo_device: match self.yolo_device { YoloDevice::Cpu => "cpu", YoloDevice::Auto => "auto" }.to_owned(),
            yolo_fallback_color: self.yolo_fallback_color,
            temporal_filter_enabled: self.temporal_filter_enabled,
            temporal_window_frames: self.temporal_lookahead_frames,
            temporal_required_frames: self.temporal_required_frames,
            temporal_overlap_threshold: self.temporal_overlap_threshold,
            track_overlap_threshold: self.track_overlap_threshold,
            sequence_playback_fps: self.sequence_playback_fps,
            sequence_loop: self.sequence_loop,
            sequence_wait_processing: self.sequence_wait_processing,
            capture_save_original: self.capture_save_original,
            capture_save_processed: self.capture_save_processed,
            original_view: Some(PersistedViewState::from_view(&self.original_view)),
            processed_view: Some(PersistedViewState::from_view(&self.processed_view)),
            ai_view: Some(PersistedViewState::from_view(&self.ai_view)),
            controls_window: Some(PersistedWindowRect::new(self.controls_window_pos, self.controls_window_size)),
            shape_plot_window: Some(PersistedWindowRect::new(self.shape_plot_window_pos, self.shape_plot_window_size)),
            shape_plot_open: self.shape_plot_open,
            url_input: self.url_input.clone(),
            image_history_input: self.image_history_input.clone(),
            sequence_history_input: self.sequence_history_input.clone(),
            sequence_glue_inputs: self.sequence_glue_inputs.clone(),
        });
    }

    fn tick_preference_persistence(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        if ctx.input(|input| !input.events.is_empty()) {
            self.preferences_dirty = true;
            self.next_preferences_save = now + Duration::from_secs(1);
        }
        if self.preferences_dirty && now >= self.next_preferences_save {
            self.save_preferences();
            self.preferences_dirty = false;
        }
        if self.preferences_dirty && self.next_preferences_save > now {
            ctx.request_repaint_after((self.next_preferences_save - now).min(Duration::from_secs(1)));
        }
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

    fn add_sequence_glue_input(&mut self, path: PathBuf) {
        let value = path.display().to_string();
        if !self.sequence_glue_inputs.iter().any(|entry| entry == &value) {
            self.sequence_glue_inputs.push(value);
            self.save_preferences();
        }
    }

    fn glue_sequences(&mut self) {
        if self.sequence_glue_inputs.len() < 2 {
            self.error = Some("Add at least two sequence folders to glue.".to_owned());
            return;
        }
        let parent = PathBuf::from(self.capture_base_dir_input.trim());
        if parent.as_os_str().is_empty() {
            self.error = Some("Set a capture/save directory before gluing sequences.".to_owned());
            return;
        }
        if let Err(error) = std::fs::create_dir_all(&parent) {
            self.error = Some(format!("Failed to create glue output directory {}: {error}", parent.display()));
            return;
        }

        let stamp = Local::now().format("%Y%m%d_%H%M%S_%3f").to_string();
        let root = parent.join(format!("glued_sequence_{stamp}"));
        let original_dir = root.join("original");
        if let Err(error) = std::fs::create_dir_all(&original_dir) {
            self.error = Some(format!("Failed to create {}: {error}", original_dir.display()));
            return;
        }

        let inputs = self.sequence_glue_inputs.clone();
        let result = (|| -> Result<(usize, Option<SystemTime>, Option<SystemTime>)> {
            let mut all_frames = Vec::<SequenceFrame>::new();
            for input in inputs {
                let sequence = load_image_sequence(Path::new(&input))
                    .with_context(|| format!("cannot load glue input {input}"))?;
                all_frames.extend(sequence.frames.into_iter().filter(|frame| frame.path.is_file()));
            }
            all_frames.sort_by(|a, b| {
                a.timestamp
                    .cmp(&b.timestamp)
                    .then_with(|| a.path.cmp(&b.path))
            });
            all_frames.dedup_by(|a, b| a.path == b.path);
            if all_frames.is_empty() {
                return Err(anyhow!("the selected sequences contain no existing image frames"));
            }

            let first_time = all_frames.first().map(|frame| frame.timestamp);
            let last_time = all_frames.last().map(|frame| frame.timestamp);
            let mut output_frames = Vec::with_capacity(all_frames.len());
            for (index, frame) in all_frames.into_iter().enumerate() {
                let extension = frame
                    .path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("png")
                    .to_ascii_lowercase();
                let file_name = format!("frame_{:08}.{extension}", index + 1);
                let destination = original_dir.join(&file_name);
                std::fs::copy(&frame.path, &destination).with_context(|| {
                    format!("cannot copy {} to {}", frame.path.display(), destination.display())
                })?;
                output_frames.push(SequenceFrame {
                    path: destination,
                    timestamp: frame.timestamp,
                });
            }

            let sequence = ImageSequence {
                root: root.clone(),
                frames: output_frames,
                selected: 0,
            };
            write_sequence_manifest(&sequence)?;
            Ok((sequence.frames.len(), first_time, last_time))
        })();

        match result {
            Ok((frame_count, first_time, last_time)) => {
                self.error = None;
                self.open_sequence(root.clone());
                if self.active_sequence.as_ref().is_some_and(|sequence| sequence.root == root) {
                    let span = match (first_time, last_time) {
                        (Some(first), Some(last)) => format!(
                            "{} → {}",
                            format_absolute_time(first),
                            format_absolute_time(last)
                        ),
                        _ => "unknown time span".to_owned(),
                    };
                    self.status = format!(
                        "Glued {frame_count} frame(s) chronologically ({span}) into {} and opened it",
                        root.display()
                    );
                }
            }
            Err(error) => {
                let _ = std::fs::remove_dir_all(&root);
                self.error = Some(format!("Failed to glue sequences: {error:#}"));
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
        self.active_sequence = Some(ImageSequence {
            root: root.clone(),
            frames: Vec::new(),
            selected: 0,
        });
        self.sequence_playing = false;
        self.next_sequence_frame_due = None;
        self.current_final_mask = None;
        self.active_processing_sequence_frame = None;
        self.sequence_mask_cache.clear();
        self.sequence_mask_cache_order.clear();
        self.shape_tracks.clear();
        self.capture_url = Some(url);
        self.pending_capture = None;
        self.processing_capture = None;
        self.deferred_sequence_frame = None;
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
        self.pending_capture = None;
        self.processing_capture = None;
        self.deferred_sequence_frame = None;
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
        {
            let Some(url) = self.capture_url.clone() else {
                self.stop_continuous_capture();
                return;
            };
            let captured_at = SystemTime::now();
            let timestamp: DateTime<Local> = captured_at.into();
            let (file_name, session_root) = {
                let Some(session) = self.capture_session.as_mut() else {
                    self.stop_continuous_capture();
                    return;
                };
                let file_name = format!(
                    "frame_{:06}_{}.png",
                    session.next_frame_index,
                    timestamp.format("%Y%m%d_%H%M%S_%3f")
                );
                session.next_frame_index = session.next_frame_index.saturating_add(1);
                (file_name, session.root.clone())
            };

            if self
                .active_sequence
                .as_ref()
                .is_none_or(|sequence| sequence.root != session_root)
            {
                self.active_sequence = Some(ImageSequence {
                    root: session_root.clone(),
                    frames: Vec::new(),
                    selected: 0,
                });
            }

            // Do not expose a frame to the live sequence until its backing file exists.
            // Previously the frame was appended optimistically here, so tracking/playback
            // could try to open a path that had not been written yet.
            self.queue_source(SourceRequest::CaptureUrl(url));
            self.pending_capture = Some(PendingCapture {
                source_id: self.active_source_id,
                processing_job_id: None,
                file_name,
                captured_at,
                sequence_frame_index: None,
            });
            if !self.source_loading {
                self.pending_capture = None;
            }
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

    fn remove_pending_capture_sequence_frame(&mut self) {
        let Some(frame_index) = self
            .pending_capture
            .as_ref()
            .and_then(|pending| pending.sequence_frame_index)
        else {
            return;
        };
        let manifest_result = {
            let Some(sequence) = self.active_sequence.as_mut() else {
                return;
            };
            if frame_index >= sequence.frames.len() {
                return;
            }
            sequence.frames.remove(frame_index);
            sequence.selected = sequence.selected.min(sequence.frames.len().saturating_sub(1));
            write_sequence_manifest(sequence)
        };
        if let Err(error) = manifest_result {
            self.error = Some(format!("Failed to update sequence manifest: {error:#}"));
        }
    }

    fn append_capture_to_sequence(
        &mut self,
        path: PathBuf,
        captured_at: SystemTime,
    ) -> Option<usize> {
        if !path.is_file() {
            self.error = Some(format!(
                "Capture was not added to sequence because {} does not exist",
                path.display()
            ));
            return None;
        }

        let session_root = self.capture_session.as_ref()?.root.clone();
        if self
            .active_sequence
            .as_ref()
            .is_none_or(|sequence| sequence.root != session_root)
        {
            self.active_sequence = Some(ImageSequence {
                root: session_root,
                frames: Vec::new(),
                selected: 0,
            });
        }

        let committed_path = path.clone();
        let (index, manifest_result) = {
            let sequence = self.active_sequence.as_mut()?;
            if !sequence.frames.iter().any(|frame| frame.path == committed_path) {
                sequence.frames.push(SequenceFrame {
                    path,
                    timestamp: captured_at,
                });
            }
            sequence.frames.sort_by(|a, b| {
                a.timestamp
                    .cmp(&b.timestamp)
                    .then_with(|| a.path.cmp(&b.path))
            });
            let index = sequence
                .frames
                .iter()
                .position(|frame| frame.path == committed_path)
                .unwrap_or_else(|| sequence.frames.len().saturating_sub(1));
            sequence.selected = index;
            (index, write_sequence_manifest(sequence))
        };

        if let Err(error) = manifest_result {
            self.error = Some(format!("Failed to update sequence manifest: {error:#}"));
        }
        Some(index)
    }

    fn save_pending_capture_original(&mut self) -> Option<usize> {
        if !self.capture_save_original {
            return None;
        }
        let (path, captured_at) = {
            let session = self.capture_session.as_ref()?;
            let pending = self.pending_capture.as_ref()?;
            (session.original_dir.join(&pending.file_name), pending.captured_at)
        };
        let image = self.original_rgba.as_ref()?.as_ref().clone();
        if let Err(error) = DynamicImage::ImageRgba8(image).save(&path) {
            self.error = Some(format!("Failed to save capture {}: {error}", path.display()));
            return None;
        }
        let index = self.append_capture_to_sequence(path, captured_at);
        if let (Some(index), Some(pending)) = (index, self.pending_capture.as_mut()) {
            pending.sequence_frame_index = Some(index);
        }
        index
    }

    fn save_pending_capture_processed(
        &mut self,
        completed_job_id: u64,
        image: &RgbaImage,
    ) -> Option<usize> {
        let should_finish = self
            .processing_capture
            .as_ref()
            .and_then(|pending| pending.processing_job_id)
            .is_some_and(|id| id == completed_job_id);
        if !should_finish {
            return None;
        }

        let mut committed_index = self
            .processing_capture
            .as_ref()
            .and_then(|pending| pending.sequence_frame_index);
        if self.capture_save_processed {
            let save_target = self.capture_session.as_ref().and_then(|session| {
                self.processing_capture.as_ref().map(|pending| {
                    (
                        session.processed_dir.join(&pending.file_name),
                        pending.captured_at,
                    )
                })
            });
            if let Some((path, captured_at)) = save_target {
                if let Err(error) = DynamicImage::ImageRgba8(image.clone()).save(&path) {
                    self.error = Some(format!(
                        "Failed to save processed capture {}: {error}",
                        path.display()
                    ));
                } else {
                    if committed_index.is_none() {
                        committed_index = self.append_capture_to_sequence(path.clone(), captured_at);
                    }
                    let absolute = format_absolute_time(captured_at);
                    self.status = format!("Captured {} ({absolute})", path.display());
                }
            }
        }
        self.processing_capture = None;
        committed_index
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
                            .processing_capture
                            .as_ref()
                            .and_then(|pending| pending.processing_job_id)
                            .is_some_and(|job_id| job_id == id)
                        {
                            self.processing_capture = None;
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
                    self.error = None;

                    self.processed_rgba = Some(result.processed.clone());
                    let committed_capture_index =
                        self.save_pending_capture_processed(id, &result.processed);
                    let frame_index = committed_capture_index.or(self.active_processing_sequence_frame);
                    if let Some(frame_index) = frame_index {
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
                }
                ProcessingMessage::Failed { id, error } if id == self.active_job_id => {
                    self.processing = false;
                    self.progress_stage = "Failed".to_owned();
                    if self
                        .processing_capture
                        .as_ref()
                        .and_then(|pending| pending.processing_job_id)
                        .is_some_and(|job_id| job_id == id)
                    {
                        self.processing_capture = None;
                    }
                    self.error = Some(error);
                }
                _ => {}
            }
        }
    }

    fn temporal_support_radius(&self) -> usize {
        (self.temporal_lookahead_frames.max(4) / 2)
            .saturating_sub(1)
            .max(1)
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
        let radius = self.temporal_support_radius();
        let mut support = Vec::with_capacity(radius * 2);

        for offset in 1..=radius {
            if let Some(index) = current.checked_sub(offset) {
                if let Some(cached) = self.sequence_mask_cache.get(&index) {
                    if expected_dims.is_none_or(|dims| dims == (cached.width, cached.height)) {
                        support.push(Arc::clone(&cached.mask));
                    }
                }
            }
            if let Some(index) = current.checked_add(offset).filter(|index| *index < sequence.frames.len()) {
                if let Some(cached) = self.sequence_mask_cache.get(&index) {
                    if expected_dims.is_none_or(|dims| dims == (cached.width, cached.height)) {
                        support.push(Arc::clone(&cached.mask));
                    }
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
            if !track.enabled || track.anchor_width != width || track.anchor_height != height {
                continue;
            }
            update_one_shape_track(track, frame_index, width, &components, threshold);
        }
    }

    fn rebuild_shape_track_from_cached_masks(&mut self, track_id: u64) {
        let Some(track_index) = self.shape_tracks.iter().position(|track| track.id == track_id) else {
            return;
        };
        let anchor_frame = self.shape_tracks[track_index].anchor_frame;
        let anchor_width = self.shape_tracks[track_index].anchor_width;
        let anchor_height = self.shape_tracks[track_index].anchor_height;
        let anchor_pixels = self.shape_tracks[track_index].anchor_pixels.clone();
        let anchor_pivot = self.shape_tracks[track_index].anchor_pivot;
        let anchor_area = anchor_pixels.len();
        let threshold = self.track_overlap_threshold.clamp(0.0, 1.0);

        // Snapshot the cache before mutably borrowing the track. Recording already
        // populated many of these masks, so pivoting can immediately backfill the
        // historical plot without requiring playback.
        let cached = self
            .sequence_mask_cache
            .iter()
            .filter(|(_, cached)| cached.width == anchor_width && cached.height == anchor_height)
            .map(|(&index, cached)| (index, Arc::clone(&cached.mask)))
            .collect::<BTreeMap<_, _>>();

        let track = &mut self.shape_tracks[track_index];
        track.observations.clear();
        track.matched_pixels.clear();
        track.matched_centroids.clear();
        track.observations.insert(
            anchor_frame,
            ShapeObservation {
                area: anchor_area,
                overlap: 1.0,
            },
        );
        track.matched_pixels.insert(anchor_frame, anchor_pixels);
        track.matched_centroids.insert(anchor_frame, anchor_pivot);

        let max_distance = cached
            .keys()
            .map(|index| index.abs_diff(anchor_frame))
            .max()
            .unwrap_or(0);
        for distance in 1..=max_distance {
            for index in [
                anchor_frame.checked_sub(distance),
                anchor_frame.checked_add(distance),
            ]
            .into_iter()
            .flatten()
            {
                let Some(mask) = cached.get(&index) else {
                    continue;
                };
                let components = extract_mask_components(mask.as_ref(), anchor_width, anchor_height);
                if components.is_empty() {
                    continue;
                }
                update_one_shape_track(track, index, anchor_width, &components, threshold);
            }
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
        let track_id = self.next_track_id;
        let anchor_pixels = component.pixels;
        let mut observations = BTreeMap::new();
        observations.insert(
            frame_index,
            ShapeObservation {
                area: component.area,
                overlap: 1.0,
            },
        );
        let mut matched_pixels = BTreeMap::new();
        matched_pixels.insert(frame_index, anchor_pixels.clone());
        let mut matched_centroids = BTreeMap::new();
        matched_centroids.insert(frame_index, pixel);

        self.shape_tracks.push(ShapeTrack {
            id: track_id,
            name: format!("shape-{track_id}"),
            anchor_pivot: pixel,
            anchor_frame: frame_index,
            anchor_width: width,
            anchor_height: height,
            anchor_pixels,
            matched_pixels,
            matched_centroids,
            observations,
            enabled: true,
        });
        self.rebuild_shape_track_from_cached_masks(track_id);
        self.shape_plot_open = true;
        let samples = self
            .shape_tracks
            .iter()
            .find(|track| track.id == track_id)
            .map(|track| track.observations.len())
            .unwrap_or(1);
        self.status = format!(
            "Tracking shape {track_id} anchored at frame {} / pivot ({}, {}); {} cached sample(s) matched.",
            frame_index.saturating_add(1),
            pixel.0,
            pixel.1,
            samples
        );
    }

    fn track_overlays(&self) -> Vec<viewer::OverlayPoint> {
        let current_frame = self.active_sequence.as_ref().map(|sequence| sequence.selected);
        self.shape_tracks
            .iter()
            .filter(|track| track.enabled)
            .map(|track| {
                let pixel = current_frame
                    .and_then(|frame| track.matched_centroids.get(&frame).copied())
                    .unwrap_or(track.anchor_pivot);
                viewer::OverlayPoint {
                    pixel,
                    color: track_color(track.id),
                    label: if current_frame == Some(track.anchor_frame) {
                        format!("T{} anchor", track.id)
                    } else {
                        format!("T{}", track.id)
                    },
                }
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

        ui.collapsing("Glue sequences", |ui| {
            ui.small("Add two or more sequence folders. Glue order is deduced automatically from every frame timestamp (oldest → newest); list order does not matter. Output is created inside the Save directory below.");
            ui.horizontal_wrapped(|ui| {
                if ui.button("Add active").clicked() {
                    if let Some(root) = self.active_sequence.as_ref().map(|sequence| sequence.root.clone()) {
                        self.add_sequence_glue_input(root);
                    }
                }
                if ui.button("Add folder…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_title("Add sequence to glue list")
                        .pick_folder()
                    {
                        self.add_sequence_glue_input(path);
                    }
                }
                if ui.button("Clear list").clicked() {
                    self.sequence_glue_inputs.clear();
                    self.save_preferences();
                }
            });
            let mut remove = None;
            for (index, entry) in self.sequence_glue_inputs.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.monospace(format!("{:>2}. {}", index + 1, entry));
                    if ui.small_button("×").clicked() {
                        remove = Some(index);
                    }
                });
            }
            if let Some(index) = remove {
                self.sequence_glue_inputs.remove(index);
                self.save_preferences();
            }
            if ui
                .add_enabled(
                    self.sequence_glue_inputs.len() >= 2,
                    egui::Button::new("Glue chronologically into new sequence…"),
                )
                .clicked()
            {
                self.glue_sequences();
            }
        });

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
        ui.small("Capture acquisition is independent from mask processing: raw originals keep the requested interval even if YOLO/temporal processing is slower. If only Processed saving is enabled, frames can be skipped when processing cannot keep up.");

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
            let overlap_changed = ui
                .add(
                    egui::Slider::new(&mut self.track_overlap_threshold, 0.0..=1.0)
                        .text("Same-shape overlap")
                        .fixed_decimals(2),
                )
                .changed();
            if overlap_changed {
                let track_ids = self.shape_tracks.iter().map(|track| track.id).collect::<Vec<_>>();
                for track_id in track_ids {
                    self.rebuild_shape_track_from_cached_masks(track_id);
                }
                self.save_preferences();
            }
            ui.small("Hover a closed component in Processed and press P. The clicked pixel + frame become an immutable identity anchor; matching then propagates outward through neighbouring cached masks.");

            ui.separator();
            let mut temporal_changed = false;
            temporal_changed |= ui
                .checkbox(
                    &mut self.temporal_filter_enabled,
                    "Filter transient components using surrounding frames",
                )
                .changed();
            temporal_changed |= ui
                .add(
                    egui::Slider::new(&mut self.temporal_lookahead_frames, 4..=26)
                        .text("Temporal window n"),
                )
                .changed();
            let support_radius = self.temporal_support_radius();
            let max_confirmations = support_radius * 2;
            if self.temporal_required_frames > max_confirmations {
                self.temporal_required_frames = max_confirmations;
            }
            temporal_changed |= ui
                .add(
                    egui::Slider::new(
                        &mut self.temporal_required_frames,
                        1..=max_confirmations.max(1),
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
                self.save_preferences();
                self.schedule_processing();
            }

            let radius = self.temporal_support_radius();
            ui.small(format!("Support radius = floor(n / 2) - 1 = {radius} frame(s) in each direction."));
            let (back_cached, forward_cached) = self
                .active_sequence
                .as_ref()
                .map(|sequence| {
                    let back = (1..=radius)
                        .filter_map(|offset| sequence.selected.checked_sub(offset))
                        .filter(|index| self.sequence_mask_cache.contains_key(index))
                        .count();
                    let forward = (1..=radius)
                        .filter_map(|offset| sequence.selected.checked_add(offset))
                        .filter(|index| *index < sequence.frames.len())
                        .filter(|index| self.sequence_mask_cache.contains_key(index))
                        .count();
                    (back, forward)
                })
                .unwrap_or((0, 0));
            ui.small(format!(
                "Window n={} → {} backward + {} forward; cached support {}/{} + {}/{}; required {}.",
                self.temporal_lookahead_frames, radius, radius, back_cached, radius, forward_cached, radius, self.temporal_required_frames
            ));
            ui.small("Temporal filtering uses cached masks on both sides of the current frame. Playing/analyzing the sequence populates the cache; edge frames naturally have fewer neighbours.");

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
                    ui.monospace(format!(
                        "{} samples · anchor f{} @ {},{}",
                        track.observations.len(),
                        track.anchor_frame.saturating_add(1),
                        track.anchor_pivot.0,
                        track.anchor_pivot.1
                    ));
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
        let response = egui::Window::new("Tracked mask size")
            .id(egui::Id::new("tracked-mask-size-window"))
            .open(&mut open)
            .default_pos(self.shape_plot_window_pos)
            .default_size(self.shape_plot_window_size)
            .current_pos(self.shape_plot_window_pos)
            .min_size(egui::vec2(420.0, 240.0))
            .constrain(false)
            .resizable([true, true])
            .show(ctx, |ui| {
                ui.small("X = absolute capture time · Y = mask area in pixels. Dashed vertical line = currently rendered frame.");

                // Size the canvas from the actual persisted/user-picked outer window
                // rectangle, not from the Resize max-rect. Using the max-rect here makes
                // a canvas self-request the maximum height, which in turn prevents the
                // user from shrinking the plot window. During a drag this trails the
                // resize by at most one frame, then follows the new size exactly.
                let desired_plot_width = (self.shape_plot_window_size.x - 28.0).max(360.0);
                let desired_plot_height = (self.shape_plot_window_size.y - 104.0).max(130.0);
                let available_width = ui.available_width();
                let plot_width = if available_width.is_finite() {
                    desired_plot_width.min(available_width.max(360.0))
                } else {
                    desired_plot_width
                };
                let plot_size = egui::vec2(plot_width, desired_plot_height);
                let (response, painter) =
                    ui.allocate_painter(plot_size, egui::Sense::hover());
                let rect = response.rect.shrink2(egui::vec2(58.0, 32.0));
                let border = egui::Stroke::new(1.0, egui::Color32::DARK_GRAY);
                painter.line_segment([rect.left_top(), rect.right_top()], border);
                painter.line_segment([rect.right_top(), rect.right_bottom()], border);
                painter.line_segment([rect.right_bottom(), rect.left_bottom()], border);
                painter.line_segment([rect.left_bottom(), rect.left_top()], border);

                let mut max_area = 1usize;
                for track in self.shape_tracks.iter().filter(|track| track.enabled) {
                    max_area = max_area.max(
                        track
                            .observations
                            .values()
                            .map(|observation| observation.area)
                            .max()
                            .unwrap_or(1),
                    );
                }

                let sequence = self.active_sequence.as_ref();
                let (start_time, end_time) = sequence
                    .and_then(|sequence| {
                        Some((
                            sequence.frames.first()?.timestamp,
                            sequence.frames.last()?.timestamp,
                        ))
                    })
                    .unwrap_or((UNIX_EPOCH, UNIX_EPOCH + Duration::from_secs(1)));
                let span = end_time
                    .duration_since(start_time)
                    .unwrap_or(Duration::from_secs(1))
                    .max(Duration::from_millis(1));

                let x_for_frame = |frame: usize| -> Option<f32> {
                    let sequence = sequence?;
                    let timestamp = sequence.frames.get(frame)?.timestamp;
                    let offset = timestamp.duration_since(start_time).ok()?;
                    Some(
                        rect.left()
                            + (offset.as_secs_f64() / span.as_secs_f64()) as f32 * rect.width(),
                    )
                };

                for step in 0..=4 {
                    let t = step as f32 / 4.0;
                    let y = egui::lerp(rect.bottom()..=rect.top(), t);
                    painter.line_segment(
                        [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                        egui::Stroke::new(0.5, egui::Color32::from_gray(45)),
                    );
                    painter.text(
                        egui::pos2(rect.left() - 7.0, y),
                        egui::Align2::RIGHT_CENTER,
                        format!("{} px", (max_area as f32 * t).round() as usize),
                        egui::FontId::monospace(10.0),
                        egui::Color32::GRAY,
                    );
                }

                for step in 0..=4 {
                    let t = step as f64 / 4.0;
                    let tick_time = start_time + Duration::from_secs_f64(span.as_secs_f64() * t);
                    let x = egui::lerp(rect.left()..=rect.right(), t as f32);
                    painter.line_segment(
                        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                        egui::Stroke::new(0.5, egui::Color32::from_gray(35)),
                    );
                    painter.text(
                        egui::pos2(x, rect.bottom() + 8.0),
                        egui::Align2::CENTER_TOP,
                        format_plot_axis_time(tick_time, span),
                        egui::FontId::monospace(10.0),
                        egui::Color32::GRAY,
                    );
                }

                for track in self.shape_tracks.iter().filter(|track| track.enabled) {
                    let color = track_color(track.id);
                    let mut points = Vec::new();
                    for (&frame, observation) in &track.observations {
                        let Some(x) = x_for_frame(frame) else {
                            continue;
                        };
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

                    if let Some(x) = x_for_frame(track.anchor_frame)
                        && let Some(observation) = track.observations.get(&track.anchor_frame)
                    {
                        let y = rect.bottom()
                            - observation.area as f32 / max_area.max(1) as f32 * rect.height();
                        painter.circle_stroke(
                            egui::pos2(x, y),
                            6.0,
                            egui::Stroke::new(2.0, color),
                        );
                    }
                }

                if let Some(sequence) = sequence
                    && let Some(x) = x_for_frame(sequence.selected)
                {
                    let current_stroke =
                        egui::Stroke::new(1.5, egui::Color32::from_rgb(245, 245, 245));
                    let dash = 7.0;
                    let gap = 5.0;
                    let mut y = rect.top();
                    while y < rect.bottom() {
                        let y2 = (y + dash).min(rect.bottom());
                        painter.line_segment(
                            [egui::pos2(x, y), egui::pos2(x, y2)],
                            current_stroke,
                        );
                        y += dash + gap;
                    }
                    if let Some(frame) = sequence.frames.get(sequence.selected) {
                        painter.text(
                            egui::pos2(x, rect.top() - 6.0),
                            egui::Align2::CENTER_BOTTOM,
                            format!(
                                "F{} {}",
                                sequence.selected.saturating_add(1),
                                format_plot_axis_time(frame.timestamp, span)
                            ),
                            egui::FontId::monospace(10.0),
                            egui::Color32::WHITE,
                        );
                    }
                }

                ui.horizontal_wrapped(|ui| {
                    for track in self.shape_tracks.iter().filter(|track| track.enabled) {
                        let last = track.observations.last_key_value();
                        let suffix = last
                            .map(|(_, observation)| {
                                format!("{} px · overlap {:.2}", observation.area, observation.overlap)
                            })
                            .unwrap_or_else(|| "no sample".to_owned());
                        ui.colored_label(
                            track_color(track.id),
                            format!(
                                "T{} {}: {} · anchor F{} @ {},{}",
                                track.id,
                                track.name,
                                suffix,
                                track.anchor_frame.saturating_add(1),
                                track.anchor_pivot.0,
                                track.anchor_pivot.1
                            ),
                        );
                    }
                });
            });
        if let Some(response) = response {
            self.shape_plot_window_pos = response.response.rect.min;
            self.shape_plot_window_size = response.response.rect.size();
        }
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
    fn save(&mut self, _storage: &mut dyn eframe::Storage) {
        self.save_preferences();
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_dropped_files(&ctx);
        self.poll_source_worker(&ctx);
        self.poll_yolo_worker(&ctx);
        self.poll_processing_worker(&ctx);
        self.tick_continuous_capture(&ctx);
        self.tick_sequence_playback(&ctx);

        egui::CentralPanel::default().show(ui, |ui| self.previews(ui));

        if let Some(response) = egui::Window::new("Controls / sequence")
            .id(egui::Id::new("controls-sequence-window"))
            .default_pos(self.controls_window_pos)
            .default_size(self.controls_window_size)
            .current_pos(self.controls_window_pos)
            .min_size(egui::vec2(330.0, 320.0))
            .constrain(false)
            .resizable([true, true])
            .hscroll(true)
            .vscroll(true)
            .show(&ctx, |ui| {
                // Keep a stable content width so shrinking the floating window exposes
                // a horizontal scrollbar instead of collapsing controls into oblivion.
                ui.set_min_width(760.0);
                self.controls(ui)
            })
        {
            self.controls_window_pos = response.response.rect.min;
            self.controls_window_size = response.response.rect.size();
        }

        self.shape_size_plot_window(&ctx);
        self.tick_preference_persistence(&ctx);
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
        .user_agent("rust-edge-gui/0.7.3")
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


fn component_overlap_score(a: &[u32], b: &[u32]) -> f32 {
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
    let containment = intersection as f32 / a.len().min(b.len()).max(1) as f32;
    component_iou(a, b).max(containment)
}

fn centroid_distance_sq(a: (u32, u32), b: (u32, u32)) -> u64 {
    let dx = a.0 as i64 - b.0 as i64;
    let dy = a.1 as i64 - b.1 as i64;
    (dx * dx + dy * dy) as u64
}

fn update_one_shape_track(
    track: &mut ShapeTrack,
    frame_index: usize,
    width: usize,
    components: &[MaskComponent],
    threshold: f32,
) {
    if frame_index == track.anchor_frame {
        return;
    }

    let Some((reference_frame_ref, reference_pixels)) = track
        .matched_pixels
        .iter()
        .min_by_key(|(known_frame, _)| (**known_frame).abs_diff(frame_index))
    else {
        return;
    };
    let reference_frame = *reference_frame_ref;
    let reference_centroid = track
        .matched_centroids
        .get(&reference_frame)
        .copied()
        .unwrap_or(track.anchor_pivot);

    let mut best: Option<(&MaskComponent, f32, u64, (u32, u32))> = None;
    for component in components {
        let overlap = component_overlap_score(reference_pixels, &component.pixels);
        let centroid = component_centroid(component, width);
        let distance = centroid_distance_sq(reference_centroid, centroid);
        let replace = match best {
            None => true,
            Some((_, best_overlap, best_distance, _)) => {
                overlap > best_overlap + f32::EPSILON
                    || ((overlap - best_overlap).abs() <= f32::EPSILON
                        && distance < best_distance)
            }
        };
        if replace {
            best = Some((component, overlap, distance, centroid));
        }
    }

    let Some((component, overlap, _, centroid)) = best else {
        return;
    };
    if overlap + f32::EPSILON < threshold {
        return;
    }

    track.observations.insert(
        frame_index,
        ShapeObservation {
            area: component.area,
            overlap,
        },
    );
    track.matched_pixels.insert(frame_index, component.pixels.clone());
    track.matched_centroids.insert(frame_index, centroid);
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

fn write_sequence_manifest(sequence: &ImageSequence) -> Result<()> {
    std::fs::create_dir_all(&sequence.root)
        .with_context(|| format!("cannot create {}", sequence.root.display()))?;
    let mut manifest_frames = Vec::with_capacity(sequence.frames.len());
    for frame in &sequence.frames {
        if !frame.path.is_file() {
            continue;
        }
        let relative = frame
            .path
            .strip_prefix(&sequence.root)
            .unwrap_or(frame.path.as_path())
            .to_string_lossy()
            .replace('\\', "/");
        let timestamp_ms = frame
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(u64::MAX as u128) as u64;
        manifest_frames.push(SequenceManifestFrame {
            path: relative,
            timestamp_ms,
        });
    }
    manifest_frames.sort_by(|a, b| {
        a.timestamp_ms
            .cmp(&b.timestamp_ms)
            .then_with(|| a.path.cmp(&b.path))
    });
    let manifest = SequenceManifest {
        version: 1,
        frames: manifest_frames,
    };
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    let target = sequence.root.join("sequence.json");
    let temporary = sequence.root.join("sequence.json.tmp");
    std::fs::write(&temporary, bytes)
        .with_context(|| format!("cannot write {}", temporary.display()))?;
    std::fs::rename(&temporary, &target)
        .with_context(|| format!("cannot replace {}", target.display()))?;
    Ok(())
}

fn load_image_sequence(root: &Path) -> Result<ImageSequence> {
    if !root.exists() {
        return Err(anyhow!("{} does not exist", root.display()));
    }
    if !root.is_dir() {
        return Err(anyhow!("{} is not a directory", root.display()));
    }

    let manifest_path = root.join("sequence.json");
    if manifest_path.is_file() {
        let bytes = std::fs::read(&manifest_path)
            .with_context(|| format!("cannot read {}", manifest_path.display()))?;
        let manifest: SequenceManifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("cannot parse {}", manifest_path.display()))?;
        if manifest.version != 1 {
            return Err(anyhow!("unsupported sequence manifest version {} in {}", manifest.version, manifest_path.display()));
        }
        let mut frames = Vec::with_capacity(manifest.frames.len());
        for frame in manifest.frames {
            let path = PathBuf::from(&frame.path);
            let path = if path.is_absolute() { path } else { root.join(path) };
            if path.is_file() && is_supported_image_path(&path) {
                frames.push(SequenceFrame {
                    path,
                    timestamp: UNIX_EPOCH + Duration::from_millis(frame.timestamp_ms),
                });
            }
        }
        if !frames.is_empty() {
            frames.sort_by(|a, b| {
                a.timestamp
                    .cmp(&b.timestamp)
                    .then_with(|| a.path.cmp(&b.path))
            });
            return Ok(ImageSequence { root: root.to_path_buf(), frames, selected: 0 });
        }
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

    frames.sort_by(|a, b| {
        a.timestamp
            .cmp(&b.timestamp)
            .then_with(|| a.path.cmp(&b.path))
    });
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

fn format_plot_axis_time(time: SystemTime, span: Duration) -> String {
    let date_time: DateTime<Local> = time.into();
    if span >= Duration::from_secs(24 * 60 * 60) {
        date_time.format("%m-%d %H:%M").to_string()
    } else if span >= Duration::from_secs(60 * 60) {
        date_time.format("%H:%M").to_string()
    } else {
        date_time.format("%H:%M:%S").to_string()
    }
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
