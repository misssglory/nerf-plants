mod viewer;
mod yolo;
mod home_assistant;
mod telegram;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
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
use home_assistant::{HaMessage, HaRequest, HaRequestKind, HaSeriesRequest, HaWorker};
use telegram::{load_telegram_config, TelegramConfig, TelegramNotifier};

const APP_TITLE: &str = "Rust Edge GUI v0.8.2 — Capture Reliability & Telegram Alerts";
const MAX_HISTORY: usize = 20;
const MIN_CAPTURE_INTERVAL_SECONDS: f32 = 0.1;
const MAX_SEQUENCE_MASK_CACHE: usize = 64;
const MAX_EXTERNAL_SERIES_POINTS: usize = 50_000;
const MAX_STATUS_HISTORY: usize = 250;

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
    /// Component-filtered mask before temporal persistence filtering. Keeping this
    /// separately lets later frames both remove false positives and restore newly
    /// confirmed sprouts in older frames.
    raw_shape_mask: Vec<bool>,
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
            selection_start: None,
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

fn default_plot_center() -> f64 { 0.5 }
fn default_plot_center_f32() -> f32 { 0.5 }
fn default_plot_zoom() -> f32 { 1.0 }

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SeriesMeta {
    key: String,
    name: String,
    group: String,
    plot_id: u64,
    #[serde(default = "default_true")]
    visible: bool,
}

fn default_true() -> bool { true }

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HaSeriesConfig {
    key: String,
    entity_id: String,
    #[serde(default)]
    attribute: String,
}

#[derive(Clone, Debug)]
struct HaSeriesData {
    unit: String,
    friendly_name: String,
    points: BTreeMap<SystemTime, f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedPlotWindow {
    id: u64,
    title: String,
    open: bool,
    pos: [f32; 2],
    size: [f32; 2],
    #[serde(default = "default_plot_center")]
    x_center: f64,
    #[serde(default = "default_plot_zoom")]
    x_zoom: f32,
    #[serde(default = "default_plot_center_f32")]
    y_center: f32,
    #[serde(default = "default_plot_zoom")]
    y_zoom: f32,
}

#[derive(Clone, Debug)]
struct PlotWindowState {
    id: u64,
    title: String,
    open: bool,
    pos: egui::Pos2,
    size: egui::Vec2,
    x_center: f64,
    x_zoom: f32,
    y_center: f32,
    y_zoom: f32,
}

impl PlotWindowState {
    fn from_persisted(value: PersistedPlotWindow) -> Self {
        Self {
            id: value.id.max(1),
            title: if value.title.trim().is_empty() { format!("Plot {}", value.id.max(1)) } else { value.title },
            open: value.open,
            pos: egui::pos2(value.pos[0], value.pos[1]),
            size: egui::vec2(value.size[0].max(160.0), value.size[1].max(58.0)),
            x_center: value.x_center.clamp(0.0, 1.0),
            x_zoom: value.x_zoom.clamp(1.0, 10_000.0),
            y_center: value.y_center.clamp(0.0, 1.0),
            y_zoom: value.y_zoom.clamp(1.0, 10_000.0),
        }
    }

    fn persisted(&self) -> PersistedPlotWindow {
        PersistedPlotWindow {
            id: self.id,
            title: self.title.clone(),
            open: self.open,
            pos: [self.pos.x, self.pos.y],
            size: [self.size.x, self.size.y],
            x_center: self.x_center,
            x_zoom: self.x_zoom,
            y_center: self.y_center,
            y_zoom: self.y_zoom,
        }
    }
}

#[derive(Clone, Debug)]
struct PlotPoint {
    time: SystemTime,
    value: f64,
    frame: Option<usize>,
}

#[derive(Clone, Debug)]
struct PlotSeriesData {
    key: String,
    name: String,
    group: String,
    unit: String,
    color_seed: u64,
    points: Vec<PlotPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedStatusEvent {
    timestamp_ms: u64,
    text: String,
    is_error: bool,
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct PersistedState {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    source_history: Vec<String>,
    #[serde(default)]
    status_history: Vec<PersistedStatusEvent>,
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
    plot_click_seek_enabled: bool,
    #[serde(default = "default_plot_center")]
    plot_x_center: f64,
    #[serde(default = "default_plot_zoom")]
    plot_x_zoom: f32,
    #[serde(default = "default_plot_center_f32")]
    plot_y_center: f32,
    #[serde(default = "default_plot_zoom")]
    plot_y_zoom: f32,
    #[serde(default)]
    url_input: String,
    #[serde(default)]
    image_history_input: String,
    #[serde(default)]
    sequence_history_input: String,
    #[serde(default)]
    sequence_glue_inputs: Vec<String>,
    #[serde(default)]
    plot_windows: Vec<PersistedPlotWindow>,
    #[serde(default)]
    series_meta: Vec<SeriesMeta>,
    #[serde(default)]
    ha_base_url: String,
    #[serde(default)]
    ha_token: String,
    #[serde(default)]
    ha_remember_token: bool,
    #[serde(default)]
    ha_auto_poll: bool,
    #[serde(default)]
    ha_poll_interval_secs: f32,
    #[serde(default)]
    ha_history_hours: f32,
    #[serde(default)]
    ha_series: Vec<HaSeriesConfig>,
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
    /// Permanent group identity. Tracks are grouped when their masks collide into
    /// the same connected component in any frame. Grouping is retroactive.
    group_id: u64,
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
    /// Auto-created pivots may be retired when temporal filtering confirms that
    /// their shape disappeared. Manual P pivots are kept until the user removes them.
    auto_created: bool,
}


#[derive(Clone, Debug)]
struct ShapeGroupSeries {
    id: u64,
    name: String,
    member_count: usize,
    anchor_frame: usize,
    areas: BTreeMap<usize, usize>,
    centroids: BTreeMap<usize, (u32, u32)>,
}

#[derive(Clone, Debug)]
struct StatusEvent {
    time: SystemTime,
    text: String,
    is_error: bool,
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
    retry_count: usize,
    failure_notified: bool,
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
    status_changed_at: SystemTime,
    status_history: VecDeque<StatusEvent>,
    last_status_signature: String,

    original_view: viewer::ImageViewState,
    processed_view: viewer::ImageViewState,

    current_final_mask: Option<Arc<Vec<bool>>>,
    active_processing_sequence_frame: Option<usize>,
    /// Final masks after temporal filtering.
    sequence_mask_cache: BTreeMap<usize, CachedSequenceMask>,
    /// Pre-temporal component masks. Temporal support must be computed from this
    /// cache so a component that was initially filtered out can be restored once
    /// future frames confirm it.
    sequence_raw_mask_cache: BTreeMap<usize, CachedSequenceMask>,
    sequence_mask_cache_order: VecDeque<usize>,
    temporal_filter_enabled: bool,
    temporal_lookahead_frames: usize,
    temporal_required_frames: usize,
    temporal_overlap_threshold: f32,
    track_overlap_threshold: f32,
    shape_tracks: Vec<ShapeTrack>,
    next_track_id: u64,
    /// Set whenever a new sequence is opened/started. The first processed frame
    /// seeds one pivot for every closed connected mask component automatically.
    auto_seed_pivots_pending: bool,
    shape_plot_open: bool,
    plot_click_seek_enabled: bool,
    /// Normalized center/zoom of the interactive plot viewport. X is capture time,
    /// Y is grouped mask area. Keeping these normalized makes the view stable when
    /// sequences are extended while recording.
    plot_x_center: f64,
    plot_x_zoom: f32,
    plot_y_center: f32,
    plot_y_zoom: f32,
    sequence_playing: bool,
    sequence_playback_fps: f32,
    sequence_loop: bool,
    sequence_wait_processing: bool,
    next_sequence_frame_due: Option<Instant>,
    sequence_glue_inputs: Vec<String>,

    // Generic timeseries dashboard. Shape-area series and external Home Assistant
    // sensors use the same metadata/plot assignment model.
    series_meta: BTreeMap<String, SeriesMeta>,
    plot_windows: Vec<PlotWindowState>,
    next_plot_window_id: u64,
    next_selection_group_id: u64,
    auto_track_highwater_frame: Option<usize>,

    ha_worker: HaWorker,
    ha_base_url_input: String,
    ha_token_input: String,
    ha_remember_token: bool,
    ha_auto_poll: bool,
    ha_poll_interval_secs: f32,
    ha_history_hours: f32,
    ha_entity_id_input: String,
    ha_attribute_input: String,
    ha_series: Vec<HaSeriesConfig>,
    ha_data: BTreeMap<String, HaSeriesData>,
    ha_request_id: u64,
    ha_request_pending: bool,
    next_ha_poll: Option<Instant>,
    ha_status: String,

    telegram_config: TelegramConfig,
    telegram_notifier: TelegramNotifier,
    telegram_config_status: String,
    capture_retry_due: Option<Instant>,

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
        let persisted_v3 = persisted.version >= 3;
        let persisted_v4 = persisted.version >= 4;
        let persisted_v5 = persisted.version >= 5;
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

        let mut plot_windows = if persisted_v5 && !persisted.plot_windows.is_empty() {
            persisted
                .plot_windows
                .clone()
                .into_iter()
                .map(PlotWindowState::from_persisted)
                .collect::<Vec<_>>()
        } else {
            vec![PlotWindowState {
                id: 1,
                title: "Plant / sensor plot".to_owned(),
                open: if persisted_v2 { persisted.shape_plot_open } else { false },
                pos: shape_plot_window.pos(),
                size: shape_plot_window.size(),
                x_center: if persisted_v4 { persisted.plot_x_center.clamp(0.0, 1.0) } else { 0.5 },
                x_zoom: if persisted_v4 { persisted.plot_x_zoom.clamp(1.0, 10_000.0) } else { 1.0 },
                y_center: if persisted_v4 { persisted.plot_y_center.clamp(0.0, 1.0) } else { 0.5 },
                y_zoom: if persisted_v4 { persisted.plot_y_zoom.clamp(1.0, 10_000.0) } else { 1.0 },
            }]
        };
        if plot_windows.is_empty() {
            plot_windows.push(PlotWindowState {
                id: 1,
                title: "Plant / sensor plot".to_owned(),
                open: false,
                pos: shape_plot_window.pos(),
                size: shape_plot_window.size(),
                x_center: 0.5,
                x_zoom: 1.0,
                y_center: 0.5,
                y_zoom: 1.0,
            });
        }
        let next_plot_window_id = plot_windows.iter().map(|plot| plot.id).max().unwrap_or(0) + 1;
        let series_meta = if persisted_v5 {
            persisted
                .series_meta
                .clone()
                .into_iter()
                .map(|meta| (meta.key.clone(), meta))
                .collect::<BTreeMap<_, _>>()
        } else {
            BTreeMap::new()
        };
        let ha_poll_interval_secs = if persisted.ha_poll_interval_secs >= 1.0 {
            persisted.ha_poll_interval_secs.clamp(1.0, 86_400.0)
        } else {
            60.0
        };
        let ha_history_hours = if persisted.ha_history_hours > 0.0 {
            persisted.ha_history_hours.clamp(0.1, 24.0 * 365.0)
        } else {
            24.0
        };

        let (telegram_config, telegram_config_status) = load_telegram_config();
        let telegram_notifier = TelegramNotifier::spawn(&telegram_config);
        let initial_status = "Open an image from disk or enter a camera IP/address.".to_owned();
        let initial_status_time = SystemTime::now();
        let mut loaded_status_history = persisted
            .status_history
            .iter()
            .filter_map(|entry| {
                UNIX_EPOCH
                    .checked_add(Duration::from_millis(entry.timestamp_ms))
                    .map(|time| StatusEvent {
                        time,
                        text: entry.text.clone(),
                        is_error: entry.is_error,
                    })
            })
            .collect::<VecDeque<_>>();
        while loaded_status_history.len() >= MAX_STATUS_HISTORY {
            loaded_status_history.pop_front();
        }
        loaded_status_history.push_back(StatusEvent {
            time: initial_status_time,
            text: initial_status.clone(),
            is_error: false,
        });

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
            status: initial_status.clone(),
            error: None,
            status_changed_at: initial_status_time,
            status_history: loaded_status_history,
            last_status_signature: format!("status:{initial_status}"),
            original_view,
            processed_view,
            current_final_mask: None,
            active_processing_sequence_frame: None,
            sequence_mask_cache: BTreeMap::new(),
            sequence_raw_mask_cache: BTreeMap::new(),
            sequence_mask_cache_order: VecDeque::new(),
            temporal_filter_enabled: if persisted_v2 { persisted.temporal_filter_enabled } else { false },
            temporal_lookahead_frames: temporal_window_frames,
            temporal_required_frames,
            temporal_overlap_threshold: if persisted_v2 { persisted.temporal_overlap_threshold.clamp(0.0, 1.0) } else { 0.35 },
            track_overlap_threshold: if persisted_v2 { persisted.track_overlap_threshold.clamp(0.0, 1.0) } else { 0.30 },
            shape_tracks: Vec::new(),
            next_track_id: 0,
            auto_seed_pivots_pending: false,
            shape_plot_open: if persisted_v2 { persisted.shape_plot_open } else { false },
            plot_click_seek_enabled: if persisted_v3 { persisted.plot_click_seek_enabled } else { true },
            plot_x_center: if persisted_v4 { persisted.plot_x_center.clamp(0.0, 1.0) } else { 0.5 },
            plot_x_zoom: if persisted_v4 { persisted.plot_x_zoom.clamp(1.0, 10_000.0) } else { 1.0 },
            plot_y_center: if persisted_v4 { persisted.plot_y_center.clamp(0.0, 1.0) } else { 0.5 },
            plot_y_zoom: if persisted_v4 { persisted.plot_y_zoom.clamp(1.0, 10_000.0) } else { 1.0 },
            sequence_playing: false,
            sequence_playback_fps: if persisted_v2 { persisted.sequence_playback_fps.clamp(0.1, 120.0) } else { 5.0 },
            sequence_loop: if persisted_v2 { persisted.sequence_loop } else { true },
            sequence_wait_processing: if persisted_v2 { persisted.sequence_wait_processing } else { true },
            next_sequence_frame_due: None,
            sequence_glue_inputs: if persisted_v2 { persisted.sequence_glue_inputs.clone() } else { Vec::new() },
            series_meta,
            plot_windows,
            next_plot_window_id,
            next_selection_group_id: 1,
            auto_track_highwater_frame: None,
            ha_worker: HaWorker::spawn(cc.egui_ctx.clone()),
            ha_base_url_input: if persisted_v5 { persisted.ha_base_url.clone() } else { String::new() },
            ha_token_input: if persisted_v5 && persisted.ha_remember_token { persisted.ha_token.clone() } else { String::new() },
            ha_remember_token: persisted_v5 && persisted.ha_remember_token,
            ha_auto_poll: persisted_v5 && persisted.ha_auto_poll,
            ha_poll_interval_secs,
            ha_history_hours,
            ha_entity_id_input: String::new(),
            ha_attribute_input: String::new(),
            ha_series: if persisted_v5 { persisted.ha_series.clone() } else { Vec::new() },
            ha_data: BTreeMap::new(),
            ha_request_id: 0,
            ha_request_pending: false,
            next_ha_poll: None,
            ha_status: "Home Assistant idle".to_owned(),
            telegram_config,
            telegram_notifier,
            telegram_config_status,
            capture_retry_due: None,
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

        app.sync_home_assistant_series_metadata();
        app.next_selection_group_id = app
            .series_meta
            .values()
            .filter_map(|meta| meta.group.strip_prefix("selection-"))
            .filter_map(|suffix| suffix.parse::<u64>().ok())
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        if app.ha_auto_poll {
            app.next_ha_poll = Some(Instant::now());
        }

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

    fn sync_status_history(&mut self) {
        let (is_error, text) = match self.error.as_ref() {
            Some(error) => (true, format!("{} — {error}", self.status)),
            None => (false, self.status.clone()),
        };
        let signature = format!("{}:{text}", if is_error { "error" } else { "status" });
        if signature == self.last_status_signature {
            return;
        }
        let now = SystemTime::now();
        self.status_changed_at = now;
        self.last_status_signature = signature;
        self.status_history.push_back(StatusEvent { time: now, text, is_error });
        while self.status_history.len() > MAX_STATUS_HISTORY {
            self.status_history.pop_front();
        }
        self.preferences_dirty = true;
        self.next_preferences_save = Instant::now() + Duration::from_secs(1);
    }

    fn notify_telegram(&self, text: impl Into<String>) {
        self.telegram_notifier.send(text);
    }

    fn capture_retry_cooldown(&self) -> Duration {
        Duration::from_secs_f32(self.telegram_config.retry_cooldown_seconds.clamp(0.1, 3600.0))
    }

    fn abandon_failed_capture_slot(&mut self, reason: &str) {
        let retry_count = self.pending_capture.as_ref().map(|pending| pending.retry_count).unwrap_or(0);
        self.pending_capture = None;
        self.capture_retry_due = None;
        self.source_loading = false;
        let message = format!("Capture slot abandoned after {retry_count} retries: {reason}");
        self.status = message.clone();
        self.error = Some(message);
        if let Some(path) = self.deferred_sequence_frame.take() {
            self.queue_source(SourceRequest::SequenceFrame(path));
        }
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
                let message = format!(
                    "Sequence frame disappeared before it could be opened: {}. The sequence was pruned to existing files.",
                    path.display()
                );
                self.notify_telegram(format!(
                    "⚠️ rust-edge-gui image load failed at {}
{message}",
                    format_status_time(SystemTime::now())
                ));
                self.error = Some(message);
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
            self.capture_retry_due = None;
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
                let message = format!("Image loader stopped: {error}");
                self.notify_telegram(format!(
                    "⚠️ rust-edge-gui image loader failed at {}
{message}",
                    format_status_time(SystemTime::now())
                ));
                self.error = Some(message);
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
                    let recovered_after_retries = self
                        .pending_capture
                        .as_ref()
                        .filter(|pending| pending.source_id == id)
                        .map(|pending| pending.retry_count)
                        .unwrap_or(0);
                    if is_pending_capture && self.continuous_capture {
                        self.capture_url = Some(image.label.clone());
                        self.capture_retry_due = None;
                        if recovered_after_retries > 0 {
                            self.notify_telegram(format!(
                                "✅ rust-edge-gui capture recovered after {recovered_after_retries} retr{} at {}",
                                if recovered_after_retries == 1 { "y" } else { "ies" },
                                format_status_time(SystemTime::now())
                            ));
                        }
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
                    let is_pending_capture = self
                        .pending_capture
                        .as_ref()
                        .is_some_and(|pending| pending.source_id == id);

                    if is_pending_capture && self.continuous_capture {
                        let now = Instant::now();
                        let cooldown = self.capture_retry_cooldown();
                        let retry_at = now + cooldown;
                        let next_frame_due = self.next_capture_due.unwrap_or(retry_at + Duration::from_secs(1));
                        let can_retry = retry_at < next_frame_due;

                        let should_notify = self
                            .pending_capture
                            .as_ref()
                            .is_some_and(|pending| !pending.failure_notified);
                        if should_notify {
                            let file_name = self
                                .pending_capture
                                .as_ref()
                                .map(|pending| pending.file_name.clone())
                                .unwrap_or_else(|| "capture frame".to_owned());
                            let retry_note = if can_retry {
                                format!(
                                    "Retrying every {:.1}s until the next scheduled frame.",
                                    self.telegram_config.retry_cooldown_seconds
                                )
                            } else {
                                "No retry fits before the next scheduled frame; this slot will be skipped.".to_owned()
                            };
                            self.notify_telegram(format!(
                                "⚠️ rust-edge-gui failed to load {file_name} at {}
{error}
{retry_note}",
                                format_status_time(SystemTime::now())
                            ));
                            if let Some(pending) = self.pending_capture.as_mut() {
                                pending.failure_notified = true;
                            }
                        }

                        self.error = Some(error.clone());
                        if can_retry {
                            self.capture_retry_due = Some(retry_at);
                            self.status = format!(
                                "Capture load failed; retry in {:.1}s (before next frame)",
                                cooldown.as_secs_f32()
                            );
                        } else {
                            self.abandon_failed_capture_slot(&error);
                        }
                    } else {
                        self.notify_telegram(format!(
                            "⚠️ rust-edge-gui image load failed at {}
{error}",
                            format_status_time(SystemTime::now())
                        ));
                        self.error = Some(error);
                        self.status = "Load failed".to_owned();
                        if let Some(path) = self.deferred_sequence_frame.take() {
                            self.queue_source(SourceRequest::SequenceFrame(path));
                        }
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
            version: 6,
            source_history: self.source_history.clone(),
            status_history: self
                .status_history
                .iter()
                .map(|entry| PersistedStatusEvent {
                    timestamp_ms: entry
                        .time
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis()
                        .min(u64::MAX as u128) as u64,
                    text: entry.text.clone(),
                    is_error: entry.is_error,
                })
                .collect(),
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
            plot_click_seek_enabled: self.plot_click_seek_enabled,
            plot_x_center: self.plot_x_center,
            plot_x_zoom: self.plot_x_zoom,
            plot_y_center: self.plot_y_center,
            plot_y_zoom: self.plot_y_zoom,
            url_input: self.url_input.clone(),
            image_history_input: self.image_history_input.clone(),
            sequence_history_input: self.sequence_history_input.clone(),
            sequence_glue_inputs: self.sequence_glue_inputs.clone(),
            plot_windows: self.plot_windows.iter().map(PlotWindowState::persisted).collect(),
            series_meta: self.series_meta.values().cloned().collect(),
            ha_base_url: self.ha_base_url_input.clone(),
            ha_token: if self.ha_remember_token { self.ha_token_input.clone() } else { String::new() },
            ha_remember_token: self.ha_remember_token,
            ha_auto_poll: self.ha_auto_poll,
            ha_poll_interval_secs: self.ha_poll_interval_secs,
            ha_history_hours: self.ha_history_hours,
            ha_series: self.ha_series.clone(),
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

    fn default_plot_id(&self) -> u64 {
        self.plot_windows.first().map(|plot| plot.id).unwrap_or(1)
    }

    fn spawn_plot_window(&mut self) -> u64 {
        let id = self.next_plot_window_id.max(1);
        self.next_plot_window_id = id.saturating_add(1);
        let cascade = ((self.plot_windows.len() % 8) as f32) * 28.0;
        self.plot_windows.push(PlotWindowState {
            id,
            title: format!("Plot {id}"),
            open: true,
            pos: egui::pos2(720.0 + cascade, 80.0 + cascade),
            size: egui::vec2(680.0, 360.0),
            x_center: 0.5,
            x_zoom: 1.0,
            y_center: 0.5,
            y_zoom: 1.0,
        });
        self.save_preferences();
        id
    }

    fn sync_shape_series_metadata(&mut self) {
        let width = self
            .original_rgba
            .as_ref()
            .map(|image| image.width() as usize)
            .unwrap_or(0);
        let groups = build_shape_group_series(&self.shape_tracks, width);
        let active_keys = groups
            .iter()
            .map(|group| shape_series_key(group.id))
            .collect::<BTreeSet<_>>();

        self.series_meta.retain(|key, _| {
            !key.starts_with("shape:") || active_keys.contains(key)
        });
        let default_plot = self.default_plot_id();
        for group in groups {
            let key = shape_series_key(group.id);
            self.series_meta.entry(key.clone()).or_insert_with(|| SeriesMeta {
                key,
                name: group.name.clone(),
                group: "Plant shapes".to_owned(),
                plot_id: default_plot,
                visible: true,
            });
        }
    }

    fn sync_home_assistant_series_metadata(&mut self) {
        let default_plot = self.default_plot_id();
        for config in &self.ha_series {
            self.series_meta.entry(config.key.clone()).or_insert_with(|| SeriesMeta {
                key: config.key.clone(),
                name: config.entity_id.clone(),
                group: "Home Assistant".to_owned(),
                plot_id: default_plot,
                visible: true,
            });
            self.ha_data.entry(config.key.clone()).or_insert_with(|| HaSeriesData {
                unit: String::new(),
                friendly_name: config.entity_id.clone(),
                points: BTreeMap::new(),
            });
        }
        let configured = self.ha_series.iter().map(|s| s.key.clone()).collect::<BTreeSet<_>>();
        self.ha_data.retain(|key, _| configured.contains(key));
        self.series_meta.retain(|key, _| !key.starts_with("ha:") || configured.contains(key));
    }

    fn group_selected_shape_series(&mut self, group_ids: &[u64]) {
        if group_ids.is_empty() {
            return;
        }
        self.sync_shape_series_metadata();
        let group_name = format!("selection-{}", self.next_selection_group_id);
        self.next_selection_group_id = self.next_selection_group_id.saturating_add(1);
        let mut changed = 0usize;
        for group_id in group_ids {
            if let Some(meta) = self.series_meta.get_mut(&shape_series_key(*group_id)) {
                meta.group = group_name.clone();
                changed += 1;
            }
        }
        if changed > 0 {
            self.status = format!("Grouped {changed} selected shape timeseries as {group_name}.");
            self.save_preferences();
        }
    }

    fn add_home_assistant_series(&mut self) {
        let entity_id = self.ha_entity_id_input.trim().to_owned();
        if entity_id.is_empty() {
            self.error = Some("Enter a Home Assistant entity_id first.".to_owned());
            return;
        }
        let attribute = self.ha_attribute_input.trim().to_owned();
        let key = ha_series_key(&entity_id, &attribute);
        if self.ha_series.iter().any(|config| config.key == key) {
            self.status = format!("{entity_id} is already configured.");
            return;
        }
        self.ha_series.push(HaSeriesConfig {
            key: key.clone(),
            entity_id: entity_id.clone(),
            attribute,
        });
        self.sync_home_assistant_series_metadata();
        self.ha_entity_id_input.clear();
        self.ha_attribute_input.clear();
        self.status = format!("Added Home Assistant series {entity_id}.");
        self.save_preferences();
    }

    fn queue_home_assistant_request(&mut self, kind: HaRequestKind) {
        if self.ha_request_pending {
            return;
        }
        if self.ha_series.is_empty() {
            self.ha_status = "Add at least one Home Assistant sensor first.".to_owned();
            return;
        }
        if self.ha_base_url_input.trim().is_empty() || self.ha_token_input.trim().is_empty() {
            self.ha_status = "Home Assistant URL/token is missing.".to_owned();
            return;
        }
        self.ha_request_id = self.ha_request_id.wrapping_add(1).max(1);
        let request = HaRequest {
            id: self.ha_request_id,
            base_url: self.ha_base_url_input.trim().to_owned(),
            token: self.ha_token_input.clone(),
            series: self
                .ha_series
                .iter()
                .map(|config| HaSeriesRequest {
                    key: config.key.clone(),
                    entity_id: config.entity_id.clone(),
                    attribute: config.attribute.clone(),
                })
                .collect(),
            kind,
        };
        match self.ha_worker.request_tx.send(request) {
            Ok(()) => {
                self.ha_request_pending = true;
                self.ha_status = "Home Assistant request in progress…".to_owned();
            }
            Err(error) => {
                self.ha_status = format!("Home Assistant worker unavailable: {error}");
            }
        }
    }

    fn poll_home_assistant_worker(&mut self) {
        while let Ok(message) = self.ha_worker.message_rx.try_recv() {
            match message {
                HaMessage::Finished { id, samples } if id == self.ha_request_id => {
                    self.ha_request_pending = false;
                    let sample_count = samples.len();
                    for sample in samples {
                        let data = self.ha_data.entry(sample.key.clone()).or_insert_with(|| HaSeriesData {
                            unit: sample.unit.clone(),
                            friendly_name: sample.friendly_name.clone(),
                            points: BTreeMap::new(),
                        });
                        if !sample.unit.is_empty() {
                            data.unit = sample.unit.clone();
                        }
                        if !sample.friendly_name.is_empty() {
                            data.friendly_name = sample.friendly_name.clone();
                        }
                        data.points.insert(sample.timestamp, sample.value);
                        while data.points.len() > MAX_EXTERNAL_SERIES_POINTS {
                            let Some(oldest) = data.points.keys().next().copied() else { break; };
                            data.points.remove(&oldest);
                        }
                        if let Some(meta) = self.series_meta.get_mut(&sample.key) {
                            let config_name = self
                                .ha_series
                                .iter()
                                .find(|config| config.key == sample.key)
                                .map(|config| config.entity_id.as_str())
                                .unwrap_or_default();
                            if meta.name == config_name && !sample.friendly_name.is_empty() {
                                meta.name = sample.friendly_name;
                            }
                        }
                    }
                    self.ha_status = format!("Home Assistant: received {sample_count} sample(s).");
                }
                HaMessage::Failed { id, error } if id == self.ha_request_id || id == 0 => {
                    self.ha_request_pending = false;
                    self.ha_status = format!("Home Assistant error: {error}");
                }
                _ => {}
            }
        }
    }

    fn tick_home_assistant_poll(&mut self, ctx: &egui::Context) {
        if !self.ha_auto_poll {
            self.next_ha_poll = None;
            return;
        }
        let now = Instant::now();
        let due = self.next_ha_poll.unwrap_or(now);
        if now >= due && !self.ha_request_pending {
            self.queue_home_assistant_request(HaRequestKind::Snapshot);
            self.next_ha_poll = Some(now + Duration::from_secs_f32(self.ha_poll_interval_secs.max(1.0)));
        }
        if let Some(next) = self.next_ha_poll {
            ctx.request_repaint_after(next.saturating_duration_since(now).min(Duration::from_secs(1)));
        }
    }

    fn collect_plot_series_for_plot(&self, plot_id: u64) -> Vec<PlotSeriesData> {
        let width = self
            .original_rgba
            .as_ref()
            .map(|image| image.width() as usize)
            .unwrap_or(0);
        let shape_groups = build_shape_group_series(&self.shape_tracks, width)
            .into_iter()
            .map(|group| (group.id, group))
            .collect::<BTreeMap<_, _>>();
        let mut result = Vec::new();
        for meta in self.series_meta.values().filter(|meta| meta.visible && meta.plot_id == plot_id) {
            if let Some(group_id) = parse_shape_series_key(&meta.key) {
                let Some(group) = shape_groups.get(&group_id) else { continue; };
                let Some(sequence) = self.active_sequence.as_ref() else { continue; };
                let points = group
                    .areas
                    .iter()
                    .filter_map(|(&frame, &area)| {
                        let time = sequence.frames.get(frame)?.timestamp;
                        Some(PlotPoint { time, value: area as f64, frame: Some(frame) })
                    })
                    .collect::<Vec<_>>();
                if !points.is_empty() {
                    result.push(PlotSeriesData {
                        key: meta.key.clone(),
                        name: meta.name.clone(),
                        group: meta.group.clone(),
                        unit: "px".to_owned(),
                        color_seed: stable_series_seed(&meta.key),
                        points,
                    });
                }
            } else if meta.key.starts_with("ha:") {
                let Some(data) = self.ha_data.get(&meta.key) else { continue; };
                let points = data
                    .points
                    .iter()
                    .map(|(&time, &value)| PlotPoint { time, value, frame: None })
                    .collect::<Vec<_>>();
                if !points.is_empty() {
                    result.push(PlotSeriesData {
                        key: meta.key.clone(),
                        name: meta.name.clone(),
                        group: meta.group.clone(),
                        unit: data.unit.clone(),
                        color_seed: stable_series_seed(&meta.key),
                        points,
                    });
                }
            }
        }
        result
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
                self.sequence_raw_mask_cache.clear();
                self.sequence_mask_cache_order.clear();
                self.shape_tracks.clear();
                self.auto_track_highwater_frame = None;
                self.auto_seed_pivots_pending = true;
                self.shape_plot_open = false;
                self.sync_shape_series_metadata();
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

    fn continue_continuous_capture_in_active_sequence(&mut self) {
        if !self.capture_save_original && !self.capture_save_processed {
            self.error = Some("Enable original and/or processed capture saving first.".to_owned());
            return;
        }
        let Some((root, frames)) = self
            .active_sequence
            .as_ref()
            .map(|sequence| (sequence.root.clone(), sequence.frames.clone()))
        else {
            self.error = Some("Open an existing image sequence first.".to_owned());
            return;
        };

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
            self.error = Some(format!("Failed to continue capture: {error:#}"));
            return;
        }

        let mut max_index = 0u64;
        for frame in &frames {
            let Some(name) = frame.path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(rest) = name.strip_prefix("frame_") else {
                continue;
            };
            let digits = rest.split('_').next().unwrap_or_default();
            if let Ok(index) = digits.parse::<u64>() {
                max_index = max_index.max(index);
            }
        }
        let next_frame_index = max_index
            .max(frames.len() as u64)
            .saturating_add(1)
            .max(1);

        self.capture_session = Some(CaptureSession {
            root: root.clone(),
            original_dir,
            processed_dir,
            next_frame_index,
        });
        self.capture_url = Some(url);
        self.pending_capture = None;
        self.processing_capture = None;
        self.deferred_sequence_frame = None;
        self.capture_retry_due = None;
        self.next_capture_due = Some(Instant::now());
        self.continuous_capture = true;
        self.sequence_playing = false;
        self.next_sequence_frame_due = None;
        // Existing pivots/groups and cached history stay intact. If this sequence
        // somehow has no pivots yet, the next frame-1 analysis can still seed them.
        if self.shape_tracks.is_empty() && frames.is_empty() {
            self.auto_seed_pivots_pending = true;
        }
        self.remember_sequence(root.display().to_string());
        self.save_preferences();
        self.status = format!(
            "Continuing capture in existing sequence: every {:.3}s → {} (next frame {:06})",
            self.capture_interval_secs,
            root.display(),
            next_frame_index
        );
        self.error = None;
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
        self.sequence_raw_mask_cache.clear();
        self.sequence_mask_cache_order.clear();
        self.shape_tracks.clear();
        self.auto_track_highwater_frame = None;
        self.auto_seed_pivots_pending = true;
        self.sync_shape_series_metadata();
        self.capture_url = Some(url);
        self.pending_capture = None;
        self.processing_capture = None;
        self.deferred_sequence_frame = None;
        self.capture_retry_due = None;
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
        self.capture_retry_due = None;
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

        // A failed scheduled frame may retry during its own time slot, but never
        // blocks the next scheduled frame indefinitely.
        if self.pending_capture.is_some() {
            if now >= due && !self.source_loading {
                self.abandon_failed_capture_slot("next scheduled frame is due");
            } else if self
                .capture_retry_due
                .is_some_and(|retry_due| now >= retry_due)
                && !self.source_loading
            {
                if let Some(url) = self.capture_url.clone() {
                    self.capture_retry_due = None;
                    self.queue_source(SourceRequest::CaptureUrl(url));
                    let new_source_id = self.active_source_id;
                    if let Some(pending) = self.pending_capture.as_mut() {
                        pending.source_id = new_source_id;
                        pending.retry_count = pending.retry_count.saturating_add(1);
                    }
                }
            }
        }

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
                retry_count: 0,
                failure_notified: false,
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

        let frame_delay = match self.next_capture_due {
            Some(next) if next > now => next.duration_since(now),
            _ => Duration::from_millis(100),
        };
        let retry_delay = match self.capture_retry_due {
            Some(next) if next > now => next.duration_since(now),
            Some(_) => Duration::from_millis(10),
            None => Duration::from_secs(1),
        };
        ctx.request_repaint_after(frame_delay.min(retry_delay).min(Duration::from_secs(1)));
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
                        self.cache_sequence_masks(
                            frame_index,
                            result.width,
                            result.height,
                            Arc::new(result.raw_shape_mask.clone()),
                            Arc::clone(&frame_mask),
                        );
                        self.refresh_temporal_cache_around(frame_index);
                        self.seed_all_shapes_from_mask(
                            frame_index,
                            result.width,
                            result.height,
                            frame_mask.as_ref(),
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
                if let Some(cached) = self.sequence_raw_mask_cache.get(&index) {
                    if expected_dims.is_none_or(|dims| dims == (cached.width, cached.height)) {
                        support.push(Arc::clone(&cached.mask));
                    }
                }
            }
            if let Some(index) = current.checked_add(offset).filter(|index| *index < sequence.frames.len()) {
                if let Some(cached) = self.sequence_raw_mask_cache.get(&index) {
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

    fn cache_sequence_masks(
        &mut self,
        frame_index: usize,
        width: usize,
        height: usize,
        raw_mask: Arc<Vec<bool>>,
        final_mask: Arc<Vec<bool>>,
    ) {
        self.sequence_raw_mask_cache.insert(
            frame_index,
            CachedSequenceMask {
                width,
                height,
                mask: raw_mask,
            },
        );
        self.sequence_mask_cache.insert(
            frame_index,
            CachedSequenceMask {
                width,
                height,
                mask: final_mask,
            },
        );
        self.sequence_mask_cache_order.retain(|index| *index != frame_index);
        self.sequence_mask_cache_order.push_back(frame_index);
        while self.sequence_mask_cache_order.len() > MAX_SEQUENCE_MASK_CACHE {
            if let Some(oldest) = self.sequence_mask_cache_order.pop_front() {
                self.sequence_mask_cache.remove(&oldest);
                self.sequence_raw_mask_cache.remove(&oldest);
            }
        }
    }

    fn refresh_temporal_cache_around(&mut self, newest_frame: usize) {
        if !self.temporal_filter_enabled || self.temporal_required_frames == 0 {
            return;
        }
        let radius = self.temporal_support_radius();
        let first = newest_frame.saturating_sub(radius);
        let last = newest_frame.saturating_add(radius);
        let affected = (first..=last)
            .filter(|frame_index| *frame_index != newest_frame)
            .filter(|frame_index| self.sequence_raw_mask_cache.contains_key(frame_index))
            .collect::<Vec<_>>();
        for frame_index in affected {
            let Some(cached) = self.sequence_raw_mask_cache.get(&frame_index) else { continue; };
            let width = cached.width;
            let height = cached.height;
            let base = Arc::clone(&cached.mask);
            let mut support = Vec::new();
            for offset in 1..=radius {
                if let Some(index) = frame_index.checked_sub(offset) {
                    if let Some(mask) = self.sequence_raw_mask_cache.get(&index) {
                        if mask.width == width && mask.height == height {
                            support.push(Arc::clone(&mask.mask));
                        }
                    }
                }
                if let Some(index) = frame_index.checked_add(offset) {
                    if let Some(mask) = self.sequence_raw_mask_cache.get(&index) {
                        if mask.width == width && mask.height == height {
                            support.push(Arc::clone(&mask.mask));
                        }
                    }
                }
            }
            if support.len() < self.temporal_required_frames {
                continue;
            }
            let (filtered, _) = filter_mask_by_temporal_support(
                base.as_ref(),
                width,
                height,
                &support,
                self.temporal_required_frames,
                self.temporal_overlap_threshold,
            );
            let unchanged = self
                .sequence_mask_cache
                .get(&frame_index)
                .is_some_and(|current| current.mask.as_ref() == &filtered);
            if unchanged {
                continue;
            }
            let filtered = Arc::new(filtered);
            if let Some(entry) = self.sequence_mask_cache.get_mut(&frame_index) {
                entry.mask = Arc::clone(&filtered);
            }
            self.retire_auto_anchors_removed_by_filter(frame_index, width, height, filtered.as_ref());
            // Historical observations are recomputed, but lifecycle spawning is
            // intentionally reserved for the high-water/newest frame.
            let components = extract_mask_components(filtered.as_ref(), width, height);
            let threshold = self.track_overlap_threshold.clamp(0.0, 1.0);
            for track in &mut self.shape_tracks {
                if !track.enabled || track.anchor_width != width || track.anchor_height != height {
                    continue;
                }
                if frame_index != track.anchor_frame {
                    track.observations.remove(&frame_index);
                    track.matched_pixels.remove(&frame_index);
                    track.matched_centroids.remove(&frame_index);
                    update_one_shape_track(track, frame_index, width, &components, threshold);
                }
            }
            self.merge_colliding_shape_groups(frame_index);
        }
        self.sync_shape_series_metadata();
    }

    fn retire_auto_anchors_removed_by_filter(
        &mut self,
        frame_index: usize,
        width: usize,
        height: usize,
        mask: &[bool],
    ) {
        let threshold = self.track_overlap_threshold.clamp(0.0, 1.0);
        let components = extract_mask_components(mask, width, height);
        let invalid = self
            .shape_tracks
            .iter()
            .filter(|track| {
                track.auto_created
                    && track.anchor_frame == frame_index
                    && track.anchor_width == width
                    && track.anchor_height == height
            })
            .filter(|track| {
                !components.iter().any(|component| {
                    component_overlap_score(&track.anchor_pixels, &component.pixels) + f32::EPSILON >= threshold
                })
            })
            .map(|track| track.id)
            .collect::<BTreeSet<_>>();
        if !invalid.is_empty() {
            self.shape_tracks.retain(|track| !invalid.contains(&track.id));
        }
    }

    fn update_shape_tracks_for_frame(
        &mut self,
        frame_index: usize,
        width: usize,
        height: usize,
        mask: &[bool],
    ) {
        if mask.len() != width.saturating_mul(height) {
            return;
        }
        let components = extract_mask_components(mask, width, height);
        let threshold = self.track_overlap_threshold.clamp(0.0, 1.0);
        for track in &mut self.shape_tracks {
            if !track.enabled || track.anchor_width != width || track.anchor_height != height {
                continue;
            }
            // Re-processing a frame after temporal support changes must not leave a
            // stale match behind. Auto anchors are validated separately below.
            if frame_index != track.anchor_frame {
                track.observations.remove(&frame_index);
                track.matched_pixels.remove(&frame_index);
                track.matched_centroids.remove(&frame_index);
            }
            update_one_shape_track(track, frame_index, width, &components, threshold);
        }
        self.merge_colliding_shape_groups(frame_index);
        self.reconcile_auto_shape_tracks(frame_index, width, height, &components);
        self.sync_shape_series_metadata();
    }

    fn reconcile_auto_shape_tracks(
        &mut self,
        frame_index: usize,
        width: usize,
        height: usize,
        components: &[MaskComponent],
    ) {
        // Lifecycle changes only move forward in time. Seeking backwards should
        // inspect history, not delete or spawn identities again.
        if self
            .auto_track_highwater_frame
            .is_some_and(|highwater| frame_index < highwater)
        {
            return;
        }
        self.auto_track_highwater_frame = Some(frame_index);
        let threshold = self.track_overlap_threshold.clamp(0.0, 1.0);
        let closed = components
            .iter()
            .filter(|component| component_is_closed(component, width, height))
            .collect::<Vec<_>>();

        // If temporal re-filtering removes the very component that created an
        // automatic pivot, retire that false-positive identity immediately.
        let mut invalid_auto_ids = BTreeSet::new();
        for track in self.shape_tracks.iter().filter(|track| {
            track.enabled
                && track.auto_created
                && track.anchor_frame == frame_index
                && track.anchor_width == width
                && track.anchor_height == height
        }) {
            let still_exists = closed.iter().any(|component| {
                component_overlap_score(&track.anchor_pixels, &component.pixels) + f32::EPSILON
                    >= threshold
            });
            if !still_exists {
                invalid_auto_ids.insert(track.id);
            }
        }
        if !invalid_auto_ids.is_empty() {
            self.shape_tracks
                .retain(|track| !invalid_auto_ids.contains(&track.id));
        }

        // Once a whole automatically-created logical group is absent for a
        // temporal-support-sized grace period, remove its pivots. Manual P tracks
        // deliberately survive until explicitly removed.
        let grace = if self.temporal_filter_enabled {
            self.temporal_support_radius().max(1)
        } else {
            2
        };
        let mut group_members = BTreeMap::<u64, Vec<usize>>::new();
        for (index, track) in self.shape_tracks.iter().enumerate() {
            group_members.entry(track.group_id).or_default().push(index);
        }
        let mut retire_groups = BTreeSet::new();
        for (group_id, members) in group_members {
            if members.iter().any(|&index| !self.shape_tracks[index].auto_created) {
                continue;
            }
            let last_seen = members
                .iter()
                .flat_map(|&index| self.shape_tracks[index].matched_pixels.keys().copied())
                .max();
            if let Some(last_seen) = last_seen {
                if frame_index > last_seen.saturating_add(grace) {
                    retire_groups.insert(group_id);
                }
            }
        }
        if !retire_groups.is_empty() {
            self.shape_tracks
                .retain(|track| !retire_groups.contains(&track.group_id));
        }

        // Every closed component that is not already represented by an existing
        // track becomes a new automatic pivot. This is what makes new sprouts
        // appear in the dashboard without manual P presses.
        let represented = self
            .shape_tracks
            .iter()
            .filter(|track| track.enabled && track.anchor_width == width && track.anchor_height == height)
            .filter_map(|track| track.matched_pixels.get(&frame_index))
            .cloned()
            .collect::<Vec<_>>();
        let mut created_ids = Vec::new();
        for component in closed {
            let already_tracked = represented.iter().any(|pixels| {
                component_overlap_score(pixels, &component.pixels) + f32::EPSILON >= threshold
            }) || self.shape_tracks.iter().any(|track| {
                track.enabled
                    && track.anchor_width == width
                    && track.anchor_height == height
                    && component_overlap_score(&track.anchor_pixels, &component.pixels) + f32::EPSILON >= threshold
                    && frame_index.abs_diff(track.anchor_frame) <= grace
            });
            if already_tracked {
                continue;
            }
            self.next_track_id = self.next_track_id.wrapping_add(1).max(1);
            let track_id = self.next_track_id;
            let pivot = component_centroid(component, width);
            let anchor_pixels = component.pixels.clone();
            let mut observations = BTreeMap::new();
            observations.insert(
                frame_index,
                ShapeObservation { area: component.area, overlap: 1.0 },
            );
            let mut matched_pixels = BTreeMap::new();
            matched_pixels.insert(frame_index, anchor_pixels.clone());
            let mut matched_centroids = BTreeMap::new();
            matched_centroids.insert(frame_index, pivot);
            self.shape_tracks.push(ShapeTrack {
                id: track_id,
                group_id: track_id,
                name: format!("shape-{track_id}"),
                anchor_pivot: pivot,
                anchor_frame: frame_index,
                anchor_width: width,
                anchor_height: height,
                anchor_pixels,
                matched_pixels,
                matched_centroids,
                observations,
                enabled: true,
                auto_created: true,
            });
            created_ids.push(track_id);
        }
        if !created_ids.is_empty() {
            self.shape_plot_open = true;
            if let Some(plot) = self.plot_windows.first_mut() {
                plot.open = true;
            }
        }
        for track_id in created_ids {
            self.rebuild_shape_track_from_cached_masks(track_id);
        }
        self.merge_colliding_shape_groups(frame_index);
    }

    /// Permanently merge track groups when two previously independent shapes
    /// resolve to the same connected component in a frame. Group membership is
    /// then used retroactively by overlays and the plot, so separated components
    /// before/after the collision are still treated as one logical shape.
    fn merge_colliding_shape_groups(&mut self, frame_index: usize) {
        let mut merges = Vec::<(usize, usize)>::new();
        for a in 0..self.shape_tracks.len() {
            if !self.shape_tracks[a].enabled {
                continue;
            }
            let Some(a_pixels) = self.shape_tracks[a].matched_pixels.get(&frame_index) else {
                continue;
            };
            for b in (a + 1)..self.shape_tracks.len() {
                if !self.shape_tracks[b].enabled {
                    continue;
                }
                if self.shape_tracks[a].group_id == self.shape_tracks[b].group_id {
                    continue;
                }
                let Some(b_pixels) = self.shape_tracks[b].matched_pixels.get(&frame_index) else {
                    continue;
                };
                // If both tracks selected exactly the same connected component,
                // the physical shapes have collided/merged in this frame.
                if a_pixels == b_pixels {
                    merges.push((a, b));
                }
            }
        }

        // Apply by track indices rather than stale group IDs so transitive
        // collisions A↔B and B↔C collapse to one canonical group in one pass.
        for (a, b) in merges {
            let ga = self.shape_tracks[a].group_id;
            let gb = self.shape_tracks[b].group_id;
            if ga == gb {
                continue;
            }
            let target = ga.min(gb);
            for track in &mut self.shape_tracks {
                if track.group_id == ga || track.group_id == gb {
                    track.group_id = target;
                }
            }
        }
    }

    fn seed_all_shapes_from_mask(
        &mut self,
        frame_index: usize,
        width: usize,
        height: usize,
        mask: &[bool],
    ) {
        if !self.auto_seed_pivots_pending || frame_index != 0 || !self.shape_tracks.is_empty() {
            return;
        }
        let components = extract_mask_components(mask, width, height)
            .into_iter()
            .filter(|component| component_is_closed(component, width, height))
            .collect::<Vec<_>>();
        if components.is_empty() {
            self.auto_seed_pivots_pending = false;
            return;
        }

        let mut created = Vec::with_capacity(components.len());
        for component in components {
            self.next_track_id = self.next_track_id.wrapping_add(1).max(1);
            let track_id = self.next_track_id;
            let pivot = component_centroid(&component, width);
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
            matched_centroids.insert(frame_index, pivot);
            self.shape_tracks.push(ShapeTrack {
                id: track_id,
                group_id: track_id,
                name: format!("shape-{track_id}"),
                anchor_pivot: pivot,
                anchor_frame: frame_index,
                anchor_width: width,
                anchor_height: height,
                anchor_pixels,
                matched_pixels,
                matched_centroids,
                observations,
                enabled: true,
                auto_created: true,
            });
            created.push(track_id);
        }
        self.auto_seed_pivots_pending = false;
        self.auto_track_highwater_frame = Some(frame_index);
        self.shape_plot_open = !created.is_empty();
        if !created.is_empty() {
            if let Some(plot) = self.plot_windows.first_mut() {
                plot.open = true;
            }
        }
        for track_id in created.iter().copied() {
            self.rebuild_shape_track_from_cached_masks(track_id);
        }
        let cached_frames = self.sequence_mask_cache.keys().copied().collect::<Vec<_>>();
        for frame in cached_frames {
            self.merge_colliding_shape_groups(frame);
        }
        self.sync_shape_series_metadata();
        self.status = format!(
            "Automatically added {} pivot(s) from closed shapes in sequence frame 1.",
            created.len()
        );
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
            group_id: track_id,
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
            auto_created: false,
        });
        self.rebuild_shape_track_from_cached_masks(track_id);
        let cached_frames = self.sequence_mask_cache.keys().copied().collect::<Vec<_>>();
        for frame in cached_frames {
            self.merge_colliding_shape_groups(frame);
        }
        self.shape_plot_open = true;
        if let Some(plot) = self.plot_windows.first_mut() {
            plot.open = true;
        }
        self.sync_shape_series_metadata();
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
        let width = self
            .original_rgba
            .as_ref()
            .map(|image| image.width() as usize)
            .unwrap_or(0);
        build_shape_group_series(&self.shape_tracks, width)
            .into_iter()
            .map(|group| {
                let frame_for_value = current_frame.unwrap_or(group.anchor_frame);
                let pixel = group
                    .centroids
                    .get(&frame_for_value)
                    .copied()
                    .or_else(|| group.centroids.get(&group.anchor_frame).copied())
                    .unwrap_or((0, 0));
                let area = if let Some(frame) = current_frame {
                    group.areas.get(&frame).copied().unwrap_or(0)
                } else {
                    group.areas.get(&group.anchor_frame).copied().unwrap_or(0)
                };
                let label = if group.member_count > 1 {
                    format!("G{}", group.id)
                } else if current_frame == Some(group.anchor_frame) {
                    format!("T{}*", group.id)
                } else {
                    format!("T{}", group.id)
                };
                let kind = if group.member_count > 1 {
                    format!("merged group · {} shapes", group.member_count)
                } else {
                    "tracked shape".to_owned()
                };
                let mut hover_text = format!(
                    "{} {}
area: {} px
frame: {}
pivot: {}, {}",
                    label,
                    kind,
                    area,
                    frame_for_value.saturating_add(1),
                    pixel.0,
                    pixel.1,
                );
                if let Some(timestamp) = self
                    .active_sequence
                    .as_ref()
                    .and_then(|sequence| sequence.frames.get(frame_for_value))
                    .map(|frame| frame.timestamp)
                {
                    hover_text.push_str(&format!("
time: {}", format_absolute_time(timestamp)));
                }
                viewer::OverlayPoint {
                    id: Some(group.id),
                    pixel,
                    color: shape_series_color(group.id),
                    label,
                    radius: pivot_radius_for_area(area),
                    hover_text,
                }
            })
            .collect()
    }

    fn delete_current_sequence_frame(&mut self) {
        if self.source_loading
            || self.processing
            || self.pending_capture.is_some()
            || self.processing_capture.is_some()
        {
            self.error = Some(
                "Wait for the current load/capture/processing job to finish before deleting a frame."
                    .to_owned(),
            );
            return;
        }

        let Some((root, frame_index, frame_path)) = self.active_sequence.as_ref().and_then(|sequence| {
            sequence
                .frames
                .get(sequence.selected)
                .map(|frame| (sequence.root.clone(), sequence.selected, frame.path.clone()))
        }) else {
            self.error = Some("There is no current sequence frame to delete.".to_owned());
            return;
        };

        self.sequence_playing = false;
        self.next_sequence_frame_due = None;

        if let Err(error) = std::fs::remove_file(&frame_path) {
            self.error = Some(format!(
                "Failed to delete current frame {}: {error}",
                frame_path.display()
            ));
            return;
        }

        // Captured sequences normally use the same filename in original/ and processed/.
        // Remove the counterpart as well so deleting one timeline frame cannot leave a stale
        // processed image behind. Missing counterparts are harmless.
        let mut cleanup_warnings = Vec::new();
        if let Some(file_name) = frame_path.file_name() {
            for candidate in [root.join("original").join(file_name), root.join("processed").join(file_name)] {
                if candidate != frame_path && candidate.is_file() {
                    if let Err(error) = std::fs::remove_file(&candidate) {
                        cleanup_warnings.push(format!("{}: {error}", candidate.display()));
                    }
                }
            }
        }

        let (next_path, manifest_error) = {
            let Some(sequence) = self.active_sequence.as_mut() else {
                return;
            };
            if frame_index >= sequence.frames.len() {
                self.error = Some("The selected frame changed before deletion completed.".to_owned());
                return;
            }
            sequence.frames.remove(frame_index);
            if sequence.frames.is_empty() {
                sequence.selected = 0;
            } else {
                sequence.selected = frame_index.min(sequence.frames.len() - 1);
            }
            let next_path = sequence
                .frames
                .get(sequence.selected)
                .map(|frame| frame.path.clone());
            let manifest_error = write_sequence_manifest(sequence).err();
            (next_path, manifest_error)
        };

        remove_and_shift_index_map(&mut self.sequence_mask_cache, frame_index);
        remove_and_shift_index_map(&mut self.sequence_raw_mask_cache, frame_index);
        let old_order = std::mem::take(&mut self.sequence_mask_cache_order);
        let mut new_order = VecDeque::new();
        for index in old_order {
            if index == frame_index {
                continue;
            }
            let shifted = if index > frame_index { index - 1 } else { index };
            if !new_order.contains(&shifted) {
                new_order.push_back(shifted);
            }
        }
        self.sequence_mask_cache_order = new_order;

        let frame_dimensions = self
            .sequence_mask_cache
            .iter()
            .map(|(&index, cached)| (index, (cached.width, cached.height)))
            .chain(
                self.sequence_raw_mask_cache
                    .iter()
                    .map(|(&index, cached)| (index, (cached.width, cached.height))),
            )
            .collect::<BTreeMap<_, _>>();

        let mut tracks_without_anchor = BTreeSet::new();
        for track in &mut self.shape_tracks {
            remove_and_shift_index_map(&mut track.matched_pixels, frame_index);
            remove_and_shift_index_map(&mut track.matched_centroids, frame_index);
            remove_and_shift_index_map(&mut track.observations, frame_index);

            if track.anchor_frame > frame_index {
                track.anchor_frame -= 1;
            } else if track.anchor_frame == frame_index {
                let replacement = track
                    .matched_pixels
                    .keys()
                    .copied()
                    .min_by_key(|candidate| candidate.abs_diff(frame_index))
                    .and_then(|new_anchor| {
                        let pixels = track.matched_pixels.get(&new_anchor)?.clone();
                        let centroid = track.matched_centroids.get(&new_anchor).copied()?;
                        Some((new_anchor, pixels, centroid))
                    });
                if let Some((new_anchor, pixels, centroid)) = replacement {
                    track.anchor_frame = new_anchor;
                    track.anchor_pixels = pixels;
                    track.anchor_pivot = centroid;
                    if let Some(&(width, height)) = frame_dimensions.get(&new_anchor) {
                        track.anchor_width = width;
                        track.anchor_height = height;
                    }
                } else {
                    tracks_without_anchor.insert(track.id);
                }
            }
        }
        if !tracks_without_anchor.is_empty() {
            self.shape_tracks
                .retain(|track| !tracks_without_anchor.contains(&track.id));
        }

        self.auto_track_highwater_frame = self.auto_track_highwater_frame.and_then(|index| {
            if index > frame_index {
                Some(index - 1)
            } else if index == frame_index {
                index.checked_sub(1)
            } else {
                Some(index)
            }
        });
        self.active_processing_sequence_frame = self.active_processing_sequence_frame.and_then(|index| {
            if index > frame_index {
                Some(index - 1)
            } else if index == frame_index {
                None
            } else {
                Some(index)
            }
        });

        self.current_final_mask = None;
        self.ai_mask = None;
        self.ai_mask_rgba = None;
        self.ai_mask_texture = None;
        self.sync_shape_series_metadata();
        self.auto_seed_pivots_pending = self
            .active_sequence
            .as_ref()
            .is_some_and(|sequence| !sequence.frames.is_empty() && self.shape_tracks.is_empty());

        let mut status = format!("Deleted frame {} from {}", frame_index + 1, root.display());
        if let Some(error) = manifest_error {
            status.push_str(&format!("; warning: failed to update sequence.json: {error:#}"));
        }
        if !cleanup_warnings.is_empty() {
            status.push_str(&format!("; counterpart cleanup warning: {}", cleanup_warnings.join(" | ")));
        }
        self.status = status;
        self.error = None;

        if let Some(path) = next_path {
            self.queue_source(SourceRequest::SequenceFrame(path));
        } else {
            self.original_rgba = None;
            self.original_gray = None;
            self.original_texture = None;
            self.processed_rgba = None;
            self.processed_texture = None;
            self.source_label = "Sequence is empty".to_owned();
        }
        self.save_preferences();
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

    fn seek_sequence_frame(&mut self, frame_index: usize) {
        let path = {
            let Some(sequence) = self.active_sequence.as_mut() else {
                return;
            };
            if sequence.frames.is_empty() {
                return;
            }
            let index = frame_index.min(sequence.frames.len().saturating_sub(1));
            sequence.selected = index;
            sequence.frames[index].path.clone()
        };
        self.sequence_playing = false;
        self.next_sequence_frame_due = None;
        self.queue_source(SourceRequest::SequenceFrame(path));
    }

    fn handle_global_shortcuts(&mut self, ctx: &egui::Context) {
        if self.active_sequence.is_none() || ctx.egui_wants_keyboard_input() {
            return;
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Space)) {
            self.sequence_playing = !self.sequence_playing;
            self.next_sequence_frame_due = Some(Instant::now());
            ctx.request_repaint();
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
        self.timeseries_controls(ui);
        self.home_assistant_controls(ui);

        ui.separator();
        ui.collapsing("Status", |ui| {
            let current_time = format_status_time(self.status_changed_at);
            if let Some(error) = self.error.as_ref() {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    format!("[{current_time}] {error}"),
                );
                if self.status != *error {
                    ui.monospace(format!("state: {}", self.status));
                }
            } else {
                ui.monospace(format!("[{current_time}] {}", self.status));
            }

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
            }

            ui.small(&self.telegram_config_status);
            ui.collapsing(format!("History ({})", self.status_history.len()), |ui| {
                if ui.small_button("Clear status history").clicked() {
                    self.status_history.clear();
                    self.preferences_dirty = true;
                }
                egui::ScrollArea::vertical()
                    .id_salt("status-history-scroll")
                    .max_height(260.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for entry in self.status_history.iter() {
                            let text = format!("[{}] {}", format_status_time(entry.time), entry.text);
                            if entry.is_error {
                                ui.colored_label(egui::Color32::LIGHT_RED, text);
                            } else {
                                ui.monospace(text);
                            }
                        }
                    });
            });
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
        ui.horizontal_wrapped(|ui| {
            if !self.continuous_capture {
                if ui.button("Start new capture sequence").clicked() {
                    self.start_continuous_capture();
                    ui.ctx().request_repaint();
                }
                if ui
                    .add_enabled(
                        self.active_sequence.is_some(),
                        egui::Button::new("Continue active sequence"),
                    )
                    .on_hover_text("Append new captured frames to the currently open sequence")
                    .clicked()
                {
                    self.continue_continuous_capture_in_active_sequence();
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
        ui.small("Start new creates a fresh capture folder. Continue active sequence appends new frames to the currently open sequence and preserves its history/tracking. Capture acquisition is independent from mask processing: raw originals keep the requested interval even if YOLO/temporal processing is slower.");

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
            self.sequence_raw_mask_cache.clear();
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
        let mut delete_current = false;

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
                let can_delete = self
                    .active_sequence
                    .as_ref()
                    .is_some_and(|sequence| !sequence.frames.is_empty())
                    && !self.source_loading
                    && !self.processing
                    && self.pending_capture.is_none()
                    && self.processing_capture.is_none();
                if ui
                    .add_enabled(can_delete, egui::Button::new("Delete current frame"))
                    .on_hover_text("Delete the current sequence frame from disk, its processed counterpart if present, and sequence.json")
                    .clicked()
                {
                    delete_current = true;
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
                ui.monospace("Space = play/pause");
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

        if delete_current {
            self.delete_current_sequence_frame();
        } else if step_delta != 0 {
            self.step_sequence(step_delta);
        } else if let Some(path) = selected_path {
            self.queue_source(SourceRequest::SequenceFrame(path));
        }
    }

    fn timeseries_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Timeseries / plot windows", |ui| {
            ui.small("Every tracked shape area and Home Assistant sensor is a timeseries. Rename it, assign a logical group, and move either one series or a whole group between plot windows.");
            ui.horizontal_wrapped(|ui| {
                if ui.button("New plot window").clicked() {
                    self.spawn_plot_window();
                }
                if ui.button("Open all plots").clicked() {
                    for plot in &mut self.plot_windows {
                        plot.open = true;
                    }
                }
                ui.small("Processed view: Shift+drag a rectangle around pivots to assign those shape timeseries to one new group.");
            });

            ui.separator();
            ui.label("Plot windows");
            let mut remove_plot = None;
            let can_remove_plot = self.plot_windows.len() > 1;
            for (index, plot) in self.plot_windows.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut plot.open, "");
                    ui.monospace(format!("P{}", plot.id));
                    ui.add(egui::TextEdit::singleline(&mut plot.title).desired_width(220.0));
                    if can_remove_plot && ui.small_button("×").clicked() {
                        remove_plot = Some(index);
                    }
                });
            }
            if let Some(index) = remove_plot {
                let removed_id = self.plot_windows[index].id;
                self.plot_windows.remove(index);
                let fallback = self.default_plot_id();
                for meta in self.series_meta.values_mut() {
                    if meta.plot_id == removed_id {
                        meta.plot_id = fallback;
                    }
                }
                self.save_preferences();
            }

            self.sync_shape_series_metadata();
            self.sync_home_assistant_series_metadata();
            let plot_options = self
                .plot_windows
                .iter()
                .map(|plot| (plot.id, plot.title.clone()))
                .collect::<Vec<_>>();

            if !self.series_meta.is_empty() {
                ui.separator();
                ui.label("Series");
                let keys = self.series_meta.keys().cloned().collect::<Vec<_>>();
                for key in keys {
                    let color = series_color(stable_series_seed(&key));
                    if let Some(meta) = self.series_meta.get_mut(&key) {
                        ui.horizontal_wrapped(|ui| {
                            ui.checkbox(&mut meta.visible, "");
                            ui.colored_label(color, "●");
                            ui.add(egui::TextEdit::singleline(&mut meta.name).desired_width(170.0));
                            ui.label("group");
                            ui.add(egui::TextEdit::singleline(&mut meta.group).desired_width(130.0));
                            let selected = plot_options
                                .iter()
                                .find(|(id, _)| *id == meta.plot_id)
                                .map(|(_, title)| title.as_str())
                                .unwrap_or("plot");
                            egui::ComboBox::from_id_salt(("series-plot", key.clone()))
                                .selected_text(selected)
                                .show_ui(ui, |ui| {
                                    for (plot_id, title) in &plot_options {
                                        ui.selectable_value(&mut meta.plot_id, *plot_id, title);
                                    }
                                });
                            ui.monospace(&key);
                        });
                    }
                }

                ui.separator();
                ui.label("Move whole group");
                let groups = self
                    .series_meta
                    .values()
                    .map(|meta| meta.group.clone())
                    .filter(|group| !group.trim().is_empty())
                    .collect::<BTreeSet<_>>();
                for group_name in groups {
                    let mut targets = self
                        .series_meta
                        .values()
                        .filter(|meta| meta.group == group_name)
                        .map(|meta| meta.plot_id);
                    let Some(first_target) = targets.next() else { continue; };
                    let rest = targets.collect::<Vec<_>>();
                    let member_count = rest.len() + 1;
                    let mixed = rest.iter().any(|target| *target != first_target);
                    let mut target = if mixed { 0 } else { first_target };
                    ui.horizontal(|ui| {
                        ui.monospace(format!("{} ({} series)", group_name, member_count));
                        let selected = if target == 0 {
                            "mixed"
                        } else {
                            plot_options
                                .iter()
                                .find(|(id, _)| *id == target)
                                .map(|(_, title)| title.as_str())
                                .unwrap_or("plot")
                        };
                        egui::ComboBox::from_id_salt(("group-plot", group_name.clone()))
                            .selected_text(selected)
                            .show_ui(ui, |ui| {
                                for (plot_id, title) in &plot_options {
                                    ui.selectable_value(&mut target, *plot_id, title);
                                }
                            });
                    });
                    if target != 0 && (mixed || target != first_target) {
                        for meta in self.series_meta.values_mut().filter(|meta| meta.group == group_name) {
                            meta.plot_id = target;
                        }
                    }
                }
            }
        });
    }

    fn home_assistant_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Home Assistant / FlowerCare timeseries", |ui| {
            ui.small("Uses Home Assistant REST API. FlowerCare usually exposes temperature, moisture, conductivity, illuminance and battery as separate sensor.* entities; add each entity here. Leave attribute blank to plot the entity state.");
            ui.label("Home Assistant URL");
            ui.add(
                egui::TextEdit::singleline(&mut self.ha_base_url_input)
                    .hint_text("http://homeassistant.local:8123")
                    .desired_width(430.0),
            );
            ui.label("Long-Lived Access Token");
            ui.add(
                egui::TextEdit::singleline(&mut self.ha_token_input)
                    .password(true)
                    .desired_width(430.0),
            );
            ui.checkbox(&mut self.ha_remember_token, "Remember token in local state.json");
            ui.horizontal_wrapped(|ui| {
                if ui.checkbox(&mut self.ha_auto_poll, "Auto poll").changed() {
                    self.next_ha_poll = self.ha_auto_poll.then(Instant::now);
                }
                ui.add(
                    egui::DragValue::new(&mut self.ha_poll_interval_secs)
                        .range(1.0..=86_400.0)
                        .speed(1.0)
                        .suffix(" s"),
                );
                if ui
                    .add_enabled(!self.ha_request_pending, egui::Button::new("Fetch now"))
                    .clicked()
                {
                    self.queue_home_assistant_request(HaRequestKind::Snapshot);
                }
                ui.label("Backfill");
                ui.add(
                    egui::DragValue::new(&mut self.ha_history_hours)
                        .range(0.1..=8760.0)
                        .speed(1.0)
                        .suffix(" h"),
                );
                if ui
                    .add_enabled(!self.ha_request_pending, egui::Button::new("Load history"))
                    .clicked()
                {
                    let end = SystemTime::now();
                    let start = end
                        .checked_sub(Duration::from_secs_f64((self.ha_history_hours.max(0.1) as f64) * 3600.0))
                        .unwrap_or(UNIX_EPOCH);
                    self.queue_home_assistant_request(HaRequestKind::History { start, end });
                }
            });
            ui.small(&self.ha_status);

            ui.separator();
            ui.label("Add sensor");
            ui.horizontal_wrapped(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.ha_entity_id_input)
                        .hint_text("sensor.flowercare_moisture")
                        .desired_width(270.0),
                );
                ui.add(
                    egui::TextEdit::singleline(&mut self.ha_attribute_input)
                        .hint_text("attribute (optional)")
                        .desired_width(180.0),
                );
                if ui.button("Add").clicked() {
                    self.add_home_assistant_series();
                }
            });

            let mut remove = None;
            for (index, config) in self.ha_series.iter().enumerate() {
                let sample_info = self.ha_data.get(&config.key).map(|data| {
                    if data.unit.is_empty() {
                        format!("{} samples", data.points.len())
                    } else {
                        format!("{} samples · {}", data.points.len(), data.unit)
                    }
                }).unwrap_or_else(|| "0 samples".to_owned());
                ui.horizontal_wrapped(|ui| {
                    ui.monospace(&config.entity_id);
                    if !config.attribute.trim().is_empty() {
                        ui.monospace(format!(".{}", config.attribute));
                    }
                    ui.small(sample_info);
                    if ui.small_button("×").clicked() {
                        remove = Some(index);
                    }
                });
            }
            if let Some(index) = remove {
                let config = self.ha_series.remove(index);
                self.ha_data.remove(&config.key);
                self.series_meta.remove(&config.key);
                self.save_preferences();
            }
        });
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
                let cached_frames = self.sequence_mask_cache.keys().copied().collect::<Vec<_>>();
                for frame in cached_frames {
                    self.merge_colliding_shape_groups(frame);
                }
                self.sync_shape_series_metadata();
                self.save_preferences();
            }
            ui.small("Frame 1 seeds closed components automatically. P remains available for manual add/remove: the clicked pixel + frame become an immutable identity anchor; hover an existing pivot (yellow) and press P to remove it.");
            if ui
                .checkbox(&mut self.plot_click_seek_enabled, "Click plot X-axis to seek frame")
                .changed()
            {
                self.save_preferences();
            }

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
            ui.small("Frame 1 auto-seeds one pivot for every closed mask component. If pivots ever resolve to the same component, their groups merge permanently and are plotted as one union shape.");
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.shape_tracks.is_empty(), egui::Button::new("Open size plot"))
                    .clicked()
                {
                    self.shape_plot_open = true;
                    if let Some(plot) = self.plot_windows.first_mut() {
                        plot.open = true;
                    }
                }
                if ui
                    .add_enabled(!self.shape_tracks.is_empty(), egui::Button::new("Clear tracks"))
                    .clicked()
                {
                    self.shape_tracks.clear();
                    self.sync_shape_series_metadata();
                }
            });
            let mut group_counts = BTreeMap::<u64, usize>::new();
            for track in &self.shape_tracks {
                *group_counts.entry(track.group_id).or_default() += 1;
            }
            let mut remove_track = None;
            for (index, track) in self.shape_tracks.iter_mut().enumerate() {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut track.enabled, "");
                    let grouped = group_counts.get(&track.group_id).copied().unwrap_or(1);
                    ui.colored_label(
                        shape_series_color(track.group_id),
                        if grouped > 1 {
                            format!("G{} / T{}", track.group_id, track.id)
                        } else {
                            format!("T{}", track.id)
                        },
                    );
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
                self.sync_shape_series_metadata();
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
        if !processed_interaction.selected_overlay_ids.is_empty() {
            self.group_selected_shape_series(&processed_interaction.selected_overlay_ids);
        }
        if let Some(group_id) = processed_interaction.toggle_overlay_id {
            let before = self.shape_tracks.len();
            self.shape_tracks.retain(|track| track.group_id != group_id);
            let removed = before.saturating_sub(self.shape_tracks.len());
            if removed > 0 {
                self.sync_shape_series_metadata();
                self.status = if removed == 1 {
                    format!("Removed tracked pivot T{group_id}.")
                } else {
                    format!("Removed merged shape group G{group_id} ({removed} pivots).")
                };
                if self.shape_tracks.is_empty() {
                    self.shape_plot_open = false;
                }
            }
        } else if let Some(pixel) = processed_interaction.pivot_pixel {
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

    fn timeseries_plot_windows(&mut self, ctx: &egui::Context) {
        if self.plot_windows.is_empty() {
            return;
        }
        let sequence_times = self
            .active_sequence
            .as_ref()
            .map(|sequence| sequence.frames.iter().map(|frame| frame.timestamp).collect::<Vec<_>>())
            .unwrap_or_default();
        let selected_frame = self.active_sequence.as_ref().map(|sequence| sequence.selected);
        let click_seek_enabled = self.plot_click_seek_enabled;
        let mut pending_seek = None;

        for index in 0..self.plot_windows.len() {
            let plot_id = self.plot_windows[index].id;
            if !self.plot_windows[index].open {
                continue;
            }
            let series = self.collect_plot_series_for_plot(plot_id);
            let mut open = self.plot_windows[index].open;
            let title = self.plot_windows[index].title.clone();
            let pos = self.plot_windows[index].pos;
            let size = self.plot_windows[index].size;
            let mut x_center = self.plot_windows[index].x_center.clamp(0.0, 1.0);
            let mut x_zoom = self.plot_windows[index].x_zoom.clamp(1.0, 10_000.0);
            let mut y_center = self.plot_windows[index].y_center.clamp(0.0, 1.0);
            let mut y_zoom = self.plot_windows[index].y_zoom.clamp(1.0, 10_000.0);
            let mut local_seek = None;

            let response = egui::Window::new(title.clone())
                .id(egui::Id::new(("timeseries-plot-window", plot_id)))
                .open(&mut open)
                .default_pos(pos)
                .default_size(size)
                .current_pos(pos)
                .min_size(egui::vec2(160.0, 58.0))
                .constrain(false)
                .resizable([true, true])
                .show(ctx, |ui| {
                    ui.set_min_size(egui::Vec2::ZERO);
                    let available_h = ui.available_height();
                    if available_h > 130.0 {
                        ui.horizontal(|ui| {
                            ui.small("scroll X/Y = pan · Ctrl+scroll X/Y = zoom · hover = value");
                            if ui.small_button("Reset view").clicked() {
                                x_center = 0.5;
                                x_zoom = 1.0;
                                y_center = 0.5;
                                y_zoom = 1.0;
                            }
                        });
                    }
                    if available_h > 92.0 && !series.is_empty() {
                        egui::ScrollArea::horizontal()
                            .id_salt(("plot-legend", plot_id))
                            .max_height(22.0)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    for item in &series {
                                        let suffix = item
                                            .points
                                            .last()
                                            .map(|point| format_plot_value(point.value, &item.unit))
                                            .unwrap_or_else(|| "no samples".to_owned());
                                        ui.colored_label(
                                            series_color(item.color_seed),
                                            format!("{} / {} [{}]", item.group, item.name, suffix),
                                        );
                                        ui.separator();
                                    }
                                });
                            });
                    }

                    if series.is_empty() {
                        ui.centered_and_justified(|ui| {
                            ui.monospace("No timeseries assigned to this plot yet.");
                        });
                        return;
                    }

                    let available = ui.available_size();
                    let plot_width = if available.x.is_finite() { available.x.max(120.0) } else { size.x.max(220.0) - 12.0 };
                    let plot_height = if available.y.is_finite() { available.y.max(12.0) } else { (size.y - 50.0).max(12.0) };
                    let sense = if click_seek_enabled { egui::Sense::click() } else { egui::Sense::hover() };
                    let (plot_response, painter) = ui.allocate_painter(egui::vec2(plot_width, plot_height), sense);
                    let horizontal_margin = if plot_width > 260.0 { 58.0 } else if plot_width > 150.0 { 26.0 } else { 4.0 };
                    let vertical_margin = if plot_height > 95.0 { 28.0 } else if plot_height > 40.0 { 10.0 } else { 1.0 };
                    let rect = plot_response.rect.shrink2(egui::vec2(horizontal_margin, vertical_margin));
                    if rect.width() <= 2.0 || rect.height() <= 2.0 {
                        return;
                    }

                    let mut start_time = None::<SystemTime>;
                    let mut end_time = None::<SystemTime>;
                    let mut data_min = f64::INFINITY;
                    let mut data_max = f64::NEG_INFINITY;
                    let shape_only = series.iter().all(|item| item.unit == "px");
                    for item in &series {
                        for point in &item.points {
                            start_time = Some(start_time.map_or(point.time, |current| current.min(point.time)));
                            end_time = Some(end_time.map_or(point.time, |current| current.max(point.time)));
                            data_min = data_min.min(point.value);
                            data_max = data_max.max(point.value);
                        }
                    }
                    let (start_time, end_time) = match (start_time, end_time) {
                        (Some(start), Some(end)) => (start, end),
                        _ => return,
                    };
                    let full_span = end_time
                        .duration_since(start_time)
                        .unwrap_or(Duration::from_secs(1))
                        .max(Duration::from_millis(1));
                    if shape_only && data_min >= 0.0 {
                        data_min = 0.0;
                    }
                    if !data_min.is_finite() || !data_max.is_finite() {
                        return;
                    }
                    if (data_max - data_min).abs() < 1e-9 {
                        let delta = data_max.abs().max(1.0) * 0.1;
                        data_min -= delta;
                        data_max += delta;
                    } else {
                        let padding = (data_max - data_min) * 0.05;
                        data_min -= padding;
                        data_max += padding;
                    }
                    let data_span = (data_max - data_min).max(1e-9);

                    if plot_response.contains_pointer() {
                        let mut pan_scroll = egui::Vec2::ZERO;
                        let mut zoom_scroll = egui::Vec2::ZERO;
                        ui.input(|input| {
                            for event in &input.events {
                                if let egui::Event::MouseWheel { unit, delta, modifiers, .. } = event {
                                    let scale = match unit {
                                        egui::MouseWheelUnit::Point => 1.0,
                                        egui::MouseWheelUnit::Line => 24.0,
                                        egui::MouseWheelUnit::Page => 180.0,
                                    };
                                    if modifiers.ctrl {
                                        zoom_scroll += *delta * scale;
                                    } else {
                                        pan_scroll += *delta * scale;
                                    }
                                }
                            }
                        });
                        let pointer = plot_response.hover_pos().unwrap_or_else(|| rect.center());
                        let anchor_x = ((pointer.x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64;
                        let anchor_y = ((rect.bottom() - pointer.y) / rect.height()).clamp(0.0, 1.0);
                        if pan_scroll.x.abs() > f32::EPSILON {
                            let visible = 1.0 / x_zoom as f64;
                            x_center -= pan_scroll.x as f64 / rect.width() as f64 * visible;
                        }
                        if pan_scroll.y.abs() > f32::EPSILON {
                            let visible = 1.0 / y_zoom;
                            y_center += pan_scroll.y / rect.height() * visible;
                        }
                        if zoom_scroll.x.abs() > f32::EPSILON {
                            let old_visible = 1.0 / x_zoom as f64;
                            let old_min = x_center - old_visible * 0.5;
                            let anchor_value = old_min + anchor_x * old_visible;
                            x_zoom = (x_zoom * (zoom_scroll.x * 0.0025).exp()).clamp(1.0, 10_000.0);
                            let new_visible = 1.0 / x_zoom as f64;
                            x_center = anchor_value + (0.5 - anchor_x) * new_visible;
                        }
                        if zoom_scroll.y.abs() > f32::EPSILON {
                            let old_visible = 1.0 / y_zoom;
                            let old_min = y_center - old_visible * 0.5;
                            let anchor_value = old_min + anchor_y * old_visible;
                            y_zoom = (y_zoom * (zoom_scroll.y * 0.0025).exp()).clamp(1.0, 10_000.0);
                            let new_visible = 1.0 / y_zoom;
                            y_center = anchor_value + (0.5 - anchor_y) * new_visible;
                        }
                        let x_half = 0.5 / x_zoom as f64;
                        x_center = x_center.clamp(x_half, 1.0 - x_half);
                        let y_half = 0.5 / y_zoom;
                        y_center = y_center.clamp(y_half, 1.0 - y_half);
                    }

                    let x_visible = 1.0 / x_zoom as f64;
                    let x_min = (x_center - x_visible * 0.5).clamp(0.0, 1.0 - x_visible);
                    let x_max = x_min + x_visible;
                    let y_visible = 1.0 / y_zoom;
                    let y_min_norm = (y_center - y_visible * 0.5).clamp(0.0, 1.0 - y_visible);
                    let y_max_norm = y_min_norm + y_visible;
                    let visible_span = Duration::from_secs_f64((full_span.as_secs_f64() * x_visible).max(0.001));

                    let x_for_time = |time: SystemTime| -> Option<f32> {
                        let offset = time.duration_since(start_time).ok()?;
                        let norm = (offset.as_secs_f64() / full_span.as_secs_f64()).clamp(0.0, 1.0);
                        Some(rect.left() + ((norm - x_min) / (x_max - x_min)) as f32 * rect.width())
                    };
                    let y_for_value = |value: f64| -> f32 {
                        let norm = ((value - data_min) / data_span) as f32;
                        rect.bottom() - ((norm - y_min_norm) / (y_max_norm - y_min_norm)) * rect.height()
                    };

                    let border = egui::Stroke::new(1.0, egui::Color32::DARK_GRAY);
                    painter.rect_stroke(rect, 0.0, border, egui::StrokeKind::Inside);
                    let grid_steps = if rect.height() > 85.0 { 4 } else { 2 };
                    for step in 0..=grid_steps {
                        let t = step as f32 / grid_steps.max(1) as f32;
                        let y = egui::lerp(rect.bottom()..=rect.top(), t);
                        painter.line_segment(
                            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                            egui::Stroke::new(0.5, egui::Color32::from_gray(45)),
                        );
                        if rect.height() > 55.0 {
                            let value_norm = y_min_norm + (y_max_norm - y_min_norm) * t;
                            let value = data_min + data_span * value_norm as f64;
                            painter.text(
                                egui::pos2(rect.left() - 5.0, y),
                                egui::Align2::RIGHT_CENTER,
                                format_compact_number(value),
                                egui::FontId::monospace(9.0),
                                egui::Color32::GRAY,
                            );
                        }
                    }
                    let time_steps = if rect.width() > 520.0 { 4 } else if rect.width() > 300.0 { 2 } else { 1 };
                    for step in 0..=time_steps {
                        let t = step as f64 / time_steps.max(1) as f64;
                        let full_fraction = x_min + (x_max - x_min) * t;
                        let tick_time = start_time + Duration::from_secs_f64(full_span.as_secs_f64() * full_fraction);
                        let x = egui::lerp(rect.left()..=rect.right(), t as f32);
                        painter.line_segment(
                            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                            egui::Stroke::new(0.5, egui::Color32::from_gray(35)),
                        );
                        if rect.width() > 180.0 && plot_height > 48.0 {
                            painter.text(
                                egui::pos2(x, rect.bottom() + 5.0),
                                egui::Align2::CENTER_TOP,
                                format_plot_axis_time(tick_time, visible_span),
                                egui::FontId::monospace(9.0),
                                egui::Color32::GRAY,
                            );
                        }
                    }

                    let data_painter = painter.with_clip_rect(rect);
                    let plot_hover = plot_response.hover_pos().filter(|pos| rect.contains(*pos));
                    let mut hovered: Option<(f32, usize, usize, egui::Pos2)> = None;
                    for (series_index, item) in series.iter().enumerate() {
                        let color = series_color(item.color_seed);
                        let mut points = Vec::with_capacity(item.points.len());
                        for (point_index, point) in item.points.iter().enumerate() {
                            let Some(x) = x_for_time(point.time) else { continue; };
                            let y = y_for_value(point.value);
                            let screen = egui::pos2(x, y);
                            points.push(screen);
                            if rect.contains(screen) {
                                if let Some(pointer) = plot_hover {
                                    let distance_sq = screen.distance_sq(pointer);
                                    if distance_sq <= 9.0 * 9.0
                                        && hovered.as_ref().is_none_or(|current| distance_sq < current.0)
                                    {
                                        hovered = Some((distance_sq, series_index, point_index, screen));
                                    }
                                }
                            }
                        }
                        for pair in points.windows(2) {
                            data_painter.line_segment([pair[0], pair[1]], egui::Stroke::new(2.0, color));
                        }
                        for point in points {
                            data_painter.circle_filled(point, 2.8, color);
                        }
                    }

                    if let Some((_, series_index, point_index, screen)) = hovered {
                        let item = &series[series_index];
                        let point = &item.points[point_index];
                        data_painter.circle_filled(screen, 4.5, egui::Color32::YELLOW);
                        data_painter.circle_stroke(screen, 7.0, egui::Stroke::new(1.2, egui::Color32::YELLOW));
                        let frame_text = point.frame.map(|frame| format!("\nframe: {}", frame + 1)).unwrap_or_default();
                        let text = format!(
                            "{}\ngroup: {}\nvalue: {}{}\ntime: {}",
                            item.name,
                            item.group,
                            format_compact_number(point.value),
                            if item.unit.is_empty() { String::new() } else { format!(" {}", item.unit) },
                            format_absolute_time(point.time),
                        ) + &frame_text;
                        let lines = text.lines().count() as f32;
                        let tooltip_size = egui::vec2(250.0, lines * 15.0 + 12.0);
                        let mut tooltip_pos = screen + egui::vec2(10.0, 10.0);
                        if tooltip_pos.x + tooltip_size.x > plot_response.rect.right() {
                            tooltip_pos.x = screen.x - tooltip_size.x - 10.0;
                        }
                        if tooltip_pos.y + tooltip_size.y > plot_response.rect.bottom() {
                            tooltip_pos.y = screen.y - tooltip_size.y - 10.0;
                        }
                        let tooltip_rect = egui::Rect::from_min_size(tooltip_pos, tooltip_size);
                        painter.rect_filled(tooltip_rect, 3.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 230));
                        painter.text(
                            tooltip_rect.min + egui::vec2(7.0, 6.0),
                            egui::Align2::LEFT_TOP,
                            text,
                            egui::FontId::monospace(10.5),
                            egui::Color32::YELLOW,
                        );
                    }

                    if let Some(frame_index) = selected_frame
                        && let Some(timestamp) = sequence_times.get(frame_index)
                        && let Some(x) = x_for_time(*timestamp)
                        && x >= rect.left()
                        && x <= rect.right()
                    {
                        let stroke = egui::Stroke::new(1.4, egui::Color32::WHITE);
                        let mut y = rect.top();
                        while y < rect.bottom() {
                            let y2 = (y + 6.0).min(rect.bottom());
                            painter.line_segment([egui::pos2(x, y), egui::pos2(x, y2)], stroke);
                            y += 10.0;
                        }
                        if plot_height > 62.0 {
                            painter.text(
                                egui::pos2(x, rect.top() - 4.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!("F{} {}", frame_index + 1, format_plot_axis_time(*timestamp, visible_span)),
                                egui::FontId::monospace(9.0),
                                egui::Color32::WHITE,
                            );
                        }
                    }

                    if click_seek_enabled && !sequence_times.is_empty() {
                        let axis_hit = egui::Rect::from_min_max(
                            egui::pos2(rect.left(), (rect.bottom() - 8.0).max(rect.top())),
                            egui::pos2(rect.right(), plot_response.rect.bottom()),
                        );
                        if let Some(pointer) = plot_response.hover_pos() {
                            if axis_hit.contains(pointer) {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                            }
                        }
                        if plot_response.clicked()
                            && let Some(pointer) = plot_response.interact_pointer_pos()
                            && axis_hit.contains(pointer)
                        {
                            let fraction = ((pointer.x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64;
                            let target_norm = x_min + fraction * (x_max - x_min);
                            let target_time = start_time + Duration::from_secs_f64(full_span.as_secs_f64() * target_norm);
                            local_seek = sequence_times
                                .iter()
                                .enumerate()
                                .min_by_key(|(_, time)| system_time_distance(**time, target_time))
                                .map(|(frame, _)| frame);
                        }
                    }
                });

            if let Some(response) = response {
                self.plot_windows[index].pos = response.response.rect.min;
                self.plot_windows[index].size = response.response.rect.size();
            }
            self.plot_windows[index].open = open;
            self.plot_windows[index].x_center = x_center;
            self.plot_windows[index].x_zoom = x_zoom;
            self.plot_windows[index].y_center = y_center;
            self.plot_windows[index].y_zoom = y_zoom;
            if local_seek.is_some() {
                pending_seek = local_seek;
            }
        }

        if let Some(frame) = pending_seek {
            self.seek_sequence_frame(frame);
        }
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
        self.poll_home_assistant_worker();
        self.tick_home_assistant_poll(&ctx);
        self.tick_continuous_capture(&ctx);
        self.handle_global_shortcuts(&ctx);
        self.tick_sequence_playback(&ctx);
        self.sync_status_history();

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

        self.timeseries_plot_windows(&ctx);
        self.sync_status_history();
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
        .user_agent("rust-edge-gui/0.8.2")
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
    let raw_shape_mask = shape_mask.clone();
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
        raw_shape_mask,
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

fn component_is_closed(component: &MaskComponent, width: usize, height: usize) -> bool {
    if component.pixels.is_empty() || width < 2 || height < 2 {
        return false;
    }
    component.pixels.iter().all(|&index| {
        let index = index as usize;
        let x = index % width;
        let y = index / width;
        x > 0 && y > 0 && x + 1 < width && y + 1 < height
    })
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


fn pivot_radius_for_area(area: usize) -> f32 {
    // Screen-space radius follows log10(area), so a 10× larger leaf is visibly
    // larger without letting large masks dominate the image. Zero/missing areas
    // keep the smallest marker.
    if area == 0 {
        return 4.5;
    }
    (4.5 + (area as f32 + 1.0).log10() * 2.25).clamp(4.5, 18.0)
}

fn build_shape_group_series(tracks: &[ShapeTrack], width: usize) -> Vec<ShapeGroupSeries> {
    let mut grouped: BTreeMap<u64, Vec<&ShapeTrack>> = BTreeMap::new();
    for track in tracks.iter().filter(|track| track.enabled) {
        grouped.entry(track.group_id).or_default().push(track);
    }

    let mut result = Vec::with_capacity(grouped.len());
    for (group_id, members) in grouped {
        let mut frame_pixels: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
        let mut anchor_frame = usize::MAX;
        for track in &members {
            anchor_frame = anchor_frame.min(track.anchor_frame);
            for (&frame, pixels) in &track.matched_pixels {
                frame_pixels
                    .entry(frame)
                    .or_default()
                    .extend(pixels.iter().copied());
            }
        }
        let anchor_frame = if anchor_frame == usize::MAX { 0 } else { anchor_frame };
        let mut areas = BTreeMap::new();
        let mut centroids = BTreeMap::new();
        for (frame, pixels) in frame_pixels {
            areas.insert(frame, pixels.len());
            if width > 0 && !pixels.is_empty() {
                let mut sum_x = 0u64;
                let mut sum_y = 0u64;
                for pixel in &pixels {
                    let index = *pixel as usize;
                    sum_x += (index % width) as u64;
                    sum_y += (index / width) as u64;
                }
                let count = pixels.len() as u64;
                centroids.insert(frame, ((sum_x / count) as u32, (sum_y / count) as u32));
            }
        }
        let name = if members.len() == 1 {
            members[0].name.clone()
        } else {
            format!("merged-{}", group_id)
        };
        result.push(ShapeGroupSeries {
            id: group_id,
            name,
            member_count: members.len(),
            anchor_frame,
            areas,
            centroids,
        });
    }
    result
}

fn shape_series_key(group_id: u64) -> String {
    format!("shape:{group_id}")
}

fn parse_shape_series_key(key: &str) -> Option<u64> {
    key.strip_prefix("shape:")?.parse().ok()
}

fn ha_series_key(entity_id: &str, attribute: &str) -> String {
    let attribute = attribute.trim();
    if attribute.is_empty() || attribute.eq_ignore_ascii_case("state") {
        format!("ha:{}:state", entity_id.trim())
    } else {
        format!("ha:{}:{}", entity_id.trim(), attribute)
    }
}

fn stable_series_seed(key: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn series_color(seed: u64) -> egui::Color32 {
    track_color(seed.max(1))
}

fn shape_series_color(group_id: u64) -> egui::Color32 {
    series_color(stable_series_seed(&shape_series_key(group_id)))
}

fn format_compact_number(value: f64) -> String {
    let abs = value.abs();
    if abs >= 1_000_000.0 {
        format!("{:.2}M", value / 1_000_000.0)
    } else if abs >= 1_000.0 {
        format!("{:.2}k", value / 1_000.0)
    } else if abs >= 100.0 {
        format!("{value:.0}")
    } else if abs >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

fn format_plot_value(value: f64, unit: &str) -> String {
    if unit.trim().is_empty() {
        format_compact_number(value)
    } else {
        format!("{} {}", format_compact_number(value), unit)
    }
}

fn system_time_distance(a: SystemTime, b: SystemTime) -> Duration {
    if a >= b {
        a.duration_since(b).unwrap_or_default()
    } else {
        b.duration_since(a).unwrap_or_default()
    }
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

fn remove_and_shift_index_map<T>(map: &mut BTreeMap<usize, T>, removed_index: usize) {
    let old = std::mem::take(map);
    *map = old
        .into_iter()
        .filter_map(|(index, value)| {
            if index == removed_index {
                None
            } else {
                Some((if index > removed_index { index - 1 } else { index }, value))
            }
        })
        .collect();
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

fn format_status_time(time: SystemTime) -> String {
    let date_time: DateTime<Local> = time.into();
    date_time.format("%Y-%m-%d %H:%M:%S").to_string()
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
