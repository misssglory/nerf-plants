use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Instant;

use anyhow::{anyhow, Context as _, Result};
use eframe::egui;
use image::{DynamicImage, RgbaImage};
use ultralytics_inference::{Device, InferenceConfig, SemanticMask, YOLOModel};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YoloDevice {
    Auto,
    Cpu,
}

impl YoloDevice {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Cpu => "CPU",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct YoloRuntimeSettings {
    pub model_path: String,
    pub confidence: f32,
    pub iou: f32,
    pub input_size: usize,
    pub device: YoloDevice,
    /// Incremented by the GUI when the user explicitly requests a reload.
    pub generation: u64,
}

#[derive(Clone, Debug)]
pub struct YoloPostSettings {
    /// Empty means "all detected classes" for instance segmentation.
    /// Semantic segmentation requires at least one selected class because every
    /// pixel has a semantic class (including background).
    pub class_ids: Vec<usize>,
    pub mask_threshold: f32,
}

#[derive(Clone)]
pub struct YoloRequest {
    pub id: u64,
    pub rgba: Arc<RgbaImage>,
    pub source_label: String,
    pub runtime: YoloRuntimeSettings,
    pub post: YoloPostSettings,
}

#[derive(Clone, Debug)]
pub struct YoloModelInfo {
    pub model_path: String,
    pub task: String,
    pub provider: String,
    pub input_size: (usize, usize),
    pub class_count: usize,
}

#[derive(Clone, Debug)]
pub struct YoloOutput {
    pub mask: Vec<bool>,
    pub width: usize,
    pub height: usize,
    pub instance_count: usize,
    pub mask_pixels: usize,
    pub summary: String,
    pub elapsed_ms: f64,
    pub model_info: YoloModelInfo,
}

pub enum YoloMessage {
    Finished { id: u64, output: YoloOutput },
    Failed { id: u64, error: String },
}

pub struct YoloWorker {
    pub tx: mpsc::Sender<YoloRequest>,
    pub rx: mpsc::Receiver<YoloMessage>,
    pub latest_id: Arc<AtomicU64>,
    _thread: thread::JoinHandle<()>,
}

impl YoloWorker {
    pub fn spawn(repaint_ctx: egui::Context) -> Self {
        let (tx, job_rx) = mpsc::channel::<YoloRequest>();
        let (message_tx, rx) = mpsc::channel::<YoloMessage>();
        let latest_id = Arc::new(AtomicU64::new(0));
        let worker_latest = Arc::clone(&latest_id);

        let worker = thread::Builder::new()
            .name("yolo-segmentation-worker".to_owned())
            .spawn(move || yolo_loop(job_rx, message_tx, worker_latest, repaint_ctx))
            .expect("failed to spawn YOLO worker");

        Self {
            tx,
            rx,
            latest_id,
            _thread: worker,
        }
    }
}

fn yolo_loop(
    job_rx: mpsc::Receiver<YoloRequest>,
    message_tx: mpsc::Sender<YoloMessage>,
    latest_id: Arc<AtomicU64>,
    repaint_ctx: egui::Context,
) {
    let mut model: Option<YOLOModel> = None;
    let mut loaded_runtime: Option<YoloRuntimeSettings> = None;

    while let Ok(mut request) = job_rx.recv() {
        // Segmentation is expensive. If the UI queued several frames/settings
        // while we were busy, process only the newest one.
        while let Ok(newer) = job_rx.try_recv() {
            request = newer;
        }

        if latest_id.load(Ordering::Acquire) != request.id {
            continue;
        }

        let result = (|| -> Result<YoloOutput> {
            if request.runtime.model_path.trim().is_empty() {
                return Err(anyhow!("YOLO model path is empty"));
            }

            if loaded_runtime.as_ref() != Some(&request.runtime) || model.is_none() {
                let mut config = InferenceConfig::new()
                    .with_confidence(request.runtime.confidence.clamp(0.0, 1.0))
                    .with_iou(request.runtime.iou.clamp(0.0, 1.0))
                    .with_max_det(1000);

                if request.runtime.input_size > 0 {
                    config = config.with_imgsz(
                        request.runtime.input_size,
                        request.runtime.input_size,
                    );
                }
                if request.runtime.device == YoloDevice::Cpu {
                    config = config.with_device(Device::Cpu);
                }

                let loaded = YOLOModel::load_with_config(&request.runtime.model_path, config)
                    .with_context(|| {
                        format!("failed to load YOLO model {}", request.runtime.model_path)
                    })?;
                model = Some(loaded);
                loaded_runtime = Some(request.runtime.clone());
            }

            let model = model.as_mut().expect("model loaded above");
            let info = YoloModelInfo {
                model_path: request.runtime.model_path.clone(),
                task: format!("{:?}", model.task()),
                provider: model.execution_provider().to_owned(),
                input_size: model.imgsz(),
                class_count: model.num_classes(),
            };

            let image = DynamicImage::ImageRgba8((*request.rgba).clone());
            let start = Instant::now();
            let results = model
                .predict_image(&image, request.source_label.clone())
                .context("YOLO inference failed")?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let result = results
                .first()
                .ok_or_else(|| anyhow!("YOLO returned no Results object"))?;

            let source_width = request.rgba.width() as usize;
            let source_height = request.rgba.height() as usize;
            let mut output_mask = vec![false; source_width * source_height];
            let selected = &request.post.class_ids;
            let threshold = request.post.mask_threshold.clamp(0.0, 1.0);
            let mut instance_count = 0usize;
            let mut class_counts = BTreeMap::<usize, usize>::new();

            if let Some(masks) = &result.masks {
                let (n, mask_h, mask_w) = masks.data.dim();
                let boxes = result.boxes.as_ref();

                for mask_index in 0..n {
                    let class_id = boxes.and_then(|boxes| {
                        if mask_index < boxes.len() {
                            Some(boxes.cls()[mask_index] as usize)
                        } else {
                            None
                        }
                    });
                    if !selected.is_empty() {
                        match class_id {
                            Some(class_id) if selected.contains(&class_id) => {}
                            _ => continue,
                        }
                    }

                    instance_count += 1;
                    if let Some(class_id) = class_id {
                        *class_counts.entry(class_id).or_default() += 1;
                    }

                    for y in 0..source_height {
                        let sy = (y * mask_h / source_height).min(mask_h.saturating_sub(1));
                        for x in 0..source_width {
                            let sx = (x * mask_w / source_width).min(mask_w.saturating_sub(1));
                            if masks.data[[mask_index, sy, sx]] >= threshold {
                                output_mask[y * source_width + x] = true;
                            }
                        }
                    }
                }
            } else if let Some(semantic) = &result.semantic_mask {
                if selected.is_empty() {
                    return Err(anyhow!(
                        "semantic segmentation needs at least one class ID; set Plant class IDs in the GUI"
                    ));
                }
                let (mask_h, mask_w) = semantic.data.dim();
                for y in 0..source_height {
                    let sy = (y * mask_h / source_height).min(mask_h.saturating_sub(1));
                    for x in 0..source_width {
                        let sx = (x * mask_w / source_width).min(mask_w.saturating_sub(1));
                        let class_id = semantic.data[[sy, sx]];
                        if class_id != SemanticMask::IGNORE
                            && selected.contains(&(class_id as usize))
                        {
                            output_mask[y * source_width + x] = true;
                        }
                    }
                }
                instance_count = semantic
                    .class_ids()
                    .into_iter()
                    .filter(|class_id| selected.contains(class_id))
                    .count();
                for class_id in semantic.class_ids() {
                    if selected.contains(&class_id) {
                        class_counts.insert(class_id, 1);
                    }
                }
            } else {
                return Err(anyhow!(
                    "model task {:?} produced neither instance masks nor a semantic mask; choose a *-seg.onnx or *-sem.onnx model",
                    model.task()
                ));
            }

            let mask_pixels = output_mask.iter().filter(|&&value| value).count();
            let summary = if class_counts.is_empty() {
                if instance_count == 0 {
                    "no selected plant instances".to_owned()
                } else {
                    format!("{instance_count} selected instance(s)")
                }
            } else {
                class_counts
                    .into_iter()
                    .map(|(class_id, count)| {
                        let name = result
                            .names
                            .get(&class_id)
                            .map(String::as_str)
                            .unwrap_or("class");
                        format!("{name}[{class_id}] ×{count}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };

            Ok(YoloOutput {
                mask: output_mask,
                width: source_width,
                height: source_height,
                instance_count,
                mask_pixels,
                summary,
                elapsed_ms,
                model_info: info,
            })
        })();

        if latest_id.load(Ordering::Acquire) != request.id {
            continue;
        }

        let message = match result {
            Ok(output) => YoloMessage::Finished {
                id: request.id,
                output,
            },
            Err(error) => YoloMessage::Failed {
                id: request.id,
                error: format!("YOLO segmentation failed: {error:#}"),
            },
        };
        let _ = message_tx.send(message);
        repaint_ctx.request_repaint();
    }
}
