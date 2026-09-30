use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};
use eframe::egui;
use reqwest::blocking::Client;
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct HaSeriesRequest {
    pub key: String,
    pub entity_id: String,
    /// Empty or "state" means the entity's primary state. Otherwise this is
    /// looked up in the state object's attributes map.
    pub attribute: String,
}

#[derive(Clone, Debug)]
pub struct HaSample {
    pub key: String,
    pub timestamp: SystemTime,
    pub value: f64,
    pub unit: String,
    pub friendly_name: String,
}

#[derive(Clone, Debug)]
pub enum HaRequestKind {
    Snapshot,
    History { start: SystemTime, end: SystemTime },
}

#[derive(Clone, Debug)]
pub struct HaRequest {
    pub id: u64,
    pub base_url: String,
    pub token: String,
    pub series: Vec<HaSeriesRequest>,
    pub kind: HaRequestKind,
}

#[derive(Debug)]
pub enum HaMessage {
    Finished { id: u64, samples: Vec<HaSample> },
    Failed { id: u64, error: String },
}

pub struct HaWorker {
    pub request_tx: mpsc::Sender<HaRequest>,
    pub message_rx: mpsc::Receiver<HaMessage>,
    _thread: thread::JoinHandle<()>,
}

impl HaWorker {
    pub fn spawn(repaint_ctx: egui::Context) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<HaRequest>();
        let (message_tx, message_rx) = mpsc::channel::<HaMessage>();
        let worker = thread::Builder::new()
            .name("home-assistant-worker".to_owned())
            .spawn(move || worker_loop(request_rx, message_tx, repaint_ctx))
            .expect("failed to spawn Home Assistant worker");
        Self { request_tx, message_rx, _thread: worker }
    }
}

fn worker_loop(
    request_rx: mpsc::Receiver<HaRequest>,
    message_tx: mpsc::Sender<HaMessage>,
    repaint_ctx: egui::Context,
) {
    let client = match Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(45))
        .user_agent("rust-edge-gui/0.8.2")
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            let _ = message_tx.send(HaMessage::Failed { id: 0, error: error.to_string() });
            return;
        }
    };

    while let Ok(request) = request_rx.recv() {
        let id = request.id;
        let result = match request.kind.clone() {
            HaRequestKind::Snapshot => fetch_snapshot(&client, &request),
            HaRequestKind::History { start, end } => fetch_history(&client, &request, start, end),
        };
        let message = match result {
            Ok(samples) => HaMessage::Finished { id, samples },
            Err(error) => HaMessage::Failed { id, error },
        };
        if message_tx.send(message).is_err() {
            return;
        }
        repaint_ctx.request_repaint();
    }
}

fn authenticated_get(client: &Client, url: reqwest::Url, token: &str) -> Result<Value, String> {
    client
        .get(url)
        .bearer_auth(token)
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?
        .json::<Value>()
        .map_err(|error| error.to_string())
}

fn api_url(base_url: &str, path: &str) -> Result<reqwest::Url, String> {
    let base = base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err("Home Assistant base URL is empty".to_owned());
    }
    reqwest::Url::parse(&format!("{base}{path}"))
        .map_err(|error| format!("invalid Home Assistant URL: {error}"))
}

fn fetch_snapshot(client: &Client, request: &HaRequest) -> Result<Vec<HaSample>, String> {
    let mut samples = Vec::new();
    for spec in &request.series {
        let url = api_url(
            &request.base_url,
            &format!("/api/states/{}", spec.entity_id.trim()),
        )?;
        let value = authenticated_get(client, url, request.token.trim())?;
        if let Some(sample) = sample_from_state(spec, &value) {
            samples.push(sample);
        }
    }
    Ok(samples)
}

fn fetch_history(
    client: &Client,
    request: &HaRequest,
    start: SystemTime,
    end: SystemTime,
) -> Result<Vec<HaSample>, String> {
    if request.series.is_empty() {
        return Ok(Vec::new());
    }
    let start: DateTime<Utc> = start.into();
    let end: DateTime<Utc> = end.into();
    let mut url = api_url(&request.base_url, "/api/history/period")?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| "Home Assistant base URL cannot be used as a base URL".to_owned())?;
        segments.push(&start.to_rfc3339());
    }
    let entity_ids = request
        .series
        .iter()
        .map(|spec| spec.entity_id.trim())
        .collect::<Vec<_>>()
        .join(",");
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("end_time", &end.to_rfc3339());
        query.append_pair("filter_entity_id", &entity_ids);
    }

    let value = authenticated_get(client, url, request.token.trim())?;
    let mut samples = Vec::new();
    let Some(entity_histories) = value.as_array() else {
        return Err("Home Assistant history response is not an array".to_owned());
    };
    for history in entity_histories {
        let Some(states) = history.as_array() else { continue; };
        for state in states {
            let entity_id = state.get("entity_id").and_then(Value::as_str).unwrap_or_default();
            for spec in request.series.iter().filter(|spec| spec.entity_id == entity_id) {
                if let Some(sample) = sample_from_state(spec, state) {
                    samples.push(sample);
                }
            }
        }
    }
    samples.sort_by_key(|sample| sample.timestamp);
    Ok(samples)
}

fn sample_from_state(spec: &HaSeriesRequest, state: &Value) -> Option<HaSample> {
    let attribute = spec.attribute.trim();
    let raw = if attribute.is_empty() || attribute.eq_ignore_ascii_case("state") {
        state.get("state")?
    } else {
        state.get("attributes")?.get(attribute)?
    };
    let value = json_number(raw)?;
    let timestamp_text = state
        .get("last_updated")
        .or_else(|| state.get("last_changed"))?
        .as_str()?;
    let timestamp = DateTime::parse_from_rfc3339(timestamp_text).ok()?;
    let timestamp: SystemTime = timestamp.with_timezone(&Utc).into();
    let attributes = state.get("attributes");
    let unit = attributes
        .and_then(|attrs| attrs.get("unit_of_measurement"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let friendly_name = attributes
        .and_then(|attrs| attrs.get("friendly_name"))
        .and_then(Value::as_str)
        .unwrap_or(&spec.entity_id)
        .to_owned();
    Some(HaSample {
        key: spec.key.clone(),
        timestamp,
        value,
        unit,
        friendly_name,
    })
}

fn json_number(value: &Value) -> Option<f64> {
    if let Some(number) = value.as_f64() {
        return number.is_finite().then_some(number);
    }
    let text = value.as_str()?.trim();
    let parsed = text.parse::<f64>().ok()?;
    parsed.is_finite().then_some(parsed)
}
