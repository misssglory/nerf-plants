use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use eframe::egui;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PumpCommand {
    Forward,
    Off,
}

impl PumpCommand {
    pub const fn payload(self) -> &'static str {
        match self {
            Self::Forward => "FORWARD",
            Self::Off => "OFF",
        }
    }

    pub const fn label(self) -> &'static str {
        self.payload()
    }
}

#[derive(Clone, Debug)]
pub struct MqttPublishConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub topic: String,
    pub timeout: Duration,
}

#[derive(Clone, Debug)]
pub struct MqttRequest {
    pub id: u64,
    pub config: MqttPublishConfig,
    pub command: PumpCommand,
}

#[derive(Debug)]
pub enum MqttMessage {
    Finished { id: u64, command: PumpCommand },
    Failed { id: u64, command: PumpCommand, error: String },
}

pub struct MqttWorker {
    pub request_tx: mpsc::Sender<MqttRequest>,
    pub message_rx: mpsc::Receiver<MqttMessage>,
    _thread: thread::JoinHandle<()>,
}

impl MqttWorker {
    pub fn spawn(repaint_ctx: egui::Context) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<MqttRequest>();
        let (message_tx, message_rx) = mpsc::channel::<MqttMessage>();
        let worker = thread::Builder::new()
            .name("mqtt-pump-worker".to_owned())
            .spawn(move || worker_loop(request_rx, message_tx, repaint_ctx))
            .expect("failed to spawn MQTT pump worker");
        Self { request_tx, message_rx, _thread: worker }
    }
}

fn worker_loop(
    request_rx: mpsc::Receiver<MqttRequest>,
    message_tx: mpsc::Sender<MqttMessage>,
    repaint_ctx: egui::Context,
) {
    while let Ok(request) = request_rx.recv() {
        let result = publish_qos1(&request.config, request.command.payload().as_bytes());
        let message = match result {
            Ok(()) => MqttMessage::Finished { id: request.id, command: request.command },
            Err(error) => MqttMessage::Failed { id: request.id, command: request.command, error },
        };
        if message_tx.send(message).is_err() {
            return;
        }
        repaint_ctx.request_repaint();
    }
}

fn publish_qos1(config: &MqttPublishConfig, payload: &[u8]) -> Result<(), String> {
    let host = config.host.trim();
    let topic = config.topic.trim();
    if host.is_empty() {
        return Err("MQTT host is empty".to_owned());
    }
    if topic.is_empty() {
        return Err("MQTT topic is empty".to_owned());
    }

    let timeout = config.timeout.max(Duration::from_millis(250));
    let mut stream = connect_with_timeout(host, config.port, timeout)?;
    stream.set_read_timeout(Some(timeout)).map_err(|error| error.to_string())?;
    stream.set_write_timeout(Some(timeout)).map_err(|error| error.to_string())?;

    let client_id = format!(
        "rust-edge-gui-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );
    let connect = build_connect_packet(
        &client_id,
        config.username.trim(),
        &config.password,
    )?;
    stream.write_all(&connect).map_err(|error| format!("MQTT CONNECT write failed: {error}"))?;
    stream.flush().map_err(|error| format!("MQTT CONNECT flush failed: {error}"))?;

    let (packet_type, connack) = read_packet(&mut stream)?;
    if packet_type != 2 || connack.len() != 2 {
        return Err(format!("unexpected MQTT CONNACK packet type={packet_type} len={}", connack.len()));
    }
    if connack[1] != 0 {
        return Err(format!("MQTT broker rejected connection (CONNACK code {})", connack[1]));
    }

    let packet_id = 1u16;
    let publish = build_publish_qos1_packet(topic, packet_id, payload)?;
    stream.write_all(&publish).map_err(|error| format!("MQTT PUBLISH write failed: {error}"))?;
    stream.flush().map_err(|error| format!("MQTT PUBLISH flush failed: {error}"))?;

    loop {
        let (packet_type, body) = read_packet(&mut stream)?;
        match packet_type {
            4 if body.len() == 2 => {
                let ack_id = u16::from_be_bytes([body[0], body[1]]);
                if ack_id == packet_id {
                    break;
                }
            }
            13 => {}
            _ => {}
        }
    }

    let _ = stream.write_all(&[0xE0, 0x00]);
    let _ = stream.flush();
    Ok(())
}

fn connect_with_timeout(host: &str, port: u16, timeout: Duration) -> Result<TcpStream, String> {
    let addresses = (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("cannot resolve MQTT broker {host}:{port}: {error}"))?
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        return Err(format!("MQTT broker {host}:{port} resolved to no addresses"));
    }
    let mut last_error = None;
    for address in addresses {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = Some(error),
        }
    }
    Err(format!(
        "failed to connect to MQTT broker {host}:{port}: {}",
        last_error.map(|error| error.to_string()).unwrap_or_else(|| "unknown error".to_owned())
    ))
}

fn build_connect_packet(client_id: &str, username: &str, password: &str) -> Result<Vec<u8>, String> {
    let mut variable_and_payload = Vec::new();
    push_utf8(&mut variable_and_payload, "MQTT")?;
    variable_and_payload.push(4); // MQTT 3.1.1
    let mut flags = 0x02; // clean session
    if !username.is_empty() {
        flags |= 0x80;
    }
    if !password.is_empty() {
        flags |= 0x40;
    }
    variable_and_payload.push(flags);
    variable_and_payload.extend_from_slice(&10u16.to_be_bytes());
    push_utf8(&mut variable_and_payload, client_id)?;
    if !username.is_empty() {
        push_utf8(&mut variable_and_payload, username)?;
    }
    if !password.is_empty() {
        push_binary(&mut variable_and_payload, password.as_bytes())?;
    }

    let mut packet = vec![0x10];
    encode_remaining_length(variable_and_payload.len(), &mut packet)?;
    packet.extend_from_slice(&variable_and_payload);
    Ok(packet)
}

fn build_publish_qos1_packet(topic: &str, packet_id: u16, payload: &[u8]) -> Result<Vec<u8>, String> {
    let mut variable_and_payload = Vec::new();
    push_utf8(&mut variable_and_payload, topic)?;
    variable_and_payload.extend_from_slice(&packet_id.to_be_bytes());
    variable_and_payload.extend_from_slice(payload);

    let mut packet = vec![0x32]; // PUBLISH, QoS 1
    encode_remaining_length(variable_and_payload.len(), &mut packet)?;
    packet.extend_from_slice(&variable_and_payload);
    Ok(packet)
}

fn push_utf8(out: &mut Vec<u8>, text: &str) -> Result<(), String> {
    push_binary(out, text.as_bytes())
}

fn push_binary(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), String> {
    let len = u16::try_from(bytes.len()).map_err(|_| "MQTT string/payload field is too long".to_owned())?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn encode_remaining_length(mut length: usize, out: &mut Vec<u8>) -> Result<(), String> {
    if length > 268_435_455 {
        return Err("MQTT packet is too large".to_owned());
    }
    loop {
        let mut encoded = (length % 128) as u8;
        length /= 128;
        if length > 0 {
            encoded |= 0x80;
        }
        out.push(encoded);
        if length == 0 {
            return Ok(());
        }
    }
}

fn read_packet(stream: &mut TcpStream) -> Result<(u8, Vec<u8>), String> {
    let mut first = [0u8; 1];
    stream.read_exact(&mut first).map_err(|error| format!("MQTT read failed: {error}"))?;
    let packet_type = first[0] >> 4;
    let remaining = read_remaining_length(stream)?;
    let mut body = vec![0u8; remaining];
    stream.read_exact(&mut body).map_err(|error| format!("MQTT packet body read failed: {error}"))?;
    Ok((packet_type, body))
}

fn read_remaining_length(stream: &mut TcpStream) -> Result<usize, String> {
    let mut multiplier = 1usize;
    let mut value = 0usize;
    for _ in 0..4 {
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte).map_err(|error| format!("MQTT remaining-length read failed: {error}"))?;
        value = value
            .checked_add(((byte[0] & 0x7f) as usize).saturating_mul(multiplier))
            .ok_or_else(|| "MQTT remaining length overflow".to_owned())?;
        if byte[0] & 0x80 == 0 {
            return Ok(value);
        }
        multiplier = multiplier.saturating_mul(128);
    }
    Err("invalid MQTT remaining length".to_owned())
}
