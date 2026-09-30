use std::net::{TcpStream, ToSocketAddrs};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context as _, Result};

#[derive(Clone, Debug)]
pub struct CaptureNetworkRequest {
    /// NetworkManager connection id/name or UUID. The profile must already exist.
    pub connection: String,
    pub timeout: Duration,
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavedWifiProfile {
    pub name: String,
    pub uuid: String,
}

#[derive(Clone, Debug)]
pub struct ActiveWifiProfile {
    pub name: String,
    pub uuid: String,
    pub device: String,
}

#[derive(Debug)]
struct NetworkSwitchGuard {
    previous_uuid: Option<String>,
    target_uuid: String,
    switched: bool,
    timeout: Duration,
}

impl NetworkSwitchGuard {
    fn begin(request: &CaptureNetworkRequest, target_url: &str) -> Result<Self> {
        let target = request.connection.trim();
        if target.is_empty() {
            return Err(anyhow!("capture Wi-Fi connection is empty"));
        }

        let target_uuid = resolve_connection_uuid(target)
            .with_context(|| format!("cannot resolve NetworkManager connection {target:?}"))?;
        let previous = active_wifi_profile()?;

        // Compare canonical UUIDs, not SSIDs/names. This avoids disconnect/reconnect
        // when the requested capture profile is already active. We still wait for
        // the host to become reachable because a profile can be reported active
        // slightly before IPv4/routes/the camera service are usable.
        if previous.as_ref().is_some_and(|profile| profile.uuid == target_uuid) {
            wait_for_capture_network_ready(&target_uuid, target_url, request.timeout)
                .with_context(|| format!(
                    "capture Wi-Fi is already active, but camera host is not ready: {target_url}"
                ))?;
            return Ok(Self {
                previous_uuid: previous.map(|profile| profile.uuid),
                target_uuid,
                switched: false,
                timeout: request.timeout,
            });
        }

        let switch_started = Instant::now();
        if let Err(switch_error) = activate_connection_uuid(&target_uuid, request.timeout) {
            let previous_label = previous
                .as_ref()
                .map(|profile| format!("{} ({})", profile.name, profile.uuid))
                .unwrap_or_else(|| "no active Wi-Fi".to_owned());
            let restore_error = previous
                .as_ref()
                .and_then(|profile| activate_connection_uuid(&profile.uuid, request.timeout).err());
            return match restore_error {
                Some(restore_error) => Err(anyhow!(
                    "failed to switch from {previous_label} to capture Wi-Fi {target:?} ({target_uuid}): {switch_error:#}; restoring {previous_label} also failed: {restore_error:#}"
                )),
                None => Err(anyhow!(
                    "failed to switch from {previous_label} to capture Wi-Fi {target:?} ({target_uuid}): {switch_error:#}; previous Wi-Fi was restored"
                )),
            };
        }

        let remaining = request
            .timeout
            .checked_sub(switch_started.elapsed())
            .unwrap_or(Duration::ZERO);
        if let Err(readiness_error) = wait_for_capture_network_ready(&target_uuid, target_url, remaining) {
            let previous_label = previous
                .as_ref()
                .map(|profile| format!("{} ({})", profile.name, profile.uuid))
                .unwrap_or_else(|| "no active Wi-Fi".to_owned());
            let restore_error = previous
                .as_ref()
                .and_then(|profile| activate_connection_uuid(&profile.uuid, request.timeout).err());
            return match restore_error {
                Some(restore_error) => Err(anyhow!(
                    "capture Wi-Fi activated but camera host {target_url} did not become ready: {readiness_error:#}; restoring {previous_label} also failed: {restore_error:#}"
                )),
                None => Err(anyhow!(
                    "capture Wi-Fi activated but camera host {target_url} did not become ready: {readiness_error:#}; previous Wi-Fi was restored"
                )),
            };
        }

        Ok(Self {
            previous_uuid: previous.map(|profile| profile.uuid),
            target_uuid,
            switched: true,
            timeout: request.timeout,
        })
    }

    fn restore(&mut self) -> Result<()> {
        if !self.switched {
            return Ok(());
        }
        self.switched = false;
        let Some(previous_uuid) = self.previous_uuid.as_deref() else {
            // There was no Wi-Fi profile to restore before the capture attempt.
            return Ok(());
        };
        if previous_uuid == self.target_uuid {
            return Ok(());
        }
        activate_connection_uuid(previous_uuid, self.timeout).with_context(|| {
            format!("failed to restore previous Wi-Fi connection {previous_uuid}")
        })
    }
}

/// Run one capture attempt on the requested NetworkManager Wi-Fi profile and
/// restore the profile that was active before the attempt before returning.
/// This means retry cooldowns happen on the previous network automatically.
pub fn with_capture_network<T>(
    request: Option<&CaptureNetworkRequest>,
    target_url: &str,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let Some(request) = request else {
        return operation();
    };

    let mut guard = NetworkSwitchGuard::begin(request, target_url)?;
    let result = operation();
    let restore = guard.restore();

    match (result, restore) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(restore_error)) => Err(restore_error),
        (Err(error), Err(restore_error)) => Err(anyhow!(
            "{error:#}; additionally, restoring the previous Wi-Fi failed: {restore_error:#}"
        )),
    }
}

pub fn validate_connection(connection: &str) -> Result<String> {
    let connection = connection.trim();
    if connection.is_empty() {
        return Err(anyhow!("capture Wi-Fi connection is empty"));
    }
    resolve_connection_uuid(connection)
}

pub fn saved_wifi_profiles() -> Result<Vec<SavedWifiProfile>> {
    let text = nmcli_output(&[
        "-t",
        "-f",
        "UUID,TYPE",
        "connection",
        "show",
    ])?;

    let mut profiles = Vec::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let mut fields = line.splitn(2, ':');
        let uuid = fields.next().unwrap_or_default().trim();
        let kind = fields.next().unwrap_or_default().trim();
        if uuid.is_empty() || (kind != "802-11-wireless" && kind != "wifi") {
            continue;
        }
        let name = connection_id_for_uuid(uuid).unwrap_or_else(|_| uuid.to_owned());
        profiles.push(SavedWifiProfile {
            name,
            uuid: uuid.to_owned(),
        });
    }
    profiles.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.uuid.cmp(&b.uuid))
    });
    profiles.dedup_by(|a, b| a.uuid == b.uuid);
    Ok(profiles)
}

pub fn active_wifi_profile() -> Result<Option<ActiveWifiProfile>> {
    let text = nmcli_output(&[
        "-t",
        "-f",
        "UUID,TYPE,DEVICE",
        "connection",
        "show",
        "--active",
    ])?;

    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let mut fields = line.splitn(3, ':');
        let uuid = fields.next().unwrap_or_default().trim();
        let kind = fields.next().unwrap_or_default().trim();
        let device = fields.next().unwrap_or_default().trim();
        if uuid.is_empty() || device.is_empty() {
            continue;
        }
        if kind != "802-11-wireless" && kind != "wifi" {
            continue;
        }
        let name = connection_id_for_uuid(uuid).unwrap_or_else(|_| uuid.to_owned());
        return Ok(Some(ActiveWifiProfile {
            name,
            uuid: uuid.to_owned(),
            device: device.to_owned(),
        }));
    }
    Ok(None)
}


pub fn active_wifi_profile_for_uuid(target_uuid: &str) -> Result<Option<ActiveWifiProfile>> {
    let text = nmcli_output(&[
        "-t",
        "-f",
        "UUID,TYPE,DEVICE",
        "connection",
        "show",
        "--active",
    ])?;

    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let mut fields = line.splitn(3, ':');
        let uuid = fields.next().unwrap_or_default().trim();
        let kind = fields.next().unwrap_or_default().trim();
        let device = fields.next().unwrap_or_default().trim();
        if uuid != target_uuid || device.is_empty() {
            continue;
        }
        if kind != "802-11-wireless" && kind != "wifi" {
            continue;
        }
        let name = connection_id_for_uuid(uuid).unwrap_or_else(|_| uuid.to_owned());
        return Ok(Some(ActiveWifiProfile {
            name,
            uuid: uuid.to_owned(),
            device: device.to_owned(),
        }));
    }
    Ok(None)
}

fn resolve_connection_uuid(connection: &str) -> Result<String> {
    let text = nmcli_output(&[
        "-g",
        "connection.uuid",
        "connection",
        "show",
        connection,
    ])?;
    first_nonempty_line(&text).ok_or_else(|| anyhow!("NetworkManager returned an empty UUID"))
}

fn connection_id_for_uuid(uuid: &str) -> Result<String> {
    let text = nmcli_output(&[
        "-g",
        "connection.id",
        "connection",
        "show",
        uuid,
    ])?;
    first_nonempty_line(&text).ok_or_else(|| anyhow!("NetworkManager returned an empty connection id"))
}

fn activate_connection_uuid(uuid: &str, timeout: Duration) -> Result<()> {
    let wait_seconds = timeout.as_secs().max(1).min(600).to_string();
    nmcli_status(&[
        "--wait",
        &wait_seconds,
        "connection",
        "up",
        "uuid",
        uuid,
    ])
}


fn wait_for_capture_network_ready(target_uuid: &str, target_url: &str, timeout: Duration) -> Result<()> {
    if timeout.is_zero() {
        return Err(anyhow!("network readiness timeout expired immediately"));
    }

    let parsed = reqwest::Url::parse(target_url)
        .with_context(|| format!("cannot parse capture URL {target_url}"))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow!("capture URL has no host: {target_url}"))?
        .to_owned();
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| anyhow!("capture URL has no usable port: {target_url}"))?;

    let deadline = Instant::now() + timeout;
    let mut last_reason = "waiting for NetworkManager".to_owned();

    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(anyhow!(
                "timed out after {:.1}s waiting for capture network/camera readiness ({last_reason})",
                timeout.as_secs_f32()
            ));
        }

        match active_wifi_profile_for_uuid(target_uuid) {
            Ok(Some(active)) => match device_has_ipv4(&active.device) {
                Ok(true) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    let connect_timeout = remaining.min(Duration::from_millis(750));
                    match tcp_host_ready(&host, port, connect_timeout) {
                        Ok(()) => return Ok(()),
                        Err(error) => {
                            last_reason = format!(
                                "{} has IPv4, but {host}:{port} is not reachable yet: {error:#}",
                                active.device
                            );
                        }
                    }
                }
                Ok(false) => {
                    last_reason = format!(
                        "target Wi-Fi is active on {}, waiting for IPv4",
                        active.device
                    );
                }
                Err(error) => {
                    last_reason = format!(
                        "target Wi-Fi is active on {}, IPv4 check failed: {error:#}",
                        active.device
                    );
                }
            },
            Ok(None) => {
                last_reason = match active_wifi_profile() {
                    Ok(Some(active)) => format!(
                        "waiting for target UUID {target_uuid}; another active Wi-Fi is {} ({})",
                        active.name, active.uuid
                    ),
                    Ok(None) => "waiting for target Wi-Fi profile to become active".to_owned(),
                    Err(error) => format!("cannot query active Wi-Fi yet: {error:#}"),
                };
            }
            Err(error) => {
                last_reason = format!("cannot query target Wi-Fi yet: {error:#}");
            }
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        thread::sleep(remaining.min(Duration::from_millis(250)));
    }
}

fn device_has_ipv4(device: &str) -> Result<bool> {
    let text = nmcli_output(&["-g", "IP4.ADDRESS", "device", "show", device])?;
    Ok(text.lines().any(|line| {
        let line = line.trim();
        !line.is_empty() && !line.starts_with("127.")
    }))
}

fn tcp_host_ready(host: &str, port: u16, timeout: Duration) -> Result<()> {
    if timeout.is_zero() {
        return Err(anyhow!("no time left for host readiness check"));
    }

    let addresses = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("cannot resolve {host}:{port}"))?
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        return Err(anyhow!("{host}:{port} resolved to no addresses"));
    }

    let per_address_timeout = timeout.min(Duration::from_millis(500));
    let mut last_error = None;
    for address in addresses {
        match TcpStream::connect_timeout(&address, per_address_timeout) {
            Ok(stream) => {
                let _ = stream.shutdown(std::net::Shutdown::Both);
                return Ok(());
            }
            Err(error) => last_error = Some((address, error)),
        }
    }

    match last_error {
        Some((address, error)) => Err(anyhow!("TCP connect to {address} failed: {error}")),
        None => Err(anyhow!("camera host is not reachable")),
    }
}

fn first_nonempty_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

fn nmcli_output(args: &[&str]) -> Result<String> {
    let output = Command::new("nmcli")
        .args(args)
        .output()
        .context("failed to execute nmcli; NetworkManager/nmcli is required for Wi-Fi switching")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(anyhow!(
            "nmcli {} failed ({}): {}",
            args.join(" "),
            output.status,
            if stderr.is_empty() { "no error text" } else { &stderr }
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn nmcli_status(args: &[&str]) -> Result<()> {
    nmcli_output(args).map(|_| ())
}
