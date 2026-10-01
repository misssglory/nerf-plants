use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct TelegramConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub bot_token: String,
    #[serde(default)]
    pub chat_id: String,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bot_token: String::new(),
            chat_id: String::new(),
        }
    }
}


#[derive(Default, Deserialize)]
struct AppConfig {
    #[serde(default)]
    telegram: TelegramConfig,
}

pub fn load_telegram_config() -> (TelegramConfig, String) {
    let path = std::env::var_os("RUST_EDGE_GUI_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config.toml"));

    if !path.is_file() {
        return (
            TelegramConfig::default(),
            format!("Telegram disabled ({} not found)", path.display()),
        );
    }

    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            return (
                TelegramConfig::default(),
                format!("Telegram config read error ({}): {error}", path.display()),
            );
        }
    };

    let parsed: AppConfig = match toml::from_str(&text) {
        Ok(config) => config,
        Err(error) => {
            return (
                TelegramConfig::default(),
                format!("Telegram config parse error ({}): {error}", path.display()),
            );
        }
    };

    let mut config = parsed.telegram;
    if config.enabled && (config.bot_token.trim().is_empty() || config.chat_id.trim().is_empty()) {
        config.enabled = false;
        return (
            config,
            format!("Telegram disabled: bot_token/chat_id missing in {}", path.display()),
        );
    }

    let status = if config.enabled {
        format!("Telegram enabled · {}", path.display())
    } else {
        format!("Telegram disabled · {}", path.display())
    };
    (config, status)
}

pub struct TelegramNotifier {
    tx: Option<mpsc::Sender<String>>,
}

impl TelegramNotifier {
    pub fn spawn(config: &TelegramConfig) -> Self {
        if !config.enabled {
            return Self { tx: None };
        }

        let (tx, rx) = mpsc::channel::<String>();
        let token = config.bot_token.clone();
        let chat_id = config.chat_id.clone();
        thread::Builder::new()
            .name("telegram-notifier".to_owned())
            .spawn(move || {
                let client = match reqwest::blocking::Client::builder()
                    .connect_timeout(Duration::from_secs(5))
                    .timeout(Duration::from_secs(15))
                    .user_agent("rust-edge-gui/0.8.9")
                    .build()
                {
                    Ok(client) => client,
                    Err(error) => {
                        eprintln!("Telegram client init failed: {error}");
                        return;
                    }
                };
                let url = format!("https://api.telegram.org/bot{token}/sendMessage");
                while let Ok(text) = rx.recv() {
                    let payload = serde_json::json!({
                        "chat_id": chat_id.clone(),
                        "text": text,
                        "disable_web_page_preview": true
                    });
                    match client.post(&url).json(&payload).send() {
                        Ok(response) => {
                            if let Err(error) = response.error_for_status() {
                                eprintln!("Telegram notification failed: {error}");
                            }
                        }
                        Err(error) => eprintln!("Telegram notification failed: {error}"),
                    }
                }
            })
            .expect("failed to spawn telegram notifier");

        Self { tx: Some(tx) }
    }

    pub fn send(&self, text: impl Into<String>) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(text.into());
        }
    }
}
