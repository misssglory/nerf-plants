use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Result};
use eframe::egui;
use image::{GrayImage, RgbaImage};

#[derive(Clone)]
struct CachedFrame {
    width: u32,
    height: u32,
    raw_bytes: usize,
    compressed_rgba: Arc<Vec<u8>>,
    compressed_gray: Arc<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameCacheStats {
    pub frames: usize,
    pub raw_bytes: usize,
    pub compressed_bytes: usize,
    pub hits: u64,
    pub misses: u64,
}

impl FrameCacheStats {
    pub fn compression_ratio(self) -> f64 {
        if self.compressed_bytes == 0 {
            0.0
        } else {
            self.raw_bytes as f64 / self.compressed_bytes as f64
        }
    }

    pub fn savings_percent(self) -> f64 {
        if self.raw_bytes == 0 {
            0.0
        } else {
            (1.0 - self.compressed_bytes as f64 / self.raw_bytes as f64) * 100.0
        }
    }

    pub fn hit_percent(self) -> f64 {
        let total = self.hits.saturating_add(self.misses);
        if total == 0 {
            0.0
        } else {
            self.hits as f64 * 100.0 / total as f64
        }
    }
}

#[derive(Default)]
struct CacheInner {
    frames: HashMap<PathBuf, CachedFrame>,
    raw_bytes: usize,
    compressed_bytes: usize,
    hits: u64,
    misses: u64,
}

#[derive(Clone, Default)]
pub struct FrameMemoryCache {
    inner: Arc<Mutex<CacheInner>>,
}

pub struct DecodedFrame {
    pub rgba: RgbaImage,
    pub gray: GrayImage,
}

impl FrameMemoryCache {
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.frames.clear();
            inner.raw_bytes = 0;
            inner.compressed_bytes = 0;
            inner.hits = 0;
            inner.misses = 0;
        }
    }

    pub fn remove(&self, path: &Path) {
        if let Ok(mut inner) = self.inner.lock() {
            if let Some(old) = inner.frames.remove(path) {
                inner.raw_bytes = inner.raw_bytes.saturating_sub(old.raw_bytes);
                inner.compressed_bytes = inner
                    .compressed_bytes
                    .saturating_sub(old.compressed_rgba.len().saturating_add(old.compressed_gray.len()));
            }
        }
    }

    pub fn contains(&self, path: &Path) -> bool {
        self.inner
            .lock()
            .map(|inner| inner.frames.contains_key(path))
            .unwrap_or(false)
    }

    pub fn stats(&self) -> FrameCacheStats {
        self.inner
            .lock()
            .map(|inner| FrameCacheStats {
                frames: inner.frames.len(),
                raw_bytes: inner.raw_bytes,
                compressed_bytes: inner.compressed_bytes,
                hits: inner.hits,
                misses: inner.misses,
            })
            .unwrap_or_default()
    }

    pub fn get(&self, path: &Path) -> Result<Option<DecodedFrame>> {
        let cached = {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| anyhow!("frame cache lock was poisoned"))?;
            if let Some(frame) = inner.frames.get(path).cloned() {
                inner.hits = inner.hits.saturating_add(1);
                Some(frame)
            } else {
                inner.misses = inner.misses.saturating_add(1);
                None
            }
        };

        let Some(cached) = cached else {
            return Ok(None);
        };

        let rgba_raw = lz4_flex::decompress_size_prepended(cached.compressed_rgba.as_slice())
            .map_err(|error| anyhow!("failed to decompress cached RGBA LZ4 frame: {error}"))?;
        let gray_raw = lz4_flex::decompress_size_prepended(cached.compressed_gray.as_slice())
            .map_err(|error| anyhow!("failed to decompress cached grayscale LZ4 frame: {error}"))?;
        let rgba_len = cached.width as usize * cached.height as usize * 4;
        let gray_len = cached.width as usize * cached.height as usize;
        if rgba_raw.len() != rgba_len || gray_raw.len() != gray_len {
            return Err(anyhow!(
                "cached LZ4 frame size mismatch: expected RGBA {rgba_len} + gray {gray_len}, got {} + {}",
                rgba_raw.len(),
                gray_raw.len()
            ));
        }
        if rgba_raw.len().saturating_add(gray_raw.len()) != cached.raw_bytes {
            return Err(anyhow!("cached frame payload has invalid decoded size"));
        }
        let rgba = RgbaImage::from_raw(cached.width, cached.height, rgba_raw)
            .ok_or_else(|| anyhow!("failed to reconstruct cached RGBA frame"))?;
        let gray = GrayImage::from_raw(cached.width, cached.height, gray_raw)
            .ok_or_else(|| anyhow!("failed to reconstruct cached grayscale frame"))?;
        Ok(Some(DecodedFrame { rgba, gray }))
    }

    pub fn insert(&self, path: PathBuf, rgba: &RgbaImage, gray: &GrayImage) -> Result<()> {
        if rgba.width() != gray.width() || rgba.height() != gray.height() {
            return Err(anyhow!("RGBA/gray dimensions differ while caching frame"));
        }

        let raw_bytes = rgba.as_raw().len().saturating_add(gray.as_raw().len());
        // Keep RGBA and grayscale in separate LZ4 payloads. Playback can then
        // decompress directly into the Vec consumed by ImageBuffer::from_raw,
        // avoiding a second full-frame memcpy after decompression.
        let compressed_rgba = Arc::new(lz4_flex::compress_prepend_size(rgba.as_raw()));
        let compressed_gray = Arc::new(lz4_flex::compress_prepend_size(gray.as_raw()));
        let frame = CachedFrame {
            width: rgba.width(),
            height: rgba.height(),
            raw_bytes,
            compressed_rgba,
            compressed_gray,
        };

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow!("frame cache lock was poisoned"))?;
        if let Some(old) = inner.frames.insert(path, frame.clone()) {
            inner.raw_bytes = inner.raw_bytes.saturating_sub(old.raw_bytes);
            inner.compressed_bytes = inner
                .compressed_bytes
                .saturating_sub(old.compressed_rgba.len().saturating_add(old.compressed_gray.len()));
        }
        inner.raw_bytes = inner.raw_bytes.saturating_add(frame.raw_bytes);
        inner.compressed_bytes = inner
            .compressed_bytes
            .saturating_add(frame.compressed_rgba.len().saturating_add(frame.compressed_gray.len()));
        Ok(())
    }


}

pub fn spawn_sequence_prefetch(
    paths: Vec<PathBuf>,
    cache: FrameMemoryCache,
    generation: Arc<AtomicU64>,
    expected_generation: u64,
    repaint_ctx: egui::Context,
) {
    let _ = thread::Builder::new()
        .name("sequence-lz4-prefetch".to_owned())
        .spawn(move || {
            for path in paths {
                if generation.load(Ordering::Acquire) != expected_generation {
                    break;
                }
                if !cache.contains(&path) {
                    if let Ok(decoded) = image::open(&path) {
                        let rgba = decoded.to_rgba8();
                        let gray = decoded.to_luma8();
                        // Opening a different sequence invalidates the generation. Check
                        // again after the expensive image decode so an old prefetch cannot
                        // repopulate a freshly-cleared cache with a stale frame.
                        if generation.load(Ordering::Acquire) != expected_generation {
                            break;
                        }
                        let _ = cache.insert(path.clone(), &rgba, &gray);
                    }
                }
                repaint_ctx.request_repaint();
                // Avoid monopolizing a CPU core while a long sequence is being warmed.
                thread::sleep(Duration::from_millis(1));
            }
        });
}
