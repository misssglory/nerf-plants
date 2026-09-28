use eframe::egui;
use image::RgbaImage;

pub const MIN_ZOOM: f32 = 0.02;
pub const MAX_ZOOM: f32 = 32.0;
const MIN_VISIBLE_EDGE: f32 = 48.0;

#[derive(Clone, Debug)]
pub struct ImageViewState {
    pub open: bool,
    pub zoom: f32,
    pub pan: egui::Vec2,
    pub fit_to_window: bool,
    pub window_pos: Option<egui::Pos2>,
    pub window_size: Option<egui::Vec2>,
}

impl Default for ImageViewState {
    fn default() -> Self {
        Self {
            open: true,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            fit_to_window: true,
            window_pos: None,
            window_size: None,
        }
    }
}

impl ImageViewState {
    pub fn reset_fit(&mut self) {
        self.fit_to_window = true;
        self.pan = egui::Vec2::ZERO;
    }

    pub fn center(&mut self) {
        self.fit_to_window = false;
        self.pan = egui::Vec2::ZERO;
    }

    pub fn one_to_one(&mut self) {
        self.fit_to_window = false;
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
    }
}

#[derive(Clone, Debug)]
pub struct OverlayPoint {
    pub pixel: (u32, u32),
    pub color: egui::Color32,
    pub label: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ViewInteraction {
    pub pivot_pixel: Option<(u32, u32)>,
}

pub fn show_floating_image_window(
    ctx: &egui::Context,
    title: &str,
    texture: Option<&egui::TextureHandle>,
    pixel_source: Option<&RgbaImage>,
    overlays: &[OverlayPoint],
    allow_pivot_hotkey: bool,
    state: &mut ImageViewState,
    default_pos: egui::Pos2,
) -> ViewInteraction {
    if !state.open {
        return ViewInteraction::default();
    }

    let mut open = state.open;
    let mut interaction = ViewInteraction::default();

    let mut window = egui::Window::new(title)
        .id(egui::Id::new(("image-window", title)))
        .open(&mut open)
        .default_pos(state.window_pos.unwrap_or(default_pos))
        .default_size(state.window_size.unwrap_or_else(|| egui::vec2(620.0, 620.0)))
        .min_size(egui::vec2(300.0, 240.0))
        // Preserve exact persisted floating-window coordinates. Egui's default
        // screen constraint may otherwise clamp restored positions during the
        // first frame while image contents are still changing size.
        .constrain(false)
        .resizable(true);
    if let Some(pos) = state.window_pos {
        window = window.current_pos(pos);
    }
    if let Some(response) = window.show(ctx, |ui| {
        interaction = show_image_window_contents(
            ui,
            texture,
            pixel_source,
            overlays,
            allow_pivot_hotkey,
            state,
        );
    }) {
        state.window_pos = Some(response.response.rect.min);
        state.window_size = Some(response.response.rect.size());
    }

    state.open = open;
    interaction
}

fn show_image_window_contents(
    ui: &mut egui::Ui,
    texture: Option<&egui::TextureHandle>,
    pixel_source: Option<&RgbaImage>,
    overlays: &[OverlayPoint],
    allow_pivot_hotkey: bool,
    state: &mut ImageViewState,
) -> ViewInteraction {
    let mut interaction = ViewInteraction::default();

    ui.horizontal_wrapped(|ui| {
        if ui.selectable_label(state.fit_to_window, "Fit").clicked() {
            state.reset_fit();
        }
        if ui.button("1:1").clicked() {
            state.one_to_one();
        }
        if ui.button("Center").clicked() {
            state.center();
        }
        if ui.small_button("−").clicked() {
            state.fit_to_window = false;
            state.zoom = (state.zoom / 1.25).clamp(MIN_ZOOM, MAX_ZOOM);
        }

        if ui
            .add(
                egui::Slider::new(&mut state.zoom, MIN_ZOOM..=MAX_ZOOM)
                    .logarithmic(true)
                    .show_value(false)
                    .text("zoom"),
            )
            .changed()
        {
            state.fit_to_window = false;
        }

        if ui.small_button("+").clicked() {
            state.fit_to_window = false;
            state.zoom = (state.zoom * 1.25).clamp(MIN_ZOOM, MAX_ZOOM);
        }

        ui.monospace(format!("{:>5.0}%", state.zoom * 100.0));
    });

    if allow_pivot_hotkey {
        ui.small("Drag = move · two-finger scroll = pan · pinch/Ctrl+wheel = zoom · P = track mask shape under cursor");
    } else {
        ui.small("Drag = move · two-finger scroll = pan · pinch/Ctrl+wheel = zoom");
    }
    ui.separator();

    let viewport_size = finite_available_size(ui, egui::vec2(420.0, 320.0));
    let (response, painter) = ui.allocate_painter(viewport_size, egui::Sense::click_and_drag());
    let viewport = response.rect;
    let painter = painter.with_clip_rect(viewport);
    painter.rect_filled(viewport, 0.0, egui::Color32::BLACK);

    let Some(texture) = texture else {
        painter.text(
            viewport.center(),
            egui::Align2::CENTER_CENTER,
            "No image loaded",
            egui::FontId::monospace(15.0),
            egui::Color32::GRAY,
        );
        return interaction;
    };

    let image_size = texture.size_vec2();
    if image_size.x <= 0.0 || image_size.y <= 0.0 {
        return interaction;
    }

    if state.fit_to_window {
        state.zoom = (viewport.width() / image_size.x)
            .min(viewport.height() / image_size.y)
            .clamp(MIN_ZOOM, MAX_ZOOM);
        state.pan = egui::Vec2::ZERO;
    }

    // Every window owns its own ImageViewState. Gestures are applied only to the
    // image viewport currently under the pointer; no transforms are mirrored.
    if response.dragged() {
        let delta = ui.input(|input| input.pointer.delta());
        if delta != egui::Vec2::ZERO {
            state.fit_to_window = false;
            state.pan += delta;
        }
    }

    if response.contains_pointer() {
        let (zoom_delta, translation_delta, hover_pos) = ui.input(|input| {
            (
                input.zoom_delta(),
                input.translation_delta(),
                input.pointer.hover_pos(),
            )
        });

        if (zoom_delta - 1.0).abs() > 0.001 {
            let anchor = hover_pos
                .filter(|pos| viewport.contains(*pos))
                .unwrap_or_else(|| viewport.center());
            zoom_about_point(state, image_size, viewport, anchor, zoom_delta);
        } else if translation_delta.length_sq() > 0.01 && !response.dragged() {
            state.fit_to_window = false;
            state.pan += translation_delta;
        }
    }

    clamp_pan(state, image_size, viewport.size());

    let display_size = image_size * state.zoom;
    let image_center = viewport.center() + state.pan;
    let image_rect = egui::Rect::from_center_size(image_center, display_size);

    painter.image(
        texture.id(),
        image_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );

    for point in overlays {
        let px = point.pixel.0 as f32 + 0.5;
        let py = point.pixel.1 as f32 + 0.5;
        if px >= 0.0 && py >= 0.0 && px < image_size.x && py < image_size.y {
            let screen = egui::pos2(
                image_rect.min.x + px / image_size.x * image_rect.width(),
                image_rect.min.y + py / image_size.y * image_rect.height(),
            );
            painter.circle_filled(screen, 5.0, point.color);
            painter.circle_stroke(screen, 8.0, egui::Stroke::new(1.5, point.color));
            painter.text(
                screen + egui::vec2(10.0, -10.0),
                egui::Align2::LEFT_BOTTOM,
                &point.label,
                egui::FontId::monospace(11.0),
                point.color,
            );
        }
    }

    if response.contains_pointer() {
        if let Some(pointer) = response.hover_pos() {
            if image_rect.contains(pointer) {
            let u = ((pointer.x - image_rect.min.x) / image_rect.width()).clamp(0.0, 0.999_999);
            let v = ((pointer.y - image_rect.min.y) / image_rect.height()).clamp(0.0, 0.999_999);
            let x = (u * image_size.x).floor() as u32;
            let y = (v * image_size.y).floor() as u32;
            if allow_pivot_hotkey && ui.input(|input| input.key_pressed(egui::Key::P)) {
                interaction.pivot_pixel = Some((x, y));
            }

            if let Some(source) = pixel_source {
                if x < source.width() && y < source.height() {
                    let pixel = source.get_pixel(x, y).0;
                    let text = format!(
                        "x:{x} y:{y}  RGBA {}, {}, {}, {}  #{:02X}{:02X}{:02X}{:02X}",
                        pixel[0], pixel[1], pixel[2], pixel[3], pixel[0], pixel[1], pixel[2], pixel[3]
                    );

                    let mut label_pos = pointer + egui::vec2(12.0, 12.0);
                    let label_size = egui::vec2(350.0, 24.0);
                    if label_pos.x + label_size.x > viewport.right() {
                        label_pos.x = pointer.x - label_size.x - 12.0;
                    }
                    if label_pos.y + label_size.y > viewport.bottom() {
                        label_pos.y = pointer.y - label_size.y - 12.0;
                    }
                    let label_rect = egui::Rect::from_min_size(label_pos, label_size);
                    painter.rect_filled(
                        label_rect,
                        3.0,
                        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 220),
                    );
                    painter.text(
                        label_rect.min + egui::vec2(6.0, 5.0),
                        egui::Align2::LEFT_TOP,
                        text,
                        egui::FontId::monospace(12.0),
                        egui::Color32::WHITE,
                    );
                }
            }
            }
        }
    }

    interaction
}

fn finite_available_size(ui: &egui::Ui, fallback: egui::Vec2) -> egui::Vec2 {
    let available = ui.available_size();
    egui::vec2(
        if available.x.is_finite() {
            available.x.max(120.0)
        } else {
            fallback.x
        },
        if available.y.is_finite() {
            available.y.max(120.0)
        } else {
            fallback.y
        },
    )
}

fn zoom_about_point(
    state: &mut ImageViewState,
    image_size: egui::Vec2,
    viewport: egui::Rect,
    anchor: egui::Pos2,
    zoom_delta: f32,
) {
    let old_zoom = state.zoom.max(MIN_ZOOM);
    let new_zoom = (old_zoom * zoom_delta).clamp(MIN_ZOOM, MAX_ZOOM);
    if (new_zoom - old_zoom).abs() <= f32::EPSILON {
        return;
    }

    let old_center = viewport.center() + state.pan;
    let anchor_from_center = anchor - old_center;
    let image_space_offset = anchor_from_center / old_zoom;
    let new_center = anchor - image_space_offset * new_zoom;

    state.fit_to_window = false;
    state.zoom = new_zoom;
    state.pan = new_center - viewport.center();
    clamp_pan(state, image_size, viewport.size());
}

fn clamp_pan(state: &mut ImageViewState, image_size: egui::Vec2, viewport_size: egui::Vec2) {
    let displayed = image_size * state.zoom;
    let max_x = ((displayed.x + viewport_size.x) * 0.5 - MIN_VISIBLE_EDGE).max(0.0);
    let max_y = ((displayed.y + viewport_size.y) * 0.5 - MIN_VISIBLE_EDGE).max(0.0);
    state.pan.x = state.pan.x.clamp(-max_x, max_x);
    state.pan.y = state.pan.y.clamp(-max_y, max_y);
}
