use eframe::egui;

pub const MIN_ZOOM: f32 = 0.02;
pub const MAX_ZOOM: f32 = 32.0;
const MIN_VISIBLE_EDGE: f32 = 48.0;

#[derive(Clone, Debug)]
pub struct ImageViewState {
    pub open: bool,
    pub zoom: f32,
    pub pan: egui::Vec2,
    pub fit_to_window: bool,
}

impl Default for ImageViewState {
    fn default() -> Self {
        Self {
            open: true,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            fit_to_window: true,
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

    pub fn copy_transform_from(&mut self, other: &Self) {
        self.zoom = other.zoom;
        self.pan = other.pan;
        self.fit_to_window = other.fit_to_window;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ViewInteraction {
    pub transform_changed: bool,
}

pub fn show_floating_image_window(
    ctx: &egui::Context,
    title: &str,
    texture: Option<&egui::TextureHandle>,
    state: &mut ImageViewState,
    default_pos: egui::Pos2,
) -> ViewInteraction {
    if !state.open {
        return ViewInteraction::default();
    }

    let mut open = state.open;
    let mut interaction = ViewInteraction::default();

    egui::Window::new(title)
        .open(&mut open)
        .default_pos(default_pos)
        .default_size(egui::vec2(620.0, 620.0))
        .min_size(egui::vec2(300.0, 240.0))
        .resizable(true)
        .show(ctx, |ui| {
            interaction = show_image_window_contents(ui, texture, state);
        });

    state.open = open;
    interaction
}

fn show_image_window_contents(
    ui: &mut egui::Ui,
    texture: Option<&egui::TextureHandle>,
    state: &mut ImageViewState,
) -> ViewInteraction {
    let mut interaction = ViewInteraction::default();

    ui.horizontal_wrapped(|ui| {
        if ui.selectable_label(state.fit_to_window, "Fit").clicked() {
            state.reset_fit();
            interaction.transform_changed = true;
        }
        if ui.button("1:1").clicked() {
            state.one_to_one();
            interaction.transform_changed = true;
        }
        if ui.button("Center").clicked() {
            state.center();
            interaction.transform_changed = true;
        }
        if ui.small_button("−").clicked() {
            state.fit_to_window = false;
            state.zoom = (state.zoom / 1.25).clamp(MIN_ZOOM, MAX_ZOOM);
            interaction.transform_changed = true;
        }

        let before = state.zoom;
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
            interaction.transform_changed = true;
        }

        if ui.small_button("+").clicked() {
            state.fit_to_window = false;
            state.zoom = (state.zoom * 1.25).clamp(MIN_ZOOM, MAX_ZOOM);
            interaction.transform_changed = true;
        }

        if (state.zoom - before).abs() > f32::EPSILON {
            interaction.transform_changed = true;
        }
        ui.monospace(format!("{:>5.0}%", state.zoom * 100.0));
    });

    ui.small("Drag = move · two-finger scroll = pan · pinch / Ctrl+wheel = zoom");
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
            egui::FontId::proportional(15.0),
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

    if response.dragged() {
        let delta = ui.input(|input| input.pointer.delta());
        if delta != egui::Vec2::ZERO {
            state.fit_to_window = false;
            state.pan += delta;
            interaction.transform_changed = true;
        }
    }

    if response.hovered() {
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
            interaction.transform_changed = true;
        } else if translation_delta.length_sq() > 0.01 && !response.dragged() {
            state.fit_to_window = false;
            state.pan += translation_delta;
            interaction.transform_changed = true;
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
