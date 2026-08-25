//! GTK4 + libadwaita UI shell for Darkroom.
//!
//! Phase 3-ui-12: live exposure preview in the darkroom view via c41-core.

use adw::prelude::*;
use adw::Application;
use anyhow::Result;
use gtk4::ApplicationWindow;
use glib::clone;

pub mod catalog;
pub mod crop_overlay;
pub mod bauhaus;
pub mod darkroom;
pub mod theme;
pub mod dialogs;
pub mod export;
pub mod export_panel;
pub mod history;
pub mod lighttable;
pub mod panels;
pub mod persist;
pub mod preview;
pub mod raw_preview;
pub mod snapshots;

pub const APP_ID:        &str = "org.c41.C41";
pub const DEFAULT_WIDTH:  i32 = 1280;
pub const DEFAULT_HEIGHT: i32 = 800;

/// Lighttable thumb-size control (bottom toolbar, m4-98a): the value is the grid's
/// *upper* column bound — fewer columns ⇒ larger thumbnails. It caps columns rather
/// than fixing them (`min_columns` stays low) so a narrow framebuffer still fits
/// the row instead of clipping.
const THUMB_COLS_MIN:     u32 = 2;
const THUMB_COLS_MAX:     u32 = 12;

/// `darkroom_ui_prefs` key under which the lighttable rating filter (comparator +
/// floor) is persisted across sessions (m4-98d). The value is the compact token
/// from [`lighttable::rating_filter_token`] (`off` / `ge:N` / `eq:N` / `le:N` /
/// `rej`).
const RATING_FILTER_PREF_KEY: &str = "rating_filter";

/// `darkroom_ui_prefs` key under which the thumbnail overlay mode is persisted
/// across sessions (m4-98e). The value is the token from
/// [`lighttable::overlay_mode_token`] (`none` / `normal` / `extended`).
const OVERLAY_MODE_PREF_KEY: &str = "overlay_mode";

/// `darkroom_ui_prefs` key under which the lighttable view mode is persisted across
/// sessions (m4-98c). The value is the token from
/// [`lighttable::view_mode_token_for`] (`filemanager` / `zoomable` / `culling`).
const VIEW_MODE_PREF_KEY: &str = "view_mode";

/// `darkroom_ui_prefs` keys for the draggable side-panel widths (parity audit
/// 1.1). Stored as the panels' **widths in pixels**, not as the `Paned` handle
/// positions: the right panel's handle position is measured from the left of the
/// centre area, so it means a different width in a different window size, and
/// restoring it into a resized window would move the panel.
const LEFT_PANEL_WIDTH_PREF_KEY: &str = "left_panel_width";
const RIGHT_PANEL_WIDTH_PREF_KEY: &str = "right_panel_width";

/// `darkroom_ui_prefs` keys under which each side panel's **collapsed** state is
/// persisted (parity audit 1.2). darktable collapses either panel with the
/// triangles at the screen edges and remembers the choice; a panel the user hid to
/// get screen back should stay hidden next session rather than coming back
/// uninvited.
const LEFT_PANEL_COLLAPSED_PREF_KEY: &str = "left_panel_collapsed";
const RIGHT_PANEL_COLLAPSED_PREF_KEY: &str = "right_panel_collapsed";

/// Bounds a restored panel width is clamped into. The lower bound keeps a corrupt
/// or hostile pref from restoring a panel too narrow to grab and drag back out;
/// the upper bound keeps one from eating the whole window.
const PANEL_WIDTH_MIN: i32 = 140;
const PANEL_WIDTH_MAX: i32 = 700;

/// Width to expand a panel to when there is no remembered one — a panel collapsed
/// across a restart before it was ever dragged. Matches the panels' own
/// `width_request` so the first expand looks like the default layout.
const PANEL_WIDTH_DEFAULT: i32 = 210;

/// How long to wait after the last drag before writing a panel width. `position`
/// notifies on every pixel of a drag, and each write is an SQLite transaction on
/// the GTK main thread.
const PANEL_WIDTH_SAVE_DEBOUNCE_MS: u32 = 400;

pub fn run() -> Result<glib::ExitCode> {
    let app = Application::builder()
        .application_id(APP_ID)
        .build();
    // Register app-level keyboard accelerators
    app.set_accels_for_action("win.import", &["<Control>i"]);
    app.set_accels_for_action("win.export-selected", &["<Control>e"]);
    app.connect_activate(build_main_window);

    // SIGTERM/SIGINT graceful shutdown. A GtkApplication installs no handler,
    // so the default disposition terminates the process instantly — the
    // container autostart trap that forwards SIGTERM and waits up to 15s would
    // be waiting on an already-dead process, dropping any pending autosave. Call
    // app.quit() so the main loop unwinds through the normal teardown path.
    // 15 = SIGTERM, 2 = SIGINT (avoids pulling in libc for two constants).
    for signum in [15, 2] {
        let app = app.downgrade();
        glib::unix_signal_add_local(signum, move || {
            if let Some(app) = app.upgrade() {
                app.quit();
            }
            glib::ControlFlow::Break
        });
    }

    Ok(app.run())
}

/// Build the darktable-style `Lighttable | Darkroom | Other` linked view switcher.
/// Returns the container plus the Lighttable and Darkroom toggles so the caller
/// can set the active view and wire navigation. "Other" (map/print/tethering) is
/// always disabled — those views aren't ported. Shared by the lighttable header
/// (Lighttable active; Darkroom pushes the selected image) and the darkroom
/// header (Darkroom active; Lighttable pops back to the grid), so the control
/// looks and behaves identically in both views.
///
/// Hand-rolled linked ToggleButton group rather than adw::ViewSwitcher: the
/// latter binds to an adw::ViewStack, but our views are a NavigationView
/// (push/pop), so this is the right fit — don't "upgrade" it to a ViewSwitcher
/// without also changing the navigation model.
pub(crate) struct ViewSwitcher {
    pub container: gtk4::Box,
    pub lighttable: gtk4::ToggleButton,
    pub darkroom: gtk4::ToggleButton,
}

pub(crate) fn build_view_switcher() -> ViewSwitcher {
    let container = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
    container.add_css_class("linked");

    let lighttable = gtk4::ToggleButton::with_label("Lighttable");
    let darkroom = gtk4::ToggleButton::with_label("Darkroom");
    darkroom.set_group(Some(&lighttable));
    // "Other" (map / print / tethering) is a disabled placeholder — those views
    // aren't ported. Insensitive so it can't break the toggle-group's
    // single-active invariant; the greyed appearance signals "unavailable". (No
    // tooltip: an insensitive widget never emits query-tooltip, so one here would
    // never show — the greying is the affordance.)
    let other = gtk4::ToggleButton::with_label("Other");
    other.set_group(Some(&lighttable));
    other.set_sensitive(false);

    container.append(&lighttable);
    container.append(&darkroom);
    container.append(&other);
    ViewSwitcher { container, lighttable, darkroom }
}

/// Wrap a view-switcher container in the standard header title: the switcher on
/// top with a dim caption below it (the app name in the lighttable, the open
/// filename in the darkroom). Shared so both view headers are the same height and
/// layout — the switcher is identical and always sits over a one-line caption.
pub(crate) fn view_switcher_title(container: &gtk4::Box, subtitle: &str) -> gtk4::Box {
    container.set_halign(gtk4::Align::Center);
    let subtitle_label = gtk4::Label::new(Some(subtitle));
    subtitle_label.add_css_class("caption");
    subtitle_label.add_css_class("dim-label");
    subtitle_label.set_ellipsize(gtk4::pango::EllipsizeMode::End);
    subtitle_label.set_max_width_chars(30);
    let title_box = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    title_box.set_valign(gtk4::Align::Center);
    title_box.set_halign(gtk4::Align::Center);
    title_box.append(container);
    title_box.append(&subtitle_label);
    title_box
}

/// Encode a panel's collapsed state for `darkroom_ui_prefs`.
///
/// Words rather than `1`/`0` so the row means something in a sqlite shell, and
/// decoded strictly by [`parse_collapsed_token`]: anything unrecognised is *no
/// opinion*, which leaves the panel **shown**. A corrupt or hand-edited pref must
/// never hide chrome the user then has to guess how to get back.
pub(crate) fn collapsed_token(collapsed: bool) -> &'static str {
    if collapsed { "hidden" } else { "shown" }
}

pub(crate) fn parse_collapsed_token(tok: &str) -> Option<bool> {
    match tok.trim() {
        "hidden" => Some(true),
        "shown" => Some(false),
        _ => None,
    }
}

/// Which side panel a key toggles, if any (parity audit 1.2). darktable uses
/// `Ctrl+Shift+L` / `Ctrl+Shift+R`; bare `L` / `R` are free in our lighttable
/// (ratings are digits, colour labels F1–F5) and are what darktable's own
/// documentation calls the panel shortcuts in the shortcut-mapping default set.
///
/// Pure, so the mapping is testable with no display.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PanelSide {
    Left,
    Right,
}

pub(crate) fn panel_toggle_key(keyval: gtk4::gdk::Key) -> Option<PanelSide> {
    use gtk4::gdk::Key;
    match keyval {
        Key::l | Key::L => Some(PanelSide::Left),
        Key::r | Key::R => Some(PanelSide::Right),
        _ => None,
    }
}

/// A side panel's persisted width, clamped into the grabbable range. `None` when
/// unset or unparseable — callers leave the layout default alone in that case.
fn stored_panel_width(db_path: &str, key: &str) -> Option<i32> {
    persist::load_ui_pref(db_path, key)
        .and_then(|v| v.parse::<i32>().ok())
        .map(|w| w.clamp(PANEL_WIDTH_MIN, PANEL_WIDTH_MAX))
}

/// Put the LEFT panel at `width`. It is the outer `Paned`'s start child, so its
/// width *is* the handle position.
fn apply_left_panel_width(outer: &gtk4::Paned, width: i32) {
    outer.set_position(width);
}

/// Put the RIGHT panel at `width`. It is an *end* child, so its handle position is
/// `centre_width - width` and depends on an allocation that hasn't happened yet at
/// build time — and hasn't happened again right after the panel is re-shown. Both
/// callers therefore go through an idle, and a still-unallocated `Paned` is left
/// alone rather than handed a position computed from a zero width, which would slam
/// the panel shut.
/// `applied` (when given) is latched **only if the position was really set**, so a
/// caller using it as a once-guard retries on the next map instead of giving up on
/// a window that hadn't been allocated yet.
fn apply_right_panel_width(
    inner: &gtk4::Paned,
    width: i32,
    applied: Option<std::rc::Rc<std::cell::Cell<bool>>>,
) {
    let inner = inner.clone();
    glib::idle_add_local_once(move || {
        let total = inner.width();
        if total <= 0 {
            return;
        }
        if let Some(applied) = applied {
            applied.set(true);
        }
        inner.set_position((total - width).max(0));
    });
}

/// Restore the persisted side-panel widths onto the two `Paned`s.
///
/// The left panel's position can be set immediately; the right panel's waits for
/// the first map (see [`apply_right_panel_width`]), and only once — re-applying on
/// every map would undo the user's drag the next time the window is shown.
fn restore_panel_widths(db_path: &str, outer: &gtk4::Paned, inner: &gtk4::Paned) {
    if let Some(left) = stored_panel_width(db_path, LEFT_PANEL_WIDTH_PREF_KEY) {
        apply_left_panel_width(outer, left);
    }

    if let Some(right) = stored_panel_width(db_path, RIGHT_PANEL_WIDTH_PREF_KEY) {
        let applied = std::rc::Rc::new(std::cell::Cell::new(false));
        inner.connect_map(move |paned| {
            if applied.get() {
                return;
            }
            apply_right_panel_width(paned, right, Some(applied.clone()));
        });
    }
}

/// Collapse/expand controller for the two side panels (parity audit 1.2).
///
/// Collapsing **hides the panel widget**, so the `Paned` hands the whole area to the
/// other child and the image area really grows — setting the handle to the edge
/// instead would leave the panel squeezed to a sliver, still drawing, and (with
/// `shrink=false`) wouldn't go all the way anyway.
///
/// The width is remembered before hiding and re-applied on expand. A `Paned`'s
/// position doesn't survive a child's visibility change intact — the end child's
/// position is a function of an allocation that changes while the child is gone —
/// and a panel that comes back narrower than it left reads as a bug, so the width is
/// carried explicitly rather than left to GTK.
#[derive(Clone)]
struct PanelCollapse {
    outer: gtk4::Paned,
    inner: gtk4::Paned,
    left_widget: gtk4::Widget,
    right_widget: gtk4::Widget,
    db_path: String,
    /// Widths to restore on expand. Seeded from the persisted widths so the very
    /// first expand of a panel that started collapsed still lands somewhere sensible.
    left_width: std::rc::Rc<std::cell::Cell<i32>>,
    right_width: std::rc::Rc<std::cell::Cell<i32>>,
    /// Collapsed state, shared with [`persist_panel_widths`]: a collapsed panel's
    /// measured width is not a width the user chose, and writing it would overwrite
    /// the real one.
    left_collapsed: std::rc::Rc<std::cell::Cell<bool>>,
    right_collapsed: std::rc::Rc<std::cell::Cell<bool>>,
}

impl PanelCollapse {
    fn new(
        db_path: &str,
        outer: &gtk4::Paned,
        inner: &gtk4::Paned,
        left_widget: &impl IsA<gtk4::Widget>,
        right_widget: &impl IsA<gtk4::Widget>,
    ) -> Self {
        let seed = |key: &str| {
            std::rc::Rc::new(std::cell::Cell::new(
                stored_panel_width(db_path, key).unwrap_or(PANEL_WIDTH_DEFAULT),
            ))
        };
        Self {
            outer: outer.clone(),
            inner: inner.clone(),
            left_widget: left_widget.clone().upcast(),
            right_widget: right_widget.clone().upcast(),
            db_path: db_path.to_string(),
            left_width: seed(LEFT_PANEL_WIDTH_PREF_KEY),
            right_width: seed(RIGHT_PANEL_WIDTH_PREF_KEY),
            left_collapsed: std::rc::Rc::new(std::cell::Cell::new(false)),
            right_collapsed: std::rc::Rc::new(std::cell::Cell::new(false)),
        }
    }

    /// The width to remember for a panel about to be hidden: what it is actually
    /// showing now, or — if that isn't a believable width, which is the case before
    /// the first allocation (restoring a collapsed panel at startup) — whatever we
    /// were already going to restore it to.
    fn remember(cell: &std::cell::Cell<i32>, measured: i32) {
        if measured >= PANEL_WIDTH_MIN {
            cell.set(measured.min(PANEL_WIDTH_MAX));
        }
    }

    fn set_left_collapsed(&self, collapsed: bool) {
        if collapsed {
            Self::remember(&self.left_width, self.outer.position());
        }
        // Order matters on expand: give the `Paned` its position before the child
        // becomes visible, so the panel doesn't appear at the wrong width for a frame.
        self.left_collapsed.set(collapsed);
        if !collapsed {
            apply_left_panel_width(&self.outer, self.left_width.get());
        }
        self.left_widget.set_visible(!collapsed);
        persist::save_ui_pref(
            &self.db_path,
            LEFT_PANEL_COLLAPSED_PREF_KEY,
            collapsed_token(collapsed),
        );
    }

    fn set_right_collapsed(&self, collapsed: bool) {
        if collapsed {
            Self::remember(
                &self.right_width,
                (self.inner.width() - self.inner.position()).max(0),
            );
        }
        self.right_collapsed.set(collapsed);
        self.right_widget.set_visible(!collapsed);
        if !collapsed {
            // The end child's position depends on the centre's width, which only
            // settles after the re-shown child is allocated — hence the idle, and
            // hence (unlike the left) after making it visible rather than before.
            //
            // Ordering assumption: set_visible(true) queues a resize; the actual
            // layout/paint runs in the next allocation phase. idle_add_local_once
            // fires after pending events are drained but before layout processes the
            // queued resize, so the position is set before the panel paints. If this
            // one-frame-glitches on a particular compositor/driver, use a higher-
            // priority idle or a frame-clock callback instead.
            apply_right_panel_width(&self.inner, self.right_width.get(), None);
        }
        persist::save_ui_pref(
            &self.db_path,
            RIGHT_PANEL_COLLAPSED_PREF_KEY,
            collapsed_token(collapsed),
        );
    }
}

/// Write panel widths back as the user drags, debounced. A panel that is currently
/// collapsed is skipped: its measured width is an artefact of being hidden, and
/// persisting it would lose the width the user actually chose.
fn persist_panel_widths(db_path: &str, outer: &gtk4::Paned, inner: &gtk4::Paned, collapse: &PanelCollapse) {
    // One pending timeout shared by both handles: whichever moved last wins the
    // timer, and both widths are written together when it fires.
    let pending: std::rc::Rc<std::cell::RefCell<Option<glib::SourceId>>> =
        std::rc::Rc::new(std::cell::RefCell::new(None));

    let schedule = {
        let db = db_path.to_string();
        let outer = outer.clone();
        let inner = inner.clone();
        let pending = pending.clone();
        let left_collapsed = collapse.left_collapsed.clone();
        let right_collapsed = collapse.right_collapsed.clone();
        std::rc::Rc::new(move || {
            if let Some(id) = pending.borrow_mut().take() {
                id.remove();
            }
            let db = db.clone();
            let outer = outer.clone();
            let inner = inner.clone();
            let pending_inner = pending.clone();
            let left_collapsed = left_collapsed.clone();
            let right_collapsed = right_collapsed.clone();
            let id = glib::timeout_add_local_once(
                std::time::Duration::from_millis(u64::from(PANEL_WIDTH_SAVE_DEBOUNCE_MS)),
                move || {
                    pending_inner.replace(None);
                    if !left_collapsed.get() {
                        persist::save_ui_pref(
                            &db,
                            LEFT_PANEL_WIDTH_PREF_KEY,
                            &outer.position().to_string(),
                        );
                    }
                    // The right panel's width, not its handle position — see the
                    // pref-key doc.
                    let right = (inner.width() - inner.position()).max(0);
                    if right > 0 && !right_collapsed.get() {
                        persist::save_ui_pref(
                            &db,
                            RIGHT_PANEL_WIDTH_PREF_KEY,
                            &right.to_string(),
                        );
                    }
                },
            );
            pending.replace(Some(id));
        })
    };

    for paned in [outer, inner] {
        let schedule = schedule.clone();
        paned.connect_position_notify(move |_| schedule());
    }
}

fn build_main_window(app: &Application) {
    // darktable ships a dark grey theme; match that first impression by forcing
    // libadwaita's dark colour scheme (the default follows the desktop setting,
    // which is light in the KasmVNC container), then darktable's own palette on
    // top — see `theme` for why the canvas greys in particular are functional.
    adw::StyleManager::default().set_color_scheme(adw::ColorScheme::ForceDark);
    theme::install();

    let window = ApplicationWindow::builder()
        .application(app)
        .title("C-41")
        .default_width(DEFAULT_WIDTH)
        .default_height(DEFAULT_HEIGHT)
        .build();

    let db_path = std::env::var("DARKROOM_LIBRARY_DB").unwrap_or_default();

    // Bootstrap the DURABLE catalog schema once on a real (non-demo) library.db
    // so a fresh /config — where no C app ever ran to create the tables — has a
    // working catalog before the lighttable query, an import, or a session-only
    // tag read (open_catalog_session) touches it. open_catalog attaches data.db +
    // a throwaway in-memory `memory` db and ensures main.* + data.tags; only the
    // durable part needs to persist, so the connection is immediately dropped.
    if !db_path.is_empty() {
        match c41_db::schema::open_catalog(&db_path) {
            Ok(_) => {}
            Err(e) => tracing::warn!("failed to bootstrap catalog schema: {e}"),
        }
    }

    // Restore the persisted lighttable rating filter (m4-98d) BEFORE the first
    // load below, so the initial grid already reflects the saved comparator +
    // floor. Seeds thread-local state only (no reload — no loader is registered
    // yet); the bottom bar reads it back when it builds its stars + dropdown.
    if let Some(tok) = persist::load_ui_pref(&db_path, RATING_FILTER_PREF_KEY) {
        lighttable::apply_rating_filter_token(&tok);
    }
    // And the colour-label quick filter (m4-126), same contract: seed the
    // thread-local mask before anything builds a control that mirrors it. Both
    // bars' circles and the left-panel checks read it back when they build.
    if let Some(tok) = persist::load_ui_pref(&db_path, lighttable::COLOUR_FILTER_PREF_KEY) {
        lighttable::apply_colour_filter_token(&tok);
    }
    // And the aspect-ratio quick filter (m4-128), same contract: seed before any
    // control that mirrors it is built (the left panel's Collection-filters
    // dropdown reads it back when it builds).
    if let Some(tok) = persist::load_ui_pref(&db_path, lighttable::ASPECT_FILTER_PREF_KEY) {
        lighttable::apply_aspect_filter_token(&tok);
    }
    // Likewise restore the thumbnail overlay mode (m4-98e) before the grid binds
    // its first cells, so they're laid out right the first time (no visible flip).
    if let Some(tok) = persist::load_ui_pref(&db_path, OVERLAY_MODE_PREF_KEY) {
        lighttable::apply_overlay_mode_token(&tok);
    }
    // And the lighttable view mode (m4-98c), before the bottom bar seeds its
    // switcher from it. A mode this build can't render is refused by the parser,
    // so a stale/hand-edited pref can't open onto an empty lighttable.
    if let Some(tok) = persist::load_ui_pref(&db_path, VIEW_MODE_PREF_KEY) {
        lighttable::apply_view_mode_token(&tok);
    }

    // ── Toast overlay (wraps everything for in-app notifications) ──────────
    let toast_overlay = adw::ToastOverlay::new();
    let make_toast = {
        let to = toast_overlay.clone();
        move |msg: String| {
            to.add_toast(adw::Toast::new(&msg));
        }
    };

    // ── Lighttable ─────────────────────────────────────────────────────────
    // Destructured rather than re-derived: the bottom-bar controls and the
    // navigation wiring below all need the GridView, and fishing it back out with
    // `scroll.child().and_downcast()` would make every one of them silently go
    // inert the day a view mode changes the scroller's child (m4-98c).
    let lighttable::LighttablePage {
        scroll,
        grid: lt_grid,
        model: lt_model,
        selection: lt_selection,
    } = lighttable::lighttable_page(db_path.clone());
    lighttable::lighttable_load_from_db(&lt_model, &db_path);

    // ── Panels ─────────────────────────────────────────────────────────────
    // Shared "currently-filtering tag prefix" (None = no tag filter): the full
    // `parent|child` path of the clicked tag-tree node. Set by the left-panel
    // folder/tag clicks and the search/import paths below; read by the
    // tag-mutation callbacks to decide whether to re-run the hierarchical filter.
    let active_tag: std::rc::Rc<std::cell::RefCell<Option<String>>> =
        std::rc::Rc::new(std::cell::RefCell::new(None));

    let left  = panels::LeftPanel::new(&db_path, &lt_model, &active_tag);
    let right = panels::MetadataPanel::new();

    // Re-run the grid filter after a tag mutation, but ONLY when a tag filter is
    // active — so e.g. detaching the filtered-on tag drops the image from the
    // grid, while ordinary tagging under an All/Folder/Search view leaves the
    // grid (and selection) untouched. Across the reload we preserve the selected
    // image so an unrelated attach/rename doesn't yank the user back to index 0;
    // if the image left the grid (detached the filtered-on tag) the default
    // selection stands.
    let reapply_tag_filter = {
        let at  = active_tag.clone();
        let mdl = lt_model.clone();
        let sel = lt_selection.clone();
        let db  = db_path.clone();
        move || {
            // Clone the prefix out before the load so the RefCell borrow is
            // released (the loader doesn't touch active_tag, but keep it tight).
            let cur = at.borrow().clone();
            if let Some(prefix) = cur {
                let prev = lighttable::selected_path(&sel);
                lighttable::lighttable_load_by_tag_prefix(&mdl, &db, &prefix);
                lighttable::reselect_path(&sel, prev.as_deref());
            }
        }
    };

    // Bidirectional tag-change refresh:
    //  • attaching/detaching a tag in the metadata panel refreshes the
    //    left-panel Tags list (new tags / changed counts);
    //  • renaming/deleting a tag in the left panel re-renders the metadata
    //    panel's chips for the current image.
    // Both then re-run the active tag filter. Neither callback re-enters the
    // other's mutation path, so there is no loop.
    {
        let lp = left.clone();
        let reapply = reapply_tag_filter.clone();
        right.set_on_tags_changed(move || { lp.refresh_tags(); reapply(); });
    }
    {
        let meta = right.clone();
        let reapply = reapply_tag_filter.clone();
        left.set_on_tags_changed(move || { meta.refresh_tags_display(); reapply(); });
    }

    // Selection → metadata
    {
        let meta = right.clone();
        let db   = db_path.clone();
        // Through `selected_path`, never through the full model: `selected()` is an
        // index into whatever the grid is *showing*, which culling narrows to a
        // window (m4-98c b). Indexing the full collection with it would show the
        // metadata of a different image, silently and plausibly.
        lt_selection.connect_selection_changed(move |sel, _, _| {
            if let Some(path) = lighttable::selected_path(sel) {
                meta.update(&path, &db);
            }
        });
    }

    // ── Lighttable page layout ─────────────────────────────────────────────
    scroll.set_hexpand(true);

    let hbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .build();
    // Full preview (m4-98c c) shares the centre slot with the grid: the scroller
    // goes on one page of a Stack and the preview on the other, so the GridView,
    // its model and every bottom-bar control stay live underneath it. The side
    // panels are outside the stack, as in darktable, where `f` covers the centre
    // view and leaves the panels up.
    let (preview_overlay, preview) = lighttable::full_preview::FullPreview::wrap(&scroll);

    // Panels are DRAGGABLE, not fixed (parity audit 1.1). Two nested `Paned`s:
    // [left | [centre | right]]. Separators come from the Paned handles, so the
    // explicit ones are gone. `resize` is false on both panels and true on the
    // centre, so growing the window grows the image area rather than the chrome —
    // and `shrink` is false everywhere, so a drag can't collapse a panel to a
    // sliver the user then can't grab. Collapsing entirely is the panel toggles
    // (audit 1.2), not an accidental drag.
    let inner_paned = gtk4::Paned::new(gtk4::Orientation::Horizontal);
    inner_paned.set_start_child(Some(&preview_overlay));
    inner_paned.set_end_child(Some(&right.widget));
    inner_paned.set_resize_start_child(true);
    inner_paned.set_resize_end_child(false);
    inner_paned.set_shrink_start_child(false);
    inner_paned.set_shrink_end_child(false);

    let outer_paned = gtk4::Paned::new(gtk4::Orientation::Horizontal);
    outer_paned.set_start_child(Some(&left.widget));
    outer_paned.set_end_child(Some(&inner_paned));
    outer_paned.set_resize_start_child(false);
    outer_paned.set_resize_end_child(true);
    outer_paned.set_shrink_start_child(false);
    outer_paned.set_shrink_end_child(false);
    outer_paned.set_hexpand(true);

    // Wide handles: the default is a ~1px line that is hard to hit and reads as a
    // border rather than a control. The complaint that started this ("the side
    // panels cannot be resized") is as much about discoverability as capability.
    outer_paned.set_wide_handle(true);
    inner_paned.set_wide_handle(true);

    // Restore the widths from the last session, then persist every drag. The
    // handle position is what the user set, so it is the thing to store — not a
    // width computed from it, which would drift with the window size.
    restore_panel_widths(&db_path, &outer_paned, &inner_paned);

    // Collapsible panels (parity audit 1.2). Built before the width persistence so
    // that persistence can see the collapsed state and skip writing a hidden
    // panel's width. Restored state is applied further down, once the header
    // toggles exist to reflect it.
    let collapse = PanelCollapse::new(
        &db_path,
        &outer_paned,
        &inner_paned,
        &left.widget,
        &right.widget,
    );
    persist_panel_widths(&db_path, &outer_paned, &inner_paned, &collapse);

    hbox.append(&outer_paned);

    // Paint the metadata for whatever is selected right now (parity audit 1.4).
    // `SingleSelection` auto-selects index 0 when the model is filled, and that
    // fires no `selection-changed` — so without this the panel sits on "Select an
    // image to view metadata" over a grid that plainly has an image selected,
    // until the user clicks a *different* cell.
    if let Some(path) = lighttable::selected_path(&lt_selection) {
        right.update(&path, &db_path);
    }

    // Styles (parity 2.4). "Save current" needs whatever edit the selected
    // image carries *at click time*, so this is a getter rather than a captured
    // value — the selection changes constantly and the panel is built once.
    // In the lighttable the edit lives in the database (the darkroom view owns
    // the live one), so read it back from there.
    {
        let sel = lt_selection.clone();
        let db = db_path.to_string();
        let get_params: panels::StyleParamsGetter = std::rc::Rc::new(move || {
            let path = lighttable::selected_path(&sel)?;
            let params = persist::load_params(&db, &path);
            Some((path, params))
        });
        // Report outcomes in a toast, and do NOT reload the grid. Thumbnails are
        // decoded from the file's own bytes and know nothing about
        // PreviewParams, so a reload could not show the applied style anyway —
        // and `lighttable_load_from_db` loads the WHOLE library, which would
        // throw away an active folder/tag/search collection and reset the
        // selection to image 0. That is a destructive no-op.
        let notify: std::rc::Rc<dyn Fn(String)> = std::rc::Rc::new(make_toast.clone());
        right.wire_styles(&db_path, get_params, notify);
    }

    // Metadata editor (parity 2.3) reports failed writes the same way.
    right.set_on_notify(make_toast.clone());

    // The full preview follows the SELECTION, not just its own keys — the same
    // observer shape the metadata panel uses. Without this the ← / → step moves
    // the selection and the metadata panel while the preview keeps showing the
    // OLD image, and any reload that drops the previewed image leaves a
    // full-screen photo the app no longer considers selected.
    {
        let preview = preview.clone();
        lt_selection.connect_selection_changed(move |sel, _, _| {
            let target = lighttable::full_preview::preview_target(
                lighttable::selected_path(sel).as_deref(),
            );
            preview.follow_selection(&target);
        });
    }

    // ── Header bar ─────────────────────────────────────────────────────────
    let lt_header = adw::HeaderBar::new();
    lt_header.set_title_widget(Some(&adw::WindowTitle::new("C-41", "Lighttable")));

    // Panel toggles (parity audit 1.2) — one per side, at the header's two ends so
    // each button sits over the panel it hides. darktable puts triangles on the
    // window edges; a header toggle is the GNOME idiom for the same thing and, being
    // a real button with a tooltip, is findable, which the audit's complaint was
    // half about. `L` / `R` do the same from the keyboard (wired with the grid's
    // other keys).
    let (left_toggle, right_toggle) = {
        let mk = |icon: &str, tip: &str| {
            gtk4::ToggleButton::builder()
                .icon_name(icon)
                .tooltip_text(tip)
                .active(true) // panels start shown; restored state is applied below
                .build()
        };
        let l = mk("sidebar-show-symbolic", "Show/hide the left panel (L)");
        let r = mk("sidebar-show-right-symbolic", "Show/hide the right panel (R)");
        // Active = panel SHOWN, so the pressed-in look means "there is a panel
        // there", matching how the view-mode toggles read.
        {
            let collapse = collapse.clone();
            l.connect_toggled(move |b| collapse.set_left_collapsed(!b.is_active()));
        }
        {
            let collapse = collapse.clone();
            r.connect_toggled(move |b| collapse.set_right_collapsed(!b.is_active()));
        }
        lt_header.pack_start(&l);
        lt_header.pack_end(&r);
        (l, r)
    };

    // Apply the persisted collapsed state through the toggles, so the buttons and
    // the panels can't disagree about what is on screen. An unrecognised token is
    // no opinion and leaves the panel shown (see `parse_collapsed_token`).
    for (key, toggle) in [
        (LEFT_PANEL_COLLAPSED_PREF_KEY, &left_toggle),
        (RIGHT_PANEL_COLLAPSED_PREF_KEY, &right_toggle),
    ] {
        if let Some(true) = persist::load_ui_pref(&db_path, key)
            .as_deref()
            .and_then(parse_collapsed_token)
        {
            // Drives the same handler a click would, which hides the panel and
            // re-persists the state it just read — a harmless idempotent write, and
            // cheaper than a second code path that could drift from the click one.
            toggle.set_active(false);
        }
    }

    // Search bar
    {
        let search = gtk4::SearchEntry::builder()
            .placeholder_text("Search images…")
            .width_request(200)
            .build();
        let db = db_path.clone();
        let at = active_tag.clone();
        let lp = left.clone();
        search.connect_search_changed(clone!(@weak lt_model => move |s| {
            let query = s.text().to_string();
            lp.clear_filter_highlights();   // a name search supersedes any filter highlight
            *at.borrow_mut() = None;   // a name search supersedes any tag filter
            lighttable::lighttable_filter_by_name(&lt_model, &db, &query);
        }));
        lt_header.pack_start(&search);
    }

    // Import button
    {
        let btn = gtk4::Button::builder()
            .icon_name("list-add-symbolic")
            .tooltip_text("Import folder")
            .build();
        let db         = db_path.clone();
        let toast_fn   = make_toast.clone();
        let at         = active_tag.clone();
        let lp         = left.clone();
        btn.connect_clicked(clone!(@weak window, @weak lt_model => move |_| {
            let db_inner    = db.clone();
            let toast_inner = toast_fn.clone();
            let lp_inner    = lp.clone();
            dialogs::show_import_dialog(
                window.upcast_ref::<gtk4::Window>(),
                db.clone(),
                clone!(@weak lt_model, @strong db_inner, @strong at => move || {
                    lp_inner.clear_filter_highlights();   // post-import shows the quick-filtered collection
                    *at.borrow_mut() = None;   // (quick filters like stars/colours stay armed)
                    lighttable::lighttable_load_from_db(&lt_model, &db_inner);
                }),
                toast_inner,
            );
        }));
        lt_header.pack_start(&btn);
    }

    // Export selected button
    {
        let btn = gtk4::Button::builder()
            .icon_name("document-send-symbolic")
            .tooltip_text("Export selected")
            .build();
        let toast_fn = make_toast.clone();
        let export_db = db_path.clone();
        // `selected_path` resolves against the model the grid is showing, so this
        // exports the image the user is looking at in every layout. Indexing the
        // full collection with a window-relative `selected()` would export a
        // DIFFERENT FILE under culling, with nothing to signal it (m4-98c b).
        btn.connect_clicked(clone!(@weak lt_selection, @weak window => move |_| {
            let paths: Vec<String> =
                lighttable::selected_path(&lt_selection).into_iter().collect();
            dialogs::show_export_dialog(
                window.upcast_ref::<gtk4::Window>(),
                paths,
                None, // no fixed edit — each image's edit is loaded from the catalog
                Some(export_db.clone()),
                toast_fn.clone(),
            );
        }));
        lt_header.pack_end(&btn);
    }

    // Image-count indicator (m4-97a) — darktable shows the current view's image
    // count in the top bar. Bound to the model's items-changed signal so it
    // tracks every load / filter / import / tag-filter without touching each
    // load call site. Capped views (GRID_CAP) read as e.g. "2000 images".
    {
        // The grid model also carries non-image sentinel rows (a "(No results)"
        // / empty-state placeholder, or a truncation notice) — the codebase's
        // convention is that real image rows contain '/', sentinels don't. Count
        // only real images so an empty collection reads "0 images", not "1".
        fn image_count(m: &gtk4::StringList) -> u32 {
            (0..m.n_items())
                .filter(|&i| m.string(i).is_some_and(|s| s.contains('/')))
                .count() as u32
        }
        fn fmt(n: u32) -> String {
            format!("{n} image{}", if n == 1 { "" } else { "s" })
        }
        let count_label = gtk4::Label::new(Some(&fmt(image_count(&lt_model))));
        count_label.add_css_class("dim-label");
        lt_model.connect_items_changed(clone!(@weak count_label => move |m, _, _, _| {
            count_label.set_label(&fmt(image_count(m)));
        }));
        lt_header.pack_end(&count_label);
    }

    // Quick-filter preset dropdown (m4-97c) — darktable's top-bar
    // `filter [all images ▾]`. Rows come from FilterPreset::ALL (labels live beside
    // the variants) plus a trailing "custom" row that is only *shown*, never
    // chosen: it's what the dropdown displays when the bottom bar has set a filter
    // no preset names (e.g. `≤ 3`). Presets don't implement filtering themselves —
    // each is a named (comparator, stars) pair applied to the same state the bottom
    // bar drives, and an observer keeps the two in sync in both directions.
    {
        const CUSTOM_ROW: &str = "custom";
        let mut labels: Vec<String> =
            lighttable::FilterPreset::ALL.iter().map(|p| p.label()).collect();
        labels.push(CUSTOM_ROW.to_string());
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

        let filter_dd = gtk4::DropDown::from_strings(&label_refs);
        filter_dd.set_tooltip_text(Some("Quick filter the collection"));

        // Reflect the live filter state into the row selection. Registered as a
        // filter observer so a change made in the BOTTOM bar moves this dropdown
        // too, landing on "custom" when the state isn't a named preset.
        let sync_from_state = {
            let filter_dd = filter_dd.clone();
            move || {
                let idx = lighttable::current_filter_preset()
                    .map_or(lighttable::FilterPreset::CUSTOM_INDEX, |p| p.to_index());
                // (Belt-and-braces: GTK already no-ops an unchanged position, and
                // the sync guard covers the emission either way.)
                if filter_dd.selected() != idx {
                    filter_dd.set_selected(idx);
                }
            }
        };
        // Seeding BEFORE the handler is connected is load-bearing: it mirrors the
        // filter restored at startup without the write being taken for a user edit
        // (same ordering the bottom bar's comparator relies on).
        sync_from_state();
        lighttable::add_filter_observer(sync_from_state);

        filter_dd.connect_selected_notify(move |dd| {
            // Ignore the programmatic write an observer just made (it's a mirror of
            // the state, not a user edit) — otherwise this would recurse.
            if lighttable::filter_sync_in_progress() {
                return;
            }
            // "custom" is display-only. `>=` (not `==`) also catches GTK's
            // INVALID_LIST_POSITION (u32::MAX), which `from_index` would otherwise
            // map to `AllImages` and *apply*. Snap the control back to the real
            // state via the observer bus rather than calling our own sync closure —
            // that would re-enter this handler unguarded.
            if dd.selected() >= lighttable::FilterPreset::CUSTOM_INDEX {
                lighttable::sync_filter_controls();
                return;
            }
            lighttable::set_filter_preset(lighttable::FilterPreset::from_index(dd.selected()));
        });
        lt_header.pack_start(&filter_dd);
    }

    // Sort-by dropdown (m4-97b) + direction toggle (m4-97e) — darktable's top-bar
    // "sort by" with the ascending/descending arrow beside it. Changing either
    // re-renders the *current* view under the new order; the loaders self-register
    // a reload closure, so this doesn't need to know which view is showing. The
    // two controls share a `.linked` box so the arrow stays glued to the dropdown.
    {
        let sort_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
        sort_box.add_css_class("linked");

        let sort_dd = gtk4::DropDown::from_strings(&["Filename", "Date taken", "Rating"]);
        sort_dd.set_tooltip_text(Some("Sort images by"));
        sort_dd.connect_selected_notify(|dd| {
            let order = match dd.selected() {
                1 => lighttable::SortOrder::DateTaken,
                2 => lighttable::SortOrder::Rating,
                _ => lighttable::SortOrder::Filename,
            };
            lighttable::set_sort_order(order);
        });
        sort_box.append(&sort_dd);

        // Direction toggle: unpressed = ascending (natural), pressed = descending
        // (reversed). Swap the icon so the arrow always reflects the active state.
        let dir_btn = gtk4::ToggleButton::builder()
            .icon_name("view-sort-ascending-symbolic")
            .tooltip_text("Reverse sort order")
            .build();
        dir_btn.connect_toggled(|b| {
            let reverse = b.is_active();
            b.set_icon_name(if reverse {
                "view-sort-descending-symbolic"
            } else {
                "view-sort-ascending-symbolic"
            });
            lighttable::set_sort_reverse(reverse);
        });
        sort_box.append(&dir_btn);

        lt_header.pack_end(&sort_box);
    }

    // Colour-label quick filter (m4-126) — darktable's top-bar colour circles.
    // Toggling a circle composes an AND-on-top predicate with whatever collection
    // is active; the row observes the bus, so it re-dots itself on every change.
    // The bottom bar builds a second mirror of the same state below — both stay in
    // step through the filter-observer bus, and ONE persist observer (registered
    // right after both rows exist) saves the token once per change.
    lt_header.pack_start(&lighttable::colour_circles_row());
    {
        let db = db_path.clone();
        lighttable::add_filter_observer(move || {
            crate::persist::save_ui_pref(
                &db,
                lighttable::COLOUR_FILTER_PREF_KEY,
                &lighttable::colour_filter_token(),
            );
        });
    }

    // Aspect quick filter (m4-128) — same one-writer-per-key persistence shape as
    // the colour filter above: an app-level observer saves the token on every
    // filter change, so the left-panel dropdown (and any future mirror) never
    // persists anything itself.
    {
        let db = db_path.clone();
        lighttable::add_filter_observer(move || {
            crate::persist::save_ui_pref(
                &db,
                lighttable::ASPECT_FILTER_PREF_KEY,
                &lighttable::aspect_filter_token(),
            );
        });
    }

    let lt_toolbar = adw::ToolbarView::new();
    lt_toolbar.add_top_bar(&lt_header);
    lt_toolbar.set_content(Some(&hbox));

    // ── Bottom toolbar (m4-98a/b/d) ────────────────────────────────────────
    // darktable's lighttable bottom bar. Right (m4-98a): the thumb-size stepper
    // (its "images per row" ± control) driving the grid's max-column bound live —
    // fewer columns ⇒ bigger thumbnails. Left (m4-98b/d): a star-rating filter
    // plus the m4-126 colour circles — both compose on top of whatever collection
    // is active and persist across sessions. Later: view-mode switcher.
    {
        let bottom = gtk4::CenterBox::new();
        bottom.add_css_class("toolbar");

        // Rating filter (m4-98b/d). The comparator dropdown picks how the star
        // count is applied; clicking star N sets the count (re-clicking the current
        // floor drops it to 0). Both the lit stars and the dropdown selection read
        // `current_min_rating()`/`current_rating_compare()` — the single source of
        // truth — so the display can't drift from the DB query, and every change
        // persists the compact filter token so it survives a restart.
        {
            let filter_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 4);
            filter_box.set_margin_start(6);

            let star_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
            star_box.add_css_class("linked");
            star_box.set_tooltip_text(Some("Filter by star rating"));

            let stars: std::rc::Rc<Vec<gtk4::Button>> = std::rc::Rc::new(
                (1..=5u8)
                    .map(|_| {
                        gtk4::Button::builder().icon_name("non-starred-symbolic").build()
                    })
                    .collect(),
            );
            for b in stars.iter() {
                star_box.append(b);
            }

            // Persist the current filter (comparator + floor) as its compact token.
            let persist_filter: std::rc::Rc<dyn Fn()> = {
                let db = db_path.clone();
                std::rc::Rc::new(move || {
                    persist::save_ui_pref(
                        &db,
                        RATING_FILTER_PREF_KEY,
                        &lighttable::rating_filter_token(),
                    );
                })
            };

            // Repaint the stars so 1..=floor are lit and the rest hollow; in the
            // `Rejected` mode the star count is irrelevant, so grey the whole row
            // out (and light none) to signal the dropdown alone drives the filter.
            let refresh_stars: std::rc::Rc<dyn Fn()> = {
                let stars = stars.clone();
                let star_box = star_box.clone();
                std::rc::Rc::new(move || {
                    let rejected =
                        lighttable::current_rating_compare() == lighttable::RatingCompare::Rejected;
                    star_box.set_sensitive(!rejected);
                    let floor = lighttable::current_min_rating();
                    for (i, b) in stars.iter().enumerate() {
                        let lit = !rejected && (i as u8) < floor; // star i+1 lit iff i+1 <= floor
                        b.set_icon_name(if lit { "starred-symbolic" } else { "non-starred-symbolic" });
                    }
                })
            };

            // Comparator dropdown, seeded from the restored filter BEFORE the
            // handler is connected so the seeding doesn't fire a spurious reload.
            let compare = gtk4::DropDown::from_strings(&["≥", "=", "≤", "⚑"]);
            compare.set_valign(gtk4::Align::Center);
            compare.set_tooltip_text(Some("Rating comparator: ≥ / = / ≤ / rejected only"));
            compare.set_selected(lighttable::current_rating_compare().to_index());
            compare.connect_selected_notify(move |d| {
                // Skip the programmatic write an observer just made (see below).
                if lighttable::filter_sync_in_progress() {
                    return;
                }
                lighttable::set_rating_compare(lighttable::RatingCompare::from_index(d.selected()));
            });

            // One observer owns re-syncing this bar's display and persisting the
            // filter, for changes from ANY control (these stars, the comparator, or
            // the top bar's preset dropdown). The click handlers below therefore
            // only set state — `set_*` runs this via `filter_changed()`.
            {
                let compare = compare.clone();
                let refresh_stars = refresh_stars.clone();
                let persist_filter = persist_filter.clone();
                lighttable::add_filter_observer(move || {
                    let idx = lighttable::current_rating_compare().to_index();
                    if compare.selected() != idx {
                        compare.set_selected(idx);
                    }
                    refresh_stars();
                    persist_filter();
                });
            }

            refresh_stars(); // sync stars to the restored filter

            for (i, b) in stars.iter().enumerate() {
                let n = (i + 1) as u8;
                b.connect_clicked(move |_| {
                    // Consult the sync guard like every other filter control. A
                    // plain Button emits `clicked` only on real input today, so this
                    // is inert — but it stops being inert the moment these become
                    // toggles an observer sets (darktable's stars are toggles), and
                    // "safe by accident" is not a property worth relying on.
                    if lighttable::filter_sync_in_progress() {
                        return;
                    }
                    // Toggle: re-clicking the current floor drops the count to 0.
                    let new = if lighttable::current_min_rating() == n { 0 } else { n };
                    lighttable::set_min_rating(new);
                });
            }

            filter_box.append(&compare);
            filter_box.append(&star_box);
            // Colour circles (m4-126): a second mirror of the same thread-local
            // mask the top bar's row drives. Both rows observe the bus, so a
            // click in either bar re-dots both; persistence is the single
            // app-level observer registered with the top row above.
            filter_box.append(&lighttable::colour_circles_row());
            bottom.set_start_widget(Some(&filter_box));
        }

        {
            let grid = lt_grid.clone();
            // The grid's own `max-columns` property is the single source of truth
            // for the current thumb size — no separate counter to keep in sync.
            grid.set_max_columns(lighttable::THUMB_COLS_DEFAULT);

            // Thumb-size stepper (m4-98a). Built here, ahead of the controls that
            // sit to its left, because the view-mode switcher has to refresh it:
            // in culling this same control sets how many images are compared, so
            // its range, label and sensitivity all change with the mode.
            let zoom_out = gtk4::Button::builder()
                .icon_name("zoom-out-symbolic")
                .tooltip_text("Larger thumbnails (fewer per row)")
                .build();
            let count = gtk4::Label::new(Some(&lighttable::THUMB_COLS_DEFAULT.to_string()));
            count.set_width_chars(2);
            let zoom_in = gtk4::Button::builder()
                .icon_name("zoom-in-symbolic")
                .tooltip_text("Smaller thumbnails (more per row)")
                .build();

            // Sync the label + button sensitivity to the grid's current bound, and
            // grey out a button once its end of the range is reached so the control
            // can't run past it. The range is the stepper's own in the file manager
            // and culling's comparison-set range in culling: without that, steps
            // past the culling maximum would count up on the label while nothing on
            // screen moved — a control that looks live and isn't.
            let refresh: std::rc::Rc<dyn Fn()> = {
                let grid = grid.clone();
                let count = count.clone();
                let zoom_out = zoom_out.clone();
                let zoom_in = zoom_in.clone();
                std::rc::Rc::new(move || {
                    // Re-fit the culling window + cell sizes first — both depend
                    // on the viewport, so this is what fits culling after the
                    // first allocation and after every resize (width AND height,
                    // since m4-132). No-op in the file manager.
                    lighttable::cull_resync(&grid);
                    // In culling the label is the comparison-set size; since
                    // m4-132 every step in the range is fully visible (cells
                    // shrink to fit), so it always matches what the stepper asked
                    // for — no dead zone to explain away.
                    let (n, lo, hi) = lighttable::cull_stepper_state(&grid)
                        .unwrap_or((grid.max_columns(), THUMB_COLS_MIN, THUMB_COLS_MAX));
                    count.set_label(&n.to_string());
                    zoom_out.set_sensitive(n > lo);
                    zoom_in.set_sensitive(n < hi);
                })
            };
            refresh(); // sync initial label + sensitivity to the default

            // Re-fit when the viewport changes. At startup the view mode is
            // restored before the grid is allocated, so culling's window has no
            // size to fit to yet; the scroller's page-sizes are the viewport
            // dimensions and notify on allocation as well as on every later
            // resize. Two hooks since m4-132: culling cells now derive their
            // HEIGHT from the viewport too, and a height-only change (e.g. a
            // header/toolbar toggling) moves only the vertical adjustment.
            scroll.hadjustment().connect_page_size_notify({
                let refresh = refresh.clone();
                move |_| refresh()
            });
            scroll.vadjustment().connect_page_size_notify({
                let refresh = refresh.clone();
                move |_| refresh()
            });

            // Centre of the bar carries two controls, so it is a Box rather than a
            // bare widget: the CenterBox has exactly three slots and the rating
            // filter (start) and thumb stepper (end) hold the other two.
            let center_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);

            // View-mode switcher (m4-98c): darktable's file manager / zoomable /
            // culling layouts as a linked group of icon-only toggles, built from
            // `ViewMode::ALL` so the control can't drift from the variants. Modes
            // this build can't render are insensitive — like the header's "Other"
            // view — so they can't break the group's single-active invariant, and
            // the greying is the "unavailable" affordance. The explanation lives on
            // the box, not the buttons: GTK4 never emits query-tooltip for an
            // insensitive widget, so a per-button tooltip on exactly the modes that
            // need explaining would be text nobody can read.
            {
                let mode_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
                mode_box.add_css_class("linked");
                mode_box.set_valign(gtk4::Align::Center);
                mode_box.set_tooltip_text(Some(&lighttable::view_mode_switcher_tooltip()));

                // Phase 1 — build and group every button, activating none. Joining a
                // group clears the joiner's active flag, so seeding is a separate
                // pass rather than something interleaved with group construction.
                let buttons: Vec<(lighttable::ViewMode, gtk4::ToggleButton)> =
                    lighttable::ViewMode::ALL
                        .into_iter()
                        .map(|mode| {
                            let btn = gtk4::ToggleButton::builder()
                                .icon_name(mode.icon_name())
                                .sensitive(mode.is_available())
                                .build();
                            mode_box.append(&btn);
                            (mode, btn)
                        })
                        .collect();
                if let Some((_, first)) = buttons.first() {
                    for (_, btn) in buttons.iter().skip(1) {
                        btn.set_group(Some(first));
                    }
                }

                // Push the live mode back onto the buttons. Used to seed them, and
                // to roll the display back if a switch is ever refused — refusing to
                // *persist* a mode while leaving its button lit would show a mode
                // that was never entered. `set_active` re-enters `toggled`, so the
                // guard is what keeps that from recursing.
                let syncing = std::rc::Rc::new(std::cell::Cell::new(false));
                let resync: std::rc::Rc<dyn Fn()> = {
                    let buttons = buttons.clone();
                    let syncing = syncing.clone();
                    std::rc::Rc::new(move || {
                        syncing.set(true);
                        let live = lighttable::current_view_mode();
                        for (mode, btn) in &buttons {
                            btn.set_active(*mode == live);
                        }
                        syncing.set(false);
                    })
                };

                // Phase 2 — seed from the restored mode BEFORE any handler exists,
                // so seeding can't fire a spurious re-apply + re-persist.
                debug_assert!(
                    lighttable::current_view_mode().is_available(),
                    "the current view mode must always be one this build can render",
                );
                resync();

                // Phase 3 — connect the handlers.
                for (mode, btn) in &buttons {
                    let mode = *mode;
                    btn.connect_toggled({
                        let grid = grid.clone();
                        let db = db_path.clone();
                        let resync = resync.clone();
                        let syncing = syncing.clone();
                        let refresh = refresh.clone();
                        move |b| {
                            // Skip the writes `resync` itself makes, and the `toggled`
                            // a radio group emits on the button going OFF — only the
                            // button turning ON is a mode change.
                            if syncing.get() || !b.is_active() {
                                return;
                            }
                            if lighttable::set_view_mode(&grid, mode) {
                                persist::save_ui_pref(
                                    &db,
                                    VIEW_MODE_PREF_KEY,
                                    lighttable::view_mode_token_for(mode),
                                );
                                // The stepper's range differs per mode (in culling
                                // it sets the comparison-set size), so re-clamp and
                                // relabel it for the mode just entered.
                                refresh();
                            } else {
                                // Refused (a mode this build can't render): put the
                                // buttons back on the mode actually in effect rather
                                // than leaving the UI claiming one that isn't.
                                resync();
                            }
                        }
                    });
                }
                // Phase 4 — the restored mode has so far only lit a button; the grid
                // still has to be told, once, outside the handlers (going through
                // them would re-persist a pref that never changed). A no-op for
                // FileManager today, load-bearing from the culling increment on:
                // without it, restoring culling would show a lit culling button over
                // a file-manager grid, with nothing to say so.
                if !lighttable::set_view_mode(&grid, lighttable::current_view_mode()) {
                    tracing::warn!(
                        "restored view mode is not renderable by this build; \
                         lighttable stays in the file-manager layout"
                    );
                    resync();
                }
                refresh(); // the restored mode may narrow the stepper's range

                center_box.append(&mode_box);
            }

            // Thumbnail overlay mode (m4-98e), centre of the bar: how much metadata
            // each cell shows. Seeded from the restored pref BEFORE the handler is
            // connected, so seeding can't fire a spurious re-apply.
            {
                // Rows come from OverlayMode::ALL so the control can never drift
                // from the variants (labels live next to the enum). They're terse
                // on purpose: this CenterBox already carries the rating filter at
                // the start and the thumb stepper at the end, and its minimum width
                // is the sum of all three — the ~915px lighttable overflow gotcha.
                let labels: Vec<&str> =
                    lighttable::OverlayMode::ALL.iter().map(|m| m.label()).collect();
                let overlays = gtk4::DropDown::from_strings(&labels);
                overlays.set_valign(gtk4::Align::Center);
                overlays.set_tooltip_text(Some("Thumbnail overlays: none / stars + labels / full"));
                overlays.set_selected(lighttable::current_overlay_mode().to_index());
                overlays.connect_selected_notify({
                    let grid = grid.clone();
                    let db = db_path.clone();
                    move |d| {
                        let mode = lighttable::OverlayMode::from_index(d.selected());
                        lighttable::set_overlay_mode(&grid, mode);
                        // Persist the mode just applied (not a re-read of the global).
                        persist::save_ui_pref(
                            &db,
                            OVERLAY_MODE_PREF_KEY,
                            lighttable::overlay_mode_token_for(mode),
                        );
                    }
                });
                center_box.append(&overlays);
            }

            bottom.set_center_widget(Some(&center_box));

            // Each step also re-applies the culling window (a no-op in the file
            // manager), and clamps into whichever range the current mode uses.
            let step = |grid: gtk4::GridView, refresh: std::rc::Rc<dyn Fn()>, up: bool| {
                move |_: &gtk4::Button| {
                    // Step within whichever range the current mode uses — in
                    // culling that's the comparison-set bounds, and since m4-132
                    // every step there is fully visible (the cells shrink to fit,
                    // so window count and `max_columns` always agree).
                    let (cur, lo, hi) = lighttable::cull_stepper_state(&grid)
                        .unwrap_or((grid.max_columns(), THUMB_COLS_MIN, THUMB_COLS_MAX));
                    let n = if up { cur.saturating_add(1) } else { cur.saturating_sub(1) };
                    grid.set_max_columns(n.clamp(lo, hi));
                    lighttable::cull_resync(&grid);
                    refresh();
                }
            };
            zoom_out.connect_clicked(step(grid.clone(), refresh.clone(), false));
            zoom_in.connect_clicked(step(grid.clone(), refresh.clone(), true));

            let zoom_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
            zoom_box.set_margin_start(6);
            zoom_box.set_margin_end(6);
            zoom_box.append(&zoom_out);
            zoom_box.append(&count);
            zoom_box.append(&zoom_in);
            bottom.set_end_widget(Some(&zoom_box));
        }

        lt_toolbar.add_bottom_bar(&bottom);
    }

    // ── Date timeline (m4-99) ──────────────────────────────────────────────
    // darktable's bottom date-histogram strip: one bar per year, click to filter.
    // Added as a SECOND bottom bar so it sits below the toolbar, as it does there
    // (ToolbarView stacks bottom bars in the order they're added). It composes
    // with the rating/preset filters through the same observer bus.
    lt_toolbar.add_bottom_bar(&lighttable::timeline::timeline_strip(&db_path));

    let lt_page = adw::NavigationPage::builder()
        .title("Lighttable")
        .tag("lighttable")
        .child(&lt_toolbar)
        .build();

    // ── Navigation view ────────────────────────────────────────────────────
    let nav = adw::NavigationView::new();
    nav.push(&lt_page);

    // Double-click → darkroom page; F1–F5 → toggle colour label on selection.
    {
        let grid = lt_grid.clone();
        // `pos` is an index into the model the GRID is showing, which is not the
        // full collection in every layout — culling installs a window over it
        // (m4-98c b). Resolving through `gv.model()` keeps this handler correct in
        // both; resolving through the full model would silently open the wrong
        // image, off by the culling offset.
        let preview_for_activate = preview.clone();
        grid.connect_activate(clone!(@weak nav, @strong db_path => move |gv, pos| {
            if let Some(path) = gv.model()
                .and_then(|m| m.item(pos))
                .and_downcast::<gtk4::StringObject>()
                .map(|o| o.string().to_string())
                .filter(|p| p.contains('/'))
            {
                // Leaving the lighttable closes the preview, or coming back from
                // the editor would land on a full-screen image of whatever was up
                // before — over a grid that has since moved on.
                preview_for_activate.close();
                let page = darkroom::darkroom_page(&path, &db_path);
                // Tag the page with its image path so the pop handler below can
                // recover which cell to re-sync (m4-25), regardless of how the
                // page was dismissed (back button / Escape / swipe gesture).
                page.set_tag(Some(&path));
                nav.push(&page);
            }
        }));

        // m4-25/m4-28: when a darkroom page is popped, its colour labels and/or
        // star rating may have been changed in that view; re-query the DB and
        // repaint the returning grid cell's dot + star rows so the lighttable
        // doesn't show stale metadata until it rebinds. Every page pushed past
        // the lighttable root IS a darkroom page, and its tag carries the image
        // path; we guard on the `/` just like the loaders.
        nav.connect_popped(clone!(@weak grid, @strong db_path => move |_, page| {
            if let Some(path) = page.tag().map(|s| s.to_string()).filter(|p| p.contains('/')) {
                lighttable::refresh_color_dots_for_path(&grid, &db_path, &path);
                lighttable::refresh_stars_for_path(&grid, &db_path, &path);
            }
        }));

        // Metadata keyboard shortcuts on the selected grid image (darktable):
        // plain F1..F5 toggle colour label 0..4; plain 0..5 set the star rating.
        // Both repaint just that cell's row in place. The controller lives on the
        // grid, so it only fires when the lighttable has focus — not the darkroom
        // page, and not while the user is typing in the search entry. Modifier
        // combos (Ctrl/Alt + key) are left to propagate so they can't be mistaken
        // for a metadata shortcut.
        // Full-preview keys, factored out of the controller body below. The
        // preview is an Overlay child rather than a Stack page precisely so this
        // keeps working: a Stack unmaps the hidden page, GTK drops focus from an
        // unmapped widget, and this controller lives on the grid — which made the
        // preview a keyboard trap (`f`, Escape and the arrows all landing
        // elsewhere). Found by pressing the keys in the container.
        let handle_preview_key: std::rc::Rc<dyn Fn(gtk4::gdk::Key) -> bool> = {
            let preview = preview.clone();
            let sel = lt_selection.clone();
            let grid = lt_grid.clone();
            std::rc::Rc::new(move |keyval| {
                use lighttable::full_preview::PreviewAction;
                let Some(action) =
                    lighttable::full_preview::preview_key_action(keyval, preview.is_open())
                else {
                    return false;
                };
                match action {
                    // Swallowed: these would move the collection under the preview.
                    PreviewAction::Ignore => {}
                    PreviewAction::Close => preview.close(),
                    PreviewAction::Toggle if preview.is_open() => preview.close(),
                    PreviewAction::Toggle => {
                        // Nothing selected means nothing to preview; leave the grid
                        // up rather than opening onto an empty page.
                        if let Some(path) = lighttable::selected_path(&sel) {
                            preview.open(&path);
                        }
                    }
                    PreviewAction::Next | PreviewAction::Prev => {
                        let forward = action == PreviewAction::Next;
                        let n = sel.model().map_or(0, |m| m.n_items());
                        // Moving the selection is what drives the preview — the
                        // observer below repaints it — so the metadata panel and
                        // the grid underneath always follow the same image, and
                        // closing lands on it.
                        match lighttable::full_preview::preview_step_index(
                            sel.selected(),
                            n,
                            forward,
                        ) {
                            Some(next) => sel.set_selected(next),
                            // At the window's edge under culling, page the window
                            // and land on its near edge: the window is only 2..8
                            // images, so stopping there would look like a freeze
                            // a few presses in. `cull_step` is false outside
                            // culling, where the end of the collection is real.
                            None if lighttable::cull_step(&grid, forward) => {
                                let n = sel.model().map_or(0, |m| m.n_items());
                                if n > 0 {
                                    sel.set_selected(if forward { 0 } else { n - 1 });
                                }
                            }
                            None => {}
                        }
                    }
                }
                true
            })
        };

        let key = gtk4::EventControllerKey::new();
        let sel = lt_selection.clone();
        let db  = db_path.clone();
        let panel_left_toggle  = left_toggle.clone();
        let panel_right_toggle = right_toggle.clone();
        key.connect_key_pressed(clone!(@weak grid => @default-return glib::Propagation::Proceed, move |_, keyval, _, state| {
            if state.intersects(gtk4::gdk::ModifierType::CONTROL_MASK | gtk4::gdk::ModifierType::ALT_MASK | gtk4::gdk::ModifierType::SHIFT_MASK) {
                return glib::Propagation::Proceed;
            }
            // Full preview (m4-98c c) is checked FIRST: while it is up, ← / →
            // step through images rather than paging the culling window
            // underneath it, and every one of its keys except the toggle is inert
            // when it is closed, so nothing else loses a shortcut.
            if handle_preview_key(keyval) {
                return glib::Propagation::Stop;
            }
            // Culling pages a whole screenful with ← / → (m4-98c b). `cull_step`
            // reports whether the key was culling's at all, so arrow keys keep
            // moving the cursor normally in the file manager instead of being
            // swallowed by a mode that isn't active.
            if let Some(forward) = lighttable::cull_key_direction(keyval) {
                if lighttable::cull_step(&grid, forward) {
                    return glib::Propagation::Stop;
                }
            }
            // Panel toggles (parity audit 1.2). Driven through the header buttons so
            // the key and the click share one path and the button can't be left
            // showing the wrong state.
            if let Some(side) = panel_toggle_key(keyval) {
                let toggle = match side {
                    PanelSide::Left => &panel_left_toggle,
                    PanelSide::Right => &panel_right_toggle,
                };
                toggle.set_active(!toggle.is_active());
                return glib::Propagation::Stop;
            }
            if let Some(color) = lighttable::fkey_to_color(keyval) {
                lighttable::toggle_selected_color(&grid, &sel, &db, color);
                return glib::Propagation::Stop;
            }
            if let Some(rating) = lighttable::digit_to_rating(keyval) {
                lighttable::set_selected_rating(&grid, &sel, &db, rating);
                return glib::Propagation::Stop;
            }
            glib::Propagation::Proceed
        }));
        grid.add_controller(key);
    }

    // ── View switcher (m4-97d) ─────────────────────────────────────────────
    // darktable-style Lighttable | Darkroom | Other toggle, as the lighttable
    // header's title widget. "Darkroom" opens the currently-selected image in
    // the editor (same push as a double-click); "Other" (map/print/tethering)
    // is disabled — those views aren't ported. The switcher shows only in the
    // lighttable header for now; mirroring it into the darkroom page header is a
    // follow-up (see RUST_MIGRATION_PLAN.md).
    {
        let sw = build_view_switcher();
        let lt_btn = sw.lighttable.clone();
        let dr_btn = sw.darkroom.clone();
        lt_btn.set_active(true); // the lighttable is the root view
        // Switcher alone as the title (no caption): the app is named "Darkroom",
        // but so is darktable's editing *view* — a "Darkroom" branding caption
        // right under the "Darkroom" switcher button reads as a confusing
        // duplicate. The darkroom view keeps a caption because there it's the
        // filename (real per-image context), not a redundant app name.
        sw.container.set_halign(gtk4::Align::Center);
        lt_header.set_title_widget(Some(&sw.container));

        // The buttons only *request* navigation on a real user click. GTK4
        // emits `clicked` for user activation only (programmatic `set_active`
        // emits `toggled`), so the nav-driven state sync below can't echo back
        // into another push. "Darkroom" opens the selected image; with no valid
        // selection (empty view / sentinel row) it snaps back to Lighttable.
        let preview_for_switch = preview.clone();
        dr_btn.connect_clicked(clone!(
            @weak nav, @weak lt_selection, @weak lt_btn, @strong db_path => move |b| {
            if !b.is_active() { return; }
            if let Some(path) = lighttable::selected_path(&lt_selection) {
                preview_for_switch.close(); // same reason as the double-click path
                let page = darkroom::darkroom_page(&path, &db_path);
                page.set_tag(Some(&path));
                nav.push(&page);
            } else {
                lt_btn.set_active(true);
            }
        }));

        // `nav` is the single source of truth for the current view; mirror it in
        // the switcher regardless of entry point (switcher / double-click /
        // keyboard push), so the toggle never lies about which view is active.
        // These fire only for pushes AFTER this point — the initial lighttable
        // root was pushed earlier, so startup keeps Lighttable active.
        nav.connect_pushed(clone!(@weak dr_btn => move |_| dr_btn.set_active(true)));
        nav.connect_popped(clone!(@weak lt_btn => move |_, _| lt_btn.set_active(true)));
    }

    // ── Window actions for keyboard shortcuts ──────────────────────────────
    {
        // win.import — Ctrl+I
        let db         = db_path.clone();
        let toast_fn   = make_toast.clone();
        let at         = active_tag.clone();
        let lp         = left.clone();
        let import_act = gtk4::gio::SimpleAction::new("import", None);
        import_act.connect_activate(clone!(@weak window, @weak lt_model => move |_, _| {
            let db_inner    = db.clone();
            let toast_inner = toast_fn.clone();
            let lp_inner    = lp.clone();
            dialogs::show_import_dialog(
                window.upcast_ref::<gtk4::Window>(),
                db.clone(),
                clone!(@weak lt_model, @strong db_inner, @strong at => move || {
                    lp_inner.clear_filter_highlights();   // post-import shows the quick-filtered collection
                    *at.borrow_mut() = None;   // (quick filters like stars/colours stay armed)
                    lighttable::lighttable_load_from_db(&lt_model, &db_inner);
                }),
                toast_inner,
            );
        }));
        window.add_action(&import_act);

        // win.export-selected — Ctrl+E
        let toast_fn2   = make_toast.clone();
        let export_act  = gtk4::gio::SimpleAction::new("export-selected", None);
        let export_db2  = db_path.clone();
        // Same as the toolbar button: resolve through the shown model, or Ctrl+E
        // exports the wrong file under culling.
        export_act.connect_activate(clone!(@weak lt_selection, @weak window => move |_, _| {
            let paths: Vec<String> =
                lighttable::selected_path(&lt_selection).into_iter().collect();
            dialogs::show_export_dialog(
                window.upcast_ref::<gtk4::Window>(),
                paths,
                None, // no fixed edit — each image's edit is loaded from the catalog
                Some(export_db2.clone()),
                toast_fn2.clone(),
            );
        }));
        window.add_action(&export_act);
    }

    // ── Wire toast overlay + present ───────────────────────────────────────
    toast_overlay.set_child(Some(&nav));
    window.set_child(Some(&toast_overlay));
    // Maximize so the window always fills the display. In the KasmVNC container
    // the framebuffer is resized dynamically to the connecting browser's
    // viewport; without maximizing, the window keeps its fixed default size in
    // the top-left corner surrounded by black, which reads as "no main screen".
    // openbox (and any EWMH WM) honours the maximized hint and re-fits the
    // window when the framebuffer is later resized. The default_width/height
    // above remain the fallback for WMs that don't honour maximize.
    // Flush any in-progress metadata edit before the window goes away. GTK4 does
    // not promise a focus-leave during teardown, so relying on the entry handlers
    // alone would drop the last field the user typed — the same
    // "persist-only-on-close" trap the darkroom view hit once already.
    {
        let right = right.clone();
        window.connect_close_request(move |_| {
            right.flush_metadata_edits();
            gtk4::glib::Propagation::Proceed
        });
    }

    window.maximize();
    window.present();
}

#[cfg(test)]
mod tests {
    use super::{collapsed_token, panel_toggle_key, parse_collapsed_token, PanelSide,
                PANEL_WIDTH_DEFAULT, PANEL_WIDTH_MAX, PANEL_WIDTH_MIN};
    use gtk4::gdk::Key;

    #[test]
    fn collapsed_token_round_trips() {
        for collapsed in [true, false] {
            assert_eq!(
                parse_collapsed_token(collapsed_token(collapsed)),
                Some(collapsed),
                "round-trip {collapsed}",
            );
        }
    }

    #[test]
    fn unknown_collapsed_token_is_no_opinion() {
        // A corrupt or hand-edited pref must never hide a panel the user then has to
        // guess how to get back: `None` leaves it shown.
        for tok in ["", "1", "0", "true", "collapsed", "HIDDEN", "shown ish"] {
            assert_eq!(parse_collapsed_token(tok), None, "token {tok:?}");
        }
        // Surrounding whitespace is tolerated, though — it round-trips a value that
        // was written correctly and then padded.
        assert_eq!(parse_collapsed_token(" hidden\n"), Some(true));
    }

    #[test]
    fn panel_keys_map_to_their_side_in_either_case() {
        assert_eq!(panel_toggle_key(Key::l), Some(PanelSide::Left));
        assert_eq!(panel_toggle_key(Key::L), Some(PanelSide::Left));
        assert_eq!(panel_toggle_key(Key::r), Some(PanelSide::Right));
        assert_eq!(panel_toggle_key(Key::R), Some(PanelSide::Right));
    }

    #[test]
    fn panel_keys_dont_steal_the_lighttables_own_shortcuts() {
        // Ratings are digits, colour labels are F1–F5, the full preview is `f`, and
        // culling pages with the arrows. None of them may map to a panel toggle.
        for k in [Key::f, Key::F, Key::_0, Key::_5, Key::F1, Key::F5,
                  Key::Left, Key::Right, Key::Home, Key::End,
                  Key::Return, Key::BackSpace, Key::Delete,
                  Key::Escape, Key::space] {
            assert_eq!(panel_toggle_key(k), None, "key {k:?}");
        }
    }

    #[test]
    fn default_expand_width_is_grabbable() {
        // The width a never-dragged panel expands to has to sit inside the range a
        // restored width is clamped into, or the first expand would land somewhere
        // the user can't drag back.
        assert!(PANEL_WIDTH_DEFAULT >= PANEL_WIDTH_MIN);
        assert!(PANEL_WIDTH_DEFAULT <= PANEL_WIDTH_MAX);
    }
}
