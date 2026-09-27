//! Print composer: print-to-PDF (u1) plus physical printing through the
//! native GTK print dialog (u9; parity audit 3.4, print second leg).
//!
//! Scope: compose one image per page on a chosen paper size / orientation /
//! margin and render it either to a PDF file ([`gtk4::PrintOperation`] in
//! Export mode) or to a physical printer (the same operation in PrintDialog
//! mode — GTK owns printer discovery, so there is no CUPS-direct
//! enumeration code here). No ICC/print-profile handling and no saved print
//! layout templates — both stay out of scope.
//!
//! The file follows the established pure-model-then-widget discipline (see
//! [`crate::export`]): all layout math ([`PaperSize`], [`Orientation`],
//! [`clamp_margin_mm`], [`layout_rect`]) is GTK-free and unit-tested headless;
//! the composer widget over it is just the GTK control surface.
//!
//! Image decoding reuses exactly the lighttable full preview's decode path
//! (see `lighttable::full_preview`): camera raws go through
//! [`crate::raw_preview::decode_raw_preview`] (demosaic + white balance +
//! linear downscale) followed by [`crate::preview::render_linear_to_srgb8`] at
//! default ("as shot") params; standard raster formats go through
//! gdk-pixbuf, the same source the grid thumbnails show. Nothing here invents
//! a new decoder — [`decode_page_async`] just repackages those two branches
//! for print use (a bounded longest side, no on-screen widget involved).

use adw::prelude::*;
use gtk4::glib;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

// ── Pure layout model (GTK-free, headless-tested) ────────────────────────────

/// Millimetres per inch; paper sizes below are defined from inch standards.
const MM_PER_INCH: f64 = 25.4;
/// PDF points per millimetre (72 pt per inch). Used to map [`layout_rect`]'s
/// millimetre output onto the print context, whose unit is set to Points.
const PT_PER_MM: f64 = 72.0 / MM_PER_INCH;
/// Longest side, in pixels, that [`decode_page_async`] decodes to. Bounds raw
/// demosaic memory (a 3000 px linear-RGBA f32 frame is ~150 MB transient) while
/// staying sharp enough for an A4 page at ~250 dpi.
const PRINT_DECODE_MAX_DIM: usize = 3000;

/// Printable paper sizes, in portrait orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaperSize {
    A4,
    Letter,
    Photo10x15,
    Photo13x18,
}

impl PaperSize {
    /// All sizes in combo order (drives the paper row's `StringList`).
    pub const ALL: [PaperSize; 4] =
        [PaperSize::A4, PaperSize::Letter, PaperSize::Photo10x15, PaperSize::Photo13x18];

    /// Human label for the paper combo row.
    pub fn label(self) -> &'static str {
        match self {
            PaperSize::A4 => "A4 (210 × 297 mm)",
            PaperSize::Letter => "US Letter (8.5 × 11 in)",
            PaperSize::Photo10x15 => "Photo 10 × 15 cm",
            PaperSize::Photo13x18 => "Photo 13 × 18 cm",
        }
    }

    /// Map a combo-row index to a size (out-of-range → A4, the first entry).
    /// Keep in sync with [`PaperSize::ALL`].
    pub fn from_index(i: u32) -> PaperSize {
        match i {
            0 => PaperSize::A4,
            1 => PaperSize::Letter,
            2 => PaperSize::Photo10x15,
            3 => PaperSize::Photo13x18,
            _ => PaperSize::A4,
        }
    }

    /// Portrait `(width, height)` in millimetres.
    pub fn dims_mm(self) -> (f64, f64) {
        match self {
            PaperSize::A4 => (210.0, 297.0),
            PaperSize::Letter => (8.5 * MM_PER_INCH, 11.0 * MM_PER_INCH),
            PaperSize::Photo10x15 => (100.0, 150.0),
            PaperSize::Photo13x18 => (130.0, 180.0),
        }
    }
}

/// Page orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    Portrait,
    Landscape,
}

impl Orientation {
    /// Both orientations in combo order.
    pub const ALL: [Orientation; 2] = [Orientation::Portrait, Orientation::Landscape];

    /// Human label for the orientation combo row.
    pub fn label(self) -> &'static str {
        match self {
            Orientation::Portrait => "Portrait",
            Orientation::Landscape => "Landscape",
        }
    }

    /// Map a combo-row index to an orientation (out-of-range → Portrait).
    pub fn from_index(i: u32) -> Orientation {
        match i {
            1 => Orientation::Landscape,
            _ => Orientation::Portrait,
        }
    }
}

/// Page `(width, height)` in millimetres for a paper size + orientation.
pub fn page_size_mm(paper: PaperSize, orient: Orientation) -> (f64, f64) {
    let (w, h) = paper.dims_mm();
    match orient {
        Orientation::Portrait => (w, h),
        Orientation::Landscape => (h, w),
    }
}

/// Clamp a uniform margin (mm) to `[0, min(page_w, page_h) / 2]`: negative
/// margins become 0 and margins past half the page clamp to exactly half, so
/// the content rect below can shrink to zero but never goes negative.
pub fn clamp_margin_mm(margin_mm: f64, page_w: f64, page_h: f64) -> f64 {
    margin_mm.max(0.0).min(page_w.min(page_h) / 2.0)
}

/// A rectangle in page space (millimetres from the page's top-left).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Fit an image of aspect `image_aspect` (width / height) into the page's
/// content box (page minus the uniform `margin_mm`), preserving aspect and
/// centring the result — the contain rule. A non-positive, infinite, or NaN
/// aspect, or a margin-clamped empty content box, yields the content box
/// itself; the result is never negative in any dimension.
pub fn layout_rect(page_w: f64, page_h: f64, margin_mm: f64, image_aspect: f64) -> Rect {
    let m = clamp_margin_mm(margin_mm, page_w, page_h);
    let (cx, cy) = (m, m);
    let (cw, ch) = ((page_w - 2.0 * m).max(0.0), (page_h - 2.0 * m).max(0.0));
    // NaN aspect is caught by `is_finite` below; spelled `<=` (not `!>`)
    // per clippy::neg_cmp_op_on_partial_ord — equivalent here.
    if image_aspect <= 0.0 || !image_aspect.is_finite() || cw <= 0.0 || ch <= 0.0 {
        return Rect { x: cx, y: cy, w: cw, h: ch };
    }
    let (w, h) = if cw / ch > image_aspect {
        (ch * image_aspect, ch) // height-bound
    } else {
        (cw, cw / image_aspect) // width-bound
    };
    Rect { x: cx + (cw - w) / 2.0, y: cy + (ch - h) / 2.0, w, h }
}

/// Default PDF file name for the save chooser: the first image's stem with a
/// `.pdf` extension, falling back to `print.pdf` for an empty selection.
pub fn default_pdf_name(paths: &[String]) -> String {
    paths
        .first()
        .and_then(|p| {
            std::path::Path::new(p).file_stem().and_then(|s| s.to_str())
        })
        .map(|s| format!("{s}.pdf"))
        .unwrap_or_else(|| "print.pdf".to_string())
}

// ── Physical-print helpers (u9; GTK-free except the action map, headless) ──

/// Printer hardware (non-printable) margins in millimetres, one per side.
/// Mirrors the C print view's `printer.hw_margin_*` fields
/// (`src/views/print.c`). Unlike C's borderless-driver path (which flags
/// the driver and lets it handle the area), the GTK flow conservatively
/// insets the uniform content box to clear the worst side — never printing
/// into non-printable area.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HwMarginsMm {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
}

impl HwMarginsMm {
    /// Zero margins: no printer involved (PDF export, or hardware margins
    /// unknown — the draw handler falls back to this when the print context
    /// reports none).
    pub const ZERO: HwMarginsMm = HwMarginsMm { top: 0.0, bottom: 0.0, left: 0.0, right: 0.0 };

    /// Widest single side. The composer margin is uniform, so the effective
    /// margin must clear every side at once.
    pub fn max_side(self) -> f64 {
        self.top.max(self.bottom).max(self.left).max(self.right)
    }
}

/// Points to millimetres. The print operation runs in Points, so the print
/// context reports hardware margins in points; layout math is in millimetres.
pub fn pt_to_mm(pt: f64) -> f64 {
    pt / PT_PER_MM
}

/// Effective uniform margin (mm): the user's margin raised — never lowered —
/// to clear the printer's widest hardware margin, then clamped to the page
/// exactly like [`clamp_margin_mm`]. A zero `hw` (PDF export or unknown
/// hardware margins) reduces to `clamp_margin_mm` unchanged.
pub fn effective_margin_mm(user_mm: f64, hw: HwMarginsMm, page_w: f64, page_h: f64) -> f64 {
    clamp_margin_mm(user_mm.max(hw.max_side()), page_w, page_h)
}

/// Select the [`gtk4::PrintOperationAction`] for a destination: the native
/// print dialog for a physical printer, Export mode for print-to-PDF.
/// A plain enum-to-enum map — constructing the values needs no display, so
/// this stays headless-tested like the rest of the pure model.
pub fn operation_action(want_dialog: bool) -> gtk4::PrintOperationAction {
    if want_dialog {
        gtk4::PrintOperationAction::PrintDialog
    } else {
        gtk4::PrintOperationAction::Export
    }
}

// ── Decode (the full preview's path, repackaged for print) ───────────────────

/// Decode one print page to an sRGB pixbuf, reusing exactly the lighttable
/// full preview's two branches: raws via `decode_raw_preview` (demosaic +
/// white balance + linear downscale) + `render_linear_to_srgb8` at default
/// params; standard formats via gdk-pixbuf. The longest side is bounded by
/// [`PRINT_DECODE_MAX_DIM`]. `None` when the file cannot be decoded.
///
/// Must be called from the main thread (the pixbuf half is not `Send`); the
/// blocking halves run on the thread pool internally.
async fn decode_page_async(path: String) -> Option<gtk4::gdk_pixbuf::Pixbuf> {
    use gtk4::gdk_pixbuf::{Colorspace, Pixbuf, PixbufLoader};
    if crate::raw_preview::is_raw_path(&path) {
        // Raw branch: every intermediate is an owned `Send` buffer, so the
        // demosaic + downscale + sRGB encode can leave the main thread — the
        // same split the full preview uses.
        let p = path.clone();
        let frame = gtk4::gio::spawn_blocking(move || {
            crate::raw_preview::decode_raw_preview(&p, PRINT_DECODE_MAX_DIM).map(|rp| {
                let bytes = crate::preview::render_linear_to_srgb8(
                    &rp.pixels,
                    rp.width,
                    rp.height,
                    &crate::preview::PreviewParams::default(),
                );
                (rp.width, rp.height, bytes)
            })
        })
        .await
        .ok()
        .flatten()?;
        let (w, h, bytes) = frame;
        Some(Pixbuf::from_bytes(
            &glib::Bytes::from_owned(bytes),
            Colorspace::Rgb,
            false,
            8,
            w as i32,
            h as i32,
            w as i32 * 3,
        ))
    } else {
        // Raster branch: only the *read* goes off-thread (`Pixbuf` is not
        // `Send`, so it can't cross back from a worker — the full preview
        // splits the work the same way). Decodes at a bounded size so a large
        // JPEG never materialises at full size just to be downscaled.
        let p = path.clone();
        let data =
            gtk4::gio::spawn_blocking(move || std::fs::read(&p).ok()).await.ok().flatten()?;
        let loader = PixbufLoader::new();
        let max = PRINT_DECODE_MAX_DIM as i32;
        loader.connect_size_prepared(move |loader, w, h| {
            let longest = w.max(h);
            if longest > max {
                // One scale factor on both axes: `set_size` does NOT preserve
                // aspect ratio for you.
                let scale = f64::from(max) / f64::from(longest);
                loader.set_size(
                    ((f64::from(w) * scale) as i32).max(1),
                    ((f64::from(h) * scale) as i32).max(1),
                );
            }
        });
        // Both unconditional: a loader finalized without `close()` emits a
        // g_warning, so an early return on a rejected header would print one
        // on every page step that lands on that file.
        let _ = loader.write(&data);
        let _ = loader.close();
        loader.pixbuf()
    }
}

/// The file's bare name for user-facing messages.
fn file_display_name(path: &str) -> &str {
    std::path::Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path)
}

// ── Composer widget ──────────────────────────────────────────────────────────

/// Shared composer state: the control selections plus one decoded pixbuf per
/// page (`None` until decoded or when undecodable — decode-once, then cached).
struct Composer {
    paths: Vec<String>,
    paper: Rc<Cell<u32>>,
    orient: Rc<Cell<u32>>,
    margin: Rc<Cell<f64>>,
    page: Rc<Cell<usize>>,
    pixbufs: Rc<RefCell<Vec<Option<gtk4::gdk_pixbuf::Pixbuf>>>>,
    area: gtk4::DrawingArea,
    status: gtk4::Label,
}

impl Composer {
    fn settings(&self) -> (PaperSize, Orientation, f64) {
        (
            PaperSize::from_index(self.paper.get()),
            Orientation::from_index(self.orient.get()),
            self.margin.get(),
        )
    }

    /// Decode page `idx` unless cached, then redraw. Failures land in the
    /// status label, never as a blank preview the user has to guess about.
    fn request_page(self: &Rc<Self>, idx: usize) {
        let this = self.clone();
        if idx >= this.paths.len() {
            return;
        }
        if this.pixbufs.borrow()[idx].is_some() {
            this.area.queue_draw();
            return;
        }
        let path = this.paths[idx].clone();
        this.status.set_label(&format!("Decoding {}…", file_display_name(&path)));
        let pixbufs = this.pixbufs.clone();
        let area = this.area.downgrade();
        let status = this.status.downgrade();
        glib::spawn_future_local(async move {
            let pb = decode_page_async(path.clone()).await;
            if let Some(pb) = pb {
                pixbufs.borrow_mut()[idx] = Some(pb);
                if let Some(s) = status.upgrade() {
                    s.set_label("");
                }
            } else if let Some(s) = status.upgrade() {
                s.set_label(&format!("No preview available for {}", file_display_name(&path)));
            }
            if let Some(a) = area.upgrade() {
                a.queue_draw();
            }
        });
    }
}

/// Build the [`gtk4::PrintOperation`] shared by both destinations (u9): the
/// default page setup carries the composer's paper + orientation with zero
/// GTK margins (our own margin math owns the content box), and the draw
/// handler reads the EFFECTIVE page setup back from the print context via
/// `PrintContext::page_setup` — under PrintDialog that is what the user
/// picked in the dialog (paper/orientation may differ from the composer),
/// under Export it is this same default. Hardware margins come from
/// `PrintContext::hard_margins` in the operation's unit (Points — converted
/// with [`pt_to_mm`]); when the context reports none (PDF export),
/// [`HwMarginsMm::ZERO`] keeps the export path identical to u1. With no
/// printers configured GTK shows an empty printer list in its own dialog —
/// no crash path, nothing to guard here.
fn build_operation(
    paper: PaperSize,
    orient: Orientation,
    user_margin_mm: f64,
    pages: Rc<Vec<gtk4::gdk_pixbuf::Pixbuf>>,
) -> gtk4::PrintOperation {
    let op = gtk4::PrintOperation::new();
    op.set_job_name("darkroom print");
    op.set_unit(gtk4::Unit::Points);
    let setup = gtk4::PageSetup::new();
    // Canonical portrait dims here: GtkPageSetup applies the orientation on
    // top of the stored paper size, so passing already-oriented dims would
    // double-rotate landscape output (u1 review fix — kept here).
    let (sw_mm, sh_mm) = paper.dims_mm();
    let size =
        gtk4::PaperSize::new_custom("c41-print", paper.label(), sw_mm, sh_mm, gtk4::Unit::Mm);
    setup.set_paper_size(&size);
    setup.set_orientation(match orient {
        Orientation::Portrait => gtk4::PageOrientation::Portrait,
        Orientation::Landscape => gtk4::PageOrientation::Landscape,
    });
    setup.set_top_margin(0.0, gtk4::Unit::Points);
    setup.set_bottom_margin(0.0, gtk4::Unit::Points);
    setup.set_left_margin(0.0, gtk4::Unit::Points);
    setup.set_right_margin(0.0, gtk4::Unit::Points);
    op.set_default_page_setup(Some(&setup));
    let n_draw = pages.len();
    op.connect_begin_print(move |op, _| {
        op.set_n_pages(n_draw as i32);
    });
    op.connect_draw_page(move |_, ctx, nr| {
        let Some(pb) = pages.get(nr as usize) else { return };
        let cr = ctx.cairo_context();
        cr.set_source_rgb(1.0, 1.0, 1.0);
        let _ = cr.paint();
        // Effective geometry: what the dialog (or the export default) settled
        // on. PageSetup widths take orientation into account, so no manual
        // swap is needed here.
        let effective = ctx.page_setup();
        let ew = effective.page_width(gtk4::Unit::Mm);
        let eh = effective.page_height(gtk4::Unit::Mm);
        let hw = ctx
            .hard_margins()
            .map(|(t, b, l, r)| HwMarginsMm {
                top: pt_to_mm(t),
                bottom: pt_to_mm(b),
                left: pt_to_mm(l),
                right: pt_to_mm(r),
            })
            .unwrap_or(HwMarginsMm::ZERO);
        let m = effective_margin_mm(user_margin_mm, hw, ew, eh);
        let aspect = f64::from(pb.width()) / f64::from(pb.height().max(1));
        let r = layout_rect(ew, eh, m, aspect);
        let _ = cr.save();
        cr.translate(r.x * PT_PER_MM, r.y * PT_PER_MM);
        cr.scale(
            r.w * PT_PER_MM / f64::from(pb.width()),
            r.h * PT_PER_MM / f64::from(pb.height().max(1)),
        );
        cr.set_source_pixbuf(pb, 0.0, 0.0);
        let _ = cr.paint();
        let _ = cr.restore();
    });
    op
}

/// Build the print composer page for `paths` (one image per page).
/// Tag is `"print"` — deliberately slash-free, so the lighttable's `popped`
/// handler (which re-syncs grid cells only for tags containing `/`) ignores it.
pub fn print_page(paths: Vec<String>) -> adw::NavigationPage {
    use gtk4::gdk::prelude::GdkCairoContextExt;

    let n = paths.len();
    let composer = Rc::new(Composer {
        paths,
        paper: Rc::new(Cell::new(0)),
        orient: Rc::new(Cell::new(0)),
        margin: Rc::new(Cell::new(10.0)),
        page: Rc::new(Cell::new(0)),
        pixbufs: Rc::new(RefCell::new(vec![None; n])),
        area: gtk4::DrawingArea::new(),
        status: gtk4::Label::new(None),
    });

    let content = gtk4::Box::new(gtk4::Orientation::Vertical, 12);
    content.set_margin_start(12);
    content.set_margin_end(12);
    content.set_margin_top(12);
    content.set_margin_bottom(12);

    // ── Page preview: grey canvas, white page, image fit into layout_rect ──
    composer.area.set_hexpand(true);
    composer.area.set_vexpand(true);
    composer.area.set_content_width(480);
    composer.area.set_content_height(360);
    {
        let c = composer.clone();
        composer.area.set_draw_func(move |_, cr, aw, ah| {
            // Canvas backdrop.
            cr.set_source_rgb(0.42, 0.42, 0.42);
            let _ = cr.paint();
            let (paper, orient, margin) = c.settings();
            let (pw, ph) = page_size_mm(paper, orient);
            let pad = 16.0;
            let s = ((f64::from(aw) - 2.0 * pad) / pw)
                .min((f64::from(ah) - 2.0 * pad) / ph)
                .max(0.0);
            let (ox, oy) = ((f64::from(aw) - pw * s) / 2.0, (f64::from(ah) - ph * s) / 2.0);
            // White page.
            cr.set_source_rgb(1.0, 1.0, 1.0);
            cr.rectangle(ox, oy, pw * s, ph * s);
            let _ = cr.fill();
            // Image fit into the layout rect, if this page decoded.
            let bufs = c.pixbufs.borrow();
            if let Some(pb) = bufs.get(c.page.get()).and_then(|p| p.as_ref()) {
                let aspect = f64::from(pb.width()) / f64::from(pb.height().max(1));
                let r = layout_rect(pw, ph, margin, aspect);
                let _ = cr.save();
                cr.translate(ox + r.x * s, oy + r.y * s);
                cr.scale(
                    r.w * s / f64::from(pb.width()),
                    r.h * s / f64::from(pb.height().max(1)),
                );
                cr.set_source_pixbuf(pb, 0.0, 0.0);
                let _ = cr.paint();
                let _ = cr.restore();
            }
        });
    }
    content.append(&composer.area);

    // ── Paper controls (PreferencesGroup + ComboRow/SpinRow, like export) ──
    let paper_group = adw::PreferencesGroup::builder().title("Paper").build();

    let paper_row = adw::ComboRow::builder().title("Paper size").build();
    let paper_labels: Vec<&str> = PaperSize::ALL.iter().map(|p| p.label()).collect();
    paper_row.set_model(Some(&gtk4::StringList::new(&paper_labels)));
    paper_group.add(&paper_row);

    let orient_row = adw::ComboRow::builder().title("Orientation").build();
    let orient_labels: Vec<&str> = Orientation::ALL.iter().map(|o| o.label()).collect();
    orient_row.set_model(Some(&gtk4::StringList::new(&orient_labels)));
    paper_group.add(&orient_row);

    let margin_row = adw::SpinRow::builder().title("Margins (mm)").build();
    margin_row.set_adjustment(Some(&gtk4::Adjustment::new(10.0, 0.0, 100.0, 1.0, 5.0, 0.0)));
    paper_group.add(&margin_row);
    content.append(&paper_group);

    // ── Page selector: one image per page ──────────────────────────────────
    let pages_group = adw::PreferencesGroup::builder()
        .title("Pages")
        .description("One image per page.")
        .build();
    let page_row = adw::SpinRow::builder().title("Page").build();
    page_row.set_adjustment(Some(&gtk4::Adjustment::new(
        1.0,
        1.0,
        n.max(1) as f64,
        1.0,
        1.0,
        0.0,
    )));
    page_row.set_sensitive(n > 1);
    pages_group.add(&page_row);
    content.append(&pages_group);

    {
        let c = composer.clone();
        paper_row.connect_selected_notify(move |row| {
            c.paper.set(row.selected());
            c.area.queue_draw();
        });
        let c = composer.clone();
        orient_row.connect_selected_notify(move |row| {
            c.orient.set(row.selected());
            c.area.queue_draw();
        });
        let c = composer.clone();
        margin_row.connect_value_notify(move |row| {
            c.margin.set(row.value());
            c.area.queue_draw();
        });
        let c = composer.clone();
        page_row.connect_value_notify(move |row| {
            let idx = (row.value() as usize).saturating_sub(1).min(n.saturating_sub(1));
            c.page.set(idx);
            c.request_page(idx);
        });
    }

    // ── Print + Export + status ────────────────────────────────────────────
    let actions = gtk4::Box::new(gtk4::Orientation::Horizontal, 12);
    actions.set_halign(gtk4::Align::Center);
    let print_btn = gtk4::Button::with_label("Print…");
    print_btn.set_sensitive(n > 0);
    actions.append(&print_btn);
    let export_btn = gtk4::Button::with_label("Export PDF…");
    export_btn.add_css_class("suggested-action");
    export_btn.set_sensitive(n > 0);
    actions.append(&export_btn);
    content.append(&actions);

    composer.status.set_wrap(true);
    composer.status.set_halign(gtk4::Align::Center);
    composer.status.add_css_class("dim-label");
    if n == 0 {
        composer.status.set_label("No images selected.");
    }
    content.append(&composer.status);

    // ── Export PDF wiring ──────────────────────────────────────────────────
    // Print-to-file via PrintOperation in Export mode with an export filename:
    // no print dialog, no printer enumeration, no CUPS. Page setup carries the
    // paper size + orientation (zero GTK margins — our own margin math owns the
    // content box); each successfully decoded image draws exactly one page.
    {
        let c = composer.clone();
        let page_w = content.downgrade();
        let status_w = composer.status.downgrade();
        // Both buttons gate each other (mirrors the Print handler): an
        // export in flight must not race a dialog-backed print over the
        // shared status label and preview cache.
        let other_w = print_btn.downgrade();
        export_btn.connect_clicked(move |btn| {
            let btn_w = btn.downgrade();
            // Shadow-clone for the single-shot file-chooser callback below:
            // `other_w` belongs to this `Fn` closure and cannot move into it.
            let other_w = other_w.clone();
            let win: Option<gtk4::Window> = page_w
                .upgrade()
                .and_then(|w| w.root())
                .and_then(|r| r.downcast::<gtk4::Window>().ok());
            let chooser = gtk4::FileDialog::builder()
                .title("Export PDF")
                .initial_name(default_pdf_name(&c.paths))
                .build();
            let c2 = c.clone();
            let status_w2 = status_w.clone();
            let win2 = win.clone();
            chooser.save(
                win.as_ref(),
                gtk4::gio::Cancellable::NONE,
                move |result| {
                    let dest = match result.ok().and_then(|f| f.path()) {
                        Some(p) => p,
                        None => return, // dismissed or non-local: nothing to do
                    };
                    let (paper, orient, margin) = c2.settings();
                    let paths = c2.paths.clone();
                    let pixbufs = c2.pixbufs.clone();
                    if let Some(b) = btn_w.upgrade() {
                        b.set_sensitive(false);
                    }
                    if let Some(b) = other_w.clone().upgrade() {
                        b.set_sensitive(false);
                    }
                    if let Some(s) = status_w2.upgrade() {
                        s.set_label("Rendering PDF…");
                    }
                    let status_w3 = status_w2.clone();
                    let btn_w2 = btn_w.clone();
                    let other_w2 = other_w.clone();
                    glib::spawn_future_local(async move {
                        // Decode every page (reusing the preview cache when the
                        // user already viewed it); failures are counted and
                        // reported, never silently dropped into a blank page.
                        let mut pages: Vec<gtk4::gdk_pixbuf::Pixbuf> = Vec::new();
                        let mut failed = 0usize;
                        for (i, path) in paths.iter().enumerate() {
                            let cached = pixbufs.borrow()[i].clone();
                            match cached {
                                Some(pb) => pages.push(pb),
                                None => match decode_page_async(path.clone()).await {
                                    Some(pb) => {
                                        pixbufs.borrow_mut()[i] = Some(pb.clone());
                                        pages.push(pb);
                                    }
                                    None => failed += 1,
                                },
                            }
                        }
                        let done = |msg: String| {
                            if let Some(s) = status_w3.upgrade() {
                                s.set_label(&msg);
                            }
                            if let Some(b) = btn_w2.upgrade() {
                                b.set_sensitive(true);
                            }
                            if let Some(b) = other_w2.upgrade() {
                                b.set_sensitive(true);
                            }
                        };
                        if pages.is_empty() {
                            done("Export failed: none of the images could be decoded.".to_string());
                            return;
                        }
                        let drawable = Rc::new(pages);
                        let n_draw = drawable.len();
                        let op = build_operation(paper, orient, margin, drawable);
                        op.set_export_filename(&dest);
                        match op.run(operation_action(false), win2.as_ref()) {
                            Ok(_) if failed == 0 => done(format!(
                                "Exported {} page(s) to {}",
                                n_draw,
                                dest.to_string_lossy()
                            )),
                            Ok(_) => done(format!(
                                "Exported {} of {} page(s) ({failed} failed) to {}",
                                n_draw,
                                n_draw + failed,
                                dest.to_string_lossy()
                            )),
                            Err(e) => done(format!("Export failed: {e}")),
                        }
                    });
                },
            );
        });
    }

    // ── Physical printing via the native GTK dialog (u9) ───────────────────
    // The same operation as Export, run in PrintDialog mode: GTK enumerates
    // printers (CUPS) and shows its own dialog — an empty printer list
    // renders as an empty list there, never as our crash path. The
    // composer's paper/orientation/margins seed the dialog through the
    // default page setup; the shared draw handler re-reads the effective
    // setup, so dialog-side changes apply. Nothing is persisted.
    {
        let c = composer.clone();
        let page_w = content.downgrade();
        let status_w = composer.status.downgrade();
        let export_w = export_btn.downgrade();
        print_btn.connect_clicked(move |btn| {
            let win: Option<gtk4::Window> = page_w
                .upgrade()
                .and_then(|w| w.root())
                .and_then(|r| r.downcast::<gtk4::Window>().ok());
            let (paper, orient, margin) = c.settings();
            let paths = c.paths.clone();
            let pixbufs = c.pixbufs.clone();
            let btn_w = btn.downgrade();
            if let Some(b) = btn_w.upgrade() {
                b.set_sensitive(false);
            }
            if let Some(e) = export_w.upgrade() {
                e.set_sensitive(false);
            }
            if let Some(s) = status_w.upgrade() {
                s.set_label("Preparing pages…");
            }
            let status_w2 = status_w.clone();
            let btn_w2 = btn_w.clone();
            let export_w2 = export_w.clone();
            glib::spawn_future_local(async move {
                // Same decode-every-page pass as Export (preview cache first);
                // failures are counted and reported, never silently dropped.
                let mut pages: Vec<gtk4::gdk_pixbuf::Pixbuf> = Vec::new();
                let mut failed = 0usize;
                for (i, path) in paths.iter().enumerate() {
                    let cached = pixbufs.borrow()[i].clone();
                    match cached {
                        Some(pb) => pages.push(pb),
                        None => match decode_page_async(path.clone()).await {
                            Some(pb) => {
                                pixbufs.borrow_mut()[i] = Some(pb.clone());
                                pages.push(pb);
                            }
                            None => failed += 1,
                        },
                    }
                }
                let done = |msg: String| {
                    if let Some(s) = status_w2.upgrade() {
                        s.set_label(&msg);
                    }
                    if let Some(b) = btn_w2.upgrade() {
                        b.set_sensitive(true);
                    }
                    if let Some(e) = export_w2.upgrade() {
                        e.set_sensitive(true);
                    }
                };
                if pages.is_empty() {
                    done("Print failed: none of the images could be decoded.".to_string());
                    return;
                }
                let n_draw = pages.len();
                let op = build_operation(paper, orient, margin, Rc::new(pages));
                // Cancel is Ok(Cancel), not an error — only Apply means the
                // job went to the printer.
                match op.run(operation_action(true), win.as_ref()) {
                    Ok(gtk4::PrintOperationResult::Apply) if failed == 0 => {
                        done(format!("Sent {n_draw} page(s) to the printer"))
                    }
                    Ok(gtk4::PrintOperationResult::Apply) => done(format!(
                        "Sent {n_draw} of {} page(s) ({failed} failed)",
                        n_draw + failed
                    )),
                    // Cancel (and the unreachable-synchronous InProgress/Error
                    // variants) land here; only Apply reports a send.
                    Ok(_) => done("Print cancelled.".to_string()),
                    Err(e) => done(format!("Print failed: {e}")),
                }
            });
        });
    }

    let page = adw::NavigationPage::builder()
        .title("Print")
        .tag("print")
        .child(&content)
        .build();

    // Paint the first page as soon as it decodes.
    if n > 0 {
        composer.request_page(0);
    }
    page
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn paper_dims_and_orientation_are_exact() {
        assert_eq!(PaperSize::A4.dims_mm(), (210.0, 297.0));
        let (w, h) = PaperSize::Letter.dims_mm();
        assert!(approx(w, 215.9), "letter width {w}");
        assert!(approx(h, 279.4), "letter height {h}");
        assert_eq!(PaperSize::Photo10x15.dims_mm(), (100.0, 150.0));
        assert_eq!(PaperSize::Photo13x18.dims_mm(), (130.0, 180.0));
        // Portrait keeps dims; landscape swaps them.
        assert_eq!(page_size_mm(PaperSize::A4, Orientation::Portrait), (210.0, 297.0));
        assert_eq!(page_size_mm(PaperSize::A4, Orientation::Landscape), (297.0, 210.0));
        // Combo dispatch round-trips ALL; out-of-range falls back to the first.
        for (i, p) in PaperSize::ALL.iter().enumerate() {
            assert_eq!(PaperSize::from_index(i as u32), *p);
        }
        assert_eq!(PaperSize::from_index(99), PaperSize::A4);
        assert_eq!(Orientation::from_index(0), Orientation::Portrait);
        assert_eq!(Orientation::from_index(1), Orientation::Landscape);
        assert_eq!(Orientation::from_index(7), Orientation::Portrait);
    }

    #[test]
    fn a4_portrait_10mm_margins_with_3x2_locks_rect() {
        // Content box: x=10, y=10, w=190, h=277. Width-bound at 3:2
        // (190/1.5 = 126.67 < 277), centred vertically.
        let r = layout_rect(210.0, 297.0, 10.0, 3.0 / 2.0);
        let h = 190.0 / 1.5;
        assert!(approx(r.x, 10.0), "x {}", r.x);
        assert!(approx(r.w, 190.0), "w {}", r.w);
        assert!(approx(r.h, h), "h {}", r.h);
        assert!(approx(r.y, 10.0 + (277.0 - h) / 2.0), "y {}", r.y);
    }

    #[test]
    fn letter_landscape_zero_margin_square_image() {
        // Letter landscape: 279.4 × 215.9 mm; a square fills the height and
        // centres horizontally with 31.75 mm pillars each side.
        let (pw, ph) = page_size_mm(PaperSize::Letter, Orientation::Landscape);
        let r = layout_rect(pw, ph, 0.0, 1.0);
        assert!(approx(r.w, ph), "w {}", r.w);
        assert!(approx(r.h, ph), "h {}", r.h);
        assert!(approx(r.x, (pw - ph) / 2.0), "x {}", r.x);
        assert!(approx(r.y, 0.0), "y {}", r.y);
        assert!(approx(r.x, 31.75), "pillar {}", r.x);
    }

    #[test]
    fn tall_image_is_height_bound_and_centred() {
        // 2:3 portrait image on A4 portrait with 10 mm margins: content is
        // 190 × 277; height-bound (190/277 < 2/3), centred horizontally.
        let r = layout_rect(210.0, 297.0, 10.0, 2.0 / 3.0);
        assert!(approx(r.h, 277.0), "h {}", r.h);
        assert!(approx(r.w, 277.0 * 2.0 / 3.0), "w {}", r.w);
        assert!(approx(r.y, 10.0), "y {}", r.y);
        assert!(approx(r.x, 10.0 + (190.0 - 277.0 * 2.0 / 3.0) / 2.0), "x {}", r.x);
    }

    #[test]
    fn margin_clamping_never_yields_a_negative_rect() {
        // Negative margins behave as zero.
        assert_eq!(clamp_margin_mm(-5.0, 210.0, 297.0), 0.0);
        // Margins past half the short side clamp to exactly half: 105 mm.
        assert_eq!(clamp_margin_mm(200.0, 210.0, 297.0), 105.0);
        let r = layout_rect(210.0, 297.0, 200.0, 1.5);
        assert!(r.w >= 0.0 && r.h >= 0.0, "negative rect {r:?}");
        assert!(approx(r.x, 105.0), "x {}", r.x);
        assert!(approx(r.w, 0.0), "w {}", r.w);
        // Degenerate aspects fall back to the (clamped) content box itself.
        for aspect in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let r = layout_rect(210.0, 297.0, 10.0, aspect);
            assert_eq!(r, Rect { x: 10.0, y: 10.0, w: 190.0, h: 277.0 }, "aspect {aspect}");
        }
    }

    #[test]
    fn default_pdf_name_uses_first_stem() {
        assert_eq!(
            default_pdf_name(&["/a/b/IMG_1234.CR2".to_string()]),
            "IMG_1234.pdf"
        );
        assert_eq!(default_pdf_name(&[]), "print.pdf");
        assert_eq!(default_pdf_name(&["noext".to_string()]), "noext.pdf");
    }

    #[test]
    fn pt_to_mm_converts_print_points() {
        assert!(approx(pt_to_mm(72.0), MM_PER_INCH), "72pt {}", pt_to_mm(72.0));
        assert!(approx(pt_to_mm(PT_PER_MM), 1.0), "unit {}", pt_to_mm(PT_PER_MM));
        assert_eq!(pt_to_mm(0.0), 0.0);
    }

    #[test]
    fn hw_margins_raise_but_never_lower_the_margin() {
        let hw = HwMarginsMm { top: 3.0, bottom: 5.0, left: 4.0, right: 2.0 };
        // The uniform composer margin must clear every side at once.
        assert_eq!(hw.max_side(), 5.0);
        assert_eq!(HwMarginsMm::ZERO.max_side(), 0.0);
        // User margin below the widest hardware margin: hardware wins.
        assert_eq!(effective_margin_mm(2.0, hw, 210.0, 297.0), 5.0);
        // User margin above: unchanged.
        assert_eq!(effective_margin_mm(10.0, hw, 210.0, 297.0), 10.0);
        // Zero hardware margins (PDF export / unknown): plain clamp.
        assert_eq!(
            effective_margin_mm(10.0, HwMarginsMm::ZERO, 210.0, 297.0),
            clamp_margin_mm(10.0, 210.0, 297.0)
        );
        assert_eq!(effective_margin_mm(-3.0, HwMarginsMm::ZERO, 210.0, 297.0), 0.0);
        // The raised margin still clamps at half the short side, never below.
        assert_eq!(effective_margin_mm(200.0, hw, 210.0, 297.0), 105.0);
    }

    #[test]
    fn operation_action_selects_dialog_or_export() {
        assert_eq!(operation_action(true), gtk4::PrintOperationAction::PrintDialog);
        assert_eq!(operation_action(false), gtk4::PrintOperationAction::Export);
    }
}
