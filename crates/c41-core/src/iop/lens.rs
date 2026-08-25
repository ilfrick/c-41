//! Lens correction via liblensfun (lens.cc `DT_IOP_LENS_METHOD_LENSFUN` path).
//!
//! darktable's lens module resolves the image's camera and lens against the
//! lensfun database, builds an [`lfModifier`](c41_sys::lensfun::lfModifier)
//! for the buffer geometry, and runs two passes: a per-pixel colour scaling
//! (vignetting) and a per-channel coordinate warp (TCA + distortion +
//! geometry + scale) resampled with bilinear interpolation. This module is
//! the whole-frame Rust equivalent: one entry point ([`process`]) over an
//! interleaved f32 RGBA buffer, mirroring `_process_lf` (src/iop/lens.cc) at
//! roi x/y = 0, roi scale = 1 — C41 has no per-module ROI machinery, stages
//! see whole buffers.
//!
//! Database handling mirrors the C module's shape: one process-wide database
//! (darktable keeps it in `init_global`), lookups serialised by a mutex
//! (darktable's `plugin_threadsafe`), and resolved lenses are *borrowed*
//! pointers into that database, which stays loaded for the process lifetime.
//! The modifier itself is created per render inside [`process`].
//!
//! Documented deviations from the C module (slice 2 scope):
//! - **Embedded-metadata methods not ported** (`method != LENSFUN`, the
//!   `knots_dist`/`knots_vig`/`cor_rgb` spline paths): C41 records no
//!   lensfun-XML metadata in files; the GUI-driven lensfun path is what the
//!   UI exercises.
//! - **Custom TCA override not ported** (`tca_override` + R/B sliders): needs
//!   mutable access to calibration lists; defaults use db calibration like
//!   darktable does when the box is unchecked.
//! - **Monochrome TCA suppression lives in the caller**: darktable clears
//!   `MODIFY_FLAG_TCA` for monochrome sensors in commit_params using image
//!   metadata the stage does not carry; the UI passes flags already cleared.
//! - **Interpolation is bilinear** (darktable's default "warp" preference);
//!   C41 exposes no interpolation preference yet.

use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

use c41_sys::lensfun::*;

// ── Correction-selection bits (dt_iop_lens_modflag_t mirror) ─────────────────

/// `DT_IOP_LENS_MODIFY_FLAG_TCA` — correct lateral chromatic aberration.
pub const MODIFY_FLAG_TCA: i32 = 1;
/// `DT_IOP_LENS_MODIFY_FLAG_VIGNETTING`.
pub const MODIFY_FLAG_VIGNETTING: i32 = 1 << 1;
/// `DT_IOP_LENS_MODIFY_FLAG_DISTORTION`.
pub const MODIFY_FLAG_DISTORTION: i32 = 1 << 2;
/// `DT_IOP_LENS_MODFLAG_ALL` — darktable's dropdown default.
pub const MODFLAG_ALL: i32 = MODIFY_FLAG_DISTORTION | MODIFY_FLAG_TCA | MODIFY_FLAG_VIGNETTING;

/// `_modflags_to_lensfun_mods`: geometry and scale corrections are always
/// enabled in darktable; the three checkboxes map onto their LF_MODIFY bits.
fn to_lensfun_mods(modify_flags: i32) -> i32 {
    let mut mods = LF_MODIFY_GEOMETRY | LF_MODIFY_SCALE;
    if modify_flags & MODIFY_FLAG_DISTORTION != 0 {
        mods |= LF_MODIFY_DISTORTION;
    }
    if modify_flags & MODIFY_FLAG_VIGNETTING != 0 {
        mods |= LF_MODIFY_VIGNETTING;
    }
    if modify_flags & MODIFY_FLAG_TCA != 0 {
        mods |= LF_MODIFY_TCA;
    }
    mods
}

// ── Parameters (the GUI-facing subset of dt_iop_lens_params_t) ───────────────

/// User-facing lens-correction parameters. Field names track
/// `dt_iop_lens_params_t`; values arrive from the catalog/EXIF autodetect
/// (slice 3 wires that up) exactly as darktable's reload_defaults does.
#[derive(Clone, Debug, PartialEq)]
pub struct LensParams {
    pub camera_maker: String,
    pub camera_model: String,
    /// Lens name as stored by the catalog (matched with lensfun scoring).
    pub lens: String,
    /// Bitmask of [`MODIFY_FLAG_TCA`] / [`MODIFY_FLAG_VIGNETTING`] /
    /// [`MODIFY_FLAG_DISTORTION`] (darktable's modflag dropdown value).
    pub modify_flags: i32,
    /// Run the transform backwards (simulate rather than correct).
    pub inverse: bool,
    /// Manual scale applied after correction (autoscale writes this).
    pub scale: f32,
    /// Camera crop factor used for the normalised film plane.
    pub crop: f32,
    pub focal: f32,
    pub aperture: f32,
    pub distance: f32,
    /// Target projection (`lfLensType` constants) for geometry conversion.
    pub target_geom: i32,
}

impl Default for LensParams {
    fn default() -> Self {
        LensParams {
            camera_maker: String::new(),
            camera_model: String::new(),
            lens: String::new(),
            modify_flags: MODFLAG_ALL,
            inverse: false,
            scale: 1.0,
            crop: 1.0,
            focal: 50.0,
            aperture: 3.5,
            distance: 10.0,
            target_geom: LF_RECTILINEAR,
        }
    }
}

// ── Process-wide database ────────────────────────────────────────────────────

struct Db(*mut lfDatabase);
unsafe impl Send for Db {}
unsafe impl Sync for Db {}

static DB: OnceLock<Option<Db>> = OnceLock::new();
/// Serialises lookups (and the first load). darktable guards the same calls
/// with `plugin_threadsafe`; lensfun's find functions write match `Score`s
/// into the shared database objects, so concurrent lookups are not safe.
static DB_LOCK: Mutex<()> = Mutex::new(());

fn db() -> Option<&'static Db> {
    DB.get_or_init(|| unsafe {
        let db = lf_db_new();
        if db.is_null() {
            return None;
        }
        // LF_NO_DATABASE (nothing installed / no files found) still leaves a
        // usable empty database — lookups just return NULL, which callers
        // already handle as "no gear".
        let _err = lf_db_load(db);
        Some(Db(db))
    })
    .as_ref()
}

/// A camera resolved against the global database. `ptr` borrows from the
/// process-lifetime database; identity fields are copied out so callers can
/// display them without touching FFI again.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedCamera {
    pub(crate) ptr: *const lfCamera,
    pub maker: String,
    pub model: String,
    /// Mount id ("Canon EF", "Nikon F", …) — the key [`list_lenses`] filters
    /// the lens database by.
    pub mount: String,
    /// Crop factor relative to 35 mm (always defined for a match). This is
    /// what feeds `LensParams::crop` — darktable's commit_params takes it from
    /// the camera (`p->crop = cam->CropFactor`), never from the lens.
    pub crop_factor: f32,
}
unsafe impl Send for ResolvedCamera {}
unsafe impl Sync for ResolvedCamera {}
// PartialEq compares the borrowed pointer: two resolutions of the same gear
// hit the same database-owned object, so pointer equality IS identity.

/// A lens resolved against the global database (see [`ResolvedCamera`] for
/// the borrowing discipline). Calibration data stays inside lensfun; the
/// modifier reads it directly during [`process`].
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedLens {
    pub(crate) ptr: *const lfLens,
    pub maker: String,
    pub model: String,
    pub crop_factor: f32,
    /// `lfLensType` of the physical lens (drives the nan-check rule and
    /// geometry conversion decisions).
    pub lens_type: i32,
    pub min_focal: f32,
    pub max_focal: f32,
}
unsafe impl Send for ResolvedLens {}
unsafe impl Sync for ResolvedLens {}

fn cstr(s: &str) -> CString {
    // Interior NUL can only come from corrupted catalog strings; treat as
    // "no match" by mapping to an empty wildcard-free name that won't match.
    CString::new(s.replace('\0', "")).expect("NUL removed")
}

/// Resolve a camera by EXIF maker/model (exact strings, like
/// `lf_db_find_cameras`). Returns the best-scored match.
pub fn resolve_camera(maker: &str, model: &str) -> Option<ResolvedCamera> {
    let maker = cstr(maker);
    let model = cstr(model);
    let _g = DB_LOCK.lock().unwrap();
    unsafe {
        let list =
            lf_db_find_cameras(db()?.0, maker.as_ptr(), model.as_ptr());
        if list.is_null() {
            return None;
        }
        let cam = *list;
        let resolved = (!cam.is_null()).then(|| {
            let cam = &*cam;
            ResolvedCamera {
                ptr: cam,
                maker: str_of(cam.Maker),
                model: str_of(cam.Model),
                mount: str_of(cam.Mount),
                crop_factor: cam.CropFactor,
            }
        });
        lf_free(list as *mut _);
        resolved
    }
}

/// Does this database lens fit `mount`? (A lens lists every mount it works
/// on — adapters included, e.g. an M42 screw lens under "Canon EF".)
fn mount_fits(lens: &lfLens, mount: &str) -> bool {
    if mount.is_empty() || lens.Mounts.is_null() {
        return false;
    }
    unsafe {
        let mut m = lens.Mounts;
        while !(*m).is_null() {
            if str_of(*m) == mount {
                return true;
            }
            m = m.add(1);
        }
    }
    false
}

/// Resolve a lens for `camera` by exact identity: `maker` and `model` must
/// byte-equal the database entry's fields, and the lens must fit the camera's
/// mount. This round-trips [`list_lenses`]' structured pairs — deliberately
/// NOT `lf_db_find_lenses_hd`, whose substring scoring is a lossy "did you
/// mean" search (measured against this database it fails outright on ~half of
/// all real entries when fed their exact maker+model), so a persisted pick
/// could never be re-found through it.
pub fn resolve_lens(camera: &ResolvedCamera, maker: &str, model: &str) -> Option<ResolvedLens> {
    let _g = DB_LOCK.lock().unwrap();
    let db = db()?;
    unsafe {
        // The database holds several entries per model (one per mount family)
        // under identical Maker/Model strings; among the mount-compatible ones
        // prefer the crop factor closest to the camera's, so the picked entry
        // is independent of the database's enumeration order.
        let mut list = lf_db_get_lenses(db.0);
        let mut best: Option<(&lfLens, f32)> = None;
        while !(*list).is_null() {
            let l = &**list;
            if !camera.mount.is_empty()
                && str_of(l.Maker) == maker
                && str_of(l.Model) == model
                && mount_fits(l, &camera.mount)
            {
                let diff = (l.CropFactor - camera.crop_factor).abs();
                if best.map_or(true, |(_, d)| diff < d) {
                    best = Some((l, diff));
                }
            }
            list = list.add(1);
        }
        best.map(|(l, _)| ResolvedLens {
            ptr: l,
            maker: str_of(l.Maker),
            model: str_of(l.Model),
            crop_factor: l.CropFactor,
            lens_type: l.Type,
            min_focal: l.MinFocal,
            max_focal: l.MaxFocal,
        })
    }
}

/// Full resolution helper: camera by maker/model, then the lens by exact
/// identity (see [`resolve_lens`]). Both lens fields must be the database's
/// own spelling; an empty `lens_model` never matches anything.
pub fn resolve(
    cam_maker: &str,
    cam_model: &str,
    lens_maker: &str,
    lens_model: &str,
) -> Option<(ResolvedCamera, ResolvedLens)> {
    debug_assert!(!lens_model.is_empty(), "empty lens model matches nothing");
    let cam = resolve_camera(cam_maker, cam_model)?;
    let lens = resolve_lens(&cam, lens_maker, lens_model)?;
    Some((cam, lens))
}

/// Every camera in the database as sorted, deduplicated `(maker, model)`
/// pairs — the population of the UI's camera dropdown. The database holds one
/// entry per mount/variant of a model; display collapses those like
/// darktable's combo does.
pub fn list_cameras() -> Vec<(String, String)> {
    let Some(db) = db() else { return Vec::new() };
    unsafe {
        let mut list = lf_db_get_cameras(db.0);
        let mut out: Vec<(String, String)> = Vec::new();
        while !(*list).is_null() {
            let cam = &**list;
            out.push((str_of(cam.Maker), str_of(cam.Model)));
            list = list.add(1);
        }
        out.sort();
        out.dedup();
        out
    }
}

/// Display label for a lens: the model, prefixed with the maker unless the
/// model string already carries it. Display only — identity never round-trips
/// through this label; callers keep the structured `(maker, model)` pair and
/// re-resolve it with [`resolve_lens`].
pub fn lens_label(maker: &str, model: &str) -> String {
    if maker.is_empty() || model.starts_with(maker) {
        model.to_string()
    } else {
        format!("{maker} {model}")
    }
}

/// Lenses in the database as sorted structured `(maker, model)` pairs,
/// optionally restricted to lenses that fit `mount` (a
/// [`ResolvedCamera::mount`] value). The database's own enumeration order is
/// unspecified, so the result is explicitly sorted; render the pairs with
/// [`lens_label`]. Reads only identity fields (`Maker/Model/Mounts`) from
/// database-owned objects — never the `Score` that concurrent lookups write —
/// so no lock is needed.
///
/// Every returned pair round-trips exactly through [`resolve_lens`] for a
/// camera with that mount.
pub fn list_lenses(mount: Option<&str>) -> Vec<(String, String)> {
    let Some(db) = db() else { return Vec::new() };
    unsafe {
        let mut list = lf_db_get_lenses(db.0);
        let mut out: Vec<(String, String)> = Vec::new();
        while !(*list).is_null() {
            let lens = &**list;
            let fits = match mount {
                None => true,
                Some(m) => mount_fits(lens, m),
            };
            if fits {
                out.push((str_of(lens.Maker), str_of(lens.Model)));
            }
            list = list.add(1);
        }
        out.sort();
        // Adapter-listed lenses can appear once per mount family under the same
        // name; the display collapses them like the camera list does.
        out.dedup();
        out
    }
}

/// Read one C string. For `lfLens`/`lfCamera` name fields this deliberately
/// reads the raw `lfMLstr` pointer — lensfun's multi-language struct starts
/// with the default-language string, so this yields the same bytes regardless
/// of locale, which is what the identity round-trip (list → persist →
/// [`resolve_lens`]) depends on. Switching to locale-aware `lf_mlstr_get`
/// would silently unresolve every persisted pick.
unsafe fn str_of(p: *const std::ffi::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

// ── Modifier RAII ────────────────────────────────────────────────────────────

struct Modifier(*mut lfModifier);
impl Modifier {
    fn new(lens: &ResolvedLens, crop: f32, width: i32, height: i32) -> Option<Self> {
        let m = unsafe { lf_modifier_new(lens.ptr, crop, width, height) };
        (!m.is_null()).then_some(Modifier(m))
    }
    fn initialize(
        &self,
        lens: &ResolvedLens,
        p: &LensParams,
        mods: i32,
        reverse: bool,
    ) -> i32 {
        unsafe {
            lf_modifier_initialize(
                self.0,
                lens.ptr,
                LF_PF_F32,
                p.focal,
                p.aperture,
                p.distance,
                p.scale,
                p.target_geom,
                mods,
                reverse as i32,
            )
        }
    }
    fn auto_scale(&self, reverse: bool) -> f32 {
        unsafe { lf_modifier_get_auto_scale(self.0, reverse as i32) }
    }
}
impl Drop for Modifier {
    fn drop(&mut self) {
        unsafe { lf_modifier_destroy(self.0) }
    }
}

/// The scale factor that keeps the corrected frame fully inside the image
/// (darktable's `_get_autoscale_lf` → `GetAutoScale`). The UI offers this as
/// the "automatic" scale; applying it is just `params.scale = autoscale(..)`.
pub fn autoscale(lens: &ResolvedLens, p: &LensParams, width: usize, height: usize) -> Option<f32> {
    // Modifier construction walks the shared database's calibration lists, so —
    // exactly like darktable, whose `_get_autoscale_lf` holds the plugin mutex
    // across Find + modifier + GetAutoScale — the whole build is serialized.
    let _g = DB_LOCK.lock().unwrap();
    let m = Modifier::new(lens, p.crop, width as i32, height as i32)?;
    let mods = to_lensfun_mods(p.modify_flags);
    // Measured from the unzoomed frame: _get_autoscale_lf builds dummy params
    // with scale = 1.0, so a second "automatic" press recomputes identically
    // instead of compounding through the current scale.
    let unzoomed = LensParams { scale: 1.0, ..p.clone() };
    m.initialize(lens, &unzoomed, mods, p.inverse);
    Some(m.auto_scale(p.inverse))
}

// ── The whole-frame transform ────────────────────────────────────────────────

/// Bilinear — darktable's default warp interpolation preference.
const WARP_INTERP: u32 = 0;

/// The commit_params nan-check rule: coordinates can go non-finite only when
/// the conversion *widens* the field of view — impossible staying
/// rectilinear or keeping the lens's own projection.
pub fn nan_checks(target_geom: i32, lens_type: i32) -> bool {
    !(target_geom == LF_RECTILINEAR || target_geom == lens_type)
}

/// Apply lens correction to a whole interleaved f32 RGBA frame.
///
/// Mirrors `_process_lf` with roi_in == roi_out == whole buffer: vignetting
/// colour modification runs on a copy of the input *before* the geometric
/// warp in the forward (correct) direction and *after* it in reverse, the
/// warp resamples each of R/G/B at its own distorted source coordinate
/// (subpixel/TCA), alpha is carried through untouched, and non-finite
/// coordinates (increased FOV conversions) zero the pixel when the nan-check
/// rule says they can occur — darktable enables the checks whenever the
/// target projection is not rectilinear and differs from the lens type.
pub fn process(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    lens: &ResolvedLens,
    p: &LensParams,
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len(), width * height * 4);

    let w = width as i32;
    let h = height as i32;
    // Corrupt catalog data (crop ≤ 0) would poison lf_modifier_new's scale
    // math into inf/garbage coordinates that the rectilinear nan-check
    // suppresses — C falls back to a plain copy there (lens.cc:1074), so do
    // the same instead of silently smearing edges.
    if p.crop <= 0.0 {
        output.copy_from_slice(input);
        return;
    }
    // Building the modifier walks the shared database's calibration lists,
    // which concurrent find/resolve calls are re-scoring (Score is written in
    // place), so construction is serialized behind DB_LOCK — mirroring
    // darktable holding its plugin mutex across `_get_modifier`
    // (lens.cc:1082-1090). The heavy per-row Apply calls below take only const
    // state and run unlocked, like upstream.
    let mods = to_lensfun_mods(p.modify_flags);
    let (m, modflags) = {
        let _g = DB_LOCK.lock().unwrap();
        let Some(m) = Modifier::new(lens, p.crop, w, h) else {
            output.copy_from_slice(input);
            return;
        };
        let mf = m.initialize(lens, p, mods, p.inverse);
        (m, mf)
    };

    let do_nan_checks = nan_checks(p.target_geom, lens.lens_type);

    let geom_warp = modflags & (LF_MODIFY_TCA | LF_MODIFY_DISTORTION | LF_MODIFY_GEOMETRY | LF_MODIFY_SCALE) != 0;
    let vig = modflags & LF_MODIFY_VIGNETTING != 0;

    let apply_vignetting = |buf: &mut [f32]| unsafe {
        lf_modifier_apply_color_modification(
            m.0,
            buf.as_mut_ptr(),
            0.0,
            0.0,
            w,
            h,
            LF_CR_4_RGB_UNKNOWN,
            w * 4, // row_stride in floats
        );
    };

    if !p.inverse {
        if vig {
            // Forward direction: vignetting works in place on a scratch copy
            // so the warp resamples vignetted data — the only branch that
            // needs a copy of the input.
            let mut scratch = input.to_vec();
            apply_vignetting(&mut scratch);
            if geom_warp {
                warp_into(&m, &scratch, output, width, height, do_nan_checks);
            } else {
                output.copy_from_slice(&scratch);
            }
        } else if geom_warp {
            warp_into(&m, input, output, width, height, do_nan_checks);
        } else {
            output.copy_from_slice(input);
        }
    } else {
        // Reverse direction: warp first, then re-apply the falloff.
        if geom_warp {
            warp_into(&m, input, output, width, height, do_nan_checks);
            if vig {
                apply_vignetting(output);
            }
        } else {
            output.copy_from_slice(input);
            if vig {
                apply_vignetting(output);
            }
        }
    }
}

/// One row at a time through `ApplySubpixelGeometryDistortion`, then per
/// channel clamped bilinear resampling — the loop bodies of `_process_lf`
/// with roi offsets zeroed and mask-display branches dropped.
fn warp_into(
    m: &Modifier,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    do_nan_checks: bool,
) {
    let w_i = width as i32;
    let mut coords = vec![0.0f32; width * 6];
    for y in 0..height {
        unsafe {
            lf_modifier_apply_subpixel_geometry_distortion(
                m.0,
                0.0,
                y as f32,
                w_i,
                1,
                coords.as_mut_ptr(),
            );
        }
        let out_row = &mut dst[y * width * 4..(y + 1) * width * 4];
        for x in 0..width {
            let co = &coords[x * 6..x * 6 + 6];
            for c in 0..3 {
                let sx = co[c * 2];
                let sy = co[c * 2 + 1];
                if do_nan_checks && (!sx.is_finite() || !sy.is_finite()) {
                    out_row[x * 4 + c] = 0.0;
                    continue;
                }
                // Clamp into the buffer like the C (fmaxf(fminf(...))) before
                // sampling; edge replication comes from the sampler itself.
                let cx = sx.clamp(0.0, (width - 1) as f32);
                let cy = sy.clamp(0.0, (height - 1) as f32);
                // `&src[c..]` reproduces the C's per-channel base pointer:
                // the strided sampler steps samplestride floats per pixel,
                // landing on lane c of every RGBA quad.
                out_row[x * 4 + c] = crate::interp::compute_sample_strided(
                    &src[c..],
                    cx,
                    cy,
                    width as i32,
                    height as i32,
                    4, // samplestride: RGBA
                    (width * 4) as i32,
                    WARP_INTERP,
                );
            }
            // Alpha carries through untouched (darktable only resamples it in
            // mask-display mode; its plain path leaves it stale).
            out_row[x * 4 + 3] = src[(y * width + x) * 4 + 3];
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The db-backed behaviour tests run only where lensfun data is installed
    /// (dev/CI images ship liblensfun-data-v1); bare host checkouts skip.
    fn resolved_test_lens() -> Option<ResolvedLens> {
        if !std::path::Path::new("/usr/share/lensfun").is_dir() {
            return None;
        }
        let (_, lens) = resolve(
            "Canon",
            "Canon EOS 5D Mark II",
            "Canon",
            "Canon EF 24-70mm f/2.8L USM",
        )?;
        Some(lens)
    }

    #[test]
    fn modflags_map_onto_lensfun_bits() {
        assert_eq!(to_lensfun_mods(MODFLAG_ALL),
                   LF_MODIFY_GEOMETRY | LF_MODIFY_SCALE | LF_MODIFY_DISTORTION | LF_MODIFY_TCA | LF_MODIFY_VIGNETTING);
        assert_eq!(to_lensfun_mods(0), LF_MODIFY_GEOMETRY | LF_MODIFY_SCALE);
        assert_eq!(to_lensfun_mods(MODIFY_FLAG_TCA | MODIFY_FLAG_VIGNETTING),
                   LF_MODIFY_GEOMETRY | LF_MODIFY_SCALE | LF_MODIFY_TCA | LF_MODIFY_VIGNETTING);
    }

    #[test]
    fn nan_rule_matches_commit_params() {
        // Rectilinear target: never any NaN, regardless of the lens.
        assert!(!nan_checks(LF_RECTILINEAR, LF_FISHEYE));
        // Same projection as the lens: no widening, no NaN.
        assert!(!nan_checks(LF_FISHEYE, LF_FISHEYE));
        // Fisheye lens widened to... no wait — fisheye→rectilinear NARROWS;
        // rectilinear→fisheye widens and CAN produce non-finite coordinates.
        assert!(nan_checks(LF_FISHEYE, LF_RECTILINEAR));
        assert!(nan_checks(LF_PANORAMIC, LF_FISHEYE));
    }

    /// The new per-channel strided sampler must agree with the established
    /// single-channel one on a genuinely single-channel buffer (stride 1).
    #[test]
    fn strided_sampler_agrees_with_1c_sampler() {
        let w = 23usize;
        let h = 11usize;
        let buf: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 251) as f32 / 31.0).collect();
        for interp in [0u32, 1, 2, 3] {
            for y in [0.0f32, 3.4, 7.0, 10.9] {
                for x in [0.0f32, 5.6, 22.9] {
                    let a = crate::interp::compute_sample_1c(&buf, x, y, w as i32, h as i32, w as i32, interp);
                    let b = crate::interp::compute_sample_strided(&buf, x, y, w as i32, h as i32, 1, w as i32, interp);
                    assert_eq!(a, b, "interp {interp} at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn database_lists_cameras_and_mount_filtered_lenses() {
        if !std::path::Path::new("/usr/share/lensfun").is_dir() {
            return;
        }
        let cams = list_cameras();
        assert!(cams.len() > 100, "database should hold many cameras");
        assert!(
            cams.iter().any(|(m, mo)| m == "Canon" && mo.contains("EOS 5D Mark II")),
            "known camera missing from enumeration"
        );
        // Sorted, deduplicated: adjacent duplicates impossible.
        for pair in cams.windows(2) {
            assert!(pair[0] <= pair[1], "camera list not sorted: {pair:?}");
        }

        let cam = resolve_camera("Canon", "Canon EOS 5D Mark II").expect("test camera");
        assert!(!cam.mount.is_empty(), "matched camera carries its mount");

        let lenses = list_lenses(Some(&cam.mount));
        assert!(!lenses.is_empty(), "no lenses enumerated for mount {}", cam.mount);
        // Sorted, and every listed pair must re-resolve EXACTLY — the identity
        // round-trip persistence depends on. (The old fuzzy
        // lf_db_find_lenses_hd search failed outright on ~half of these.)
        for pair in lenses.windows(2) {
            assert!(pair[0] <= pair[1], "lens list not sorted: {pair:?}");
        }
        for (maker, model) in &lenses {
            let back = resolve_lens(&cam, maker, model)
                .unwrap_or_else(|| panic!("listed lens '{maker} | {model}' must resolve"));
            assert_eq!(back.maker, *maker, "maker mismatch for '{model}'");
            assert_eq!(back.model, *model, "model mismatch");
        }
        // Near-miss identities don't resolve (exactness, not fuzzy search).
        assert!(resolve_lens(&cam, "Canon", "Canon EF 24-70mm f/2.8L").is_none());
        assert!(resolve_lens(&cam, "", "Canon EF 24-70mm f/2.8L USM").is_none());
    }

    #[test]
    fn unknown_camera_resolves_nothing() {
        if !std::path::Path::new("/usr/share/lensfun").is_dir() {
            return;
        }
        assert!(resolve_camera("Definitely Not A Camera", "Nor Is This").is_none());
    }

    #[test]
    fn listed_cameras_resolve_exactly() {
        // Same invariant as the lens round-trip above, for the camera side:
        // everything the dropdown offers must re-resolve, or a persisted pick
        // would silently stop correcting. Sampling keeps the
        // enumeration×find cost bounded while covering the whole list.
        if !std::path::Path::new("/usr/share/lensfun").is_dir() {
            return;
        }
        let cams = list_cameras();
        assert!(cams.len() > 100, "database should hold many cameras");
        let step = cams.len().div_ceil(250).max(1);
        for (mk, md) in cams.iter().step_by(step) {
            assert!(
                resolve_camera(mk, md).is_some(),
                "listed camera '{mk} | {md}' must resolve"
            );
        }
    }

    #[test]
    fn correction_moves_corners_but_not_the_centre() {
        let Some(lens) = resolved_test_lens() else { return };
        let width = 64usize;
        let height = 48usize;
        let n = width * height * 4;
        // Flat grey frame with a unique value per pixel so any resample of a
        // shifted coordinate is detectable; alpha = 1 everywhere.
        let input: Vec<f32> = (0..n)
            .map(|i| if i % 4 == 3 { 1.0 } else { ((i / 4) % 97) as f32 / 97.0 + 0.05 })
            .collect();

        let p = LensParams {
            crop: 1.0,
            focal: 24.0,
            aperture: 2.8,
            distance: 10.0,
            modify_flags: MODIFY_FLAG_DISTORTION | MODIFY_FLAG_TCA,
            ..Default::default()
        };

        let mut out = vec![0.0f32; n];
        process(&input, &mut out, width, height, &lens, &p);

        // The warp is radial about the optical centre, so the centre pixel is
        // nearly stationary while corners move a lot — but not exactly
        // stationary: lensfun places its radial origin at (width/2, height/2)
        // (not (width−1)/2), so on even-sized frames the centre pixel sits
        // half a pixel off-axis and calibration decentering (CenterX/Y) can
        // shift it further. Measured for this lens/frame: centre ≈ 0.007
        // value-units vs corner ≈ 0.15 — a ~20× separation we assert on.
        let cx = width / 2;
        let cy = height / 2;
        let ci = (cy * width + cx) * 4;
        let corner = 0usize; // top-left pixel
        for c in 0..3 {
            let centre_d = (out[ci + c] - input[ci + c]).abs();
            let corner_d = (out[corner * 4 + c] - input[corner * 4 + c]).abs();
            assert!(
                centre_d < 0.02,
                "centre channel {c} moved too far: {centre_d}"
            );
            assert!(
                corner_d > centre_d * 10.0 && corner_d > 0.05,
                "corner must move much more than the centre \
                 (centre {centre_d} vs corner {corner_d})"
            );
        }
        // Alpha carried through untouched everywhere.
        assert!(out.iter().skip(3).step_by(4).all(|&a| a == 1.0));

        // Vignetting on a calibrated lens: the FORWARD direction *corrects*
        // the falloff, so corners brighten (24mm f/2.8 wide open loses >2
        // stops there) while staying neutral at the optical centre (gain is
        // exactly 1 there — measured Δ=0).
        let pv = LensParams { modify_flags: MODIFY_FLAG_VIGNETTING, ..p.clone() };
        let mut vigged = vec![0.0f32; n];
        process(&input, &mut vigged, width, height, &lens, &pv);
        let cc = vigged[corner * 4];
        assert!(
            cc > input[corner * 4],
            "forward vignetting must brighten (correct) the corner ({} vs {})",
            cc, input[corner * 4]
        );
        for c in 0..3 {
            let d = (vigged[ci + c] - input[ci + c]).abs();
            assert!(d < 1e-5, "vignetting must be neutral at centre ({d})");
        }

        // Inverse mode (lens simulation) re-applies the falloff: corners
        // darken again, and the whole pass stays finite.
        let pi = LensParams { inverse: true, ..pv.clone() };
        let mut inv = vec![0.0f32; n];
        process(&input, &mut inv, width, height, &lens, &pi);
        assert!(inv.iter().all(|v| v.is_finite()));
        assert!(
            inv[corner * 4] < input[corner * 4],
            "inverse vignetting must darken the corner ({} vs {})",
            inv[corner * 4], input[corner * 4]
        );

        // Autoscale exists and is positive/finite for this setup.
        let pa = LensParams { modify_flags: MODFLAG_ALL, ..p.clone() };
        let sc = autoscale(&lens, &pa, width, height);
        let sc = sc.expect("autoscale computed");
        assert!(sc.is_finite() && sc > 0.0);
    }
}
