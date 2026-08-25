//! Display-free probe of the hand-written liblensfun bindings: load the
//! system database, find a known camera, list its lenses. Skips silently when
//! no lensfun data is installed (bare host checkouts) — CI/dev containers
//! always have `lensfun-data`.

use c41_sys::lensfun::*;
use std::ffi::CString;

/// Canonical C-string helper: panics on interior NUL, which our fixed inputs
/// never contain.
fn cstr(s: &str) -> CString {
    CString::new(s).expect("no interior NUL")
}

#[test]
fn database_loads_and_resolves_known_gear() {
    // lensfun also searches /usr/local/share/lensfun and ~/.local/share/lensfun,
    // but our dev/CI images install liblensfun-data-v1 into /usr/share/lensfun;
    // a host with data only in the other locations just skips this probe.
    if !std::path::Path::new("/usr/share/lensfun").is_dir() {
        return;
    }

    unsafe {
        let db = lf_db_new();
        assert!(!db.is_null());
        // lf_db_load reports LF_NO_ERROR even when no file was found, so an
        // empty database is detected downstream via lookup results, not here.
        assert_eq!(lf_db_load(db), LF_NO_ERROR);

        // A body present in every lensfun release since forever — keeps this
        // probe deterministic without pinning a database version.
        let make = cstr("Canon");
        let model = cstr("Canon EOS 5D Mark II");

        let cams = lf_db_find_cameras(db, make.as_ptr(), model.as_ptr());
        assert!(!cams.is_null(), "camera lookup returned no list");
        let cam_count = (0..)
            .take_while(|i| !(*cams.add(*i)).is_null())
            .count();
        assert!(cam_count >= 1, "Canon EOS 5D Mark II not found in database");

        let cam = *cams;
        assert!(!(*cam).Maker.is_null() && !(*cam).Model.is_null());
        assert!((*cam).CropFactor > 0.0, "crop factor must be defined");

        let maker = std::ffi::CStr::from_ptr((*cam).Maker).to_string_lossy();
        assert_eq!(maker, "Canon", "matched camera carries the EXIF maker");

        let lenses = lf_db_find_lenses_hd(
            db,
            cam,
            std::ptr::null(),
            std::ptr::null(),
            LF_SEARCH_SORT_AND_UNIQUIFY,
        );
        assert!(!lenses.is_null(), "lens lookup returned no list");
        let lens_count = (0..)
            .take_while(|i| !(*lenses.add(*i)).is_null())
            .count();
        assert!(lens_count >= 1, "no lenses resolved for the 5D Mark II mount");

        // Best-score lens must carry readable identity + calibration basics.
        let lens = *lenses;
        assert!(!(*lens).Model.is_null());
        assert!((*lens).CropFactor > 0.0);

        // Release the lists themselves (never the elements — db-owned).
        lf_free(lenses as *mut _);
        lf_free(cams as *mut _);
        lf_db_destroy(db);
    }
}
