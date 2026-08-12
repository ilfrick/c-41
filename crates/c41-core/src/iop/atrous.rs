use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Atrous;

impl IopProcess for Atrous {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "atrous" }
}

/// Add `add_buf` into `out_buf` element-wise, for `n` floats.
///
/// Replaces the DT_OMP_FOR_SIMD residue-add loop in atrous.c process().
#[no_mangle]
pub unsafe extern "C" fn darkroom_add_buffers(
    out_buf: *mut f32,
    add_buf: *const f32,
    n: usize,
) {
    let out = std::slice::from_raw_parts_mut(out_buf, n);
    let add = std::slice::from_raw_parts(add_buf, n);
    for k in 0..n {
        out[k] += add[k];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_buffers_basic() {
        let add = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![10.0f32, 20.0, 30.0, 40.0];
        unsafe { darkroom_add_buffers(out.as_mut_ptr(), add.as_ptr(), 4); }
        assert!((out[0] - 11.0).abs() < 1e-6);
        assert!((out[1] - 22.0).abs() < 1e-6);
        assert!((out[2] - 33.0).abs() < 1e-6);
        assert!((out[3] - 44.0).abs() < 1e-6);
    }

    #[test]
    fn add_buffers_zeros() {
        let add = vec![0.0f32; 8];
        let mut out = vec![5.0f32; 8];
        unsafe { darkroom_add_buffers(out.as_mut_ptr(), add.as_ptr(), 8); }
        for v in &out { assert!((*v - 5.0).abs() < 1e-6); }
    }
}
