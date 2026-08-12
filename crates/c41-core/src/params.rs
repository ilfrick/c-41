/// Opaque blob of IOP-specific parameters.
///
/// Phase 1 will introduce typed params structs per IOP module.
/// This type carries the raw bytes through the pipeline dispatcher.
pub struct IopParams {
    data: Vec<u8>,
}

impl IopParams {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Reinterpret the byte blob as `T`.
    ///
    /// # Safety
    /// Caller must ensure `T`'s layout matches the IOP's C parameter struct.
    pub unsafe fn cast<T: Copy>(&self) -> Option<T> {
        if self.data.len() < std::mem::size_of::<T>() {
            return None;
        }
        Some(std::ptr::read_unaligned(self.data.as_ptr().cast::<T>()))
    }
}
