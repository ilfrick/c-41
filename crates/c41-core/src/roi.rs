/// Region of interest passed to each IOP module's process call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoiIn {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub scale: f32,
}
