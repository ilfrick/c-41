use std::fmt;

#[derive(Debug)]
pub enum Error {
    InvalidImageId,
    OpenCl(String),
    Pipeline(String),
    Raw(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidImageId => write!(f, "invalid image ID"),
            Self::OpenCl(s) => write!(f, "OpenCL error: {s}"),
            Self::Pipeline(s) => write!(f, "pipeline error: {s}"),
            Self::Raw(s) => write!(f, "raw decode error: {s}"),
        }
    }
}

impl std::error::Error for Error {}
