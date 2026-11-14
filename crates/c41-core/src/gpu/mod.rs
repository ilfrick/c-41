//! Optional, fail-closed OpenCL acceleration for selected pipeline stages.
//!
//! The embedded kernel is an adaptation of darktable's
//! `data/kernels/sigmoid.cl::sigmoid_loglogistic_rgb_ratio`, pinned with the
//! repository's darktable baseline. The source is copyright (c) 2023-2026
//! darktable developers and licensed under GNU GPL-3.0-or-later; see the
//! original file's license notice. This derivative preserves that attribution.

use std::sync::{atomic::{AtomicUsize, Ordering}, Mutex, OnceLock};

use anyhow::{anyhow, Result};
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY},
    platform::get_platforms,
    program::Program,
    types::{cl_float, cl_ulong, CL_BLOCKING},
};

const KERNEL: &str = include_str!("sigmoid_rgb_ratio.cl");
const WORKGROUP: usize = 128;

struct Backend {
    _context: Context,
    queue: CommandQueue,
    kernel: Kernel,
    device_name: String,
}

static BACKEND: OnceLock<Option<Mutex<Backend>>> = OnceLock::new();

fn initialize() -> Option<Mutex<Backend>> {
    if std::env::var_os("C41_GPU").is_some_and(|value| value == "off") {
        return None;
    }
    let backend = (|| -> Result<Backend> {
        let mut selected = None;
        for platform in get_platforms()? {
            if let Ok(devices) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
                if let Some(id) = devices.first() {
                    selected = Some(Device::new(*id));
                    break;
                }
            }
        }
        let device = selected.ok_or_else(|| anyhow!("no OpenCL GPU device"))?;
        let device_name = device.name()?;
        let context = Context::from_device(&device)?;
        let queue = CommandQueue::create_default(&context, 0)?;
        let program = Program::create_and_build_from_source(&context, KERNEL, "")
            .map_err(|log| anyhow!("OpenCL sigmoid kernel build failed: {log}"))?;
        let kernel = Kernel::create(&program, "sigmoid_rgb_ratio")?;
        Ok(Backend {
            _context: context,
            queue,
            kernel,
            device_name,
        })
    })();
    backend.ok().map(Mutex::new)
}

fn backend() -> Option<&'static Mutex<Backend>> {
    BACKEND.get_or_init(initialize).as_ref()
}

pub(crate) fn explicitly_disabled() -> bool {
    std::env::var_os("C41_GPU").is_some_and(|value| value == "off")
}

static DISPATCH_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn dispatch_count() -> usize {
    DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// Check backend initialization without allocating image-sized host buffers.
pub(crate) fn backend_available() -> bool {
    backend().is_some()
}

static FALLBACK_REPORTED: OnceLock<()> = OnceLock::new();

/// Emit one actionable diagnostic, without repeating it for every frame.
pub(crate) fn report_fallback(error: &anyhow::Error) {
    FALLBACK_REPORTED.get_or_init(|| {
        eprintln!("C-41 GPU sigmoid unavailable; using CPU fallback: {error}");
    });
}

/// Name of the selected real GPU, if the backend initialized successfully.
/// Return the name of the OpenCL GPU selected by the initialized backend.
pub fn gpu_name() -> Option<String> {
    backend()?
        .lock()
        .ok()
        .map(|backend| backend.device_name.clone())
}

/// Run the transfer-inclusive RGB-ratio stage. Any error is returned to the
/// caller, which must execute the unchanged CPU reference instead.
pub(crate) fn sigmoid_rgb_ratio(input: &[f32], output: &mut [f32], params: [f32; 6]) -> Result<()> {
    if input.len() != output.len() || input.len() % 4 != 0 {
        return Err(anyhow!("invalid packed RGBA buffer lengths"));
    }
    let pixels = input.len() / 4;
    if pixels == 0 {
        return Ok(());
    }
    let bytes = input
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow!("OpenCL buffer byte size overflow"))?;
    let global = pixels
        .checked_add(WORKGROUP - 1)
        .ok_or_else(|| anyhow!("OpenCL global size overflow"))?
        .checked_div(WORKGROUP)
        .and_then(|groups| groups.checked_mul(WORKGROUP))
        .ok_or_else(|| anyhow!("OpenCL global size overflow"))?;
    let count = cl_ulong::try_from(pixels)
        .map_err(|_| anyhow!("pixel count exceeds OpenCL kernel range"))?;
    // Select/initialize the backend before allocating or writing any host frame.
    let runtime = backend().ok_or_else(|| anyhow!("OpenCL GPU backend unavailable"))?;
    let backend = runtime
        .lock()
        .map_err(|_| anyhow!("OpenCL backend lock poisoned"))?;
    let _ = bytes;
    unsafe {
        let mut input_buffer = Buffer::<cl_float>::create(
            &backend._context,
            CL_MEM_READ_ONLY,
            input.len(),
            std::ptr::null_mut(),
        )?;
        let output_buffer = Buffer::<cl_float>::create(
            &backend._context,
            CL_MEM_WRITE_ONLY,
            output.len(),
            std::ptr::null_mut(),
        )?;
        backend
            .queue
            .enqueue_write_buffer(&mut input_buffer, CL_BLOCKING, 0, input, &[])?;
        backend.kernel.set_arg(0, &input_buffer)?;
        backend.kernel.set_arg(1, &output_buffer)?;
        backend.kernel.set_arg(2, &count)?;
        for (index, value) in params.iter().enumerate() {
            backend.kernel.set_arg((index + 3) as u32, value)?;
        }
        backend.queue.enqueue_nd_range_kernel(
            backend.kernel.get(),
            1,
            std::ptr::null(),
            &global,
            &WORKGROUP,
            &[],
        )?;
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        // Read directly into the caller buffer. If readback fails part-way,
        // the caller must overwrite the entire output via the CPU fallback.
        backend
            .queue
            .enqueue_read_buffer(&output_buffer, CL_BLOCKING, 0, output, &[])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        iop::sigmoid,
        pipeline::{Pipeline, Stage},
    };
    use std::time::Instant;

    fn sigmoid_stage(params: [f32; 6]) -> Stage {
        Stage::Sigmoid {
            white_target: params[0],
            black_target: params[1],
            paper_exp: params[2],
            film_fog: params[3],
            film_power: params[4],
            paper_power: params[5],
        }
    }

    /// Hardware-only: run with `cargo test -p c41-core sigmoid_gpu_matches_cpu_when_gpu_is_available -- --ignored --nocapture`.
    #[test]
    #[ignore = "requires an OpenCL GPU device"]
    fn sigmoid_gpu_matches_cpu_when_gpu_is_available() {
        let name = gpu_name().expect("OpenCL GPU backend unavailable");
        eprintln!("GPU differential test device: {name}");
        let params = sigmoid::rgb_ratio_params(1.5, 0.0, 100.0, 1.52);
        let stage = sigmoid_stage(params);
        // Cross the pipeline's 64k-pixel Rayon threshold to exercise one
        // whole-stage GPU submission rather than per-band dispatch.
        let input: Vec<f32> = (0..65_537)
            .flat_map(|i| {
                let colors = match i % 8 {
                    0 => [0.18, 0.18, 0.18],
                    1 => [0.0, 0.8, 0.02],
                    2 => [-0.2, 0.3, 1.4],
                    3 => [2.0, 0.1, 0.7],
                    4 => [0.0, 0.0, 0.0],
                    5 => [-1.0, -0.5, -0.2],
                    6 => [0.01, 0.2, 0.9],
                    _ => [1.2, 0.8, 0.4],
                };
                [colors[0], colors[1], colors[2], (i as f32) / 1003.0]
            })
            .collect();
        let mut cpu = vec![0.0; input.len()];
        stage.apply(&input, &mut cpu, 65_537, 1);
        let mut gpu = vec![f32::NAN; input.len()];
        sigmoid_rgb_ratio(&input, &mut gpu, params).expect("GPU dispatch/readback");
        for (pixel, (actual, expected)) in gpu.chunks_exact(4).zip(cpu.chunks_exact(4)).enumerate()
        {
            assert_eq!(actual[3], input[pixel * 4 + 3], "alpha contract");
            for channel in 0..3 {
                assert!(
                    actual[channel].is_finite(),
                    "nonfinite GPU output at {pixel}:{channel}"
                );
                assert!(
                    actual[channel] >= params[1] - 2.0e-5 && actual[channel] <= params[0] + 2.0e-5,
                    "GPU output outside sigmoid bounds at {pixel}:{channel}: {}",
                    actual[channel]
                );
                let tolerance = 2.0e-5 + 2.0e-4 * expected[channel].abs();
                assert!(
                    (actual[channel] - expected[channel]).abs() <= tolerance,
                    "GPU mismatch at {pixel}:{channel}: {} vs {} (tol {tolerance})",
                    actual[channel],
                    expected[channel]
                );
            }
        }
        assert!(
            gpu.iter().all(|value| value.is_finite()),
            "tail or output lanes were not written"
        );
        let pipeline = Pipeline::with_stages(vec![
            Stage::Exposure { black: 0.0, scale: 0.75 },
            stage.clone(),
            Stage::Exposure { black: 0.01, scale: 1.1 },
        ]);
        let mut expected = vec![0.0; input.len()];
        let mut intermediate = vec![0.0; input.len()];
        pipeline.stages[0].apply(&input, &mut expected, 65_537, 1);
        stage.apply(&expected, &mut intermediate, 65_537, 1);
        pipeline.stages[2].apply(&intermediate, &mut expected, 65_537, 1);
        let before = dispatch_count();
        let actual = pipeline.process(&input, 65_537, 1);
        assert!(dispatch_count() > before, "pipeline silently fell back to CPU");
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = 2.0e-5 + 2.0e-4 * expected.abs();
            assert!((actual - expected).abs() <= tolerance, "mixed pipeline mismatch at {index}: {actual} vs {expected}");
        }
    }

    /// Repeated end-to-end pipeline timing; diagnostic only, not a performance claim.
    /// Run with `cargo test -p c41-core gpu_sigmoid_transfer_inclusive_diagnostic -- --ignored --nocapture`.
    #[test]
    #[ignore = "manual transfer-inclusive GPU timing diagnostic"]
    fn gpu_sigmoid_transfer_inclusive_diagnostic() {
        let Some(name) = gpu_name() else {
            eprintln!("SKIP: no OpenCL CL_DEVICE_TYPE_GPU available");
            return;
        };
        const REPETITIONS: usize = 5;
        let params = sigmoid::rgb_ratio_params(1.5, 0.0, 100.0, 1.52);
        let pipeline = Pipeline::with_stages(vec![sigmoid_stage(params)]);
        eprintln!("GPU pipeline diagnostic device: {name}");

        // Pay one-time runtime initialization and kernel dispatch before timing.
        let warmup_input = vec![0.18f32; 256 * 256 * 4];
        let before_warmup = dispatch_count();
        drop(pipeline.process(&warmup_input, 256, 256));
        assert!(dispatch_count() > before_warmup, "warmup did not dispatch to OpenCL");
        drop(pipeline.process_cpu(&warmup_input, 256, 256));
        drop(warmup_input);

        for (width, height) in [(1280, 720), (6000, 4000)] {
            let input = vec![0.18f32; width * height * 4];
            let mut gpu_times = Vec::with_capacity(REPETITIONS);
            let mut cpu_times = Vec::with_capacity(REPETITIONS);
            for _ in 0..REPETITIONS {
                let before = dispatch_count();
                let start = Instant::now();
                let output = pipeline.process(&input, width, height);
                gpu_times.push(start.elapsed());
                std::hint::black_box(&output);
                drop(output);
                assert!(dispatch_count() > before, "timed pipeline silently fell back to CPU");

                let start = Instant::now();
                let output = pipeline.process_cpu(&input, width, height);
                cpu_times.push(start.elapsed());
                std::hint::black_box(&output);
                drop(output);
            }
            gpu_times.sort_unstable();
            cpu_times.sort_unstable();
            eprintln!(
                "{width}x{height} on {name} ({REPETITIONS} runs): median GPU whole-pipeline incl. transfers/readback {:?}; median Rayon CPU process_cpu {:?}",
                gpu_times[REPETITIONS / 2],
                cpu_times[REPETITIONS / 2],
            );
            drop(input);
        }
    }
}
