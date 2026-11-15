//! Darktable-compatible adaptive resource budgets.
//!
//! These are configured limits derived from the selected level and reported
//! total memory; they are not measurements of currently free memory.
//!
//! CPU-available parity follows pinned darktable `src/common/darktable.c`:
//! `dt_get_available_mem()` returns `MAX(512 MiB, total_memory / 1024 * fraction)`.
//! `total_memory` is normalized to whole MiB (and reduced by 1 GiB below 8 GiB)
//! before this calculation. No unified-memory deduction is part of that CPU
//! calculation.

use std::fmt;

/// Darktable's four adaptive resource levels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceLevel {
    Small,
    Default,
    Large,
    Unrestricted,
}

/// Darktable fraction-table columns: CPU available, CPU single buffer,
/// mipmap cache and OpenCL available, each expressed in 1/1024ths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceFractions {
    pub cpu_available: u32,
    pub cpu_single_buffer: u32,
    pub mipmap: u32,
    pub opencl_available: u32,
}

impl ResourceLevel {
    /// Fractions copied from darktable `src/common/darktable.c` at the pinned
    /// baseline (97912518a5a9a9786aeee3c38116fbc999412ff8).
    pub const fn fractions(self) -> ResourceFractions {
        match self {
            Self::Small => ResourceFractions {
                cpu_available: 128,
                cpu_single_buffer: 4,
                mipmap: 64,
                opencl_available: 400,
            },
            Self::Default => ResourceFractions {
                cpu_available: 512,
                cpu_single_buffer: 8,
                mipmap: 128,
                opencl_available: 700,
            },
            Self::Large => ResourceFractions {
                cpu_available: 700,
                cpu_single_buffer: 16,
                mipmap: 128,
                opencl_available: 900,
            },
            Self::Unrestricted => ResourceFractions {
                cpu_available: 16384,
                cpu_single_buffer: 1024,
                mipmap: 128,
                opencl_available: 900,
            },
        }
    }
}

/// Optional darktable debug/reference profile, in MiB. None is the normal mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbsoluteReferenceProfile {
    Reference,
    Mini,
    Notebook,
    Common2026,
}

impl AbsoluteReferenceProfile {
    const fn mib(self) -> [u64; 4] {
        match self {
            Self::Reference => [8192, 32, 512, 2048],
            Self::Mini => [1024, 2, 128, 200],
            Self::Notebook => [4096, 32, 512, 1024],
            Self::Common2026 => [16384, 64, 1024, 5500],
        }
    }
}

/// Calculated configured limits; no field claims to measure free memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceBudget {
    pub cpu_available_bytes: u64,
    pub cpu_single_buffer_bytes: u64,
    pub mipmap_bytes: u64,
    pub opencl_available_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BudgetError {
    ArithmeticOverflow,
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticOverflow => f.write_str("resource budget arithmetic overflow"),
        }
    }
}
impl std::error::Error for BudgetError {}

impl ResourceBudget {
    /// Calculate configured resource limits from total memory. The optional
    /// cache deduction affects only the CPU-available budget.
    ///
    /// Darktable's pinned CPU-available calculation has no unified-memory
    /// deduction: `src/common/darktable.c:2448-2456` applies the fraction to
    /// normalized `total_memory` and floors at 512 MiB.
    pub fn calculate(
        level: ResourceLevel,
        total_memory_bytes: u64,
        cache_deduction_bytes: Option<u64>,
        reference: Option<AbsoluteReferenceProfile>,
    ) -> Result<Self, BudgetError> {
        // darktable normalizes physical RAM to whole MiB, deducting 1 GiB on
        // systems below 8 GiB before applying the fraction table.
        let total_memory_bytes = if reference.is_some() {
            total_memory_bytes
        } else {
            let total_mib = total_memory_bytes / (1024 * 1024);
            let normalized_mib = if total_mib < 8192 {
                total_mib.saturating_sub(1024)
            } else {
                total_mib
            };
            normalized_mib
                .checked_mul(1024 * 1024)
                .ok_or(BudgetError::ArithmeticOverflow)?
        };
        let values = if let Some(profile) = reference {
            let [cpu, single, mipmap, opencl] = profile.mib();
            [cpu, single, mipmap, opencl].map(|mib| {
                mib.checked_mul(1024 * 1024)
                    .ok_or(BudgetError::ArithmeticOverflow)
            })
        } else {
            let fractions = level.fractions();
            [
                fractions.cpu_available,
                fractions.cpu_single_buffer,
                fractions.mipmap,
                fractions.opencl_available,
            ]
            .map(|fraction| {
                (total_memory_bytes / 1024)
                    .checked_mul(u64::from(fraction))
                    .ok_or(BudgetError::ArithmeticOverflow)
            })
        };
        let [cpu, single, mipmap, opencl] = values;
        let cpu_available = cpu?;
        let single_buffer = single?;
        let mipmap = mipmap?;
        let opencl = opencl?;
        let (cpu_available_bytes, single_buffer_bytes) = if reference.is_none() {
            (
                cpu_available.max(512 * 1024 * 1024),
                single_buffer.max(2 * 1024 * 1024),
            )
        } else {
            (cpu_available, single_buffer)
        };
        let cpu_available_bytes = cpu_available_bytes
            .saturating_sub(cache_deduction_bytes.unwrap_or(0));
        Ok(Self {
            cpu_available_bytes,
            cpu_single_buffer_bytes: single_buffer_bytes,
            mipmap_bytes: mipmap,
            opencl_available_bytes: opencl,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractions_scale_and_unrestricted_has_no_arbitrary_cap() {
        let budget =
            ResourceBudget::calculate(ResourceLevel::Small, 7 * 1024 * 1024 * 1024, None, None)
                .unwrap();
        assert_eq!(budget.cpu_available_bytes, 768 * 1024 * 1024);
        assert_eq!(budget.cpu_single_buffer_bytes, 24 * 1024 * 1024);

        let threshold = ResourceBudget::calculate(
            ResourceLevel::Small,
            8 * 1024 * 1024 * 1024,
            None,
            None,
        )
        .unwrap();
        assert_eq!(threshold.cpu_available_bytes, 1024 * 1024 * 1024);
        assert_eq!(threshold.cpu_single_buffer_bytes, 32 * 1024 * 1024);

        let low_memory = ResourceBudget::calculate(
            ResourceLevel::Small,
            1024 * 1024 * 1024,
            None,
            None,
        )
        .unwrap();
        assert_eq!(low_memory.cpu_available_bytes, 512 * 1024 * 1024);
        assert_eq!(low_memory.cpu_single_buffer_bytes, 2 * 1024 * 1024);

        let input = u64::MAX / 16384;
        let normalized = (input / (1024 * 1024)) * (1024 * 1024);
        let huge = ResourceBudget::calculate(ResourceLevel::Unrestricted, input, None, None).unwrap();
        assert_eq!(huge.cpu_available_bytes, (normalized / 1024) * 16384);
    }

    #[test]
    fn cache_deduction_is_opt_in_and_saturating() {
        let total = 8 * 1024 * 1024 * 1024;
        let plain = ResourceBudget::calculate(ResourceLevel::Default, total, None, None).unwrap();
        let deducted =
            ResourceBudget::calculate(ResourceLevel::Default, total, Some(100), None).unwrap();
        let exhausted =
            ResourceBudget::calculate(ResourceLevel::Default, total, Some(u64::MAX), None).unwrap();
        assert_eq!(plain.cpu_available_bytes, 4 * 1024 * 1024 * 1024);
        assert_eq!(deducted.cpu_available_bytes, 4 * 1024 * 1024 * 1024 - 100);
        assert_eq!(exhausted.cpu_available_bytes, 0);
    }

    #[test]
    fn absolute_reference_profiles_are_explicit() {
        let p = ResourceBudget::calculate(
            ResourceLevel::Small,
            1,
            None,
            Some(AbsoluteReferenceProfile::Mini),
        )
        .unwrap();
        assert_eq!(p.cpu_available_bytes, 1024 * 1024 * 1024);
        assert_eq!(p.cpu_single_buffer_bytes, 2 * 1024 * 1024);
    }

    #[test]
    fn arithmetic_overflow_is_reported() {
        assert_eq!(
            ResourceBudget::calculate(ResourceLevel::Unrestricted, u64::MAX, None, None),
            Err(BudgetError::ArithmeticOverflow)
        );
    }
}
