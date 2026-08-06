use std::env;

pub(super) fn env_u32(k: &str, d: u32) -> u32 {
    env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

pub(super) fn env_usize(k: &str, d: usize) -> usize {
    env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

pub(super) fn env_bool(k: &str) -> bool {
    env::var(k)
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

pub(super) fn env_string(k: &str, d: &str) -> String {
    env::var(k).unwrap_or_else(|_| d.into())
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum OutputMode {
    Table,
    Csv,
}

impl OutputMode {
    pub(super) fn from_env() -> Self {
        match env_string("ML_OUTPUT", "table")
            .to_ascii_lowercase()
            .as_str()
        {
            "csv" => Self::Csv,
            _ => Self::Table,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum SweepMode {
    Smoke,
    Standard,
    Full,
}

impl SweepMode {
    pub(super) fn from_env(default: Self) -> Self {
        match env_string("ML_SWEEP", "").to_ascii_lowercase().as_str() {
            "smoke" => Self::Smoke,
            "standard" => Self::Standard,
            "full" => Self::Full,
            _ => default,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_mode_defaults_to_table() {
        // SAFETY: these tests restore the env var they touch.
        unsafe { env::remove_var("ML_OUTPUT") };
        assert_eq!(OutputMode::from_env(), OutputMode::Table);
        unsafe { env::set_var("ML_OUTPUT", "csv") };
        assert_eq!(OutputMode::from_env(), OutputMode::Csv);
        unsafe { env::set_var("ML_OUTPUT", "CSV") };
        assert_eq!(OutputMode::from_env(), OutputMode::Csv);
        unsafe { env::set_var("ML_OUTPUT", "wibble") };
        assert_eq!(OutputMode::from_env(), OutputMode::Table);
        unsafe { env::remove_var("ML_OUTPUT") };
    }

    #[test]
    fn sweep_mode_honors_env_then_default() {
        unsafe { env::remove_var("ML_SWEEP") };
        assert_eq!(
            SweepMode::from_env(SweepMode::Standard),
            SweepMode::Standard
        );
        assert_eq!(SweepMode::from_env(SweepMode::Smoke), SweepMode::Smoke);
        unsafe { env::set_var("ML_SWEEP", "full") };
        assert_eq!(SweepMode::from_env(SweepMode::Smoke), SweepMode::Full);
        unsafe { env::set_var("ML_SWEEP", "SMOKE") };
        assert_eq!(SweepMode::from_env(SweepMode::Standard), SweepMode::Smoke);
        unsafe { env::set_var("ML_SWEEP", "bogus") };
        assert_eq!(
            SweepMode::from_env(SweepMode::Standard),
            SweepMode::Standard
        );
        unsafe { env::remove_var("ML_SWEEP") };
    }

    #[test]
    fn bool_env_accepts_1_or_true() {
        unsafe { env::remove_var("ML_BOOL_TEST") };
        assert!(!env_bool("ML_BOOL_TEST"));
        unsafe { env::set_var("ML_BOOL_TEST", "1") };
        assert!(env_bool("ML_BOOL_TEST"));
        unsafe { env::set_var("ML_BOOL_TEST", "true") };
        assert!(env_bool("ML_BOOL_TEST"));
        unsafe { env::set_var("ML_BOOL_TEST", "TRUE") };
        assert!(env_bool("ML_BOOL_TEST"));
        unsafe { env::set_var("ML_BOOL_TEST", "yes") };
        assert!(!env_bool("ML_BOOL_TEST"));
        unsafe { env::remove_var("ML_BOOL_TEST") };
    }
}
