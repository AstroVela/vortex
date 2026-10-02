// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::sync::LazyLock;
use std::time::Instant;

use serde::Serialize;

use super::ValidationMode;

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("VORTEX_INDEX_TIMING").is_ok_and(|value| value == "1"));

#[derive(Default, Serialize)]
struct Phases {
    reference_check_ms: f64,
    source_validation_ms: f64,
    provider_open_ms: f64,
    native_search_ms: f64,
    hit_validation_ms: f64,
    take_ms: f64,
    result_materialization_ms: f64,
}

pub(super) enum Phase {
    ReferenceCheck,
    SourceValidation,
    ProviderOpen,
    NativeSearch,
    HitValidation,
    Take,
    ResultMaterialization,
}

pub(super) struct SearchTiming {
    start: Option<Instant>,
    checkpoint: Option<Instant>,
    phases: Phases,
    provider_cache_hit: bool,
    validation_mode: ValidationMode,
    snapshot_cache_hit: bool,
}

impl SearchTiming {
    pub(super) fn new(is_search: bool) -> Self {
        let start = (is_search && *ENABLED).then(Instant::now);
        Self {
            start,
            checkpoint: start,
            phases: Phases::default(),
            provider_cache_hit: false,
            validation_mode: ValidationMode::Strict,
            snapshot_cache_hit: false,
        }
    }

    pub(super) fn provider_cache_hit(&mut self, hit: bool) {
        self.provider_cache_hit = hit;
    }

    pub(super) fn validation_mode(&mut self, mode: ValidationMode) {
        self.validation_mode = mode;
    }

    pub(super) fn snapshot_cache_hit(&mut self, hit: bool) {
        self.snapshot_cache_hit = hit;
    }

    pub(super) fn mark(&mut self, phase: Phase) {
        let Some(previous) = self.checkpoint else {
            return;
        };
        let now = Instant::now();
        let ms = now.duration_since(previous).as_secs_f64() * 1000.0;
        match phase {
            Phase::ReferenceCheck => self.phases.reference_check_ms = ms,
            Phase::SourceValidation => self.phases.source_validation_ms = ms,
            Phase::ProviderOpen => self.phases.provider_open_ms = ms,
            Phase::NativeSearch => self.phases.native_search_ms = ms,
            Phase::HitValidation => self.phases.hit_validation_ms = ms,
            Phase::Take => self.phases.take_ms = ms,
            Phase::ResultMaterialization => self.phases.result_materialization_ms = ms,
        }
        self.checkpoint = Some(now);
    }

    pub(super) fn emit(&self) {
        #[derive(Serialize)]
        struct Event<'a> {
            event: &'static str,
            format_version: u32,
            total_ms: f64,
            phases: &'a Phases,
            provider_cache_hit: bool,
            validation_mode: ValidationMode,
            snapshot_cache_hit: bool,
        }
        if let (Some(start), Some(end)) = (self.start, self.checkpoint) {
            let event = Event {
                event: "vortex_index_search_timing",
                format_version: 1,
                total_ms: end.duration_since(start).as_secs_f64() * 1000.0,
                phases: &self.phases,
                provider_cache_hit: self.provider_cache_hit,
                validation_mode: self.validation_mode,
                snapshot_cache_hit: self.snapshot_cache_hit,
            };
            if let Ok(json) = serde_json::to_string(&event) {
                // Diagnostic output must not change query success on a closed pipe.
                drop(writeln!(std::io::stderr().lock(), "{json}"));
            }
        }
    }
}
