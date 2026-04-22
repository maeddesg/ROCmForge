//! JSONL telemetry log (`ga_tuning_spec §5.10`).
//!
//! The logger is plug-in — it either writes to a file handle or, for
//! tests, captures lines into a `Vec<String>` that the caller can
//! inspect. Every event carries the `run_id`, a timestamp, and the
//! event tag so post-hoc analysis can grep cheaply.
//!
//! Event types implemented here: `shape_start`, `eval`,
//! `generation_complete`, `early_exit`, `shape_complete`.

use std::io::{self, Write};

use serde_json::{json, Value};

/// Telemetry sink. `File` is production; `Capture` is tests.
pub enum LoggerSink {
    File(Box<dyn Write + Send>),
    Capture(Vec<String>),
    Null,
}

impl std::fmt::Debug for LoggerSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(_) => write!(f, "LoggerSink::File(<dyn Write>)"),
            Self::Capture(lines) => f.debug_tuple("LoggerSink::Capture").field(lines).finish(),
            Self::Null => write!(f, "LoggerSink::Null"),
        }
    }
}

#[derive(Debug)]
pub struct GaLogger {
    run_id: String,
    sink: LoggerSink,
}

impl GaLogger {
    /// Logger that drops every event. Used by tests that don't care
    /// about log content.
    pub fn null(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            sink: LoggerSink::Null,
        }
    }

    /// Logger that captures every event into memory. Tests can
    /// inspect `captured_lines()` to assert on specific records.
    pub fn capturing(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            sink: LoggerSink::Capture(Vec::new()),
        }
    }

    /// Logger backed by a `Write` (typically a file).
    pub fn writing(run_id: impl Into<String>, w: Box<dyn Write + Send>) -> Self {
        Self {
            run_id: run_id.into(),
            sink: LoggerSink::File(w),
        }
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn captured_lines(&self) -> Option<&[String]> {
        if let LoggerSink::Capture(lines) = &self.sink {
            Some(lines)
        } else {
            None
        }
    }

    fn emit(&mut self, mut obj: Value) -> io::Result<()> {
        // Every event carries ts + run_id for post-hoc grep.
        let ts = now_iso8601();
        if let Value::Object(map) = &mut obj {
            map.entry("ts".to_string()).or_insert_with(|| Value::String(ts));
            map.entry("run_id".to_string())
                .or_insert_with(|| Value::String(self.run_id.clone()));
        }
        let line = serde_json::to_string(&obj)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        match &mut self.sink {
            LoggerSink::File(w) => {
                w.write_all(line.as_bytes())?;
                w.write_all(b"\n")?;
            }
            LoggerSink::Capture(lines) => lines.push(line),
            LoggerSink::Null => {}
        }
        Ok(())
    }

    pub fn log_shape_start(&mut self, shape: &str) -> io::Result<()> {
        self.emit(json!({
            "event": "shape_start",
            "shape": shape,
        }))
    }

    pub fn log_shape_complete(&mut self, shape: &str, best_fitness: f64) -> io::Result<()> {
        self.emit(json!({
            "event": "shape_complete",
            "shape": shape,
            "best_fitness": best_fitness,
        }))
    }

    pub fn log_eval(
        &mut self,
        shape: &str,
        generation: usize,
        individual: usize,
        genome_json: Value,
        metrics: Value,
        seed: u64,
    ) -> io::Result<()> {
        self.emit(json!({
            "event": "eval",
            "shape": shape,
            "generation": generation,
            "individual": individual,
            "genome": genome_json,
            "metrics": metrics,
            "seed": seed,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn log_generation(
        &mut self,
        shape: &str,
        generation: usize,
        compile_hits: u64,
        compile_misses: u64,
        pre_compile_rejects: usize,
        post_compile_vgpr_rejects: usize,
        benchmarked_individuals: usize,
        best_fitness: f64,
        median_fitness: f64,
        compile_wall_ms: u64,
        benchmark_wall_ms: u64,
    ) -> io::Result<()> {
        let hit_rate = if compile_hits + compile_misses > 0 {
            compile_hits as f64 / (compile_hits + compile_misses) as f64
        } else {
            0.0
        };
        self.emit(json!({
            "event": "generation_complete",
            "shape": shape,
            "generation": generation,
            "compile_cache_hits": compile_hits,
            "compile_cache_misses": compile_misses,
            "compile_cache_hit_rate": hit_rate,
            "pre_compile_rejects": pre_compile_rejects,
            "post_compile_vgpr_rejects": post_compile_vgpr_rejects,
            "benchmarked_individuals": benchmarked_individuals,
            "compile_wall_ms": compile_wall_ms,
            "benchmark_wall_ms": benchmark_wall_ms,
            "best_fitness": best_fitness,
            "median_fitness": median_fitness,
        }))
    }

    pub fn log_early_exit(
        &mut self,
        shape: &str,
        generation: usize,
        best_fitness: f64,
    ) -> io::Result<()> {
        self.emit(json!({
            "event": "early_exit",
            "shape": shape,
            "generation": generation,
            "best_fitness": best_fitness,
        }))
    }
}

fn now_iso8601() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_and_parse() {
        let mut log = GaLogger::capturing("run-test");
        log.log_shape_start("toy").unwrap();
        log.log_generation(
            "toy", 0, 0, 50, 10, 2, 38, 1.35, 0.94, 120, 500,
        )
        .unwrap();
        log.log_early_exit("toy", 12, 1.35).unwrap();

        let lines = log.captured_lines().unwrap();
        assert_eq!(lines.len(), 3);
        for line in lines {
            // Every line is valid JSON, carries run_id.
            let v: Value = serde_json::from_str(line).expect("valid JSONL");
            assert_eq!(v.get("run_id").and_then(|x| x.as_str()), Some("run-test"));
            assert!(v.get("event").is_some());
            assert!(v.get("ts").is_some());
        }
    }
}
