// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Custom console-table rendering for random-access benchmark results.
//!
//! Random-access has one extra dimension beyond what the shared
//! [`vortex_bench::display::render_table`] handles: the access pattern, which
//! we keep on the row. Rather than push that concern into the shared renderer,
//! we keep this layout local.
//!
//! Rows are keyed by `(dataset, Option<AccessPattern>)` in the order they are
//! first observed in `runs`; columns are the `formats`.

use std::io::Write;

use anyhow::Result;
use tabled::builder::Builder;
use tabled::settings::Color;
use tabled::settings::Style;
use tabled::settings::themes::Colorization;
use vortex_bench::Format;
use vortex_bench::measurements::TimingMeasurement;
use vortex_bench::utils::aliases::hash_map::HashMap;

use crate::AccessPattern;

/// One row of random-access timing for a `(dataset, pattern)` benchmark and a
/// specific format. Carries the same [`TimingMeasurement`] used for JSON output
/// so the two emitters stay in sync.
pub struct RandomAccessRun {
    pub timing: TimingMeasurement,
    pub dataset: String,
    pub pattern: Option<AccessPattern>,
    /// Row label for this run (e.g. `random-access/taxi/uniform`). Format is
    /// implied by the column header, so it is not part of the label.
    pub display_name: String,
}

/// Render a random-access benchmark result table.
///
/// Columns are the `formats`, headed by `{format-ext}` (e.g. `vortex`,
/// `parquet`). Rows are unique `(dataset, pattern)` pairs in the order they
/// were observed in `runs`.
///
/// The first column (the baseline) is the leftmost format; each non-baseline
/// cell is colored relative to the baseline value in its row.
pub fn render_random_access_table<W: Write>(
    writer: &mut W,
    runs: &[RandomAccessRun],
    formats: &[Format],
) -> Result<()> {
    // Storing the cell key alongside its display label keeps the lookup loop
    // straightforward.
    let columns: Vec<(Format, String)> = formats
        .iter()
        .map(|format| (*format, column_label(*format)))
        .collect();

    // Rows: unique (dataset, pattern) keys in insertion order. Insertion
    // order matches the outer iteration in `run_random_access`, so taxi/legacy
    // rows appear before pattern rows.
    let mut rows: Vec<(String, Option<AccessPattern>, String)> = Vec::new();
    let mut row_index: HashMap<(String, Option<AccessPattern>), usize> = HashMap::new();
    let mut cells: HashMap<(usize, Format), u128> = HashMap::new();

    for run in runs {
        let key = (run.dataset.clone(), run.pattern);
        let idx = match row_index.get(&key) {
            Some(&idx) => idx,
            None => {
                let idx = rows.len();
                row_index.insert(key.clone(), idx);
                rows.push((run.dataset.clone(), run.pattern, run.display_name.clone()));
                idx
            }
        };
        let format = run.timing.target.format;
        cells.insert((idx, format), run.timing.median_time().as_micros());
    }

    let mut table_builder = Builder::default();

    // Single header row: `Benchmark | format | format | ...`.
    let header: Vec<String> = std::iter::once("Benchmark".to_owned())
        .chain(columns.iter().map(|(_, label)| label.clone()))
        .collect();
    table_builder.push_record(header);

    // Baseline for each row is the leftmost column. We capture it before
    // emitting the row so non-baseline cells can be colored relative to it.
    let baseline_column = columns.first().map(|(format, _)| *format);

    let mut colors = Vec::new();
    for (row_idx, (_, _, label)) in rows.iter().enumerate() {
        let baseline_value =
            baseline_column.and_then(|format| cells.get(&(row_idx, format)).copied());

        let mut record = vec![label.clone()];
        for (col_idx, (format, _)) in columns.iter().enumerate() {
            let value = cells.get(&(row_idx, *format)).copied();
            record.push(match (value, baseline_value) {
                (Some(v), Some(base)) if base > 0 => {
                    let ratio = v as f64 / base as f64;
                    if col_idx > 0 {
                        colors.push(Colorization::exact(
                            vec![color(base, v)],
                            (row_idx + 1, col_idx + 1),
                        ));
                    }
                    format!("{v} μs ({ratio:.2})")
                }
                (Some(v), _) => format!("{v} μs"),
                (None, _) => "-".to_string(),
            });
        }
        table_builder.push_record(record);
    }

    let mut table = table_builder.build();
    table.with(Style::modern());
    for c in colors {
        table.with(c);
    }

    writeln!(writer, "{table}")?;
    Ok(())
}

/// `{format-ext}` (e.g. `vortex`, `parquet`). Using `Format::ext()` keeps
/// headers narrow (`vortex` rather than `vortex-file-compressed`).
fn column_label(format: Format) -> String {
    format.ext().to_string()
}

/// Mirror of the coloring used by `vortex_bench::display::render_table`:
/// green when within 10% of baseline, yellow when within 50%, red beyond.
fn color(baseline: u128, value: u128) -> Color {
    if value > baseline + baseline / 2 {
        Color::BG_RED | Color::FG_BLACK
    } else if value > baseline + baseline / 10 {
        Color::BG_YELLOW | Color::FG_BLACK
    } else {
        Color::BG_BRIGHT_GREEN | Color::FG_BLACK
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use vortex_bench::Engine;
    use vortex_bench::Target;

    use super::*;

    fn strip_ansi(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\u{1b}' {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    fn run(
        dataset: &str,
        pattern: Option<AccessPattern>,
        format: Format,
        micros: u64,
    ) -> RandomAccessRun {
        RandomAccessRun {
            timing: TimingMeasurement {
                name: format!("random-access/{dataset}/{}-tokio-local-disk", format.ext()),
                target: Target::new(Engine::Vortex, format),
                storage: "nvme".to_string(),
                runs: vec![Duration::from_micros(micros)],
            },
            dataset: dataset.to_string(),
            pattern,
            display_name: match pattern {
                Some(p) => format!("random-access/{dataset}/{}", p.name()),
                None => format!("random-access/{dataset}"),
            },
        }
    }

    #[test]
    fn column_label_uses_format_ext() {
        assert_eq!(column_label(Format::Parquet), "parquet");
        assert_eq!(column_label(Format::OnDiskVortex), "vortex");
    }

    #[test]
    fn render_emits_single_header_row_with_format_columns() -> Result<()> {
        let runs = vec![
            run("taxi", None, Format::Parquet, 100),
            run("taxi", None, Format::OnDiskVortex, 50),
            run("taxi", Some(AccessPattern::Uniform), Format::Parquet, 300),
            run(
                "taxi",
                Some(AccessPattern::Uniform),
                Format::OnDiskVortex,
                150,
            ),
        ];

        let mut buf = Vec::new();
        render_random_access_table(&mut buf, &runs, &[Format::Parquet, Format::OnDiskVortex])?;
        let rendered = strip_ansi(&String::from_utf8(buf)?);

        assert!(
            rendered.contains("parquet") && rendered.contains("vortex"),
            "expected ext-based column headers, got:\n{rendered}"
        );
        assert!(
            rendered.contains("random-access/taxi")
                && rendered.contains("random-access/taxi/uniform"),
            "expected display-name row labels, got:\n{rendered}"
        );
        Ok(())
    }

    #[test]
    fn render_renders_dash_for_missing_cells() -> Result<()> {
        // No vortex run supplied, but the column is still listed in `formats`
        // — that cell should render as `-`.
        let runs = vec![run("taxi", None, Format::Parquet, 100)];

        let mut buf = Vec::new();
        render_random_access_table(&mut buf, &runs, &[Format::Parquet, Format::OnDiskVortex])?;
        let rendered = strip_ansi(&String::from_utf8(buf)?);

        assert!(
            rendered.contains('-'),
            "expected `-` placeholder for missing cell, got:\n{rendered}"
        );
        Ok(())
    }
}
