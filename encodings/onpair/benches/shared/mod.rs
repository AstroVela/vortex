// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Corpus shapes shared by the OnPair benchmarks.

#![allow(dead_code, clippy::cast_possible_truncation)]

/// Corpus shape. Each one stresses a different part of the compress / decode
/// path: token reuse, per-row overhead, or dictionary pressure.
#[derive(Copy, Clone, Debug)]
pub enum Shape {
    /// URL / HTTP-log shaped — high lexical overlap, ~35–45 bytes per row.
    UrlLog,
    /// Short uniform strings — 4–8 bytes per row, very low cardinality. Every
    /// row is inlined in its `BinaryView`, so no data buffer is read.
    Short,
    /// Long log-line shaped — ~120 bytes per row, more tokens per row.
    Long,
    /// High cardinality — every row unique.
    HighCard,
}

/// Deterministic corpus of `n` rows in the given [`Shape`].
pub fn corpus(n: usize, shape: Shape) -> Vec<String> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let mut out = Vec::with_capacity(n);
    match shape {
        Shape::UrlLog => {
            let templates: &[&str] = &[
                "https://www.example.com/products/{id}",
                "https://cdn.example.com/img/{id}.webp",
                "https://api.example.com/v2/orders/{id}",
                "https://www.example.com/users/{id}/profile",
                "INFO  request_id={id} status=200 method=GET",
                "WARN  request_id={id} status=429 method=POST",
                "ERROR request_id={id} status=500 method=PUT",
            ];
            for _ in 0..n {
                let s = next();
                let pick = (s as usize) % templates.len();
                let id = s as u32;
                out.push(templates[pick].replace("{id}", &format!("{id:08x}")));
            }
        }
        Shape::Short => {
            let templates: &[&str] = &["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta"];
            for _ in 0..n {
                let s = next();
                out.push(templates[(s as usize) % templates.len()].to_string());
            }
        }
        Shape::Long => {
            let templates: &[&str] = &[
                "2026-05-14T12:34:56.789012Z INFO  request_id={id} method=GET path=/api/v1/users/{id}/profile status=200",
                "2026-05-14T12:34:56.789012Z WARN  request_id={id} method=POST path=/api/v1/users/{id}/sessions status=429",
                "2026-05-14T12:34:56.789012Z ERROR request_id={id} method=PUT  path=/api/v1/users/{id}/settings status=500",
            ];
            for _ in 0..n {
                let s = next();
                let pick = (s as usize) % templates.len();
                let id = s as u32;
                out.push(templates[pick].replace("{id}", &format!("{id:08x}")));
            }
        }
        Shape::HighCard => {
            for i in 0..n {
                out.push(format!("row-{i:010x}-{rand:016x}", rand = next()));
            }
        }
    }
    out
}
