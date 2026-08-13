// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Support for benchmarks measured once per CPU feature set.
//!
//! One attribute, [`cpu_features`]. Which feature sets exist, what each is built with, and
//! where it runs are all in `.github/workflows/codspeed.yml`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::ItemFn;
use syn::parse_macro_input;

/// Measure this benchmark on every walltime CPU-feature leg.
///
/// Every tagged benchmark runs on all of the legs; the argument decides only whether it *also*
/// stays in the sharded simulation job:
///
/// * `#[cpu_features]` — walltime legs only. For a benchmark added to measure the feature sets
///   in the first place, with no simulation history worth keeping.
/// * `#[cpu_features(with_simulation)]` — both. For a benchmark that already reports to CodSpeed
///   in simulation: the walltime legs add per-feature-set series next to it, and the existing
///   simulation series keeps its name and its history, so PRs still get the instruction-count
///   comparison they got before.
///
/// An untagged benchmark is unaffected by any of this and runs in the simulation job alone.
///
/// Write it *above* `#[divan::bench]`, whose arguments it fills in — the name is qualified with
/// the leg that produced it, so the legs report one series each rather than fighting over a
/// shared name, and (without `with_simulation`) `ignore` takes the benchmark out of the sharded
/// simulation job. A plain `cargo bench` runs it as before, under its bare name.
///
/// ```ignore
/// #[vortex_bench_support::cpu_features]
/// #[divan::bench(args = INPUT_SIZE)]
/// fn words_gather_dispatch(bencher: Bencher, len: usize) { /* ... */ }
///
/// #[vortex_bench_support::cpu_features(with_simulation)]
/// #[divan::bench]
/// fn compare_int(bencher: Bencher) { /* ... */ }
/// ```
///
/// Spell it out in full rather than importing it: benchmark files are read a function at a
/// time, and the path says where the behaviour comes from.
///
/// This is for code that is written once and *compiled* differently per feature set — a
/// shipped entry point that selects its kernel through `cfg(target_feature)`, or a scalar
/// loop whose auto-vectorization depends on the build. A hand-written kernel for one
/// instruction set extension is a different thing: it cannot run on the other legs, so it
/// does not belong here. Keep those on `#[cfg(not(codspeed))]` for local A/B runs.
#[proc_macro_attribute]
pub fn cpu_features(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = TokenStream2::from(attr);
    let keep_simulation = match attr.to_string().as_str() {
        "" => false,
        "with_simulation" => true,
        _ => {
            return syn::Error::new_spanned(
                attr,
                "`#[cpu_features]` takes at most `with_simulation`, which keeps the benchmark in \
                 the simulation job as well as on every feature-set leg",
            )
            .to_compile_error()
            .into();
        }
    };

    let mut function = parse_macro_input!(item as ItemFn);

    let Some(bench) = function.attrs.iter_mut().find(|attr| {
        attr.path()
            .segments
            .last()
            .is_some_and(|s| s.ident == "bench")
    }) else {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "`#[cpu_features]` must be written directly above the `#[divan::bench]` it applies to",
        )
        .to_compile_error()
        .into();
    };

    let existing: TokenStream2 = match &bench.meta {
        syn::Meta::Path(_) => TokenStream2::new(),
        syn::Meta::List(list) => list.tokens.clone(),
        syn::Meta::NameValue(value) => {
            return syn::Error::new_spanned(
                value,
                "expected `#[divan::bench]` or `#[divan::bench(..)]`",
            )
            .to_compile_error()
            .into();
        }
    };

    for reserved in ["name", "ignore"] {
        if existing
            .clone()
            .into_iter()
            .any(|token| matches!(&token, proc_macro2::TokenTree::Ident(i) if i == reserved))
        {
            return syn::Error::new_spanned(
                &existing,
                format!(
                    "`{reserved}` is set by `#[cpu_features]`; remove it from `#[divan::bench]`"
                ),
            )
            .to_compile_error()
            .into();
        }
    }

    // Skipping simulation, rather than naming the legs to run on, is what keeps the leg list
    // in the workflow alone. `env!` rather than reading the environment during expansion:
    // rustc records it as a dependency of the crate, so changing legs rebuilds the benchmarks.
    let name = function.sig.ident.to_string();
    // A multi-line `#[divan::bench(..)]` usually ends in a trailing comma; adding another
    // would leave `,,` behind.
    let trailing_comma = matches!(
        existing.clone().into_iter().last(),
        Some(proc_macro2::TokenTree::Punct(punct)) if punct.as_char() == ','
    );
    let separator = if existing.is_empty() || trailing_comma {
        quote!()
    } else {
        quote!(,)
    };
    let bench_path = bench.path().clone();
    // With `with_simulation` there is nothing to skip: the walltime legs pick the benchmark out by
    // the name prefix, and the simulation job runs it under its bare name as it always has.
    let ignore = if keep_simulation {
        quote!()
    } else {
        quote!(ignore = env!("VORTEX_BENCH_VARIANT") == "simulation",)
    };
    bench.meta = syn::parse_quote! {
        #bench_path(
            #existing #separator
            name = concat!(env!("VORTEX_BENCH_PREFIX"), #name),
            #ignore
        )
    };

    quote!(#function).into()
}
