// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::net::SocketAddr;
use std::net::TcpListener;
use std::thread;

use anyhow::Context;
use axum::Router;
use axum::body::Body;
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use tracing::error;
use tracing::info;

const DEFAULT_ADDRESS: &str = "127.0.0.1:6060";
const ADDRESS_ENV: &str = "VORTEX_HEAP_PROFILE_ADDR";

/// Start a localhost HTTP server that exposes the process's sampled jemalloc heap data.
pub(crate) fn start_heap_profile_server() -> anyhow::Result<()> {
    let address = env::var(ADDRESS_ENV).unwrap_or_else(|_| DEFAULT_ADDRESS.to_owned());
    let address = address
        .parse::<SocketAddr>()
        .with_context(|| format!("invalid {ADDRESS_ENV} value {address:?}"))?;
    let listener = TcpListener::bind(address)
        .with_context(|| format!("failed to bind jemalloc pprof server to {address}"))?;
    listener.set_nonblocking(true)?;

    thread::Builder::new()
        .name("jemalloc-pprof".to_owned())
        .spawn(move || {
            if let Err(error) = serve(listener, address) {
                error!(%error, "jemalloc pprof server stopped");
            }
        })
        .context("failed to start jemalloc pprof server thread")?;

    Ok(())
}

fn serve(listener: TcpListener, address: SocketAddr) -> anyhow::Result<()> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener)?;
            let router = Router::new()
                .route("/debug/pprof/allocs", get(heap_profile))
                .route("/debug/pprof/heap", get(heap_profile));

            info!(%address, "jemalloc pprof server listening");
            axum::serve(listener, router).await?;
            Ok::<(), anyhow::Error>(())
        })
}

async fn heap_profile() -> Response {
    let Some(prof_ctl) = jemalloc_pprof::PROF_CTL.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "jemalloc profiling is unavailable",
        )
            .into_response();
    };

    let mut prof_ctl = prof_ctl.lock().await;
    if !prof_ctl.activated() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "jemalloc profiling is inactive",
        )
            .into_response();
    }

    match prof_ctl.dump_pprof() {
        Ok(pprof) => Response::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "application/octet-stream")
            .body(Body::from(pprof))
            .unwrap_or_else(|error| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to build heap profile response: {error}"),
                )
                    .into_response()
            }),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to dump jemalloc heap profile: {error}"),
        )
            .into_response(),
    }
}
