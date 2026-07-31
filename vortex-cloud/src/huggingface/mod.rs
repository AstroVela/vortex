// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Resolution of `hf://` URLs against the Hugging Face Hub.
//!
//! The Hub serves repository files from a `resolve` endpoint that supports HTTP range requests, so
//! a Vortex file in a Hub repository can be scanned in place rather than downloaded first. This
//! module maps the `hf://` URIs used across the Hub ecosystem onto that endpoint, so that
//!
//! ```text
//! hf://datasets/<org>/<name>[@<revision>]/<path>
//! ```
//!
//! resolves to an HTTP store rooted at
//!
//! ```text
//! <endpoint>/datasets/<org>/<name>/resolve/<revision>
//! ```
//!
//! with `<path>` as the key within that store. The revision defaults to `main` and is passed
//! through exactly as written: one containing `/` (e.g. `refs/convert/parquet`) must be
//! percent-encoded by the caller, as `HfFileSystem` paths require, because the Hub only routes the
//! encoded form.
//!
//! Only dataset repositories are addressable, matching the repository type Vortex files are
//! published under. Model and Space repositories are rejected rather than silently mis-resolved.
//!
//! # Configuration
//!
//! * `HF_ENDPOINT` — the Hub endpoint, defaulting to `https://huggingface.co`.
//! * `HF_TOKEN`, or `HUGGING_FACE_HUB_TOKEN` — a token sent as an `authorization: Bearer` header on
//!   every request, for private and gated repositories. Reads are anonymous when neither is set.
//!
//! Client configuration otherwise follows every other scheme in the registry: the environment is
//! consulted for [`ClientConfigKey`] values such as `allow_http` and the request timeouts.

use std::sync::Arc;

use http::HeaderMap;
use http::HeaderValue;
use http::header::AUTHORIZATION;
use object_store::ClientConfigKey;
use object_store::ClientOptions;
use object_store::ObjectStore;
use object_store::http::HttpBuilder;
use object_store::path::Path;
use url::Url;

use crate::registry::path_segments;

/// The URL scheme served by this module.
pub const HF_SCHEME: &str = "hf";

/// The repository type addressable through `hf://`, taken from the URL authority.
const DATASETS_REPO_TYPE: &str = "datasets";

/// The Hub endpoint used when `HF_ENDPOINT` is unset.
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// The revision used when the URL does not name one.
const DEFAULT_REVISION: &str = "main";

/// The variable naming the Hub endpoint.
const ENDPOINT_VAR: &str = "hf_endpoint";

/// The variables consulted for a Hub token, in precedence order.
const TOKEN_VARS: [&str; 2] = ["hf_token", "hugging_face_hub_token"];

/// Whether `scheme` is served by this module.
pub fn supports_scheme(scheme: &str) -> bool {
    scheme == HF_SCHEME
}

/// Error building a store for an `hf://` URL.
#[derive(Debug)]
pub enum HuggingFaceError {
    /// The URL named a repository type other than `datasets`.
    UnsupportedRepoType(String),
    /// The URL was not of the form `hf://datasets/<org>/<name>[@<revision>][/<path>]`.
    MalformedUrl(String),
}

impl std::fmt::Display for HuggingFaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HuggingFaceError::UnsupportedRepoType(repo_type) => write!(
                f,
                "unsupported HuggingFace repository type '{repo_type}': only hf://datasets/... is supported"
            ),
            HuggingFaceError::MalformedUrl(url) => write!(
                f,
                "malformed HuggingFace URL '{url}': expected hf://datasets/<org>/<name>[@revision][/path]"
            ),
        }
    }
}

impl std::error::Error for HuggingFaceError {}

impl From<HuggingFaceError> for object_store::Error {
    fn from(error: HuggingFaceError) -> Self {
        object_store::Error::Generic {
            store: "HuggingFace",
            source: Box::new(error),
        }
    }
}

/// A `hf://` URL split into the repository it addresses and the file within it.
#[derive(Debug, PartialEq, Eq)]
struct HfLocation {
    /// `<org>/<name>`.
    repo_id: String,
    /// The revision exactly as written in the URL, still percent-encoded.
    revision: String,
    /// The path within the repository, empty when the URL names only the repository.
    path: String,
}

/// Splits `url` into the repository it addresses and the path within it.
fn parse_hf_url(url: &Url) -> Result<HfLocation, HuggingFaceError> {
    let repo_type = url.host_str().unwrap_or_default();
    if repo_type != DATASETS_REPO_TYPE {
        return Err(HuggingFaceError::UnsupportedRepoType(repo_type.to_string()));
    }

    let malformed = || HuggingFaceError::MalformedUrl(url.to_string());
    let mut segments = path_segments(url.path());
    let org = segments.next().ok_or_else(malformed)?;
    let name_and_revision = segments.next().ok_or_else(malformed)?;

    // A revision is appended to the repository name, as in `HfFileSystem` paths. Both halves must
    // be non-empty, so `name@` and `@main` are rejected rather than resolving a bogus endpoint.
    let (name, revision) = match name_and_revision.split_once('@') {
        Some((name, revision)) if !name.is_empty() && !revision.is_empty() => (name, revision),
        Some(_) => return Err(malformed()),
        None => (name_and_revision, DEFAULT_REVISION),
    };

    Ok(HfLocation {
        repo_id: format!("{org}/{name}"),
        revision: revision.to_string(),
        path: segments.collect::<Vec<_>>().join("/"),
    })
}

/// Looks up a configuration variable among `vars`, whose keys are already lowercased.
fn var<'a>(vars: &'a [(String, String)], key: &str) -> Option<&'a str> {
    vars.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// The client options for a Hub store: the [`ClientConfigKey`] values present in `vars`, as every
/// other scheme in the registry resolves them, plus the bearer token when one is configured.
fn client_options(vars: &[(String, String)]) -> object_store::Result<ClientOptions> {
    let mut options = vars
        .iter()
        .filter_map(|(key, value)| key.parse::<ClientConfigKey>().ok().map(|key| (key, value)))
        .fold(ClientOptions::new(), |options, (key, value)| {
            options.with_config(key, value)
        });

    if let Some(token) = TOKEN_VARS.iter().find_map(|name| var(vars, name)) {
        let mut value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|e| {
            object_store::Error::Generic {
                store: "HuggingFace",
                source: Box::new(e),
            }
        })?;
        // Marking the header sensitive keeps the token out of any `Debug` rendering of the client.
        value.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, value);
        options = options.with_default_headers(headers);
    }

    Ok(options)
}

/// The Hub `resolve` URL that `location`'s repository revision is served from.
fn base_url(location: &HfLocation, vars: &[(String, String)]) -> String {
    let endpoint = var(vars, ENDPOINT_VAR).unwrap_or(DEFAULT_ENDPOINT);
    format!(
        "{}/{DATASETS_REPO_TYPE}/{}/resolve/{}",
        endpoint.trim_end_matches('/'),
        location.repo_id,
        location.revision
    )
}

/// Builds the store serving `url`, alongside the path to the file within that store.
///
/// `vars` supplies configuration with lowercased keys; see the [module docs](self) for the ones
/// consulted. The store is rooted at the repository revision, so every file in one revision of one
/// repository shares a single client.
pub(crate) fn make_hf_store(
    url: &Url,
    vars: &[(String, String)],
) -> object_store::Result<(Arc<dyn ObjectStore>, Path)> {
    let location = parse_hf_url(url)?;
    let store = HttpBuilder::new()
        .with_url(base_url(&location, vars))
        .with_client_options(client_options(vars)?)
        .build()?;

    Ok((Arc::new(store), Path::from_url_path(&location.path)?))
}

#[cfg(test)]
mod tests;
