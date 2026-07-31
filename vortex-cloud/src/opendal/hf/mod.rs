// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The HuggingFace Hub, served over OpenDAL's `services::Hf`.
//!
//! The Hub serves repository files from a `resolve` endpoint that supports HTTP range requests, so
//! a Vortex file in a Hub repository is scanned in place rather than downloaded first. URLs take
//! the form the Hub ecosystem uses:
//!
//! ```text
//! hf://<repo-type>/<org>/<name>[@<revision>]/<path>
//! ```
//!
//! where `<repo-type>` is `datasets`, `models` or `spaces`. The revision defaults to `main`.
//!
//! Unlike the bucket-style OpenDAL schemes, the store is rooted at a repository revision rather
//! than at the URL authority, so the key within the store is only the path *inside* the
//! repository. [`make_hf_store_for_url`] returns both halves for that reason.
//!
//! Going through OpenDAL rather than a plain HTTP store is what makes listing work: the Hub is not
//! a WebDAV server, so `PROPFIND` is refused, and directory listings have to come from the Hub's
//! JSON API (`/api/<repo-type>/<repo>/tree/<revision>`), which this service calls.

use std::sync::Arc;

use ::opendal::services;
use object_store::ObjectStore;
use object_store::path::Path;
use object_store_opendal::OpendalStore;
use url::Url;
use vortex_utils::aliases::hash_map::HashMap;

use crate::opendal::OpenDALStoreError;
use crate::opendal::build_operator;
use crate::opendal::property_or_env;
use crate::opendal::warn_on_unknown_properties;

/// The URL scheme served by the HuggingFace Hub.
pub const HF_SCHEME: &str = "hf";

/// Property keys recognized for `hf://` URLs. Anything else is warned about and dropped.
const KNOWN_PROPERTIES: &[&str] = &[
    "repo_type",
    "repo_id",
    "revision",
    "root",
    "token",
    "endpoint",
];

/// The revision used when the URL does not name one.
const DEFAULT_REVISION: &str = "main";

/// The variables consulted for a Hub token, in precedence order.
const TOKEN_VARS: [&str; 2] = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"];

/// The variable naming the Hub endpoint.
const ENDPOINT_VAR: &str = "HF_ENDPOINT";

/// Strongly-typed configuration for building an OpenDAL store against the HuggingFace Hub.
///
/// Building from an `HfConfig` avoids the URL round-trip that
/// [`crate::opendal::make_opendal_store`] uses, and is the preferred way to construct a Hub store.
#[derive(Debug, Clone, Default)]
pub struct HfConfig {
    /// Repository type: `dataset`, `model` or `space`.
    pub repo_type: String,
    /// Repository id, `<org>/<name>`.
    pub repo_id: String,
    /// Revision (branch, tag or commit). Defaults to `main`.
    ///
    /// Stored decoded: OpenDAL percent-encodes it when building the resolve URL, so a revision
    /// containing `/` (e.g. `refs/convert/parquet`) must *not* be pre-encoded here.
    pub revision: Option<String>,
    /// Optional root prefix within the repository, applied to all operations.
    pub root: Option<String>,
    /// Hub token, for private and gated repositories (mapped to `HF_TOKEN`).
    pub token: Option<String>,
    /// Hub endpoint (mapped to `HF_ENDPOINT`). Defaults to OpenDAL's `https://huggingface.co`.
    pub endpoint: Option<String>,
}

/// The non-empty segments of a URL path.
///
/// Defined here rather than shared with the registry so that `hf` builds without `registry`.
fn path_segments(path: &str) -> impl Iterator<Item = &str> {
    path.split('/').filter(|segment| !segment.is_empty())
}

/// The repository types the Hub addresses, keyed by the URL authority that names them.
///
/// The authority is plural (as in `hf://datasets/...`, matching `HfFileSystem` paths) while
/// OpenDAL's builder takes the singular form.
const REPO_TYPES: &[(&str, &str)] = &[
    ("datasets", "dataset"),
    ("models", "model"),
    ("spaces", "space"),
];

/// Build an [`object_store::ObjectStore`] for the HuggingFace Hub directly from an [`HfConfig`].
pub fn make_hf_store(config: HfConfig) -> Result<Arc<dyn ObjectStore>, OpenDALStoreError> {
    if config.repo_type.is_empty() {
        return Err(OpenDALStoreError::MissingConfig("repo_type"));
    }
    if config.repo_id.is_empty() {
        return Err(OpenDALStoreError::MissingConfig("repo_id"));
    }

    let mut builder = services::Hf::default()
        .repo_type(&config.repo_type)
        .repo_id(&config.repo_id)
        .revision(config.revision.as_deref().unwrap_or(DEFAULT_REVISION));

    if let Some(root) = config.root.as_deref() {
        builder = builder.root(root);
    }
    if let Some(token) = config.token.as_deref() {
        builder = builder.token(token);
    }
    if let Some(endpoint) = config.endpoint.as_deref() {
        builder = builder.endpoint(endpoint);
    }

    let operator = build_operator(builder)?;
    Ok(Arc::new(OpendalStore::new(operator)))
}

/// Translate an (`hf://` URL, properties) pair into an [`HfConfig`] plus the path within the
/// repository.
///
/// The repository type comes from the URL authority, the repository id and revision from the first
/// two path segments, and everything after them is the path within the repository. Explicit
/// properties win over the URL; the token and endpoint fall back to the environment.
pub(crate) fn url_and_properties_to_config<F>(
    url: &Url,
    properties: &HashMap<String, String>,
    env_lookup: F,
) -> Result<(HfConfig, String), OpenDALStoreError>
where
    F: Fn(&str) -> Option<String>,
{
    warn_on_unknown_properties(properties, KNOWN_PROPERTIES);

    let invalid = || OpenDALStoreError::InvalidUrl(url.to_string());

    let authority = url.host_str().unwrap_or_default();
    let repo_type = properties
        .get("repo_type")
        .cloned()
        .or_else(|| {
            REPO_TYPES
                .iter()
                .find(|(plural, _)| *plural == authority)
                .map(|(_, singular)| (*singular).to_string())
        })
        .ok_or_else(invalid)?;

    let mut segments = path_segments(url.path());
    let (Some(org), Some(name_and_revision)) = (segments.next(), segments.next()) else {
        return Err(invalid());
    };

    // A revision is appended to the repository name, as in `HfFileSystem` paths. Both halves must
    // be non-empty, so `name@` and `@main` are rejected rather than addressing a bogus repository.
    let (name, revision) = match name_and_revision.split_once('@') {
        Some((name, revision)) if !name.is_empty() && !revision.is_empty() => {
            (name, Some(revision))
        }
        Some(_) => return Err(invalid()),
        None => (name_and_revision, None),
    };

    // OpenDAL percent-encodes the revision when it builds the resolve URL, so hand it the decoded
    // form: `@refs%2Fconvert%2Fparquet` in the URL means the revision `refs/convert/parquet`.
    let revision = properties.get("revision").cloned().or_else(|| {
        revision.map(|revision| {
            percent_encoding::percent_decode_str(revision)
                .decode_utf8_lossy()
                .into_owned()
        })
    });

    let path = segments.collect::<Vec<_>>().join("/");

    let config = HfConfig {
        repo_type,
        repo_id: properties
            .get("repo_id")
            .cloned()
            .unwrap_or_else(|| format!("{org}/{name}")),
        revision,
        root: properties.get("root").cloned(),
        token: property_or_env(properties, "token", TOKEN_VARS[0], &env_lookup)
            .or_else(|| env_lookup(TOKEN_VARS[1])),
        endpoint: property_or_env(properties, "endpoint", ENDPOINT_VAR, &env_lookup),
    };
    Ok((config, path))
}

/// Build the store serving `url`, alongside the path to the file within the repository.
///
/// The bucket-style OpenDAL schemes let the registry derive the path itself, because their store
/// is rooted at the URL authority and the whole path is the key. A Hub store is rooted at a
/// repository revision instead, so the split has to come from here.
pub(crate) fn make_hf_store_for_url<F>(
    url: &Url,
    properties: &HashMap<String, String>,
    env_lookup: F,
) -> Result<(Arc<dyn ObjectStore>, Path), OpenDALStoreError>
where
    F: Fn(&str) -> Option<String>,
{
    let (config, path) = url_and_properties_to_config(url, properties, env_lookup)?;
    let store = make_hf_store(config)?;
    // The path comes from a URL, so percent-decode it: OpenDAL re-encodes when building the
    // resolve URL, and double-encoding would address a file that does not exist.
    let path =
        Path::from_url_path(&path).map_err(|_| OpenDALStoreError::InvalidUrl(url.to_string()))?;
    Ok((store, path))
}

#[cfg(test)]
mod tests;
