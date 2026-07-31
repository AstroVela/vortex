// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;

use super::*;

/// An env lookup that finds nothing, so these tests never read the process environment.
fn no_env(_: &str) -> Option<String> {
    None
}

fn config_for(url: &str) -> Result<(HfConfig, String), Box<dyn std::error::Error>> {
    let url = Url::parse(url)?;
    Ok(url_and_properties_to_config(&url, &HashMap::new(), no_env)?)
}

#[rstest]
#[case(
    "hf://datasets/org/name/train.vortex",
    "dataset",
    "org/name",
    None,
    "train.vortex"
)]
#[case(
    "hf://models/org/name/model.vortex",
    "model",
    "org/name",
    None,
    "model.vortex"
)]
#[case(
    "hf://spaces/org/name/app.vortex",
    "space",
    "org/name",
    None,
    "app.vortex"
)]
#[case("hf://datasets/org/name", "dataset", "org/name", None, "")]
#[case(
    "hf://datasets/org/name/data/nested/train.vortex",
    "dataset",
    "org/name",
    None,
    "data/nested/train.vortex"
)]
#[case(
    "hf://datasets/org/name@v1.0/train.vortex",
    "dataset",
    "org/name",
    Some("v1.0"),
    "train.vortex"
)]
// OpenDAL percent-encodes the revision when building the resolve URL, so the decoded form is what
// must reach the builder — otherwise `refs/convert/parquet` would be double-encoded.
#[case(
    "hf://datasets/org/name@refs%2Fconvert%2Fparquet/data/train.vortex",
    "dataset",
    "org/name",
    Some("refs/convert/parquet"),
    "data/train.vortex"
)]
fn test_url_to_config(
    #[case] url: &str,
    #[case] repo_type: &str,
    #[case] repo_id: &str,
    #[case] revision: Option<&str>,
    #[case] path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (config, resolved_path) = config_for(url)?;
    assert_eq!(config.repo_type, repo_type);
    assert_eq!(config.repo_id, repo_id);
    assert_eq!(config.revision.as_deref(), revision);
    assert_eq!(resolved_path, path);
    Ok(())
}

#[rstest]
// A bare `hf://<org>/<name>` is ambiguous with the repo-type authorities, so it is rejected.
#[case("hf://org/name/train.vortex")]
#[case("hf://datasets/org")]
#[case("hf://datasets/org/name@/train.vortex")]
#[case("hf://datasets/org/@main/train.vortex")]
fn test_url_rejected(#[case] url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let url_parsed = Url::parse(url)?;
    let err = url_and_properties_to_config(&url_parsed, &HashMap::new(), no_env)
        .expect_err("URL should be rejected");
    assert!(
        matches!(err, OpenDALStoreError::InvalidUrl(_)),
        "unexpected error for {url}: {err}"
    );
    // The message must point at the shape a caller should have written.
    assert!(
        err.to_string().contains("hf://"),
        "unhelpful message: {err}"
    );
    Ok(())
}

/// Explicit properties must win over what the URL says, matching the other OpenDAL services.
#[test]
fn test_properties_override_url() -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse("hf://datasets/url-org/url-name@url-rev/train.vortex")?;
    let mut properties = HashMap::new();
    properties.insert("repo_id".to_string(), "prop-org/prop-name".to_string());
    properties.insert("revision".to_string(), "prop-rev".to_string());
    properties.insert("token".to_string(), "prop-token".to_string());

    let (config, path) = url_and_properties_to_config(&url, &properties, no_env)?;
    assert_eq!(config.repo_id, "prop-org/prop-name");
    assert_eq!(config.revision.as_deref(), Some("prop-rev"));
    assert_eq!(config.token.as_deref(), Some("prop-token"));
    // The path within the repository still comes from the URL.
    assert_eq!(path, "train.vortex");
    Ok(())
}

/// Both token variables are consulted, with `HF_TOKEN` taking precedence.
#[rstest]
#[case(&[("HF_TOKEN", "primary")], "primary")]
#[case(&[("HUGGING_FACE_HUB_TOKEN", "fallback")], "fallback")]
#[case(&[("HF_TOKEN", "primary"), ("HUGGING_FACE_HUB_TOKEN", "fallback")], "primary")]
fn test_token_from_env(
    #[case] env: &[(&str, &str)],
    #[case] expected: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse("hf://datasets/org/name/train.vortex")?;
    let env: Vec<(String, String)> = env
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    let lookup = |key: &str| env.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());

    let (config, _path) = url_and_properties_to_config(&url, &HashMap::new(), lookup)?;
    assert_eq!(config.token.as_deref(), Some(expected));
    Ok(())
}

#[test]
fn test_endpoint_from_env() -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse("hf://datasets/org/name/train.vortex")?;
    let lookup = |key: &str| (key == "HF_ENDPOINT").then(|| "https://hub.example.com".to_string());

    let (config, _path) = url_and_properties_to_config(&url, &HashMap::new(), lookup)?;
    assert_eq!(config.endpoint.as_deref(), Some("https://hub.example.com"));
    Ok(())
}

/// The strongly-typed entry point must reject an empty repository before touching the builder.
#[test]
fn test_config_rejects_empty_fields() {
    assert!(matches!(
        make_hf_store(HfConfig::default()),
        Err(OpenDALStoreError::MissingConfig("repo_type"))
    ));
    assert!(matches!(
        make_hf_store(HfConfig {
            repo_type: "dataset".to_string(),
            ..HfConfig::default()
        }),
        Err(OpenDALStoreError::MissingConfig("repo_id"))
    ));
}

/// The store is rooted at the repository revision, so the key it reports is only the path inside
/// the repository. The registry derives its cache depth from that split.
#[test]
fn test_store_builds_with_repo_relative_path() -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse("hf://datasets/org/name/data/train.vortex")?;
    let (store, path) = make_hf_store_for_url(&url, &HashMap::new(), no_env)?;

    assert_eq!(path.as_ref(), "data/train.vortex");
    assert!(Arc::strong_count(&store) >= 1);
    Ok(())
}
