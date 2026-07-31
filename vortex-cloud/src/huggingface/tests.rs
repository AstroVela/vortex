// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use url::Url;

use super::*;

fn vars(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

#[rstest]
#[case(
    "hf://datasets/org/name/train.vortex",
    "org/name",
    "main",
    "train.vortex"
)]
#[case("hf://datasets/org/name", "org/name", "main", "")]
#[case("hf://datasets/org/name/", "org/name", "main", "")]
#[case(
    "hf://datasets/org/name/data/nested/train.vortex",
    "org/name",
    "main",
    "data/nested/train.vortex"
)]
#[case(
    "hf://datasets/org/name@v1.0/train.vortex",
    "org/name",
    "v1.0",
    "train.vortex"
)]
// A revision containing `/` is percent-encoded by the caller and must be passed through as
// written, because the Hub only routes the encoded form.
#[case(
    "hf://datasets/org/name@refs%2Fconvert%2Fparquet/data/train.vortex",
    "org/name",
    "refs%2Fconvert%2Fparquet",
    "data/train.vortex"
)]
fn test_parse_hf_url(
    #[case] url: &str,
    #[case] repo_id: &str,
    #[case] revision: &str,
    #[case] path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let location = parse_hf_url(&Url::parse(url)?)?;
    assert_eq!(location.repo_id, repo_id);
    assert_eq!(location.revision, revision);
    assert_eq!(location.path, path);
    Ok(())
}

#[rstest]
// Model repositories put the org in the authority, so they are not dataset URLs.
#[case("hf://org/name/train.vortex")]
#[case("hf://spaces/org/name/train.vortex")]
#[case("hf://datasets/org")]
#[case("hf://datasets/org/name@/train.vortex")]
#[case("hf://datasets/org/@main/train.vortex")]
fn test_parse_hf_url_rejected(#[case] url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let err = parse_hf_url(&Url::parse(url)?).expect_err("URL should be rejected");
    // The message must point at the shape a caller should have written.
    assert!(
        err.to_string().contains("hf://datasets/"),
        "unhelpful rejection for {url}: {err}"
    );
    Ok(())
}

/// The store is rooted at the repository revision, so the base URL carries the repository and the
/// returned path carries only the file. The registry derives its cache depth from that split.
#[rstest]
#[case(
    "hf://datasets/org/name/data/train.vortex",
    &[],
    "https://huggingface.co/datasets/org/name/resolve/main",
    "data/train.vortex"
)]
#[case(
    "hf://datasets/org/name@dev/train.vortex",
    &[("hf_endpoint", "https://hub.example.com")],
    "https://hub.example.com/datasets/org/name/resolve/dev",
    "train.vortex"
)]
// A trailing slash on the endpoint must not double up in the resolved base URL.
#[case(
    "hf://datasets/org/name/train.vortex",
    &[("hf_endpoint", "https://hub.example.com/")],
    "https://hub.example.com/datasets/org/name/resolve/main",
    "train.vortex"
)]
#[case(
    "hf://datasets/org/name@refs%2Fconvert%2Fparquet/data/train.vortex",
    &[],
    "https://huggingface.co/datasets/org/name/resolve/refs%2Fconvert%2Fparquet",
    "data/train.vortex"
)]
fn test_hf_store_root_and_path(
    #[case] url: &str,
    #[case] env: &[(&str, &str)],
    #[case] expected_base: &str,
    #[case] expected_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse(url)?;
    let env = vars(env);

    assert_eq!(base_url(&parse_hf_url(&url)?, &env), expected_base);

    // The store must actually build from that base, and report the path the registry will cache on.
    let (_store, path) = make_hf_store(&url, &env)?;
    assert_eq!(path.as_ref(), expected_path);
    Ok(())
}

/// A token must be sent as a bearer header, and must not leak into the `Debug` rendering that the
/// registry's own tests print on failure.
#[rstest]
#[case("hf_token")]
#[case("hugging_face_hub_token")]
fn test_token_becomes_redacted_bearer_header(
    #[case] name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = client_options(&vars(&[(name, "hf_secret_value")]))?;
    let headers = options
        .get_default_headers()
        .expect("a token should install default headers");
    let authorization = headers
        .get(AUTHORIZATION)
        .expect("the token should be sent as an authorization header");

    assert!(authorization.is_sensitive());
    assert_eq!(authorization.to_str()?, "Bearer hf_secret_value");
    assert!(
        !format!("{options:?}").contains("hf_secret_value"),
        "the token leaked into the client options debug output"
    );
    Ok(())
}

#[test]
fn test_no_token_reads_anonymously() -> Result<(), Box<dyn std::error::Error>> {
    assert!(client_options(&[])?.get_default_headers().is_none());
    Ok(())
}

/// Client configuration reaches the store the same way it does for every other scheme, so a
/// plaintext Hub endpoint needs the same `allow_http` opt-in as any other plaintext URL.
#[test]
fn test_client_config_keys_are_applied() -> Result<(), Box<dyn std::error::Error>> {
    let allow_http = |vars: &[(&str, &str)]| -> object_store::Result<Option<String>> {
        Ok(client_options(&self::vars(vars))?.get_config_value(&ClientConfigKey::AllowHttp))
    };

    assert_eq!(allow_http(&[])?.as_deref(), Some("false"));
    assert_eq!(
        allow_http(&[("allow_http", "true")])?.as_deref(),
        Some("true")
    );
    Ok(())
}
