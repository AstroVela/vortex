// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared whole-file and streaming SHA-256 identities.

use base16ct::HexDisplay;
use ring::digest;

pub(super) fn sha256(bytes: &[u8]) -> String {
    format_digest(digest::digest(&digest::SHA256, bytes))
}

#[cfg(any(test, all(unix, feature = "local-store")))]
pub(super) struct Sha256(digest::Context);

#[cfg(any(test, all(unix, feature = "local-store")))]
impl Sha256 {
    pub fn new() -> Self {
        Self(digest::Context::new(&digest::SHA256))
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    pub fn finalize(self) -> String {
        format_digest(self.0.finish())
    }
}

fn format_digest(digest: digest::Digest) -> String {
    format!("sha256:{:x}", HexDisplay(digest.as_ref()))
}

#[cfg(test)]
mod tests;
