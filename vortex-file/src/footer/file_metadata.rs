// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the file metadata component of the Vortex file footer.
//!
//! File metadata is an arbitrary user-provided key/value mapping that is stored in its own
//! segment and is fully opaque to Vortex, for example an Iceberg field-id mapping.

use std::sync::Arc;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::Follow;
use flatbuffers::WIPOffset;
use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::ReadFlatBuffer;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::footer as fb;
use vortex_utils::aliases::hash_map::HashMap;

/// Arbitrary user-provided key/value metadata stored in a Vortex file.
///
/// The mapping is fully opaque to Vortex: keys and values are arbitrary bytes. Entries are
/// serialized sorted by key so that writes are deterministic.
#[derive(Clone, Debug, Default)]
pub struct FileMetadata {
    entries: Arc<HashMap<ByteBuffer, ByteBuffer>>,
}

impl FileMetadata {
    /// Creates a new [`FileMetadata`] from the given key/value entries.
    pub fn new(entries: HashMap<ByteBuffer, ByteBuffer>) -> Self {
        Self {
            entries: Arc::new(entries),
        }
    }

    /// Returns the value for the given key, if present.
    pub fn get(&self, key: impl AsRef<[u8]>) -> Option<&ByteBuffer> {
        self.entries.get(&ByteBuffer::copy_from(key))
    }

    /// Returns the key/value entries.
    pub fn entries(&self) -> &HashMap<ByteBuffer, ByteBuffer> {
        &self.entries
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<K: Into<ByteBuffer>, V: Into<ByteBuffer>> FromIterator<(K, V)> for FileMetadata {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        Self::new(
            iter.into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        )
    }
}

impl FlatBufferRoot for FileMetadata {}

impl WriteFlatBuffer for FileMetadata {
    type Target<'a> = fb::FileMetadata<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        let entries = self
            .entries
            .iter()
            .sorted_by(|(left, _), (right, _)| left.as_slice().cmp(right.as_slice()))
            .map(|(key, value)| {
                let key = fbb.create_vector(key.as_slice());
                let value = fbb.create_vector(value.as_slice());
                fb::MetadataEntry::create(
                    fbb,
                    &fb::MetadataEntryArgs {
                        key: Some(key),
                        value: Some(value),
                    },
                )
            })
            .collect::<Vec<_>>();
        let entries = fbb.create_vector(entries.as_slice());

        Ok(fb::FileMetadata::create(
            fbb,
            &fb::FileMetadataArgs {
                entries: Some(entries),
            },
        ))
    }
}

impl ReadFlatBuffer for FileMetadata {
    type Source<'a> = fb::FileMetadata<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error> {
        let fb_entries = fb.entries().unwrap_or_default();
        let mut entries = HashMap::with_capacity(fb_entries.len());
        for entry in fb_entries {
            let key = ByteBuffer::copy_from(entry.key().bytes());
            if entries
                .insert(key, ByteBuffer::copy_from(entry.value().bytes()))
                .is_some()
            {
                vortex_bail!("Duplicate key in file metadata");
            }
        }
        Ok(Self::new(entries))
    }
}
