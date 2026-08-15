// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;

use super::SharedBytes;
use super::UniqueBytes;
use crate::Alignment;

/// A window of `len` bytes filled with the low byte of each index.
fn filled(len: usize, alignment: Alignment) -> UniqueBytes {
    let mut bytes = UniqueBytes::with_capacity(len, alignment);
    bytes.extend_from_slice(&pattern(len), alignment);
    bytes
}

/// The low byte of each index in `0..len`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "truncating to the low byte is the point of the pattern"
)]
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| i as u8).collect()
}

#[rstest]
#[case(1)]
#[case(8)]
#[case(64)]
#[case(256)]
#[case(4096)]
fn allocations_meet_their_alignment(#[case] alignment: usize) {
    let alignment = Alignment::new(alignment);
    let bytes = filled(1000, alignment);
    assert!(alignment.is_ptr_aligned(bytes.as_ptr()));
    assert_eq!(bytes.len(), 1000);
    assert!(bytes.capacity() >= 1000);
}

#[test]
fn empty_allocates_nothing_and_satisfies_any_alignment() {
    let bytes = UniqueBytes::with_capacity(0, Alignment::new(4096));
    assert_eq!(bytes.capacity(), 0);
    assert!(Alignment::MAX.is_ptr_aligned(bytes.as_ptr()));

    let shared = SharedBytes::empty();
    assert_eq!(shared.len(), 0);
    assert!(Alignment::MAX.is_ptr_aligned(shared.as_ptr()));
    assert!(shared.as_slice().is_empty());
}

#[test]
fn zeroed_is_zeroed() {
    let bytes = UniqueBytes::zeroed(129, Alignment::new(64));
    assert_eq!(bytes.len(), 129);
    assert_eq!(bytes.as_slice(), &[0u8; 129]);
}

#[test]
fn growth_preserves_contents_and_alignment() {
    let alignment = Alignment::new(512);
    let mut bytes = filled(16, alignment);
    for i in 0..1000u32 {
        bytes.extend_from_slice(&i.to_le_bytes(), alignment);
    }
    assert!(alignment.is_ptr_aligned(bytes.as_ptr()));
    assert_eq!(bytes.len(), 16 + 4000);
    assert_eq!(&bytes.as_slice()[..16], &pattern(16)[..]);
    assert_eq!(&bytes.as_slice()[16..20], &0u32.to_le_bytes());
    assert_eq!(&bytes.as_slice()[4012..4016], &999u32.to_le_bytes());
}

#[test]
fn shared_slices_share_the_allocation() {
    let shared = filled(64, Alignment::none()).freeze();
    assert!(shared.is_unique());

    let head = shared.slice(0, 8);
    assert!(!shared.is_unique());
    assert_eq!(head.as_slice(), &pattern(8)[..]);
    assert_eq!(head.as_ptr(), shared.as_ptr());

    let tail = shared.slice(8, 16);
    assert_eq!(tail.as_slice(), &pattern(16)[8..]);

    drop(head);
    drop(tail);
    assert!(shared.is_unique());
}

#[test]
fn slice_ref_recovers_the_offset() {
    let shared = filled(64, Alignment::none()).freeze();
    let subset = &shared.as_slice()[10..20];
    let sliced = shared.slice_ref(subset);
    assert_eq!(sliced.len(), 10);
    assert_eq!(sliced.as_ptr(), subset.as_ptr());
}

#[test]
#[should_panic(expected = "subset pointer")]
fn slice_ref_rejects_foreign_slices() {
    let shared = filled(64, Alignment::none()).freeze();
    let other = filled(64, Alignment::none()).freeze();
    shared.slice_ref(&other.as_slice()[..8]);
}

#[test]
fn try_into_unique_requires_sole_ownership() {
    let shared = filled(16, Alignment::none()).freeze();
    let clone = shared.clone();

    let shared = shared.try_into_unique().expect_err("two handles exist");
    drop(clone);
    assert!(shared.try_into_unique().is_ok());
}

#[test]
fn try_into_unique_recovers_capacity_to_the_end_of_the_region() {
    let mut bytes = UniqueBytes::with_capacity(1024, Alignment::none());
    bytes.extend_from_slice(&[1, 2, 3, 4], Alignment::none());
    let unique = bytes
        .freeze()
        .try_into_unique()
        .expect("sole handle to the region");
    assert_eq!(unique.len(), 4);
    assert!(unique.capacity() >= 1024);
}

#[test]
fn split_off_windows_are_disjoint_and_rejoin_in_place() {
    let mut a = filled(64, Alignment::none());
    let ptr = a.as_ptr();
    let b = a.split_off(16);

    assert_eq!(a.len(), 16);
    assert_eq!(a.capacity(), 16);
    assert_eq!(b.len(), 48);
    assert_eq!(b.as_ptr(), unsafe { ptr.add(16) });

    // Neither half can grow in place while the other is alive.
    assert!(!a.freeze().is_unique());

    let mut a = filled(64, Alignment::none());
    let b = a.split_off(16);
    a.unsplit(b, Alignment::none());
    assert_eq!(a.len(), 64);
    assert_eq!(a.as_slice(), &pattern(64)[..]);
}

#[test]
fn unsplit_of_unrelated_windows_copies() {
    let mut a = filled(4, Alignment::none());
    let b = filled(4, Alignment::none());
    a.unsplit(b, Alignment::none());
    assert_eq!(a.as_slice(), &[0, 1, 2, 3, 0, 1, 2, 3]);
}

#[test]
fn reclaim_after_sibling_is_dropped() {
    let mut a = UniqueBytes::with_capacity(1024, Alignment::none());
    a.extend_from_slice(&[1, 2, 3, 4], Alignment::none());
    let b = a.split_off(4);
    let ptr = a.as_ptr();
    assert_eq!(a.capacity(), 4);

    drop(b);
    // The whole region is ours again, so growing must not move the data.
    a.reserve(500, Alignment::none());
    assert_eq!(a.as_ptr(), ptr);
    assert!(a.capacity() >= 504);
    assert_eq!(a.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn advance_gives_up_the_front_of_the_window() {
    let mut bytes = filled(16, Alignment::none());
    bytes.advance(4);
    assert_eq!(bytes.len(), 12);
    assert_eq!(bytes.as_slice()[0], 4);

    let mut shared = filled(16, Alignment::none()).freeze();
    shared.advance(4);
    assert_eq!(shared.len(), 12);
    assert_eq!(shared.as_slice()[0], 4);
}

#[test]
fn vec_round_trip_keeps_the_allocation() {
    let vec: Vec<u32> = (0..100).collect();
    let ptr = vec.as_ptr();

    let bytes = UniqueBytes::from_vec(vec);
    assert_eq!(bytes.as_ptr(), ptr.cast());
    assert_eq!(bytes.len(), 400);

    let vec = bytes
        .try_into_vec::<u32>()
        .expect("adopted from a Vec<u32>");
    assert_eq!(vec.as_ptr(), ptr);
    assert_eq!(vec.len(), 100);
    assert_eq!(vec[99], 99);
}

#[test]
fn vec_round_trip_survives_growth() {
    let vec: Vec<u32> = (0..100).collect();
    let mut bytes = UniqueBytes::from_vec(vec);
    bytes.extend_from_slice(&[0u8; 4096], Alignment::of::<u32>());

    let vec = bytes.try_into_vec::<u32>().expect("still a u32 allocation");
    assert_eq!(vec.len(), 1124);
    assert_eq!(vec[99], 99);
}

#[test]
fn vec_round_trip_rejects_a_mismatched_layout() {
    // Over-aligned: `Vec<u32>` would free it with an alignment of 4.
    let bytes = filled(64, Alignment::new(256));
    assert!(bytes.try_into_vec::<u32>().is_err());

    // Offset: `Vec` requires the pointer to be the start of the allocation.
    let mut bytes = filled(64, Alignment::of::<u32>());
    bytes.advance(4);
    assert!(bytes.try_into_vec::<u32>().is_err());
}

#[test]
fn owned_regions_release_their_owner() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    struct Counted {
        data: Vec<u8>,
        dropped: Arc<AtomicUsize>,
    }

    impl AsRef<[u8]> for Counted {
        fn as_ref(&self) -> &[u8] {
            &self.data
        }
    }

    impl Drop for Counted {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }

    let dropped = Arc::new(AtomicUsize::new(0));
    let owner = Counted {
        data: vec![1, 2, 3, 4],
        dropped: Arc::clone(&dropped),
    };

    let shared = SharedBytes::from_owner::<_, u8>(owner);
    assert_eq!(shared.as_slice(), &[1, 2, 3, 4]);
    let slice = shared.slice(1, 3);
    drop(shared);
    assert_eq!(dropped.load(Ordering::SeqCst), 0, "still referenced");

    assert_eq!(slice.as_slice(), &[2, 3]);
    // Read-only provenance: the owner only ever lent us a shared reference.
    assert!(slice.try_into_unique().is_err());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn static_regions_are_never_writable() {
    static VALUES: [u8; 4] = [1, 2, 3, 4];
    let shared = SharedBytes::from_static(&VALUES);
    assert_eq!(shared.as_ptr(), VALUES.as_ptr());
    assert!(shared.try_into_unique().is_err());
}
