use std::{
    cmp::Ordering,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

use ahash::AHashMap;

mod characters;
#[cfg(feature = "tokenizers")]
mod huggingface;
#[cfg(feature = "tiktoken-rs")]
mod tiktoken;

pub use characters::Characters;

/// Result returned from a `ChunkSizer`. Includes the size of the chunk, in units
/// determined by the sizer, as well as the max byte offset of the text that
/// would fit within the given `ChunkCapacity`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ChunkSize {
    /// Whether or not the entire chunk fits within the `ChunkCapacity`
    fits: Ordering,
    /// max byte offset of the text that fit within the given `ChunkCapacity`.
    max_chunk_size_offset: Option<usize>,
    /// Size of the chunk, in units used by the sizer.
    size: usize,
}

impl ChunkSize {
    /// Generate a chunk size from a given size. Will not be able to compute the
    /// max byte offset that fits within the capacity.
    pub fn from_size(size: usize, capacity: &impl ChunkCapacity) -> Self {
        Self {
            fits: capacity.fits(size),
            max_chunk_size_offset: None,
            size,
        }
    }

    /// Generate a chunk size from an iterator of byte ranges for each encoded
    /// element in the chunk.
    pub fn from_offsets(
        offsets: impl Iterator<Item = Range<usize>>,
        capacity: &impl ChunkCapacity,
    ) -> Self {
        let mut chunk_size = offsets.fold(
            Self {
                fits: Ordering::Less,
                max_chunk_size_offset: None,
                size: 0,
            },
            |mut acc, range| {
                acc.size += 1;
                if acc.size <= capacity.end() {
                    acc.max_chunk_size_offset = Some(range.end);
                }
                acc
            },
        );
        chunk_size.fits = capacity.fits(chunk_size.size);
        chunk_size
    }

    /// Determine whether the chunk size fits within the capacity or not
    #[must_use]
    pub fn fits(&self) -> Ordering {
        self.fits
    }

    /// max byte offset of the text that fit within the given `ChunkCapacity`.
    #[must_use]
    pub fn max_chunk_size_offset(&self) -> Option<usize> {
        self.max_chunk_size_offset
    }

    /// Size of the chunk, in units used by the sizer.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Determines the size of a given chunk.
pub trait ChunkSizer {
    /// Determine the size of a given chunk to use for validation
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize;
}

/// A memoized chunk sizer that caches the size of chunks.
/// Very helpful when the same chunk is being validated multiple times, which
/// happens often, and can be expensive to compute, such as with tokenizers.
#[derive(Debug)]
pub struct MemoizedChunkSizer<'sizer, C, S>
where
    C: ChunkCapacity,
    S: ChunkSizer,
{
    /// Cache of chunk sizes per byte offset range
    cache: AHashMap<Range<usize>, ChunkSize>,
    /// How big can each chunk be
    chunk_capacity: C,
    /// The sizer we are wrapping
    sizer: &'sizer S,
}

impl<'sizer, C, S> MemoizedChunkSizer<'sizer, C, S>
where
    C: ChunkCapacity,
    S: ChunkSizer,
{
    /// Wrap any chunk sizer for memoization
    pub fn new(chunk_capacity: C, sizer: &'sizer S) -> Self {
        Self {
            cache: AHashMap::new(),
            chunk_capacity,
            sizer,
        }
    }

    /// Determine the size of a given chunk to use for validation,
    /// returning a cached value if it exists, and storing the result if not.
    pub fn chunk_size(&mut self, offset: usize, chunk: &str) -> ChunkSize {
        *self
            .cache
            .entry(offset..(offset + chunk.len()))
            .or_insert_with(|| self.sizer.chunk_size(chunk, &self.chunk_capacity))
    }

    /// Check if the chunk is within the capacity. Chunk should be trimmed if necessary beforehand.
    pub fn check_capacity(&mut self, (offset, chunk): (usize, &str)) -> ChunkSize {
        let mut chunk_size = self.chunk_size(offset, chunk);
        if let Some(max_chunk_size_offset) = chunk_size.max_chunk_size_offset.as_mut() {
            *max_chunk_size_offset += offset;
        }
        chunk_size
    }

    /// Clear the cached values. Once we've moved the cursor,
    /// we don't need to keep the old values around.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Describes the largest valid chunk size(s) that can be generated.
///
/// An `end` size is required, which is the maximum possible chunk size that
/// can be generated.
///
/// A `start` size is optional. By specifying `start` and `end` it means a
/// range of sizes will be considered valid. Once a chunk has reached a length
/// that falls between `start` and `end` it will be returned.
///
/// It is always possible that a chunk may be returned that is less than the
/// `start` value, as adding the next piece of text may have made it larger
/// than the `end` capacity.
pub trait ChunkCapacity {
    /// An optional `start` value. If both `start` and `end` are specified, a
    /// valid chunk can fall anywhere between the two values (inclusive).
    fn start(&self) -> Option<usize> {
        None
    }

    /// The maximum size that a chunk can be.
    #[must_use]
    fn end(&self) -> usize;

    /// Validate if a given chunk fits within the capacity
    ///
    /// - `Ordering::Less` indicates more could be added
    /// - `Ordering::Equal` indicates the chunk is within the capacity range
    /// - `Ordering::Greater` indicates the chunk is larger than the capacity
    fn fits(&self, chunk_size: usize) -> Ordering {
        let end = self.end();

        match self.start() {
            Some(start) => {
                if chunk_size < start {
                    Ordering::Less
                } else if chunk_size > end {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }
            None => chunk_size.cmp(&end),
        }
    }
}

impl ChunkCapacity for usize {
    fn end(&self) -> usize {
        *self
    }
}

impl ChunkCapacity for Range<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> usize {
        self.end.saturating_sub(1).max(self.start)
    }
}

impl ChunkCapacity for RangeFrom<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> usize {
        usize::MAX
    }
}

impl ChunkCapacity for RangeFull {
    fn start(&self) -> Option<usize> {
        Some(usize::MIN)
    }

    fn end(&self) -> usize {
        usize::MAX
    }
}

impl ChunkCapacity for RangeInclusive<usize> {
    fn start(&self) -> Option<usize> {
        Some(*self.start())
    }

    fn end(&self) -> usize {
        *self.end()
    }
}

impl ChunkCapacity for RangeTo<usize> {
    fn start(&self) -> Option<usize> {
        Some(usize::MIN)
    }

    fn end(&self) -> usize {
        self.end.saturating_sub(1)
    }
}

impl ChunkCapacity for RangeToInclusive<usize> {
    fn start(&self) -> Option<usize> {
        Some(usize::MIN)
    }

    fn end(&self) -> usize {
        self.end
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{self, AtomicUsize};

    use super::*;

    #[test]
    fn check_chunk_capacity() {
        let chunk = "12345";

        assert_eq!(Characters.chunk_size(chunk, &4).fits, Ordering::Greater);
        assert_eq!(Characters.chunk_size(chunk, &5).fits, Ordering::Equal);
        assert_eq!(Characters.chunk_size(chunk, &6).fits, Ordering::Less);
    }

    #[test]
    fn check_chunk_capacity_for_range() {
        let chunk = "12345";

        assert_eq!(
            Characters.chunk_size(chunk, &(0..0)).fits,
            Ordering::Greater
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(0..5)).fits,
            Ordering::Greater
        );
        assert_eq!(Characters.chunk_size(chunk, &(5..6)).fits, Ordering::Equal);
        assert_eq!(Characters.chunk_size(chunk, &(6..100)).fits, Ordering::Less);
    }

    #[test]
    fn check_chunk_capacity_for_range_from() {
        let chunk = "12345";

        assert_eq!(Characters.chunk_size(chunk, &(0..)).fits, Ordering::Equal);
        assert_eq!(Characters.chunk_size(chunk, &(5..)).fits, Ordering::Equal);
        assert_eq!(Characters.chunk_size(chunk, &(6..)).fits, Ordering::Less);
    }

    #[test]
    fn check_chunk_capacity_for_range_full() {
        let chunk = "12345";

        assert_eq!(Characters.chunk_size(chunk, &..).fits, Ordering::Equal);
    }

    #[test]
    fn check_chunk_capacity_for_range_inclusive() {
        let chunk = "12345";

        assert_eq!(
            Characters.chunk_size(chunk, &(0..=4)).fits,
            Ordering::Greater
        );
        assert_eq!(Characters.chunk_size(chunk, &(5..=6)).fits, Ordering::Equal);
        assert_eq!(Characters.chunk_size(chunk, &(4..=5)).fits, Ordering::Equal);
        assert_eq!(
            Characters.chunk_size(chunk, &(6..=100)).fits,
            Ordering::Less
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_to() {
        let chunk = "12345";

        assert_eq!(Characters.chunk_size(chunk, &(..0)).fits, Ordering::Greater);
        assert_eq!(Characters.chunk_size(chunk, &(..5)).fits, Ordering::Greater);
        assert_eq!(Characters.chunk_size(chunk, &(..6)).fits, Ordering::Equal);
    }

    #[test]
    fn check_chunk_capacity_for_range_to_inclusive() {
        let chunk = "12345";

        assert_eq!(
            Characters.chunk_size(chunk, &(..=4)).fits,
            Ordering::Greater
        );
        assert_eq!(Characters.chunk_size(chunk, &(..=5)).fits, Ordering::Equal);
        assert_eq!(Characters.chunk_size(chunk, &(..=6)).fits, Ordering::Equal);
    }

    #[test]
    fn chunk_size_from_offsets() {
        let offsets = [0..1, 1..2, 2..3];
        let chunk_size = ChunkSize::from_offsets(offsets.clone().into_iter(), &1);
        assert_eq!(
            ChunkSize {
                fits: Ordering::Greater,
                size: offsets.len(),
                max_chunk_size_offset: Some(1)
            },
            chunk_size
        );
    }

    #[test]
    fn chunk_size_from_empty_offsets() {
        let offsets = [];
        let chunk_size = ChunkSize::from_offsets(offsets.clone().into_iter(), &1);
        assert_eq!(
            ChunkSize {
                fits: Ordering::Less,
                size: offsets.len(),
                max_chunk_size_offset: None
            },
            chunk_size
        );
    }

    #[test]
    fn chunk_size_from_small_offsets() {
        let offsets = [0..1, 1..2, 2..3];
        let chunk_size = ChunkSize::from_offsets(offsets.clone().into_iter(), &4);
        assert_eq!(
            ChunkSize {
                fits: Ordering::Less,
                size: offsets.len(),
                max_chunk_size_offset: Some(3)
            },
            chunk_size
        );
    }

    #[derive(Default)]
    struct CountingSizer {
        calls: AtomicUsize,
    }

    impl ChunkSizer for CountingSizer {
        // Return character version, but count calls
        fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
            self.calls.fetch_add(1, atomic::Ordering::SeqCst);
            Characters.chunk_size(chunk, capacity)
        }
    }

    #[test]
    fn memoized_sizer_only_calculates_once_per_text() {
        let sizer = CountingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(10, &sizer);
        let text = "1234567890";
        for _ in 0..10 {
            memoized_sizer.chunk_size(0, text);
        }

        assert_eq!(memoized_sizer.sizer.calls.load(atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn memoized_sizer_calculates_once_per_different_text() {
        let sizer = CountingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(10, &sizer);
        let text = "1234567890";
        for i in 0..10 {
            memoized_sizer.chunk_size(0, text.get(0..i).unwrap());
        }

        assert_eq!(
            memoized_sizer.sizer.calls.load(atomic::Ordering::SeqCst),
            10
        );
    }

    #[test]
    fn can_clear_cache_on_memoized_sizer() {
        let sizer = CountingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(10, &sizer);
        let text = "1234567890";
        for _ in 0..10 {
            memoized_sizer.chunk_size(0, text);
            memoized_sizer.clear_cache();
        }

        assert_eq!(
            memoized_sizer.sizer.calls.load(atomic::Ordering::SeqCst),
            10
        );
    }

    #[test]
    fn test_chunk_size_from_size() {
        let chunk_size = ChunkSize::from_size(10, &10);
        assert_eq!(
            ChunkSize {
                fits: Ordering::Equal,
                size: 10,
                max_chunk_size_offset: None
            },
            chunk_size
        );
    }
}
