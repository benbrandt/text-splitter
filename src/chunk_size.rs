use std::{
    cmp::Ordering,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

use ahash::AHashMap;

mod characters;
#[cfg(feature = "tokenizers")]
mod huggingface;
#[cfg(feature = "rust-tokenizers")]
mod rust_tokenizers;
#[cfg(feature = "tiktoken-rs")]
mod tiktoken;

pub use characters::Characters;
use thiserror::Error;

/// Indicates there was an error with the chunk capacity configuration.
/// The `Display` implementation will provide a human-readable error message to
/// help debug the issue that caused the error.
#[derive(Error, Debug)]
#[error(transparent)]
pub struct ChunkCapacityError(#[from] ChunkCapacityErrorRepr);

/// Private error and free to change across minor version of the crate.
#[derive(Error, Debug)]
enum ChunkCapacityErrorRepr {
    #[error("Max chunk size must be greater than or equal to the desired chunk size")]
    MaxLessThanDesired,
}

/// Describes the valid chunk size(s) that can be generated.
///
/// The `desired` size is the target size for the chunk. In most cases, this
/// will also serve as the maximum size of the chunk. It is always possible
/// that a chunk may be returned that is less than the `desired` value, as
/// adding the next piece of text may have made it larger than the `desired`
/// capacity.
///
/// The `max` size is the maximum possible chunk size that can be generated.
/// By setting this to a larger value than `desired`, it means that the chunk
/// should be as close to `desired` as possible, but can be larger if it means
/// staying at a larger semantic level.
///
/// The splitter will consume text until at maxumum somewhere between `desired`
/// and `max`, if they differ, but never above `max`.
///
/// If you need to ensure a fixed size, set `desired` and `max` to the same
/// value. For example, if you are trying to maximize the context window for an
/// embedding.
///
/// If you are loosely targeting a size, but have some extra room, for example
/// in a RAG use case where you roughly want a certain part of a document, you
/// can set `max` to your absolute maxumum, and the splitter can stay at a
/// higher semantic level when determining the chunk.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ChunkCapacity {
    desired: usize,
    max: usize,
}

impl ChunkCapacity {
    /// Create a new `ChunkCapacity` with the same `desired` and `max` size.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            desired: size,
            max: size,
        }
    }

    /// The `desired` size is the target size for the chunk. In most cases, this
    /// will also serve as the maximum size of the chunk. It is always possible
    /// that a chunk may be returned that is less than the `desired` value, as
    /// adding the next piece of text may have made it larger than the `desired`
    /// capacity.
    #[must_use]
    pub fn desired(&self) -> usize {
        self.desired
    }

    /// The `max` size is the maximum possible chunk size that can be generated.
    /// By setting this to a larger value than `desired`, it means that the chunk
    /// should be as close to `desired` as possible, but can be larger if it means
    /// staying at a larger semantic level.
    #[must_use]
    pub fn max(&self) -> usize {
        self.max
    }

    /// If you need to ensure a fixed size, set `desired` and `max` to the same
    /// value. For example, if you are trying to maximize the context window for an
    /// embedding.
    ///
    /// If you are loosely targeting a size, but have some extra room, for example
    /// in a RAG use case where you roughly want a certain part of a document, you
    /// can set `max` to your absolute maxumum, and the splitter can stay at a
    /// higher semantic level when determining the chunk.
    ///
    /// # Errors
    ///
    /// If the `max` size is less than the `desired` size, an error is returned.
    pub fn with_max(self, max: usize) -> Result<Self, ChunkCapacityError> {
        if max < self.desired {
            Err(ChunkCapacityError(
                ChunkCapacityErrorRepr::MaxLessThanDesired,
            ))
        } else {
            Ok(Self { max, ..self })
        }
    }

    /// Validate if a given chunk fits within the capacity
    ///
    /// - `Ordering::Less` indicates more could be added
    /// - `Ordering::Equal` indicates the chunk is within the capacity range
    /// - `Ordering::Greater` indicates the chunk is larger than the capacity
    #[must_use]
    pub fn fits(&self, chunk_size: usize) -> Ordering {
        if chunk_size < self.desired {
            Ordering::Less
        } else if chunk_size > self.max {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl From<usize> for ChunkCapacity {
    fn from(size: usize) -> Self {
        ChunkCapacity::new(size)
    }
}

impl From<Range<usize>> for ChunkCapacity {
    fn from(range: Range<usize>) -> Self {
        ChunkCapacity::new(range.start)
            .with_max(range.end.saturating_sub(1).max(range.start))
            .expect("invalid range")
    }
}

impl From<RangeFrom<usize>> for ChunkCapacity {
    fn from(range: RangeFrom<usize>) -> Self {
        ChunkCapacity::new(range.start)
            .with_max(usize::MAX)
            .expect("invalid range")
    }
}

impl From<RangeFull> for ChunkCapacity {
    fn from(_: RangeFull) -> Self {
        ChunkCapacity::new(usize::MIN)
            .with_max(usize::MAX)
            .expect("invalid range")
    }
}

impl From<RangeInclusive<usize>> for ChunkCapacity {
    fn from(range: RangeInclusive<usize>) -> Self {
        ChunkCapacity::new(*range.start())
            .with_max(*range.end())
            .expect("invalid range")
    }
}

impl From<RangeTo<usize>> for ChunkCapacity {
    fn from(range: RangeTo<usize>) -> Self {
        ChunkCapacity::new(usize::MIN)
            .with_max(range.end.saturating_sub(1))
            .expect("invalid range")
    }
}

impl From<RangeToInclusive<usize>> for ChunkCapacity {
    fn from(range: RangeToInclusive<usize>) -> Self {
        ChunkCapacity::new(usize::MIN)
            .with_max(range.end)
            .expect("invalid range")
    }
}

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
    #[must_use]
    pub fn from_size(size: usize, capacity: &ChunkCapacity) -> Self {
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
        capacity: &ChunkCapacity,
    ) -> Self {
        let mut chunk_size = offsets.fold(
            Self {
                fits: Ordering::Less,
                max_chunk_size_offset: None,
                size: 0,
            },
            |mut acc, range| {
                acc.size += 1;
                if acc.size <= capacity.max {
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
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize;
}

/// Configuration for how chunks should be created
#[derive(Debug)]
pub struct ChunkConfig<Sizer>
where
    Sizer: ChunkSizer,
{
    /// The chunk capacity to use for filling chunks
    capacity: ChunkCapacity,
    /// The chunk sizer to use for determining the size of each chunk
    sizer: Sizer,
    /// Whether whitespace will be trimmed from the beginning and end of each chunk
    trim: bool,
}

impl ChunkConfig<Characters> {
    /// Create a basic configuration for chunking with only the required value a chunk capacity.
    ///
    /// By default, chunk sizes will be calculated based on the number of characters in each chunk.
    /// You can set a custom chunk sizer by calling [`Self::with_sizer`].
    ///
    /// By default, chunks will be trimmed. If you want to preserve whitespace,
    /// call [`Self::with_trim`] and set it to `false`.
    #[must_use]
    pub fn new(capacity: impl Into<ChunkCapacity>) -> Self {
        Self {
            capacity: capacity.into(),
            sizer: Characters,
            trim: true,
        }
    }
}

impl<Sizer> ChunkConfig<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Retrieve a reference to the chunk capacity for this configuration.
    pub fn capacity(&self) -> &ChunkCapacity {
        &self.capacity
    }

    /// Retrieve a reference to the chunk sizer for this configuration.
    pub fn sizer(&self) -> &Sizer {
        &self.sizer
    }

    /// Set a custom chunk sizer to use for determining the size of each chunk
    ///
    /// ```
    /// use text_splitter::{Characters, ChunkConfig};
    ///
    /// let config = ChunkConfig::new(512).with_sizer(Characters);
    /// ```
    #[must_use]
    pub fn with_sizer<S: ChunkSizer>(self, sizer: S) -> ChunkConfig<S> {
        ChunkConfig {
            capacity: self.capacity,
            sizer,
            trim: self.trim,
        }
    }

    /// Whether chunkd should have whitespace trimmed from the beginning and end or not.
    pub fn trim(&self) -> bool {
        self.trim
    }

    /// Specify whether chunks should have whitespace trimmed from the
    /// beginning and end or not.
    ///
    /// If `false` (default), joining all chunks should return the original
    /// string.
    /// If `true`, all chunks will have whitespace removed from beginning and end.
    ///
    /// ```
    /// use text_splitter::ChunkConfig;
    ///
    /// let config = ChunkConfig::new(512).with_trim(false);
    /// ```
    #[must_use]
    pub fn with_trim(mut self, trim: bool) -> Self {
        self.trim = trim;
        self
    }
}

impl<T> From<T> for ChunkConfig<Characters>
where
    T: Into<ChunkCapacity>,
{
    fn from(capacity: T) -> Self {
        Self::new(capacity)
    }
}

/// A memoized chunk sizer that caches the size of chunks.
/// Very helpful when the same chunk is being validated multiple times, which
/// happens often, and can be expensive to compute, such as with tokenizers.
#[derive(Debug)]
pub struct MemoizedChunkSizer<'sizer, Sizer>
where
    Sizer: ChunkSizer,
{
    /// Cache of chunk sizes per byte offset range
    cache: AHashMap<Range<usize>, ChunkSize>,
    /// How big can each chunk be
    chunk_capacity: &'sizer ChunkCapacity,
    /// The sizer we are wrapping
    sizer: &'sizer Sizer,
}

impl<'sizer, Sizer> MemoizedChunkSizer<'sizer, Sizer>
where
    Sizer: ChunkSizer,
{
    /// Wrap any chunk sizer for memoization
    pub fn new(chunk_capacity: &'sizer ChunkCapacity, sizer: &'sizer Sizer) -> Self {
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
            .or_insert_with(|| self.sizer.chunk_size(chunk, self.chunk_capacity))
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{self, AtomicUsize};

    use super::*;

    #[test]
    fn check_chunk_capacity() {
        let chunk = "12345";

        assert_eq!(
            Characters.chunk_size(chunk, &4.into()).fits,
            Ordering::Greater
        );
        assert_eq!(
            Characters.chunk_size(chunk, &5.into()).fits,
            Ordering::Equal
        );
        assert_eq!(Characters.chunk_size(chunk, &6.into()).fits, Ordering::Less);
    }

    #[test]
    fn check_chunk_capacity_for_range() {
        let chunk = "12345";

        assert_eq!(
            Characters.chunk_size(chunk, &(0..0).into()).fits,
            Ordering::Greater
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(0..5).into()).fits,
            Ordering::Greater
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(5..6).into()).fits,
            Ordering::Equal
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(6..100).into()).fits,
            Ordering::Less
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_from() {
        let chunk = "12345";

        assert_eq!(
            Characters.chunk_size(chunk, &(0..).into()).fits,
            Ordering::Equal
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(5..).into()).fits,
            Ordering::Equal
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(6..).into()).fits,
            Ordering::Less
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_full() {
        let chunk = "12345";

        assert_eq!(
            Characters.chunk_size(chunk, &(..).into()).fits,
            Ordering::Equal
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_inclusive() {
        let chunk = "12345";

        assert_eq!(
            Characters.chunk_size(chunk, &(0..=4).into()).fits,
            Ordering::Greater
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(5..=6).into()).fits,
            Ordering::Equal
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(4..=5).into()).fits,
            Ordering::Equal
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(6..=100).into()).fits,
            Ordering::Less
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_to() {
        let chunk = "12345";

        assert_eq!(
            Characters.chunk_size(chunk, &(..0).into()).fits,
            Ordering::Greater
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(..5).into()).fits,
            Ordering::Greater
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(..6).into()).fits,
            Ordering::Equal
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_to_inclusive() {
        let chunk = "12345";

        assert_eq!(
            Characters.chunk_size(chunk, &(..=4).into()).fits,
            Ordering::Greater
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(..=5).into()).fits,
            Ordering::Equal
        );
        assert_eq!(
            Characters.chunk_size(chunk, &(..=6).into()).fits,
            Ordering::Equal
        );
    }

    #[test]
    fn chunk_size_from_offsets() {
        let offsets = [0..1, 1..2, 2..3];
        let chunk_size = ChunkSize::from_offsets(offsets.clone().into_iter(), &1.into());
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
        let chunk_size = ChunkSize::from_offsets(offsets.clone().into_iter(), &1.into());
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
        let chunk_size = ChunkSize::from_offsets(offsets.clone().into_iter(), &4.into());
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
        fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
            self.calls.fetch_add(1, atomic::Ordering::SeqCst);
            Characters.chunk_size(chunk, capacity)
        }
    }

    #[test]
    fn memoized_sizer_only_calculates_once_per_text() {
        let sizer = CountingSizer::default();
        let capacity = ChunkCapacity::new(10);
        let mut memoized_sizer = MemoizedChunkSizer::new(&capacity, &sizer);
        let text = "1234567890";
        for _ in 0..10 {
            memoized_sizer.chunk_size(0, text);
        }

        assert_eq!(memoized_sizer.sizer.calls.load(atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn memoized_sizer_calculates_once_per_different_text() {
        let sizer = CountingSizer::default();
        let capacity = ChunkCapacity::new(10);
        let mut memoized_sizer = MemoizedChunkSizer::new(&capacity, &sizer);
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
        let capacity = ChunkCapacity::new(10);
        let mut memoized_sizer = MemoizedChunkSizer::new(&capacity, &sizer);
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
        let chunk_size = ChunkSize::from_size(10, &10.into());
        assert_eq!(
            ChunkSize {
                fits: Ordering::Equal,
                size: 10,
                max_chunk_size_offset: None
            },
            chunk_size
        );
    }

    #[test]
    fn basic_chunk_config() {
        let config = ChunkConfig::new(10);
        assert_eq!(config.capacity, 10.into());
        assert_eq!(config.sizer, Characters);
        assert!(config.trim);
    }

    #[test]
    fn disable_trimming() {
        let config = ChunkConfig::new(10).with_trim(false);
        assert!(!config.trim);
    }

    #[test]
    fn new_sizer() {
        #[derive(Debug, PartialEq)]
        struct BasicSizer;

        impl ChunkSizer for BasicSizer {
            fn chunk_size(&self, _chunk: &str, _capacity: &ChunkCapacity) -> ChunkSize {
                unimplemented!()
            }
        }

        let config = ChunkConfig::new(10).with_sizer(BasicSizer);
        assert_eq!(config.capacity, 10.into());
        assert_eq!(config.sizer, BasicSizer);
        assert!(config.trim);
    }

    #[test]
    fn chunk_capacity_max_and_desired_equal() {
        let capacity = ChunkCapacity::new(10);
        assert_eq!(capacity.desired(), 10);
        assert_eq!(capacity.max(), 10);
    }

    #[test]
    fn chunk_capacity_can_adjust_max() {
        let capacity = ChunkCapacity::new(10).with_max(20).unwrap();
        assert_eq!(capacity.desired(), 10);
        assert_eq!(capacity.max(), 20);
    }

    #[test]
    fn chunk_capacity_max_cant_be_less_than_desired() {
        let capacity = ChunkCapacity::new(10);
        let err = capacity.with_max(5).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Max chunk size must be greater than or equal to the desired chunk size"
        );
        assert_eq!(capacity.desired(), 10);
        assert_eq!(capacity.max(), 10);
    }
}
