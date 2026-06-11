use std::{
    borrow::Cow,
    cell::{Ref, RefMut},
    cmp::Ordering,
    fmt,
    ops::{Deref, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
    rc::Rc,
    sync::Arc,
};

use ahash::AHashMap;
use itertools::Itertools;
use thiserror::Error;

mod characters;
#[cfg(feature = "tokenizers")]
mod huggingface;
#[cfg(feature = "tiktoken-rs")]
mod tiktoken;

use crate::trim::Trim;
pub use characters::Characters;

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
    pub(crate) desired: usize,
    pub(crate) max: usize,
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
    pub fn with_max(mut self, max: usize) -> Result<Self, ChunkCapacityError> {
        if max < self.desired {
            Err(ChunkCapacityError(
                ChunkCapacityErrorRepr::MaxLessThanDesired,
            ))
        } else {
            self.max = max;
            Ok(self)
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

/// Determines the size of a given chunk.
pub trait ChunkSizer {
    /// Determine the size of a given chunk to use for validation
    fn size(&self, chunk: &str) -> usize;
}

impl<T> ChunkSizer for &T
where
    T: ChunkSizer,
{
    fn size(&self, chunk: &str) -> usize {
        (*self).size(chunk)
    }
}

impl<T> ChunkSizer for Ref<'_, T>
where
    T: ChunkSizer,
{
    fn size(&self, chunk: &str) -> usize {
        self.deref().size(chunk)
    }
}

impl<T> ChunkSizer for RefMut<'_, T>
where
    T: ChunkSizer,
{
    fn size(&self, chunk: &str) -> usize {
        self.deref().size(chunk)
    }
}

impl<T> ChunkSizer for Box<T>
where
    T: ChunkSizer,
{
    fn size(&self, chunk: &str) -> usize {
        self.deref().size(chunk)
    }
}

impl<T> ChunkSizer for Cow<'_, T>
where
    T: ChunkSizer + ToOwned + ?Sized,
    <T as ToOwned>::Owned: ChunkSizer,
{
    fn size(&self, chunk: &str) -> usize {
        self.as_ref().size(chunk)
    }
}

impl<T> ChunkSizer for Rc<T>
where
    T: ChunkSizer,
{
    fn size(&self, chunk: &str) -> usize {
        self.deref().size(chunk)
    }
}

impl<T> ChunkSizer for Arc<T>
where
    T: ChunkSizer,
{
    fn size(&self, chunk: &str) -> usize {
        self.as_ref().size(chunk)
    }
}

/// Indicates there was an error with the chunk configuration.
/// The `Display` implementation will provide a human-readable error message to
/// help debug the issue that caused the error.
#[derive(Error, Debug)]
#[error(transparent)]
pub struct ChunkConfigError(#[from] ChunkConfigErrorRepr);

/// Private error and free to change across minor version of the crate.
#[derive(Error, Debug)]
enum ChunkConfigErrorRepr {
    #[error("The overlap is larger than or equal to the desired chunk capacity")]
    OverlapLargerThanCapacity,
}

/// Configuration for how chunks should be created
#[derive(Debug)]
pub struct ChunkConfig<Sizer>
where
    Sizer: ChunkSizer,
{
    /// The chunk capacity to use for filling chunks
    pub(crate) capacity: ChunkCapacity,
    /// The amount of overlap between chunks. Defaults to 0.
    pub(crate) overlap: usize,
    /// The chunk sizer to use for determining the size of each chunk
    pub(crate) sizer: Sizer,
    /// Whether whitespace will be trimmed from the beginning and end of each chunk
    pub(crate) trim: bool,
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
            overlap: 0,
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

    /// Retrieve the amount of overlap between chunks.
    pub fn overlap(&self) -> usize {
        self.overlap
    }

    /// Set the amount of overlap between chunks.
    ///
    /// # Errors
    ///
    /// Will return an error if the overlap is larger than or equal to the chunk capacity.
    pub fn with_overlap(mut self, overlap: usize) -> Result<Self, ChunkConfigError> {
        if overlap >= self.capacity.desired {
            Err(ChunkConfigError(
                ChunkConfigErrorRepr::OverlapLargerThanCapacity,
            ))
        } else {
            self.overlap = overlap;
            Ok(self)
        }
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
            overlap: self.overlap,
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

/// Conservative upper bound of bytes per size unit, used to bound the prefix probe in
/// [`MemoizedChunkSizer::fits_with_early_exit`]. Only chunks larger than twice
/// `capacity.max * EARLY_EXIT_PROBE_FACTOR` bytes are checked via a prefix probe first.
const EARLY_EXIT_PROBE_FACTOR: usize = 32;

/// A memoized chunk sizer that caches the size of chunks.
/// Very helpful when the same chunk is being validated multiple times, which
/// happens often, and can be expensive to compute, such as with tokenizers.
#[derive(Debug)]
pub struct MemoizedChunkSizer<'sizer, Sizer>
where
    Sizer: ChunkSizer,
{
    /// Cache of chunk sizes per byte offset range for base capacity
    size_cache: AHashMap<Range<usize>, usize>,
    /// The sizer used for caluclating chunk sizes
    sizer: &'sizer Sizer,
}

impl<'sizer, Sizer> MemoizedChunkSizer<'sizer, Sizer>
where
    Sizer: ChunkSizer,
{
    /// Wrap any chunk sizer for memoization
    pub fn new(sizer: &'sizer Sizer) -> Self {
        Self {
            size_cache: AHashMap::new(),
            sizer,
        }
    }

    /// Determine the size of a given chunk to use for validation,
    /// returning a cached value if it exists, and storing the result if not.
    pub fn chunk_size(&mut self, offset: usize, chunk: &str, trim: Trim) -> usize {
        let (offset, chunk) = trim.trim(offset, chunk);
        *self
            .size_cache
            .entry(offset..(offset + chunk.len()))
            .or_insert_with(|| self.sizer.size(chunk))
    }

    /// Determine how a given chunk fits within the capacity, sizing only a prefix of very
    /// large chunks when that already proves the chunk is too big.
    ///
    /// The prefix always ends at an ASCII whitespace boundary, so for sizers where a
    /// whitespace-delimited prefix never sizes larger than the whole chunk (such as
    /// [`Characters`] and whitespace-pretokenizing tokenizers) the classification is
    /// identical to sizing the entire chunk. If no whitespace cut point exists, or the
    /// prefix alone doesn't already exceed the capacity, the entire chunk is sized as
    /// usual.
    ///
    /// This keeps the cost of checking oversized semantic levels proportional to the
    /// chunk capacity instead of to the length of the remaining document.
    pub fn fits_with_early_exit(
        &mut self,
        offset: usize,
        chunk: &str,
        capacity: &ChunkCapacity,
        trim: Trim,
    ) -> Ordering {
        let probe_len = capacity.max.saturating_mul(EARLY_EXIT_PROBE_FACTOR);
        if chunk.len() > probe_len.saturating_mul(2) {
            let mut window_end = probe_len;
            while !chunk.is_char_boundary(window_end) {
                window_end -= 1;
            }
            if let Some(cut) = chunk[..window_end].rfind([' ', '\n', '\t', '\r']) {
                if cut > 0 {
                    let prefix_size = self.chunk_size(offset, &chunk[..cut], trim);
                    if capacity.fits(prefix_size).is_gt() {
                        return Ordering::Greater;
                    }
                }
            }
        }
        capacity.fits(self.chunk_size(offset, chunk, trim))
    }

    /// Find the best level to start splitting the text
    pub fn find_correct_level<'text, L: fmt::Debug>(
        &mut self,
        offset: usize,
        capacity: &ChunkCapacity,
        levels_with_first_chunk: impl Iterator<Item = (L, &'text str)>,
        trim: Trim,
    ) -> (Option<L>, Option<usize>) {
        let mut semantic_level = None;
        let mut max_offset = None;

        // We assume that larger levels are also longer. We can skip lower levels if going to a higher level would result in a shorter text
        let levels_with_first_chunk =
            levels_with_first_chunk.coalesce(|(a_level, a_str), (b_level, b_str)| {
                if a_str.len() >= b_str.len() {
                    Ok((b_level, b_str))
                } else {
                    Err(((a_level, a_str), (b_level, b_str)))
                }
            });

        for (level, str) in levels_with_first_chunk {
            // Skip tokenizing levels that we know are too small anyway.
            let len = str.len();
            if len > capacity.max {
                let fits = self.fits_with_early_exit(offset, str, capacity, trim);
                // If this no longer fits, we use the level we are at.
                if fits.is_gt() {
                    max_offset = Some(offset + len);
                    break;
                }
            }
            // Otherwise break up the text with the next level
            semantic_level = Some(level);
        }

        (semantic_level, max_offset)
    }

    /// Clear the cached values. Once we've moved the cursor,
    /// we don't need to keep the old values around.
    pub fn clear_cache(&mut self) {
        self.size_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        sync::atomic::{self, AtomicUsize},
    };

    use crate::trim::Trim;

    use super::*;

    #[test]
    fn check_chunk_capacity() {
        let chunk = "12345";

        assert_eq!(
            ChunkCapacity::from(4).fits(Characters.size(chunk)),
            Ordering::Greater
        );
        assert_eq!(
            ChunkCapacity::from(5).fits(Characters.size(chunk)),
            Ordering::Equal
        );
        assert_eq!(
            ChunkCapacity::from(6).fits(Characters.size(chunk)),
            Ordering::Less
        );
    }

    #[test]
    fn check_chunk_capacity_for_range() {
        let chunk = "12345";

        assert_eq!(
            ChunkCapacity::from(0..0).fits(Characters.size(chunk)),
            Ordering::Greater
        );
        assert_eq!(
            ChunkCapacity::from(0..5).fits(Characters.size(chunk)),
            Ordering::Greater
        );
        assert_eq!(
            ChunkCapacity::from(5..6).fits(Characters.size(chunk)),
            Ordering::Equal
        );
        assert_eq!(
            ChunkCapacity::from(6..100).fits(Characters.size(chunk)),
            Ordering::Less
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_from() {
        let chunk = "12345";

        assert_eq!(
            ChunkCapacity::from(0..).fits(Characters.size(chunk)),
            Ordering::Equal
        );
        assert_eq!(
            ChunkCapacity::from(5..).fits(Characters.size(chunk)),
            Ordering::Equal
        );
        assert_eq!(
            ChunkCapacity::from(6..).fits(Characters.size(chunk)),
            Ordering::Less
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_full() {
        let chunk = "12345";

        assert_eq!(
            ChunkCapacity::from(..).fits(Characters.size(chunk)),
            Ordering::Equal
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_inclusive() {
        let chunk = "12345";

        assert_eq!(
            ChunkCapacity::from(0..=4).fits(Characters.size(chunk)),
            Ordering::Greater
        );
        assert_eq!(
            ChunkCapacity::from(5..=6).fits(Characters.size(chunk)),
            Ordering::Equal
        );
        assert_eq!(
            ChunkCapacity::from(4..=5).fits(Characters.size(chunk)),
            Ordering::Equal
        );
        assert_eq!(
            ChunkCapacity::from(6..=100).fits(Characters.size(chunk)),
            Ordering::Less
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_to() {
        let chunk = "12345";

        assert_eq!(
            ChunkCapacity::from(..0).fits(Characters.size(chunk)),
            Ordering::Greater
        );
        assert_eq!(
            ChunkCapacity::from(..5).fits(Characters.size(chunk)),
            Ordering::Greater
        );
        assert_eq!(
            ChunkCapacity::from(..6).fits(Characters.size(chunk)),
            Ordering::Equal
        );
    }

    #[test]
    fn check_chunk_capacity_for_range_to_inclusive() {
        let chunk = "12345";

        assert_eq!(
            ChunkCapacity::from(..=4).fits(Characters.size(chunk)),
            Ordering::Greater
        );
        assert_eq!(
            ChunkCapacity::from(..=5).fits(Characters.size(chunk)),
            Ordering::Equal
        );
        assert_eq!(
            ChunkCapacity::from(..=6).fits(Characters.size(chunk)),
            Ordering::Equal
        );
    }

    #[derive(Default)]
    struct CountingSizer {
        calls: AtomicUsize,
    }

    impl ChunkSizer for CountingSizer {
        // Return character version, but count calls
        fn size(&self, chunk: &str) -> usize {
            self.calls.fetch_add(1, atomic::Ordering::SeqCst);
            Characters.size(chunk)
        }
    }

    #[test]
    fn memoized_sizer_only_calculates_once_per_text() {
        let sizer = CountingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let text = "1234567890";
        for _ in 0..10 {
            memoized_sizer.chunk_size(0, text, Trim::All);
        }

        assert_eq!(memoized_sizer.sizer.calls.load(atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn memoized_sizer_calculates_once_per_different_text() {
        let sizer = CountingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let text = "1234567890";
        for i in 0..10 {
            memoized_sizer.chunk_size(0, text.get(0..i).unwrap(), Trim::All);
        }

        assert_eq!(
            memoized_sizer.sizer.calls.load(atomic::Ordering::SeqCst),
            10
        );
    }

    #[test]
    fn can_clear_cache_on_memoized_sizer() {
        let sizer = CountingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let text = "1234567890";
        for _ in 0..10 {
            memoized_sizer.chunk_size(0, text, Trim::All);
            memoized_sizer.clear_cache();
        }

        assert_eq!(
            memoized_sizer.sizer.calls.load(atomic::Ordering::SeqCst),
            10
        );
    }

    #[derive(Default)]
    struct RecordingSizer {
        chunk_lens: RefCell<Vec<usize>>,
    }

    impl ChunkSizer for RecordingSizer {
        // Character count, recording the length of every chunk it is asked to size
        fn size(&self, chunk: &str) -> usize {
            self.chunk_lens.borrow_mut().push(chunk.len());
            Characters.size(chunk)
        }
    }

    #[test]
    fn early_exit_skips_sizing_full_oversized_chunk() {
        let sizer = RecordingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let chunk = "word ".repeat(10_000);
        let capacity = ChunkCapacity::new(10);

        let fits = memoized_sizer.fits_with_early_exit(0, &chunk, &capacity, Trim::None);

        assert_eq!(fits, Ordering::Greater);
        let max_len = sizer.chunk_lens.borrow().iter().copied().max().unwrap();
        assert!(max_len <= capacity.max() * EARLY_EXIT_PROBE_FACTOR);
    }

    #[test]
    fn early_exit_inconclusive_probe_sizes_full_chunk() {
        #[derive(Default)]
        struct CountXs(RefCell<Vec<usize>>);

        impl ChunkSizer for CountXs {
            fn size(&self, chunk: &str) -> usize {
                self.0.borrow_mut().push(chunk.len());
                chunk.matches('X').count()
            }
        }

        let sizer = CountXs::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let chunk = format!("{}{}", "a ".repeat(5_000), "X ".repeat(50));
        let capacity = ChunkCapacity::new(10);

        let fits = memoized_sizer.fits_with_early_exit(0, &chunk, &capacity, Trim::None);

        // The prefix probe didn't already exceed the capacity, so the entire chunk must
        // have been sized for an exact result.
        assert_eq!(fits, Ordering::Greater);
        assert!(sizer.0.borrow().contains(&chunk.len()));
    }

    #[test]
    fn early_exit_disengaged_for_chunks_near_capacity() {
        let sizer = RecordingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let chunk = "word ".repeat(10);
        let capacity = ChunkCapacity::new(10);

        let fits = memoized_sizer.fits_with_early_exit(0, &chunk, &capacity, Trim::None);

        assert_eq!(fits, Ordering::Greater);
        assert_eq!(sizer.chunk_lens.borrow().as_slice(), &[chunk.len()]);
    }

    #[test]
    fn early_exit_disengaged_for_unbounded_capacity() {
        let sizer = RecordingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let chunk = "word ".repeat(10_000);
        let capacity = ChunkCapacity::from(10..);

        let fits = memoized_sizer.fits_with_early_exit(0, &chunk, &capacity, Trim::None);

        assert_eq!(fits, Ordering::Equal);
        assert_eq!(sizer.chunk_lens.borrow().as_slice(), &[chunk.len()]);
    }

    #[test]
    fn early_exit_without_whitespace_sizes_full_chunk() {
        let sizer = RecordingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let chunk = "x".repeat(50_000);
        let capacity = ChunkCapacity::new(10);

        let fits = memoized_sizer.fits_with_early_exit(0, &chunk, &capacity, Trim::None);

        assert_eq!(fits, Ordering::Greater);
        assert_eq!(sizer.chunk_lens.borrow().as_slice(), &[chunk.len()]);
    }

    #[test]
    fn early_exit_probe_respects_char_boundaries() {
        let sizer = RecordingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        // Multi-byte chars so the probe window cannot end on the initial byte offset
        let chunk = "日".repeat(20_000);
        let capacity = ChunkCapacity::new(10);

        let fits = memoized_sizer.fits_with_early_exit(0, &chunk, &capacity, Trim::None);

        assert_eq!(fits, Ordering::Greater);
        assert_eq!(sizer.chunk_lens.borrow().as_slice(), &[chunk.len()]);
    }

    #[test]
    fn basic_chunk_config() {
        let config = ChunkConfig::new(10);
        assert_eq!(config.capacity, 10.into());
        assert_eq!(config.sizer, Characters);
        assert!(config.trim());
    }

    #[test]
    fn disable_trimming() {
        let config = ChunkConfig::new(10).with_trim(false);
        assert!(!config.trim());
    }

    #[test]
    fn new_sizer() {
        #[derive(Debug, PartialEq)]
        struct BasicSizer;

        impl ChunkSizer for BasicSizer {
            fn size(&self, _chunk: &str) -> usize {
                unimplemented!()
            }
        }

        let config = ChunkConfig::new(10).with_sizer(BasicSizer);
        assert_eq!(config.capacity, 10.into());
        assert_eq!(config.sizer, BasicSizer);
        assert!(config.trim());
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

    #[test]
    fn set_chunk_overlap() {
        let config = ChunkConfig::new(10).with_overlap(5).unwrap();
        assert_eq!(config.overlap(), 5);
    }

    #[test]
    fn cant_set_overlap_larger_than_capacity() {
        let chunk_config = ChunkConfig::new(5);
        let err = chunk_config.with_overlap(10).unwrap_err();
        assert_eq!(
            err.to_string(),
            "The overlap is larger than or equal to the desired chunk capacity"
        );
    }

    #[test]
    fn cant_set_overlap_larger_than_desired() {
        let chunk_config = ChunkConfig::new(5..15);
        let err = chunk_config.with_overlap(10).unwrap_err();
        assert_eq!(
            err.to_string(),
            "The overlap is larger than or equal to the desired chunk capacity"
        );
    }

    #[test]
    fn chunk_size_reference() {
        let config = ChunkConfig::new(1).with_sizer(&Characters);
        config.sizer().size("chunk");
    }

    #[test]
    fn chunk_size_cow() {
        let sizer: Cow<'_, Characters> = Cow::Owned(Characters);
        let config = ChunkConfig::new(1).with_sizer(sizer);
        config.sizer().size("chunk");

        let sizer = Cow::Borrowed(&Characters);
        let config = ChunkConfig::new(1).with_sizer(sizer);
        config.sizer().size("chunk");
    }

    #[test]
    fn chunk_size_arc() {
        let sizer = Arc::new(Characters);
        let config = ChunkConfig::new(1).with_sizer(sizer);
        config.sizer().size("chunk");
    }

    #[test]
    fn chunk_size_ref() {
        let sizer = RefCell::new(Characters);
        let config = ChunkConfig::new(1).with_sizer(sizer.borrow());
        config.sizer().size("chunk");
    }

    #[test]
    fn chunk_size_ref_mut() {
        let sizer = RefCell::new(Characters);
        let config = ChunkConfig::new(1).with_sizer(sizer.borrow_mut());
        config.sizer().size("chunk");
    }

    #[test]
    fn chunk_size_box() {
        let sizer = Box::new(Characters);
        let config = ChunkConfig::new(1).with_sizer(sizer);
        config.sizer().size("chunk");
    }

    #[test]
    fn chunk_size_rc() {
        let sizer = Rc::new(Characters);
        let config = ChunkConfig::new(1).with_sizer(sizer);
        config.sizer().size("chunk");
    }
}
