use std::{
    borrow::Cow,
    cell::{Ref, RefMut},
    cmp::Ordering,
    collections::hash_map::Entry,
    convert::Infallible,
    fmt,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
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

const PROBE_OVERSHOOT_DIVISOR: usize = 4;
const PROBE_START_BYTE_FACTOR: usize = 8;

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

/// The result of measuring a whole chunk against an inclusive limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChunkSize {
    /// The exact size, even if it exceeds the limit. The splitter can cache it.
    Exact(usize),
    /// The size is strictly greater than the limit; the exact size was not computed.
    ExceedsLimit,
}

/// Determines the size of a given chunk.
pub trait ChunkSizer {
    /// Determine the size of a given chunk to use for validation
    fn size(&self, chunk: &str) -> usize;

    /// Measure `chunk`, optionally stopping once its size is known to exceed `limit`.
    ///
    /// By default, returns [`ChunkSize::Exact`] with the result of [`Self::size`],
    /// including when that result is over the limit. An override may
    /// return [`ChunkSize::ExceedsLimit`] only if `size(chunk) > limit`. A
    /// prefix being over budget is not sufficient if extending it can reduce
    /// the size (for example, when tokens merge).
    ///
    /// ```
    /// use text_splitter::{Characters, ChunkSize, ChunkSizer};
    ///
    /// assert_eq!(Characters.size_with_limit("🍎🍐", 2), ChunkSize::Exact(2));
    /// assert_eq!(Characters.size_with_limit("🍎🍐", 1), ChunkSize::ExceedsLimit);
    /// ```
    fn size_with_limit(&self, chunk: &str, _limit: usize) -> ChunkSize {
        ChunkSize::Exact(self.size(chunk))
    }
}

/// Determines the size of a chunk, returning an error if sizing fails.
///
/// All [`ChunkSizer`] implementations automatically implement this trait with
/// [`Infallible`] as their error type. This adapter preserves their behavior,
/// including any panics; it does not catch panics or convert them into errors.
/// A type implementing [`ChunkSizer`] cannot also provide a separate
/// implementation of this trait. Use a newtype for different fallible behavior.
///
/// ```
/// use text_splitter::{ChunkConfig, TextSplitter, TryChunkSizer};
///
/// struct CheckedCharacters;
///
/// impl TryChunkSizer for CheckedCharacters {
///     type Error = &'static str;
///
///     fn try_size(&self, chunk: &str) -> Result<usize, Self::Error> {
///         if chunk.contains('\0') {
///             Err("NUL is not supported")
///         } else {
///             Ok(chunk.chars().count())
///         }
///     }
/// }
///
/// let splitter = TextSplitter::new(ChunkConfig::new(10).with_sizer(CheckedCharacters));
/// let chunks = splitter.try_chunks("hello\0world").collect::<Result<Vec<_>, _>>();
/// assert_eq!(chunks, Err("NUL is not supported"));
/// ```
pub trait TryChunkSizer {
    /// The error returned when sizing fails.
    type Error;

    /// Determine the size of a chunk.
    ///
    /// # Errors
    ///
    /// Returns the sizer's error if the chunk cannot be sized.
    fn try_size(&self, chunk: &str) -> Result<usize, Self::Error>;

    /// Fallible version of [`ChunkSizer::size_with_limit`]. By default, measures
    /// the whole chunk with [`Self::try_size`]. A chunk exactly at `limit` fits.
    ///
    /// # Errors
    ///
    /// Preserve errors from [`Self::try_size`]; do not report an invalid chunk
    /// as merely over the limit.
    fn try_size_with_limit(&self, chunk: &str, _limit: usize) -> Result<ChunkSize, Self::Error> {
        self.try_size(chunk).map(ChunkSize::Exact)
    }
}

impl<T: ChunkSizer + ?Sized> TryChunkSizer for T {
    type Error = Infallible;

    fn try_size(&self, chunk: &str) -> Result<usize, Self::Error> {
        Ok(self.size(chunk))
    }

    fn try_size_with_limit(&self, chunk: &str, limit: usize) -> Result<ChunkSize, Self::Error> {
        Ok(self.size_with_limit(chunk, limit))
    }
}

// Wrapping a sizer must preserve its budget check as well as its exact measurement.
macro_rules! delegate_sizing {
    () => {
        fn size(&self, chunk: &str) -> usize {
            (**self).size(chunk)
        }

        fn size_with_limit(&self, chunk: &str, limit: usize) -> ChunkSize {
            (**self).size_with_limit(chunk, limit)
        }
    };
}

impl<T: ChunkSizer> ChunkSizer for &T {
    delegate_sizing!();
}

impl<T: ChunkSizer> ChunkSizer for Ref<'_, T> {
    delegate_sizing!();
}

impl<T: ChunkSizer> ChunkSizer for RefMut<'_, T> {
    delegate_sizing!();
}

impl<T: ChunkSizer> ChunkSizer for Box<T> {
    delegate_sizing!();
}

impl<T> ChunkSizer for Cow<'_, T>
where
    T: ChunkSizer + ToOwned + ?Sized,
    <T as ToOwned>::Owned: ChunkSizer,
{
    delegate_sizing!();
}

impl<T: ChunkSizer> ChunkSizer for Rc<T> {
    delegate_sizing!();
}

impl<T: ChunkSizer> ChunkSizer for Arc<T> {
    delegate_sizing!();
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
    Sizer: TryChunkSizer,
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
    Sizer: TryChunkSizer,
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
    pub fn with_sizer<S: TryChunkSizer>(self, sizer: S) -> ChunkConfig<S> {
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

/// A memoized chunk sizer that caches the size of chunks.
/// Very helpful when the same chunk is being validated multiple times, which
/// happens often, and can be expensive to compute, such as with tokenizers.
#[derive(Debug)]
pub struct MemoizedChunkSizer<'sizer, Sizer>
where
    Sizer: TryChunkSizer,
{
    /// Cache of chunk sizes per byte offset range for base capacity
    size_cache: AHashMap<Range<usize>, usize>,
    /// The sizer used for caluclating chunk sizes
    sizer: &'sizer Sizer,
}

impl<'sizer, Sizer> MemoizedChunkSizer<'sizer, Sizer>
where
    Sizer: TryChunkSizer,
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
    pub fn chunk_size(
        &mut self,
        offset: usize,
        chunk: &str,
        trim: Trim,
    ) -> Result<usize, Sizer::Error> {
        let (range, chunk) = trimmed_range(offset, chunk, trim);
        match self.size_cache.entry(range) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let size = self.sizer.try_size(chunk)?;
                entry.insert(size);
                Ok(size)
            }
        }
    }

    /// Classify the same trimmed slice as `chunk_size`, without requiring its
    /// exact size when the sizer can prove the whole slice is too large.
    fn chunk_fits(
        &mut self,
        offset: usize,
        chunk: &str,
        capacity: &ChunkCapacity,
        trim: Trim,
    ) -> Result<Ordering, Sizer::Error> {
        let (range, chunk) = trimmed_range(offset, chunk, trim);
        let size = match self.size_cache.entry(range) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => match self.sizer.try_size_with_limit(chunk, capacity.max)? {
                ChunkSize::Exact(size) => *entry.insert(size),
                ChunkSize::ExceedsLimit => return Ok(Ordering::Greater),
            },
        };
        Ok(capacity.fits(size))
    }

    /// Find the best level to start splitting the text
    pub fn find_correct_level<'text, L, Boundaries>(
        &mut self,
        offset: usize,
        capacity: &ChunkCapacity,
        levels_with_first_chunk: impl Iterator<Item = (L, &'text str, Option<L>)>,
        mut lower_boundaries_for: impl FnMut(Option<L>, usize) -> Boundaries,
        trim: Trim,
    ) -> Result<(Option<L>, Option<usize>), Sizer::Error>
    where
        L: Copy + fmt::Debug,
        Boundaries: Iterator<Item = usize>,
    {
        let mut semantic_level = None;
        let mut max_offset = None;

        // We assume that larger levels are also longer. We can skip lower levels if going to a higher level would result in a shorter text
        let levels_with_first_chunk = levels_with_first_chunk.coalesce(
            |(a_level, a_str, a_lower_level), (b_level, b_str, b_lower_level)| {
                if a_str.len() >= b_str.len() {
                    Ok((b_level, b_str, b_lower_level))
                } else {
                    Err((
                        (a_level, a_str, a_lower_level),
                        (b_level, b_str, b_lower_level),
                    ))
                }
            },
        );

        for (level, str, lower_level) in levels_with_first_chunk {
            // Skip tokenizing levels that we know are too small anyway.
            let len = str.len();
            if len > capacity.max {
                let mut lower_boundaries = lower_boundaries_for(lower_level, offset + len);
                let fits = self.chunk_fits_with_boundaries(
                    offset,
                    str,
                    capacity,
                    &mut lower_boundaries,
                    trim,
                )?;
                // If this no longer fits, we use the level we are at.
                if fits.is_gt() {
                    // Keep the old search window while avoiding the full-size call
                    // when lower semantic boundaries prove this section is too large.
                    max_offset = Some(offset + len);
                    break;
                }
            }
            // Otherwise break up the text with the next level
            semantic_level = Some(level);
        }

        Ok((semantic_level, max_offset))
    }

    fn chunk_fits_with_boundaries(
        &mut self,
        offset: usize,
        chunk: &str,
        capacity: &ChunkCapacity,
        lower_boundaries: &mut impl Iterator<Item = usize>,
        trim: Trim,
    ) -> Result<Ordering, Sizer::Error> {
        let chunk_end = offset + chunk.len();
        let probe_max = probe_max_size(capacity);
        let probe_start = offset
            .saturating_add(capacity.max.saturating_mul(PROBE_START_BYTE_FACTOR).max(1))
            .min(chunk_end);
        if probe_start == chunk_end {
            return self.chunk_fits(offset, chunk, capacity, trim);
        }

        let mut step: usize = 1;
        let mut next_boundary =
            lower_boundaries.find(|&boundary| boundary > offset && boundary >= probe_start);

        while let Some(boundary) = next_boundary {
            if boundary >= chunk_end {
                break;
            }

            let prefix = chunk
                .get(..(boundary - offset))
                .expect("valid character boundary");
            let chunk_size = self.chunk_size(offset, prefix, trim)?;
            if chunk_size > probe_max {
                return Ok(Ordering::Greater);
            }

            let skip = step.saturating_sub(1);
            step = step.saturating_mul(2);
            next_boundary = lower_boundaries.nth(skip);
        }

        self.chunk_fits(offset, chunk, capacity, trim)
    }

    /// Clear the cached values. Once we've moved the cursor,
    /// we don't need to keep the old values around.
    pub fn clear_cache(&mut self) {
        self.size_cache.clear();
    }
}

fn trimmed_range(offset: usize, chunk: &str, trim: Trim) -> (Range<usize>, &str) {
    let (start, chunk) = trim.trim(offset, chunk);
    (start..start + chunk.len(), chunk)
}

fn probe_max_size(capacity: &ChunkCapacity) -> usize {
    capacity
        .max
        .saturating_add((capacity.max / PROBE_OVERSHOOT_DIVISOR).max(1))
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
            memoized_sizer.chunk_size(0, text, Trim::All).unwrap();
        }

        assert_eq!(memoized_sizer.sizer.calls.load(atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn memoized_sizer_calculates_once_per_different_text() {
        let sizer = CountingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let text = "1234567890";
        for i in 0..10 {
            memoized_sizer
                .chunk_size(0, text.get(0..i).unwrap(), Trim::All)
                .unwrap();
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
            memoized_sizer.chunk_size(0, text, Trim::All).unwrap();
            memoized_sizer.clear_cache();
        }

        assert_eq!(
            memoized_sizer.sizer.calls.load(atomic::Ordering::SeqCst),
            10
        );
    }

    #[test]
    fn memoized_sizer_caches_successes_but_not_errors() {
        struct RetrySizer(AtomicUsize);

        impl TryChunkSizer for RetrySizer {
            type Error = &'static str;

            fn try_size(&self, chunk: &str) -> Result<usize, Self::Error> {
                if self.0.fetch_add(1, atomic::Ordering::SeqCst) == 0 {
                    Err("first call failed")
                } else {
                    Ok(Characters.size(chunk))
                }
            }
        }

        for limited in [false, true] {
            let sizer = RetrySizer(AtomicUsize::new(0));
            let mut memoized = MemoizedChunkSizer::new(&sizer);
            let result = if limited {
                memoized
                    .chunk_fits(0, "abc", &1.into(), Trim::All)
                    .map(|_| ())
            } else {
                memoized.chunk_size(0, "abc", Trim::All).map(|_| ())
            };
            assert_eq!(result, Err("first call failed"));
            assert!(memoized.size_cache.is_empty());
            assert_eq!(memoized.chunk_size(0, "abc", Trim::All), Ok(3));
            assert_eq!(memoized.chunk_size(0, "abc", Trim::All), Ok(3));
            assert_eq!(sizer.0.load(atomic::Ordering::SeqCst), 2);
        }
    }

    #[test]
    fn budget_check_shares_exact_sizes_even_when_over_budget() {
        let sizer = CountingSizer::default();
        let mut memo = MemoizedChunkSizer::new(&sizer);
        // The default hook computes the exact count even when it's over budget.
        assert_eq!(
            memo.chunk_fits(5, "  aé🦀  ", &2.into(), Trim::All),
            Ok(Ordering::Greater)
        );
        // Same trimmed byte range with a different limit, then an exact query.
        assert_eq!(
            memo.chunk_fits(7, "aé🦀", &(3..=5).into(), Trim::None),
            Ok(Ordering::Equal)
        );
        assert_eq!(memo.chunk_size(7, "aé🦀", Trim::None), Ok(3));
        assert_eq!(sizer.calls.load(atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn whole_section_over_limit_proof_is_not_cached_as_a_size() {
        let mut memo = MemoizedChunkSizer::new(&Characters);
        let text = "one long section";
        assert_eq!(
            memo.chunk_fits_with_boundaries(
                0,
                text,
                &1.into(),
                &mut [text.len()].into_iter(),
                Trim::None,
            ),
            Ok(Ordering::Greater)
        );
        assert!(memo.size_cache.is_empty());
        assert_eq!(memo.chunk_size(0, text, Trim::None), Ok(text.len()));
        assert_eq!(
            memo.chunk_fits(0, text, &text.len().into(), Trim::None),
            Ok(Ordering::Equal)
        );
    }

    #[test]
    fn boundary_probe_is_skipped_when_probe_start_is_chunk_end() {
        let sizer = CountingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let mut lower_boundaries = [usize::MAX].into_iter();

        let fits = memoized_sizer.chunk_fits_with_boundaries(
            0,
            "12345678901",
            &ChunkCapacity::new(10),
            &mut lower_boundaries,
            Trim::All,
        );

        assert_eq!(fits, Ok(Ordering::Greater));
        assert_eq!(sizer.calls.load(atomic::Ordering::SeqCst), 1);
        assert_eq!(lower_boundaries.next(), Some(usize::MAX));
    }

    #[test]
    fn boundary_probe_stops_when_next_boundary_exceeds_chunk_end() {
        let sizer = CountingSizer::default();
        let mut memoized_sizer = MemoizedChunkSizer::new(&sizer);
        let chunk = "1234567890".repeat(9);
        let mut lower_boundaries = [usize::MAX].into_iter();

        let fits = memoized_sizer.chunk_fits_with_boundaries(
            0,
            &chunk,
            &ChunkCapacity::new(10),
            &mut lower_boundaries,
            Trim::All,
        );

        assert_eq!(fits, Ok(Ordering::Greater));
        assert_eq!(sizer.calls.load(atomic::Ordering::SeqCst), 1);
        assert_eq!(lower_boundaries.next(), None);
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

    fn assert_delegates_sizing<S: ChunkSizer>(sizer: S) {
        let config = ChunkConfig::new(1).with_sizer(sizer);
        assert_eq!(config.sizer().size("chunk"), 5);
        // The default implementation would return Exact(5); Characters can stop early.
        assert_eq!(
            config.sizer().try_size_with_limit("chunk", 1),
            Ok(ChunkSize::ExceedsLimit)
        );
    }

    #[test]
    fn chunk_size_reference() {
        assert_delegates_sizing::<&Characters>(&Characters);
    }

    #[test]
    fn chunk_size_cow() {
        let sizer: Cow<'_, Characters> = Cow::Owned(Characters);
        assert_delegates_sizing(sizer);

        let sizer = Cow::Borrowed(&Characters);
        assert_delegates_sizing(sizer);
    }

    #[test]
    fn chunk_size_arc() {
        let sizer = Arc::new(Characters);
        assert_delegates_sizing(sizer);
    }

    #[test]
    fn chunk_size_ref() {
        let sizer = RefCell::new(Characters);
        assert_delegates_sizing(sizer.borrow());
    }

    #[test]
    fn chunk_size_ref_mut() {
        let sizer = RefCell::new(Characters);
        assert_delegates_sizing(sizer.borrow_mut());
    }

    #[test]
    fn chunk_size_box() {
        let sizer = Box::new(Characters);
        assert_delegates_sizing(sizer);
    }

    #[test]
    fn chunk_size_rc() {
        let sizer = Rc::new(Characters);
        assert_delegates_sizing(sizer);
    }
}
