#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_auto_cfg, doc_cfg))]

use std::{
    cmp::Ordering,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

use ahash::AHashMap;
use itertools::Itertools;

mod characters;
#[cfg(feature = "tokenizers")]
mod huggingface;
#[cfg(feature = "markdown")]
mod markdown;
mod text;
#[cfg(feature = "tiktoken-rs")]
mod tiktoken;

pub use characters::Characters;
#[cfg(feature = "markdown")]
pub use markdown::MarkdownSplitter;
pub use text::TextSplitter;

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
struct MemoizedChunkSizer<'sizer, C, S>
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
    fn new(chunk_capacity: C, sizer: &'sizer S) -> Self {
        Self {
            cache: AHashMap::new(),
            chunk_capacity,
            sizer,
        }
    }

    /// Determine the size of a given chunk to use for validation,
    /// returning a cached value if it exists, and storing the result if not.
    fn chunk_size(&mut self, offset: usize, chunk: &str) -> ChunkSize {
        *self
            .cache
            .entry(offset..(offset + chunk.len()))
            .or_insert_with(|| self.sizer.chunk_size(chunk, &self.chunk_capacity))
    }

    /// Check if the chunk is within the capacity. Chunk should be trimmed if necessary beforehand.
    fn check_capacity(&mut self, (offset, chunk): (usize, &str)) -> ChunkSize {
        let mut chunk_size = self.chunk_size(offset, chunk);
        if let Some(max_chunk_size_offset) = chunk_size.max_chunk_size_offset.as_mut() {
            *max_chunk_size_offset += offset;
        }
        chunk_size
    }

    /// Clear the cached values. Once we've moved the cursor,
    /// we don't need to keep the old values around.
    fn clear_cache(&mut self) {
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

/// Implementation that dictates the semantic split points available.
/// For plain text, this goes from characters, to grapheme clusters, to words,
/// to sentences, to linebreaks.
/// For something like Markdown, this would also include things like headers,
/// lists, and code blocks.
trait SemanticSplit {
    /// Internal type used to represent the level of semantic splitting.
    type Level: Copy + Ord + PartialOrd + 'static;

    /// Levels that are always considered in splitting text, because they are always present.
    const PERSISTENT_LEVELS: &'static [Self::Level];

    /// Generate a new instance from a given text.
    fn new(text: &str) -> Self;

    /// Retrieve ranges for each semantic level in the entire text
    fn ranges(&self) -> impl Iterator<Item = &(Self::Level, Range<usize>)> + '_;

    /// Retrieve ranges for all sections of a given level after an offset
    fn ranges_after_offset(
        &self,
        offset: usize,
        level: Self::Level,
    ) -> impl Iterator<Item = &(Self::Level, Range<usize>)> + '_ {
        let first_item = self.ranges().find(|(l, _)| l == &level);
        self.ranges()
            .filter(move |(l, sep)| l >= &level && sep.start >= offset)
            .skip_while(move |(l, r)| {
                first_item.is_some_and(|(_, fir)| l > &level && r.contains(&fir.start))
            })
    }

    /// Return a unique, sorted list of all line break levels present before the next max level, added
    /// to all of the base semantic levels, in order from smallest to largest
    fn levels_in_remaining_text(&self, offset: usize) -> impl Iterator<Item = Self::Level> + '_ {
        let existing_levels = self
            .ranges()
            // Only start taking them from the offset
            .filter(|(_, sep)| sep.start >= offset)
            .map(|(l, _)| l);

        Self::PERSISTENT_LEVELS
            .iter()
            .chain(existing_levels)
            .sorted()
            .dedup()
            .copied()
    }

    /// Split a given text into iterator over each semantic chunk
    fn semantic_chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        offset: usize,
        text: &'text str,
        semantic_level: Self::Level,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter;

    /// Trim the str and adjust the offset if necessary.
    /// This is the default behavior, but custom semantic levels may need different behavior.
    fn trim_chunk<'splitter, 'text: 'splitter>(
        &'splitter self,
        offset: usize,
        chunk: &'text str,
    ) -> (usize, &'text str) {
        // Figure out how many bytes we lose trimming the beginning
        let diff = chunk.len() - chunk.trim_start().len();
        (offset + diff, chunk.trim())
    }
}

/// Returns chunks of text with their byte offsets as an iterator.
#[derive(Debug)]
struct TextChunks<'text, 'sizer, C, S, Sp>
where
    C: ChunkCapacity,
    S: ChunkSizer,
    Sp: SemanticSplit,
{
    /// How to validate chunk sizes
    chunk_sizer: MemoizedChunkSizer<'sizer, C, S>,
    /// Current byte offset in the `text`
    cursor: usize,
    /// Reusable container for next sections to avoid extra allocations
    next_sections: Vec<(usize, &'text str)>,
    /// Splitter used for determining semantic levels.
    semantic_split: Sp,
    /// Original text to iterate over and generate chunks from
    text: &'text str,
    /// Whether or not chunks should be trimmed
    trim_chunks: bool,
}

impl<'sizer, 'text: 'sizer, C, S, Sp> TextChunks<'text, 'sizer, C, S, Sp>
where
    C: ChunkCapacity,
    S: ChunkSizer,
    Sp: SemanticSplit,
{
    /// Generate new [`TextChunks`] iterator for a given text.
    /// Starts with an offset of 0
    fn new(chunk_capacity: C, chunk_sizer: &'sizer S, text: &'text str, trim_chunks: bool) -> Self {
        Self {
            cursor: 0,
            chunk_sizer: MemoizedChunkSizer::new(chunk_capacity, chunk_sizer),
            next_sections: Vec::new(),
            semantic_split: Sp::new(text),
            text,
            trim_chunks,
        }
    }

    /// If trim chunks is on, trim the str and adjust the offset
    fn trim_chunk(&self, offset: usize, chunk: &'text str) -> (usize, &'text str) {
        if self.trim_chunks {
            self.semantic_split.trim_chunk(offset, chunk)
        } else {
            (offset, chunk)
        }
    }

    /// Generate the next chunk, applying trimming settings.
    /// Returns final byte offset and str.
    /// Will return `None` if given an invalid range.
    fn next_chunk(&mut self) -> Option<(usize, &'text str)> {
        // Reset caches so we can reuse the memory allocation
        self.chunk_sizer.clear_cache();
        self.update_next_sections();

        let start = self.cursor;
        let mut end = self.cursor;
        let mut equals_found = false;
        let mut low = 0;
        let mut high = self.next_sections.len().saturating_sub(1);
        let mut successful_index = None;
        let mut successful_chunk_size = None;

        while low <= high {
            let mid = low + (high - low) / 2;
            let (offset, str) = self.next_sections[mid];
            let text_end = offset + str.len();
            let chunk = self.text.get(start..text_end)?;
            let chunk_size = self
                .chunk_sizer
                .check_capacity(self.trim_chunk(start, chunk));

            match chunk_size.fits {
                Ordering::Less => {
                    // We got further than the last one, so update end
                    if text_end > end {
                        end = text_end;
                        successful_index = Some(mid);
                        successful_chunk_size = Some(chunk_size);
                    }
                }
                Ordering::Equal => {
                    // If we found a smaller equals use it. Or if this is the first equals we found
                    if text_end < end || !equals_found {
                        end = text_end;
                        successful_index = Some(mid);
                        successful_chunk_size = Some(chunk_size);
                    }
                    equals_found = true;
                }
                Ordering::Greater => {
                    // If we're too big on our smallest run, we must return at least one section
                    if mid == 0 && start == end {
                        end = text_end;
                        successful_index = Some(mid);
                        successful_chunk_size = Some(chunk_size);
                    }
                }
            };

            // Adjust search area
            if chunk_size.fits.is_lt() {
                low = mid + 1;
            } else if mid > 0 {
                high = mid - 1;
            } else {
                // Nothing to adjust
                break;
            }
        }

        // Sometimes with tokenization, we can get a bigger chunk for the same amount of tokens.
        if let (Some(successful_index), Some(chunk_size)) =
            (successful_index, successful_chunk_size)
        {
            let mut range = successful_index..self.next_sections.len();
            // We've already checked the successful index
            range.next();

            for index in range {
                let (offset, str) = self.next_sections[index];
                let text_end = offset + str.len();
                let chunk = self.text.get(start..text_end)?;
                let size = self
                    .chunk_sizer
                    .check_capacity(self.trim_chunk(start, chunk));
                if size.size <= chunk_size.size {
                    if text_end > end {
                        end = text_end;
                    }
                } else {
                    break;
                }
            }
        }

        self.cursor = end;

        let chunk = self.text.get(start..self.cursor)?;

        // Trim whitespace if user requested it
        Some(self.trim_chunk(start, chunk))
    }

    /// Find the ideal next sections, breaking it up until we find the largest chunk.
    /// Increasing length of chunk until we find biggest size to minimize validation time
    /// on huge chunks
    fn update_next_sections(&mut self) {
        // First thing, clear out the list, but reuse the allocated memory
        self.next_sections.clear();
        // Get starting level
        let mut levels_in_remaining_text =
            self.semantic_split.levels_in_remaining_text(self.cursor);
        let mut semantic_level = levels_in_remaining_text
            .next()
            .expect("Need at least one level to progress");
        // If we aren't at the highest semantic level, stop iterating sections that go beyond the range of the next level.
        let mut max_encoded_offset = None;

        let remaining_text = self.text.get(self.cursor..).unwrap();

        for (level, str) in levels_in_remaining_text.filter_map(|level| {
            self.semantic_split
                .semantic_chunks(self.cursor, remaining_text, level)
                .next()
                .map(|(_, str)| (level, str))
        }) {
            let chunk_size = self
                .chunk_sizer
                .check_capacity(self.trim_chunk(self.cursor, str));
            // If this no longer fits, we use the level we are at.
            if chunk_size.fits.is_gt() {
                max_encoded_offset = chunk_size.max_chunk_size_offset;
                break;
            }
            // Otherwise break up the text with the next level
            semantic_level = level;
        }

        let sections = self
            .semantic_split
            .semantic_chunks(self.cursor, remaining_text, semantic_level)
            // We don't want to return items at this level that go beyond the next highest semantic level, as that is most
            // likely a meaningful breakpoint we want to preserve. We already know that the next highest doesn't fit anyway,
            // so we should be safe to break once we reach it.
            .take_while_inclusive(move |(offset, _)| {
                max_encoded_offset.map_or(true, |max| offset <= &max)
            })
            .filter(|(_, str)| !str.is_empty());

        self.next_sections.extend(sections);
    }
}

impl<'sizer, 'text: 'sizer, C, S, Sp> Iterator for TextChunks<'text, 'sizer, C, S, Sp>
where
    C: ChunkCapacity,
    S: ChunkSizer,
    Sp: SemanticSplit,
{
    type Item = (usize, &'text str);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Make sure we haven't reached the end
            if self.cursor >= self.text.len() {
                return None;
            }

            match self.next_chunk()? {
                // Make sure we didn't get an empty chunk. Should only happen in
                // cases where we trim.
                (_, "") => continue,
                c => return Some(c),
            }
        }
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
}
