#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_auto_cfg, doc_cfg))]

use std::{
    cmp::Ordering,
    fmt,
    iter::once,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

use either::Either;
use itertools::Itertools;

mod characters;
#[cfg(feature = "tokenizers")]
mod huggingface;
mod text;
#[cfg(feature = "tiktoken-rs")]
mod tiktoken;
#[cfg(feature = "markdown")]
pub mod unstable_markdown;

pub use characters::Characters;
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

/// How a particular semantic level relates to surrounding text elements.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum SemanticSplitPosition {
    /// The semantic level should be included in the previous chunk.
    Prev,
    /// The semantic level should be treated as its own chunk.
    Own,
    /// The semantic level should be included in the next chunk.
    Next,
}

/// Information required by generic Semantic Levels
trait Level: fmt::Debug {
    fn split_position(&self) -> SemanticSplitPosition;
}

/// Implementation that dictates the semantic split points available.
/// For plain text, this goes from characters, to grapheme clusters, to words,
/// to sentences, to linebreaks.
/// For something like Markdown, this would also include things like headers,
/// lists, and code blocks.
trait SemanticSplit {
    /// Internal type used to represent the level of semantic splitting.
    type Level: Copy + Level + Ord + PartialOrd + 'static;

    /// Levels that are always considered in splitting text, because they are always present.
    const PERSISTENT_LEVELS: &'static [Self::Level];

    /// Generate a new instance from a given text.
    fn new(text: &str) -> Self;

    /// Retrieve ranges for each semantic level in the entire text
    fn ranges(&self) -> impl Iterator<Item = &(Self::Level, Range<usize>)> + '_;

    /// Maximum level of semantic splitting in the text
    fn max_level(&self) -> Self::Level;

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
    fn levels_in_next_max_chunk(&self, offset: usize) -> impl Iterator<Item = Self::Level> + '_ {
        let max_level = self.max_level();
        let existing_levels = self
            .ranges()
            // Only start taking them from the offset
            .filter(|(_, sep)| sep.start >= offset)
            // Stop once we hit the first of the max level
            .take_while_inclusive(|(l, _)| l < &max_level)
            .map(|(l, _)| l)
            .copied();

        Self::PERSISTENT_LEVELS
            .iter()
            .copied()
            .chain(existing_levels)
            .sorted()
            .dedup()
    }

    /// Split a given text into iterator over each semantic chunk
    fn semantic_chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        offset: usize,
        text: &'text str,
        semantic_level: Self::Level,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter;
}

/// Returns chunks of text with their byte offsets as an iterator.
#[derive(Debug)]
struct TextChunks<'text, 'sizer, C, S, Sp>
where
    C: ChunkCapacity,
    S: ChunkSizer,
    Sp: SemanticSplit,
{
    /// Size of the chunks to generate
    chunk_capacity: C,
    /// How to validate chunk sizes
    chunk_sizer: &'sizer S,
    /// Current byte offset in the `text`
    cursor: usize,
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
            chunk_capacity,
            chunk_sizer,
            semantic_split: Sp::new(text),
            text,
            trim_chunks,
        }
    }

    /// If trim chunks is on, trim the str and adjust the offset
    fn trim_chunk(&self, offset: usize, chunk: &'text str) -> (usize, &'text str) {
        if self.trim_chunks {
            // Figure out how many bytes we lose trimming the beginning
            let diff = chunk.len() - chunk.trim_start().len();
            (offset + diff, chunk.trim())
        } else {
            (offset, chunk)
        }
    }

    /// Is the given text within the chunk size?
    fn check_capacity(&self, offset: usize, chunk: &str) -> ChunkSize {
        let (offset, chunk) = self.trim_chunk(offset, chunk);
        let mut chunk_size = self.chunk_sizer.chunk_size(chunk, &self.chunk_capacity);
        if let Some(max_chunk_size_offset) = chunk_size.max_chunk_size_offset.as_mut() {
            *max_chunk_size_offset += offset;
        }
        chunk_size
    }

    /// Generate the next chunk, applying trimming settings.
    /// Returns final byte offset and str.
    /// Will return `None` if given an invalid range.
    fn next_chunk(&mut self) -> Option<(usize, &'text str)> {
        let start = self.cursor;
        let mut end = self.cursor;
        let mut equals_found = false;

        let sections = self.next_sections()?.collect::<Vec<_>>();
        let mut sizes = sections
            .iter()
            .map(|_| None)
            .collect::<Vec<Option<ChunkSize>>>();
        let mut low = 0;
        let mut high = sections.len().saturating_sub(1);
        let mut successful_index = None;

        while low <= high {
            let mid = low + (high - low) / 2;
            let (offset, str) = sections[mid];
            let text_end = offset + str.len();
            let chunk = self.text.get(start..text_end)?;
            let chunk_size = self.check_capacity(start, chunk);
            sizes[mid] = Some(chunk_size);

            match chunk_size.fits {
                Ordering::Less => {
                    // We got further than the last one, so update end
                    if text_end > end {
                        end = text_end;
                        successful_index = Some(mid);
                    }
                }
                Ordering::Equal => {
                    // If we found a smaller equals use it. Or if this is the first equals we found
                    if text_end < end || !equals_found {
                        end = text_end;
                        successful_index = Some(mid);
                    }
                    equals_found = true;
                }
                Ordering::Greater => {
                    // If we're too big on our smallest run, we must return at least one section
                    if mid == 0 && start == end {
                        end = text_end;
                        successful_index = Some(mid);
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
        if let Some((successful_index, chunk_size)) =
            successful_index.and_then(|successful_index| {
                Some((successful_index, sizes.get(successful_index)?.as_ref()?))
            })
        {
            for (size, (offset, str)) in sizes.iter().zip(sections).skip(successful_index) {
                let text_end = offset + str.len();
                match size {
                    Some(size) if size.size <= chunk_size.size => {
                        if text_end > end {
                            end = text_end;
                        }
                    }
                    // We didn't tokenize this section yet
                    None => {
                        let chunk = self.text.get(start..text_end)?;
                        let size = self.check_capacity(start, chunk);
                        if size.size <= chunk_size.size {
                            if text_end > end {
                                end = text_end;
                            }
                        } else {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        }

        self.cursor = end;

        let chunk = self.text.get(start..self.cursor)?;

        // Trim whitespace if user requested it
        Some(if self.trim_chunks {
            // Figure out how many bytes we lose trimming the beginning
            let offset = chunk.len() - chunk.trim_start().len();
            (start + offset, chunk.trim())
        } else {
            (start, chunk)
        })
    }

    /// Find the ideal next sections, breaking it up until we find the largest chunk.
    /// Increasing length of chunk until we find biggest size to minimize validation time
    /// on huge chunks
    fn next_sections(&'sizer self) -> Option<impl Iterator<Item = (usize, &'text str)> + 'sizer> {
        // Next levels to try. Will stop at max level. We check only levels in the next max level
        // chunk so we don't bypass it if not all levels are present in every chunk.
        let mut levels = self.semantic_split.levels_in_next_max_chunk(self.cursor);
        // Get starting level
        let mut semantic_level = levels.next()?;
        // If we aren't at the highest semantic level, stop iterating sections that go beyond the range of the next level.
        let mut max_encoded_offset = None;

        for level in levels {
            let (_, str) = self.semantic_chunks(level).next()?;
            let chunk_size = self.check_capacity(self.cursor, str);
            // If this no longer fits, we use the level we are at. Or if we already
            // have the rest of the string
            if chunk_size.fits.is_gt() || self.text.get(self.cursor..)? == str {
                max_encoded_offset = chunk_size.max_chunk_size_offset;
                break;
            }
            // Otherwise break up the text with the next level
            semantic_level = level;
        }

        Some(
            self.semantic_chunks(semantic_level)
                // We don't want to return items at this level that go beyond the next highest semantic level, as that is most
                // likely a meaningful breakpoint we want to preserve. We already know that the next highest doesn't fit anyway,
                // so we should be safe to break once we reach it.
                .take_while_inclusive(move |(offset, _)| {
                    max_encoded_offset.map_or(true, |max| offset <= &max)
                })
                .filter(|(_, str)| !str.is_empty()),
        )
    }

    fn semantic_chunks(
        &'sizer self,
        level: <Sp as SemanticSplit>::Level,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'sizer {
        self.semantic_split.semantic_chunks(
            self.cursor,
            self.text.get(self.cursor..).unwrap(),
            level,
        )
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

/// Given a list of separator ranges, construct the sections of the text
fn split_str_by_separator<L: Level>(
    text: &str,
    separator_ranges: impl Iterator<Item = (L, Range<usize>)>,
) -> impl Iterator<Item = (usize, &str)> {
    let mut cursor = 0;
    let mut final_match = false;
    separator_ranges
        .batching(move |it| {
            loop {
                match it.next() {
                    // If we've hit the end, actually return None
                    None if final_match => return None,
                    // First time we hit None, return the final section of the text
                    None => {
                        final_match = true;
                        return text.get(cursor..).map(|t| Either::Left(once((cursor, t))));
                    }
                    // Return text preceding match + the match
                    Some((level, range)) => {
                        let offset = cursor;
                        match level.split_position() {
                            SemanticSplitPosition::Prev => {
                                if range.end < cursor {
                                    continue;
                                }
                                let section = text
                                    .get(cursor..range.end)
                                    .expect("invalid character sequence");
                                cursor = range.end;
                                return Some(Either::Left(once((offset, section))));
                            }
                            SemanticSplitPosition::Own => {
                                if range.start < cursor {
                                    continue;
                                }
                                let prev_section = text
                                    .get(cursor..range.start)
                                    .expect("invalid character sequence");
                                let separator = text
                                    .get(range.start..range.end)
                                    .expect("invalid character sequence");
                                cursor = range.end;
                                return Some(Either::Right(
                                    [(offset, prev_section), (range.start, separator)].into_iter(),
                                ));
                            }
                            SemanticSplitPosition::Next => {
                                if range.start < cursor {
                                    continue;
                                }
                                let prev_section = text
                                    .get(cursor..range.start)
                                    .expect("invalid character sequence");
                                // Separator will be part of the next chunk
                                cursor = range.start;
                                return Some(Either::Left(once((offset, prev_section))));
                            }
                        }
                    }
                }
            }
        })
        .flatten()
        .filter(|(_, s)| !s.is_empty())
}

#[cfg(test)]
mod tests {
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
}
