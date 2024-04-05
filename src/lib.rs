#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_auto_cfg, doc_cfg))]

use std::{cmp::Ordering, ops::Range};

use chunk_size::MemoizedChunkSizer;
use itertools::Itertools;

mod chunk_size;
#[cfg(feature = "markdown")]
mod markdown;
mod text;

pub use chunk_size::{Characters, ChunkCapacity, ChunkSize, ChunkSizer};
#[cfg(feature = "markdown")]
pub use markdown::MarkdownSplitter;
pub use text::TextSplitter;

/// Captures information about document structure for a given text, and their
/// various semantic levels
#[derive(Debug)]
struct SemanticSplitRanges<Level>
where
    Level: Copy + Ord + PartialOrd + 'static,
{
    /// Levels that are always considered in splitting text, because they are always present.
    peristent_levels: &'static [Level],
    /// Range of each semantic item and its precalculated semantic level
    ranges: Vec<(Level, Range<usize>)>,
}

impl<Level> SemanticSplitRanges<Level>
where
    Level: Copy + Ord + PartialOrd + 'static,
{
    /// Retrieve ranges for all sections of a given level after an offset
    fn ranges_after_offset(
        &self,
        offset: usize,
    ) -> impl Iterator<Item = (Level, Range<usize>)> + '_ {
        self.ranges
            .iter()
            .filter(move |(_, sep)| sep.start >= offset)
            .map(|(l, r)| (*l, r.start..r.end))
    }
    /// Retrieve ranges for all sections of a given level after an offset
    fn level_ranges_after_offset(
        &self,
        offset: usize,
        level: Level,
    ) -> impl Iterator<Item = (Level, Range<usize>)> + '_ {
        // Find the first item of this level. Allows us to skip larger items of a higher level that surround this one.
        // Otherwise all lower levels would only return the first item of the higher level that wraps it.
        let first_item = self
            .ranges_after_offset(offset)
            .position(|(l, _)| l == level)
            .and_then(|i| {
                self.ranges_after_offset(offset)
                    .skip(i)
                    .coalesce(|(a_level, a_range), (b_level, b_range)| {
                        // If we are at the first item, if two neighboring elements have the same level and start, take the shorter one
                        if a_level == b_level && a_range.start == b_range.start && i == 0 {
                            Ok((b_level, b_range))
                        } else {
                            Err(((a_level, a_range), (b_level, b_range)))
                        }
                    })
                    // Just take the first of these items
                    .next()
            });
        // let first_item = self.ranges_after_offset(offset).find(|(l, _)| l == &level);
        self.ranges_after_offset(offset)
            .filter(move |(l, _)| l >= &level)
            .skip_while(move |(l, r)| {
                first_item.as_ref().is_some_and(|(_, fir)| {
                    (l > &level && r.contains(&fir.start))
                        || (l == &level && r.start == fir.start && r.end > fir.end)
                })
            })
    }

    /// Return a unique, sorted list of all line break levels present before the next max level, added
    /// to all of the base semantic levels, in order from smallest to largest
    fn levels_in_remaining_text(&self, offset: usize) -> impl Iterator<Item = Level> + '_ {
        let existing_levels = self.ranges_after_offset(offset).map(|(l, _)| l);

        self.peristent_levels
            .iter()
            .copied()
            .chain(existing_levels)
            .sorted()
            .dedup()
    }

    /// Clear out ranges we have moved past so future iterations are faster
    fn update_ranges(&mut self, cursor: usize) {
        self.ranges.retain(|(_, range)| range.start >= cursor);
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

    /// Generate a new instance from a given text.
    fn new(text: &str) -> Self;

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
struct TextChunks<'text, 'sizer, C, S, L>
where
    C: ChunkCapacity,
    S: ChunkSizer,
    L: Copy + Ord + PartialOrd + 'static,
    SemanticSplitRanges<L>: SemanticSplit,
{
    /// How to validate chunk sizes
    chunk_sizer: MemoizedChunkSizer<'sizer, C, S>,
    /// Current byte offset in the `text`
    cursor: usize,
    /// Reusable container for next sections to avoid extra allocations
    next_sections: Vec<(usize, &'text str)>,
    /// Splitter used for determining semantic levels.
    semantic_split: SemanticSplitRanges<L>,
    /// Original text to iterate over and generate chunks from
    text: &'text str,
    /// Whether or not chunks should be trimmed
    trim_chunks: bool,
}

impl<'sizer, 'text: 'sizer, C, S, L> TextChunks<'text, 'sizer, C, S, L>
where
    C: ChunkCapacity,
    S: ChunkSizer,
    L: Copy + Ord + PartialOrd + 'static,
    SemanticSplitRanges<L>: SemanticSplit<Level = L>,
{
    /// Generate new [`TextChunks`] iterator for a given text.
    /// Starts with an offset of 0
    fn new(chunk_capacity: C, chunk_sizer: &'sizer S, text: &'text str, trim_chunks: bool) -> Self {
        Self {
            cursor: 0,
            chunk_sizer: MemoizedChunkSizer::new(chunk_capacity, chunk_sizer),
            next_sections: Vec::new(),
            semantic_split: SemanticSplitRanges::<L>::new(text),
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
        self.semantic_split.update_ranges(self.cursor);
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

            match chunk_size.fits() {
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
            if chunk_size.fits().is_lt() {
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
                if size.size() <= chunk_size.size() {
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

        let levels_with_chunks = levels_in_remaining_text
            .filter_map(|level| {
                self.semantic_split
                    .semantic_chunks(self.cursor, remaining_text, level)
                    .next()
                    .map(|(_, str)| (level, str))
            })
            // We assume that larger levels are also longer. We can skip lower levels if going to a higher level would result in a shorter text
            .coalesce(|(a_level, a_str), (b_level, b_str)| {
                if a_str.len() >= b_str.len() {
                    Ok((b_level, b_str))
                } else {
                    Err(((a_level, a_str), (b_level, b_str)))
                }
            });
        for (level, str) in levels_with_chunks {
            let chunk_size = self
                .chunk_sizer
                .check_capacity(self.trim_chunk(self.cursor, str));
            // If this no longer fits, we use the level we are at.
            if chunk_size.fits().is_gt() {
                max_encoded_offset = chunk_size.max_chunk_size_offset();
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

impl<'sizer, 'text: 'sizer, C, S, L> Iterator for TextChunks<'text, 'sizer, C, S, L>
where
    C: ChunkCapacity,
    S: ChunkSizer,
    L: Copy + Ord + PartialOrd + 'static,
    SemanticSplitRanges<L>: SemanticSplit<Level = L>,
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
