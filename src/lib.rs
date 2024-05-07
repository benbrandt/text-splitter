#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_auto_cfg, doc_cfg))]

use std::{cmp::Ordering, fmt, ops::Range};

use auto_enums::auto_enum;
use chunk_size::MemoizedChunkSizer;
use itertools::Itertools;
use strum::{EnumIter, IntoEnumIterator};
use trim::Trim;

mod chunk_size;
#[cfg(feature = "markdown")]
mod markdown;
mod text;
mod trim;

pub use chunk_size::{
    Characters, ChunkCapacity, ChunkCapacityError, ChunkConfig, ChunkConfigError, ChunkSize,
    ChunkSizer,
};
#[cfg(feature = "markdown")]
pub use markdown::MarkdownSplitter;
pub use text::TextSplitter;
use unicode_segmentation::UnicodeSegmentation;

/// When using a custom semantic level, it is possible that none of them will
/// be small enough to fit into the chunk size. In order to make sure we can
/// still move the cursor forward, we fallback to unicode segmentation.
#[derive(Clone, Copy, Debug, EnumIter, Eq, PartialEq, Ord, PartialOrd)]
enum FallbackLevel {
    /// Split by individual chars. May be larger than a single byte,
    /// but we don't go lower so we always have valid UTF str's.
    Char,
    /// Split by [unicode grapheme clusters](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)    Grapheme,
    GraphemeCluster,
    /// Split by [unicode words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    Word,
    /// Split by [unicode sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    Sentence,
}

impl FallbackLevel {
    #[auto_enum(Iterator)]
    fn sections(self, text: &str) -> impl Iterator<Item = (usize, &str)> {
        match self {
            Self::Char => text.char_indices().map(move |(i, c)| {
                (
                    i,
                    text.get(i..i + c.len_utf8()).expect("char should be valid"),
                )
            }),
            Self::GraphemeCluster => text.grapheme_indices(true),
            Self::Word => text.split_word_bound_indices(),
            Self::Sentence => text.split_sentence_bound_indices(),
        }
    }
}

/// Custom-defined levels of semantic splitting for custom document types.
trait SemanticLevel: Copy + fmt::Debug + Ord + PartialOrd + 'static {
    /// Trimming behavior to use when trimming chunks
    const TRIM: Trim = Trim::All;

    /// Generate a list of offsets for each semantic level within the text.
    fn offsets(text: &str) -> impl Iterator<Item = (Self, Range<usize>)>;

    /// Given a level, split the text into sections based on the level.
    /// Level ranges are also provided of items that are equal to or greater than the current level.
    fn sections(
        self,
        text: &str,
        level_ranges: impl Iterator<Item = (Self, Range<usize>)>,
    ) -> impl Iterator<Item = (usize, &str)>;
}

/// Captures information about document structure for a given text, and their
/// various semantic levels
#[derive(Debug)]
struct SemanticSplitRanges<Level>
where
    Level: SemanticLevel,
{
    /// Range of each semantic item and its precalculated semantic level
    ranges: Vec<(Level, Range<usize>)>,
}

impl<Level> SemanticSplitRanges<Level>
where
    Level: SemanticLevel,
{
    fn new(ranges: Vec<(Level, Range<usize>)>) -> Self {
        Self { ranges }
    }

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
        self.ranges_after_offset(offset)
            .map(|(l, _)| l)
            .sorted()
            .dedup()
    }

    /// Split a given text into iterator over each semantic chunk
    fn semantic_chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        offset: usize,
        text: &'text str,
        semantic_level: Level,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        semantic_level
            .sections(
                text,
                self.level_ranges_after_offset(offset, semantic_level)
                    .map(move |(l, sep)| (l, sep.start - offset..sep.end - offset)),
            )
            .map(move |(i, str)| (offset + i, str))
    }

    /// Clear out ranges we have moved past so future iterations are faster
    fn update_ranges(&mut self, cursor: usize) {
        self.ranges.retain(|(_, range)| range.start >= cursor);
    }
}

/// Returns chunks of text with their byte offsets as an iterator.
#[derive(Debug)]
struct TextChunks<'text, 'sizer, Sizer, Level>
where
    Sizer: ChunkSizer,
    Level: SemanticLevel,
{
    /// Chunk configuration for this iterator
    chunk_config: &'sizer ChunkConfig<Sizer>,
    /// How to validate chunk sizes
    chunk_sizer: MemoizedChunkSizer<'sizer, Sizer>,
    /// Current byte offset in the `text`
    cursor: usize,
    /// Reusable container for next sections to avoid extra allocations
    next_sections: Vec<(usize, &'text str)>,
    /// Previous item's end byte offset
    prev_item_end: usize,
    /// Splitter used for determining semantic levels.
    semantic_split: SemanticSplitRanges<Level>,
    /// Original text to iterate over and generate chunks from
    text: &'text str,
}

impl<'sizer, 'text: 'sizer, Sizer, Level> TextChunks<'text, 'sizer, Sizer, Level>
where
    Sizer: ChunkSizer,
    Level: SemanticLevel,
{
    /// Generate new [`TextChunks`] iterator for a given text.
    /// Starts with an offset of 0
    fn new(chunk_config: &'sizer ChunkConfig<Sizer>, text: &'text str) -> Self {
        Self {
            chunk_config,
            chunk_sizer: MemoizedChunkSizer::new(chunk_config, Level::TRIM),
            cursor: 0,
            next_sections: Vec::new(),
            prev_item_end: 0,
            semantic_split: SemanticSplitRanges::new(Level::offsets(text).collect()),
            text,
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

        let (start, end) = self.binary_search_next_chunk()?;

        // Optionally move cursor back if overlap is desired
        self.update_cursor(end);

        let chunk = self.text.get(start..end)?;
        // Trim whitespace if user requested it
        Some(self.chunk_sizer.trim_chunk(start, chunk))
    }

    /// Use binary search to find the next chunk that fits within the chunk size
    fn binary_search_next_chunk(&mut self) -> Option<(usize, usize)> {
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
            let chunk_size = self.chunk_sizer.check_capacity(start, chunk, false);

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
                let size = self.chunk_sizer.check_capacity(start, chunk, false);
                if size.size() <= chunk_size.size() {
                    if text_end > end {
                        end = text_end;
                    }
                } else {
                    break;
                }
            }
        }

        Some((start, end))
    }

    /// Use binary search to find the sections that fit within the overlap size.
    /// If no overlap deisired, return end.
    fn update_cursor(&mut self, end: usize) {
        if self.chunk_config.overlap() == 0 {
            self.cursor = end;
            return;
        }

        // Binary search for overlap
        let mut start = end;
        let mut low = 0;
        // Find closest index that would work
        let mut high = match self
            .next_sections
            .binary_search_by_key(&end, |(offset, str)| offset + str.len())
        {
            Ok(i) | Err(i) => i,
        };

        while low <= high {
            let mid = low + (high - low) / 2;
            let (offset, _) = self.next_sections[mid];
            let chunk_size = self.chunk_sizer.check_capacity(
                offset,
                self.text.get(offset..end).expect("Invalid range"),
                true,
            );

            // We got further than the last one, so update start
            if chunk_size.fits().is_le() && offset < start && offset > self.cursor {
                start = offset;
            }

            // Adjust search area
            if chunk_size.fits().is_lt() && mid > 0 {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }

        self.cursor = start;
    }

    /// Find the ideal next sections, breaking it up until we find the largest chunk.
    /// Increasing length of chunk until we find biggest size to minimize validation time
    /// on huge chunks
    fn update_next_sections(&mut self) {
        // First thing, clear out the list, but reuse the allocated memory
        self.next_sections.clear();

        let remaining_text = self.text.get(self.cursor..).unwrap();

        let (semantic_level, max_encoded_offset) = self.chunk_sizer.find_correct_level(
            self.cursor,
            self.semantic_split
                .levels_in_remaining_text(self.cursor)
                .filter_map(|level| {
                    self.semantic_split
                        .semantic_chunks(self.cursor, remaining_text, level)
                        .next()
                        .map(|(_, str)| (level, str))
                }),
        );

        if let Some(semantic_level) = semantic_level {
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
        } else {
            let (semantic_level, fallback_max_encoded_offset) =
                self.chunk_sizer.find_correct_level(
                    self.cursor,
                    FallbackLevel::iter().filter_map(|level| {
                        level
                            .sections(remaining_text)
                            .next()
                            .map(|(_, str)| (level, str))
                    }),
                );

            let max_encoded_offset = match (fallback_max_encoded_offset, max_encoded_offset) {
                (Some(fallback), Some(max)) => Some(fallback.min(max)),
                (fallback, max) => fallback.or(max),
            };

            let sections = semantic_level
                .unwrap_or(FallbackLevel::Char)
                .sections(remaining_text)
                .map(|(offset, text)| (self.cursor + offset, text))
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
}

impl<'sizer, 'text: 'sizer, Sizer, Level> Iterator for TextChunks<'text, 'sizer, Sizer, Level>
where
    Sizer: ChunkSizer,
    Level: SemanticLevel,
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
                c => {
                    let item_end = c.0 + c.1.len();
                    // Skip because we've emitted a chunk whose content we've already emitted
                    if item_end <= self.prev_item_end {
                        continue;
                    }
                    self.prev_item_end = item_end;
                    return Some(c);
                }
            }
        }
    }
}
