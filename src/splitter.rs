use std::{cmp::Ordering, fmt, iter::once, ops::Range};

use either::Either;
use itertools::Itertools;
use strum::IntoEnumIterator;

use self::fallback::FallbackLevel;
use crate::{chunk_size::MemoizedChunkSizer, trim::Trim, ChunkCapacity, ChunkConfig, ChunkSizer};

#[cfg(feature = "code")]
mod code;
mod fallback;
#[cfg(feature = "markdown")]
mod markdown;
mod text;

#[cfg(feature = "code")]
pub use code::{CodeSplitter, CodeSplitterError};
#[cfg(feature = "markdown")]
pub use markdown::MarkdownSplitter;
pub use text::TextSplitter;

/// Initial fallback segmentation window, in bytes per unit of chunk capacity.
const FALLBACK_WINDOW_FACTOR: usize = 8;
/// Smallest fallback segmentation window, in bytes.
const MIN_FALLBACK_WINDOW: usize = 1024;

/// Shared interface for splitters that can generate chunks of text based on the
/// associated semantic level.
trait Splitter<Sizer>
where
    Sizer: ChunkSizer,
{
    type Level: SemanticLevel;

    /// Trimming behavior to use when trimming chunks
    const TRIM: Trim = Trim::All;

    /// Retrieve the splitter chunk configuration
    fn chunk_config(&self) -> &ChunkConfig<Sizer>;

    /// Generate a list of offsets for each semantic level within the text.
    fn parse(&self, text: &str) -> Vec<(Self::Level, Range<usize>)>;

    /// Returns an iterator over chunks of the text and their byte offsets.
    /// Each chunk will be up to the max size of the `ChunkConfig`.
    fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter
    where
        Sizer: 'splitter,
    {
        TextChunks::<Sizer, Self::Level>::new(
            self.chunk_config(),
            text,
            self.parse(text),
            Self::TRIM,
        )
    }

    /// Returns an iterator over chunks of the text and their byte and character offsets.
    /// Each chunk will be up to the max size of the `ChunkConfig`.
    ///
    /// This will be more expensive than just byte offsets, and for most usage in Rust, just having byte offsets is sufficient.
    /// But when interfacing with other languages or systems that require character offsets, this will track the character offsets
    /// for you, accounting for any trimming that may have occurred.
    fn chunk_char_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = ChunkCharIndex<'text>> + 'splitter
    where
        Sizer: 'splitter,
    {
        TextChunksWithCharIndices::<Sizer, Self::Level>::new(
            self.chunk_config(),
            text,
            self.parse(text),
            Self::TRIM,
        )
    }

    /// Generate a list of chunks from a given text.
    /// Each chunk will be up to the max size of the `ChunkConfig`.
    fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = &'text str> + 'splitter
    where
        Sizer: 'splitter,
    {
        self.chunk_indices(text).map(|(_, t)| t)
    }
}

/// Custom-defined levels of semantic splitting for custom document types.
trait SemanticLevel: Copy + fmt::Debug + Ord + PartialOrd + 'static {
    /// Given a level, split the text into sections based on the level.
    /// Level ranges are also provided of items that are equal to or greater than the current level.
    /// Default implementation assumes that all level ranges should be treated
    /// as their own item.
    fn sections(
        text: &str,
        level_ranges: impl Iterator<Item = (Self, Range<usize>)>,
    ) -> impl Iterator<Item = (usize, &str)> {
        let mut cursor = 0;
        let mut final_match = false;
        level_ranges
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
                        Some((_, range)) => {
                            if range.start < cursor {
                                continue;
                            }
                            let offset = cursor;
                            let prev_section = text
                                .get(offset..range.start)
                                .expect("invalid character sequence");
                            let separator = text
                                .get(range.start..range.end)
                                .expect("invalid character sequence");
                            cursor = range.end;
                            return Some(Either::Right(
                                [(offset, prev_section), (range.start, separator)].into_iter(),
                            ));
                        }
                    }
                }
            })
            .flatten()
            .filter(|(_, s)| !s.is_empty())
    }
}

/// Captures information about document structure for a given text, and their
/// various semantic levels
#[derive(Debug)]
struct SemanticSplitRanges<Level>
where
    Level: SemanticLevel,
{
    /// Current cursor in the ranges list, so that we can skip over items we've
    /// already processed.
    cursor: usize,
    /// Range of each semantic item and its precalculated semantic level
    ranges: Vec<(Level, Range<usize>)>,
}

impl<Level> SemanticSplitRanges<Level>
where
    Level: SemanticLevel,
{
    fn new(mut ranges: Vec<(Level, Range<usize>)>) -> Self {
        // Sort by start. If start is equal, sort by end in reverse order, so larger ranges come first.
        ranges.sort_unstable_by(|(_, a), (_, b)| {
            a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end))
        });
        Self { cursor: 0, ranges }
    }

    /// Retrieve ranges for all sections of a given level after an offset
    fn ranges_after_offset(
        &self,
        offset: usize,
    ) -> impl Iterator<Item = (Level, Range<usize>)> + '_ {
        self.ranges[self.cursor..]
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
        Level::sections(
            text,
            self.level_ranges_after_offset(offset, semantic_level)
                .map(move |(l, sep)| (l, sep.start - offset..sep.end - offset)),
        )
        .map(move |(i, str)| (offset + i, str))
    }

    /// Clear out ranges we have moved past so future iterations are faster
    fn update_cursor(&mut self, cursor: usize) {
        self.cursor += self.ranges[self.cursor..]
            .iter()
            .position(|(_, range)| range.start >= cursor)
            .unwrap_or_else(|| self.ranges.len() - self.cursor);
    }
}

/// Returns chunks of text with their byte offsets as an iterator.
#[derive(Debug)]
struct TextChunks<'text, 'sizer, Sizer, Level>
where
    Sizer: ChunkSizer,
    Level: SemanticLevel,
{
    /// Overal capacity of the chunk
    capacity: ChunkCapacity,
    /// How to validate chunk sizes
    chunk_sizer: MemoizedChunkSizer<'sizer, Sizer>,
    /// Average number of sections in a chunk for each level
    chunk_stats: ChunkStats,
    /// Current byte offset in the `text`
    cursor: usize,
    /// Reusable container for next sections to avoid extra allocations
    next_sections: Vec<(usize, &'text str)>,
    /// Overlap capacity
    overlap: ChunkCapacity,
    /// Previous item's end byte offset
    prev_item_end: usize,
    /// Splitter used for determining semantic levels.
    semantic_split: SemanticSplitRanges<Level>,
    /// Original text to iterate over and generate chunks from
    text: &'text str,
    /// The trimming method to apply
    trim: Trim,
}

impl<'sizer, 'text: 'sizer, Sizer, Level> TextChunks<'text, 'sizer, Sizer, Level>
where
    Sizer: ChunkSizer,
    Level: SemanticLevel,
{
    /// Generate new [`TextChunks`] iterator for a given text.
    /// Starts with an offset of 0
    fn new(
        chunk_config: &'sizer ChunkConfig<Sizer>,
        text: &'text str,
        offsets: Vec<(Level, Range<usize>)>,
        trim: Trim,
    ) -> Self {
        let ChunkConfig {
            capacity,
            overlap,
            sizer,
            trim: trim_enabled,
        } = chunk_config;
        Self {
            capacity: *capacity,
            chunk_sizer: MemoizedChunkSizer::new(sizer),
            chunk_stats: ChunkStats::new(),
            cursor: 0,
            next_sections: Vec::new(),
            overlap: (*overlap).into(),
            prev_item_end: 0,
            semantic_split: SemanticSplitRanges::new(offsets),
            text,
            trim: if *trim_enabled { trim } else { Trim::None },
        }
    }

    /// Generate the next chunk, applying trimming settings.
    /// Returns final byte offset and str.
    /// Will return `None` if given an invalid range.
    fn next_chunk(&mut self) -> Option<(usize, &'text str)> {
        self.semantic_split.update_cursor(self.cursor);
        let low = self.update_next_sections();
        let (start, end) = self.binary_search_next_chunk(low)?;
        let chunk = self.text.get(start..end)?;
        self.chunk_stats.update_max_chunk_size(end - start);

        // Reset caches so we can reuse the memory allocation
        self.chunk_sizer.clear_cache();
        // Optionally move cursor back if overlap is desired
        self.update_cursor(end);

        // Trim whitespace if user requested it
        Some(self.trim.trim(start, chunk))
    }

    /// Use binary search to find the next chunk that fits within the chunk size
    fn binary_search_next_chunk(&mut self, mut low: usize) -> Option<(usize, usize)> {
        let start = self.cursor;
        let mut end = self.cursor;
        let mut equals_found = false;
        let mut high = self.next_sections.len().saturating_sub(1);
        let mut successful_index = None;
        let mut successful_chunk_size = None;

        while low <= high {
            let mid = low + (high - low) / 2;
            let (offset, str) = self.next_sections[mid];
            let text_end = offset + str.len();
            let chunk = self.text.get(start..text_end)?;
            let chunk_size = self.chunk_sizer.chunk_size(start, chunk, self.trim);
            let fits = self.capacity.fits(chunk_size);

            match fits {
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
            }

            // Adjust search area
            if fits.is_lt() {
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
                let size = self.chunk_sizer.chunk_size(start, chunk, self.trim);
                if size <= chunk_size {
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
        if self.overlap.max == 0 {
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
            let chunk_size = self.chunk_sizer.chunk_size(
                offset,
                self.text.get(offset..end).expect("Invalid range"),
                self.trim,
            );
            let fits = self.overlap.fits(chunk_size);

            // We got further than the last one, so update start
            if fits.is_le() && offset < start && offset > self.cursor {
                start = offset;
            }

            // Adjust search area
            if fits.is_lt() && mid > 0 {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }

        self.cursor = start;
    }

    /// First section of each fallback level, for `find_correct_level`.
    ///
    /// Each level is segmented within a byte window. A level with no boundary
    /// in the window is reported with the window prefix instead of its full
    /// first section. The window doubles until either no level is truncated, or
    /// the prefix is too big for the chunk. In the second case each truncated
    /// section contains that prefix, so it is too big as well. Every chunk must
    /// then end inside the window, so the exact section ends cannot change the
    /// chunk.
    fn fallback_first_sections(
        &mut self,
        remaining_text: &'text str,
    ) -> Vec<(FallbackLevel, &'text str)> {
        let mut window = self
            .capacity
            .max
            .saturating_mul(FALLBACK_WINDOW_FACTOR)
            .max(MIN_FALLBACK_WINDOW);

        loop {
            let mut truncated_prefix = None;
            let sections = FallbackLevel::iter()
                .filter_map(|level| {
                    let (str, truncated) = level.first_section_within(remaining_text, window)?;
                    if truncated {
                        truncated_prefix = Some(str);
                    }
                    Some((level, str))
                })
                .collect::<Vec<_>>();

            let Some(prefix) = truncated_prefix else {
                return sections;
            };
            let size = self.chunk_sizer.chunk_size(self.cursor, prefix, self.trim);
            if self.capacity.fits(size).is_gt() {
                return sections;
            }
            window = window.saturating_mul(2);
        }
    }

    /// Find the ideal next sections, breaking it up until we find the largest chunk.
    /// Increasing length of chunk until we find biggest size to minimize validation time
    /// on huge chunks
    #[expect(clippy::too_many_lines)]
    fn update_next_sections(&mut self) -> usize {
        // First thing, clear out the list, but reuse the allocated memory
        self.next_sections.clear();

        let remaining_text = self.text.get(self.cursor..).unwrap();
        let mut lower_level = None;

        let (semantic_level, mut max_offset) = self.chunk_sizer.find_correct_level(
            self.cursor,
            &self.capacity,
            self.semantic_split
                .levels_in_remaining_text(self.cursor)
                .filter_map(|level| {
                    let first_chunk = self
                        .semantic_split
                        .semantic_chunks(self.cursor, remaining_text, level)
                        .next();

                    let result = first_chunk.map(|(_, str)| {
                        let candidate_lower_level = lower_level;
                        (level, str, candidate_lower_level)
                    });

                    lower_level = Some(level);
                    result
                }),
            |lower_level, chunk_end| {
                lower_level.map_or_else(
                    // The lowest semantic level has no smaller level to probe with.
                    // Probe with words for fixed capacities, so a long section
                    // is not sized whole. Keep the original behavior for ranges:
                    // nonmonotone sizers can select a different chunk otherwise.
                    || {
                        let cursor = self.cursor;
                        Either::Left(
                            (self.capacity.desired == self.capacity.max)
                                .then(|| {
                                    FallbackLevel::Word
                                        .sections(remaining_text)
                                        .map(move |(offset, text)| cursor + offset + text.len())
                                        .take_while(move |end| *end <= chunk_end)
                                })
                                .into_iter()
                                .flatten(),
                        )
                    },
                    |lower_level| {
                        Either::Right(
                            self.semantic_split
                                .semantic_chunks(self.cursor, remaining_text, lower_level)
                                .map(|(offset, text)| offset + text.len())
                                .take_while(move |end| *end <= chunk_end),
                        )
                    },
                )
            },
            self.trim,
        );

        let sections = if let Some(semantic_level) = semantic_level {
            Either::Left(self.semantic_split.semantic_chunks(
                self.cursor,
                remaining_text,
                semantic_level,
            ))
        } else {
            // A ranged capacity may select a chunk based on the number of
            // candidate sections. Keep the original bounds for that case:
            // token sizers are not necessarily monotone across prefixes.
            let first_sections = if self.capacity.desired == self.capacity.max {
                self.fallback_first_sections(remaining_text)
            } else {
                FallbackLevel::iter()
                    .filter_map(|level| {
                        level
                            .sections(remaining_text)
                            .next()
                            .map(|(_, section)| (level, section))
                    })
                    .collect()
            };
            let (semantic_level, fallback_max_offset) = self.chunk_sizer.find_correct_level(
                self.cursor,
                &self.capacity,
                first_sections
                    .into_iter()
                    .map(|(level, str)| (level, str, level.boundary_level_for_probe())),
                |lower_level, chunk_end| {
                    lower_level.map_or_else(
                        || Either::Left(std::iter::empty()),
                        |lower_level| {
                            Either::Right(
                                lower_level
                                    .sections(remaining_text)
                                    .map(|(offset, text)| self.cursor + offset + text.len())
                                    .take_while(move |end| *end <= chunk_end),
                            )
                        },
                    )
                },
                self.trim,
            );

            max_offset = match (fallback_max_offset, max_offset) {
                (Some(fallback), Some(max)) => Some(fallback.min(max)),
                (fallback, max) => fallback.or(max),
            };

            let fallback_level = semantic_level.unwrap_or(FallbackLevel::Char);

            Either::Right(
                fallback_level
                    .sections(remaining_text)
                    .map(|(offset, text)| (self.cursor + offset, text)),
            )
        };

        let mut sections = sections
            .take_while(move |(offset, _)| max_offset.is_none_or(|max| *offset <= max))
            .filter(|(_, str)| !str.is_empty());

        // Start filling up the next sections. Since calculating the size of the chunk gets more expensive
        // the farther we go, we conservatively check for a smaller range to do the later binary search in.
        let mut low = 0;
        let mut prev_equals: Option<usize> = None;
        let max = self.capacity.max;
        let mut target_offset = self.chunk_stats.max_chunk_size.unwrap_or(max);

        loop {
            let prev_num = self.next_sections.len();
            for (offset, str) in sections.by_ref() {
                self.next_sections.push((offset, str));
                if offset + str.len() > (self.cursor.saturating_add(target_offset)) {
                    break;
                }
            }
            let new_num = self.next_sections.len();
            // If we've iterated through the whole iterator, break here.
            if new_num - prev_num == 0 {
                break;
            }

            // Check if the last item fits
            if let Some(&(offset, str)) = self.next_sections.last() {
                let text_end = offset + str.len();
                if (text_end - self.cursor) < target_offset {
                    break;
                }
                let chunk_size = self.chunk_sizer.chunk_size(
                    offset,
                    self.text.get(self.cursor..text_end).expect("Invalid range"),
                    self.trim,
                );
                let fits = self.capacity.fits(chunk_size);

                if fits.is_le() {
                    let final_offset = offset + str.len() - self.cursor;
                    let size = chunk_size.max(1);
                    let diff = (max - size).max(1);
                    let avg_size = final_offset.div_ceil(size);

                    target_offset = final_offset
                        .saturating_add(diff.saturating_mul(avg_size))
                        .saturating_add(final_offset.div_ceil(10));
                }

                match fits {
                    Ordering::Less => {
                        // We know we can go higher
                        low = new_num.saturating_sub(1);
                    }
                    Ordering::Equal => {
                        // Don't update low because it could be a range
                        // If we've seen a previous equals, we can break if the size is bigger already
                        if let Some(prev) = prev_equals {
                            if prev < chunk_size {
                                break;
                            }
                        }
                        prev_equals = Some(chunk_size);
                    }
                    Ordering::Greater => {
                        break;
                    }
                }
            }
        }

        low
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
                (_, "") => {}
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

/// Represents a chunk of text along with its byte and character offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkCharIndex<'text> {
    /// The text of the generated chunk.
    pub chunk: &'text str,
    /// The byte offset of the chunk within the original text.
    pub byte_offset: usize,
    /// The character offset of the chunk within the original text.
    pub char_offset: usize,
}

/// Returns chunks of text with their byte and character offsets as an iterator.
#[derive(Debug)]
struct TextChunksWithCharIndices<'text, 'sizer, Sizer, Level>
where
    Sizer: ChunkSizer,
    Level: SemanticLevel,
{
    /// The text being chunked.
    text: &'text str,
    /// The main iterator over chunks of text.
    text_chunks: TextChunks<'text, 'sizer, Sizer, Level>,
    /// The byte offset of the previous chunk within the original text.
    byte_offset: usize,
    /// The character offset of the previous chunk within the original text.
    char_offset: usize,
}

impl<'sizer, 'text: 'sizer, Sizer, Level> TextChunksWithCharIndices<'text, 'sizer, Sizer, Level>
where
    Sizer: ChunkSizer,
    Level: SemanticLevel,
{
    /// Generate new [`TextChunksWithCharIndices`] iterator for a given text.
    /// Starts with an offset of 0
    fn new(
        chunk_config: &'sizer ChunkConfig<Sizer>,
        text: &'text str,
        offsets: Vec<(Level, Range<usize>)>,
        trim: Trim,
    ) -> Self {
        Self {
            text,
            text_chunks: TextChunks::new(chunk_config, text, offsets, trim),
            byte_offset: 0,
            char_offset: 0,
        }
    }
}

impl<'sizer, 'text: 'sizer, Sizer, Level> Iterator
    for TextChunksWithCharIndices<'text, 'sizer, Sizer, Level>
where
    Sizer: ChunkSizer,
    Level: SemanticLevel,
{
    type Item = ChunkCharIndex<'text>;

    fn next(&mut self) -> Option<Self::Item> {
        let (byte_offset, chunk) = self.text_chunks.next()?;
        let preceding_text = self
            .text
            .get(self.byte_offset..byte_offset)
            .expect("Invalid byte sequence");
        self.byte_offset = byte_offset;
        self.char_offset += preceding_text.chars().count();
        Some(ChunkCharIndex {
            chunk,
            byte_offset,
            char_offset: self.char_offset,
        })
    }
}

/// Keeps track of the average size of chunks as we go
#[derive(Debug, Default)]
struct ChunkStats {
    /// The size of the biggest chunk we've seen, if we have seen at least one
    max_chunk_size: Option<usize>,
}

impl ChunkStats {
    fn new() -> Self {
        Self::default()
    }

    /// Update statistics after the chunk has been produced
    fn update_max_chunk_size(&mut self, size: usize) {
        self.max_chunk_size = self.max_chunk_size.map(|s| s.max(size)).or(Some(size));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CompressedBytes;

    impl ChunkSizer for CompressedBytes {
        fn size(&self, chunk: &str) -> usize {
            chunk.len().div_ceil(64)
        }
    }

    #[test]
    fn fallback_window_doubles_until_its_prefix_exceeds_capacity() {
        let text = "word ".repeat(4_000);
        let config = ChunkConfig::new(20)
            .with_sizer(CompressedBytes)
            .with_trim(false);
        let mut chunks = TextChunks::<_, usize>::new(&config, &text, vec![], Trim::None);

        let first_sections = chunks.fallback_first_sections(&text);
        let sentence = first_sections.last().unwrap().1;

        assert_eq!(sentence.len(), 2_048);
    }

    #[test]
    fn chunk_stats_empty() {
        let stats = ChunkStats::new();
        assert_eq!(stats.max_chunk_size, None);
    }

    #[test]
    fn chunk_stats_one() {
        let mut stats = ChunkStats::new();
        stats.update_max_chunk_size(10);
        assert_eq!(stats.max_chunk_size, Some(10));
    }

    #[test]
    fn chunk_stats_multiple() {
        let mut stats = ChunkStats::new();
        stats.update_max_chunk_size(10);
        stats.update_max_chunk_size(20);
        stats.update_max_chunk_size(30);
        assert_eq!(stats.max_chunk_size, Some(30));
    }

    impl SemanticLevel for usize {}

    #[test]
    fn semantic_ranges_are_sorted() {
        let ranges = SemanticSplitRanges::new(vec![(0, 0..1), (1, 0..2), (0, 1..2), (2, 0..4)]);

        assert_eq!(
            ranges.ranges,
            vec![(2, 0..4), (1, 0..2), (0, 0..1), (0, 1..2)]
        );
    }

    #[test]
    fn semantic_ranges_skip_previous_ranges() {
        let mut ranges = SemanticSplitRanges::new(vec![(0, 0..1), (1, 0..2), (0, 1..2), (2, 0..4)]);

        ranges.update_cursor(1);

        assert_eq!(
            ranges.ranges_after_offset(0).collect::<Vec<_>>(),
            vec![(0, 1..2)]
        );
    }
}
