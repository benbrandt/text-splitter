use std::{
    cell::Cell, cmp::Ordering, collections::BTreeMap, convert::Infallible, fmt, iter::once,
    ops::Range,
};

use either::Either;
use itertools::Itertools;
use strum::IntoEnumIterator;

use self::fallback::{FallbackLevel, SentenceProbe, WordProbe};
use crate::{
    chunk_size::MemoizedChunkSizer, trim::Trim, ChunkCapacity, ChunkConfig, TryChunkSizer,
};

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

fn unwrap_infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

/// Shared interface for splitters that can generate chunks of text based on the
/// associated semantic level.
trait Splitter<Sizer>
where
    Sizer: TryChunkSizer,
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
    fn try_chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = Result<(usize, &'text str), Sizer::Error>> + 'splitter
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
    fn try_chunk_char_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = Result<ChunkCharIndex<'text>, Sizer::Error>> + 'splitter
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
    fn try_chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = Result<&'text str, Sizer::Error>> + 'splitter
    where
        Sizer: 'splitter,
    {
        self.try_chunk_indices(text)
            .map(|chunk| chunk.map(|(_, t)| t))
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
    /// First index at or after `cursor` for each level that still has ranges.
    next_range_by_level: BTreeMap<Level, usize>,
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
        let mut next_range_by_level = BTreeMap::new();
        for (index, (level, _)) in ranges.iter().enumerate() {
            next_range_by_level.entry(*level).or_insert(index);
        }
        Self {
            cursor: 0,
            ranges,
            next_range_by_level,
        }
    }

    /// Find the first range of this exact level. At the cursor, prefer the shortest
    /// of consecutive ranges with the same level and start.
    fn first_level_range(&self, level: Level) -> Option<&Range<usize>> {
        let &index = self.next_range_by_level.get(&level)?;
        let first = &self.ranges[index].1;
        if index != self.cursor {
            return Some(first);
        }
        Some(
            self.ranges[index + 1..]
                .iter()
                .take_while(|(l, r)| *l == level && r.start == first.start)
                .last()
                .map_or(first, |(_, r)| r),
        )
    }

    /// Retrieve remaining ranges at this level or higher, skipping initial ranges
    /// that would hide the first item of the requested level.
    fn level_ranges(&self, level: Level) -> impl Iterator<Item = (Level, Range<usize>)> + '_ {
        let first = self.first_level_range(level);
        // Only search the index if the first remaining range is below the requested level.
        let start = match self.ranges.get(self.cursor) {
            Some((current, _)) if *current < level => self
                .next_range_by_level
                .range(level..)
                .map(|(_, &index)| index)
                .min()
                .unwrap_or(self.ranges.len()),
            _ => self.cursor,
        };
        self.ranges[start..]
            .iter()
            .filter(move |(l, _)| *l >= level)
            .skip_while(move |(l, r)| {
                first.is_some_and(|first| {
                    (*l > level && r.contains(&first.start))
                        || (*l == level && r.start == first.start && r.end > first.end)
                })
            })
            .cloned()
    }

    /// Return the distinct levels with a range at or after the cursor, in ascending order.
    fn levels_in_remaining_text(&self) -> impl Iterator<Item = Level> + '_ {
        self.next_range_by_level.keys().copied()
    }

    /// Split the suffix beginning at the byte offset last passed to `update_cursor`.
    fn semantic_chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        offset: usize,
        text: &'text str,
        semantic_level: Level,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        Level::sections(
            text,
            self.level_ranges(semantic_level)
                .map(move |(l, sep)| (l, sep.start - offset..sep.end - offset)),
        )
        .map(move |(i, str)| (offset + i, str))
    }

    /// Advance to the first range starting at or after the byte offset.
    fn update_cursor(&mut self, offset: usize) {
        let skipped = self.ranges[self.cursor..]
            .iter()
            .take_while(|(_, range)| range.start < offset)
            .count();
        if skipped == 0 {
            return;
        }
        self.cursor += skipped;
        let remaining = &self.ranges[self.cursor..];
        self.next_range_by_level.retain(|level, index| {
            // Only search again once the cursor passes the cached range.
            if *index < self.cursor {
                let Some(next) = remaining.iter().position(|(l, _)| l == level) else {
                    return false;
                };
                *index = self.cursor + next;
            }
            true
        });
    }
}

/// Returns chunks of text with their byte offsets as an iterator.
#[derive(Debug)]
struct TextChunks<'text, 'sizer, Sizer, Level>
where
    Sizer: TryChunkSizer,
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
    /// Avoid searching for the same distant fallback sentence boundary on every chunk.
    sentence_probe: SentenceProbe,
    /// Avoid segmenting the same long run without word breaks on every chunk.
    word_probe: WordProbe,
    /// Original text to iterate over and generate chunks from
    text: &'text str,
    /// The trimming method to apply
    trim: Trim,
    /// Whether iteration has ended, including after a sizing error.
    finished: bool,
}

impl<'sizer, 'text: 'sizer, Sizer, Level> TextChunks<'text, 'sizer, Sizer, Level>
where
    Sizer: TryChunkSizer,
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
        let trim = if *trim_enabled { trim } else { Trim::None };
        #[cfg(any(feature = "markdown", feature = "code"))]
        let trim = trim.for_text(text);
        Self {
            capacity: *capacity,
            chunk_sizer: MemoizedChunkSizer::new(sizer),
            chunk_stats: ChunkStats::new(),
            cursor: 0,
            next_sections: Vec::new(),
            overlap: (*overlap).into(),
            prev_item_end: 0,
            semantic_split: SemanticSplitRanges::new(offsets),
            sentence_probe: SentenceProbe::default(),
            word_probe: WordProbe::default(),
            text,
            trim,
            finished: false,
        }
    }

    /// Generate the next chunk, applying trimming settings.
    /// Returns final byte offset and str.
    fn next_chunk(&mut self) -> Result<(usize, &'text str), Sizer::Error> {
        self.semantic_split.update_cursor(self.cursor);
        let start = self.cursor;
        let end = if let Some(end) = self.find_chunk_end()? {
            end
        } else {
            // The sizer disproved the byte-length estimate. Keep cached sizes
            // and retry at the same cursor, measuring levels from now on.
            self.chunk_sizer.require_measured_levels();
            self.find_chunk_end()?
                .expect("a measured first section fits or is one character")
        };
        // Chunk selection uses offsets from slices of this same text, so both
        // endpoints are valid UTF-8 boundaries.
        let chunk = &self.text[start..end];
        self.chunk_stats.update_max_chunk_size(end - start);

        // Reset caches so we can reuse the memory allocation
        self.chunk_sizer.clear_cache();
        // Optionally move cursor back if overlap is desired
        self.update_cursor(end)?;

        // Trim whitespace if user requested it
        Ok(self.trim.trim(start, chunk))
    }

    /// Select a level and search for a fitting chunk end. `None` means the first
    /// section needs to be subdivided instead of emitted over capacity.
    fn find_chunk_end(&mut self) -> Result<Option<usize>, Sizer::Error> {
        let mut low = self.update_next_sections()?;
        let start = self.cursor;
        let mut end = self.cursor;
        let mut equals_found = false;
        let mut high = self.next_sections.len().saturating_sub(1);
        let mut lookahead_from = None;

        while low <= high {
            let mid = low + (high - low) / 2;
            let (offset, str) = self.next_sections[mid];
            let text_end = offset + str.len();
            let chunk = &self.text[start..text_end];
            let chunk_size = self.chunk_sizer.chunk_size(start, chunk, self.trim)?;
            let fits = self.capacity.fits(chunk_size);

            match fits {
                Ordering::Less => {
                    // We got further than the last one, so update end
                    if text_end > end {
                        end = text_end;
                        lookahead_from = Some((mid, chunk_size));
                    }
                }
                Ordering::Equal => {
                    // If we found a smaller equals use it. Or if this is the first equals we found
                    if text_end < end || !equals_found {
                        end = text_end;
                        lookahead_from = Some((mid, chunk_size));
                    }
                    equals_found = true;
                }
                Ordering::Greater => {
                    if mid == 0 && start == end {
                        // Don't accept this prefix. A longer one can still
                        // tokenize to a smaller size in the lookahead below.
                        lookahead_from = Some((mid, chunk_size));
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

        if let Some((index, chunk_size)) = lookahead_from {
            for &(offset, str) in &self.next_sections[index + 1..] {
                let text_end = offset + str.len();
                let chunk = &self.text[start..text_end];
                let size = self.chunk_sizer.chunk_size(start, chunk, self.trim)?;
                if size <= chunk_size {
                    if size <= self.capacity.max && text_end > end {
                        end = text_end;
                    }
                } else {
                    break;
                }
            }
        }

        if end > start {
            return Ok(Some(end));
        }
        // Only a single Unicode scalar is indivisible, even if the sizer
        // charges the same (over-budget) amount for several scalars.
        let (offset, first) = self.next_sections[0];
        Ok(first
            .chars()
            .nth(1)
            .is_none()
            .then_some(offset + first.len()))
    }

    /// Use binary search to find the sections that fit within the overlap size.
    /// If no overlap deisired, return end.
    fn update_cursor(&mut self, end: usize) -> Result<(), Sizer::Error> {
        if self.overlap.max == 0 {
            self.cursor = end;
            return Ok(());
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
            )?;
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
        Ok(())
    }

    /// Find the ideal next sections, breaking it up until we find the largest chunk.
    /// Increasing length of chunk until we find biggest size to minimize validation time
    /// on huge chunks
    #[expect(clippy::too_many_lines)]
    fn update_next_sections(&mut self) -> Result<usize, Sizer::Error> {
        // First thing, clear out the list, but reuse the allocated memory
        self.next_sections.clear();

        let remaining_text = self.text.get(self.cursor..).unwrap();
        let mut lower_level = None;

        let (semantic_level, mut max_offset) = self.chunk_sizer.find_correct_level(
            self.cursor,
            &self.capacity,
            self.semantic_split
                .levels_in_remaining_text()
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
                    || Either::Left(std::iter::empty()),
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
        )?;

        let sections = if let Some(semantic_level) = semantic_level {
            Either::Left(self.semantic_split.semantic_chunks(
                self.cursor,
                remaining_text,
                semantic_level,
            ))
        } else {
            // The next word consumer at this cursor can use the iterator from level selection.
            let probed_words = Cell::new(None);
            let sections_for = |level: FallbackLevel| {
                if level == FallbackLevel::Word {
                    if let Some(words) = probed_words.take() {
                        return Either::Left(words);
                    }
                }
                Either::Right(level.sections(remaining_text))
            };
            let (semantic_level, fallback_max_offset) = self.chunk_sizer.find_correct_level(
                self.cursor,
                &self.capacity,
                FallbackLevel::iter().filter_map(|level| {
                    let first = match level {
                        FallbackLevel::Sentence => {
                            self.sentence_probe.first(self.text, self.cursor)
                        }
                        FallbackLevel::Word => {
                            let mut words = level.sections(remaining_text).peekable();
                            let first =
                                self.word_probe.first(self.text, self.cursor, &mut words)?;
                            // On a miss, skip the buffered first word. On a cache hit,
                            // ICU won't be asked for any boundaries unless we need a second word.
                            probed_words.set(Some(once((0, first)).chain(words.skip(1))));
                            Some(first)
                        }
                        _ => level.sections(remaining_text).next().map(|(_, s)| s),
                    };
                    first.map(|s| (level, s, level.boundary_level_for_probe()))
                }),
                |lower_level, chunk_end| {
                    // If sizing never asks for a boundary, leave the words for output.
                    lower_level
                        .into_iter()
                        .flat_map(&sections_for)
                        .map(|(offset, text)| self.cursor + offset + text.len())
                        .take_while(move |end| *end <= chunk_end)
                },
                self.trim,
            )?;

            max_offset = match (fallback_max_offset, max_offset) {
                (Some(fallback), Some(max)) => Some(fallback.min(max)),
                (fallback, max) => fallback.or(max),
            };

            let fallback_level = semantic_level.unwrap_or(FallbackLevel::Char);

            Either::Right(
                sections_for(fallback_level).map(|(offset, text)| (self.cursor + offset, text)),
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
                    self.cursor,
                    self.text.get(self.cursor..text_end).expect("Invalid range"),
                    self.trim,
                )?;
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

        Ok(low)
    }
}

impl<'sizer, 'text: 'sizer, Sizer, Level> Iterator for TextChunks<'text, 'sizer, Sizer, Level>
where
    Sizer: TryChunkSizer,
    Level: SemanticLevel,
{
    type Item = Result<(usize, &'text str), Sizer::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            // Make sure we haven't reached the end
            if self.cursor >= self.text.len() {
                self.finished = true;
                return None;
            }

            match self.next_chunk() {
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
                // Make sure we didn't get an empty chunk. Should only happen in
                // cases where we trim.
                Ok((_, "")) => {}
                Ok(c) => {
                    let item_end = c.0 + c.1.len();
                    // Skip because we've emitted a chunk whose content we've already emitted
                    if item_end <= self.prev_item_end {
                        continue;
                    }
                    self.prev_item_end = item_end;
                    return Some(Ok(c));
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
    Sizer: TryChunkSizer,
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
    Sizer: TryChunkSizer,
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
    Sizer: TryChunkSizer,
    Level: SemanticLevel,
{
    type Item = Result<ChunkCharIndex<'text>, Sizer::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.text_chunks.next().map(|result| {
            let (byte_offset, chunk) = result?;
            let preceding_text = self
                .text
                .get(self.byte_offset..byte_offset)
                .expect("Invalid byte sequence");
            self.byte_offset = byte_offset;
            self.char_offset += preceding_text.chars().count();
            Ok(ChunkCharIndex {
                chunk,
                byte_offset,
                char_offset: self.char_offset,
            })
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

        assert_eq!(ranges.level_ranges(0).collect::<Vec<_>>(), [(0, 1..2)]);
    }

    #[test]
    fn level_ranges_keep_earlier_higher_boundaries() {
        let mut ranges =
            SemanticSplitRanges::new(vec![(0, 0..1), (3, 2..3), (2, 5..10), (1, 5..6)]);

        // The same boundary is found whether it is ahead of the cursor or right at it.
        for offset in [0, 2] {
            ranges.update_cursor(offset);
            assert_eq!(
                ranges.level_ranges(1).collect::<Vec<_>>(),
                [(3, 2..3), (2, 5..10), (1, 5..6)]
            );
        }
        // Once the earlier boundary is gone, skip the parent that wraps the first match.
        ranges.update_cursor(4);
        assert_eq!(ranges.level_ranges(1).collect::<Vec<_>>(), [(1, 5..6)]);
        ranges.update_cursor(10);
        assert_eq!(ranges.level_ranges(1).next(), None);
    }

    #[test]
    fn level_ranges_prefer_the_shortest_shared_start_only_at_the_cursor() {
        let mut ranges = SemanticSplitRanges::new(vec![(0, 0..1), (1, 2..8), (1, 2..4), (1, 2..3)]);

        assert_eq!(ranges.level_ranges(1).next(), Some((1, 2..8)));
        ranges.update_cursor(2);
        assert_eq!(ranges.level_ranges(1).next(), Some((1, 2..3)));
    }

    #[test]
    fn remaining_levels_follow_the_range_cursor() {
        let mut ranges =
            SemanticSplitRanges::new(vec![(1, 30..31), (1, 10..11), (0, 20..21), (2, 10..50)]);

        for (offset, expected) in [(10, &[0, 1, 2][..]), (11, &[0, 1]), (30, &[1]), (31, &[])] {
            ranges.update_cursor(offset);
            assert_eq!(
                ranges.levels_in_remaining_text().collect::<Vec<_>>(),
                expected,
                "offset {offset}"
            );
        }
    }
}
