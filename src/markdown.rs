/*!
# [`MarkdownSplitter`]
Semantic splitting of Markdown documents. Tries to use as many semantic units from Markdown
as possible, eventually falling back to the normal [`TextSplitter`] method.
*/

use std::ops::Range;

use auto_enums::auto_enum;
use pulldown_cmark::{Event, Parser};
use unicode_segmentation::UnicodeSegmentation;

use crate::{
    split_str_by_separator, Characters, ChunkCapacity, ChunkSizer, SemanticSplit, TextChunks,
};

/// Markdown splitter. Recursively splits chunks into the largest
/// semantic units that fit within the chunk size. Also will
/// attempt to merge neighboring chunks if they can fit within the
/// given chunk size.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct MarkdownSplitter<S>
where
    S: ChunkSizer,
{
    /// Method of determining chunk sizes.
    chunk_sizer: S,
    /// Whether or not all chunks should have whitespace trimmed.
    /// If `false`, joining all chunks should return the original string.
    /// If `true`, all chunks will have whitespace removed from beginning and end.
    trim_chunks: bool,
}

impl Default for MarkdownSplitter<Characters> {
    fn default() -> Self {
        Self::new(Characters)
    }
}

impl<S> MarkdownSplitter<S>
where
    S: ChunkSizer,
{
    /// Creates a new [`MarkdownSplitter`].
    ///
    /// ```
    /// use text_splitter::{Characters, MarkdownSplitter};
    ///
    /// // Characters is the default, so you can also do `MarkdownSplitter::default()`
    /// let splitter = MarkdownSplitter::new(Characters);
    /// ```
    #[must_use]
    pub fn new(chunk_sizer: S) -> Self {
        Self {
            chunk_sizer,
            trim_chunks: false,
        }
    }

    /// Specify whether chunks should have whitespace trimmed from the
    /// beginning and end or not.
    ///
    /// If `false` (default), joining all chunks should return the original
    /// string.
    /// If `true`, all chunks will have whitespace removed from beginning and end.
    ///
    /// ```
    /// use text_splitter::{Characters, MarkdownSplitter};
    ///
    /// let splitter = MarkdownSplitter::default().with_trim_chunks(true);
    /// ```
    #[must_use]
    pub fn with_trim_chunks(mut self, trim_chunks: bool) -> Self {
        self.trim_chunks = trim_chunks;
        self
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// ## Method
    ///
    /// To preserve as much semantic meaning within a chunk as possible, a
    /// recursive approach is used, starting at larger semantic units and, if that is
    /// too large, breaking it up into the next largest unit. Here is an example of the
    /// steps used:
    ///
    /// 1. Split the text by a given level
    /// 2. For each section, does it fit within the chunk size?
    ///   a. Yes. Merge as many of these neighboring sections into a chunk as possible to maximize chunk length.
    ///   b. No. Split by the next level and repeat.
    ///
    /// The boundaries used to split the text if using the top-level `chunks` method, in descending length:
    ///
    /// 1. [Headings](https://spec.commonmark.org/0.30/#atx-headings) - in descending levels
    /// 2. [Thematic Breaks](https://spec.commonmark.org/0.30/#thematic-break)
    /// 3. Progress through the `TextSplitter::chunks` method.
    ///
    /// ```
    /// use text_splitter::{Characters, MarkdownSplitter};
    ///
    /// let splitter = MarkdownSplitter::default();
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunks(text, 10).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec!["Some text", "\n\nfrom a\n", "document"], chunks);
    /// ```
    pub fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: impl ChunkCapacity + 'splitter,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        self.chunk_indices(text, chunk_capacity).map(|(_, t)| t)
    }

    /// Returns an iterator over chunks of the text and their byte offsets.
    /// Each chunk will be up to the `max_chunk_size`.
    ///
    /// See [`MarkdownSplitter::chunks`] for more information.
    ///
    /// ```
    /// use text_splitter::{Characters, MarkdownSplitter};
    ///
    /// let splitter = MarkdownSplitter::default();
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_indices(text, 10).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![(0, "Some text"), (9, "\n\nfrom a\n"), (18, "document")], chunks);
    pub fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: impl ChunkCapacity + 'splitter,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        TextChunks::<_, S, Markdown>::new(chunk_capacity, &self.chunk_sizer, text, self.trim_chunks)
    }
}

/// Different semantic levels that text can be split by.
/// Each level provides a method of splitting text into chunks of a given level
/// as well as a fallback in case a given fallback is too large.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum SemanticLevel {
    /// Split by individual chars. May be larger than a single byte,
    /// but we don't go lower so we always have valid UTF str's.
    Char,
    /// Split by [unicode grapheme clusters](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)    Grapheme,
    /// Falls back to [`Self::Char`]
    GraphemeCluster,
    /// Split by [unicode words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// Falls back to [`Self::GraphemeCluster`]
    Word,
    /// Split by [unicode sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// Falls back to [`Self::Word`]
    Sentence,
    /// thematic break/horizontal rule
    Rule,
}

/// Captures information about markdown structure for a given text, and their
/// various semantic levels.
#[derive(Debug)]
struct Markdown {
    /// Range of each semantic markdown item and its precalculated semantic level
    ranges: Vec<(SemanticLevel, Range<usize>)>,
    /// Maximum element level in a given text
    max_level: SemanticLevel,
}

impl SemanticSplit for Markdown {
    type Level = SemanticLevel;

    const PERSISTENT_LEVELS: &'static [Self::Level] = &[
        SemanticLevel::Char,
        SemanticLevel::GraphemeCluster,
        SemanticLevel::Word,
        SemanticLevel::Sentence,
    ];

    fn new(text: &str) -> Self {
        let ranges = Parser::new(text)
            .into_offset_iter()
            .filter_map(|(event, range)| match event {
                Event::Rule => Some((SemanticLevel::Rule, range)),
                _ => None,
            })
            .collect::<Vec<_>>();

        let max_level = ranges
            .iter()
            .map(|(level, _)| level)
            .max()
            .copied()
            .unwrap_or(SemanticLevel::Sentence);

        Self { ranges, max_level }
    }

    fn ranges(&self) -> impl Iterator<Item = &(Self::Level, Range<usize>)> + '_ {
        self.ranges.iter()
    }

    fn max_level(&self) -> Self::Level {
        self.max_level
    }

    /// Split a given text into iterator over each semantic chunk
    #[auto_enum(Iterator)]
    fn semantic_chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        offset: usize,
        text: &'text str,
        semantic_level: Self::Level,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        match semantic_level {
            SemanticLevel::Char => text.char_indices().map(move |(i, c)| {
                (
                    offset + i,
                    text.get(i..i + c.len_utf8()).expect("char should be valid"),
                )
            }),
            SemanticLevel::GraphemeCluster => text
                .grapheme_indices(true)
                .map(move |(i, str)| (offset + i, str)),
            SemanticLevel::Word => text
                .split_word_bound_indices()
                .map(move |(i, str)| (offset + i, str)),
            SemanticLevel::Sentence => text
                .split_sentence_bound_indices()
                .map(move |(i, str)| (offset + i, str)),
            SemanticLevel::Rule => split_str_by_separator(
                text,
                self.ranges_after_offset(offset, semantic_level)
                    .map(move |(_, sep)| sep.start - offset..sep.end - offset),
            )
            .map(move |(i, str)| (offset + i, str)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_markdown_separators() {
        let markdown = Markdown::new("Some text without any markdown separators");

        assert_eq!(
            Vec::<&(SemanticLevel, Range<usize>)>::new(),
            markdown.ranges().collect::<Vec<_>>()
        );
        assert_eq!(SemanticLevel::Sentence, markdown.max_level());
    }

    #[test]
    fn test_with_rule() {
        let markdown = Markdown::new("Some text\n\n---\n\nwith a rule");

        assert_eq!(
            vec![&(SemanticLevel::Rule, 11..15)],
            markdown.ranges().collect::<Vec<_>>()
        );
        assert_eq!(SemanticLevel::Rule, markdown.max_level());
    }
}
