/*!
# [`MarkdownSplitter`]
Semantic splitting of Markdown documents. Tries to use as many semantic units from Markdown
as possible, according to the Common Mark specification.
*/

use std::{iter::once, ops::Range};

use either::Either;
use itertools::Itertools;
use pulldown_cmark::{Event, Options, Parser, Tag};

use crate::{trim::Trim, ChunkConfig, ChunkSizer, SemanticLevel, TextChunks};

/// Markdown splitter. Recursively splits chunks into the largest
/// semantic units that fit within the chunk size. Also will
/// attempt to merge neighboring chunks if they can fit within the
/// given chunk size.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct MarkdownSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Method of determining chunk sizes.
    chunk_config: ChunkConfig<Sizer>,
}

impl<Sizer> MarkdownSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Creates a new [`MarkdownSplitter`].
    ///
    /// ```
    /// use text_splitter::MarkdownSplitter;
    ///
    /// // By default, the chunk sizer is based on characters.
    /// let splitter = MarkdownSplitter::new(512);
    /// ```
    #[must_use]
    pub fn new(chunk_config: impl Into<ChunkConfig<Sizer>>) -> Self {
        Self {
            chunk_config: chunk_config.into(),
        }
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// ## Method
    ///
    /// To preserve as much semantic meaning within a chunk as possible, each chunk is composed of the largest semantic units that can fit in the next given chunk. For each splitter type, there is a defined set of semantic levels. Here is an example of the steps used:
    ///
    /// 1. Characters
    /// 2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// 3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// 4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// 5. Soft line breaks (single newline) which isn't necessarily a new element in Markdown.
    /// 6. Inline elements such as: text nodes, emphasis, strong, strikethrough, link, image, table cells, inline code, footnote references, task list markers, and inline html.
    /// 7. Block elements suce as: paragraphs, code blocks, footnote definitions, metadata. Also, a block quote or row/item within a table or list that can contain other "block" type elements, and a list or table that contains items.
    /// 8. Thematic breaks or horizontal rules.
    /// 9. Headings by level
    ///
    /// Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.
    ///
    /// Markdown is parsed according to the Commonmark spec, along with some optional features such as GitHub Flavored Markdown.
    ///
    /// ```
    /// use text_splitter::MarkdownSplitter;
    ///
    /// let splitter = MarkdownSplitter::new(10);
    /// let text = "# Header\n\nfrom a\ndocument";
    /// let chunks = splitter.chunks(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec!["# Header", "from a", "document"], chunks);
    /// ```
    pub fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        self.chunk_indices(text).map(|(_, t)| t)
    }

    /// Returns an iterator over chunks of the text and their byte offsets.
    /// Each chunk will be up to the `max_chunk_size`.
    ///
    /// See [`MarkdownSplitter::chunks`] for more information.
    ///
    /// ```
    /// use text_splitter::MarkdownSplitter;
    ///
    /// let splitter = MarkdownSplitter::new(10);
    /// let text = "# Header\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_indices(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![(0, "# Header"), (10, "from a"), (17, "document")], chunks);
    pub fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        TextChunks::<Sizer, MarkdownLevel>::new(&self.chunk_config, text)
    }
}

/// Heading levels in markdown.
/// Sorted in reverse order for sorting purposes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum HeadingLevel {
    H6,
    H5,
    H4,
    H3,
    H2,
    H1,
}

impl From<pulldown_cmark::HeadingLevel> for HeadingLevel {
    fn from(value: pulldown_cmark::HeadingLevel) -> Self {
        match value {
            pulldown_cmark::HeadingLevel::H1 => HeadingLevel::H1,
            pulldown_cmark::HeadingLevel::H2 => HeadingLevel::H2,
            pulldown_cmark::HeadingLevel::H3 => HeadingLevel::H3,
            pulldown_cmark::HeadingLevel::H4 => HeadingLevel::H4,
            pulldown_cmark::HeadingLevel::H5 => HeadingLevel::H5,
            pulldown_cmark::HeadingLevel::H6 => HeadingLevel::H6,
        }
    }
}

/// How a particular semantic level relates to surrounding text elements.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum SemanticSplitPosition {
    /// The semantic level should be treated as its own chunk.
    Own,
    /// The semantic level should be included in the next chunk.
    Next,
}

/// Different semantic levels that text can be split by.
/// Each level provides a method of splitting text into chunks of a given level
/// as well as a fallback in case a given fallback is too large.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum MarkdownLevel {
    /// Single line break, which isn't necessarily a new element in Markdown
    SoftBreak,
    /// An inline element that is within a larger element such as a paragraph, but
    /// more specific than a sentence.
    InlineElement,
    /// Paragraph, code block, metadata, a row/item within a table or list, block quote, that can contain other "block" type elements, List or table that contains items
    Block,
    /// thematic break/horizontal rule
    Rule,
    /// Heading levels in markdown
    Heading(HeadingLevel),
}

impl MarkdownLevel {
    fn split_position(self) -> SemanticSplitPosition {
        match self {
            Self::SoftBreak | Self::Block | Self::Rule | Self::InlineElement => {
                SemanticSplitPosition::Own
            }
            // Attach it to the next text
            Self::Heading(_) => SemanticSplitPosition::Next,
        }
    }

    fn treat_whitespace_as_previous(self) -> bool {
        match self {
            Self::SoftBreak | Self::InlineElement | Self::Rule | Self::Heading(_) => false,
            Self::Block => true,
        }
    }

    /// Given a list of separator ranges, construct the sections of the text
    fn split_str_by_separator(
        text: &str,
        separator_ranges: impl Iterator<Item = (Self, Range<usize>)>,
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
                                SemanticSplitPosition::Own => {
                                    if range.start < cursor {
                                        continue;
                                    }
                                    let prev_section = text
                                        .get(cursor..range.start)
                                        .expect("invalid character sequence");
                                    if level.treat_whitespace_as_previous()
                                        && prev_section.chars().all(char::is_whitespace)
                                    {
                                        let section = text
                                            .get(cursor..range.end)
                                            .expect("invalid character sequence");
                                        cursor = range.end;
                                        return Some(Either::Left(once((offset, section))));
                                    }
                                    let separator = text
                                        .get(range.start..range.end)
                                        .expect("invalid character sequence");
                                    cursor = range.end;
                                    return Some(Either::Right(
                                        [(offset, prev_section), (range.start, separator)]
                                            .into_iter(),
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
}

impl SemanticLevel for MarkdownLevel {
    const TRIM: Trim = Trim::PreserveIndentation;

    fn offsets(text: &str) -> impl Iterator<Item = (Self, Range<usize>)> {
        Parser::new_ext(text, Options::all())
            .into_offset_iter()
            .filter_map(|(event, range)| match event {
                Event::Start(
                    Tag::Emphasis
                    | Tag::Strong
                    | Tag::Strikethrough
                    | Tag::Link { .. }
                    | Tag::Image { .. }
                    | Tag::TableCell,
                )
                | Event::Text(_)
                | Event::HardBreak
                | Event::Code(_)
                | Event::InlineHtml(_)
                | Event::FootnoteReference(_)
                | Event::TaskListMarker(_) => Some((Self::InlineElement, range)),
                Event::SoftBreak => Some((Self::SoftBreak, range)),
                Event::Html(_)
                | Event::Start(
                    Tag::Paragraph
                    | Tag::CodeBlock(_)
                    | Tag::FootnoteDefinition(_)
                    | Tag::MetadataBlock(_)
                    | Tag::TableHead
                    | Tag::BlockQuote
                    | Tag::TableRow
                    | Tag::Item
                    | Tag::HtmlBlock
                    | Tag::List(_)
                    | Tag::Table(_),
                ) => Some((Self::Block, range)),
                Event::Rule => Some((Self::Rule, range)),
                Event::Start(Tag::Heading { level, .. }) => {
                    Some((Self::Heading(level.into()), range))
                }
                // End events are identical to start, so no need to grab them.
                Event::End(_) => None,
            })
    }

    fn sections(
        self,
        text: &str,
        level_ranges: impl Iterator<Item = (Self, Range<usize>)>,
    ) -> impl Iterator<Item = (usize, &str)> {
        match self {
            Self::SoftBreak | Self::InlineElement | Self::Block | Self::Heading(_) | Self::Rule => {
                Self::split_str_by_separator(text, level_ranges)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::min;

    use fake::{Fake, Faker};

    use crate::SemanticSplitRanges;

    use super::*;

    #[test]
    fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
        let text = Faker.fake::<String>();
        let chunks = TextChunks::<_, MarkdownLevel>::new(
            &ChunkConfig::new(text.chars().count()).with_trim(false),
            &text,
        )
        .map(|(_, c)| c)
        .collect::<Vec<_>>();
        assert_eq!(vec![&text], chunks);
    }

    #[test]
    fn returns_two_chunks_if_text_is_longer_than_max_chunk_size() {
        let text1 = Faker.fake::<String>();
        let text2 = Faker.fake::<String>();
        let text = format!("{text1}{text2}");
        // Round up to one above half so it goes to 2 chunks
        let max_chunk_size = text.chars().count() / 2 + 1;

        let chunks = TextChunks::<_, MarkdownLevel>::new(
            &ChunkConfig::new(max_chunk_size).with_trim(false),
            &text,
        )
        .map(|(_, c)| c)
        .collect::<Vec<_>>();

        assert!(chunks.iter().all(|c| c.chars().count() <= max_chunk_size));

        // Check that beginning of first chunk and text 1 matches
        let len = min(text1.len(), chunks[0].len());
        assert_eq!(text1[..len], chunks[0][..len]);
        // Check that end of second chunk and text 2 matches
        let len = min(text2.len(), chunks[1].len());
        assert_eq!(
            text2[(text2.len() - len)..],
            chunks[1][chunks[1].len() - len..]
        );

        assert_eq!(chunks.join(""), text);
    }

    #[test]
    fn empty_string() {
        let text = "";
        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(100).with_trim(false), text)
                .map(|(_, c)| c)
                .collect::<Vec<_>>();
        assert!(chunks.is_empty());
    }

    #[test]
    fn can_handle_unicode_characters() {
        let text = "éé"; // Char that is more than one byte
        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(1).with_trim(false), text)
                .map(|(_, c)| c)
                .collect::<Vec<_>>();
        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn chunk_by_graphemes() {
        let text = "a̐éö̲\r\n";

        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(3).with_trim(false), text)
                .map(|(_, g)| g)
                .collect::<Vec<_>>();
        // \r\n is grouped together not separated
        assert_eq!(vec!["a̐é", "ö̲", "\r\n"], chunks);
    }

    #[test]
    fn trim_char_indices() {
        let text = " a b ";

        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(1), text).collect::<Vec<_>>();
        assert_eq!(vec![(1, "a"), (3, "b")], chunks);
    }

    #[test]
    fn graphemes_fallback_to_chars() {
        let text = "a̐éö̲\r\n";

        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(1).with_trim(false), text)
                .map(|(_, g)| g)
                .collect::<Vec<_>>();
        assert_eq!(
            vec!["a", "\u{310}", "é", "ö", "\u{332}", "\r", "\n"],
            chunks
        );
    }

    #[test]
    fn trim_grapheme_indices() {
        let text = "\r\na̐éö̲\r\n";

        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(3), text).collect::<Vec<_>>();
        assert_eq!(vec![(2, "a̐é"), (7, "ö̲")], chunks);
    }

    #[test]
    fn chunk_by_words() {
        let text = "The quick brown fox can jump 32.3 feet, right?";

        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(10).with_trim(false), text)
                .map(|(_, w)| w)
                .collect::<Vec<_>>();
        assert_eq!(
            vec![
                "The quick ",
                "brown fox ",
                "can jump ",
                "32.3 feet,",
                " right?"
            ],
            chunks
        );
    }

    #[test]
    fn words_fallback_to_graphemes() {
        let text = "Thé quick\r\n";
        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(2).with_trim(false), text)
                .map(|(_, w)| w)
                .collect::<Vec<_>>();
        assert_eq!(vec!["Th", "é ", "qu", "ic", "k", "\r\n"], chunks);
    }

    #[test]
    fn trim_word_indices() {
        let text = "Some text from a document";
        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(10), text).collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text"), (10, "from a"), (17, "document")],
            chunks
        );
    }

    #[test]
    fn chunk_by_sentences() {
        let text = "Mr. Fox jumped. The dog was too lazy.";
        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(21).with_trim(false), text)
                .map(|(_, s)| s)
                .collect::<Vec<_>>();
        assert_eq!(vec!["Mr. Fox jumped. ", "The dog was too lazy."], chunks);
    }

    #[test]
    fn sentences_falls_back_to_words() {
        let text = "Mr. Fox jumped. The dog was too lazy.";
        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(16).with_trim(false), text)
                .map(|(_, s)| s)
                .collect::<Vec<_>>();
        assert_eq!(
            vec!["Mr. Fox jumped. ", "The dog was too ", "lazy."],
            chunks
        );
    }

    #[test]
    fn trim_sentence_indices() {
        let text = "Some text. From a document.";
        let chunks =
            TextChunks::<_, MarkdownLevel>::new(&ChunkConfig::new(10), text).collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text."), (11, "From a"), (18, "document.")],
            chunks
        );
    }

    #[test]
    fn test_no_markdown_separators() {
        let markdown = SemanticSplitRanges::new(
            MarkdownLevel::offsets("Some text without any markdown separators").collect(),
        );

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..41),
                (MarkdownLevel::InlineElement, 0..41)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_checklist() {
        let markdown = SemanticSplitRanges::new(
            MarkdownLevel::offsets("- [ ] incomplete task\n- [x] completed task").collect(),
        );

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..42),
                (MarkdownLevel::Block, 0..22),
                (MarkdownLevel::InlineElement, 2..5),
                (MarkdownLevel::InlineElement, 6..21),
                (MarkdownLevel::Block, 22..42),
                (MarkdownLevel::InlineElement, 24..27),
                (MarkdownLevel::InlineElement, 28..42),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_footnote_reference() {
        let markdown = SemanticSplitRanges::new(MarkdownLevel::offsets("Footnote[^1]").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..12),
                (MarkdownLevel::InlineElement, 0..8),
                (MarkdownLevel::InlineElement, 8..12),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inline_code() {
        let markdown = SemanticSplitRanges::new(MarkdownLevel::offsets("`bash`").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..6),
                (MarkdownLevel::InlineElement, 0..6)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_emphasis() {
        let markdown = SemanticSplitRanges::new(MarkdownLevel::offsets("*emphasis*").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..10),
                (MarkdownLevel::InlineElement, 0..10),
                (MarkdownLevel::InlineElement, 1..9),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_strong() {
        let markdown = SemanticSplitRanges::new(MarkdownLevel::offsets("**emphasis**").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..12),
                (MarkdownLevel::InlineElement, 0..12),
                (MarkdownLevel::InlineElement, 2..10),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_strikethrough() {
        let markdown = SemanticSplitRanges::new(MarkdownLevel::offsets("~~emphasis~~").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..12),
                (MarkdownLevel::InlineElement, 0..12),
                (MarkdownLevel::InlineElement, 2..10),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_link() {
        let markdown = SemanticSplitRanges::new(MarkdownLevel::offsets("[link](url)").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..11),
                (MarkdownLevel::InlineElement, 0..11),
                (MarkdownLevel::InlineElement, 1..5),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_image() {
        let markdown = SemanticSplitRanges::new(MarkdownLevel::offsets("![link](url)").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..12),
                (MarkdownLevel::InlineElement, 0..12),
                (MarkdownLevel::InlineElement, 2..6),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inline_html() {
        let markdown =
            SemanticSplitRanges::new(MarkdownLevel::offsets("<span>Some text</span>").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..22),
                (MarkdownLevel::InlineElement, 0..6),
                (MarkdownLevel::InlineElement, 6..15),
                (MarkdownLevel::InlineElement, 15..22),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_html() {
        let markdown =
            SemanticSplitRanges::new(MarkdownLevel::offsets("<div>Some text</div>").collect());

        assert_eq!(
            vec![(MarkdownLevel::Block, 0..20), (MarkdownLevel::Block, 0..20)],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_table() {
        let markdown = SemanticSplitRanges::new(
            MarkdownLevel::offsets("| Header 1 | Header 2 |\n| --- | --- |\n| Cell 1 | Cell 2 |")
                .collect(),
        );
        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..57),
                (MarkdownLevel::Block, 0..24),
                (MarkdownLevel::InlineElement, 1..11),
                (MarkdownLevel::InlineElement, 2..10),
                (MarkdownLevel::InlineElement, 12..22),
                (MarkdownLevel::InlineElement, 13..21),
                (MarkdownLevel::Block, 38..57),
                (MarkdownLevel::InlineElement, 39..47),
                (MarkdownLevel::InlineElement, 40..46),
                (MarkdownLevel::InlineElement, 48..56),
                (MarkdownLevel::InlineElement, 49..55)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_softbreak() {
        let markdown = SemanticSplitRanges::new(
            MarkdownLevel::offsets("Some text\nwith a softbreak").collect(),
        );

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..26),
                (MarkdownLevel::InlineElement, 0..9),
                (MarkdownLevel::SoftBreak, 9..10),
                (MarkdownLevel::InlineElement, 10..26)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_hardbreak() {
        let markdown = SemanticSplitRanges::new(
            MarkdownLevel::offsets("Some text\\\nwith a hardbreak").collect(),
        );

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..27),
                (MarkdownLevel::InlineElement, 0..9),
                (MarkdownLevel::InlineElement, 9..11),
                (MarkdownLevel::InlineElement, 11..27)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_footnote_def() {
        let markdown =
            SemanticSplitRanges::new(MarkdownLevel::offsets("[^first]: Footnote").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..18),
                (MarkdownLevel::Block, 10..18),
                (MarkdownLevel::InlineElement, 10..18)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_code_block() {
        let markdown = SemanticSplitRanges::new(MarkdownLevel::offsets("```\ncode\n```").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..12),
                (MarkdownLevel::InlineElement, 4..9)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_block_quote() {
        let markdown = SemanticSplitRanges::new(MarkdownLevel::offsets("> quote").collect());

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..7),
                (MarkdownLevel::Block, 2..7),
                (MarkdownLevel::InlineElement, 2..7)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_with_rule() {
        let markdown = SemanticSplitRanges::new(
            MarkdownLevel::offsets("Some text\n\n---\n\nwith a rule").collect(),
        );

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..10),
                (MarkdownLevel::InlineElement, 0..9),
                (MarkdownLevel::Rule, 11..15),
                (MarkdownLevel::Block, 16..27),
                (MarkdownLevel::InlineElement, 16..27)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_heading() {
        for (index, (heading, level)) in [
            ("#", HeadingLevel::H1),
            ("##", HeadingLevel::H2),
            ("###", HeadingLevel::H3),
            ("####", HeadingLevel::H4),
            ("#####", HeadingLevel::H5),
            ("######", HeadingLevel::H6),
        ]
        .into_iter()
        .enumerate()
        {
            let markdown = SemanticSplitRanges::new(
                MarkdownLevel::offsets(&format!("{heading} Heading")).collect(),
            );

            assert_eq!(
                vec![
                    (MarkdownLevel::Heading(level), 0..9 + index),
                    (MarkdownLevel::InlineElement, 2 + index..9 + index)
                ],
                markdown.ranges_after_offset(0).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_ranges_after_offset_block() {
        let markdown = SemanticSplitRanges::new(
            MarkdownLevel::offsets("- [ ] incomplete task\n- [x] completed task").collect(),
        );

        assert_eq!(
            vec![
                (MarkdownLevel::Block, 0..22),
                (MarkdownLevel::Block, 22..42),
            ],
            markdown
                .level_ranges_after_offset(0, MarkdownLevel::Block)
                .collect::<Vec<_>>()
        );
    }
}
