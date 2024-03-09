/*!
# [`MarkdownSplitter`]
Semantic splitting of Markdown documents. Tries to use as many semantic units from Markdown
as possible, according to the Common Mark specification.
*/

use std::ops::Range;

use auto_enums::auto_enum;
use pulldown_cmark::{Event, Options, Parser, Tag};
use unicode_segmentation::UnicodeSegmentation;

use crate::{
    split_str_by_separator, Characters, ChunkCapacity, ChunkSizer, Level, SemanticSplit,
    SemanticSplitPosition, TextChunks,
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
    /// If `true`, all chunks will have whitespace removed from beginning and end, preserving indentation if necessary.
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
    /// Indentation however will be preserved if the chunk also includes multiple lines.
    /// Extra newlines are always removed, but if the text would include multiple indented list
    /// items, the indentation of the first element will also be preserved.
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
    /// To preserve as much semantic meaning within a chunk as possible, each chunk is composed of the largest semantic units that can fit in the next given chunk. For each splitter type, there is a defined set of semantic levels. Here is an example of the steps used:
    ///
    /// 1. Characters
    /// 2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// 3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// 4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// 5. Soft line breaks (single newline) which isn't necessarily a new element in Markdown.
    /// 6. Text nodes within elements
    /// 7. Inline elements such as: emphasis, strong, strikethrough, link, image, table cells, inline code, footnote references, task list markers, and inline html.
    /// 8. Block elements suce as: paragraphs, code blocks, and footnote definitions.
    /// 9. Container blocks such as: table rows, block quotes, list items, and HTML blocks.
    /// 10. Meta containers such as: lists and tables.
    /// 11. Thematic breaks or horizontal rules.
    /// 12. Headings by level
    /// 13. Metadata at the beginning of the document
    ///
    /// Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.
    ///
    /// Markdown is parsed according to the Commonmark spec, along with some optional features such as GitHub Flavored Markdown.
    ///
    /// ```
    /// use text_splitter::{Characters, MarkdownSplitter};
    ///
    /// let splitter = MarkdownSplitter::default();
    /// let text = "# Header\n\nfrom a\ndocument";
    /// let chunks = splitter.chunks(text, 10).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec!["# Header\n\n", "from a\n", "document"], chunks);
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
    /// let text = "# Header\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_indices(text, 10).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![(0, "# Header\n\n"), (10, "from a\n"), (17, "document")], chunks);
    pub fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: impl ChunkCapacity + 'splitter,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        TextChunks::<_, S, Markdown>::new(chunk_capacity, &self.chunk_sizer, text, self.trim_chunks)
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

/// Different semantic levels that text can be split by.
/// Each level provides a method of splitting text into chunks of a given level
/// as well as a fallback in case a given fallback is too large.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum SemanticLevel {
    /// Split by individual chars. May be larger than a single byte,
    /// but we don't go lower so we always have valid UTF str's.
    Char,
    /// Split by [unicode grapheme clusters](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)    Grapheme,
    GraphemeCluster,
    /// Split by [unicode words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    Word,
    /// Split by [unicode sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    Sentence,
    /// Single line break, which isn't necessarily a new element in Markdown
    SoftBreak,
    /// A text node within an element
    Text,
    /// An inline element that is within a larger element such as a paragraph, but
    /// more specific than a sentence.
    InlineElement(SemanticSplitPosition),
    /// Paragraph, code block,  etc
    Block,
    /// A row/item within a table or list, block quote, that can contain other "block" type elements
    ContainerBlock(SemanticSplitPosition),
    /// List or table that contains items
    MetaContainer,
    /// thematic break/horizontal rule
    Rule,
    /// Heading levels in markdown
    Heading(HeadingLevel),
    /// Metadata for the entire document
    Metadata,
}

impl Level for SemanticLevel {
    fn split_position(&self) -> SemanticSplitPosition {
        match self {
            SemanticLevel::Char
            | SemanticLevel::GraphemeCluster
            | SemanticLevel::Word
            | SemanticLevel::Sentence
            | SemanticLevel::SoftBreak
            | SemanticLevel::Text
            | SemanticLevel::Block
            | SemanticLevel::MetaContainer
            | SemanticLevel::Rule
            | SemanticLevel::Metadata => SemanticSplitPosition::Own,
            SemanticLevel::InlineElement(p) | SemanticLevel::ContainerBlock(p) => *p,
            // Attach it to the next text
            SemanticLevel::Heading(_) => SemanticSplitPosition::Next,
        }
    }

    fn treat_whitespace_as_previous(&self) -> bool {
        match self {
            SemanticLevel::Char
            | SemanticLevel::GraphemeCluster
            | SemanticLevel::Word
            | SemanticLevel::Sentence
            | SemanticLevel::SoftBreak
            | SemanticLevel::Text
            | SemanticLevel::InlineElement(_)
            | SemanticLevel::Rule
            | SemanticLevel::Heading(_)
            | SemanticLevel::Metadata => false,
            SemanticLevel::Block
            | SemanticLevel::ContainerBlock(_)
            | SemanticLevel::MetaContainer => true,
        }
    }
}

/// Captures information about markdown structure for a given text, and their
/// various semantic levels.
#[derive(Debug)]
struct Markdown {
    /// Range of each semantic markdown item and its precalculated semantic level
    ranges: Vec<(SemanticLevel, Range<usize>)>,
}

const NEWLINES: [char; 2] = ['\n', '\r'];

impl SemanticSplit for Markdown {
    type Level = SemanticLevel;

    const PERSISTENT_LEVELS: &'static [Self::Level] = &[
        SemanticLevel::Char,
        SemanticLevel::GraphemeCluster,
        SemanticLevel::Word,
        SemanticLevel::Sentence,
    ];

    fn new(text: &str) -> Self {
        let ranges = Parser::new_ext(text, Options::all())
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
                | Event::HardBreak
                | Event::Code(_)
                | Event::InlineHtml(_) => Some((
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    range,
                )),
                Event::Text(_) => Some((SemanticLevel::Text, range)),
                Event::FootnoteReference(_) => Some((
                    SemanticLevel::InlineElement(SemanticSplitPosition::Prev),
                    range,
                )),
                Event::TaskListMarker(_) => Some((
                    SemanticLevel::InlineElement(SemanticSplitPosition::Next),
                    range,
                )),
                Event::SoftBreak => Some((SemanticLevel::SoftBreak, range)),
                Event::Html(_)
                | Event::Start(Tag::Paragraph | Tag::CodeBlock(_) | Tag::FootnoteDefinition(_)) => {
                    Some((SemanticLevel::Block, range))
                }
                Event::Start(Tag::TableHead) => Some((
                    SemanticLevel::ContainerBlock(SemanticSplitPosition::Next),
                    range,
                )),
                Event::Start(Tag::BlockQuote | Tag::TableRow | Tag::Item | Tag::HtmlBlock) => {
                    Some((
                        SemanticLevel::ContainerBlock(SemanticSplitPosition::Own),
                        range,
                    ))
                }
                Event::Start(Tag::List(_) | Tag::Table(_)) => {
                    Some((SemanticLevel::MetaContainer, range))
                }
                Event::Rule => Some((SemanticLevel::Rule, range)),
                Event::Start(Tag::Heading { level, .. }) => {
                    Some((SemanticLevel::Heading(level.into()), range))
                }
                Event::Start(Tag::MetadataBlock(_)) => Some((SemanticLevel::Metadata, range)),
                // End events are identical to start, so no need to grab them.
                Event::End(_) => None,
            })
            .collect::<Vec<_>>();

        Self { ranges }
    }

    fn ranges(&self) -> impl Iterator<Item = &(Self::Level, Range<usize>)> + '_ {
        self.ranges.iter()
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
            SemanticLevel::Text
            | SemanticLevel::SoftBreak
            | SemanticLevel::InlineElement(_)
            | SemanticLevel::ContainerBlock(_)
            | SemanticLevel::Block
            | SemanticLevel::MetaContainer
            | SemanticLevel::Heading(_)
            | SemanticLevel::Rule
            | SemanticLevel::Metadata => split_str_by_separator(
                text,
                self.ranges_after_offset(offset, semantic_level)
                    .map(move |(l, sep)| (*l, sep.start - offset..sep.end - offset)),
            )
            .map(move |(i, str)| (offset + i, str)),
        }
    }

    fn trim_chunk<'splitter, 'text: 'splitter>(
        &'splitter self,
        offset: usize,
        chunk: &'text str,
    ) -> (usize, &'text str) {
        // Preserve indentation if we have newlines inside the element
        if chunk.trim().contains(NEWLINES) {
            let diff = chunk.len() - chunk.trim_start_matches(NEWLINES).len();
            (offset + diff, chunk.trim_start_matches(NEWLINES).trim_end())
        } else {
            let diff = chunk.len() - chunk.trim_start().len();
            (offset + diff, chunk.trim())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::min;

    use fake::{Fake, Faker};

    use super::*;

    #[test]
    fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
        let text = Faker.fake::<String>();
        let chunks =
            TextChunks::<_, _, Markdown>::new(text.chars().count(), &Characters, &text, false)
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

        let chunks = TextChunks::<_, _, Markdown>::new(max_chunk_size, &Characters, &text, false)
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
        let chunks = TextChunks::<_, _, Markdown>::new(100, &Characters, text, false)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        assert!(chunks.is_empty());
    }

    #[test]
    fn can_handle_unicode_characters() {
        let text = "éé"; // Char that is more than one byte
        let chunks = TextChunks::<_, _, Markdown>::new(1, &Characters, text, false)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn chunk_by_graphemes() {
        let text = "a̐éö̲\r\n";

        let chunks = TextChunks::<_, _, Markdown>::new(3, &Characters, text, false)
            .map(|(_, g)| g)
            .collect::<Vec<_>>();
        // \r\n is grouped together not separated
        assert_eq!(vec!["a̐é", "ö̲", "\r\n"], chunks);
    }

    #[test]
    fn trim_char_indices() {
        let text = " a b ";

        let chunks =
            TextChunks::<_, _, Markdown>::new(1, &Characters, text, true).collect::<Vec<_>>();
        assert_eq!(vec![(1, "a"), (3, "b")], chunks);
    }

    #[test]
    fn graphemes_fallback_to_chars() {
        let text = "a̐éö̲\r\n";

        let chunks = TextChunks::<_, _, Markdown>::new(1, &Characters, text, false)
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
            TextChunks::<_, _, Markdown>::new(3, &Characters, text, true).collect::<Vec<_>>();
        assert_eq!(vec![(2, "a̐é"), (7, "ö̲")], chunks);
    }

    #[test]
    fn chunk_by_words() {
        let text = "The quick (\"brown\") fox can't jump 32.3 feet, right?";

        let chunks = TextChunks::<_, _, Markdown>::new(10, &Characters, text, false)
            .map(|(_, w)| w)
            .collect::<Vec<_>>();
        assert_eq!(
            vec![
                "The quick ",
                "(\"brown\") ",
                "fox can't ",
                "jump 32.3 ",
                "feet, ",
                "right?"
            ],
            chunks
        );
    }

    #[test]
    fn words_fallback_to_graphemes() {
        let text = "Thé quick\r\n";
        let chunks = TextChunks::<_, _, Markdown>::new(2, &Characters, text, false)
            .map(|(_, w)| w)
            .collect::<Vec<_>>();
        assert_eq!(vec!["Th", "é ", "qu", "ic", "k", "\r\n"], chunks);
    }

    #[test]
    fn trim_word_indices() {
        let text = "Some text from a document";
        let chunks =
            TextChunks::<_, _, Markdown>::new(10, &Characters, text, true).collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text"), (10, "from a"), (17, "document")],
            chunks
        );
    }

    #[test]
    fn chunk_by_sentences() {
        let text = "Mr. Fox jumped. The dog was too lazy.";
        let chunks = TextChunks::<_, _, Markdown>::new(21, &Characters, text, false)
            .map(|(_, s)| s)
            .collect::<Vec<_>>();
        assert_eq!(vec!["Mr. Fox jumped. ", "The dog was too lazy."], chunks);
    }

    #[test]
    fn sentences_falls_back_to_words() {
        let text = "Mr. Fox jumped. The dog was too lazy.";
        let chunks = TextChunks::<_, _, Markdown>::new(16, &Characters, text, false)
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
            TextChunks::<_, _, Markdown>::new(10, &Characters, text, true).collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text."), (11, "From a"), (18, "document.")],
            chunks
        );
    }

    #[test]
    fn test_no_markdown_separators() {
        let markdown = Markdown::new("Some text without any markdown separators");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..41),
                &(SemanticLevel::Text, 0..41)
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_checklist() {
        let markdown = Markdown::new("- [ ] incomplete task\n- [x] completed task");

        assert_eq!(
            vec![
                &(SemanticLevel::MetaContainer, 0..42),
                &(
                    SemanticLevel::ContainerBlock(SemanticSplitPosition::Own),
                    0..22
                ),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Next),
                    2..5
                ),
                &(SemanticLevel::Text, 6..21),
                &(
                    SemanticLevel::ContainerBlock(SemanticSplitPosition::Own),
                    22..42
                ),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Next),
                    24..27
                ),
                &(SemanticLevel::Text, 28..42),
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_footnote_reference() {
        let markdown = Markdown::new("Footnote[^1]");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..12),
                &(SemanticLevel::Text, 0..8),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Prev),
                    8..12
                ),
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inline_code() {
        let markdown = Markdown::new("`bash`");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..6),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    0..6
                )
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_emphasis() {
        let markdown = Markdown::new("*emphasis*");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..10),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    0..10
                ),
                &(SemanticLevel::Text, 1..9),
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_strong() {
        let markdown = Markdown::new("**emphasis**");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..12),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    0..12
                ),
                &(SemanticLevel::Text, 2..10),
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_strikethrough() {
        let markdown = Markdown::new("~~emphasis~~");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..12),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    0..12
                ),
                &(SemanticLevel::Text, 2..10),
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_link() {
        let markdown = Markdown::new("[link](url)");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..11),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    0..11
                ),
                &(SemanticLevel::Text, 1..5),
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_image() {
        let markdown = Markdown::new("![link](url)");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..12),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    0..12
                ),
                &(SemanticLevel::Text, 2..6),
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inline_html() {
        let markdown = Markdown::new("<span>Some text</span>");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..22),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    0..6
                ),
                &(SemanticLevel::Text, 6..15),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    15..22
                ),
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_html() {
        let markdown = Markdown::new("<div>Some text</div>");

        assert_eq!(
            vec![
                &(
                    SemanticLevel::ContainerBlock(SemanticSplitPosition::Own),
                    0..20
                ),
                &(SemanticLevel::Block, 0..20)
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_table() {
        let markdown = Markdown::new("| Header 1 | Header 2 |\n| --- | --- |\n| Cell 1 | Cell 2 |");
        assert_eq!(
            vec![
                &(SemanticLevel::MetaContainer, 0..57),
                &(
                    SemanticLevel::ContainerBlock(SemanticSplitPosition::Next),
                    0..24
                ),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    1..11
                ),
                &(SemanticLevel::Text, 2..10),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    12..22
                ),
                &(SemanticLevel::Text, 13..21),
                &(
                    SemanticLevel::ContainerBlock(SemanticSplitPosition::Own),
                    38..57
                ),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    39..47
                ),
                &(SemanticLevel::Text, 40..46),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    48..56
                ),
                &(SemanticLevel::Text, 49..55)
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_softbreak() {
        let markdown = Markdown::new("Some text\nwith a softbreak");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..26),
                &(SemanticLevel::Text, 0..9),
                &(SemanticLevel::SoftBreak, 9..10),
                &(SemanticLevel::Text, 10..26)
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_hardbreak() {
        let markdown = Markdown::new("Some text\\\nwith a hardbreak");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..27),
                &(SemanticLevel::Text, 0..9),
                &(
                    SemanticLevel::InlineElement(SemanticSplitPosition::Own),
                    9..11
                ),
                &(SemanticLevel::Text, 11..27)
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_footnote_def() {
        let markdown = Markdown::new("[^first]: Footnote");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..18),
                &(SemanticLevel::Block, 10..18),
                &(SemanticLevel::Text, 10..18)
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_code_block() {
        let markdown = Markdown::new("```\ncode\n```");

        assert_eq!(
            vec![&(SemanticLevel::Block, 0..12), &(SemanticLevel::Text, 4..9)],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_block_quote() {
        let markdown = Markdown::new("> quote");

        assert_eq!(
            vec![
                &(
                    SemanticLevel::ContainerBlock(SemanticSplitPosition::Own),
                    0..7
                ),
                &(SemanticLevel::Block, 2..7),
                &(SemanticLevel::Text, 2..7)
            ],
            markdown.ranges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_with_rule() {
        let markdown = Markdown::new("Some text\n\n---\n\nwith a rule");

        assert_eq!(
            vec![
                &(SemanticLevel::Block, 0..10),
                &(SemanticLevel::Text, 0..9),
                &(SemanticLevel::Rule, 11..15),
                &(SemanticLevel::Block, 16..27),
                &(SemanticLevel::Text, 16..27)
            ],
            markdown.ranges().collect::<Vec<_>>()
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
            let markdown = Markdown::new(&format!("{heading} Heading"));

            assert_eq!(
                vec![
                    &(SemanticLevel::Heading(level), 0..9 + index),
                    &(SemanticLevel::Text, 2 + index..9 + index)
                ],
                markdown.ranges().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_ranges_after_offset_block() {
        let markdown = Markdown::new("- [ ] incomplete task\n- [x] completed task");

        assert_eq!(
            vec![&(SemanticLevel::MetaContainer, 0..42),],
            markdown
                .ranges_after_offset(0, SemanticLevel::MetaContainer)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_ranges_after_offset_item() {
        let markdown = Markdown::new("- [ ] incomplete task\n- [x] completed task");

        assert_eq!(
            vec![
                &(
                    SemanticLevel::ContainerBlock(SemanticSplitPosition::Own),
                    0..22
                ),
                &(
                    SemanticLevel::ContainerBlock(SemanticSplitPosition::Own),
                    22..42
                ),
            ],
            markdown
                .ranges_after_offset(0, SemanticLevel::ContainerBlock(SemanticSplitPosition::Own))
                .collect::<Vec<_>>()
        );
    }
}
