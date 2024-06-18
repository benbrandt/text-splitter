/*!
# [`TextSplitter`]
Semantic splitting of text documents.
*/

use std::ops::Range;

use once_cell::sync::Lazy;
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

use crate::{
    splitter::{SemanticLevel, Splitter},
    ChunkConfig, ChunkSizer,
};

/// Default plain-text splitter. Recursively splits chunks into the largest
/// semantic units that fit within the chunk size. Also will attempt to merge
/// neighboring chunks if they can fit within the given chunk size.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct TextSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Method of determining chunk sizes.
    chunk_config: ChunkConfig<Sizer>,
}

impl<Sizer> TextSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Creates a new [`TextSplitter`].
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// // By default, the chunk sizer is based on characters.
    /// let splitter = TextSplitter::new(512);
    /// ```
    #[must_use]
    pub fn new(chunk_config: impl Into<ChunkConfig<Sizer>>) -> Self {
        Self {
            chunk_config: chunk_config.into(),
        }
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.
    ///
    /// ## Method
    ///
    /// To preserve as much semantic meaning within a chunk as possible, each chunk is composed of the largest semantic units that can fit in the next given chunk. For each splitter type, there is a defined set of semantic levels. Here is an example of the steps used:
    //
    // 1. Split the text by a increasing semantic levels.
    // 2. Check the first item for each level and select the highest level whose first item still fits within the chunk size.
    // 3. Merge as many of these neighboring sections of this level or above into a chunk to maximize chunk length.
    //    Boundaries of higher semantic levels are always included when merging, so that the chunk doesn't inadvertantly cross semantic boundaries.
    //
    // The boundaries used to split the text if using the `chunks` method, in ascending order:
    //
    // 1. Characters
    // 2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    // 3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    // 4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    // 5. Ascending sequence length of newlines. (Newline is `\r\n`, `\n`, or `\r`)
    //    Each unique length of consecutive newline sequences is treated as its own semantic level. So a sequence of 2 newlines is a higher level than a sequence of 1 newline, and so on.
    //
    // Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunks(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec!["Some text", "from a", "document"], chunks);
    /// ```
    pub fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        Splitter::<_>::chunks(self, text)
    }

    /// Returns an iterator over chunks of the text and their byte offsets.
    /// Each chunk will be up to the `chunk_capacity`.
    ///
    /// See [`TextSplitter::chunks`] for more information.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_indices(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![(0, "Some text"), (11, "from a"), (18, "document")], chunks);
    pub fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        Splitter::<_>::chunk_indices(self, text)
    }
}

impl<Sizer> Splitter<Sizer> for TextSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    type Level = LineBreaks;

    fn chunk_config(&self) -> &ChunkConfig<Sizer> {
        &self.chunk_config
    }

    fn parse(&self, text: &str) -> Vec<(Self::Level, Range<usize>)> {
        CAPTURE_LINEBREAKS
            .find_iter(text)
            .map(|m| {
                let range = m.range();
                let level = text
                    .get(range.start..range.end)
                    .unwrap()
                    .graphemes(true)
                    .count();
                (
                    match level {
                        0 => unreachable!("regex should always match at least one newline"),
                        n => LineBreaks(n),
                    },
                    range,
                )
            })
            .collect()
    }
}

/// Different semantic levels that text can be split by.
/// Each level provides a method of splitting text into chunks of a given level
/// as well as a fallback in case a given fallback is too large.
///
/// Split by given number of linebreaks, either `\n`, `\r`, or `\r\n`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct LineBreaks(usize);

// Lazy so that we don't have to compile them more than once
static CAPTURE_LINEBREAKS: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r\n)+|\r+|\n+").unwrap());

impl SemanticLevel for LineBreaks {}

#[cfg(test)]
mod tests {
    use std::cmp::min;

    use fake::{Fake, Faker};

    use crate::splitter::SemanticSplitRanges;

    use super::*;

    #[test]
    fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
        let text = Faker.fake::<String>();
        let chunks = TextSplitter::new(ChunkConfig::new(text.chars().count()).with_trim(false))
            .chunks(&text)
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
        let chunks = TextSplitter::new(ChunkConfig::new(max_chunk_size).with_trim(false))
            .chunks(&text)
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
        let chunks = TextSplitter::new(ChunkConfig::new(100).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert!(chunks.is_empty());
    }

    #[test]
    fn can_handle_unicode_characters() {
        let text = "éé"; // Char that is more than one byte
        let chunks = TextSplitter::new(ChunkConfig::new(1).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();
        assert_eq!(vec!["é", "é"], chunks);
    }

    // Just for testing
    struct Str;

    impl ChunkSizer for Str {
        fn size(&self, chunk: &str) -> usize {
            chunk.as_bytes().len()
        }
    }

    #[test]
    fn custom_len_function() {
        let text = "éé"; // Char that is two bytes each
        let chunks = TextSplitter::new(ChunkConfig::new(2).with_sizer(Str).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn handles_char_bigger_than_len() {
        let text = "éé"; // Char that is two bytes each
        let chunks = TextSplitter::new(ChunkConfig::new(1).with_sizer(Str).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        // We can only go so small
        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn chunk_by_graphemes() {
        let text = "a̐éö̲\r\n";
        let chunks = TextSplitter::new(ChunkConfig::new(3).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        // \r\n is grouped together not separated
        assert_eq!(vec!["a̐é", "ö̲", "\r\n"], chunks);
    }

    #[test]
    fn trim_char_indices() {
        let text = " a b ";
        let chunks = TextSplitter::new(1).chunk_indices(text).collect::<Vec<_>>();

        assert_eq!(vec![(1, "a"), (3, "b")], chunks);
    }

    #[test]
    fn graphemes_fallback_to_chars() {
        let text = "a̐éö̲\r\n";
        let chunks = TextSplitter::new(ChunkConfig::new(1).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();
        assert_eq!(
            vec!["a", "\u{310}", "é", "ö", "\u{332}", "\r", "\n"],
            chunks
        );
    }

    #[test]
    fn trim_grapheme_indices() {
        let text = "\r\na̐éö̲\r\n";
        let chunks = TextSplitter::new(3).chunk_indices(text).collect::<Vec<_>>();

        assert_eq!(vec![(2, "a̐é"), (7, "ö̲")], chunks);
    }

    #[test]
    fn chunk_by_words() {
        let text = "The quick (\"brown\") fox can't jump 32.3 feet, right?";
        let chunks = TextSplitter::new(ChunkConfig::new(10).with_trim(false))
            .chunks(text)
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
        let chunks = TextSplitter::new(ChunkConfig::new(2).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();
        assert_eq!(vec!["Th", "é ", "qu", "ic", "k", "\r\n"], chunks);
    }

    #[test]
    fn trim_word_indices() {
        let text = "Some text from a document";
        let chunks = TextSplitter::new(10)
            .chunk_indices(text)
            .collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text"), (10, "from a"), (17, "document")],
            chunks
        );
    }

    #[test]
    fn chunk_by_sentences() {
        let text = "Mr. Fox jumped. [...] The dog was too lazy.";
        let chunks = TextSplitter::new(ChunkConfig::new(21).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();
        assert_eq!(
            vec!["Mr. Fox jumped. ", "[...] ", "The dog was too lazy."],
            chunks
        );
    }

    #[test]
    fn sentences_falls_back_to_words() {
        let text = "Mr. Fox jumped. [...] The dog was too lazy.";
        let chunks = TextSplitter::new(ChunkConfig::new(16).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();
        assert_eq!(
            vec!["Mr. Fox jumped. ", "[...] ", "The dog was too ", "lazy."],
            chunks
        );
    }

    #[test]
    fn trim_sentence_indices() {
        let text = "Some text. From a document.";
        let chunks = TextSplitter::new(10)
            .chunk_indices(text)
            .collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text."), (11, "From a"), (18, "document.")],
            chunks
        );
    }

    #[test]
    fn trim_paragraph_indices() {
        let text = "Some text\n\nfrom a\ndocument";
        let chunks = TextSplitter::new(10)
            .chunk_indices(text)
            .collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text"), (11, "from a"), (18, "document")],
            chunks
        );
    }

    #[test]
    fn correctly_determines_newlines() {
        let text = "\r\n\r\ntext\n\n\ntext2";
        let splitter = TextSplitter::new(10);
        let linebreaks = SemanticSplitRanges::new(splitter.parse(text));
        assert_eq!(
            vec![(LineBreaks(2), 0..4), (LineBreaks(3), 8..11)],
            linebreaks.ranges
        );
    }
}
