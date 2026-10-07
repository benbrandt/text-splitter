use std::sync::LazyLock;

use auto_enums::auto_enum;
use icu_segmenter::{
    options::{SentenceBreakInvariantOptions, WordBreakInvariantOptions},
    GraphemeClusterSegmenter, GraphemeClusterSegmenterBorrowed, SentenceSegmenter,
    SentenceSegmenterBorrowed, WordSegmenter, WordSegmenterBorrowed,
};
use itertools::Itertools;
use strum::EnumIter;

pub const GRAPHEME_SEGMENTER: GraphemeClusterSegmenterBorrowed<'static> =
    GraphemeClusterSegmenter::new();
static WORD_SEGMENTER: LazyLock<WordSegmenterBorrowed<'static>> =
    LazyLock::new(|| WordSegmenter::new_dictionary(WordBreakInvariantOptions::default()));
static SENTENCE_SEGMENTER: LazyLock<SentenceSegmenterBorrowed<'static>> =
    LazyLock::new(|| SentenceSegmenter::new(SentenceBreakInvariantOptions::default()));

/// When using a custom semantic level, it is possible that none of them will
/// be small enough to fit into the chunk size. In order to make sure we can
/// still move the cursor forward, we fallback to unicode segmentation.
#[derive(Clone, Copy, Debug, EnumIter, Eq, PartialEq, Ord, PartialOrd)]
pub enum FallbackLevel {
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
    pub fn boundary_level_for_probe(self) -> Option<Self> {
        match self {
            Self::Sentence => Some(Self::Word),
            Self::Char | Self::GraphemeCluster | Self::Word => None,
        }
    }

    /// Returns the first section of `text` at this level, segmenting only the
    /// first `window` bytes unless that window contains a boundary.
    ///
    /// The boolean is `true` when the window holds no boundary. The returned
    /// str is then the whole window, and the real first section is longer.
    ///
    /// Without a window, a level with no boundary in the text (such as a
    /// sentence in a long list of URLs) is segmented to the end of the text.
    /// That happens again for every chunk, which makes splitting quadratic.
    ///
    /// Truncating the text can only add boundaries near the cut, never remove
    /// one earlier, because the Unicode rules only look ahead to prevent a
    /// break. So a window with no boundary proves the first section is longer
    /// than the window. When the window does show a boundary, it may be an
    /// artifact of the cut, so the full text is segmented again. That scan
    /// stops at the real first boundary, which is close by.
    pub fn first_section_within(self, text: &str, window: usize) -> Option<(&str, bool)> {
        if window >= text.len() {
            return self.sections(text).next().map(|(_, s)| (s, false));
        }

        let mut end = window.max(1);
        while !text.is_char_boundary(end) {
            end += 1;
        }

        match self.sections(&text[..end]).next() {
            Some((_, section)) if section.len() < end => {
                self.sections(text).next().map(|(_, s)| (s, false))
            }
            _ => Some((&text[..end], true)),
        }
    }

    #[auto_enum(Iterator)]
    pub fn sections(self, text: &str) -> impl Iterator<Item = (usize, &str)> {
        match self {
            Self::Char => text.char_indices().map(move |(i, c)| {
                (
                    i,
                    text.get(i..i + c.len_utf8()).expect("char should be valid"),
                )
            }),
            Self::GraphemeCluster => GRAPHEME_SEGMENTER
                .segment_str(text)
                .tuple_windows()
                .map(|(i, j)| (i, &text[i..j])),
            Self::Word => WORD_SEGMENTER
                .segment_str(text)
                .tuple_windows()
                .map(|(i, j)| (i, &text[i..j])),
            Self::Sentence => SENTENCE_SEGMENTER
                .segment_str(text)
                .tuple_windows()
                .map(|(i, j)| (i, &text[i..j])),
        }
    }
}

#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;

    use super::*;

    #[test]
    fn first_section_within_matches_unbounded_first_section() {
        let text = "Short sentence one. Another sentence follows here. And a third.";

        for level in FallbackLevel::iter() {
            let expected = level.sections(text).next().map(|(_, s)| s);
            for window in [1, 5, 18, 19, 20, 40, text.len(), text.len() + 10] {
                let (section, truncated) = level.first_section_within(text, window).unwrap();
                if truncated {
                    assert!(
                        expected.unwrap().len() >= section.len(),
                        "{level:?} {window}"
                    );
                    assert!(text.starts_with(section));
                } else {
                    assert_eq!(Some(section), expected, "{level:?} {window}");
                }
            }
        }
    }

    #[test]
    fn first_section_within_reports_truncation_without_scanning_past_window() {
        // No sentence boundary anywhere: the first sentence is the whole text.
        let text = "https://example.com/a 2026-01-01T00:00:00+02:00   ".repeat(2_000);

        let (section, truncated) = FallbackLevel::Sentence
            .first_section_within(&text, 1_000)
            .unwrap();
        assert!(truncated);
        assert!(section.len() <= 1_000);
        assert!(text.starts_with(section));

        let (word, truncated) = FallbackLevel::Word
            .first_section_within(&text, 1_000)
            .unwrap();
        assert!(!truncated);
        assert_eq!(word, "https");
    }

    #[test]
    fn first_section_within_respects_char_boundaries() {
        let text = "é".repeat(1_000);
        for window in 1..10 {
            let (section, _) = FallbackLevel::Sentence
                .first_section_within(&text, window)
                .unwrap();
            assert!(text.is_char_boundary(section.len()));
        }
    }
}
