use std::{iter::Peekable, ops::Range, sync::LazyLock};

use auto_enums::auto_enum;
use icu_properties::{props::SentenceBreak, CodePointMapData};
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

/// The first sentence of the last suffix probed, in absolute byte offsets.
#[derive(Debug, Default)]
pub(super) struct SentenceProbe {
    range: Range<usize>,
}

impl SentenceProbe {
    pub(super) fn first<'text>(&mut self, text: &'text str, offset: usize) -> Option<&'text str> {
        let suffix = &text[offset..];
        if self.range.contains(&offset)
            && can_drop_sentence_prefix(&text[self.range.start..offset], suffix.chars().next()?)
        {
            // Only the newly discarded prefix needs checking on the next probe.
            self.range.start = offset;
            return Some(&text[self.range.clone()]);
        }
        let (_, sentence) = FallbackLevel::Sentence.sections(suffix).next()?;
        self.range = offset..offset + sentence.len();
        Some(sentence)
    }
}

/// The first word of the last suffix probed, in absolute byte offsets.
#[derive(Debug, Default)]
pub(super) struct WordProbe {
    range: Range<usize>,
}

impl WordProbe {
    /// First word of `text[offset..]`. Segmenting a long run without word
    /// breaks takes time proportional to the run, so a later offset inside the
    /// same run reuses the previous result. On a miss, peek into the caller's
    /// iterator so it can continue with the same ICU segmentation pass.
    pub(super) fn first<'text>(
        &mut self,
        text: &'text str,
        offset: usize,
        words: &mut Peekable<impl Iterator<Item = (usize, &'text str)>>,
    ) -> Option<&'text str> {
        let suffix = &text[offset..];
        // Dropping ASCII letters/digits before another one cannot change the word's end.
        // For other restarts, including on or across a non-ASCII scalar, ask ICU again.
        if self.range.contains(&offset)
            && suffix.starts_with(|c: char| c.is_ascii_alphanumeric())
            && text[self.range.start..offset]
                .bytes()
                .all(|b| b.is_ascii_alphanumeric())
        {
            // Only the newly discarded prefix needs checking on the next probe.
            self.range.start = offset;
            return Some(&text[self.range.clone()]);
        }
        let &(_, word) = words.peek()?;
        self.range = offset..offset + word.len();
        Some(word)
    }
}

fn can_drop_sentence_prefix(prefix: &str, next: char) -> bool {
    let properties = CodePointMapData::<SentenceBreak>::new();
    let letter_or_number = |class| {
        matches!(
            class,
            SentenceBreak::Upper
                | SentenceBreak::Lower
                | SentenceBreak::OLetter
                | SentenceBreak::Numeric
        )
    };
    // In the UAX #29 sentence rules, no context before a letter or number can
    // affect a break *after* it. A leading space is different: it can carry a
    // pending terminator from the discarded prefix (e.g. "Hello.  World.").
    // Only reuse at a space if the prefix was ordinary text with no such context.
    match properties.get(next) {
        SentenceBreak::Sp => prefix.chars().all(|c| {
            let class = properties.get(c);
            letter_or_number(class)
                || matches!(
                    class,
                    SentenceBreak::Sp | SentenceBreak::Extend | SentenceBreak::Format
                )
        }),
        class => letter_or_number(class),
    }
}

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
    use super::*;

    #[test]
    fn only_the_first_ascii_word_probe_advances_icu() {
        let text = "abC123def456 next";
        let mut probe = WordProbe::default();
        let mut scanned = 0;
        for offset in 0..12 {
            let mut words = FallbackLevel::Word
                .sections(&text[offset..])
                .inspect(|_| scanned += 1)
                .peekable();
            assert_eq!(
                probe.first(text, offset, &mut words),
                Some(&text[offset..12])
            );
        }
        assert_eq!(scanned, 1);
    }

    #[test]
    fn word_probes_match_fresh_icu_at_each_restart() {
        for text in [
            "",
            "abc123def  hello.world 3.14 can't a_b",
            "aBébc7 猫123abc",
            "abc\u{301}def\u{200d}ghi ٢٣abc 日本語abc",
            "ab-cd\r\nxyz.  Q9",
        ] {
            let offsets = text
                .char_indices()
                .map(|(i, _)| i)
                .chain([text.len()])
                .collect::<Vec<_>>();
            for (i, &from) in offsets.iter().enumerate() {
                for &at in &offsets[i..] {
                    let mut probe = WordProbe::default();
                    let words_at =
                        |offset| FallbackLevel::Word.sections(&text[offset..]).peekable();
                    let _ = probe.first(text, from, &mut words_at(from));
                    let suffix = &text[at..];
                    let expected = WORD_SEGMENTER
                        .segment_str(suffix)
                        .nth(1)
                        .map(|end| &suffix[..end]);
                    assert_eq!(
                        probe.first(text, at, &mut words_at(at)),
                        expected,
                        "{text:?}: {from} -> {at}"
                    );
                }
            }
        }
    }

    #[test]
    fn restarting_can_remove_or_introduce_a_sentence_boundary() {
        let mut probe = SentenceProbe::default();
        let text = "Hello.  World.";
        assert_eq!(probe.first(text, 0), Some("Hello.  "));
        assert_eq!(probe.first(text, 6), Some("  World."));

        let mut probe = SentenceProbe::default();
        let text = "A.B. Next.";
        assert_eq!(probe.first(text, 0), Some("A.B. "));
        assert_eq!(probe.first(text, 1), Some("."));
    }

    #[test]
    fn sentence_probes_match_fresh_icu_at_each_restart() {
        for text in [
            "",
            "U.S.A.  Price: 3.14!  \"Really?\"  Next.",
            "A\u{301}.\u{200d}B. e\u{301}lan.  אָב。東京！",
            "٢.٣ ¿Qué tal?\r\n\u{2029}🦀!  Καλημέρα。",
        ] {
            let offsets = text
                .char_indices()
                .map(|(i, _)| i)
                .chain([text.len()])
                .collect::<Vec<_>>();
            for (i, &from) in offsets.iter().enumerate() {
                for &at in &offsets[i..] {
                    let mut probe = SentenceProbe::default();
                    let _ = probe.first(text, from);
                    let suffix = &text[at..];
                    let expected = SENTENCE_SEGMENTER
                        .segment_str(suffix)
                        .nth(1)
                        .map(|end| &suffix[..end]);
                    assert_eq!(probe.first(text, at), expected, "{text:?}: {from} → {at}");
                }
            }
        }
    }
}
