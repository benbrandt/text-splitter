//! Chunks that hold only punctuation or markup marks should be avoided when a merge fits.
use std::{fmt::Write, fs};

#[cfg(feature = "markdown")]
use text_splitter::MarkdownSplitter;
use text_splitter::{ChunkConfig, TextSplitter};

fn has_content(chunk: &str) -> bool {
    chunk.chars().any(char::is_alphanumeric)
}

#[test]
fn closing_mark_after_full_chunk_joins_the_previous_words() {
    let text = format!("{}bb)", "aaaaa ".repeat(8));
    let chunks = TextSplitter::new(ChunkConfig::new(50))
        .chunks(&text)
        .collect::<Vec<_>>();
    assert!(chunks.iter().all(|c| has_content(c)), "{chunks:?}");
    assert_eq!(chunks.last(), Some(&"bb)"));
    assert!(chunks.iter().all(|c| c.chars().count() <= 50));
}

#[cfg(feature = "markdown")]
#[test]
fn closing_mark_after_full_chunk_joins_the_previous_words_markdown() {
    let text = format!("{}bb)", "aaaaa ".repeat(8));
    let chunks = MarkdownSplitter::new(ChunkConfig::new(50))
        .chunks(&text)
        .collect::<Vec<_>>();
    assert!(chunks.iter().all(|c| has_content(c)), "{chunks:?}");
}

#[cfg(feature = "markdown")]
#[test]
fn opening_quote_stays_with_the_sentence_that_follows() {
    let mut text = String::new();
    for i in 0..40 {
        write!(text, "CHAPTER {i}\n\n\"Well, Prince, so Genoa and Lucca are now just family estates of the Buonapartes, I warn you\"\n\n").unwrap();
    }
    for max in [30, 50] {
        let chunks = MarkdownSplitter::new(ChunkConfig::new(max))
            .chunks(&text)
            .collect::<Vec<_>>();
        let bad = chunks
            .iter()
            .filter(|c| !has_content(c))
            .collect::<Vec<_>>();
        assert!(
            bad.is_empty(),
            "max={max}: {} mark-only chunks, e.g. {:?}",
            bad.len(),
            bad.first()
        );
        assert!(chunks.iter().all(|c| c.chars().count() <= max));
    }
}

#[cfg(feature = "markdown")]
#[test]
fn typographic_bullet_stays_with_the_code_span_that_follows() {
    let mut text = String::new();
    for i in 0..40 {
        write!(
            text,
            "\u{2022}\u{a0}`alpha beta gamma delta epsilon zeta eta theta {i}`\n\n"
        )
        .unwrap();
    }
    for max in [30, 50] {
        let chunks = MarkdownSplitter::new(ChunkConfig::new(max))
            .chunks(&text)
            .collect::<Vec<_>>();
        let bad = chunks
            .iter()
            .filter(|c| !has_content(c))
            .collect::<Vec<_>>();
        assert!(
            bad.is_empty(),
            "max={max}: {} mark-only chunks, e.g. {:?}",
            bad.len(),
            bad.first()
        );
        assert!(chunks.iter().all(|c| c.chars().count() <= max));
    }
}

#[cfg(feature = "markdown")]
#[test]
fn github_flavored_spec_has_no_mark_only_chunks() {
    let text = fs::read_to_string("tests/inputs/markdown/github_flavored.md").unwrap();
    for max in [200, 1000] {
        let chunks = MarkdownSplitter::new(ChunkConfig::new(max))
            .chunks(&text)
            .collect::<Vec<_>>();
        let bad = chunks
            .iter()
            .filter(|c| !has_content(c))
            .collect::<Vec<_>>();
        assert!(
            bad.is_empty(),
            "max={max}: {} mark-only chunks, e.g. {:?}",
            bad.len(),
            bad.first()
        );
    }
}
