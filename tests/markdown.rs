//! Test for `MarkdownSplitter` behavior.
use std::fs;

use fake::{Fake, Faker};
use itertools::Itertools;
use more_asserts::assert_le;
#[cfg(feature = "markdown")]
use text_splitter::{ChunkConfig, ChunkSizer, MarkdownSplitter};

#[cfg(feature = "markdown")]
#[test]
fn random_chunk_size() {
    let text = fs::read_to_string("tests/inputs/markdown/github_flavored.md").unwrap();

    for _ in 0..10 {
        let max_characters = Faker.fake();
        let splitter = MarkdownSplitter::new(ChunkConfig::new(max_characters).with_trim(false));
        let chunks = splitter.chunks(&text).collect::<Vec<_>>();

        assert_eq!(chunks.join(""), text);
        for chunk in chunks {
            assert_le!(chunk.chars().count(), max_characters);
        }
    }
}

#[cfg(feature = "markdown")]
#[test]
fn random_chunk_indices_increase() {
    let text = fs::read_to_string("tests/inputs/markdown/github_flavored.md").unwrap();

    for _ in 0..10 {
        let max_characters = Faker.fake::<usize>();
        let splitter = MarkdownSplitter::new(ChunkConfig::new(max_characters).with_trim(false));
        let indices = splitter.chunk_indices(&text).map(|(i, _)| i);

        assert!(indices.tuple_windows().all(|(a, b)| a < b));
    }
}

#[cfg(feature = "markdown")]
#[test]
fn fallsback_to_normal_text_split_if_no_markdown_content() {
    let chunk_config = ChunkConfig::new(10).with_trim(false);
    let splitter = MarkdownSplitter::new(chunk_config);
    let text = "Some text\n\nfrom a\ndocument";
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(["Some text\n", "\nfrom a\n", "document"].to_vec(), chunks);
}

#[cfg(feature = "markdown")]
#[test]
fn split_by_rule() {
    let splitter = MarkdownSplitter::new(ChunkConfig::new(12).with_trim(false));
    let text = "Some text\n\n---\n\nwith a rule";
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(["Some text\n\n", "---\n", "\nwith a rule"].to_vec(), chunks);
}

#[cfg(feature = "markdown")]
#[test]
fn split_by_rule_trim() {
    let splitter = MarkdownSplitter::new(12);
    let text = "Some text\n\n---\n\nwith a rule";
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(["Some text", "---", "with a rule"].to_vec(), chunks);
}

#[cfg(feature = "markdown")]
#[test]
fn split_by_headers() {
    let splitter = MarkdownSplitter::new(ChunkConfig::new(30).with_trim(false));
    let text = "# Header 1\n\nSome text\n\n## Header 2\n\nwith headings\n";
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(
        [
            "# Header 1\n\nSome text\n\n",
            "## Header 2\n\nwith headings\n"
        ]
        .to_vec(),
        chunks
    );
}

#[cfg(feature = "markdown")]
#[test]
fn subheadings_grouped_with_top_header() {
    let splitter = MarkdownSplitter::new(ChunkConfig::new(60).with_trim(false));
    let text = "# Header 1\n\nSome text\n\n## Header 2\n\nwith headings\n\n### Subheading\n\nand more text\n";
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(
        [
            "# Header 1\n\nSome text\n\n",
            "## Header 2\n\nwith headings\n\n### Subheading\n\nand more text\n"
        ]
        .to_vec(),
        chunks
    );
}

#[cfg(feature = "markdown")]
#[test]
fn trimming_doesnt_trim_block_level_indentation_if_multiple_items() {
    let splitter = MarkdownSplitter::new(48);
    let text = "* Really long list item that is too big to fit\n\n  * Some Indented Text\n\n  * More Indented Text\n\n";
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(
        [
            "* Really long list item that is too big to fit",
            "  * Some Indented Text\n\n  * More Indented Text"
        ]
        .to_vec(),
        chunks
    );
}

#[cfg(feature = "markdown")]
#[test]
fn trimming_does_trim_block_level_indentation_if_only_one_item() {
    let splitter = MarkdownSplitter::new(30);
    let text = "1. Really long list item\n\n  1. Some Indented Text\n\n  2. More Indented Text\n\n";
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(
        [
            "1. Really long list item",
            "1. Some Indented Text",
            "2. More Indented Text"
        ]
        .to_vec(),
        chunks
    );
}

#[cfg(feature = "markdown")]
#[test]
fn distant_inline_boundary_does_not_remeasure_the_whole_gap() {
    struct BytesSized(std::cell::Cell<usize>);

    impl ChunkSizer for BytesSized {
        fn size(&self, chunk: &str) -> usize {
            self.0.set(self.0.get() + chunk.len());
            chunk.chars().count()
        }
    }

    let entry = "https://example.com/path/to/page 2026-01-01T00:00:00+02:00   ";
    // Both smart punctuation and an actual link produce a distant parsed range.
    for (marker, overlap, trim) in [
        ("https://example.com/lynk--co ", 0, true),
        ("[次](https://example.com/) ", 100, false),
    ] {
        let text = format!("{}{marker}{}", entry.repeat(4_000), entry.repeat(100));
        let bytes = BytesSized(std::cell::Cell::new(0));
        let config = ChunkConfig::new(1_000)
            .with_sizer(&bytes)
            .with_overlap(overlap)
            .unwrap()
            .with_trim(trim);
        let splitter = MarkdownSplitter::new(config);
        let chunks = splitter.chunks(&text).collect::<Vec<_>>();
        assert!(chunks.len() > 200);
        assert!(chunks.iter().all(|c| c.chars().count() <= 1_000));
        assert!(
            bytes.0.get() < 50 * text.len(),
            "{} bytes sized for {} input bytes",
            bytes.0.get(),
            text.len()
        );
    }
}

#[cfg(feature = "markdown")]
#[test]
fn tail_of_a_split_section_can_merge_with_following_markup() {
    let text = format!("{}**x**", "aa ".repeat(19));
    let splitter = MarkdownSplitter::new(ChunkConfig::new(10).with_trim(false));
    let chunks = splitter.chunk_indices(&text).collect::<Vec<_>>();
    assert_eq!(chunks.last(), Some(&(54, "aa **x**")));
}

#[cfg(feature = "markdown")]
#[test]
fn rechecks_a_fitting_tail_for_ranges_or_an_oversized_fallback_chunk() {
    struct TailSizer<'a>(&'a str);

    impl ChunkSizer for TailSizer<'_> {
        fn size(&self, chunk: &str) -> usize {
            if chunk == self.0 {
                1
            } else if chunk.starts_with('z') {
                chunk.len().max(11)
            } else {
                chunk.len()
            }
        }
    }

    let tail = "zabc ".repeat(24);
    let text = format!("abcd abcd {tail}--end");
    // At max 10 no fallback chunk starting at 'z' fits. At max 11 a short one
    // does fit, but a ranged capacity should still prefer the entire semantic tail.
    for capacity in [10..=10, 10..=11] {
        let splitter = MarkdownSplitter::new(
            ChunkConfig::new(capacity)
                .with_sizer(TailSizer(&tail))
                .with_trim(false),
        );
        assert_eq!(
            splitter.chunks(&text).collect::<Vec<_>>(),
            ["abcd abcd ", tail.as_str(), "--end"]
        );
    }
}
