#![warn(
    clippy::pedantic,
    future_incompatible,
    missing_debug_implementations,
    missing_docs,
    nonstandard_style,
    rust_2018_compatibility,
    rust_2018_idioms,
    rust_2021_compatibility,
    unused
)]

use text_splitter::MarkdownSplitter;

#[test]
fn fallsback_to_normal_text_split_if_no_markdown_content() {
    let splitter = MarkdownSplitter::default();
    let text = "Some text\n\nfrom a\ndocument";
    let chunks = splitter.chunks(text, 10).collect::<Vec<_>>();

    assert_eq!(vec!["Some text", "\n\nfrom a\n", "document"], chunks);
}

#[test]
fn splits_on_horizontal_rules() {
    let splitter = MarkdownSplitter::default();
    let text = "Text\n\nfrom\n\n---\n\na document";
    let chunks = splitter.chunks(text, 12).collect::<Vec<_>>();

    assert_eq!(vec!["Text\n\nfrom\n\n", "---\n", "\na document"], chunks);
}

#[test]
fn horizontal_rule_with_trim() {
    let splitter = MarkdownSplitter::default().with_trim_chunks(true);
    let text = "Text\n\nfrom\n\n---\n\na document";
    let chunks = splitter.chunks(text, 12).collect::<Vec<_>>();

    assert_eq!(vec!["Text\n\nfrom", "---", "a document"], chunks);
}

#[test]
fn headings() {
    let splitter = MarkdownSplitter::default();
    for heading in ["#", "##", "###", "####", "#####", "######"] {
        let text = format!("{heading} Heading\nParagraph\n{heading} Heading\nParagraph");
        let chunks = splitter
            .chunks(&text, 32 + heading.chars().count())
            .collect::<Vec<_>>();

        // Chunk size big enough to grab next heading, but it should be in the next chunk
        assert_eq!(
            vec![
                format!("{heading} Heading\nParagraph\n"),
                format!("{heading} Heading\nParagraph")
            ],
            chunks
        );
    }
}

#[test]
fn headings_trim() {
    let splitter = MarkdownSplitter::default().with_trim_chunks(true);
    for heading in ["#", "##", "###", "####", "#####", "######"] {
        let text = format!("{heading} Heading\nParagraph\n{heading} Heading\nParagraph");
        let chunks = splitter
            .chunks(&text, 32 + heading.chars().count())
            .collect::<Vec<_>>();

        // Chunk size big enough to grab next heading, but it should be in the next chunk
        assert_eq!(
            vec![
                format!("{heading} Heading\nParagraph"),
                format!("{heading} Heading\nParagraph")
            ],
            chunks
        );
    }
}
