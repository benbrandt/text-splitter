#[cfg(feature = "markdown")]
use text_splitter::MarkdownSplitter;

#[cfg(feature = "markdown")]
#[test]
fn fallsback_to_normal_text_split_if_no_markdown_content() {
    use text_splitter::TextSplitter;

    let splitter = MarkdownSplitter::default();
    let text = "Some text\n\nfrom a\ndocument";
    let chunk_size = 10;
    let chunks = splitter.chunks(text, chunk_size).collect::<Vec<_>>();

    assert_eq!(
        TextSplitter::default()
            .chunks(text, chunk_size)
            .collect::<Vec<_>>(),
        chunks
    );
}
