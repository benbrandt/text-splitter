use text_splitter::TextSplitter;

#[test]
fn chunk_by_paragraphs() {
    let text = "Mr. Fox jumped.\n[...]\r\n\r\nThe dog was too lazy.";
    let splitter = TextSplitter::default();

    let chunks = splitter.chunks(text, 21).collect::<Vec<_>>();
    assert_eq!(
        vec![
            "Mr. Fox jumped.\n[...]",
            "\r\n\r\n",
            "The dog was too lazy."
        ],
        chunks
    );
}

#[test]
fn handles_ending_on_newline() {
    let text = "Mr. Fox jumped.\n[...]\r\n\r\n";
    let splitter = TextSplitter::default();

    let chunks = splitter.chunks(text, 21).collect::<Vec<_>>();
    assert_eq!(vec!["Mr. Fox jumped.\n[...]", "\r\n\r\n"], chunks);
}

#[test]
fn regex_handles_empty_string() {
    let text = "";
    let splitter = TextSplitter::default();

    let chunks = splitter.chunks(text, 21).collect::<Vec<_>>();
    assert!(chunks.is_empty());
}

#[test]
fn double_newline_fallsback_to_single_and_sentences() {
    let text = "Mr. Fox jumped.\n[...]\r\n\r\nThe dog was too lazy. It just sat there.";
    let splitter = TextSplitter::default();

    let chunks = splitter.chunks(text, 18).collect::<Vec<_>>();
    assert_eq!(
        vec![
            "Mr. Fox jumped.\n",
            "[...]\r\n\r\n",
            "The dog was too ",
            "lazy. ",
            "It just sat there."
        ],
        chunks
    );
}

#[test]
fn trim_paragraphs() {
    let text = "Some text\n\nfrom a\ndocument";
    let splitter = TextSplitter::default().with_trim_chunks(true);

    let chunks = splitter.chunks(text, 10).collect::<Vec<_>>();
    assert_eq!(vec!["Some text", "from a", "document"], chunks);
}

#[test]
fn chunk_capacity_range() {
    let text = "12345\n12345";
    let splitter = TextSplitter::default();
    let chunks = splitter.chunks(text, 5..10).collect::<Vec<_>>();

    // \n goes in the second chunk because capacity is reached after first paragraph.
    assert_eq!(vec!["12345", "\n12345"], chunks);
}
