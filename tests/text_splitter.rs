use text_splitter::{ChunkConfig, TextSplitter};

#[test]
fn chunk_by_paragraphs() {
    let text = "Mr. Fox jumped.\n[...]\r\n\r\nThe dog was too lazy.";
    let splitter = TextSplitter::new(ChunkConfig::new(21).with_trim(false));

    let chunks = splitter.chunks(text).collect::<Vec<_>>();
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
    let splitter = TextSplitter::new(ChunkConfig::new(21).with_trim(false));

    let chunks = splitter.chunks(text).collect::<Vec<_>>();
    assert_eq!(vec!["Mr. Fox jumped.\n[...]", "\r\n\r\n"], chunks);
}

#[test]
fn regex_handles_empty_string() {
    let text = "";
    let splitter = TextSplitter::new(ChunkConfig::new(21).with_trim(false));

    let chunks = splitter.chunks(text).collect::<Vec<_>>();
    assert!(chunks.is_empty());
}

#[test]
fn double_newline_fallsback_to_single_and_sentences() {
    let text = "Mr. Fox jumped.\n[...]\r\n\r\nThe dog was too lazy. It just sat there.";
    let splitter = TextSplitter::new(ChunkConfig::new(18).with_trim(false));

    let chunks = splitter.chunks(text).collect::<Vec<_>>();
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
    let splitter = TextSplitter::new(10);

    let chunks = splitter.chunks(text).collect::<Vec<_>>();
    assert_eq!(vec!["Some text", "from a", "document"], chunks);
}

#[test]
fn chunk_capacity_range() {
    let text = "12345\n12345";
    let splitter = TextSplitter::new(ChunkConfig::new(5..10).with_trim(false));
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    // \n goes in the second chunk because capacity is reached after first paragraph.
    assert_eq!(vec!["12345", "\n12345"], chunks);
}

#[cfg(feature = "tokenizers")]
#[test]
fn huggingface_small_chunk_behavior() {
    let tokenizer =
        tokenizers::Tokenizer::from_file("./tests/tokenizers/huggingface.json").unwrap();
    let splitter = TextSplitter::new(ChunkConfig::new(5).with_sizer(tokenizer).with_trim(false));

    let text = "notokenexistsforthisword";
    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, ["notokenexistsforth", "isword"]);
}

#[cfg(feature = "tokenizers")]
#[test]
fn huggingface_tokenizer_with_padding() {
    let tokenizer = tokenizers::Tokenizer::from_pretrained("thenlper/gte-small", None).unwrap();
    let splitter = TextSplitter::new(ChunkConfig::new(5).with_sizer(tokenizer).with_trim(false));
    let text = "\nThis is an example text This is an example text\n";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(
        chunks,
        [
            "\n",
            "This is an example text ",
            "This is an example text\n"
        ]
    );
}

#[test]
fn chunk_overlap_characters() {
    let splitter = TextSplitter::new(ChunkConfig::new(4).with_overlap(2).unwrap());
    let text = "1234567890";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, ["1234", "3456", "5678", "7890"]);
}

#[test]
fn chunk_overlap_words() {
    let splitter = TextSplitter::new(
        ChunkConfig::new(4)
            .with_overlap(3)
            .unwrap()
            .with_trim(false),
    );
    let text = "An apple a day";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, ["An ", " ", "appl", "pple", " a ", "a ", " day"]);
}

#[test]
fn chunk_overlap_words_trim() {
    let splitter = TextSplitter::new(ChunkConfig::new(4).with_overlap(3).unwrap());
    let text = "An apple a day";

    let chunks = splitter.chunks(text).collect::<Vec<_>>();

    assert_eq!(chunks, ["An", "appl", "pple", "a", "day"]);
}
