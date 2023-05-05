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

use std::{cmp::min, fs};

use fake::{Fake, Faker};
use more_asserts::assert_le;
use text_splitter::{Characters, ChunkSize, TextSplitter};

#[test]
fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
    let text = Faker.fake::<String>();
    let splitter = TextSplitter::new(Characters::new(text.chars().count()));
    let chunks = splitter.chunk_by_chars(&text).collect::<Vec<_>>();
    assert_eq!(vec![&text], chunks);
}

#[test]
fn returns_two_chunks_if_text_is_longer_than_max_chunk_size() {
    let text1 = Faker.fake::<String>();
    let text2 = Faker.fake::<String>();
    let text = format!("{text1}{text2}");
    // Round up to one above half so it goes to 2 chunks
    let max_chunk_size = text.chars().count() / 2 + 1;

    let splitter = TextSplitter::new(Characters::new(max_chunk_size));
    let chunks = splitter.chunk_by_chars(&text).collect::<Vec<_>>();

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
    let splitter = TextSplitter::new(Characters::new(100));
    let chunks = splitter.chunk_by_chars(text).collect::<Vec<_>>();
    assert!(chunks.is_empty());
}

#[test]
fn can_handle_unicode_characters() {
    let text = "éé"; // Char that is more than one byte
    let splitter = TextSplitter::new(Characters::new(1));
    let chunks = splitter.chunk_by_chars(text).collect::<Vec<_>>();
    assert_eq!(vec!["é", "é"], chunks);
}

// Just for testing
struct Str {
    length: usize,
}

impl ChunkSize for Str {
    fn valid_chunk(&self, chunk: &str) -> bool {
        chunk.len() <= self.length
    }
}

#[test]
fn custom_len_function() {
    let text = "éé"; // Char that is two bytes each
    let splitter = TextSplitter::new(Str { length: 2 });
    let chunks = splitter.chunk_by_chars(text).collect::<Vec<_>>();
    assert_eq!(vec!["é", "é"], chunks);
}

#[test]
fn handles_char_bigger_than_len() {
    let text = "éé"; // Char that is two bytes each
    let splitter = TextSplitter::new(Str { length: 1 });
    let chunks = splitter.chunk_by_chars(text).collect::<Vec<_>>();
    // We can only go so small
    assert_eq!(vec!["é", "é"], chunks);
}

#[test]
fn chunk_by_graphemes() {
    let text = "a̐éö̲\r\n";
    let splitter = TextSplitter::new(Characters::new(3));

    let chunks = splitter.chunk_by_graphemes(text).collect::<Vec<_>>();
    // \r\n is grouped together not separated
    assert_eq!(vec!["a̐é", "ö̲", "\r\n"], chunks);
}

#[test]
fn graphemes_fallback_to_chars() {
    let text = "a̐éö̲\r\n";
    let splitter = TextSplitter::new(Characters::new(1));

    let chunks = splitter.chunk_by_graphemes(text).collect::<Vec<_>>();
    assert_eq!(
        vec!["a", "\u{310}", "é", "ö", "\u{332}", "\r", "\n"],
        chunks
    );
}

#[test]
fn chunk_by_words() {
    let text = "The quick (\"brown\") fox can't jump 32.3 feet, right?";
    let splitter = TextSplitter::new(Characters::new(10));

    let chunks = splitter.chunk_by_words(text).collect::<Vec<_>>();
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
    let splitter = TextSplitter::new(Characters::new(2));

    let chunks = splitter.chunk_by_words(text).collect::<Vec<_>>();
    assert_eq!(vec!["Th", "é ", "qu", "ic", "k", "\r\n"], chunks);
}

#[test]
fn chunk_by_sentences() {
    let text = "Mr. Fox jumped. [...] The dog was too lazy.";
    let splitter = TextSplitter::new(Characters::new(21));

    let chunks = splitter.chunk_by_sentences(text).collect::<Vec<_>>();
    assert_eq!(
        vec!["Mr. Fox jumped. ", "[...] ", "The dog was too lazy."],
        chunks
    );
}

#[test]
fn sentences_falls_back_to_words() {
    let text = "Mr. Fox jumped. [...] The dog was too lazy.";
    let splitter = TextSplitter::new(Characters::new(16));

    let chunks = splitter.chunk_by_sentences(text).collect::<Vec<_>>();
    assert_eq!(
        vec!["Mr. Fox jumped. ", "[...] ", "The dog was too ", "lazy."],
        chunks
    );
}

#[test]
fn chunk_by_paragraphs() {
    let text = "Mr. Fox jumped.\n[...]\r\n\r\nThe dog was too lazy.";
    let splitter = TextSplitter::new(Characters::new(21));

    let chunks = splitter.chunk_by_paragraphs(text).collect::<Vec<_>>();
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
    let splitter = TextSplitter::new(Characters::new(21));

    let chunks = splitter.chunk_by_paragraphs(text).collect::<Vec<_>>();
    assert_eq!(vec!["Mr. Fox jumped.\n[...]", "\r\n\r\n"], chunks);
}

#[test]
fn regex_handles_empty_string() {
    let text = "";
    let splitter = TextSplitter::new(Characters::new(21));

    let chunks = splitter.chunk_by_paragraphs(text).collect::<Vec<_>>();
    assert!(chunks.is_empty());
}

#[test]
fn double_newline_fallsback_to_single_and_sentences() {
    let text = "Mr. Fox jumped.\n[...]\r\n\r\nThe dog was too lazy. It just sat there.";
    let splitter = TextSplitter::new(Characters::new(18));

    let chunks = splitter.chunk_by_paragraphs(text).collect::<Vec<_>>();
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
fn trim_char_indices() {
    let text = " a b ";
    let splitter = TextSplitter::new(Characters::new(1)).with_trim_chunks(true);

    let chunks = splitter.chunk_by_char_indices(text).collect::<Vec<_>>();
    assert_eq!(vec![(1, "a"), (3, "b")], chunks);
}

#[test]
fn trim_chars() {
    let text = " a b ";
    let splitter = TextSplitter::new(Characters::new(1)).with_trim_chunks(true);

    let chunks = splitter.chunk_by_chars(text).collect::<Vec<_>>();
    assert_eq!(vec!["a", "b"], chunks);
}

#[test]
fn trim_grapheme_indices() {
    let text = "\r\na̐éö̲\r\n";
    let splitter = TextSplitter::new(Characters::new(3)).with_trim_chunks(true);

    let chunks = splitter.chunk_by_grapheme_indices(text).collect::<Vec<_>>();
    assert_eq!(vec![(2, "a̐é"), (7, "ö̲")], chunks);
}

#[test]
fn trim_graphemes() {
    let text = "\r\na̐éö̲\r\n";
    let splitter = TextSplitter::new(Characters::new(3)).with_trim_chunks(true);

    let chunks = splitter.chunk_by_graphemes(text).collect::<Vec<_>>();
    assert_eq!(vec!["a̐é", "ö̲"], chunks);
}

#[test]
fn trim_word_indices() {
    let text = "Some text from a document";
    let splitter = TextSplitter::new(Characters::new(10)).with_trim_chunks(true);

    let chunks = splitter.chunk_by_word_indices(text).collect::<Vec<_>>();
    assert_eq!(
        vec![(0, "Some text"), (10, "from a"), (17, "document")],
        chunks
    );
}

#[test]
fn trim_words() {
    let text = "Some text from a document";
    let splitter = TextSplitter::new(Characters::new(10)).with_trim_chunks(true);

    let chunks = splitter.chunk_by_words(text).collect::<Vec<_>>();
    assert_eq!(vec!["Some text", "from a", "document"], chunks);
}

#[test]
fn trim_sentence_indices() {
    let text = "Some text. From a document.";
    let splitter = TextSplitter::new(Characters::new(10)).with_trim_chunks(true);

    let chunks = splitter.chunk_by_sentence_indices(text).collect::<Vec<_>>();
    assert_eq!(
        vec![(0, "Some text."), (11, "From a"), (18, "document.")],
        chunks
    );
}

#[test]
fn trim_sentences() {
    let text = "Some text. From a document.";
    let splitter = TextSplitter::new(Characters::new(10)).with_trim_chunks(true);

    let chunks = splitter.chunk_by_sentences(text).collect::<Vec<_>>();
    assert_eq!(vec!["Some text.", "From a", "document."], chunks);
}

#[test]
fn trim_paragraph_indices() {
    let text = "Some text\n\nfrom a\ndocument";
    let splitter = TextSplitter::new(Characters::new(10)).with_trim_chunks(true);

    let chunks = splitter
        .chunk_by_paragraph_indices(text)
        .collect::<Vec<_>>();
    assert_eq!(
        vec![(0, "Some text"), (11, "from a"), (18, "document")],
        chunks
    );
}

#[test]
fn trim_paragraphs() {
    let text = "Some text\n\nfrom a\ndocument";
    let splitter = TextSplitter::new(Characters::new(10)).with_trim_chunks(true);

    let chunks = splitter.chunk_by_paragraphs(text).collect::<Vec<_>>();
    assert_eq!(vec!["Some text", "from a", "document"], chunks);
}

#[test]
fn random_chunk_size() {
    let text = fs::read_to_string("tests/texts/room_with_a_view.txt").unwrap();

    for _ in 0..100 {
        let max_characters = Faker.fake();
        let splitter = TextSplitter::new(Characters::new(max_characters));
        let chunks = splitter.chunk_by_paragraphs(&text).collect::<Vec<_>>();

        assert_eq!(chunks.join(""), text);
        for chunk in chunks {
            assert_le!(chunk.chars().count(), max_characters);
        }
    }
}
