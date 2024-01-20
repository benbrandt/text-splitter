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

use std::fs;

use fake::{Fake, Faker};
use more_asserts::assert_le;
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

#[test]
fn random_chunk_size() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    for _ in 0..100 {
        let max_characters = Faker.fake();
        let splitter = TextSplitter::default();
        let chunks = splitter.chunks(&text, max_characters).collect::<Vec<_>>();

        assert_eq!(chunks.join(""), text);
        for chunk in chunks {
            assert_le!(chunk.chars().count(), max_characters);
        }
    }
}

#[test]
fn random_chunk_range() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    for _ in 0..10 {
        let a = Faker.fake::<Option<u16>>().map(usize::from);
        let b = Faker.fake::<Option<u16>>().map(usize::from);
        let splitter = TextSplitter::default();

        let chunks = match (a, b) {
            (None, None) => splitter.chunks(&text, ..).collect::<Vec<_>>(),
            (None, Some(b)) => splitter.chunks(&text, ..b).collect::<Vec<_>>(),
            (Some(a), None) => splitter.chunks(&text, a..).collect::<Vec<_>>(),
            (Some(a), Some(b)) if b < a => splitter.chunks(&text, b..a).collect::<Vec<_>>(),
            (Some(a), Some(b)) => splitter.chunks(&text, a..=b).collect::<Vec<_>>(),
        };

        assert_eq!(chunks.join(""), text);
        let max = a.unwrap_or(usize::MIN).max(b.unwrap_or(usize::MAX));
        for chunk in chunks {
            let chars = chunk.chars().count();
            assert_le!(chars, max);
        }
    }
}

#[cfg(feature = "tokenizers")]
#[test]
fn huggingface_small_chunk_behavior() {
    let tokenizer = tokenizers::Tokenizer::from_file("./tests/alpha-03-128k.json").unwrap();
    let splitter = TextSplitter::new(tokenizer);

    let text = "notokenexistsforthisword";
    let chunks = splitter.chunks(text, 5).collect::<Vec<_>>();

    assert_eq!(chunks, ["notokenexistsforth", "isword"]);
}
