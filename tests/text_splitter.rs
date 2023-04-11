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

use std::cmp::min;

use fake::{Fake, Faker};
use text_splitter::TextSplitter;

#[test]
fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
    let text = Faker.fake::<String>();
    let splitter = TextSplitter::new(text.chars().count());
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

    let splitter = TextSplitter::new(max_chunk_size);
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
fn can_handle_unicode_characters() {
    let text = "éé"; // Char that is more than one byte
    let splitter = TextSplitter::new(1);
    let chunks = splitter.chunk_by_chars(text).collect::<Vec<_>>();
    assert_eq!(vec!["é", "é"], chunks);
}

#[test]
fn custom_len_function() {
    let text = "éé"; // Char that is two bytes each
    let splitter = TextSplitter::new(2).with_length_fn(str::len);
    let chunks = splitter.chunk_by_chars(text).collect::<Vec<_>>();
    assert_eq!(vec!["é", "é"], chunks);
}

#[test]
fn handles_char_bigger_than_len() {
    let text = "éé"; // Char that is two bytes each
    let splitter = TextSplitter::new(1).with_length_fn(str::len);
    let chunks = splitter.chunk_by_chars(text).collect::<Vec<_>>();
    // Can't squeeze it in
    assert!(chunks.is_empty());
}

#[test]
fn chunk_by_graphemes() {
    let text = "a̐éö̲\r\n";
    let splitter = TextSplitter::new(3);

    let chunks = splitter.chunk_by_graphemes(text).collect::<Vec<_>>();
    // \r\n is grouped together not separated
    assert_eq!(vec!["a̐é", "ö̲", "\r\n"], chunks);
}

#[test]
fn graphemes_fallback_to_chars() {
    let text = "a̐éö̲\r\n";
    let splitter = TextSplitter::new(1);

    let chunks = splitter.chunk_by_graphemes(text).collect::<Vec<_>>();
    assert_eq!(
        vec!["a", "\u{310}", "é", "ö", "\u{332}", "\r", "\n"],
        chunks
    );
}

#[test]
fn chunk_by_words() {
    let text = "The quick (\"brown\") fox can't jump 32.3 feet, right?";
    let splitter = TextSplitter::new(10);

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
    let splitter = TextSplitter::new(2);

    let chunks = splitter.chunk_by_words(text).collect::<Vec<_>>();
    assert_eq!(vec!["Th", "é ", "qu", "ic", "k", "\r\n"], chunks);
}
