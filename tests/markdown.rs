use std::fs;

use fake::{Fake, Faker};
use itertools::Itertools;
use more_asserts::assert_le;
#[cfg(feature = "markdown")]
use text_splitter::MarkdownSplitter;

#[cfg(feature = "markdown")]
#[test]
fn random_chunk_size() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    for _ in 0..10 {
        let max_characters = Faker.fake();
        let splitter = MarkdownSplitter::default();
        let chunks = splitter.chunks(&text, max_characters).collect::<Vec<_>>();

        assert_eq!(chunks.join(""), text);
        for chunk in chunks {
            assert_le!(chunk.chars().count(), max_characters);
        }
    }
}

#[cfg(feature = "markdown")]
#[test]
fn random_chunk_indices_increase() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    for _ in 0..10 {
        let max_characters = Faker.fake::<usize>();
        let splitter = MarkdownSplitter::default();
        let indices = splitter
            .chunk_indices(&text, max_characters)
            .map(|(i, _)| i);

        assert!(indices.tuple_windows().all(|(a, b)| a < b));
    }
}

#[cfg(feature = "markdown")]
#[test]
fn fallsback_to_normal_text_split_if_no_markdown_content() {
    let splitter = MarkdownSplitter::default();
    let text = "Some text\n\nfrom a\ndocument";
    let chunk_size = 10;
    let chunks = splitter.chunks(text, chunk_size).collect::<Vec<_>>();

    assert_eq!(
        // ["Some text", "\n\n", "from a\n", "document"],
        ["Some text\n", "\nfrom a\n", "document"].to_vec(),
        chunks
    );
}
