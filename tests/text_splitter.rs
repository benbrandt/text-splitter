use std::cmp::min;

use fake::{Fake, Faker};
use text_splitter::TextSplitter;

#[test]
fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
    let text = Faker.fake::<String>();
    let splitter = TextSplitter::new(text.chars().count());
    let chunks = splitter.chunks(&text).collect::<Vec<_>>();
    assert_eq!(text, chunks[0]);
}

#[test]
fn returns_two_chunks_if_text_is_longer_than_max_chunk_size() {
    let text1 = Faker.fake::<String>();
    let text2 = Faker.fake::<String>();
    let text = format!("{text1}{text2}");
    // Round up to one above half so it goes to 2 chunks
    let max_chunk_size = text.chars().count() / 2 + 1;

    let splitter = TextSplitter::new(max_chunk_size);
    let chunks = splitter.chunks(&text).collect::<Vec<_>>();

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
}
