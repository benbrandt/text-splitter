use fake::{Fake, Faker};
use text_splitter::TextSplitter;

#[test]
fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
    let text = Faker.fake::<String>();
    let splitter = TextSplitter::new(text.chars().count());
    let chunks = splitter.chunks(&text);
    assert_eq!(text, chunks[0]);
}
