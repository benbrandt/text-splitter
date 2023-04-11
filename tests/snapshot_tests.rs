use std::fs;

use text_splitter::TextSplitter;

#[test]
fn paragraph_snapshot() {
    let text = fs::read_to_string("tests/room_with_a_view.txt").unwrap();
    let splitter = TextSplitter::new(1000);
    let chunks = splitter.chunk_by_paragraphs(&text).collect::<Vec<_>>();
    insta::assert_yaml_snapshot!(chunks);
}
