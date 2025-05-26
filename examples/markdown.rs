//! A simple example of using the `MarkdownSplitter` to split a markdown document into chunks
//! based on character count.
use text_splitter::MarkdownSplitter;

fn main() {
    // Maximum number of characters in a chunk
    let max_characters = 1000;
    // Default implementation uses character count for chunk size
    let splitter = MarkdownSplitter::new(max_characters);

    let chunks = splitter
        .chunks(include_str!("samples/nexus_arcana.md"))
        .collect::<Vec<_>>();

    assert_eq!(chunks.len(), 6);

    for chunk in &chunks {
        // markdown chunks usually start with a header
        assert!(chunk.starts_with("#"));
    }
    println!("chunks: {:#?}", chunks);
}
