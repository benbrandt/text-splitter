//! A simple example of using the `TextSplitter` to split a plain text document into chunks
//! based on character count.
use text_splitter::TextSplitter;

fn main() {
    // Maximum number of characters in a chunk
    let max_characters = 1000;
    // Default implementation uses character count for chunk size
    let splitter = TextSplitter::new(max_characters);

    let chunks = splitter
        .chunks(include_str!("samples/nexus_arcana.md"))
        .collect::<Vec<_>>();

    assert_eq!(chunks.len(), 5);

    // note how the final chunk does not contain the header,
    // this is a loss of information so for markdown files MarkdownSplitter is preferred
    println!("chunks: {:#?}", chunks);
}
