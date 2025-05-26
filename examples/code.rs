//! A simple example of using the `CodeSplitter` to split a rust file into chunks
//! based on character count.
use text_splitter::CodeSplitter;

fn main() {
    // Maximum number of characters in a chunk. Can also use a range.
    let max_characters = 5000;
    // Default implementation uses character count for chunk size.
    // Can also use all of the same tokenizer implementations as `TextSplitter`.
    let splitter = CodeSplitter::new(tree_sitter_rust::LANGUAGE, max_characters)
        .expect("Invalid tree-sitter language");

    let chunks = splitter
        .chunks(include_str!("samples/nexus_arcana.rs"))
        .collect::<Vec<_>>();
    assert_eq!(chunks.len(), 5);

    for chunk in &chunks {
        println!("\n\nChunk:\n\n{:#?}", chunk);
    }
}
