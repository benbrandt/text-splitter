/// Example demonstrating BPE tokenization support
/// Run with: cargo run --example bpe_example --features bpe
#[cfg(feature = "bpe")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use bpe_openai::{cl100k_base, o200k_base};
    use text_splitter::{ChunkConfig, TextSplitter};

    // Create cl100k tokenizer (used by GPT-3.5 and GPT-4)
    let cl100k_tokenizer = cl100k_base();
    println!("cl100k tokenizer loaded");

    // Create o200k tokenizer (used by GPT-4o)
    let o200k_tokenizer = o200k_base();
    println!("o200k tokenizer loaded");

    let text = "This is a sample text that we want to split using BPE tokenization. \
                The BPE tokenizer will count tokens efficiently and allow us to create \
                chunks based on token count rather than character count.";

    // Example with cl100k tokenizer
    let chunk_config = ChunkConfig::new(50).with_sizer(cl100k_tokenizer);
    let splitter = TextSplitter::new(chunk_config);
    let chunks = splitter.chunks(text);

    println!("\nUsing cl100k tokenizer (chunks limited to 50 tokens):");
    for (i, chunk) in chunks.enumerate() {
        let token_count = cl100k_base().count(chunk);
        println!("Chunk {}: {} tokens\n{}", i + 1, token_count, chunk);
        println!("---");
    }

    // Example with o200k tokenizer
    let chunk_config = ChunkConfig::new(30).with_sizer(o200k_tokenizer);
    let splitter = TextSplitter::new(chunk_config);
    let chunks = splitter.chunks(text);

    println!("\nUsing o200k tokenizer (chunks limited to 30 tokens):");
    for (i, chunk) in chunks.enumerate() {
        let token_count = o200k_base().count(chunk);
        println!("Chunk {}: {} tokens\n{}", i + 1, token_count, chunk);
        println!("---");
    }

    Ok(())
}

#[cfg(not(feature = "bpe"))]
fn main() {
    println!("This example requires the 'bpe' feature to be enabled.");
    println!("Run with: cargo run --example bpe_example --features bpe");
}