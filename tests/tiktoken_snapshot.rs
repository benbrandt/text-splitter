use std::fs;

use once_cell::sync::Lazy;
use text_splitter::{TextSplitter, Tokens};
use tiktoken_rs::{cl100k_base, CoreBPE};

static TOKENIZER: Lazy<CoreBPE> = Lazy::new(|| cl100k_base().unwrap());

#[test]
fn tiktoken_paragraph_long_chunk() {
    let text = fs::read_to_string("tests/texts/room_with_a_view.txt").unwrap();
    let splitter = TextSplitter::new(Tokens::new(TOKENIZER.clone(), 1000));
    let chunks = splitter.chunk_by_paragraphs(&text).collect::<Vec<_>>();
    assert_eq!(chunks.join(""), text);
    insta::assert_yaml_snapshot!(chunks);
}

#[test]
fn tiktoken_paragraph_short_chunk() {
    let text = fs::read_to_string("tests/texts/room_with_a_view.txt").unwrap();
    let splitter = TextSplitter::new(Tokens::new(TOKENIZER.clone(), 100));
    let chunks = splitter.chunk_by_paragraphs(&text).collect::<Vec<_>>();
    assert_eq!(chunks.join(""), text);
    insta::assert_yaml_snapshot!(chunks);
}

#[test]
fn tiktoken_paragraph_tiny_chunk() {
    let text = fs::read_to_string("tests/texts/room_with_a_view.txt").unwrap();
    let splitter = TextSplitter::new(Tokens::new(TOKENIZER.clone(), 10));
    let chunks = splitter.chunk_by_paragraphs(&text).collect::<Vec<_>>();
    assert_eq!(chunks.join(""), text);
    insta::assert_yaml_snapshot!(chunks);
}

#[test]
fn tiktoken_paragraph_long_chunk_trim() {
    let text = fs::read_to_string("tests/texts/room_with_a_view.txt").unwrap();
    let splitter = TextSplitter::new(Tokens::new(TOKENIZER.clone(), 1000)).with_trim_chunks(true);
    let chunks = splitter.chunk_by_paragraphs(&text).collect::<Vec<_>>();
    insta::assert_yaml_snapshot!(chunks);
}

#[test]
fn tiktoken_paragraph_short_chunk_trim() {
    let text = fs::read_to_string("tests/texts/room_with_a_view.txt").unwrap();
    let splitter = TextSplitter::new(Tokens::new(TOKENIZER.clone(), 100)).with_trim_chunks(true);
    let chunks = splitter.chunk_by_paragraphs(&text).collect::<Vec<_>>();
    insta::assert_yaml_snapshot!(chunks);
}

#[test]
fn tiktoken_paragraph_tiny_chunk_trim() {
    let text = fs::read_to_string("tests/texts/room_with_a_view.txt").unwrap();
    let splitter = TextSplitter::new(Tokens::new(TOKENIZER.clone(), 10)).with_trim_chunks(true);
    let chunks = splitter.chunk_by_paragraphs(&text).collect::<Vec<_>>();
    insta::assert_yaml_snapshot!(chunks);
}
