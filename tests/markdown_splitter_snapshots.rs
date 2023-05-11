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

use std::fs;

use once_cell::sync::Lazy;
use text_splitter::MarkdownSplitter;
use tiktoken_rs::{cl100k_base, CoreBPE};
use tokenizers::Tokenizer;

#[test]
fn characters_default() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = MarkdownSplitter::default();
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn characters_trim() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = MarkdownSplitter::default().with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

static HUGGINGFACE_TOKENIZER: Lazy<Tokenizer> =
    Lazy::new(|| Tokenizer::from_pretrained("bert-base-cased", None).unwrap());

#[test]
fn huggingface_default() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = MarkdownSplitter::new(HUGGINGFACE_TOKENIZER.clone());
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn huggingface_trim() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter =
                MarkdownSplitter::new(HUGGINGFACE_TOKENIZER.clone()).with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

static TIKTOKEN_TOKENIZER: Lazy<CoreBPE> = Lazy::new(|| cl100k_base().unwrap());

#[test]
fn tiktoken_default() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = MarkdownSplitter::new(TIKTOKEN_TOKENIZER.clone());
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn tiktoken_trim() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = MarkdownSplitter::new(TIKTOKEN_TOKENIZER.clone()).with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            insta::assert_yaml_snapshot!(chunks);
        }
    });
}
