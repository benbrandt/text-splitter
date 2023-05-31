use std::fs;

use more_asserts::assert_le;
use once_cell::sync::Lazy;
use text_splitter::{Characters, ChunkSizer, TextSplitter};
use tiktoken_rs::{cl100k_base, CoreBPE};
use tokenizers::Tokenizer;

#[test]
fn characters_default() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = TextSplitter::default();
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in chunks.iter() {
                assert_le!(Characters.chunk_size(chunk), chunk_size);
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn characters_range() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        let splitter = TextSplitter::default();
        let chunks = splitter.chunks(&text, 500..=2000).collect::<Vec<_>>();

        assert_eq!(chunks.join(""), text);
        for chunk in chunks.iter() {
            assert_le!(Characters.chunk_size(chunk), 2000);
        }
        insta::assert_yaml_snapshot!(chunks);
    });
}

#[test]
fn characters_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = TextSplitter::default().with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

static HUGGINGFACE_TOKENIZER: Lazy<Tokenizer> =
    Lazy::new(|| Tokenizer::from_pretrained("bert-base-cased", None).unwrap());

#[test]
fn huggingface_default() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = TextSplitter::new(HUGGINGFACE_TOKENIZER.clone());
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in chunks.iter() {
                assert_le!(HUGGINGFACE_TOKENIZER.chunk_size(chunk), chunk_size);
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn huggingface_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = TextSplitter::new(HUGGINGFACE_TOKENIZER.clone()).with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

static TIKTOKEN_TOKENIZER: Lazy<CoreBPE> = Lazy::new(|| cl100k_base().unwrap());

#[test]
fn tiktoken_default() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = TextSplitter::new(TIKTOKEN_TOKENIZER.clone());
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in chunks.iter() {
                assert_le!(TIKTOKEN_TOKENIZER.chunk_size(chunk), chunk_size);
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn tiktoken_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = TextSplitter::new(TIKTOKEN_TOKENIZER.clone()).with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            insta::assert_yaml_snapshot!(chunks);
        }
    });
}
