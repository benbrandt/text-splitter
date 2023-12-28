use std::fs;

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
                assert!(Characters.chunk_size(chunk, &chunk_size).fits().is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn characters_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = TextSplitter::default().with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            for chunk in chunks.iter() {
                assert!(Characters.chunk_size(chunk, &chunk_size).fits().is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn characters_range() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for range in [500..=2000, 200..=1000] {
            let splitter = TextSplitter::default();
            let chunks = splitter.chunks(&text, range.clone()).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in chunks.iter() {
                assert!(Characters.chunk_size(chunk, &range).fits().is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn characters_range_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for range in [500..=2000, 200..=1000] {
            let splitter = TextSplitter::default().with_trim_chunks(true);
            let chunks = splitter.chunks(&text, range.clone()).collect::<Vec<_>>();

            for chunk in chunks.iter() {
                assert!(Characters.chunk_size(chunk, &range).fits().is_le());
            }
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
            let splitter = TextSplitter::new(&*HUGGINGFACE_TOKENIZER);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in chunks.iter() {
                assert!(HUGGINGFACE_TOKENIZER
                    .chunk_size(chunk, &chunk_size)
                    .fits()
                    .is_le());
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
            let splitter = TextSplitter::new(&*HUGGINGFACE_TOKENIZER).with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            for chunk in chunks.iter() {
                assert!(HUGGINGFACE_TOKENIZER
                    .chunk_size(chunk, &chunk_size)
                    .fits()
                    .is_le());
            }
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
            let splitter = TextSplitter::new(&*TIKTOKEN_TOKENIZER);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in chunks.iter() {
                assert!(TIKTOKEN_TOKENIZER
                    .chunk_size(chunk, &chunk_size)
                    .fits()
                    .is_le());
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
            let splitter = TextSplitter::new(&*TIKTOKEN_TOKENIZER).with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            for chunk in chunks.iter() {
                assert!(TIKTOKEN_TOKENIZER
                    .chunk_size(chunk, &chunk_size)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}
