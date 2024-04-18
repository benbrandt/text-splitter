use std::{fs, ops::RangeInclusive};

use fake::{Fake, Faker};
use itertools::Itertools;
use more_asserts::assert_le;
use once_cell::sync::Lazy;
#[cfg(feature = "rust-tokenizers")]
use rust_tokenizers::tokenizer::BertTokenizer;
#[cfg(feature = "markdown")]
use text_splitter::MarkdownSplitter;
use text_splitter::{Characters, ChunkConfig, ChunkSizer, TextSplitter};
#[cfg(feature = "tiktoken-rs")]
use tiktoken_rs::{cl100k_base, CoreBPE};
#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer;

#[test]
fn random_chunk_size() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    for _ in 0..10 {
        let max_characters = Faker.fake();
        let splitter = TextSplitter::new(ChunkConfig::new(max_characters).with_trim(false));
        let chunks = splitter.chunks(&text).collect::<Vec<_>>();

        assert_eq!(chunks.join(""), text);
        for chunk in chunks {
            assert_le!(chunk.chars().count(), max_characters);
        }
    }
}

#[test]
fn random_chunk_indices_increase() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    for _ in 0..10 {
        let max_characters = Faker.fake::<usize>();
        let splitter = TextSplitter::new(ChunkConfig::new(max_characters).with_trim(false));
        let indices = splitter.chunk_indices(&text).map(|(i, _)| i);

        assert!(indices.tuple_windows().all(|(a, b)| a < b));
    }
}

#[test]
fn random_chunk_range() {
    let text = fs::read_to_string("tests/inputs/text/room_with_a_view.txt").unwrap();

    for _ in 0..10 {
        let a = Faker.fake::<Option<u16>>().map(usize::from);
        let b = Faker.fake::<Option<u16>>().map(usize::from);

        let chunks = match (a, b) {
            (None, None) => TextSplitter::new(ChunkConfig::new(..).with_trim(false))
                .chunks(&text)
                .collect::<Vec<_>>(),
            (None, Some(b)) => TextSplitter::new(ChunkConfig::new(..b).with_trim(false))
                .chunks(&text)
                .collect::<Vec<_>>(),
            (Some(a), None) => TextSplitter::new(ChunkConfig::new(a..).with_trim(false))
                .chunks(&text)
                .collect::<Vec<_>>(),
            (Some(a), Some(b)) if b < a => {
                TextSplitter::new(ChunkConfig::new(b..a).with_trim(false))
                    .chunks(&text)
                    .collect::<Vec<_>>()
            }
            (Some(a), Some(b)) => TextSplitter::new(ChunkConfig::new(a..=b).with_trim(false))
                .chunks(&text)
                .collect::<Vec<_>>(),
        };

        assert_eq!(chunks.join(""), text);
        let max = a.unwrap_or(usize::MIN).max(b.unwrap_or(usize::MAX));
        for chunk in chunks {
            let chars = chunk.chars().count();
            assert_le!(chars, max);
        }
    }
}

const CHUNK_SIZES: [usize; 4] = [16, 256, 4096, 65536];
const RANGE_CHUNK_SIZES: [RangeInclusive<usize>; 2] = [64..=512, 512..=4096];

#[test]
fn characters_no_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size).with_trim(false);
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(Characters.chunk_size(chunk, &capacity).fits().is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn characters_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size);
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(Characters.chunk_size(chunk, &capacity).fits().is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn characters_range() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for range in RANGE_CHUNK_SIZES {
            let config = ChunkConfig::new(range.clone()).with_trim(false);
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(Characters.chunk_size(chunk, &capacity).fits().is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn characters_range_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for range in RANGE_CHUNK_SIZES {
            let config = ChunkConfig::new(range.clone());
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(Characters.chunk_size(chunk, &capacity).fits().is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(feature = "rust-tokenizers")]
static BERT_UNCASED_TOKENIZER: Lazy<BertTokenizer> = Lazy::new(|| {
    let path: &str = "tests/tokenizers/bert-uncased-vocab.txt";
    BertTokenizer::from_file(path, false, false).unwrap()
});

#[cfg(feature = "rust-tokenizers")]
#[test]
fn rust_tokenizers() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let splitter = TextSplitter::new(&*BERT_UNCASED_TOKENIZER);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(BERT_UNCASED_TOKENIZER
                    .chunk_size(chunk, &chunk_size)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(feature = "rust-tokenizers")]
#[test]
fn rust_tokenizers_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let splitter = TextSplitter::new(&*BERT_UNCASED_TOKENIZER).with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(BERT_UNCASED_TOKENIZER
                    .chunk_size(chunk, &chunk_size)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}
#[cfg(feature = "tokenizers")]
static HUGGINGFACE_TOKENIZER: Lazy<Tokenizer> =
    Lazy::new(|| Tokenizer::from_pretrained("bert-base-cased", None).unwrap());

#[cfg(feature = "tokenizers")]
#[test]
fn huggingface_no_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(&*HUGGINGFACE_TOKENIZER)
                .with_trim(false);
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(HUGGINGFACE_TOKENIZER
                    .chunk_size(chunk, &capacity)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(feature = "tokenizers")]
#[test]
fn huggingface_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size).with_sizer(&*HUGGINGFACE_TOKENIZER);
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(HUGGINGFACE_TOKENIZER
                    .chunk_size(chunk, &capacity)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(feature = "tiktoken-rs")]
static TIKTOKEN_TOKENIZER: Lazy<CoreBPE> = Lazy::new(|| cl100k_base().unwrap());

#[cfg(feature = "tiktoken-rs")]
#[test]
fn tiktoken_no_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(&*TIKTOKEN_TOKENIZER)
                .with_trim(false);
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(TIKTOKEN_TOKENIZER
                    .chunk_size(chunk, &capacity)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(feature = "tiktoken-rs")]
#[test]
fn tiktoken_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size).with_sizer(&*TIKTOKEN_TOKENIZER);
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(TIKTOKEN_TOKENIZER
                    .chunk_size(chunk, &capacity)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(feature = "markdown")]
#[test]
fn markdown() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size).with_trim(false);
            let capacity = *config.capacity();
            let splitter = MarkdownSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(Characters.chunk_size(chunk, &capacity).fits().is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(all(feature = "markdown", feature = "rust-tokenizers"))]
#[test]
fn rust_tokenizers_markdown() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let splitter = MarkdownSplitter::new(&*BERT_UNCASED_TOKENIZER);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(BERT_UNCASED_TOKENIZER
                    .chunk_size(chunk, &chunk_size)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(all(feature = "markdown", feature = "rust-tokenizers"))]
#[test]
fn rust_tokenizers_markdown_trim() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let splitter = MarkdownSplitter::new(&*BERT_UNCASED_TOKENIZER).with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(BERT_UNCASED_TOKENIZER
                    .chunk_size(chunk, &chunk_size)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}
#[cfg(feature = "tokenizers")]
#[test]
fn markdown_trim() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size);
            let capacity = *config.capacity();
            let splitter = MarkdownSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(Characters.chunk_size(chunk, &capacity).fits().is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(all(feature = "markdown", feature = "tokenizers"))]
#[test]
fn huggingface_markdown() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(&*HUGGINGFACE_TOKENIZER)
                .with_trim(false);
            let capacity = *config.capacity();
            let splitter = MarkdownSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(HUGGINGFACE_TOKENIZER
                    .chunk_size(chunk, &capacity)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(all(feature = "markdown", feature = "tokenizers"))]
#[test]
fn huggingface_markdown_trim() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size).with_sizer(&*HUGGINGFACE_TOKENIZER);
            let capacity = *config.capacity();
            let splitter = MarkdownSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(HUGGINGFACE_TOKENIZER
                    .chunk_size(chunk, &capacity)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(all(feature = "markdown", feature = "tiktoken-rs"))]
#[test]
fn tiktoken_markdown() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(&*TIKTOKEN_TOKENIZER)
                .with_trim(false);
            let capacity = *config.capacity();
            let splitter = MarkdownSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(TIKTOKEN_TOKENIZER
                    .chunk_size(chunk, &capacity)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[cfg(all(feature = "markdown", feature = "tiktoken-rs"))]
#[test]
fn tiktoken_markdown_trim() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in CHUNK_SIZES {
            let config = ChunkConfig::new(chunk_size).with_sizer(&*TIKTOKEN_TOKENIZER);
            let capacity = *config.capacity();
            let splitter = MarkdownSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(TIKTOKEN_TOKENIZER
                    .chunk_size(chunk, &capacity)
                    .fits()
                    .is_le());
            }
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}
