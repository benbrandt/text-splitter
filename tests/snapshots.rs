//! Snapshot tests for regressions in chunk output.
use std::{fs, ops::RangeInclusive};

use rayon::prelude::*;
use strum::{Display, EnumIter, IntoEnumIterator};
#[cfg(feature = "code")]
use text_splitter::CodeSplitter;
#[cfg(feature = "markdown")]
use text_splitter::MarkdownSplitter;
use text_splitter::{Characters, ChunkConfig, ChunkSizer, TextSplitter};
#[cfg(feature = "tiktoken-rs")]
use tiktoken_rs::{cl100k_base, CoreBPE};
#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer;

const CHUNK_SIZES: [usize; 3] = [32, 512, 8192];
const RANGE_CHUNK_SIZES: [RangeInclusive<usize>; 2] = [64..=512, 512..=4096];

#[cfg(feature = "tokenizers")]
static HUGGINGFACE_TOKENIZER: std::sync::LazyLock<Tokenizer> =
    std::sync::LazyLock::new(|| Tokenizer::from_pretrained("bert-base-cased", None).unwrap());

#[cfg(feature = "tiktoken-rs")]
static TIKTOKEN_TOKENIZER: std::sync::LazyLock<CoreBPE> =
    std::sync::LazyLock::new(|| cl100k_base().unwrap());

#[derive(Copy, Clone, Display, EnumIter)]
enum SizerOption {
    Characters,
    #[cfg(feature = "tokenizers")]
    Tokenizers,
    #[cfg(feature = "tiktoken-rs")]
    TikToken,
}

impl ChunkSizer for SizerOption {
    fn size(&self, chunk: &str) -> usize {
        match self {
            Self::Characters => Characters.size(chunk),
            #[cfg(feature = "tokenizers")]
            Self::Tokenizers => HUGGINGFACE_TOKENIZER.size(chunk),
            #[cfg(feature = "tiktoken-rs")]
            Self::TikToken => TIKTOKEN_TOKENIZER.size(chunk),
        }
    }
}

#[test]
fn trim_false() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            SizerOption::iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .for_each(|sizer| {
                    let config = ChunkConfig::new(chunk_size)
                        .with_sizer(sizer)
                        .with_trim(false);
                    let capacity = *config.capacity();
                    let splitter = TextSplitter::new(config);
                    let chunks = splitter.chunks(&text).collect::<Vec<_>>();

                    assert_eq!(chunks.join(""), text);
                    for chunk in &chunks {
                        assert!(capacity.fits(sizer.size(chunk)).is_le());
                    }
                    insta::assert_yaml_snapshot!(
                        format!(
                            "{}_{sizer}_trim_false_{chunk_size}",
                            path.file_stem().unwrap().to_string_lossy()
                        ),
                        chunks
                    );
                });
        });
    });
}

#[test]
fn trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            SizerOption::iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .for_each(|sizer| {
                    let config = ChunkConfig::new(chunk_size).with_sizer(sizer);
                    let capacity = *config.capacity();
                    let splitter = TextSplitter::new(config);
                    let chunks = splitter.chunks(&text).collect::<Vec<_>>();

                    for chunk in &chunks {
                        assert!(capacity.fits(sizer.size(chunk)).is_le());
                    }
                    insta::assert_yaml_snapshot!(
                        format!(
                            "{}_{sizer}_trim_{chunk_size}",
                            path.file_stem().unwrap().to_string_lossy()
                        ),
                        chunks
                    );
                });
        });
    });
}

#[test]
fn range_trim_false() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        RANGE_CHUNK_SIZES.into_par_iter().for_each(|range| {
            let config = ChunkConfig::new(range.clone()).with_trim(false);
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(capacity.fits(Characters.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_Characters_trim_false_range_{range:?}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[test]
fn range_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        RANGE_CHUNK_SIZES.into_par_iter().for_each(|range| {
            let config = ChunkConfig::new(range.clone());
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(capacity.fits(Characters.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_Characters_trim_range_{range:?}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[test]
fn overlap_trim_false() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            let overlap = chunk_size / 2;
            let sizer = Characters;
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(sizer)
                .with_overlap(overlap)
                .unwrap()
                .with_trim(false);
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(capacity.fits(sizer.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_Characters_trim_false_{chunk_size}_overlap_{overlap}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[test]
fn overlap_trim() {
    insta::glob!("inputs/text/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            let sizer = Characters;
            let overlap = chunk_size / 2;
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(sizer)
                .with_overlap(overlap)
                .unwrap();
            let capacity = *config.capacity();
            let splitter = TextSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(capacity.fits(sizer.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_Characters_trim_{chunk_size}_overlap_{overlap}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[cfg(feature = "markdown")]
#[test]
fn markdown_trim_false() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            let sizer = Characters;
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(sizer)
                .with_trim(false);
            let capacity = *config.capacity();
            let splitter = MarkdownSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(capacity.fits(sizer.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_markdown_Characters_trim_false_{chunk_size}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[cfg(feature = "markdown")]
#[test]
fn markdown_trim() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            let sizer = Characters;
            let config = ChunkConfig::new(chunk_size).with_sizer(sizer);
            let capacity = *config.capacity();
            let splitter = MarkdownSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(capacity.fits(sizer.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_markdown_Characters_trim_{chunk_size}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[cfg(feature = "markdown")]
#[test]
fn markdown_overlap_trim_false() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            let overlap = chunk_size / 2;
            let sizer = Characters;
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(sizer)
                .with_overlap(overlap)
                .unwrap()
                .with_trim(false);
            let capacity = *config.capacity();
            let splitter = MarkdownSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(capacity.fits(sizer.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_markdown_Characters_trim_false_{chunk_size}_overlap_{overlap}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[cfg(feature = "markdown")]
#[test]
fn markdown_overlap_trim() {
    insta::glob!("inputs/markdown/*.md", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            let sizer = Characters;
            let overlap = chunk_size / 2;
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(sizer)
                .with_overlap(overlap)
                .unwrap();
            let capacity = *config.capacity();
            let splitter = MarkdownSplitter::new(config);
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(capacity.fits(sizer.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_markdown_Characters_trim_{chunk_size}_overlap_{overlap}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[cfg(feature = "code")]
#[test]
fn code_trim_false() {
    insta::glob!("inputs/code/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            let sizer = Characters;
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(sizer)
                .with_trim(false);
            let capacity = *config.capacity();
            let splitter = CodeSplitter::new(tree_sitter_rust::LANGUAGE, config).unwrap();
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            for chunk in &chunks {
                assert!(capacity.fits(sizer.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_code_Characters_trim_false_{chunk_size}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[cfg(feature = "code")]
#[test]
fn code_trim() {
    insta::glob!("inputs/code/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            let sizer = Characters;
            let config = ChunkConfig::new(chunk_size).with_sizer(sizer);
            let capacity = *config.capacity();
            let splitter = CodeSplitter::new(tree_sitter_rust::LANGUAGE, config).unwrap();
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(capacity.fits(sizer.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_code_Characters_trim_{chunk_size}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[cfg(feature = "code")]
#[test]
fn code_overlap_trim_false() {
    insta::glob!("inputs/code/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            let overlap = chunk_size / 2;
            let sizer = Characters;
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(sizer)
                .with_overlap(overlap)
                .unwrap()
                .with_trim(false);
            let capacity = *config.capacity();
            let splitter = CodeSplitter::new(tree_sitter_rust::LANGUAGE, config).unwrap();
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(capacity.fits(sizer.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_code_Characters_trim_false_{chunk_size}_overlap_{overlap}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}

#[cfg(feature = "code")]
#[test]
fn code_overlap_trim() {
    insta::glob!("inputs/code/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        CHUNK_SIZES.into_par_iter().for_each(|chunk_size| {
            let sizer = Characters;
            let overlap = chunk_size / 2;
            let config = ChunkConfig::new(chunk_size)
                .with_sizer(sizer)
                .with_overlap(overlap)
                .unwrap();
            let capacity = *config.capacity();
            let splitter = CodeSplitter::new(tree_sitter_rust::LANGUAGE, config).unwrap();
            let chunks = splitter.chunks(&text).collect::<Vec<_>>();

            for chunk in &chunks {
                assert!(capacity.fits(sizer.size(chunk)).is_le());
            }
            insta::assert_yaml_snapshot!(
                format!(
                    "{}_code_Characters_trim_{chunk_size}_overlap_{overlap}",
                    path.file_stem().unwrap().to_string_lossy()
                ),
                chunks
            );
        });
    });
}
