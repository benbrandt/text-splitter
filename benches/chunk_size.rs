#![allow(missing_docs)]

use std::{fs, path::PathBuf};

use ahash::AHashMap;
use cached_path::Cache;
use divan::AllocProfiler;
use once_cell::sync::Lazy;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

const CHUNK_SIZES: [usize; 3] = [64, 1024, 16384];

fn main() {
    // Run registered benchmarks.
    divan::main();
}

/// Downloads a remote file to the cache directory if it doensn't already exist,
/// and returns the path to the cached file.
fn download_file_to_cache(src: &str) -> PathBuf {
    let mut cache_dir = dirs::home_dir().unwrap();
    cache_dir.push(".cache");
    cache_dir.push(".text-splitter");

    Cache::builder()
        .dir(cache_dir)
        .build()
        .unwrap()
        .cached_path(src)
        .unwrap()
}

const TEXT_FILENAMES: &[&str] = &["romeo_and_juliet", "room_with_a_view"];
const MARKDOWN_FILENAMES: &[&str] = &["commonmark_spec"];
const CODE_FILENAMES: &[&str] = &["hashbrown_set_rs"];

static FILES: Lazy<AHashMap<&'static str, String>> = Lazy::new(|| {
    let mut m = AHashMap::new();
    for &name in TEXT_FILENAMES {
        m.insert(
            name,
            fs::read_to_string(format!("tests/inputs/text/{name}.txt")).unwrap(),
        );
    }
    for &name in MARKDOWN_FILENAMES {
        m.insert(
            name,
            fs::read_to_string(format!("tests/inputs/markdown/{name}.md")).unwrap(),
        );
    }
    for &name in CODE_FILENAMES {
        m.insert(
            name,
            fs::read_to_string(format!("tests/inputs/code/{name}.txt")).unwrap(),
        );
    }
    m
});

static BERT_TOKENIZER: Lazy<rust_tokenizers::tokenizer::BertTokenizer> = Lazy::new(|| {
    let vocab_path = download_file_to_cache(
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    );
    rust_tokenizers::tokenizer::BertTokenizer::from_file(vocab_path, false, false).unwrap()
});

#[divan::bench_group]
mod text {
    use divan::{black_box_drop, counter::BytesCount, Bencher};
    use text_splitter::{ChunkConfig, ChunkSizer, TextSplitter};

    use crate::{CHUNK_SIZES, FILES, TEXT_FILENAMES};

    fn bench<S, G>(bencher: Bencher<'_, '_>, filename: &str, gen_splitter: G)
    where
        G: Fn() -> TextSplitter<S> + Sync,
        S: ChunkSizer,
    {
        bencher
            .with_inputs(|| (gen_splitter(), FILES.get(filename).unwrap().clone()))
            .input_counter(|(_, text)| BytesCount::of_str(text))
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn characters<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || TextSplitter::new(N));
    }

    #[cfg(feature = "tiktoken-rs")]
    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn tiktoken<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        use text_splitter::ChunkConfig;

        bench(bencher, filename, || {
            TextSplitter::new(ChunkConfig::new(N).with_sizer(tiktoken_rs::cl100k_base().unwrap()))
        });
    }

    #[cfg(feature = "tokenizers")]
    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            TextSplitter::new(ChunkConfig::new(N).with_sizer(
                tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
            ))
        });
    }
    #[cfg(feature = "rust-tokenizers")]
    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn rust_tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        use crate::BERT_TOKENIZER;

        bench(bencher, filename, || {
            TextSplitter::new(ChunkConfig::new(N).with_sizer(&*BERT_TOKENIZER))
        });
    }
}

#[cfg(feature = "markdown")]
#[divan::bench_group]
mod markdown {
    use divan::{black_box_drop, counter::BytesCount, Bencher};
    use text_splitter::{ChunkConfig, ChunkSizer, MarkdownSplitter};

    use crate::{CHUNK_SIZES, FILES, MARKDOWN_FILENAMES};

    fn bench<S, G>(bencher: Bencher<'_, '_>, filename: &str, gen_splitter: G)
    where
        G: Fn() -> MarkdownSplitter<S> + Sync,
        S: ChunkSizer,
    {
        bencher
            .with_inputs(|| (gen_splitter(), FILES.get(filename).unwrap().clone()))
            .input_counter(|(_, text)| BytesCount::of_str(text))
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn characters<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || MarkdownSplitter::new(N));
    }

    #[cfg(feature = "tiktoken-rs")]
    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn tiktoken<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            MarkdownSplitter::new(
                ChunkConfig::new(N).with_sizer(tiktoken_rs::cl100k_base().unwrap()),
            )
        });
    }

    #[cfg(feature = "tokenizers")]
    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            MarkdownSplitter::new(ChunkConfig::new(N).with_sizer(
                tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
            ))
        });
    }
    #[cfg(feature = "rust-tokenizers")]
    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn rust_tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        use crate::BERT_TOKENIZER;

        bench(bencher, filename, || {
            MarkdownSplitter::new(ChunkConfig::new(N).with_sizer(&*BERT_TOKENIZER))
        });
    }
}

#[cfg(feature = "code")]
#[divan::bench_group]
mod code {
    use divan::{black_box_drop, counter::BytesCount, Bencher};
    use text_splitter::{ChunkConfig, ChunkSizer, CodeSplitter};

    use crate::{CHUNK_SIZES, CODE_FILENAMES, FILES};

    fn bench<S, G>(bencher: Bencher<'_, '_>, filename: &str, gen_splitter: G)
    where
        G: Fn() -> CodeSplitter<S> + Sync,
        S: ChunkSizer,
    {
        bencher
            .with_inputs(|| (gen_splitter(), FILES.get(filename).unwrap().clone()))
            .input_counter(|(_, text)| BytesCount::of_str(text))
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = CODE_FILENAMES, consts = CHUNK_SIZES)]
    fn characters<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            CodeSplitter::new(tree_sitter_rust::language(), N).unwrap()
        });
    }

    #[cfg(feature = "tiktoken-rs")]
    #[divan::bench(args = CODE_FILENAMES, consts = CHUNK_SIZES)]
    fn tiktoken<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            CodeSplitter::new(
                tree_sitter_rust::language(),
                ChunkConfig::new(N).with_sizer(tiktoken_rs::cl100k_base().unwrap()),
            )
            .unwrap()
        });
    }

    #[cfg(feature = "tokenizers")]
    #[divan::bench(args = CODE_FILENAMES, consts = CHUNK_SIZES)]
    fn tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench(bencher, filename, || {
            CodeSplitter::new(
                tree_sitter_rust::language(),
                ChunkConfig::new(N).with_sizer(
                    tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
                ),
            )
            .unwrap()
        });
    }
    #[cfg(feature = "rust-tokenizers")]
    #[divan::bench(args = CODE_FILENAMES, consts = CHUNK_SIZES)]
    fn rust_tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        use crate::BERT_TOKENIZER;

        bench(bencher, filename, || {
            CodeSplitter::new(
                tree_sitter_rust::language(),
                ChunkConfig::new(N).with_sizer(&*BERT_TOKENIZER),
            )
            .unwrap()
        });
    }
}
