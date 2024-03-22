#![allow(missing_docs)]

use divan::AllocProfiler;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

const CHUNK_SIZES: [usize; 4] = [64, 512, 4096, 32768];

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench_group]
mod text {
    use std::fs;

    use divan::{black_box_drop, counter::BytesCount, Bencher};
    use text_splitter::{ChunkSizer, TextSplitter};

    use crate::CHUNK_SIZES;

    const TEXT_FILENAMES: &[&str] = &["romeo_and_juliet", "room_with_a_view"];

    fn bench<const N: usize, S, G>(bencher: Bencher<'_, '_>, filename: &str, gen_splitter: G)
    where
        G: Fn() -> TextSplitter<S> + Sync,
        S: ChunkSizer,
    {
        bencher
            .with_inputs(|| {
                (
                    gen_splitter(),
                    fs::read_to_string(format!("tests/inputs/text/{filename}.txt")).unwrap(),
                )
            })
            .input_counter(|(_, text)| BytesCount::of_str(text))
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text, N).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn characters<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench::<N, _, _>(bencher, filename, TextSplitter::default);
    }

    #[cfg(feature = "tiktoken-rs")]
    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn tiktoken<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench::<N, _, _>(bencher, filename, || {
            TextSplitter::new(tiktoken_rs::cl100k_base().unwrap())
        });
    }

    #[cfg(feature = "tokenizers")]
    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench::<N, _, _>(bencher, filename, || {
            TextSplitter::new(
                tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
            )
        });
    }
}

#[cfg(feature = "markdown")]
#[divan::bench_group]
mod markdown {
    use std::fs;

    use divan::{black_box_drop, counter::BytesCount, Bencher};
    use text_splitter::{ChunkSizer, MarkdownSplitter};

    use crate::CHUNK_SIZES;

    const MARKDOWN_FILENAMES: &[&str] = &["commonmark_spec"];

    fn bench<const N: usize, S, G>(bencher: Bencher<'_, '_>, filename: &str, gen_splitter: G)
    where
        G: Fn() -> MarkdownSplitter<S> + Sync,
        S: ChunkSizer,
    {
        bencher
            .with_inputs(|| {
                (
                    gen_splitter(),
                    fs::read_to_string(format!("tests/inputs/markdown/{filename}.md")).unwrap(),
                )
            })
            .input_counter(|(_, text)| BytesCount::of_str(text))
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text, N).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn characters<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench::<N, _, _>(bencher, filename, MarkdownSplitter::default);
    }

    #[cfg(feature = "tiktoken-rs")]
    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn tiktoken<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench::<N, _, _>(bencher, filename, || {
            MarkdownSplitter::new(tiktoken_rs::cl100k_base().unwrap())
        });
    }

    #[cfg(feature = "tokenizers")]
    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bench::<N, _, _>(bencher, filename, || {
            MarkdownSplitter::new(
                tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
            )
        });
    }
}
