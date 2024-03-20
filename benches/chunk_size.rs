#![allow(missing_docs)]

fn main() {
    // Run registered benchmarks.
    divan::main();
}

const CHUNK_SIZES: [usize; 4] = [64, 512, 4096, 32768];

#[divan::bench_group]
mod text {
    use std::fs;

    use divan::{black_box_drop, Bencher};
    use text_splitter::TextSplitter;
    use tiktoken_rs::cl100k_base;
    use tokenizers::Tokenizer;

    use crate::CHUNK_SIZES;

    const TEXT_FILENAMES: &[&str] = &["romeo_and_juliet", "room_with_a_view"];

    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn characters<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bencher
            .with_inputs(|| {
                (
                    TextSplitter::default(),
                    fs::read_to_string(format!("tests/inputs/text/{filename}.txt")).unwrap(),
                )
            })
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text, N).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn tiktoken<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bencher
            .with_inputs(|| {
                (
                    TextSplitter::new(cl100k_base().unwrap()),
                    fs::read_to_string(format!("tests/inputs/text/{filename}.txt")).unwrap(),
                )
            })
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text, N).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = TEXT_FILENAMES, consts = CHUNK_SIZES)]
    fn tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bencher
            .with_inputs(|| {
                (
                    TextSplitter::new(Tokenizer::from_pretrained("bert-base-cased", None).unwrap()),
                    fs::read_to_string(format!("tests/inputs/text/{filename}.txt")).unwrap(),
                )
            })
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text, N).for_each(black_box_drop);
            });
    }
}

#[divan::bench_group]
mod markdown {
    use std::fs;

    use divan::{black_box_drop, Bencher};
    use text_splitter::MarkdownSplitter;
    use tiktoken_rs::cl100k_base;
    use tokenizers::Tokenizer;

    use crate::CHUNK_SIZES;

    const MARKDOWN_FILENAMES: &[&str] = &["commonmark_spec"];

    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn characters<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bencher
            .with_inputs(|| {
                (
                    MarkdownSplitter::default(),
                    fs::read_to_string(format!("tests/inputs/markdown/{filename}.md")).unwrap(),
                )
            })
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text, N).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn tiktoken<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bencher
            .with_inputs(|| {
                (
                    MarkdownSplitter::new(cl100k_base().unwrap()),
                    fs::read_to_string(format!("tests/inputs/markdown/{filename}.md")).unwrap(),
                )
            })
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text, N).for_each(black_box_drop);
            });
    }

    #[divan::bench(args = MARKDOWN_FILENAMES, consts = CHUNK_SIZES)]
    fn tokenizers<const N: usize>(bencher: Bencher<'_, '_>, filename: &str) {
        bencher
            .with_inputs(|| {
                (
                    MarkdownSplitter::new(
                        Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
                    ),
                    fs::read_to_string(format!("tests/inputs/markdown/{filename}.md")).unwrap(),
                )
            })
            .bench_values(|(splitter, text)| {
                splitter.chunks(&text, N).for_each(black_box_drop);
            });
    }
}
