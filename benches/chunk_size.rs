use std::fs;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use text_splitter::TextSplitter;
use tiktoken_rs::cl100k_base;
use tokenizers::Tokenizer;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("characters");

    let splitter = TextSplitter::default();
    let text = fs::read_to_string("tests/inputs/text/romeo_and_juliet.txt").unwrap();

    for chunk_size in (5..17).map(|n| 2usize.pow(n)) {
        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &chunk_size,
            |b, &chunk_size| b.iter(|| splitter.chunks(&text, chunk_size).collect::<Vec<_>>()),
        );
    }

    group.finish();
}

fn huggingface_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("huggingface");

    let splitter = TextSplitter::new(Tokenizer::from_pretrained("bert-base-cased", None).unwrap());
    let text = fs::read_to_string("tests/inputs/text/romeo_and_juliet.txt").unwrap();

    for chunk_size in (5..17).map(|n| 2usize.pow(n)) {
        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &chunk_size,
            |b, &chunk_size| b.iter(|| splitter.chunks(&text, chunk_size).collect::<Vec<_>>()),
        );
    }

    group.finish();
}

fn tiktoken_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiktoken");

    let splitter = TextSplitter::new(cl100k_base().unwrap());
    let text = fs::read_to_string("tests/inputs/text/romeo_and_juliet.txt").unwrap();

    for chunk_size in (5..17).map(|n| 2usize.pow(n)) {
        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &chunk_size,
            |b, &chunk_size| b.iter(|| splitter.chunks(&text, chunk_size).collect::<Vec<_>>()),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    criterion_benchmark,
    huggingface_benchmark,
    tiktoken_benchmark
);
criterion_main!(benches);
