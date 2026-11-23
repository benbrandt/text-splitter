//! Move one existing greedy boundary; never change the number of chunks.

use std::{cmp::Reverse, iter::from_fn, ops::Range, sync::LazyLock};

use icu_properties::{props::LineBreak, CodePointMapData};
use icu_segmenter::{options::LineBreakOptions, LineSegmenter, LineSegmenterBorrowed};

use super::{FallbackLevel, SemanticLevel, SizedChunk, TextChunks, TryChunkSizer};
use crate::trim::Trim;

static LINE: LazyLock<LineSegmenterBorrowed<'static>> =
    LazyLock::new(|| LineSegmenter::new_dictionary(LineBreakOptions::default()));

// A bounded number of sizing calls per pair, including for callback/token sizers.
const MAX_CANDIDATES: usize = 3;

fn hard_break(c: char) -> bool {
    matches!(
        CodePointMapData::<LineBreak>::new().get(c),
        LineBreak::MandatoryBreak
            | LineBreak::LineFeed
            | LineBreak::CarriageReturn
            | LineBreak::NextLine
    )
}

impl<'sizer, 'text: 'sizer, Sizer: TryChunkSizer, Level: SemanticLevel>
    TextChunks<'text, 'sizer, Sizer, Level>
{
    pub(super) fn balanced_chunks(
        mut self,
    ) -> impl Iterator<Item = Result<(usize, &'text str), Sizer::Error>> + 'sizer {
        // A changed right chunk is final. An unchanged one can still help its
        // next neighbor. Errors from lookahead follow the already-valid left chunk.
        let mut pending = None;
        let mut previous_end = 0;
        from_fn(move || {
            let (mut left, final_cut) = match pending
                .take()
                .or_else(|| self.next_sized().map(|r| r.map(|chunk| (chunk, false))))?
            {
                Ok(item) => item,
                Err(error) => return Some(Err(error)),
            };
            if !final_cut {
                match self.next_sized() {
                    Some(Ok(right)) => match self.balance_pair(left, right, previous_end) {
                        Ok(Some((first, second))) => {
                            left = first;
                            pending = Some(Ok((second, true)));
                        }
                        Ok(None) => pending = Some(Ok((right, false))),
                        Err(error) => {
                            self.finished = true;
                            pending = Some(Err(error));
                        }
                    },
                    Some(Err(error)) => pending = Some(Err(error)),
                    None => {}
                }
            }
            previous_end = left.end();
            Some(Ok(left.indexed()))
        })
    }

    fn balance_pair(
        &mut self,
        left: SizedChunk<'text>,
        right: SizedChunk<'text>,
        previous_end: usize,
    ) -> Result<Option<(SizedChunk<'text>, SizedChunk<'text>)>, Sizer::Error> {
        let move_left = left.size > right.size;
        let (donor, small) = if move_left {
            (left, right)
        } else {
            (right, left)
        };
        // Help an unusually short chunk, not every pair that differs by a word.
        if left.end() > right.start
            || donor.size == small.size
            || small.size > self.capacity.desired / 3
            || small.size > donor.size / 2
        {
            return Ok(None);
        }

        // Include the whitespace hidden by trim when identifying the old cut.
        // Only the line containing that cut may donate text.
        let lo = left.start + left.text.trim_end().len();
        let hi = right.end() - right.text.trim_start().len();
        let text = self.text;
        if left.text.trim().is_empty()
            || right.text.trim().is_empty()
            || text[lo..hi].contains(hard_break)
        {
            return Ok(None);
        }
        let start = lo
            - text[left.start..lo]
                .rsplit(hard_break)
                .next()
                .unwrap()
                .len();
        let end = hi
            + text[hi..right.end()]
                .split(hard_break)
                .next()
                .unwrap()
                .len();
        let mut cuts = candidate_cuts(&text[start..end], lo - start, hi - start);
        cuts.retain(|&cut| donor.start < start + cut && start + cut < donor.end());

        // Bytes rank proposals only. A shorter prefix can cost *more* tokens,
        // so measure both actual outputs instead of searching by monotonic size.
        let gap = donor.size - small.size;
        let bytes = (donor.text.len() as u128 * gap as u128 / (2 * donor.size as u128)) as usize;
        let ideal = if move_left {
            donor.end() - bytes
        } else {
            donor.start + bytes
        };
        let nearest = cuts.len().min(MAX_CANDIDATES);
        if cuts.len() > nearest {
            cuts.select_nth_unstable_by_key(nearest, |&cut| ((start + cut).abs_diff(ideal), cut));
        }
        let mut best = None;
        let mut score = (small.size, Reverse(gap), Reverse(usize::MAX));
        for &cut in &cuts[..nearest] {
            let cut = start + cut;
            // Check only the bounded shortlist against the parsed tree, not
            // every word against every Markdown element in the donor.
            if self
                .semantic_split
                .bisects_intact_section(donor.start..donor.end(), cut)
            {
                continue;
            }
            let (a, first) = self.trim.trim(left.start, &text[left.start..cut]);
            let (b, second) = self.trim.trim(cut, &text[cut..right.end()]);
            // Keep the existing overlap with *outside* neighbors entirely in
            // the same chunk. Also preserve the two outer endpoints under trim.
            if first.is_empty()
                || second.is_empty()
                || a != left.start
                || b + second.len() != right.end()
                || a + first.len() <= previous_end
                || (self.cursor < self.text.len() && b >= self.cursor)
            {
                continue;
            }
            let x = self.chunk_sizer.chunk_size(a, first, Trim::None)?;
            if x > self.capacity.max {
                continue;
            }
            let y = self.chunk_sizer.chunk_size(b, second, Trim::None)?;
            let min = x.min(y);
            let diff = x.abs_diff(y);
            let candidate = (min, Reverse(diff), Reverse(cut));
            if y <= self.capacity.max && min > small.size && diff < gap && candidate > score {
                best = Some((
                    SizedChunk {
                        start: a,
                        text: first,
                        size: x,
                    },
                    SizedChunk {
                        start: b,
                        text: second,
                        size: y,
                    },
                ));
                score = candidate;
            }
        }
        Ok(best)
    }
}

/// If the old cut splits a unit at this level, return its enclosing range.
/// Otherwise only another cut at this level may replace it.
fn enclosing(cuts: &[usize], lo: usize, hi: usize, end: usize) -> Option<Range<usize>> {
    let i = cuts.partition_point(|&cut| cut < lo);
    let next = cuts.get(i).copied().unwrap_or(end);
    (next > hi).then(|| i.checked_sub(1).map_or(0, |i| cuts[i])..next)
}

fn candidate_cuts(text: &str, lo: usize, hi: usize) -> Vec<usize> {
    let breaks = |level: FallbackLevel| {
        level
            .sections(text)
            .skip(1)
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    };
    let sentences = breaks(FallbackLevel::Sentence);
    let Some(sentence) = enclosing(&sentences, lo, hi, text.len()) else {
        return sentences;
    };
    let words = breaks(FallbackLevel::Word);
    if let Some(word) = enclosing(&words, lo, hi, text.len()) {
        // Already inside a word: do not start breaking a different one.
        return FallbackLevel::GraphemeCluster
            .sections(text)
            .skip(1)
            .map(|(i, _)| i)
            .filter(|i| word.start.max(sentence.start) <= *i && *i <= word.end.min(sentence.end))
            .collect();
    }
    // An ICU word break can strand a mark. Require a legal line break too.
    LINE.segment_str(text)
        .filter(|i| sentence.start <= *i && *i <= sentence.end && words.binary_search(i).is_ok())
        .collect()
}
