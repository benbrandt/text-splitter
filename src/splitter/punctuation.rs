//! A punctuation-only prose chunk may belong to the word it was touching in
//! the source. Consider only actual neighboring output, after trim and overlap.

use std::{iter::from_fn, sync::LazyLock};

use icu_properties::{
    props::{GeneralCategory, GeneralCategoryGroup, LineBreak},
    CodePointMapData,
};
use icu_segmenter::{options::LineBreakOptions, LineSegmenter, LineSegmenterBorrowed};
use itertools::Itertools;

use super::{FallbackLevel, SemanticLevel, TextChunks, TryChunkSizer};
use crate::trim::Trim;

type Chunk<'a> = (usize, &'a str);

static LINE_BREAKS: LazyLock<LineSegmenterBorrowed<'static>> =
    LazyLock::new(|| LineSegmenter::new_dictionary(LineBreakOptions::default()));

fn punctuation(c: char) -> bool {
    GeneralCategoryGroup::Punctuation.contains(CodePointMapData::<GeneralCategory>::new().get(c))
}

fn glue_space(c: char) -> bool {
    c.is_whitespace() && CodePointMapData::<LineBreak>::new().get(c) == LineBreak::Glue
}

fn attached(c: char) -> bool {
    !c.is_whitespace() || glue_space(c)
}

fn hard_break(c: char) -> bool {
    matches!(
        CodePointMapData::<LineBreak>::new().get(c),
        LineBreak::MandatoryBreak
            | LineBreak::CarriageReturn
            | LineBreak::LineFeed
            | LineBreak::NextLine
    )
}

impl<'sizer, 'text: 'sizer, Sizer: TryChunkSizer, Level: SemanticLevel>
    TextChunks<'text, 'sizer, Sizer, Level>
{
    pub(super) fn attach_punctuation(
        mut self,
    ) -> impl Iterator<Item = Result<Chunk<'text>, Sizer::Error>> + 'sizer {
        let mut pending = None;
        let mut emitted_end = 0;
        from_fn(move || {
            let mut left = match pending.take().or_else(|| self.next())? {
                Ok(chunk) => chunk,
                Err(e) => return Some(Err(e)),
            };
            let unpadded = left.1.trim_end();
            // Do not size the next chunk just to emit an ordinary boundary, or
            // search through the remaining document for a non-whitespace byte.
            if !unpadded.is_empty()
                && self.text[left.0 + unpadded.len()..].starts_with(|next: char| {
                    if next.is_whitespace() {
                        glue_space(next)
                    } else {
                        punctuation(next) || unpadded.trim_start().chars().all(punctuation)
                    }
                })
            {
                pending = self.next();
                if let Some(Ok(right)) = pending.as_ref() {
                    match self.repair_pair(left, *right, emitted_end) {
                        Ok((first, second)) => {
                            left = first;
                            pending = second.map(Ok);
                        }
                        Err(e) => {
                            // Yield the valid left chunk before reporting the error.
                            self.finished = true;
                            pending = Some(Err(e));
                        }
                    }
                }
            }
            emitted_end = left.0 + left.1.len();
            Some(Ok(left))
        })
    }

    fn repair_pair(
        &mut self,
        left: Chunk<'text>,
        right: Chunk<'text>,
        emitted_end: usize,
    ) -> Result<(Chunk<'text>, Option<Chunk<'text>>), Sizer::Error> {
        let unchanged = (left, Some(right));
        let (a_start, a) = Trim::All.trim(left.0, left.1);
        let (b_start, b) = Trim::All.trim(right.0, right.1);
        if a.is_empty() || b.is_empty() || left.0 + left.1.len() > right.0 {
            return Ok(unchanged);
        }
        let prefix = a.chars().all(punctuation);
        let (donor, content) = if prefix { (right, b) } else { (left, a) };
        if (!prefix && !b.chars().all(punctuation)) || !content.chars().any(char::is_alphanumeric) {
            return Ok(unchanged);
        }
        let a_end = a_start + a.len();
        let b_end = b_start + b.len();
        // Whitespace from trimming still belongs to the source. Never glue
        // across a space or newline; a nonbreaking space does bind its sides.
        if !self.text[a_end..b_start].chars().all(glue_space) {
            return Ok(unchanged);
        }
        let visible = &self.text[a_start..b_end];
        // Don't guess a direction between two attached neighbors, or move an
        // existing cut across lines/paragraphs, Markdown blocks or table rows.
        if (prefix && self.text[..a_start].ends_with(attached))
            || (!prefix && self.text[b_end..].starts_with(attached))
            || visible.contains(hard_break)
        {
            return Ok(unchanged);
        }
        let mut breaks = LINE_BREAKS
            .segment_str(visible)
            .map(|i| a_start + i)
            .peekable();
        let before = breaks.by_ref().peeking_take_while(|&i| i < a_end).last();
        let after = breaks.next();
        // If Unicode allows the original line break, keep the splitter's choice.
        if after.is_some_and(|i| i <= b_start) {
            return Ok(unchanged);
        }
        let end = right.0 + right.1.len();
        let whole = self.trim.trim(left.0, &self.text[left.0..end]);
        if self.fits(whole)? {
            return Ok((whole, None));
        }
        // Try just the nearest legal break on the content side. It must also
        // be a word boundary: line breaking alone can divide a dictionary word.
        let Some(cut) = (if prefix { after } else { before }) else {
            return Ok(unchanged);
        };
        if cut <= a_start
            || cut >= b_end
            || self
                .semantic_split
                .bisects_intact_section(donor.0..donor.0 + donor.1.len(), cut)
            || !FallbackLevel::Word
                .sections(visible)
                .any(|(i, _)| a_start + i == cut)
        {
            return Ok(unchanged);
        }
        let first = self.trim.trim(left.0, &self.text[left.0..cut]);
        let second = self.trim.trim(cut, &self.text[cut..end]);
        // Keep both replacements in order relative to the untouched, possibly
        // overlapping chunks on either side of this pair.
        if ![first, second]
            .iter()
            .all(|(_, s)| s.chars().any(char::is_alphanumeric))
            || first.0 + first.1.len() <= emitted_end
            || (self.cursor < self.text.len() && second.0 >= self.cursor)
        {
            return Ok(unchanged);
        }
        // Sizing need not be monotonic, even for real BPE tokenizers.
        if self.fits(first)? && self.fits(second)? {
            Ok((first, Some(second)))
        } else {
            Ok(unchanged)
        }
    }

    /// Measure exactly the slice that would be emitted (already trimmed).
    fn fits(&mut self, (start, text): Chunk<'text>) -> Result<bool, Sizer::Error> {
        Ok(self.chunk_sizer.chunk_size(start, text, Trim::None)? <= self.capacity.max)
    }
}
