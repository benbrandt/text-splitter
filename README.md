# text-splitter

Large language models (LLMs) have lots of amazing use cases. But often they have a limited context size that is smaller than larger documents. In order to use documents of larger length, you often have to split your text into chunks to fit within this context size.

This crate provides methods for doing so by trying to maximize a desired chunk size, but still splitting at semantic units whenever possible.
