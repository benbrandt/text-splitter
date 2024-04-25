/*!
Shared test and benchmark utilities.

Separate crate so that it can be only included in test mode for
unit tests, benchmarks, and integration tests.
*/
use std::path::PathBuf;

use cached_path::Cache;

/// Downloads a remote file to the cache directory if it doensn't already exist,
/// and returns the path to the cached file.
pub fn download_file_to_cache(src: &str) -> PathBuf {
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
