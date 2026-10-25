# Releasing text-splitter

## Current rollout: release preparation only

Release-plz prepares a reviewable release PR with the next shared version,
`Cargo.lock`, and root `CHANGELOG.md`. **It does not publish, push release tags,
or create GitHub releases in this rollout.** The workflow explicitly runs only
`release-pr`, not `release`.

Merging a release PR therefore does not publish anything. The existing manual
release/tag process remains in place:

- `.github/workflows/release.yml` publishes the Rust crate on a `v*` tag.
- `.github/workflows/python.yml` builds and tests wheels and the sdist for that
  tag, then publishes them to PyPI with uv.
- GitHub release creation remains manual.

Do not add a release-plz `release` job while the existing crates.io publisher
is still active: the same release tag would start a second publisher.

## One-time GitHub App setup

The workflow stays skipped until `RELEASE_PLZ_APP_CLIENT_ID` is configured.
No repository settings or credentials are provisioned by the code.

1. Create a GitHub App with Webhooks disabled and these repository permissions:
   - **Contents: read and write** (release branch commits).
   - **Pull requests: read and write** (open/update the release PR).
2. Install the App on `benbrandt/text-splitter` only.
3. Generate its private key and add it directly as the repository Actions secret
   `RELEASE_PLZ_APP_PRIVATE_KEY`. Never commit the key or paste it into a PR.
4. Set the repository Actions variable `RELEASE_PLZ_APP_CLIENT_ID` to the App's
   **client ID** (not its numeric App ID). Set this last to enable the workflow.
5. After this workflow has landed on `main`, run:

   ```sh
   gh workflow run release-plz.yml --repo benbrandt/text-splitter --ref main
   ```

After setup, every push to `main` runs release preparation. Manual dispatches
against other branches and runs in forks are skipped. Removing the client-ID
variable disables preparation without affecting the existing publishers.

The App token is restricted to this repository and the two permissions above.
Unlike the default `GITHUB_TOKEN`, it allows commits to the generated release
branch to trigger the normal PR checks. No crates.io or PyPI credentials are
passed to the preparation workflow.

## Version and changelog policy

Both packages inherit `[workspace.package].version`; there is still one version
and one `vX.Y.Z` tag for the Rust crate and Python distribution.

- The Rust package is compared with its crates.io release, including API checks
  with cargo-semver-checks.
- The Python Cargo package has `publish = false`. Release-plz manages it in
  `git_only` mode using the existing shared `v*` tags as its baseline, not a
  nonexistent crates.io release.
- Both packages belong to the same `distribution` version group. The root
  changelog includes Python changes, so Python-only fixes also prepare a release.
- Python does not get a second tag, GitHub release, or generated changelog.
  Maturin continues to derive its version from the Cargo workspace.
- In the existing 0.x release convention, additive `feat` commits increment the
  minor version; ordinary fixes increment the patch version. Breaking changes
  can be marked with `!` or a `BREAKING CHANGE` footer. Rust API checking is an
  additional guard, not a substitute for reviewing Python compatibility.
- Routine `chore`, CI, test, formatting, and build commits are omitted from the
  generated notes. `ci(python)` changes are retained because wheel/runtime
  changes are user-facing. Explicitly marked breaking commits are protected
  from filtering. Changelog filtering does not prevent a version bump.
- Dependabot continues to manage dependencies; preparation does not run a full
  `cargo update`.

Keep useful manual release notes in the root changelog. Review the first
generated entry to retain the existing detailed Rust/Python explanations and
remove duplicates. Python API breaks and dependency changes that affect users
still need explicit release notes even when their commit messages are filtered.

Changes only to excluded files, such as `.github/workflows`, do not necessarily
prepare a release. For a release-worthy packaging/workflow fix, add a note to the
root changelog so the release has an explicit user-facing change.

## Reviewing a release PR

1. Review the proposed version, generated notes, and compatibility results.
2. Verify Rust and Python versions remain equal and no historical changelog
   entries or existing tags are rewritten.
3. Wait for the existing Rust, Python, runtime, and typing checks to pass.
   Check results explicitly; this change does not configure branch protection.
4. Merge the release PR when ready, then perform the existing manual release
   process from that exact commit. Do not include additional unreviewed changes.

The PR can be edited before merging to improve release notes or correct a
version. During this rollout, publication is still a separate manual decision.

### Preview locally without creating a PR

Install release-plz **0.4.1** and cargo-semver-checks, then run this in a clean,
disposable clone or branch:

```sh
release-plz update --config release-plz.toml
git diff -- Cargo.toml Cargo.lock CHANGELOG.md bindings/python/Cargo.toml
```

`update` edits local files only; it does not publish or open a PR. Check both
versions with `cargo metadata --no-deps --format-version 1`. Preview Python-only
changes as well as core changes when updating release-plz or its configuration.
Do not use `release-plz release` to preview preparation.

## Later: publication cutover

Enabling automatic publication is a separate change:

1. Retire the old crates.io publishing job before enabling `release-plz release`.
2. Supply crates.io authentication (or configure trusted publishing).
3. Retain `release_always = false` so only release-PR merges initiate releases.
4. Verify an App-created `v*` tag starts the full Python build/test/publish path.
5. Document retries for partial releases: crates.io and PyPI publication cannot
   be atomic. Do not recreate or move an already published tag.
