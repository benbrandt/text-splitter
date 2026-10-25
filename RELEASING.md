# Releasing text-splitter

## Release flow

1. Release-plz opens or updates a release PR with the shared version,
   `Cargo.lock`, and root `CHANGELOG.md`.
2. Review its version and notes, wait for required CI checks, and merge it.
   **Merging the release PR authorizes publication; do not bypass the checks.**
3. `.github/workflows/release.yml` runs `release-plz release` on `main`.
   `release_always = false` limits publication to release-PR merges. It publishes
   the Rust crate, creates the shared `vX.Y.Z` tag, and creates a draft GitHub
   release. The App-created tag triggers the Python workflow.
4. `.github/workflows/python.yml` builds and tests wheels and the sdist, then uv
   publishes them to PyPI using trusted publishing.
5. After PyPI publication succeeds, the Python workflow publishes the GitHub
   draft. GitHub selects the latest release automatically.

Release-plz is the only automatic Rust publisher and tag creator. There is no
second tag-triggered `cargo publish` job. Existing build/runtime/typing checks
and the publishers' built-in checks are used rather than a separate verification
framework.

## One-time GitHub App setup

Preparation is skipped with a notice until both App secrets are configured.
Publishing and finalization require the same App credentials.

1. Create a GitHub App with Webhooks disabled and these repository permissions:
   - **Contents: read and write** (release branch commits).
   - **Pull requests: read and write** (open/update the release PR).
2. Install the App on `benbrandt/text-splitter` only.
3. Generate its private key and add it directly as the repository Actions secret
   `RELEASE_PLZ_APP_PRIVATE_KEY`. Never commit the key or paste it into a PR.
4. Set the repository Actions secret `RELEASE_PLZ_APP_CLIENT_ID` to the App's
   **client ID** (not its numeric App ID).
5. After this workflow has landed on `main`, run:

   ```sh
   gh workflow run release-plz.yml --repo benbrandt/text-splitter --ref main
   ```

After setup, every push to `main` runs release preparation. Manual dispatches
against other branches and runs in forks are skipped. Removing either App
secret disables preparation and prevents automatic publishing/finalization.

The App token is restricted to this repository and the two permissions above.
Unlike the default `GITHUB_TOKEN`, it allows release PRs and tags to trigger the
normal CI and Python workflows. Finalization requests only Contents write
permission.

## Registry authentication

- **crates.io:** `CRATES_IO_TOKEN` is passed to release-plz as
  `CARGO_REGISTRY_TOKEN`. Crates.io trusted publishing is a separate migration.
- **PyPI:** register a trusted publisher for `benbrandt/text-splitter`, workflow
  filename **`python.yml`**, with **no environment**. The publishing job has
  `id-token: write` and runs `uv publish --trusted-publishing always`.
  Build jobs do not receive this permission. No `PYPI_API_TOKEN` or
  `UV_PUBLISH_TOKEN` is passed to publishing.

If the PyPI publisher was registered with an environment name, it must match a
job environment before releasing. After the first successful OIDC publication,
the unused PyPI token secret can be removed separately.

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
- Follow Cargo's pre-1.0 compatibility rules using release-plz's defaults:
  compatible features and fixes increment the patch version; breaking changes
  increment the minor version. Mark breaking changes with `!` or a
  `BREAKING CHANGE` footer. Rust API checking is an additional guard, not a
  substitute for reviewing Python compatibility.
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
   `main` requires the existing Rust, build, runtime, typing, and security checks.
   Keep those required check names in sync when CI jobs are renamed.
4. Merge the release PR when ready. Publication starts automatically; watch the
   Release workflow, then the tag's Python workflow.

The PR can be edited before merging to improve release notes or correct a
version. Review Python compatibility explicitly; Rust API checking cannot catch
Python API breaks.

### Preview locally without creating a PR

Install release-plz and cargo-semver-checks, then run this in a clean,
disposable clone or branch:

```sh
release-plz update --config release-plz.toml
git diff -- Cargo.toml Cargo.lock CHANGELOG.md bindings/python/Cargo.toml
```

`update` edits local files only; it does not publish or open a PR. Check both
versions with `cargo metadata --no-deps --format-version 1`. Preview Python-only
changes as well as core changes when updating release-plz or its configuration.
Do not use `release-plz release` to preview preparation.

## Retrying partial releases

Crates.io and PyPI publication cannot be atomic. Start by rerunning failed jobs:

```sh
gh run rerun <run-id> --repo benbrandt/text-splitter --failed
```

- Rerun the original run, not a newer `main` commit. Release-plz may have
  published the release PR's last commit instead of its merge commit.
- For a partial PyPI upload, rerun the publishing job with its original build
  artifacts. `uv publish --check-url https://pypi.org/simple/` skips files already
  present. Do not rebuild/re-upload a published filename with different contents.
- If only GitHub finalization failed, rerun that job; it does not publish
  packages. Publishing an already-public release is safe to repeat.
- If Rust uploaded successfully but tag/draft creation failed, finish the
  missing tag or draft manually from the **original published commit**, then
  resume the Python workflow. Release-plz can skip an already-uploaded crate or
  existing tag, so rerunning it alone may not repair that case.
- If the draft is missing, recreate it for the existing tag using the changelog
  notes, then rerun finalization.

Never move an existing release tag, republish a version with different code, or
bump the version just to retry a failed upload. Leave the GitHub release in
draft until Python publication succeeds.
