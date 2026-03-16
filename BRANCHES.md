# Branch and packaging workflow

This repository intentionally maintains two upstream development lines.

## Branch roles

- `master`
  - primary upstream branch
  - tracks the modern CMake-based toolchain work
  - branch that downstreams should assume for current development unless documented otherwise
- `0.16`
  - legacy upstream maintenance branch
  - reserved for work that must stay compatible with the legacy 0.16/autotools line
- `debian/sid`
  - Debian unstable packaging branch for the current upstream line
  - expected to merge from `master`
- `debian/sid-0.16`
  - Debian packaging branch for the legacy upstream line
  - expected to merge from `0.16`

## Maintainer expectations

- New upstream feature work should target `master`
- Legacy-only fixes should target `0.16`
- Debian unstable packaging updates for the current line should target `debian/sid`
- Debian packaging changes specific to the legacy line should target `debian/sid-0.16`
- Before changing GitHub branch protections or CI filters, verify that required checks are attached to the correct branch names

## Documentation and helper scripts

- [update_doc.sh](update_doc.sh) assumes the documentation source branch is `master` by default, but this can be overridden with `DOC_SOURCE_BRANCH`
- [dist/macosx/macports/mk_diff.sh](dist/macosx/macports/mk_diff.sh) assumes the upstream comparison branch is `master` by default, but this can be overridden with `UPSTREAM_BRANCH`
- [.gbp.conf](.gbp.conf) should keep `master` as the primary upstream branch and `debian/sid` as the Debian unstable packaging branch

## GitHub and downstream follow-up after branch renames

After any branch rename or branch-role swap, also update:

- repository default branch
- branch protection rules
- required status checks
- open pull request base branches
- any GitHub Actions branch filters
- downstream packaging references and maintainer notes

## Suggested local retargeting after the rename

Typical local maintenance commands after the server-side branch changes are:

```bash
git fetch origin --prune
git branch -m debian/sid-legacy-autotools debian/sid-0.16
git branch -m master 0.16
git branch -m pr/msvc-toolchain-compat master
git branch --set-upstream-to=origin/master master
git branch --set-upstream-to=origin/0.16 0.16
git branch --set-upstream-to=origin/debian/sid debian/sid
git branch --set-upstream-to=origin/debian/sid-0.16 debian/sid-0.16
git remote set-head origin -a
```

Run the rename commands that apply to the branches present in a given clone.
