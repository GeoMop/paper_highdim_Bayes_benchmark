#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

mkdir -p .texlive-var .texlive-config .texlive-home
export TEXMFVAR="$repo_root/.texlive-var"
export TEXMFCONFIG="$repo_root/.texlive-config"
export TEXMFHOME="$repo_root/.texlive-home"

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "No Git history found. Cannot compare main.tex against HEAD." >&2
  exit 1
fi

if ! git ls-files --error-unmatch -- main.tex >/dev/null 2>&1; then
  echo "main.tex is not tracked by Git. Cannot build a comparison PDF." >&2
  exit 1
fi

old_source_label="HEAD:main.tex"
old_source_ref="HEAD:main.tex"

if ! git diff --cached --quiet -- main.tex; then
  old_source_label="staged main.tex"
  old_source_ref=":main.tex"
fi

tmp_old="$(mktemp)"
trap 'rm -f "$tmp_old"' EXIT

git show "$old_source_ref" > "$tmp_old"

latexdiff \
  --flatten \
  --math-markup=whole \
  --graphics-markup=none \
  --append-safecmd="vc,tn,symbfit,norm,grad,div,MMD,KL,TV,Real" \
  --append-textcmd="vc,tn,symbfit,norm" \
  "$tmp_old" \
  main.tex > main-d.tex
latexmk -f -lualatex -interaction=nonstopmode -file-line-error main-d.tex

printf 'Built comparison PDF: %s/main-d.pdf\n' "$repo_root"
printf 'Reference version: %s\n' "$old_source_label"
