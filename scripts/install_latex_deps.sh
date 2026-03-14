#!/usr/bin/env bash

set -euo pipefail

packages=(
  biber
  latexmk
  latexdiff
  lmodern
  texlive-luatex
  texlive-latex-base
  texlive-latex-recommended
  texlive-latex-extra
  texlive-fonts-recommended
  texlive-science
)

sudo apt-get update
sudo apt-get install -y "${packages[@]}"

printf 'Installed LaTeX packages:\n'
printf '  %s\n' "${packages[@]}"
