#!/bin/zsh
# Regenerate the review diff files from main.tex vs the original manuscript.
# EDIT ONLY main.tex by hand. The two main_diff*.tex files are GENERATED — run this
# after each edit to refresh them; never hand-edit them (changes would be overwritten).
#
#   ./make_diff.sh
set -e
cd "$(dirname "$0")"
ORIG="../../docs/research_howard/current-manuscript/main.tex"

# Full audit: blue additions + red strikethrough deletions.
latexdiff --encoding=utf8 --math-markup=whole --append-safecmd="textcolor" \
  "$ORIG" main.tex > main_diff.tex 2>/dev/null

# Clean read: blue additions only, deletions hidden (rendering-only tweak of two macros).
cp main_diff.tex main_diff_clean.tex
perl -0pi -e 's/\\providecommand\{\\DIFadd\}\[1\]\{\{\\protect\\color\{blue\}\\uwave\{#1\}\}\}/\\providecommand{\\DIFadd}[1]{{\\protect\\color{blue}#1}}/' main_diff_clean.tex
perl -0pi -e 's/\\providecommand\{\\DIFdel\}\[1\]\{\{\\protect\\color\{red\}\\sout\{#1\}\}\}/\\providecommand{\\DIFdel}[1]{}/' main_diff_clean.tex

echo "Regenerated main_diff.tex (add+del) and main_diff_clean.tex (add-only) from main.tex"
