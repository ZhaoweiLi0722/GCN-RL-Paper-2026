#!/bin/zsh
# Regenerate the review diff files from main.tex vs the original manuscript.
# EDIT ONLY main.tex by hand. The two main_diff*.tex files are GENERATED — run this
# after each edit to refresh them; never hand-edit them (changes would be overwritten).
#
#   ./make_diff.sh
set -e
cd "$(dirname "$0")"
ORIG="../../docs/research_howard/current-manuscript/main.tex"
if ! command -v latexdiff >/dev/null 2>&1; then
  echo "latexdiff is required; generated diff files were left unchanged" >&2
  exit 127
fi
if [[ ! -f "$ORIG" ]]; then
  echo "original manuscript not found at $ORIG; generated diff files were left unchanged" >&2
  exit 2
fi

TMP_DIFF="$(mktemp "${TMPDIR:-/tmp}/main_diff.XXXXXX.tex")"
TMP_CLEAN="$(mktemp "${TMPDIR:-/tmp}/main_diff_clean.XXXXXX.tex")"
trap 'rm -f "$TMP_DIFF" "$TMP_CLEAN"' EXIT

# Full audit: blue additions + red strikethrough deletions.
latexdiff --encoding=utf8 --math-markup=whole --append-safecmd="textcolor" \
  "$ORIG" main.tex > "$TMP_DIFF" 2>/dev/null

# Clean read: blue additions only, deletions hidden (rendering-only tweak of two macros).
cp "$TMP_DIFF" "$TMP_CLEAN"
perl -0pi -e 's/\\providecommand\{\\DIFadd\}\[1\]\{\{\\protect\\color\{blue\}\\uwave\{#1\}\}\}/\\providecommand{\\DIFadd}[1]{{\\protect\\color{blue}#1}}/' "$TMP_CLEAN"
perl -0pi -e 's/\\providecommand\{\\DIFdel\}\[1\]\{\{\\protect\\color\{red\}\\sout\{#1\}\}\}/\\providecommand{\\DIFdel}[1]{}/' "$TMP_CLEAN"
mv "$TMP_DIFF" main_diff.tex
mv "$TMP_CLEAN" main_diff_clean.tex

echo "Regenerated main_diff.tex (add+del) and main_diff_clean.tex (add-only) from main.tex"
