"""Number the appendix slides A1, A2, ... in front of their titles.

Idempotent: it strips any existing "A<n>\\quad " prefix and re-prefixes every frame that
follows \\section{Appendix}, so the numbers stay right after an appendix slide is inserted
or removed. Frames without a title (a [plain] divider) are left alone.

    python number_appendix.py                 # main_v2.tex
    python number_appendix.py main_v2.tex --check
"""
from __future__ import annotations

import argparse
import io
import re
import shutil

ANCHOR = r'\section{Appendix}'
PREFIX = re.compile(r'\\begin\{frame\}\{A\d+\\quad ')
FRAME = re.compile(r'\\begin\{frame\}\{([^}]*)\}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tex', nargs='?', default='main_v2.tex')
    ap.add_argument('--check', action='store_true', help='report only, do not write')
    args = ap.parse_args()

    s = io.open(args.tex, encoding='utf-8').read()
    if ANCHOR not in s:
        raise SystemExit(f'{args.tex}: no {ANCHOR}')

    i = s.index(ANCHOR)
    head, tail = s[:i], s[i:]
    had = len(PREFIX.findall(tail))

    stripped = PREFIX.sub(r'\\begin{frame}{', tail)
    n = [0]

    def number(m):
        n[0] += 1
        return r'\begin{frame}{A%d\quad ' % n[0] + m.group(1) + '}'

    numbered = FRAME.sub(number, stripped)
    changed = numbered != tail
    state = ('would rewrite' if changed else 'already consistent') if args.check else \
            ('rewritten' if changed else 'unchanged')
    print(f'{args.tex}: {n[0]} appendix slides, {had} already numbered -> {state}')

    if changed and not args.check:
        shutil.copyfile(args.tex, args.tex + '.bak')
        io.open(args.tex, 'w', encoding='utf-8', newline='\n').write(head + numbered)
        print(f'  backup: {args.tex}.bak')


if __name__ == '__main__':
    main()
