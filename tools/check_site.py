#!/usr/bin/env python3
"""Check generated pages, local links, SVGs and example syntax without NPU access."""
from __future__ import annotations
import ast
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent

class Document(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids=[];self.links=[];self.images=[];self.h1=0;self.redirect=False;self.lang=None
        self.feed(text)
    def handle_starttag(self, tag, pairs):
        attrs=dict(pairs)
        if tag=='html':self.lang=attrs.get('lang')
        if 'id' in attrs:self.ids.append(attrs['id'])
        if tag=='h1':self.h1+=1
        if tag=='meta' and attrs.get('http-equiv','').lower()=='refresh':self.redirect=True
        if tag in ('a','link') and attrs.get('href'):self.links.append(attrs['href'])
        if tag in ('img','script') and attrs.get('src'):self.links.append(attrs['src'])
        if tag=='img':self.images.append(attrs)

def main():
    errors=[];pages={}
    paths=sorted([*ROOT.glob('*.html'),*ROOT.glob('chapter/*.html')])
    for path in paths:
        doc=Document(path.read_text(encoding='utf-8'));pages[path.resolve()]=doc
        duplicate=[key for key,n in Counter(doc.ids).items() if n>1]
        if duplicate:errors.append(f'{path.name}: duplicate ids {duplicate}')
        if doc.lang!='ko':errors.append(f'{path.name}: missing Korean document language')
        if not doc.redirect and doc.h1!=1:errors.append(f'{path.name}: expected one h1, got {doc.h1}')
        for img in doc.images:
            if not img.get('alt'):errors.append(f'{path.name}: missing image alt')
    for path,doc in pages.items():
        for href in doc.links:
            parts=urlsplit(href)
            if parts.scheme or parts.netloc or href.startswith('data:'):continue
            dest=(path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not dest.exists():errors.append(f'{path.name}: missing local target {href}');continue
            if parts.fragment and dest.suffix=='.html':
                target=pages.get(dest)
                if target and unquote(parts.fragment) not in target.ids:errors.append(f'{path.name}: missing anchor {href}')
    chapters=json.loads((ROOT/'data/chapters.json').read_text())
    for chapter in chapters:
        src=ROOT/'content'/f'{chapter["id"]}.md'
        if not src.exists():errors.append(f'missing source {src.name}');continue
        text=src.read_text(encoding='utf-8')
        if not text.startswith('# '):errors.append(f'{src.name}: source should start with h1')
        if '## 출처' not in text:errors.append(f'{src.name}: missing source section')
        if re.search(r'\b(?:lorem ipsum|TODO_WRITE|PLACEHOLDER_CONTENT)\b',text,re.I):errors.append(f'{src.name}: unfinished content')
    syntax_count=0
    for path in (ROOT/'examples').rglob('*.py'):
        try:ast.parse(path.read_text(encoding='utf-8'),filename=str(path));syntax_count+=1
        except SyntaxError as e:errors.append(str(e))
    for path in (ROOT/'assets/diagrams').glob('*.svg'):
        try:ET.fromstring(path.read_text(encoding='utf-8'))
        except ET.ParseError as e:errors.append(f'{path.name}: invalid SVG {e}')
    search=json.loads((ROOT/'search-index.json').read_text())
    if len(search)!=len(chapters):errors.append(f'search index has {len(search)} pages; expected {len(chapters)}')
    if not any('DMA' in d['text'] for d in search):errors.append('search index omits body text')
    if errors:
        print('\n'.join(errors));return 1
    print(json.dumps({'pages':len(pages),'chapters':len(chapters),'local_links':sum(len(d.links)for d in pages.values()),'example_scripts_syntax_checked':syntax_count,'result':'passed','npu_execution':'not performed'},ensure_ascii=False))
    return 0

if __name__=='__main__':sys.exit(main())
