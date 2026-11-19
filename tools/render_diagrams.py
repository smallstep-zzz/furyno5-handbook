#!/usr/bin/env python3
"""Render the handbook's deliberately small Mermaid flowchart subset as SVG.

No browser CDN is required. Original Mermaid is kept next to each generated SVG.
Unsupported syntax fails the build so a diagram is never silently misrepresented.
"""
import html
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

NODE = re.compile(r'\b([A-Za-z][A-Za-z0-9_]*)\s*(\[\[.*?\]\]|\[.*?\]|\{.*?\}|\(.*?\))')
EDGE = re.compile(r'([A-Za-z][A-Za-z0-9_]*)\s*(-->|---|==>|-\.->)(?:\s*\|([^|]*)\|)?\s*([A-Za-z][A-Za-z0-9_]*)')

def label_text(s):
    s = s.strip().strip('[]{}()').strip().strip('"').strip("'")
    s = re.sub(r'<br\s*/?>', '\n', s)
    s = html.unescape(s).replace('\\n', '\n')
    return s

def wrap_label(s):
    lines = []
    for part in s.splitlines():
        limit = 23 if re.search(r'[가-힣]', part) else 34
        lines.extend(textwrap.wrap(part, width=limit, break_long_words=False, break_on_hyphens=False) or [''])
    return '\n'.join(lines)

def render(source, out):
    nodes = {}
    edges = []
    direction = 'TB'
    for raw in source.splitlines():
        line = raw.strip().rstrip(';')
        if not line or line.startswith('%%'):
            continue
        if re.match(r'^(?:flowchart|graph)\s+', line):
            direction = line.split()[1]
            continue
        if line.startswith(('subgraph ', 'classDef ', 'class ', 'style ', 'linkStyle ')) or line == 'end':
            raise ValueError('Unsupported Mermaid statement: ' + line)
        for m in NODE.finditer(line):
            nodes[m.group(1)] = (wrap_label(label_text(m.group(2))), 'diamond' if m.group(2).startswith('{') else 'box')
        clean = NODE.sub(lambda m: m.group(1), line)
        # Standard text labels: A -- "condition" --> B.
        clean = re.sub(r'--\s*"([^"]+)"\s*-->', lambda m: '-->|'+m.group(1)+'|', clean)
        clean = re.sub(r'--\s+([^|>\-][^>]*?)\s+-->', lambda m: '-->|'+m.group(1).strip()+'|', clean)
        found = list(EDGE.finditer(clean))
        if found:
            for m in found:
                a, arrow, label, b = m.groups()
                edges.append((a, b, label_text(label or ''), arrow == '-.->', arrow != '---'))
                nodes.setdefault(a, (a, 'box'))
                nodes.setdefault(b, (b, 'box'))
            # Chained arrows need a second scan starting at every endpoint.
            for m in re.finditer(r'(?=([A-Za-z][A-Za-z0-9_]*\s*(?:-->|==>|-\.->)(?:\s*\|[^|]*\|)?\s*[A-Za-z][A-Za-z0-9_]*))', clean):
                e = EDGE.fullmatch(m.group(1))
                if e:
                    a, arrow, label, b = e.groups()
                    item = (a,b,label_text(label or ''),arrow=='-.->',True)
                    if item not in edges:
                        edges.append(item)
                        nodes.setdefault(a,(a,'box'))
                        nodes.setdefault(b,(b,'box'))
        elif not NODE.search(line) and not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', clean):
            raise ValueError('Unparsed Mermaid: ' + line)
    if not nodes:
        raise ValueError('No diagram nodes')
    if direction in ('TD','TB'):
        direction='TB'
    if direction == 'LR' and len(nodes)>5:
        direction='TB'
    def q(x): return json.dumps(x, ensure_ascii=False)
    dot=['digraph handbook {',f'graph [rankdir={direction}, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.55", outputorder="edgesfirst"];',
         'node [shape=box, style="rounded,filled", fillcolor="#f8f7f3", color="#c9ccc9", penwidth=1.2, fontname="Noto Sans CJK KR", fontsize=12, fontcolor="#283330", margin="0.20,0.14"];',
         'edge [color="#a92d35", arrowsize=0.7, penwidth=1.2, fontname="Noto Sans CJK KR", fontsize=10, fontcolor="#6e5456"];']
    for key,(label,shape) in nodes.items():
        dot.append(f'{q(key)} [label={q(label)}, shape={shape}];')
    for a,b,label,dashed,arrow in edges:
        attrs=[f'label={q(label)}']
        if dashed: attrs.append('style=dashed')
        if not arrow: attrs.append('dir=none')
        dot.append(f'{q(a)} -> {q(b)} [{", ".join(attrs)}];')
    dot.append('}')
    result=subprocess.run(['dot','-Tsvg'],input='\n'.join(dot),text=True,capture_output=True,check=True)
    svg=re.sub(r'<\?xml.*?\?>\s*','',result.stdout,flags=re.S)
    svg=re.sub(r'<!DOCTYPE.*?>\s*','',svg,flags=re.S)
    svg=re.sub(r'<!--.*?-->\s*','',svg,flags=re.S)
    svg=svg.replace('<svg ', '<svg role="img" aria-label="'+html.escape(' · '.join(x[0].replace('\n',' ') for x in nodes.values()),quote=True)+'" ',1)
    Path(out).write_text(svg,encoding='utf-8')

if __name__=='__main__':
    render(Path(sys.argv[1]).read_text(encoding='utf-8'),sys.argv[2])
