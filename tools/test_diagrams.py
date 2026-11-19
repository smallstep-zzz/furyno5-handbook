#!/usr/bin/env python3
"""Regression check: multi-character Mermaid IDs must stay whole in chains."""
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from render_diagrams import render


class DiagramTopology(unittest.TestCase):
    def test_identifiers_and_chained_edges(self):
        source = '''flowchart TB
AB["source"] --> BC["middle"] --> CD["target"]
BC --> end_2["end"]
'''
        with tempfile.TemporaryDirectory() as directory:
            with patch('render_diagrams.subprocess.run') as dot:
                dot.return_value = SimpleNamespace(stdout='<svg></svg>')
                render(source, Path(directory) / 'diagram.svg')
                graph = dot.call_args.kwargs['input']
        nodes = re.findall(r'^"([^"\n]+)" \[label=', graph, re.M)
        edges = re.findall(r'^"([^"\n]+)" -> "([^"\n]+)"', graph, re.M)
        self.assertCountEqual(nodes, ['AB', 'BC', 'CD', 'end_2'])
        self.assertCountEqual(edges, [('AB', 'BC'), ('BC', 'CD'), ('BC', 'end_2')])


if __name__ == '__main__':
    unittest.main()
