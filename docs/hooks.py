"""MkDocs hook: publish the generated validation report (benchmarks/results) inside the site.

The HTML/PDF reports are produced by ``flowtrace benchmark`` and live next to results.json;
this hook adds them as validation/report.html and validation/report.pdf without copying them
into docs/.
"""

import os

from mkdocs.structure.files import File

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = {"validation/report.html": "benchmarks/results/report.html",
           "validation/report.pdf": "benchmarks/results/report.pdf"}


def on_files(files, config):
    for dest, src in REPORTS.items():
        path = os.path.join(ROOT, src)
        if os.path.exists(path):
            files.append(File.generated(config, dest, abs_src_path=path))
    return files
