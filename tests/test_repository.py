import ast
import re
from pathlib import Path

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_compile():
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_notebooks_are_valid_and_have_no_recorded_errors():
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    assert notebooks, "No notebooks found"

    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        errors = [
            output
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, f"{path.relative_to(ROOT)} contains recorded errors"


def test_notebooks_do_not_embed_local_absolute_paths():
    forbidden = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\\\Users\\\\)")
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        raw = path.read_text(encoding="utf-8")
        assert not forbidden.search(raw), f"Local absolute path found in {path.relative_to(ROOT)}"


def test_report_assets_are_complete():
    source = (ROOT / "report" / "individual_report.tex").read_text(encoding="utf-8")
    references = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source))
    assert references
    for relative_path in references:
        assert (ROOT / "report" / relative_path).is_file(), f"Missing report asset: {relative_path}"
    assert (ROOT / "report" / "individual_report.pdf").is_file()
    assert (ROOT / "report" / "references.bib").is_file()


def test_citation_metadata_is_valid_yaml():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["repository-code"].startswith("https://github.com/")


def test_submission_debris_is_absent():
    forbidden = [
        ROOT / ".DS_Store",
        ROOT / "Group 6 DS.pdf",
        ROOT / "p_prendergast_datascience_submission.zip",
        ROOT / "notebooks" / "daily" / "return_prediction" / "probabilistic_calibration_study.ipynb.bak",
    ]
    assert not [path for path in forbidden if path.exists()]

    latex_debris = {
        ".aux",
        ".bbl",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".toc",
    }
    assert not [path for path in (ROOT / "report").iterdir() if path.suffix in latex_debris]
