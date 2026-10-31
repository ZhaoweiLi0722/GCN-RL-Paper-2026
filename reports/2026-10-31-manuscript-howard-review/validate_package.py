"""Static manuscript/export checks only; never import research models."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/Graph_Aware_Deep_Reinforcement_Learning_for_Adaptive_Capacity_Planning_in_Distributed_Personalized_Regenerative_Medicine_Manufacturing_Networks"
REPORT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def main():
    manifest = read_json(PAPER / "data/source-manifest.json")
    for entry in manifest["sources"]:
        raw = (ROOT / entry["local_source"]).read_bytes()
        assert len(raw) == entry["bytes"], entry["id"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], entry["id"]
    primary_source = read_json(ROOT / manifest["sources"][0]["local_source"])
    primary = read_json(PAPER / "data/primary-comparison.json")
    assert all(primary_source[k] == value for k, value in primary.items())
    message_source = read_json(ROOT / manifest["sources"][1]["local_source"])
    messages = read_json(PAPER / "data/bounded-message-comparison.json")
    for contrast in message_source["pairs"]:
        for scope in ("worlds", "blocks", "conditions"):
            for record in contrast[scope]:
                record.pop("extra_metrics", None)
    assert all(message_source[k] == value for k, value in messages.items())
    procurement_source = read_json(ROOT / manifest["sources"][2]["local_source"])
    assert read_json(PAPER / "data/procurement-excerpts.json") == procurement_source["formal_result_excerpts"]
    for entry in manifest["sources"][5:]:
        assert (PAPER / "data/supplements" / entry["id"]).read_bytes() == (ROOT / entry["local_source"]).read_bytes()
    assert primary["paired_worlds"] == 120 and primary["training_blocks"] == 5
    assert len(primary["pairs"]) == 6 and len(messages["pairs"]) == 3
    assert all(len(pair["worlds"]) == 120 for pair in primary["pairs"])
    assert all(len(pair["worlds"]) == 24 for pair in messages["pairs"])

    sources = [PAPER / "main.tex", PAPER / "sections/related-work.tex"]
    tex = "\n".join(path.read_text() for path in sources)
    cites = set()
    for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        cites.update(group.split(","))
    bibliography = (PAPER / "references-current.bib").read_text()
    keys = re.findall(r"@\w+\{([^,]+),", bibliography)
    assert len(keys) == len(set(keys)) == 27
    assert not cites - set(keys), cites - set(keys)
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels))
    assert not set(refs) - set(labels), set(refs) - set(labels)
    figures = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    assert len(figures) == 5 and all((PAPER / path).is_file() for path in figures)
    assert "Overfull" not in (PAPER / "main.log").read_text()
    assert "undefined" not in (PAPER / "main.log").read_text().lower()

    from pypdf import PdfReader
    pdf = PdfReader(PAPER / "main.pdf")
    texts = [page.extract_text() for page in pdf.pages]
    assert all(len(text.strip()) > 100 for text in texts)
    assert not any("??" in text or "\ufffd" in text for text in texts)
    result = dict(status="static_export_and_document_checks_passed",
        science_reexecuted=False, comparisons_recomputed=False,
        source_hashes_checked=len(manifest["sources"]), primary_saved_pairs=6,
        bounded_message_saved_pairs=3, primary_world_records_retained_per_pair=120,
        message_world_records_retained_per_pair=24, bibliography_entries=len(keys),
        cited_keys=len(cites), figures=len(figures), pdf_pages=len(pdf.pages),
        pdf_sha256=hashlib.sha256((PAPER / "main.pdf").read_bytes()).hexdigest(),
        pdf_bytes=(PAPER / "main.pdf").stat().st_size,
        accountable_human_scientific_and_reference_review="pending",
        visual_inspection="recorded separately by the coordinator")
    (REPORT / "static-validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
