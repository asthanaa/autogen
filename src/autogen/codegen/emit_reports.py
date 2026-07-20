"""Raw-only projected qp report export.

The canonical exhaustive artifact is a plain UTF-8 ``.tct`` file containing
all raw projected contractions. Portrait PDFs are optional convenience views
generated from the same metadata.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = ROOT / "generated_code" / "methods" / "qp_ccsd" / "projected_codegen_plan.json"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "projected_qp"
DEFAULT_TCT = DEFAULT_ARTIFACT_DIR / "projected_qp_raw_terms.tct"
DEFAULT_SUMMARY_JSON = DEFAULT_ARTIFACT_DIR / "projected_qp_raw_summary.json"
DEFAULT_PDF_DIR = DEFAULT_ARTIFACT_DIR / "pdf"


def _tokenize(labels: str):
    out = []
    cur = ""
    for ch in labels:
        if ch.isalpha():
            if cur:
                out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _coeff_text(coeff: float, *, max_denominator: int = 4096) -> str:
    if abs(coeff) < 1e-14:
        return "0"

    frac = Fraction(float(coeff)).limit_denominator(max_denominator)
    if abs(float(frac) - float(coeff)) < 1e-12:
        sign = "+" if frac >= 0 else "-"
        num = abs(frac.numerator)
        den = frac.denominator
        if den == 1:
            return f"{sign}{num}"
        return f"{sign}{num}/{den}"
    return f"{coeff:+.8g}"


def _output_plain(name: str, labels: str) -> str:
    return name if not labels else f"{name}[{labels}]"


def _term_plain(output_name: str, output_labels: str, term: dict) -> str:
    factors = " ".join(f"{t['name']}[{t['labels']}]" for t in term["tensors"])
    encoded = term.get("coefficient")
    coeff_value = (
        float(Fraction(int(encoded["numerator"]), int(encoded["denominator"])))
        if encoded is not None
        else float(term["coeff"])
    )
    coeff = _coeff_text(coeff_value)
    return f"{_output_plain(output_name, output_labels)} ({coeff}) {factors}".rstrip()


def _scaling(output_labels: str, tensors: list[dict]) -> str:
    labels = set(_tokenize(output_labels))
    for tensor in tensors:
        labels.update(_tokenize(tensor["labels"]))
    return f"O(N^{len(labels)})"


def _projection_lines() -> list[str]:
    return [
        "Projection formulas:",
        "  S(phi)   = <Phi|Phi(phi)>",
        "  x(phi)   = exp(-i N phi) S(phi)",
        "  y(phi)   = x(phi) / sum_phi' x(phi')",
        "  P_N      = sum_phi y(phi) exp(i phi Nhat)",
        "  Z_pq(phi)= <Phi| beta_p beta_q |Phi(phi)> / <Phi|Phi(phi)>",
        "  E^N      = sum_phi y(phi) E(phi)",
        "  R1^N     = sum_phi y(phi) R1(phi)",
        "  R2^N     = sum_phi y(phi) R2(phi)",
    ]


def build_raw_projection_report(plan: dict, *, plan_path: Path | None = None) -> dict:
    outputs = {}
    for output_key in ("scalar", "X1", "X2"):
        block = plan["outputs"][output_key]
        outputs[output_key] = {
            "name": block["name"],
            "labels": block["labels"],
            "raw_term_count": len(block["raw_terms"]),
            "raw_terms": block["raw_terms"],
        }
    return {
        "theory": "projected_qpccsd",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_plan": str(plan_path) if plan_path is not None else None,
        "outputs": outputs,
    }


def _tct_lines(report: dict) -> list[str]:
    outputs = report["outputs"]
    lines = [
        "Projected qp-CCSD Raw Projection Terms",
        "artifact_type = raw_projection_terms",
        f"theory        = {report['theory']}",
        f"generated_utc = {report['generated_at_utc']}",
        f"source_plan   = {report['source_plan']}",
        "",
        "Counts:",
        f"  energy = {outputs['scalar']['raw_term_count']}",
        f"  r1     = {outputs['X1']['raw_term_count']}",
        f"  r2     = {outputs['X2']['raw_term_count']}",
        "",
        * _projection_lines(),
        "",
        "Definitions:",
        "  raw symbolic terms = canonical projected contractions before intermediate hoisting",
        "  emitted terms are intentionally excluded from this artifact",
        "",
    ]

    for output_key in ("scalar", "X1", "X2"):
        block = outputs[output_key]
        lines.extend(
            [
                f"[{block['name']}]",
                f"labels = {block['labels']!r}",
                f"count  = {block['raw_term_count']}",
                "",
            ]
        )
        for idx, term in enumerate(block["raw_terms"], start=1):
            scaling = _scaling(block["labels"], term["tensors"])
            lines.append(f"{idx:7d}: {scaling:>8}  {_term_plain(block['name'], block['labels'], term)}")
        lines.append("")
    return lines


def write_raw_tct(plan: dict, out_path: Path, *, plan_path: Path | None = None) -> dict:
    report = build_raw_projection_report(plan, plan_path=plan_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(_tct_lines(report)) + "\n")
    return report


def _render_lines_pdf(lines: list[str], pdf_path: Path) -> bool:
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
    except Exception:
        return False

    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=110) or [""])

    with PdfPages(pdf_path) as pdf:
        page_height = 64
        for start in range(0, len(wrapped), page_height):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(
                0.05,
                0.97,
                "\n".join(wrapped[start : start + page_height]),
                family="monospace",
                fontsize=7,
                va="top",
            )
            pdf.savefig(fig)
            plt.close(fig)
    return True


def render_raw_split_pdfs(plan: dict, out_dir: Path, *, terms_per_pdf: int = 50000) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    overview = [
        "Projected qp-CCSD Raw Projection Terms",
        "",
        "Layout:",
        "  Portrait pages only.",
        "  Raw symbolic terms only.",
        "  Large sections are split across multiple PDFs.",
        "",
        * _projection_lines(),
        "",
        f"terms_per_pdf = {terms_per_pdf}",
        "",
    ]

    name_map = {"scalar": "energy", "X1": "r1", "X2": "r2"}
    for output_key in ("scalar", "X1", "X2"):
        block = plan["outputs"][output_key]
        output_name = block["name"]
        output_slug = name_map[output_key]
        terms = block["raw_terms"]
        total_parts = max(1, (len(terms) + terms_per_pdf - 1) // terms_per_pdf)
        overview.append(f"{output_name}: raw={len(terms)}")
        for part_idx in range(total_parts):
            start = part_idx * terms_per_pdf
            stop = min(len(terms), start + terms_per_pdf)
            chunk = terms[start:stop]
            pdf_name = (
                f"projected_qp_raw_{output_slug}.pdf"
                if total_parts == 1
                else f"projected_qp_raw_{output_slug}_part{part_idx + 1:03d}.pdf"
            )
            pdf_path = out_dir / pdf_name
            lines = [
                "Projected qp-CCSD Raw Terms",
                "",
                f"section = {output_name}",
                f"labels  = {block['labels']!r}",
                f"part    = {part_idx + 1}/{total_parts}",
                f"terms   = {start + 1}..{stop} of {len(terms)}",
                "",
                * _projection_lines(),
                "",
            ]
            for idx, term in enumerate(chunk, start=start + 1):
                lines.append(f"{idx:7d}: {_term_plain(output_name, block['labels'], term)}")
            ok = _render_lines_pdf(lines, pdf_path)
            if not ok:
                raise SystemExit("Failed to build portrait PDFs. Install matplotlib.")
            generated.append(pdf_path)
            overview.append(f"  {pdf_name}")

    overview_path = out_dir / "projected_qp_raw_index.pdf"
    ok = _render_lines_pdf(overview, overview_path)
    if not ok:
        raise SystemExit("Failed to build portrait PDF index. Install matplotlib.")
    generated.insert(0, overview_path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Render raw-only projected qp-CCSD artifacts.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--tct", type=Path, default=DEFAULT_TCT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--pdf-split", action="store_true")
    parser.add_argument("--terms-per-pdf", type=int, default=50000)
    args = parser.parse_args()

    with args.plan.open() as handle:
        plan = json.load(handle)

    report = write_raw_tct(plan, args.tct, plan_path=args.plan)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.tct}")
    print(f"Wrote {args.summary_json}")

    if args.pdf_split:
        generated = render_raw_split_pdfs(plan, args.pdf_dir, terms_per_pdf=args.terms_per_pdf)
        for path in generated:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
