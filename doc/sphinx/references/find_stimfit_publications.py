#!/usr/bin/env python3
"""
Find new Stimfit publications using the same strategy used for reference curation:

1) OpenAlex: papers citing the original Stimfit paper (10.3389/fninf.2014.00016)
2) Europe PMC: full-text search for "stimfit" + "patch-clamp" (methods-like signal)
3) Merge, de-duplicate, screen confounders, compare against index.rst
4) Emit JSON report and optional RST snippet for copy/paste
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Set


ORIGINAL_STIMFIT_DOI = "10.3389/fninf.2014.00016"


@dataclass
class Record:
    doi: str
    title: str
    year: int
    authors: List[str]
    venue: str
    url: str
    is_preprint: bool
    source_tags: Set[str]


def http_json(url: str, timeout: int = 60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def norm_title(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"<[^>]+>", "", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-\+αβγδ]+", " ", s)
    return norm_ws(s)


def initials_from_name(name: str) -> str:
    out = []
    for tok in re.split(r"[\s\-]+", name.strip()):
        tok = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", tok)
        if tok:
            out.append(tok[0].upper())
    return "".join(out)


def format_author_last_initials(full_name: str) -> str:
    parts = [p for p in full_name.strip().split() if p]
    if len(parts) < 2:
        return full_name.strip()
    particles = {"de", "da", "del", "della", "di", "du", "van", "von", "der", "den", "la", "le"}
    if len(parts) >= 2 and parts[-2].lower() in particles:
        last = " ".join(parts[-2:])
        first = " ".join(parts[:-2])
    else:
        last = parts[-1]
        first = " ".join(parts[:-1])
    return f"{last} {initials_from_name(first)}".strip()


def parse_existing_rst(rst_path: str):
    text = open(rst_path, "r", encoding="utf-8").read()
    dois = {d.lower() for d in re.findall(r"https?://doi.org/([^>\s]+)", text, re.I)}
    titles = set()
    for m in re.finditer(r"`_\s*(.*?)\s*\.\s*\*", text):
        titles.add(norm_title(m.group(1)))
    return dois, titles


def openalex_work_from_doi(doi: str):
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi, safe="/:")
    return http_json(url)


def openalex_citing_records(since_year: int) -> Dict[str, Record]:
    base = "https://api.openalex.org/works"
    # OpenAlex filter syntax for citation targets uses work IDs (W...), not DOI in cites:.
    root = openalex_work_from_doi(ORIGINAL_STIMFIT_DOI)
    root_id = (root.get("id") or "").rsplit("/", 1)[-1]
    if not root_id:
        raise RuntimeError("Could not resolve OpenAlex ID for original Stimfit DOI")
    filt = f"cites:{root_id},from_publication_date:{since_year}-01-01"
    params = {
        "filter": filt,
        "per-page": "200",
        "select": "display_name,publication_year,doi,primary_location,authorships,type",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    data = http_json(url)
    out: Dict[str, Record] = {}
    for w in data.get("results", []):
        doi = (w.get("doi") or "").replace("https://doi.org/", "").strip().lower()
        if not doi:
            continue
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        authors = [
            format_author_last_initials(a.get("author", {}).get("display_name", ""))
            for a in (w.get("authorships") or [])
            if a.get("author", {}).get("display_name")
        ]
        rec = Record(
            doi=doi,
            title=norm_ws(html.unescape(w.get("display_name", ""))),
            year=int(w.get("publication_year") or 0),
            authors=authors,
            venue=venue,
            url=f"https://doi.org/{doi}",
            is_preprint=(w.get("type") == "preprint" or "biorxiv" in venue.lower() or "research square" in venue.lower()),
            source_tags={"openalex_cites"},
        )
        out[doi] = rec
    return out


def europepmc_method_dois(since_year: int) -> Set[str]:
    query = f'("stimfit" AND "patch-clamp") AND FIRST_PDATE:[{since_year}-01-01 TO 2100-12-31]'
    params = {"query": query, "format": "json", "pageSize": "1000"}
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(params)
    data = http_json(url)
    out = set()
    for r in data.get("resultList", {}).get("result", []):
        doi = (r.get("doi") or "").strip().lower()
        if doi:
            out.add(doi)
    return out


def likely_confounder(title: str, venue: str) -> bool:
    s = f"{title} {venue}".lower()
    bad = [
        "mri",
        "fmr",
        "deep brain stimulation programming",
        "stimfit-a data-driven algorithm for automated deep brain stimulation",
        "renal t2 mapping",
    ]
    return any(b in s for b in bad)


def to_rst_line(r: Record) -> str:
    authors = r.authors[:4]
    author_txt = ", ".join(authors) + (", et al." if len(r.authors) > 4 else "")
    venue = r.venue or "Unknown venue"
    if r.is_preprint and "preprint" not in venue.lower():
        venue = f"{venue} [preprint]"
    return f"    * `{author_txt} ({r.year}) <https://doi.org/{r.doi}>`_ {r.title}. *{venue}*."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rst-path", default="doc/sphinx/references/index.rst")
    ap.add_argument("--since-year", type=int, default=2022)
    ap.add_argument("--out-json", default="doc/sphinx/references/stimfit_candidates.json")
    ap.add_argument("--out-rst-snippet", default="doc/sphinx/references/stimfit_new_entries.rst")
    args = ap.parse_args()

    existing_dois, existing_titles = parse_existing_rst(args.rst_path)

    cited = openalex_citing_records(args.since_year)
    method_dois = europepmc_method_dois(args.since_year)

    merged: Dict[str, Record] = dict(cited)

    # Add method-only DOIs via OpenAlex metadata lookup
    for doi in sorted(method_dois):
        if doi in merged:
            merged[doi].source_tags.add("europepmc_methods")
            continue
        try:
            w = openalex_work_from_doi(doi)
        except Exception:
            continue
        year = int(w.get("publication_year") or 0)
        if year < args.since_year:
            continue
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        authors = [
            format_author_last_initials(a.get("author", {}).get("display_name", ""))
            for a in (w.get("authorships") or [])
            if a.get("author", {}).get("display_name")
        ]
        merged[doi] = Record(
            doi=doi,
            title=norm_ws(html.unescape(w.get("display_name", ""))),
            year=year,
            authors=authors,
            venue=venue,
            url=f"https://doi.org/{doi}",
            is_preprint=(w.get("type") == "preprint" or "biorxiv" in venue.lower() or "research square" in venue.lower()),
            source_tags={"europepmc_methods"},
        )

    # Screen + novelty check
    candidates: List[Record] = []
    for rec in merged.values():
        if likely_confounder(rec.title, rec.venue):
            continue
        if rec.doi.lower() in existing_dois:
            continue
        if norm_title(rec.title) in existing_titles:
            continue
        candidates.append(rec)

    candidates.sort(key=lambda r: (r.year, r.title.lower()), reverse=True)

    # JSON report
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    **asdict(c),
                    "source_tags": sorted(list(c.source_tags)),
                }
                for c in candidates
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # RST snippet grouped by year
    by_year: Dict[int, List[Record]] = defaultdict(list)
    for c in candidates:
        by_year[c.year].append(c)

    snippet_lines: List[str] = []
    for y in sorted(by_year.keys(), reverse=True):
        snippet_lines += ["====", str(y), "====", ""]
        for r in sorted(by_year[y], key=lambda x: x.title.lower()):
            snippet_lines.append(to_rst_line(r))
            snippet_lines.append("")

    with open(args.out_rst_snippet, "w", encoding="utf-8") as f:
        f.write("\n".join(snippet_lines).rstrip() + "\n")

    print(f"Existing DOI entries: {len(existing_dois)}")
    print(f"Cited candidates (OpenAlex): {len(cited)}")
    print(f"Methods-query DOIs (Europe PMC): {len(method_dois)}")
    print(f"New non-duplicate candidates: {len(candidates)}")
    print(f"Wrote JSON: {args.out_json}")
    print(f"Wrote RST snippet: {args.out_rst_snippet}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
