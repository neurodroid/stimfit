#!/usr/bin/env python3
"""
Find new Stimfit publications using the same strategy used for reference curation:

1) OpenAlex: papers citing the original Stimfit paper (10.3389/fninf.2014.00016)
2) Europe PMC: full-text search for "stimfit" + "patch-clamp" (methods-like signal)
3) Prefer journal versions using curated DOI pairs, Crossref and bioRxiv/medRxiv
4) Merge, de-duplicate, screen confounders, compare against index.rst
5) Emit JSON report and optional RST snippet for copy/paste

publication_versions.json preserves manually verified preprint/journal DOI pairs
when public metadata is incomplete. Unpublished preprints remain eligible; those
whose publication status cannot be checked are withheld with a warning.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, asdict
from functools import cache
from pathlib import Path
from typing import Dict, List, Set


ORIGINAL_STIMFIT_DOI = "10.3389/fninf.2014.00016"
PUBLICATION_VERSIONS_PATH = Path(__file__).with_name("publication_versions.json")


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
    request = urllib.request.Request(url, headers={"User-Agent": "Stimfit-publication-finder/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as r:
        return json.load(r)


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def norm_title(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"<[^>]+>", "", s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s\-\+αβγδ]+", " ", s)
    return norm_ws(s)


def norm_doi(value: str) -> str:
    value = urllib.parse.unquote(value or "").strip().lower()
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else ""


def is_preprint(doi: str, venue: str = "", work_type: str = "") -> bool:
    return (
        work_type in {"preprint", "posted-content"}
        or bool(re.match(r"10\.(?:1101|64898)/\d|10\.21203/rs\.|10\.2139/ssrn\.", norm_doi(doi)))
        or any(name in venue.lower() for name in ("preprint", "biorxiv", "medrxiv", "research square", "ssrn", "arxiv"))
    )


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
    """Return existing DOIs and journal titles, allowing upgrades of preprints."""
    text = Path(rst_path).read_text(encoding="utf-8")
    dois = set()
    titles = set()
    for line in text.splitlines():
        for doi in re.findall(r"https?://(?:dx\.)?doi.org/([^>\s]+)", line, re.I):
            dois.add(norm_doi(doi))
        m = re.search(r"<([^>]+)>`_\s*(.*?)\s*\.\s*\*([^*]+)\*", line)
        if m and not is_preprint(m.group(1), line):
            titles.add(norm_title(m.group(2)))
    return dois, titles


@cache
def openalex_work_from_doi(doi: str):
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi, safe="/:")
    return http_json(url)


def record_from_openalex(w: dict, source_tags: Set[str]) -> Record:
    doi = norm_doi(w.get("doi"))
    venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
    return Record(
        doi=doi,
        title=norm_ws(html.unescape(w.get("display_name") or "")),
        year=int(w.get("publication_year") or 0),
        authors=[
            format_author_last_initials(a["author"]["display_name"])
            for a in (w.get("authorships") or [])
            if (a.get("author") or {}).get("display_name")
        ],
        venue=venue,
        url=f"https://doi.org/{doi}",
        is_preprint=is_preprint(doi, venue, w.get("type", "")),
        source_tags=set(source_tags),
    )


@cache
def crossref_work_from_doi(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    try:
        return http_json(url)["message"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {}
        raise


def record_from_crossref(w: dict, source_tags: Set[str]) -> Record:
    doi = norm_doi(w.get("DOI"))
    title = norm_ws(html.unescape(re.sub(r"<[^>]+>", "", (w.get("title") or [""])[0])))
    if w.get("type") != "journal-article" or not doi or not title:
        raise ValueError("Linked publication is not a journal article with a DOI and title")
    date = w.get("published") or w.get("issued") or {}
    return Record(
        doi=doi,
        title=title,
        year=int((date.get("date-parts") or [[0]])[0][0]),
        authors=[
            norm_ws(f"{a['family']} {initials_from_name(a.get('given', ''))}")
            if a.get("family") else a.get("name", "")
            for a in (w.get("author") or [])
        ],
        venue=(w.get("container-title") or [""])[0],
        url=f"https://doi.org/{doi}",
        is_preprint=False,
        source_tags=set(source_tags),
    )


def matching_journal_dois(rec: Record) -> Set[str]:
    """Conservatively cover missing links using an exact title and first author."""
    if not rec.authors or len(norm_title(rec.title)) < 25:
        return set()
    params = {"query.bibliographic": f"{rec.title} {rec.authors[0]}", "filter": "type:journal-article", "rows": "5"}
    data = http_json("https://api.crossref.org/works?" + urllib.parse.urlencode(params))
    matches = set()
    family = norm_title(rec.authors[0].rsplit(" ", 1)[0])
    for work in data["message"]["items"]:
        authors = work.get("author") or []
        if not authors or norm_title(authors[0].get("family", "")) != family:
            continue
        try:
            journal = record_from_crossref(work, set())
        except (ValueError, KeyError):
            continue
        # Earlier conference abstracts can have the same title and authors and
        # Crossref type journal-article; they do not supersede the preprint.
        if norm_title(journal.title) == norm_title(rec.title) and journal.year >= rec.year:
            matches.add(journal.doi)
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous journal title matches for {rec.doi}: {', '.join(sorted(matches))}")
    return matches


def published_dois(rec: Record) -> Set[str]:
    """Use explicit version links and exact bibliographic matches, never fuzzy titles.

    API documentation: https://api.biorxiv.org/ and
    https://www.crossref.org/documentation/retrieve-metadata/rest-api/rest-api-filters/
    """
    dois = set()
    checked = False
    errors = []
    try:
        work = crossref_work_from_doi(rec.doi)
        checked = True
        for link in (work.get("relation") or {}).get("is-preprint-of", []):
            if link.get("id-type", "").lower() == "doi":
                dois.add(norm_doi(link.get("id")))
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"Crossref: {exc}")

    if re.match(r"10\.(?:1101|64898)/\d", rec.doi) or any(
        server in rec.venue.lower() for server in ("biorxiv", "medrxiv")
    ):
        server = "medrxiv" if "medrxiv" in rec.venue.lower() else "biorxiv"
        try:
            url = f"https://api.biorxiv.org/pubs/{server}/{urllib.parse.quote(rec.doi, safe='/')}/na/json"
            data = http_json(url)
            statuses = [message.get("status", "").lower() for message in data.get("messages", [])]
            if not any(status == "ok" or status.startswith("no articles found") for status in statuses):
                raise ValueError("Unrecognized publication-status response")
            checked = True
            for entry in data.get("collection", []):
                if norm_doi(entry.get("biorxiv_doi")) == rec.doi:
                    dois.add(norm_doi(entry.get("published_doi")))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{server}: {exc}")
    if not dois and rec.authors and len(norm_title(rec.title)) >= 25:
        try:
            dois.update(matching_journal_dois(rec))
            checked = True
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"Crossref title search: {exc}")
    if not checked and not dois:
        raise RuntimeError("; ".join(errors))
    for error in errors:
        print(f"Warning: {rec.doi}: {error}", file=sys.stderr)
    return {doi for doi in dois if doi and doi != rec.doi and not is_preprint(doi)}


def prefer_journal_versions(
    records: Dict[str, Record], existing_dois: Set[str], existing_titles: Set[str],
    known_versions: Dict[str, str],
) -> Dict[str, Record]:
    out = {doi: rec for doi, rec in records.items() if not rec.is_preprint}
    journal_titles = existing_titles | {norm_title(rec.title) for rec in out.values()}
    for rec in records.values():
        if not rec.is_preprint:
            continue
        # Curated pairs also cover servers without version links in their metadata.
        try:
            versions = {known_versions[rec.doi]} if rec.doi in known_versions else published_dois(rec)
        except RuntimeError as exc:
            print(f"Warning: withholding unchecked preprint {rec.doi}: {exc}", file=sys.stderr)
            continue
        if not versions:
            if norm_title(rec.title) not in journal_titles:
                out[rec.doi] = rec
            continue
        for doi in sorted(versions):
            if doi in existing_dois:
                continue
            tags = rec.source_tags | {"journal_version"}
            if doi in out:
                out[doi].source_tags.update(tags)
                continue
            try:
                try:
                    journal = record_from_crossref(crossref_work_from_doi(doi), tags)
                except (OSError, ValueError, KeyError):
                    work = openalex_work_from_doi(doi)
                    location = work.get("primary_location") or {}
                    if (location.get("source") or {}).get("type") != "journal":
                        raise ValueError("Linked publication has no journal metadata")
                    journal = record_from_openalex(work, tags)
                if journal.is_preprint or journal.doi != doi or not journal.title or not journal.year:
                    raise ValueError("Incomplete or inconsistent journal metadata")
                out[doi] = journal
                journal_titles.add(norm_title(journal.title))
            except (OSError, ValueError, KeyError) as exc:
                print(f"Warning: withholding published preprint {rec.doi}; cannot load {doi}: {exc}", file=sys.stderr)
    # A later preprint may have resolved to a journal matching an earlier one.
    return {doi: rec for doi, rec in out.items() if not rec.is_preprint or norm_title(rec.title) not in journal_titles}


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
        rec = record_from_openalex(w, {"openalex_cites"})
        if rec.doi:
            out[rec.doi] = rec
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
    with PUBLICATION_VERSIONS_PATH.open(encoding="utf-8") as f:
        known_versions = {norm_doi(preprint): norm_doi(journal) for preprint, journal in json.load(f).items()}

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
        merged[doi] = record_from_openalex(w, {"europepmc_methods"})

    # Resolve versions before novelty checks: even an existing preprint may now
    # have a new journal version, with a different DOI, title, authors and year.
    preferred = prefer_journal_versions(merged, existing_dois, existing_titles, known_versions)

    # Screen + novelty check
    candidates: List[Record] = []
    for rec in preferred.values():
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
