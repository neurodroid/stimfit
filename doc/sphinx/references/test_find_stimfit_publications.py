"""Offline regression tests for publication-version selection.

Run with: python -m unittest discover -s doc/sphinx/references -v
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import find_stimfit_publications as finder


def record(doi="10.1101/2024.01.01.123456", title="A study of hippocampal synaptic plasticity", **kwargs):
    return finder.Record(
        doi=doi, title=title, year=kwargs.get("year", 2024),
        authors=["Smith AB"], venue=kwargs.get("venue", "bioRxiv"),
        url=f"https://doi.org/{doi}", is_preprint=kwargs.get("is_preprint", True),
        source_tags={"openalex_cites"},
    )


def journal_metadata(doi="10.1234/journal.2025", title="A published study of hippocampal synapses", family="Smith"):
    return {
        "DOI": doi, "type": "journal-article", "title": [title],
        "container-title": ["Journal of Neuroscience"],
        "published": {"date-parts": [[2025, 3, 1]]},
        "author": [{"family": family, "given": "Alice B"}],
    }


class PublicationVersionsTest(unittest.TestCase):
    def setUp(self):
        finder.crossref_work_from_doi.cache_clear()
        finder.openalex_work_from_doi.cache_clear()

    def tearDown(self):
        finder.crossref_work_from_doi.cache_clear()
        finder.openalex_work_from_doi.cache_clear()

    def test_all_curated_pairs_stay_suppressed_without_network(self):
        versions = json.loads(finder.PUBLICATION_VERSIONS_PATH.read_text())
        records = {doi: record(doi) for doi in versions}
        with patch.object(finder, "http_json", side_effect=AssertionError("Unexpected network access")):
            result = finder.prefer_journal_versions(records, set(versions.values()), set(), versions)
        self.assertEqual(result, {})

    def test_changed_title_and_year_are_replaced_from_crossref_link(self):
        preprint = record()
        journal = journal_metadata()

        def api(url):
            if url.endswith(preprint.doi):
                return {"message": {"relation": {"is-preprint-of": [
                    {"id-type": "doi", "id": "https://doi.org/" + journal["DOI"]},
                ]}}}
            if "/pubs/" in url:
                return {"messages": [{"status": "ok"}], "collection": []}
            if url.endswith(journal["DOI"]):
                return {"message": journal}
            self.fail(url)

        with patch.object(finder, "http_json", side_effect=api):
            result = finder.prefer_journal_versions({preprint.doi: preprint}, set(), set(), {})
        self.assertEqual(set(result), {journal["DOI"]})
        published = result[journal["DOI"]]
        self.assertEqual((published.title, published.year), (journal["title"][0], 2025))
        self.assertFalse(published.is_preprint)
        self.assertIn("journal_version", published.source_tags)

    def test_biorxiv_and_medrxiv_links_work_without_crossref_links(self):
        for venue, server in (("bioRxiv", "biorxiv"), ("medRxiv", "medrxiv")):
            with self.subTest(server=server):
                preprint = record(venue=venue)

                def api(url):
                    self.assertIn(f"/pubs/{server}/", url)
                    return {"messages": [{"status": "ok"}], "collection": [
                        {"biorxiv_doi": preprint.doi, "published_doi": "10.1234/journal.2025"},
                        {"biorxiv_doi": "10.1101/999999", "published_doi": "10.1234/unrelated"},
                    ]}

                with patch.object(finder, "crossref_work_from_doi", return_value={}), patch.object(finder, "http_json", side_effect=api):
                    self.assertEqual(finder.published_dois(preprint), {"10.1234/journal.2025"})

    def test_exact_title_and_author_fallback_finds_unlinked_journal(self):
        preprint = record(doi="10.21203/rs.3.rs-123/v1", venue="Research Square")
        journal = journal_metadata(title=preprint.title)
        with patch.object(finder, "crossref_work_from_doi", return_value={}), patch.object(
            finder, "http_json", return_value={"message": {"items": [journal]}},
        ):
            result = finder.prefer_journal_versions(
                {preprint.doi: preprint}, {journal["DOI"]}, set(), {},
            )
        self.assertEqual(result, {})

    def test_title_search_does_not_match_another_author_or_similar_paper(self):
        preprint = record()
        unrelated = [journal_metadata(title=preprint.title, family="Jones"), journal_metadata()]
        with patch.object(finder, "http_json", return_value={"message": {"items": unrelated}}):
            self.assertEqual(finder.matching_journal_dois(preprint), set())

    def test_title_search_excludes_earlier_conference_abstract(self):
        preprint = record()
        abstract = journal_metadata(doi="10.1234/conference.2023", title=preprint.title)
        abstract["published"] = {"date-parts": [[2023]]}
        journal = journal_metadata(title=preprint.title)
        with patch.object(finder, "http_json", return_value={"message": {"items": [abstract, journal]}}):
            self.assertEqual(finder.matching_journal_dois(preprint), {journal["DOI"]})

    def test_ambiguous_journal_matches_withhold_preprint_for_review(self):
        preprint = record(doi="10.21203/rs.3.rs-123/v1", venue="Research Square")
        matches = [journal_metadata(title=preprint.title), journal_metadata(doi="10.1234/other", title=preprint.title)]
        warnings = io.StringIO()
        with patch.object(finder, "crossref_work_from_doi", return_value={}), patch.object(
            finder, "http_json", return_value={"message": {"items": matches}},
        ), contextlib.redirect_stderr(warnings):
            result = finder.prefer_journal_versions({preprint.doi: preprint}, set(), set(), {})
        self.assertEqual(result, {})
        self.assertIn("Ambiguous journal title matches", warnings.getvalue())

    def test_unpublished_preprints_remain_eligible(self):
        preprint = record()
        with patch.object(finder, "published_dois", return_value=set()):
            self.assertEqual(
                finder.prefer_journal_versions({preprint.doi: preprint}, set(), set(), {}),
                {preprint.doi: preprint},
            )

    def test_existing_journal_title_suppresses_unlinked_preprint(self):
        preprint = record()
        with patch.object(finder, "published_dois", return_value=set()):
            result = finder.prefer_journal_versions(
                {preprint.doi: preprint}, set(), {finder.norm_title(preprint.title)}, {},
            )
        self.assertEqual(result, {})

    def test_already_discovered_journal_is_emitted_once_and_merges_provenance(self):
        preprint = record()
        journal = record("10.1234/journal.2025", is_preprint=False, venue="Journal")
        journal.source_tags = {"europepmc_methods"}
        with patch.object(finder, "http_json", side_effect=AssertionError("Unexpected network access")):
            result = finder.prefer_journal_versions(
                {preprint.doi: preprint, journal.doi: journal}, set(), set(), {preprint.doi: journal.doi},
            )
        self.assertEqual(set(result), {journal.doi})
        self.assertEqual(journal.source_tags, {"openalex_cites", "europepmc_methods", "journal_version"})

    def test_api_outage_withholds_unchecked_preprint(self):
        preprint = record()
        warnings = io.StringIO()
        with patch.object(finder, "http_json", side_effect=OSError("offline")), contextlib.redirect_stderr(warnings):
            result = finder.prefer_journal_versions({preprint.doi: preprint}, set(), set(), {})
        self.assertEqual(result, {})
        self.assertIn("withholding unchecked preprint", warnings.getvalue())

    def test_api_outage_with_missing_authors_still_withholds_preprint(self):
        preprint = record()
        preprint.authors = []
        with patch.object(finder, "http_json", side_effect=OSError("offline")), contextlib.redirect_stderr(io.StringIO()):
            result = finder.prefer_journal_versions({preprint.doi: preprint}, set(), set(), {})
        self.assertEqual(result, {})

    def test_missing_journal_metadata_never_reverts_to_preprint(self):
        preprint = record()
        with patch.object(finder, "http_json", side_effect=OSError("offline")), contextlib.redirect_stderr(io.StringIO()):
            result = finder.prefer_journal_versions(
                {preprint.doi: preprint}, set(), set(), {preprint.doi: "10.1234/journal.2025"},
            )
        self.assertEqual(result, {})

    def test_journal_metadata_falls_back_to_openalex(self):
        preprint = record()
        doi = "10.1234/journal.2025"
        work = {
            "doi": doi, "display_name": "A renamed journal article", "publication_year": 2025,
            "type": "article", "primary_location": {"source": {"type": "journal", "display_name": "Journal"}},
        }
        with patch.object(finder, "crossref_work_from_doi", side_effect=OSError("unavailable")), patch.object(
            finder, "openalex_work_from_doi", return_value=work,
        ):
            result = finder.prefer_journal_versions({preprint.doi: preprint}, set(), set(), {preprint.doi: doi})
        self.assertEqual(result[doi].title, work["display_name"])
        self.assertFalse(result[doi].is_preprint)

    def test_classifies_preprint_dois_even_when_openalex_calls_them_articles(self):
        for doi in ("10.64898/2026.03.04.709616", "10.1101/845461", "10.21203/rs.3.rs-123/v1", "10.2139/ssrn.3985190"):
            with self.subTest(doi=doi):
                self.assertTrue(finder.record_from_openalex({"doi": doi, "type": "article"}, set()).is_preprint)
        self.assertFalse(finder.is_preprint("10.1101/gad.123456"))

    def test_main_upgrades_existing_preprint_with_same_title(self):
        preprint = record(doi="10.21203/rs.3.rs-123/v1", venue="Research Square")
        journal = journal_metadata(title=preprint.title)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            rst = base / "index.rst"
            rst.write_text(finder.to_rst_line(preprint) + "\n")
            versions = base / "versions.json"
            versions.write_text(json.dumps({preprint.doi: journal["DOI"]}))
            args = ["finder", "--rst-path", str(rst), "--out-json", str(base / "report.json"),
                    "--out-rst-snippet", str(base / "snippet.rst")]
            with patch("sys.argv", args), patch.object(finder, "PUBLICATION_VERSIONS_PATH", versions), patch.object(
                finder, "openalex_citing_records", return_value={preprint.doi: preprint},
            ), patch.object(finder, "europepmc_method_dois", return_value=set()), patch.object(
                finder, "crossref_work_from_doi", return_value=journal,
            ), contextlib.redirect_stdout(io.StringIO()):
                finder.main()
            report = json.loads((base / "report.json").read_text())
            self.assertEqual([item["doi"] for item in report], [journal["DOI"]])
            snippet = (base / "snippet.rst").read_text()
            self.assertIn("2025\n====", snippet)
            self.assertNotIn("[preprint]", snippet)


if __name__ == "__main__":
    unittest.main()
