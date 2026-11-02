"""Checks that do not call the model.

Run from the project root:

    python -m unittest discover -s tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import analyze  # noqa: E402
from dashboard import canonical_products, review_call_type  # noqa: E402
from load_data import load_all_meetings, load_meeting  # noqa: E402


def _write_meeting(root, folder, emails, title, sentences, summary=None):
    path = Path(root) / folder
    path.mkdir()
    (path / "transcript.json").write_text(json.dumps({
        "data": [
            {"speaker_name": "Alex", "sentence": sentence}
            for sentence in sentences
        ]
    }))
    (path / "meeting-info.json").write_text(json.dumps({
        "meetingId": folder,
        "title": title,
        "host": emails[0],
        "startTime": "2026-03-10T09:00:00.000Z",
        "endTime": "2026-03-10T09:20:00.000Z",
        "duration": 20,
        "invitees": emails,
        "allEmails": emails,
    }))
    if summary is not None:
        (path / "summary.json").write_text(json.dumps(summary))
    return path


def _valid_result(**overrides):
    result = {
        "call_type": "support",
        "primary_topic": "Backup performance",
        "secondary_topics": ["Aegis Protect"],
        "category": "Technical Support",
        "sentiment_score": -0.4,
        "sentiment_label": "negative",
        "escalation_risk": True,
        "summary": "The backup jobs slowed down. Support escalated the ticket.",
        "evidence": "nightly backup jobs are taking four to six hours",
        "is_business_call": True,
        "products_mentioned": ["Aegis Protect"],
        "confidence_score": 0.9,
    }
    result.update(overrides)
    return result


class LoadMeetingTests(unittest.TestCase):
    def test_call_type_follows_email_domain_and_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            internal = _write_meeting(
                tmp, "internal-sync",
                ["a@aegiscloud.com", "b@aegiscloud.com"],
                "Weekly Engineering Standup",
                ["Let's review the rollout plan."],
            )
            external = _write_meeting(
                tmp, "renewal",
                ["a@aegiscloud.com", "c@customer.com"],
                "Aegis / Northwind - Renewal",
                ["We should talk through the renewal."],
            )
            support = _write_meeting(
                tmp, "case",
                ["a@aegiscloud.com", "c@customer.com"],
                "Support Case #1234 - Slow Backup",
                ["The nightly job is taking six hours."],
                summary={
                    "overallSentiment": "negative",
                    "sentimentScore": -0.4,
                    "keyMoments": [{"type": "churn_signal"}, {"type": "concern"}],
                },
            )

            internal_row = load_meeting(str(internal))
            external_row = load_meeting(str(external))
            support_row = load_meeting(str(support))

        self.assertEqual(internal_row["call_type_derived"], "internal")
        self.assertEqual(external_row["call_type_derived"], "external")
        self.assertEqual(support_row["call_type_derived"], "support")
        self.assertIn("Alex: The nightly job is taking six hours.", support_row["transcript"])
        self.assertEqual(support_row["sentence_count"], 1)
        self.assertTrue(support_row["has_churn_signal"])
        self.assertTrue(support_row["has_concern"])

    def test_incomplete_folder_is_recorded_as_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_meeting(
                tmp, "good",
                ["a@aegiscloud.com"],
                "Standup",
                ["Hello."],
            )
            bad = Path(tmp) / "missing-transcript"
            bad.mkdir()
            (bad / "meeting-info.json").write_text("{}")

            meetings, failed = load_all_meetings(tmp)

        self.assertEqual([m["folder"] for m in meetings], ["good"])
        self.assertEqual(failed, ["missing-transcript"])


class ValidateResultTests(unittest.TestCase):
    def test_complete_result_is_accepted(self):
        ok, reason = analyze.validate_llm_result(_valid_result())
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_missing_field_and_out_of_range_sentiment_are_rejected(self):
        missing = _valid_result()
        del missing["evidence"]
        ok, reason = analyze.validate_llm_result(missing)
        self.assertFalse(ok)
        self.assertIn("evidence", reason)

        out_of_range = _valid_result(sentiment_score=1.5)
        ok, reason = analyze.validate_llm_result(out_of_range)
        self.assertFalse(ok)
        self.assertIn("sentiment_score", reason)

        bad_type = _valid_result(call_type="sales")
        ok, reason = analyze.validate_llm_result(bad_type)
        self.assertFalse(ok)
        self.assertIn("call_type", reason)


class CacheTests(unittest.TestCase):
    def test_cached_response_is_returned_without_calling_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = {"call_type": "internal", "summary": "from cache"}
            (Path(tmp) / "meeting-1.json").write_text(json.dumps(cached))

            def _fail_if_called(*_args, **_kwargs):
                raise AssertionError("the model should not be called when the cache hits")

            original_cache = analyze.CACHE_DIR
            original_post = analyze.requests.post
            analyze.CACHE_DIR = tmp
            analyze.requests.post = _fail_if_called
            try:
                result = analyze.call_llm({"folder": "meeting-1"}, force=False)
            finally:
                analyze.CACHE_DIR = original_cache
                analyze.requests.post = original_post

        self.assertEqual(result, cached)


class DashboardReviewTests(unittest.TestCase):
    def test_review_moves_labels_and_groups_product_aliases(self):
        same_company = "['a@aegiscloud.com', 'b@aegiscloud.com']"
        with_customer = "['a@aegiscloud.com', 'c@customer.com']"

        self.assertEqual(
            review_call_type(same_company, "Account Review", "Account Management", "external"),
            "internal",
        )
        self.assertEqual(
            review_call_type(with_customer, "Support Case #9 - Login", "Technical Support", "internal"),
            "support",
        )
        self.assertEqual(
            review_call_type(with_customer, "Aegis / Northwind - Renewal", "Account Management", "support"),
            "external",
        )
        self.assertEqual(
            canonical_products(["Detect", "Aegis Detect", "comply v2", "not a product"]),
            ["Aegis Detect", "Comply v2"],
        )


if __name__ == "__main__":
    unittest.main()
