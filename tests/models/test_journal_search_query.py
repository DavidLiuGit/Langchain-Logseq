import unittest

from logseq_retriever.models.journal_pgvector import (
    JournalSearchQuery,
    JournalDocumentMetadata,
)
from logseq_retriever.models import MetadataFilter


class TestJournalSearchQuery(unittest.TestCase):
    def test_model_json_schema_includes_metadata_reference(self):
        """Test that JournalSearchQuery's schema includes JournalDocumentMetadata schema reference."""
        schema = JournalSearchQuery.model_json_schema()

        # Check that metadata_filters field exists
        self.assertIn("metadata_filters", schema["properties"])

        # Check that metadata_filters has metadata_schema
        metadata_filters_field = schema["properties"]["metadata_filters"]
        self.assertIn("metadata_schema", metadata_filters_field)

        # Verify the metadata_schema matches JournalDocumentMetadata's schema
        expected_metadata_schema = JournalDocumentMetadata.model_json_schema()
        actual_metadata_schema = metadata_filters_field["metadata_schema"]

        self.assertEqual(actual_metadata_schema, expected_metadata_schema)

    def test_metadata_schema_contains_expected_fields(self):
        """Test that the referenced metadata schema contains expected JournalDocumentMetadata fields."""
        schema = JournalSearchQuery.model_json_schema()
        metadata_schema = schema["properties"]["metadata_filters"]["metadata_schema"]

        # Check for JournalDocumentMetadata specific fields
        properties = metadata_schema["properties"]
        self.assertIn("date_str", properties)
        self.assertIn("chunk_len", properties)
        self.assertIn("word_count", properties)
        self.assertIn("references", properties)
        self.assertIn("anchor_ids", properties)
        self.assertIn("document_type", properties)
        self.assertIn("schema_version", properties)

    def test_metadata_schema_field_types(self):
        """Test that metadata schema fields have correct types."""
        schema = JournalSearchQuery.model_json_schema()
        properties = schema["properties"]["metadata_filters"]["metadata_schema"][
            "properties"
        ]

        self.assertEqual(properties["date_str"]["type"], "string")
        self.assertEqual(properties["chunk_len"]["type"], "integer")
        self.assertEqual(properties["references"]["type"], "array")
        self.assertEqual(properties["anchor_ids"]["type"], "array")

    def test_metadata_schema_required_fields(self):
        """Test that metadata schema has correct required fields."""
        schema = JournalSearchQuery.model_json_schema()
        metadata_schema = schema["properties"]["metadata_filters"]["metadata_schema"]

        required_fields = metadata_schema["required"]
        self.assertIn("date_str", required_fields)
        self.assertIn("chunk_len", required_fields)
        self.assertIn("word_count", required_fields)


class TestJournalSearchQueryDateFields(unittest.TestCase):
    def test_date_from_expands_to_gte_filter(self):
        q = JournalSearchQuery(date_from="2025-01-01")
        self.assertEqual(len(q.metadata_filters), 1)
        self.assertEqual(q.metadata_filters[0].field_name, "date_str")
        self.assertEqual(q.metadata_filters[0].condition, "gte")
        self.assertEqual(q.metadata_filters[0].value, "2025-01-01")

    def test_date_to_expands_to_lte_filter(self):
        q = JournalSearchQuery(date_to="2025-03-31")
        self.assertEqual(len(q.metadata_filters), 1)
        self.assertEqual(q.metadata_filters[0].condition, "lte")
        self.assertEqual(q.metadata_filters[0].value, "2025-03-31")

    def test_both_dates_expand_to_two_filters(self):
        q = JournalSearchQuery(date_from="2025-01-01", date_to="2025-03-31")
        self.assertEqual(len(q.metadata_filters), 2)
        conditions = {f.condition for f in q.metadata_filters}
        self.assertEqual(conditions, {"gte", "lte"})

    def test_date_from_after_date_to_raises(self):
        with self.assertRaises(ValueError):
            JournalSearchQuery(date_from="2025-06-01", date_to="2025-01-01")

    def test_date_only_query_is_valid(self):
        """date_from alone should satisfy ensure_criterion without text/keywords."""
        q = JournalSearchQuery(date_from="2025-01-01")
        self.assertIsNotNone(q)

    def test_invalid_date_format_raises(self):
        with self.assertRaises(ValueError):
            JournalSearchQuery(date_from="01-01-2025")

    def test_additional_metadata_filters_preserved(self):
        extra = MetadataFilter(
            field_name="references", condition="contains", value="health"
        )
        q = JournalSearchQuery(date_from="2025-01-01", metadata_filters=[extra])
        self.assertEqual(len(q.metadata_filters), 2)
        field_names = {f.field_name for f in q.metadata_filters}
        self.assertIn("date_str", field_names)
        self.assertIn("references", field_names)

    def test_date_filters_not_duplicated_on_revalidation(self):
        """Re-validating with already-expanded metadata_filters must not double-append date entries."""
        q = JournalSearchQuery(date_from="2025-01-01")
        # Simulate re-validation with the already-expanded filters passed back in
        q2 = JournalSearchQuery.model_validate(
            {
                "date_from": "2025-01-01",
                "date_to": "2025-06-01",
                "metadata_filters": [f.model_dump() for f in q.metadata_filters],
            }
        )
        date_str_filters = [
            f for f in q2.metadata_filters if f.field_name == "date_str"
        ]
        self.assertEqual(len(date_str_filters), 2)  # gte + lte, not 3

    def test_no_criteria_raises(self):
        with self.assertRaises(ValueError):
            JournalSearchQuery(limit=10)  # no text, keywords, dates, or filters

    def test_metadata_filter_re_exported(self):
        """MetadataFilter should be importable from logseq_retriever.models."""
        from logseq_retriever.models import MetadataFilter as MF

        self.assertIs(MF, MetadataFilter)


if __name__ == "__main__":
    unittest.main()
