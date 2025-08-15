import unittest

from langchain_logseq.models.journal_pgvector import JournalSearchQuery, JournalDocumentMetadata


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
        properties = schema["properties"]["metadata_filters"]["metadata_schema"]["properties"]
        
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


if __name__ == "__main__":
    unittest.main()
