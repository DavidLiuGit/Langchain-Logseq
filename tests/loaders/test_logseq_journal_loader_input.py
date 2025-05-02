import unittest
from datetime import datetime

from langchain_logseq.loaders.logseq_journal_loader_input import (
    LogseqJournalLoaderInput,
    _validate_date_fields
)


class TestLogseqJournalLoaderInput(unittest.TestCase):
    def test_validate_date_fields_valid(self):
        """Test that _validate_date_fields accepts valid date formats."""
        # Test with valid date formats
        valid_dates = ["2023-01-01", "2025-06-09", "2022-12-31"]
        for date in valid_dates:
            # This should not raise an exception
            result = _validate_date_fields(date)
            # Function should return the input value
            self.assertEqual(result, date)


    def test_validate_date_fields_invalid(self):
        """Test that _validate_date_fields rejects invalid date formats."""
        # Test with invalid date formats
        invalid_dates = [
            "2023-02-30",  # Invalid day (February 30th doesn't exist)
            "2023-13-01",  # Invalid month (month 13 doesn't exist)
            "abcd-ef-gh",  # Not a date at all
            "202-01-01",   # Year too short
            "20231-01-01", # Year too long
            "",            # Empty string
        ]
        
        for date in invalid_dates:
            print(date)
            with self.assertRaises(ValueError) as context:
                _validate_date_fields(date)
            
            self.assertEqual(str(context.exception), "Dates must be in YYYY-MM-DD format.")


    def test_create_valid_input(self):
        """Test creating a LogseqJournalLoaderInput with valid data."""
        # Create with valid data
        input_data = LogseqJournalLoaderInput(
            journal_start_date="2023-01-01",
            journal_end_date="2023-12-31"
        )
        
        # Check that the fields are set correctly
        self.assertEqual(input_data.journal_start_date, "2023-01-01")
        self.assertEqual(input_data.journal_end_date, "2023-12-31")
        self.assertEqual(input_data.max_char_length, 8192)  # Default value


    def test_create_with_custom_max_char_length(self):
        """Test creating a LogseqJournalLoaderInput with a custom max_char_length."""
        # Create with custom max_char_length
        input_data = LogseqJournalLoaderInput(
            journal_start_date="2023-01-01",
            journal_end_date="2023-12-31",
            max_char_length=4096
        )
        
        # Check that max_char_length is set correctly
        self.assertEqual(input_data.max_char_length, 4096)


    def test_create_with_invalid_start_date(self):
        """Test that creating with an invalid start date raises a validation error."""
        with self.assertRaises(ValueError) as context:
            LogseqJournalLoaderInput(
                journal_start_date="01-01-2023",  # Invalid format
                journal_end_date="2023-12-31"
            )
        
        self.assertIn("Dates must be in YYYY-MM-DD format", str(context.exception))


    def test_create_with_invalid_end_date(self):
        """Test that creating with an invalid end date raises a validation error."""
        with self.assertRaises(ValueError) as context:
            LogseqJournalLoaderInput(
                journal_start_date="2023-01-01",
                journal_end_date="2023/12/31"  # Invalid format
            )
        
        self.assertIn("Dates must be in YYYY-MM-DD format", str(context.exception))


    def test_model_validation(self):
        """Test the model validation using the model_validate method."""
        # Valid data
        valid_data = {
            "journal_start_date": "2023-01-01",
            "journal_end_date": "2023-12-31"
        }
        
        # This should not raise an exception
        input_data = LogseqJournalLoaderInput.model_validate(valid_data)
        self.assertEqual(input_data.journal_start_date, "2023-01-01")
        self.assertEqual(input_data.journal_end_date, "2023-12-31")
        
        # Invalid data
        invalid_data = {
            "journal_start_date": "01/01/2023",  # Invalid format
            "journal_end_date": "2023-12-31"
        }
        
        # This should raise a validation error
        with self.assertRaises(ValueError):
            LogseqJournalLoaderInput.model_validate(invalid_data)


if __name__ == "__main__":
    unittest.main()
