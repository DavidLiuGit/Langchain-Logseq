## Integration Tests

These tests also serve as implementation & usage examples.

It is common for integration tests to use an LLM via an API, meaning you will likely be paying for the API costs.


### Usage
Be sure to first install test dependencies.
```bash
pip install .[test]
```

To explicitly run one test, from project root dir:
```bash
PYTHONPATH=. python integ-tests/test_integ_journal_date_range_retriever.py
```

To run all integration tests, from project root dir:
```bash
pytest -v --log-cli-level=INFO \
  --cov=langchain_logseq integ-tests/ \
  --cov-report=xml --cov-report=html --cov-report=term
```
