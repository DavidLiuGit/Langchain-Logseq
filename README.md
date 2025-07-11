# Langchain Logseq
Collection of Langchain utilities for working with Logseq files.

---
## Components
This section provides an overview of the components provided, listed by type


### Retrievers
Retrievers inject context into a conversation. Works in tandem with a Contextualizer and `Document` Loader.
- **Input**:
  - natural-language user-input, usually query-like
  - (optional) chat history
- **Output**:
  - list of `Document`s to provide context for an LLM to answer the user-input

#### Implementations
- `LogseqJournalDateRangeRetriever`
  - retrieve Logseq journal `Document`s, intended for queries that require context from a date range
  - required to set up:
    - `RetrieverContextualizer`
    - `LogseqJournalLoader`
  - examples:
    - "What did I do over Christmas break 2024?"
    - "How did I spend the last Independence Day?"


### Contextualizers
Contextualizers serve as the bridge between natural-language input and a downstream component that
handles fetching of relevant `Document`s.
- **Input**:
  - natural-language user-input, usually query-like
  - (optional) chat history
- **Output**:
  - structured downstream query, based on

In this library, an instance of `RetrieverContextualizer` is provided directly to 
`Retriever`s during the latter's instantiation. To set up the `RetrieverContextualizer`, provide
`RetrieverContextualizerProps`, which includes:
- `llm` - this is the backbone of the contextualizer
- `prompt` - instructions provided to the LLM
- `output_schema` - (optional) structured schema used to fetch relevant `Document`s
  - if no schema is provided, a string shall be returned instead
- other flags and settings


### Loaders
Loaders are one type of component that can fetch relevant `Document`s. Loaders are typically specific to
a corresponding `Retriever` component.
- **Input**:
  - each loader specifies its own schema
    - the Contextualizer is usually responsible for creating an instance of the query obj to act upon
- **Output**:
  - `list[Document]`

#### Implementations
- `LogseqJournalFilesystemLoader`
  - loads from the filesystem, where journal files are expected to be present at specified path


---
## Scripts

### PGVector

#### `upload_journal`

usage: `python scripts/upload_journal_to_pgvector.py [-h] [-p PATH] from_date to_date`
