"""Retrieval evaluation harness - shared library.

Kept deliberately small: corpus loading, chunking, two retrievers, metrics, and one cached
LLM wrapper. The eval driver, the report generator and the regression gate live under
`eval/` and are not part of the importable surface, because they are tools that run the
library rather than components of it.
"""
