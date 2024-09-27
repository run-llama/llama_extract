import os
import pytest

from llama_extract import LlamaExtract
from llama_extract.base import ExtractionSchema, ExtractionResult


TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/test.pdf")


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
def test_simple() -> None:
    extractor = LlamaExtract(
        api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    )

    # Infer schema
    schema = extractor.infer_schema("my_schema", [TEST_FILE])
    assert isinstance(schema, ExtractionSchema)

    # Extract data
    results = extractor.extract(schema.id, [TEST_FILE])
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], ExtractionResult)
