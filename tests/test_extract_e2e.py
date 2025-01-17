import os
import pytest
from pathlib import Path

from llama_extract import LlamaExtract, ExtractionAgent
from dotenv import load_dotenv
from collections import namedtuple
import json
from llama_cloud.core.api_error import ApiError
from deepdiff import DeepDiff
from tests.util import json_subset_match_score

load_dotenv(Path(__file__).parent.parent / ".env.dev", override=True)


TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Get configuration from environment
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
LLAMA_CLOUD_BASE_URL = os.getenv("LLAMA_CLOUD_BASE_URL")
LLAMA_CLOUD_PROJECT_ID = os.getenv("LLAMA_CLOUD_PROJECT_ID")

TestCase = namedtuple(
    "TestCase", ["name", "schema_path", "input_file", "expected_output"]
)


def get_test_cases():
    """Get all test cases from TEST_DIR.

    Returns:
        List[TestCase]: List of test cases
    """
    test_cases = []

    for data_type in os.listdir(TEST_DIR):
        data_type_dir = os.path.join(TEST_DIR, data_type)
        if not os.path.isdir(data_type_dir):
            continue

        schema_path = os.path.join(data_type_dir, "schema.json")
        if not os.path.exists(schema_path):
            continue

        input_files = []

        for file in os.listdir(data_type_dir):
            file_path = os.path.join(data_type_dir, file)
            if (
                not os.path.isfile(file_path)
                or file == "schema.json"
                or file.endswith(".test.json")
            ):
                continue

            input_files.append(file_path)

        for input_file in sorted(input_files):
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            expected_output = os.path.join(data_type_dir, f"{base_name}.test.json")

            if not os.path.exists(expected_output):
                continue

            test_name = f"{data_type}/{os.path.basename(input_file)}"
            test_cases.append(
                TestCase(
                    name=test_name,
                    schema_path=schema_path,
                    input_file=input_file,
                    expected_output=expected_output,
                )
            )

    return test_cases


@pytest.fixture
def extraction_agent(test_case: TestCase):
    """Fixture to create and cleanup extraction agent for each test."""
    extractor = LlamaExtract(
        api_key=LLAMA_CLOUD_API_KEY,
        base_url=LLAMA_CLOUD_BASE_URL,
        project_id=LLAMA_CLOUD_PROJECT_ID,
    )
    agent_name = test_case.name.split("/")[0]
    with open(test_case.schema_path, "r") as f:
        schema = json.load(f)

    try:
        existing_agent = extractor.get_agent(agent_name)
        if existing_agent:
            extractor.delete_agent(existing_agent.id)
    except ApiError as e:
        if e.status_code == 404:
            pass
        else:
            raise

    agent = extractor.create_agent(agent_name, schema)
    yield agent  # Provide the agent to the test
    try:
        extractor.delete_agent(agent.id)
    except Exception as e:
        print(f"Warning: Failed to delete agent {agent.id}: {str(e)}")


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
@pytest.mark.parametrize("test_case", get_test_cases(), ids=lambda x: x.name)
def test_extraction(test_case: TestCase, extraction_agent: ExtractionAgent) -> None:
    _, result = extraction_agent.extract(test_case.input_file)
    result = result.data
    with open(test_case.expected_output, "r") as f:
        expected = json.load(f)
    assert json_subset_match_score(expected, result) > 0.5, DeepDiff(
        expected, result, ignore_order=True
    )
