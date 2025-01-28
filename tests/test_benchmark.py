import os
import pytest
from pathlib import Path

from llama_extract import LlamaExtract, ExtractionAgent
from dotenv import load_dotenv
from time import perf_counter
from collections import namedtuple
import json
import uuid
from llama_cloud.core.api_error import ApiError
from llama_cloud.types import (
    ExtractConfig,
    ExtractMode,
    LlamaParseParameters,
    LlamaExtractSettings,
)

load_dotenv(Path(__file__).parent.parent / ".env.dev", override=True)


TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Get configuration from environment
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
LLAMA_CLOUD_BASE_URL = os.getenv("LLAMA_CLOUD_BASE_URL")
LLAMA_CLOUD_PROJECT_ID = os.getenv("LLAMA_CLOUD_PROJECT_ID")

TestCase = namedtuple(
    "TestCase", ["name", "schema_path", "config", "input_file", "expected_output"]
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

        settings = [
            ExtractConfig(extraction_mode=ExtractMode.FAST),
            ExtractConfig(extraction_mode=ExtractMode.ACCURATE),
        ]

        for input_file in sorted(input_files):
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            expected_output = os.path.join(data_type_dir, f"{base_name}.test.json")

            if not os.path.exists(expected_output):
                continue

            test_name = f"{data_type}/{os.path.basename(input_file)}"
            for setting in settings:
                test_cases.append(
                    TestCase(
                        name=test_name,
                        schema_path=schema_path,
                        input_file=input_file,
                        config=setting,
                        expected_output=expected_output,
                    )
                )

    return test_cases


@pytest.fixture(scope="session")
def extractor():
    """Create a single LlamaExtract instance for all tests."""
    extract = LlamaExtract(
        api_key=LLAMA_CLOUD_API_KEY,
        base_url=LLAMA_CLOUD_BASE_URL,
        project_id=LLAMA_CLOUD_PROJECT_ID,
        verbose=True,
    )
    yield extract
    # Cleanup thread pool at end of session
    extract._thread_pool.shutdown()


@pytest.fixture
def extraction_agent(test_case: TestCase, extractor: LlamaExtract):
    """Fixture to create and cleanup extraction agent for each test."""
    # Create unique name with random UUID (important for CI to avoid conflicts)
    unique_id = uuid.uuid4().hex[:8]
    agent_name = f"{test_case.name}_{unique_id}"

    with open(test_case.schema_path, "r") as f:
        schema = json.load(f)

    # Clean up any existing agents with this name
    try:
        agents = extractor.list_agents()
        for agent in agents:
            if agent.name == agent_name:
                extractor.delete_agent(agent.id)
    except Exception as e:
        print(f"Warning: Failed to cleanup existing agent: {str(e)}")

    # Create new agent
    agent = extractor.create_agent(agent_name, schema, config=test_case.config)
    yield agent


@pytest.mark.skipif(
    "CI" in os.environ,
    reason="CI environment is not suitable for benchmarking",
)
@pytest.mark.parametrize("test_case", get_test_cases(), ids=lambda x: x.name)
@pytest.mark.asyncio(loop_scope="session")
async def test_extraction(
    test_case: TestCase, extraction_agent: ExtractionAgent
) -> None:
    start = perf_counter()
    result = await extraction_agent._queue_extraction_test(
        test_case.input_file,
        extract_settings=LlamaExtractSettings(
            llama_parse_params=LlamaParseParameters(
                invalidate_cache=True,
                do_not_cache=True,
            )
        ),
    )
    end = perf_counter()
    print(f"Time taken: {end - start} seconds")
    print(result)
