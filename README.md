# LlamaExtract (Experimental)

**NOTE:** This API is deprecated in favor of a more mature LlamaExtract offering that will be released soon. We will have an update for you shortly.


LlamaExtract is an API created by LlamaIndex to efficiently infer schema and extract data from unstructured files.

LlamaExtract directly integrates with [LlamaIndex](https://github.com/run-llama/llama_index).

Note: LlamaExtract is currently experimental and may change in the future.

Read below for some quickstart information, or see the [full documentation](https://docs.cloud.llamaindex.ai/).

## Getting Started

First, login and get an api-key from [**https://cloud.llamaindex.ai ↗**](https://cloud.llamaindex.ai).

Install the package:

`pip install llama-extract`

Now you can easily infer schemas and extract data from your files:

```python
import nest_asyncio

nest_asyncio.apply()

from llama_extract import LlamaExtract

extractor = LlamaExtract(
    api_key="llx-...",  # can also be set in your env as LLAMA_CLOUD_API_KEY
    num_workers=4,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
)

# Infer schema
schema = extractor.infer_schema(
    "my_schema", ["./my_file1.pdf", "./my_file2.pdf"]
)

# Extract data
results = extractor.extract(schema.id, ["./my_file1.pdf", "./my_file2.pdf"])
```

## Examples

Several end-to-end examples can be found in the examples folder

- [Getting Started](examples/demo_basic.ipynb)

## Documentation

[https://docs.cloud.llamaindex.ai/](https://docs.cloud.llamaindex.ai/)
