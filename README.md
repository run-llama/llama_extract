# LlamaExtract (EXPERIMENTAL)

LlamaExtract provides a simple API for extracting structured data from unstructured text.

> **⚠️ Warning**  
> 🚧 
>
> The released version of LlamaExtract on PyPi is no longer supported. This library is under active development and we will share an updated version 
> on PyPi very soon. In the meantime, please do not use this code on Github. If you are interested in being an early adopter, please contact us 
> at [support@llamaindex.ai](mailto:support@llamaindex.ai) or reach out on [Discord](https://discord.com/invite/eN6D2HQ4aX). 
>
> 🚧 

## Installation

```bash
# Warning: Contains breaking changes
pip install llama-extract==0.1.0  
```

## Usage

### Create a LlamaExtract client

```python
extractor = LlamaExtract(api_key="YOUR_API_KEY")
```

### Create an agent

```python
agent = extractor.create_agent(name="test_agent", data_schema={
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
    },
    "required": ["name", "email"],
})
```

You can also pass in a Pydantic model to define the data schema.

```python
from pydantic import BaseModel

class Resume(BaseModel):
    name: str
    email: str

agent = extractor.create_agent(name="test_agent", data_schema=Resume)
```

### Extract data from a file

```python
result = await agent.aextract("path/to/resume.pdf")
```

For a more detailed example and an illustration of usage patterns, please refer to the [demo notebook](examples/resume_screening.ipynb). 

