import asyncio
import os
import time

import pydantic.v1 as pydantic_v1

from io import BufferedIOBase, BufferedReader, BytesIO
from json.decoder import JSONDecodeError
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Tuple, Type, Union
import urllib.parse


from llama_cloud import (
    ExtractionJob,
    ExtractionResult,
    ExtractionSchema,
    File,
    HttpValidationError,
    StatusEnum,
    UnprocessableEntityError,
)
from llama_cloud.client import AsyncLlamaCloud
from llama_cloud.core import ApiError, jsonable_encoder
from llama_extract.utils import nest_asyncio_err, nest_asyncio_msg
from llama_index.core.schema import BaseComponent
from llama_index.core.async_utils import asyncio_run, run_jobs
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_BASE_URL

# can put in a path to the file or the file bytes itself
FileInput = Union[str, Path, bytes, BufferedIOBase]

SchemaInput = Union[dict, Type[BaseModel]]
ExtractionOutput = Union[ExtractionResult, Tuple[ExtractionResult, Optional[BaseModel]]]
ExtractionOutputList = Union[
    List[ExtractionResult], Tuple[List[ExtractionResult], List[Optional[BaseModel]]]
]


class LlamaExtract(BaseComponent):
    """A extractor for unstructured data."""

    api_key: str = Field(description="The API key for the LlamaExtract API.")
    base_url: str = Field(
        description="The base URL of the LlamaExtract API.",
    )
    check_interval: int = Field(
        default=1,
        description="The interval in seconds to check if the extraction is done.",
    )
    max_timeout: int = Field(
        default=2000,
        description="The maximum timeout in seconds to wait for the extraction to finish.",
    )
    num_workers: int = Field(
        default=4,
        gt=0,
        lt=10,
        description="The number of workers to use sending API requests for extraction.",
    )
    show_progress: bool = Field(
        default=True, description="Show progress when extracting multiple files."
    )
    verbose: bool = Field(
        default=False, description="Show verbose output when extracting files."
    )
    _async_client: AsyncLlamaCloud = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        check_interval: int = 1,
        max_timeout: int = 2000,
        num_workers: int = 4,
        show_progress: bool = True,
        verbose: bool = False,
    ):
        if not api_key:
            api_key = os.getenv("LLAMA_CLOUD_API_KEY", None)
            if api_key is None:
                raise ValueError("The API key is required.")

        if not base_url:
            base_url = os.getenv("LLAMA_CLOUD_BASE_URL", None) or DEFAULT_BASE_URL

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            check_interval=check_interval,
            max_timeout=max_timeout,
            num_workers=num_workers,
            show_progress=show_progress,
            verbose=verbose,
        )
        self._async_client = AsyncLlamaCloud(
            token=self.api_key, base_url=self.base_url, timeout=None
        )

    async def _upload_file(
        self, file_input: FileInput, project_id: Optional[str] = None
    ) -> File:
        if isinstance(file_input, BufferedIOBase):
            upload_file = file_input
        elif isinstance(file_input, bytes):
            upload_file = BytesIO(file_input)
        elif isinstance(file_input, (str, Path)):
            upload_file = open(file_input, "rb")
        else:
            raise ValueError(
                "file_input must be either a file path string, file bytes, or buffer object"
            )

        try:
            uploaded_file = await self._async_client.files.upload_file(
                project_id=project_id, upload_file=upload_file
            )

            return uploaded_file
        finally:
            if isinstance(upload_file, BufferedReader):
                upload_file.close()

    async def _wait_for_job_result(
        self, job_id: str, verbose: bool = False
    ) -> ExtractionResult:
        start = time.time()
        tries = 0
        while True:
            await asyncio.sleep(self.check_interval)
            tries += 1
            extraction_job = await self.aget_job(job_id)

            if extraction_job.status == StatusEnum.SUCCESS:
                result = await self.aget_job_result(job_id)
                return result
            elif extraction_job.status == StatusEnum.PENDING:
                end = time.time()
                if end - start > self.max_timeout:
                    raise Exception(f"Timeout while extracting the file: {job_id}")
                if verbose and tries % 10 == 0:
                    print(".", end="", flush=True)

                await asyncio.sleep(self.check_interval)

                continue
            else:
                raise Exception(
                    f"Failed to extract the file: {job_id}, status: {extraction_job.status}"
                )

    async def _extract(
        self,
        schema_id: str,
        file_input: FileInput,
        project_id: Optional[str] = None,
        verbose: bool = False,
    ) -> ExtractionResult:
        try:
            file = await self._upload_file(file_input, project_id)

            extraction_job = await self._async_client.extraction.run_job(
                schema_id=schema_id, file_id=file.id
            )

            if verbose:
                print("Started extracting the file under job_id %s" % extraction_job.id)

            result = await self._wait_for_job_result(extraction_job.id, verbose=verbose)

            return result
        except Exception as e:
            file_repr = (
                str(file_input)
                if isinstance(file_input, (str, Path))
                else "<bytes/buffer>"
            )
            print(f"Error while extracting the file '{file_repr}':", e)

            raise e

    async def ainfer_schema(
        self,
        name: str,
        seed_files: List[FileInput],
        schema_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> ExtractionSchema:
        """Infer schema for a given set of seed files."""
        file_ids: List[str] = []

        for seed_file in seed_files:
            file = await self._upload_file(seed_file, project_id)

            file_ids.append(file.id)

        body = {"name": name, "file_ids": file_ids}

        if schema_id is not None:
            body["schema_id"] = schema_id

        if project_id is not None:
            body["project_id"] = project_id

        # Using httpx_client directly to bypass timeout from LlamaCloud client
        _response = await self._async_client._client_wrapper.httpx_client.post(
            urllib.parse.urljoin(
                f"{self._async_client._client_wrapper.get_base_url()}/",
                "api/v1/extraction/schemas/infer",
            ),
            json=jsonable_encoder(body),
            headers=self._async_client._client_wrapper.get_headers(),
        )

        if 200 <= _response.status_code < 300:
            return pydantic_v1.parse_obj_as(ExtractionSchema, _response.json())
        if _response.status_code == 422:
            raise UnprocessableEntityError(
                pydantic_v1.parse_obj_as(HttpValidationError, _response.json())
            )
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)

        raise ApiError(status_code=_response.status_code, body=_response_json)

    def infer_schema(
        self,
        name: str,
        seed_files: List[FileInput],
        schema_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> ExtractionSchema:
        """Infer schema for a given set of seed files."""
        try:
            return asyncio_run(
                self.ainfer_schema(name, seed_files, schema_id, project_id)
            )
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def acreate_schema(
        self,
        name: str,
        data_schema: SchemaInput,
        project_id: Optional[str] = None,
    ) -> ExtractionSchema:
        """Create a schema."""

        if isinstance(data_schema, dict):
            json_schema = data_schema
        elif issubclass(data_schema, BaseModel):
            json_schema = data_schema.model_json_schema()
        else:
            raise ValueError(
                "data_schema must be either a dictionary or a Pydantic model"
            )

        response = await self._async_client.extraction.create_schema(
            name=name, data_schema=json_schema, project_id=project_id
        )
        return response

    def create_schema(
        self,
        name: str,
        data_schema: SchemaInput,
        project_id: Optional[str] = None,
    ) -> ExtractionSchema:
        """Create a schema."""
        try:
            return asyncio_run(self.acreate_schema(name, data_schema, project_id))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def alist_schemas(
        self, project_id: Optional[str] = None
    ) -> List[ExtractionSchema]:
        """List all schemas."""
        extraction_schemas = await self._async_client.extraction.list_schemas(
            project_id=project_id
        )
        return extraction_schemas

    def list_schemas(self, project_id: Optional[str] = None) -> List[ExtractionSchema]:
        """List all schemas."""
        try:
            return asyncio_run(self.alist_schemas(project_id=project_id))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def aget_schema(self, schema_id: str) -> ExtractionSchema:
        """Get a schema."""
        response = await self._async_client.extraction.get_schema(schema_id=schema_id)
        return response

    def get_schema(self, schema_id: str) -> ExtractionSchema:
        """Get a schema."""
        try:
            return asyncio_run(self.aget_schema(schema_id))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def aupdate_schema(
        self, schema_id: str, data_schema: SchemaInput
    ) -> ExtractionSchema:
        """Update a schema."""
        if isinstance(data_schema, dict):
            json_schema = data_schema
        elif issubclass(data_schema, BaseModel):
            json_schema = data_schema.model_json_schema()
        else:
            raise ValueError(
                "data_schema must be either a dictionary or a Pydantic model"
            )

        response = await self._async_client.extraction.update_schema(
            schema_id=schema_id, data_schema=json_schema
        )
        return response

    def update_schema(
        self, schema_id: str, data_schema: SchemaInput
    ) -> ExtractionSchema:
        """Update a schema."""
        try:
            return asyncio_run(self.aupdate_schema(schema_id, data_schema))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def alist_jobs(self, schema_id: str) -> List[ExtractionJob]:
        """List all jobs."""
        extraction_jobs = await self._async_client.extraction.list_jobs(
            schema_id=schema_id
        )
        return extraction_jobs

    def list_jobs(self, schema_id: str) -> List[ExtractionJob]:
        """List all jobs."""
        try:
            return asyncio_run(self.alist_jobs(schema_id))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def aget_job(self, job_id: str) -> ExtractionJob:
        """Get a job."""
        response = await self._async_client.extraction.get_job(job_id=job_id)
        return response

    def get_job(self, job_id: str) -> ExtractionJob:
        """Get a job."""
        try:
            return asyncio_run(self.aget_job(job_id))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def aget_job_result(
        self, job_id: str, response_model: Optional[Type[BaseModel]] = None
    ) -> ExtractionOutput:
        """Get a job result.

        Args:
            job_id: The job id.
            response_model: The response model to validate the response with.

        Returns:
            If response_model is None, directly returns the raw extraction result.
            If response_model is provided, returns a tuple of the raw extraction result and the validated model.
        """
        response = await self._async_client.extraction.get_job_result(job_id=job_id)
        if response_model is None:
            return response

        # validate response with the response model
        try:
            model = response_model.model_validate(response)
        except ValidationError:
            if self.verbose:
                print(
                    f"Failed to validate the response with the model {response_model}, returning the response as is."
                )
            model = None
        return response, model

    def get_job_result(
        self, job_id: str, response_model: Optional[Type[BaseModel]] = None
    ) -> ExtractionOutput:
        """Get a job result.

        Args:
            job_id: The job id.
            response_model: The response model to validate the response with.

        Returns:
            If response_model is None, directly returns the raw extraction result.
            If response_model is provided, returns a tuple of the raw extraction result and the validated model.
        """
        try:
            return asyncio_run(
                self.aget_job_result(job_id, response_model=response_model)
            )
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def aextract(
        self,
        schema_id: str,
        files: List[FileInput],
        project_id: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> ExtractionOutputList:
        """Extract data from a file using a schema.

        Args:
            schema_id: The schema id.
            files: The list of files to extract.
            project_id: The project id.
            response_model: The response model to validate the response with.

        Returns:
            If response_model is None, directly returns the list of raw extraction results.
            If response_model is provided, returns a tuple of the list of raw extraction results and the list of validated models.
        """

        jobs = [
            self._extract(
                schema_id,
                file,
                project_id,
                verbose=self.verbose and not self.show_progress,
            )
            for file in files
        ]

        try:
            results = await run_jobs(
                jobs,
                workers=self.num_workers,
                desc="Extracting files",
                show_progress=self.show_progress,
            )

            if response_model is None:
                return results

            try:
                models = [
                    response_model.model_validate(result.data) for result in results
                ]
            except ValidationError:
                if self.verbose:
                    print(
                        f"Failed to validate the response with the model {response_model}, returning the response as is."
                    )
                models = [None] * len(results)

            return results, models

        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    def extract(
        self,
        schema_id: str,
        file_input: List[FileInput],
        project_id: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> ExtractionOutputList:
        """Extract data from a file using a schema.

        Args:
            schema_id: The schema id.
            files: The list of files to extract.
            project_id: The project id.
            response_model: the response model to validate the response with.

        Returns:
            If response_model is None, directly returns the list of raw extraction results.
            If response_model is provided, returns a tuple of the list of raw extraction results and the list of validated
        """
        try:
            return asyncio_run(
                self.aextract(
                    schema_id, file_input, project_id, response_model=response_model
                )
            )
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e
