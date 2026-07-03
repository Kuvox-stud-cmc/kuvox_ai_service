"""Async wrapper around an S3-compatible object store (SeaweedFS/AWS S3) via boto3.

boto3 is synchronous; calls are offloaded to a thread.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from kuvox_ai.config import Settings
from kuvox_ai.logging import get_logger

if TYPE_CHECKING:
    from types_boto3_s3.client import S3Client  # type: ignore[import-not-found]

logger = get_logger(__name__)


class ObjectStorageClient:
    """S3-compatible object storage client."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        region: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        create_bucket: bool,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._region = region
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._create_bucket = create_bucket
        self._client: Any | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> ObjectStorageClient:
        access_key = settings.s3_access_key
        secret_key = settings.s3_secret_key

        if not access_key or not secret_key:
            raise ValueError(
                "S3 configuration error: KUVOX_S3_ACCESS_KEY and KUVOX_S3_SECRET_KEY are required."
            )

        return cls(
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
            access_key=access_key,
            secret_key=secret_key,
            bucket=settings.s3_bucket,
            create_bucket=settings.s3_create_bucket,
        )

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def client(self) -> S3Client:
        if self._client is None:
            raise RuntimeError("ObjectStorageClient is not connected")
        return self._client

    async def connect(self) -> None:
        logger.info("s3.connecting", endpoint=self._endpoint_url, bucket=self._bucket)
        config_kwargs: dict[str, Any] = {"s3": {"addressing_style": "path"}}
        client_config = Config(signature_version="s3v4", **config_kwargs)

        self._client = await asyncio.to_thread(
            boto3.client,
            "s3",
            endpoint_url=self._endpoint_url,
            region_name=self._region,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            config=client_config,
        )
        if self._create_bucket:
            await self._ensure_bucket(self._bucket)
        logger.info("s3.connected")

    async def close(self) -> None:
        if self._client is None:
            return
        logger.info("s3.closing")
        # boto3 clients use a session under the hood; nothing to explicitly close.
        self._client = None
        logger.info("s3.closed")

    async def _ensure_bucket(self, bucket: str) -> None:
        def _check_and_create() -> None:
            assert self._client is not None
            try:
                self._client.head_bucket(Bucket=bucket)
            except ClientError as err:
                code = err.response.get("Error", {}).get("Code")
                if code in {"404", "NoSuchBucket", "NotFound"}:
                    logger.info("s3.creating_bucket", bucket=bucket)
                    self._client.create_bucket(Bucket=bucket)
                else:
                    raise

        await asyncio.to_thread(_check_and_create)

    async def _ensure_bucket_if_enabled(self, bucket: str) -> None:
        if self._create_bucket:
            await self._ensure_bucket(bucket)

    async def put_object(self, key: str, body: bytes, *, content_type: str | None = None) -> None:
        def _put() -> None:
            assert self._client is not None
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": key, "Body": body}
            if content_type:
                kwargs["ContentType"] = content_type
            self._client.put_object(**kwargs)

        await asyncio.to_thread(_put)

    async def get_object(self, key: str) -> bytes:
        def _get() -> bytes:
            assert self._client is not None
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            body = resp["Body"].read()
            assert isinstance(body, bytes)
            return body

        return await asyncio.to_thread(_get)

    async def download_file(self, bucket: str, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)

        def _download() -> None:
            assert self._client is not None
            response = self._client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                with destination.open("wb") as destination_file:
                    for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                        if chunk:
                            destination_file.write(chunk)
            finally:
                body.close()

        await asyncio.to_thread(_download)

    async def upload_file(
        self,
        source: Path,
        bucket: str,
        key: str,
        *,
        content_type: str,
    ) -> None:
        await self._ensure_bucket_if_enabled(bucket)

        def _upload() -> None:
            assert self._client is not None
            self._client.upload_file(
                str(source),
                bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )

        await asyncio.to_thread(_upload)

    async def delete_object(self, bucket: str, key: str) -> None:
        def _delete() -> None:
            assert self._client is not None
            self._client.delete_object(Bucket=bucket, Key=key)

        await asyncio.to_thread(_delete)

    async def object_exists(self, bucket: str, key: str) -> bool:
        def _exists() -> bool:
            assert self._client is not None
            try:
                self._client.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError as err:
                code = err.response.get("Error", {}).get("Code")
                if code in {"404", "NoSuchKey", "NotFound"}:
                    return False
                raise

        return await asyncio.to_thread(_exists)

    async def health_check(self) -> bool:
        try:
            if self._client is None:
                return False
            await asyncio.to_thread(self._client.head_bucket, Bucket=self._bucket)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("s3.health_check_failed", error=str(exc))
            return False
