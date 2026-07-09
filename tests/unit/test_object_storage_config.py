from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from kuvox_ai.config import Settings
from kuvox_ai.infrastructure.object_storage_client import ObjectStorageClient
from kuvox_ai.infrastructure.qdrant_client import QdrantClient


def test_settings_require_s3_credentials() -> None:
    with pytest.raises(ValidationError, match="KUVOX_S3_ACCESS_KEY and KUVOX_S3_SECRET_KEY"):
        Settings(s3_access_key=None, s3_secret_key=None)


def test_settings_reject_partial_s3_credentials() -> None:
    with pytest.raises(ValidationError, match="KUVOX_S3_SECRET_KEY is required"):
        Settings(s3_access_key="kuvox-ai-dev", s3_secret_key=None)

    with pytest.raises(ValidationError, match="KUVOX_S3_ACCESS_KEY is required"):
        Settings(s3_access_key=None, s3_secret_key="secret")


@pytest.mark.skipif(os.name == "nt", reason="Windows drive paths are valid on Windows")
def test_settings_reject_windows_drive_paths_on_posix() -> None:
    with pytest.raises(ValidationError, match="KUVOX_MEDIA_WORK_DIR uses a Windows drive path"):
        Settings(
            s3_access_key="kuvox-ai-dev",
            s3_secret_key="secret",
            media_work_dir="D:/Kuvox/.codex-tmp/kuvox-media",
        )


def test_object_storage_client_from_settings_uses_configured_credentials_and_no_create() -> None:
    settings = Settings(
        s3_endpoint_url="http://localhost:8333",
        s3_region="us-east-1",
        s3_access_key="kuvox-ai-dev",
        s3_secret_key="secret",
        s3_bucket="kuvox-temp",
        s3_create_bucket=False,
    )

    client = ObjectStorageClient.from_settings(settings)

    assert client.bucket == "kuvox-temp"
    assert client._access_key == "kuvox-ai-dev"
    assert client._secret_key == "secret"
    assert client._create_bucket is False


def test_qdrant_blank_api_key_stays_http_for_local_dev() -> None:
    settings = Settings(
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_api_key="",
        qdrant_https=False,
        s3_access_key="kuvox-ai-dev",
        s3_secret_key="secret",
    )

    client = QdrantClient.from_settings(settings)

    assert settings.qdrant_api_key is None
    assert client._api_key is None
    assert client._https is False


def test_qdrant_https_can_be_enabled_for_hosted_endpoint() -> None:
    settings = Settings(
        qdrant_host="example.cloud.qdrant.io",
        qdrant_port=6333,
        qdrant_api_key=" hosted-key ",
        qdrant_https=True,
        s3_access_key="kuvox-ai-dev",
        s3_secret_key="secret",
    )

    client = QdrantClient.from_settings(settings)

    assert client._api_key == "hosted-key"
    assert client._https is True
