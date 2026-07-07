from __future__ import annotations

import pytest
from pydantic import ValidationError

from kuvox_ai.config import Settings
from kuvox_ai.infrastructure.object_storage_client import ObjectStorageClient


def test_settings_require_s3_credentials() -> None:
    with pytest.raises(ValidationError, match="KUVOX_S3_ACCESS_KEY and KUVOX_S3_SECRET_KEY"):
        Settings(s3_access_key=None, s3_secret_key=None)


def test_settings_reject_partial_s3_credentials() -> None:
    with pytest.raises(ValidationError, match="KUVOX_S3_SECRET_KEY is required"):
        Settings(s3_access_key="kuvox-ai-dev", s3_secret_key=None)

    with pytest.raises(ValidationError, match="KUVOX_S3_ACCESS_KEY is required"):
        Settings(s3_access_key=None, s3_secret_key="secret")


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
