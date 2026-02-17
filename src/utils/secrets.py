import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

from src.utils.logging_utils import get_logger

logger = get_logger("secrets")


@dataclass
class SecretsConfig:
    provider: str = "environment"
    vault_url: Optional[str] = None
    vault_token: Optional[str] = None
    aws_region: Optional[str] = None
    aws_secret_name: Optional[str] = None


class SecretsProvider(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def get_all(self) -> Dict[str, str]:
        pass


class EnvironmentSecretsProvider(SecretsProvider):
    def __init__(self, prefix: str = "KALSHI_"):
        self.prefix = prefix

    def get(self, key: str) -> Optional[str]:
        env_key = f"{self.prefix}{key.upper()}"
        return os.environ.get(env_key)

    def get_all(self) -> Dict[str, str]:
        secrets = {}
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                secrets[key] = value
        return secrets


class HashiCorpVaultProvider(SecretsProvider):
    def __init__(self, url: str, token: str, mount_point: str = "secret"):
        self.url = url.rstrip('/')
        self.token = token
        self.mount_point = mount_point
        self._cache: Dict[str, str] = {}
        self._cache_timeout = 300
        self._last_cache_time = 0

    async def _fetch_secret(self, path: str) -> Dict[str, Any]:
        import httpx
        url = f"{self.url}/v1/{self.mount_point}/{path}"
        headers = {"X-Vault-Token": self.token}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    def get(self, key: str) -> Optional[str]:
        import time
        if time.time() - self._last_cache_time > self._cache_timeout:
            self._cache.clear()
            self._last_cache_time = time.time()

        if key in self._cache:
            return self._cache[key]

        try:
            parts = key.split('.')
            if len(parts) == 2:
                secret_path = parts[0]
                secret_key = parts[1]
                data = self._fetch_secret(secret_path)
                value = data.get('data', {}).get(secret_key)
                if value:
                    self._cache[key] = value
                return value
        except Exception as e:
            logger.error(f"Failed to fetch secret {key}: {e}")

        return None

    def get_all(self) -> Dict[str, str]:
        return dict(self._cache)


class AWSSecretsManagerProvider(SecretsProvider):
    def __init__(self, region: str, secret_name: str):
        self.region = region
        self.secret_name = secret_name
        self._cache: Optional[Dict[str, str]] = None

    def get(self, key: str) -> Optional[str]:
        if self._cache is None:
            self._fetch_all()

        return self._cache.get(key) if self._cache else None

    def _fetch_all(self) -> None:
        try:
            import boto3
            client = boto3.client('secretsmanager', region_name=self.region)
            response = client.get_secret_value(SecretId=self.secret_name)
            secret = response['SecretString']
            self._cache = {
                k.strip(): v.strip()
                for k, v in (line.split('=', 1) for line in secret.split('\n') if '=' in line)
            }
        except Exception as e:
            logger.error(f"Failed to fetch AWS secret: {e}")
            self._cache = {}

    def get_all(self) -> Dict[str, str]:
        if self._cache is None:
            self._fetch_all()
        return dict(self._cache) if self._cache else {}


class SecretsManager:
    def __init__(self, config: Optional[SecretsConfig] = None):
        self.config = config or SecretsConfig()
        self._provider: Optional[SecretsProvider] = None

    def _get_provider(self) -> SecretsProvider:
        if self._provider:
            return self._provider

        if self.config.provider == "environment":
            self._provider = EnvironmentSecretsProvider()
        elif self.config.provider == "vault":
            self._provider = HashiCorpVaultProvider(
                url=self.config.vault_url or "",
                token=self.config.vault_token or ""
            )
        elif self.config.provider == "aws":
            self._provider = AWSSecretsManagerProvider(
                region=self.config.aws_region or "us-east-1",
                secret_name=self.config.aws_secret_name or ""
            )
        else:
            self._provider = EnvironmentSecretsProvider()

        return self._provider

    def get(self, key: str) -> Optional[str]:
        return self._get_provider().get(key)

    def get_required(self, key: str) -> str:
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required secret not found: {key}")
        return value

    def get_all(self) -> Dict[str, str]:
        return self._get_provider().get_all()

    def get_kalshi_credentials(self) -> Dict[str, str]:
        return {
            'api_key_id': self.get_required('API_KEY_ID'),
            'private_key': self.get_required('PRIVATE_KEY')
        }

    def reload(self) -> None:
        self._provider = None


def create_secrets_manager(config: Dict[str, Any]) -> SecretsManager:
    secrets_config = SecretsConfig(
        provider=config.get('provider', 'environment'),
        vault_url=config.get('vault_url'),
        vault_token=config.get('vault_token'),
        aws_region=config.get('aws_region'),
        aws_secret_name=config.get('aws_secret_name')
    )
    return SecretsManager(secrets_config)
