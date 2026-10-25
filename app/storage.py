"""Protected local storage for source text and uploaded documents.

AES (Fernet) encryption is applied to source text, fact payloads and uploaded
documents.  The key comes from ``CAREFLOW_ENCRYPTION_KEY`` or, for local
development only, is generated once at ``data/.careflow.key``.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Optional

from cryptography.fernet import Fernet, InvalidToken

from .config import Settings


class StorageError(RuntimeError):
    pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_or_create_key(settings: Settings) -> bytes:
    if settings.encryption_key:
        key = settings.encryption_key.encode("ascii")
        try:
            Fernet(key)
        except Exception as exc:  # pragma: no cover - defensive
            raise StorageError("invalid CAREFLOW_ENCRYPTION_KEY") from exc
        return key
    key_path = settings.key_file
    if key_path.exists():
        key = key_path.read_bytes().strip()
        try:
            Fernet(key)
        except Exception as exc:
            raise StorageError("corrupt local encryption key file") from exc
        return key
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    key_path.write_bytes(key)
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    return key


class Cipher:
    def __init__(self, settings: Settings):
        self._fernet = Fernet(_load_or_create_key(settings))

    def encrypt_text(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt_text(self, value: str) -> str:
        try:
            return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise StorageError("stored data could not be decrypted") from exc

    def encrypt_json(self, value: Any) -> str:
        return self.encrypt_text(json.dumps(value, ensure_ascii=False, default=str))

    def decrypt_json(self, value: str) -> Any:
        return json.loads(self.decrypt_text(value))


class DocumentStore:
    """Encrypted blob store for uploaded originals."""

    def __init__(self, settings: Settings, cipher: Cipher):
        self.settings = settings
        self.cipher = cipher
        self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    def save(self, case_id: str, filename: str, content: bytes) -> dict:
        digest = hashlib.sha256(content).hexdigest()
        safe_name = "".join(
            ch for ch in Path(filename).name if ch.isalnum() or ch in "._-"
        )[:120] or "upload.bin"
        stored_name = f"{case_id}_{digest[:16]}_{safe_name}.enc"
        path = self.settings.uploads_dir / stored_name
        if not path.exists():
            path.write_bytes(self.cipher._fernet.encrypt(content))
        return {
            "stored_name": stored_name,
            "sha256": digest,
            "size": len(content),
            "original_name": safe_name,
        }

    def read(self, stored_name: str) -> bytes:
        if "/" in stored_name or "\\" in stored_name or ".." in stored_name:
            raise StorageError("invalid stored document reference")
        path = self.settings.uploads_dir / stored_name
        if not path.exists():
            raise StorageError("document not found")
        return self.cipher._fernet.decrypt(path.read_bytes())

    def delete(self, stored_name: str) -> bool:
        if "/" in stored_name or "\\" in stored_name or ".." in stored_name:
            return False
        path = self.settings.uploads_dir / stored_name
        if path.exists():
            path.unlink()
            return True
        return False


def random_id(prefix: str, nbytes: int = 8) -> str:
    return f"{prefix}_{secrets.token_hex(nbytes)}"


def safe_error(exc: Exception) -> str:
    """Return a user-safe error message without stack traces or secrets."""
    name = type(exc).__name__
    if isinstance(exc, (StorageError,)):
        return str(exc)
    return f"internal processing error ({name})"
