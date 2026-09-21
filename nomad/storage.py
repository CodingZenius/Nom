"""Optional blob storage over any S3-compatible API (AWS, R2, MinIO, B2, Wasabi...).

Non-secret settings live in config["s3"]; credentials come from env:
  S3_ACCESS_KEY, S3_SECRET_KEY  (or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
"""
from __future__ import annotations
import mimetypes
import os
from pathlib import Path


class Blob:
    def __init__(self, cfg: dict | None):
        self.cfg = cfg or {}
        self._client = None
        self.enabled = bool(self.cfg.get("bucket"))
        self.prefix = (self.cfg.get("prefix") or "nomad/").lstrip("/")

    def _c(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                from .bootstrap import ensure_py
                ensure_py("boto3")
                import boto3
            self._client = boto3.client(
                "s3",
                endpoint_url=self.cfg.get("endpoint_url") or None,
                region_name=self.cfg.get("region") or "auto",
                aws_access_key_id=os.environ.get("S3_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("S3_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
        return self._client

    def _k(self, key: str) -> str:
        return self.prefix + key.lstrip("/")

    def put(self, key: str, src: str | Path | bytes) -> str:
        if not self.enabled:
            raise RuntimeError("S3 not configured")
        k = self._k(key)
        if isinstance(src, (bytes, bytearray)):
            self._c().put_object(Bucket=self.cfg["bucket"], Key=k, Body=bytes(src))
        else:
            ct = mimetypes.guess_type(str(src))[0] or "application/octet-stream"
            self._c().upload_file(str(src), self.cfg["bucket"], k, ExtraArgs={"ContentType": ct})
        return k

    def get(self, key: str, dest: str | Path) -> str:
        if not self.enabled:
            raise RuntimeError("S3 not configured")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._c().download_file(self.cfg["bucket"], self._k(key), str(dest))
        return str(dest)

    def list(self, prefix: str = "", limit: int = 50) -> list[str]:
        if not self.enabled:
            raise RuntimeError("S3 not configured")
        r = self._c().list_objects_v2(Bucket=self.cfg["bucket"], Prefix=self._k(prefix), MaxKeys=limit)
        return [o["Key"][len(self.prefix):] for o in r.get("Contents", [])]

    def url(self, key: str, expires: int = 3600) -> str:
        return self._c().generate_presigned_url(
            "get_object", Params={"Bucket": self.cfg["bucket"], "Key": self._k(key)}, ExpiresIn=expires)
