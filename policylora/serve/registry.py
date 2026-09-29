"""Pinned adapter registry. Unknown tenants fail closed."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from policylora.paths import repo_root


class UnknownTenant(Exception):
    pass


class AdapterPin(BaseModel):
    tenant_id: str
    adapter_name: str
    version: str
    module_name: str
    path: str

    @property
    def version_label(self) -> str:
        return f"{self.adapter_name}@{self.version}"


class AdapterRegistry:
    def __init__(self, path: Path | None = None):
        self.path = path or repo_root() / "policylora/serve/adapters/registry.json"
        self.document = json.loads(self.path.read_text())

    def resolve(self, tenant_id: str) -> AdapterPin:
        tenants = self.document["tenants"]
        if tenant_id not in tenants:
            raise UnknownTenant(tenant_id)
        adapter_name = tenants[tenant_id]
        adapter = self.document["adapters"][adapter_name]
        version = adapter["active"]
        if version not in adapter["versions"]:
            raise UnknownTenant(tenant_id)
        return AdapterPin(
            tenant_id=tenant_id,
            adapter_name=adapter_name,
            version=version,
            module_name=f"{adapter_name if adapter_name != 'shared' else 'base'}-{version}",
            path=adapter["versions"][version],
        )
