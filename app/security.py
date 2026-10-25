"""Authentication and case-level authorization.

The development mode uses trusted headers (``X-User-Id``/``X-Role``) and is
suitable only for local synthetic demos.  API-key mode is provided for simple
deployments; production should replace this with the organization's OIDC/SSO.
Case-level checks prevent cross-patient access even for authenticated roles.
"""
from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, Request

from .config import Settings
from .db import Database
from .schemas import CaseRecord

CLINICAL_ROLES = frozenset(
    {
        "clinician",
        "doctor",
        "nurse",
        "physician",
        "physician_assistant",
        "nurse_practitioner",
    }
)
VALID_ROLES = {"patient", "intake", "admin"} | set(CLINICAL_ROLES)


@dataclass
class Principal:
    user_id: str
    role: str
    patient_id: Optional[str] = None

    @property
    def is_clinician(self) -> bool:
        return self.role in CLINICAL_ROLES or self.role == "admin"


def authenticate(
    settings: Settings,
    x_user_id: Optional[str] = None,
    x_role: Optional[str] = None,
    x_patient_id: Optional[str] = None,
    x_api_key: Optional[str] = None,
) -> Principal:
    if settings.auth_mode == "api_key":
        if not x_api_key:
            raise HTTPException(status_code=401, detail="authentication required")
        matched = None
        for key, info in settings.api_keys.items():
            if hmac.compare_digest(key, x_api_key):
                matched = info
                break
        if not matched:
            raise HTTPException(status_code=401, detail="authentication required")
        role = matched["role"]
        user_id = matched["user_id"]
        patient_id = x_patient_id if role == "patient" else None
        if role == "patient" and not patient_id:
            raise HTTPException(status_code=401, detail="patient identity required")
        return Principal(user_id=user_id, role=role, patient_id=patient_id)

    if not x_user_id or not x_role:
        raise HTTPException(status_code=401, detail="authentication required")
    role = x_role.strip().lower()
    if role not in VALID_ROLES:
        raise HTTPException(status_code=403, detail="unsupported role")
    patient_id = x_patient_id
    if role == "patient":
        patient_id = patient_id or x_user_id
    return Principal(user_id=x_user_id.strip(), role=role, patient_id=patient_id)


def current_principal(
    request: Request,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
    x_patient_id: Optional[str] = Header(default=None, alias="X-Patient-Id"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Principal:
    settings: Settings = request.app.state.settings
    return authenticate(settings, x_user_id, x_role, x_patient_id, x_api_key)


def ensure_roles(principal: Principal, roles: set[str]) -> None:
    if principal.role in roles:
        return
    # Every clinical endpoint already lists "clinician". Treat named clinical
    # roles (doctor, nurse, etc.) as equivalents without duplicating the list
    # at every route.
    if "clinician" in roles and principal.role in CLINICAL_ROLES:
        return
    raise HTTPException(status_code=403, detail="insufficient role for this action")


def ensure_case_access(
    db: Database,
    principal: Principal,
    case: CaseRecord,
    *,
    write: bool = False,
) -> None:
    if principal.role == "patient":
        if case.patient_id != principal.patient_id:
            db.audit(
                principal.user_id,
                principal.role,
                "unauthorized_case_access_attempt",
                case.case_id,
                {"requested_patient_scope": "self"},
            )
            raise HTTPException(status_code=403, detail="access denied for this case")
    if write and principal.role == "patient":
        # Patients may ingest their own material and consent, but clinical
        # review actions are restricted separately at each endpoint.
        return
