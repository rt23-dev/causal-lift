"""
IncrementalIQ playground — FastAPI backend.

A thin web wrapper around the `causal-lift` library, deployed as a demo /
lead-gen surface.  All analysis logic lives in `causal_lift`; this file only
handles HTTP, validation, and CSV parsing.

Endpoints
---------
POST /upload         — parse & validate CSVs, return structured JSON
POST /analyze        — run causal incrementality analysis
GET  /sample-data    — generate synthetic data (demo / testing)
GET  /health         — liveness check
"""

from __future__ import annotations

import asyncio
import os
from functools import partial
from io import StringIO
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import causal_lift as cl
from causal_lift import __version__ as causal_lift_version
from causal_lift.data import CSVValidationError, load_sales_csv, load_spend_csv

app = FastAPI(
    title="IncrementalIQ Playground",
    version="0.2.0",
    description=f"Web demo wrapping causal-lift v{causal_lift_version}",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB hard cap


# ── Request models ────────────────────────────────────────────────────────────

class SpendRow(BaseModel):
    date: str
    channel: str
    spend: float

    @field_validator("spend")
    @classmethod
    def spend_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("spend must be non-negative")
        return v


class SalesRow(BaseModel):
    date: str
    revenue: float
    orders: Optional[int] = None

    @field_validator("revenue")
    @classmethod
    def revenue_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("revenue must be non-negative")
        return v


class AnalyzeRequest(BaseModel):
    spend_data: list[SpendRow] = Field(..., min_length=1, max_length=50_000)
    sales_data: list[SalesRow] = Field(..., min_length=1, max_length=50_000)
    contribution_margin: float = Field(default=0.30, ge=0.05, le=0.95)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _read_csv_upload(file: UploadFile, name: str) -> str:
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            400,
            f"{name} exceeds 10 MB limit ({len(content) / 1_048_576:.1f} MB received).",
        )
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, f"{name} must be UTF-8 encoded text (CSV).")


# ── /upload ───────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_files(
    spend_file: UploadFile = File(...),
    sales_file: UploadFile = File(...),
) -> dict:
    spend_text, sales_text = await asyncio.gather(
        _read_csv_upload(spend_file, "spend_file"),
        _read_csv_upload(sales_file, "sales_file"),
    )

    try:
        spend_df = load_spend_csv(StringIO(spend_text))
        sales_df = load_sales_csv(StringIO(sales_text))
    except CSVValidationError as exc:
        raise HTTPException(400, str(exc))

    overlap_start = max(spend_df["date"].min(), sales_df["date"].min())
    overlap_end = min(spend_df["date"].max(), sales_df["date"].max())
    overlap_days = max(0, (overlap_end - overlap_start).days + 1)

    if overlap_days < 14:
        raise HTTPException(
            400,
            f"Only {overlap_days} days of overlapping dates between spend and sales files. "
            "At least 14 days of overlap are required.",
        )

    return {
        "spend_data": spend_df.assign(date=spend_df["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "sales_data": sales_df.assign(date=sales_df["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "summary": {
            "channels": sorted(spend_df["channel"].unique().tolist()),
            "date_range": {
                "start": overlap_start.strftime("%Y-%m-%d"),
                "end": overlap_end.strftime("%Y-%m-%d"),
            },
            "days": overlap_days,
        },
    }


# ── /analyze ──────────────────────────────────────────────────────────────────

def _run_blocking(spend_df: pd.DataFrame, sales_df: pd.DataFrame, margin: float):
    return cl.analyze(spend_df, sales_df, contribution_margin=margin)


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> dict:
    try:
        spend_df = pd.DataFrame([r.model_dump() for r in req.spend_data])
        sales_df = pd.DataFrame([r.model_dump() for r in req.sales_data])
        spend_df["date"] = pd.to_datetime(spend_df["date"])
        sales_df["date"] = pd.to_datetime(sales_df["date"])
    except Exception as exc:
        raise HTTPException(400, f"Could not reconstruct DataFrames: {exc}")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            partial(_run_blocking, spend_df, sales_df, req.contribution_margin),
        )
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")

    return result.to_dict()


# ── /sample-data ──────────────────────────────────────────────────────────────

@app.get("/sample-data")
def sample_data() -> dict:
    data = cl.generate_synthetic_data(n_days=90, seed=42)
    spend = data.spend_df.assign(date=data.spend_df["date"].dt.strftime("%Y-%m-%d"))
    sales = data.sales_df.assign(date=data.sales_df["date"].dt.strftime("%Y-%m-%d"))
    return {
        "spend_data": spend.to_dict(orient="records"),
        "sales_data": sales.to_dict(orient="records"),
        "summary": {
            "channels": sorted(spend["channel"].unique().tolist()),
            "date_range": {
                "start": str(sales["date"].iloc[0]),
                "end": str(sales["date"].iloc[-1]),
            },
            "days": len(sales),
        },
        "_ground_truth": data.ground_truth,
        "_sample_note": data.note,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.2.0", "causal_lift_version": causal_lift_version}
