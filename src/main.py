import os
import logging
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from starlette.responses import JSONResponse

# Optional: use HuggingFace transformers if available.
# This import is safe because the package is listed in the project's dependencies.
try:
    from transformers import pipeline, Pipeline
except Exception as e:  # pragma: no cover
    raise ImportError(
        "The `transformers` package is required for the LLM-Powered Code Review Assistant. "
        f"Original error: {e}"
    ) from e

# --------------------------------------------------------------------------- #
# Configuration & Logging
# --------------------------------------------------------------------------- #

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pydantic models for request/response validation
# --------------------------------------------------------------------------- #


class ReviewRequest(BaseModel):
    """Schema for a code review request."""

    code: str = Field(..., description="The source code to be reviewed.")
    language: str = Field(
        default="python",
        description="Programming language of the source code (e.g., python, javascript).",
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Maximum number of tokens to generate for the review.",
    )

    @validator("code")
    def code_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Code cannot be empty or whitespace only.")
        return v


class ReviewResponse(BaseModel):
    """Schema for a code review response."""

    review: str = Field(..., description="Generated review comments for the submitted code.")
    model: str = Field(..., description="Identifier of the model used for generation.")
    usage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional usage statistics (e.g., token counts).",
    )


# --------------------------------------------------------------------------- #
# Model loading utilities
# --------------------------------------------------------------------------- #


def _load_model() -> Pipeline:
    """
    Load a text generation pipeline suitable for code review.

    The model name can be overridden via the ``LLM_MODEL_NAME`` environment
    variable. If the variable is not set, a lightweight open‑source model is used.
    """
    model_name = os.getenv("LLM_MODEL_NAME", "microsoft/phi-2")
    device = 0 if torch.cuda.is_available() else -1

    logger.info("Loading LLM model %s on device %s", model_name, "GPU" if device == 0 else "CPU")
    try:
        # ``pipeline`` will download the model if it is not cached locally.
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype=torch.float16 if device == 0 else torch.float32,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to load model %s", model_name)
        raise RuntimeError(f"Unable to load LLM model '{model_name}'.") from exc

    return generator


# Lazily load the model at startup; this avoids heavy import time for test collection.
model_pipeline: Pipeline = _load_model()


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #


def generate_review(code: str, language: str, max_tokens: int) -> str:
    """
    Generate a code review using the loaded LLM.

    Parameters
    ----------
    code: str
        The source code to be reviewed.
    language: str
        Programming language of the source (used for prompting).
    max_tokens: int
        Upper bound on the number of tokens the model may emit.

    Returns
    -------
    str
        A textual review of the supplied code.
    """
    # Simple prompt engineering – can be expanded later.
    prompt = (
        f"You are an expert software engineer reviewing {language} code. "
        f"Provide constructive feedback, point out bugs, style issues, and suggest improvements.\n\n"
        f"```{language}\n{code}\n```\n\nReview:"
    )

    logger.debug("Prompt for LLM: %s", prompt[:200])

    try:
        # ``generate`` returns a list of dictionaries; we only need the generated text.
        result = model_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            return_full_text=False,
        )
        review_text = result[0]["generated_text"].strip()
        logger.debug("Generated review (truncated): %s", review_text[:200])
        return review_text
    except Exception as exc:  # pragma: no cover
        logger.exception("LLM generation failed")
        raise RuntimeError("Failed to generate review.") from exc


# --------------------------------------------------------------------------- #
# FastAPI application definition
# --------------------------------------------------------------------------- #


app = FastAPI(
    title="LLM-Powered Code Review Assistant",
    description="A FastAPI service that uses a large language model to review source code.",
    version="1.0.0",
)

# Allow all origins for simplicity; adjust for production as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/review",
    response_model=ReviewResponse,
    summary="Generate a code review for a snippet of source code.",
    response_description="A textual review produced by the LLM.",
)
async def review_endpoint(request: ReviewRequest) -> JSONResponse:
    """
    Endpoint that receives source code and returns a review generated by the LLM.

    The function is asynchronous to keep the FastAPI event loop responsive,
    but the underlying model inference is CPU/GPU bound and therefore runs
    synchronously. For high‑throughput scenarios, consider moving inference
    to a background worker (e.g., Celery or a thread pool).
    """
    logger.info(
        "Received review request for language=%s, max_tokens=%d",
        request.language,
        request.max_tokens,
    )
    try:
        review_text = generate_review(request.code, request.language, request.max_tokens)
    except RuntimeError as err:
        raise HTTPException(status_code=500, detail=str(err)) from err

    response_payload = ReviewResponse(
        review=review_text,
        model=os.getenv("LLM_MODEL_NAME", "microsoft/phi-2"),
        usage={},  # Placeholder for future token usage stats.
    )
    return JSONResponse(content=response_payload.dict())


# --------------------------------------------------------------------------- #
# Development entry point
# --------------------------------------------------------------------------- #


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=LOG_LEVEL.lower(),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )