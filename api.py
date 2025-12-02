import logging
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import image_captioning
import summarization
import utils
from packages.db_utils import add_entry, db, search_as_dict
from settings import logger


def disable_uvicorn_logs() -> None:
    """Stop uvicorn from emitting its own logs so we only use our logger."""
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = False
        uvicorn_logger.setLevel(logging.CRITICAL)


app = FastAPI()
disable_uvicorn_logs()


class StoreInfoRequest(BaseModel):
    path: Optional[str] = None
    url: Optional[str] = None


class SearchInfoRequest(BaseModel):
    query: str


def _summarize_text(text: str) -> str:
    """Truncate and summarize text, raising if summarization fails."""
    truncated = utils.smart_truncate(text)
    summary = summarization.summarize_text(truncated)
    if not summary:
        logger.error("Summarization failed")
        raise HTTPException(status_code=500, detail="Failed to summarize text")
    return summary


@app.post("/storeInfo")
async def store_info(payload: StoreInfoRequest):
    try:
        logger.info("Received /storeInfo request")
        if payload.url:
            logger.info("Processing URL input")
            extracted_text = utils.extract_text_from_webpage(payload.url)
            if not extracted_text:
                logger.error("Failed to extract text from URL: %s", payload.url)
                raise HTTPException(status_code=400, detail="Could not extract text from URL")

            summary = _summarize_text(extracted_text)
            logger.info(f"Summarized text: {summary}")
            add_entry(db, summary, {"type": "url", "url": payload.url})
            logger.info("Stored summary for URL input")
            return JSONResponse(content={"message": "success"})

        if payload.path:
            logger.debug("Processing path input")
            _, ext = os.path.splitext(payload.path)
            ext = ext.lower()

            if ext in {".jpg", ".jpeg", ".png"}:
                extracted_text = utils.extract_text_from_image(payload.path)

                if extracted_text:
                    summary = _summarize_text(extracted_text)
                    logger.info(f"Summarized text: {summary}")
                    metadata = {"type": "image", "path": payload.path, "source": "ocr"}
                    add_entry(db, summary, metadata)
                    logger.info("Stored OCR-based summary for image input")
                    return JSONResponse(content={"message": "success"})

                caption = image_captioning.generate_caption(payload.path)
                if not caption:
                    logger.error("Failed to generate caption for image: %s", payload.path)
                    raise HTTPException(status_code=500, detail="Failed to generate caption")

                add_entry(db, caption, {"type": "image", "path": payload.path, "source": "caption"})
                logger.info("Stored caption for image input")
                return JSONResponse(content={"message": f"Success! Added {payload.path}"})

            elif ext in {".txt"}:
                with open(payload.path, "r") as f:
                    file_text = f.read()
                    summary = _summarize_text(file_text)
                    logger.info(f"Summarized text: {summary}")
                    add_entry(db, summary, {"type": "text", "path": payload.url})
                    return JSONResponse(content={"message": f"Success! Added {payload.path}"})

            else:
                logger.error("Unsupported file type for path: %s", payload.path)
                raise HTTPException(status_code=400, detail="Unsupported file type. Only jpg, jpeg, png allowed.")

        logger.error("Neither URL nor path provided in /storeInfo payload")
        raise HTTPException(status_code=400, detail="Either 'url' or 'path' must be provided")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.post("/searchInfo")
async def search_info(payload: SearchInfoRequest):
    logger.info("Received /searchInfo request")
    results = search_as_dict(db, payload.query)
    logger.info(f"Search returned {len(results)} results")
    return JSONResponse(content={"results": results})


if __name__ == '__main__':
    disable_uvicorn_logs()
    logger.info("Starting API server")
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level="critical", access_log=False)
