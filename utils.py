import trafilatura
import ssl

# Create an unverified SSL context to bypass certificate verification
ssl._create_default_https_context = ssl._create_unverified_context
import easyocr
import os
from settings import logger

reader = easyocr.Reader(['en'])


def extract_text_from_webpage(url):
    try:
        logger.info(f"Extracting text from webpage {url}")
        url_data = trafilatura.fetch_url(url)

        if url_data is None:
            logger.error(f"Error: Could not retrieve content from {url}")
            return None

        text_content = trafilatura.extract(
            url_data,
            favor_precision=False,
            include_comments=False,
            include_tables=False,
            no_fallback=False  # Allow internal fallbacks for better coverage
        )

        return text_content

    except Exception as e:
        logger.exception(f"Error while processing {url}: {e}")
        return None


def extract_text_from_image(image_path):
    logger.info(f"Extracting text from image {image_path}")
    if not os.path.exists(image_path):
        logger.info(f"Error: The file {image_path} does not exist.")
        return None

    try:
        results = reader.readtext(image_path, detail=0)  # detail 0 means extracts without bbox
        extracted_text = " ".join(results)
        logger.info(f"Extracted text from image {image_path}: {extracted_text}")
        return extracted_text

    except Exception as e:
        logger.exception(f"OCR Error: {e}")
        return None


def smart_truncate(text, max_chars=500):
    """
    Truncates text to fit within a character limit, ensuring it ends
    on a complete sentence to avoid abrupt cutoffs.
    """
    logger.info(f"Smart truncating text to {max_chars} characters long")

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    last_period = truncated.rfind('.')  # filter and find positions of end of sentence punctuations
    last_exclam = truncated.rfind('!')
    last_quest = truncated.rfind('?')

    split_point = max(last_period, last_exclam, last_quest)  # find the split point that is the maximum point that can be used to split.

    # If a valid sentence ending is found (and not -1), cut there
    if split_point != -1:
        return truncated[:split_point + 1]  # +1 to include the punctuation mark itself

    return truncated