import torch
from transformers import pipeline
from settings import logger, DEVICE


def load_models():
    models_dict = dict()
    models_dict["small"] = pipeline(
"text2text-generation",  # Use text2text-generation because FLAN-T5 is an instruction model
    model="google/flan-t5-small",
    device=DEVICE
    )
    models_dict["base"] = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=DEVICE
        )
    return models_dict

models = load_models()


def summarize_text(text, max_token_length=200, min_token_length=30, model_name="base", repetition_penalty=1.2, no_repeat_ngram_size=3):
    logger.info("starting summarization\n")
    try:
        prompt = f"Summarize the following text:\n\n{text}"

        # Generate the summary
        # max_length: limits the summary size
        # truncation=True: Handles texts longer than the model's limit (512 tokens) for small model
        output = models[model_name](
            prompt,
            max_length=max_token_length,
            min_length=min_token_length,
            truncation=True,
            num_beams=4, # beam search better than greedy for our case. It forces the model to look ahead at multiple potential sentence structures before committing, which significantly reduces repetition and nonsensical phrasing.
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

        return output[0]['generated_text']

    except Exception as e:
        logger.info(f"Summarization error: {e}")
        return None



