from utils import extract_text_from_webpage, smart_truncate
from summarization import summarize_text

url_text = extract_text_from_webpage("https://www.khoury.northeastern.edu/people/jin-yu/")
print(f"\nfull text-{url_text}\n")

truncated_text = smart_truncate(url_text)
print(f'truncated- {truncated_text}')

if url_text is not None:
    summarized_text = summarize_text(truncated_text)
    print(f"\nSummary-\n- {summarized_text}")
