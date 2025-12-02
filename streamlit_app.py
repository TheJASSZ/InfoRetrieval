import json
from typing import Optional

import requests
import streamlit as st


API_BASE_URL = "http://localhost:8000"


def store_info(url: Optional[str], path: Optional[str]):
    payload = {"url": url or None, "path": path or None}
    resp = requests.post(f"{API_BASE_URL}/storeInfo", json=payload, timeout=60)
    return resp


def search_info(query: str):
    resp = requests.post(f"{API_BASE_URL}/searchInfo", json={"query": query}, timeout=30)
    return resp


st.set_page_config(page_title="Info Store", layout="centered")

st.title("Info Store")
st.caption("Send URLs or file paths to /storeInfo and search with /searchInfo.")

st.header("Store Info")
col1, col2 = st.columns(2)
with col1:
    url_input = st.text_input("Webpage URL (optional)", placeholder="https://example.com/article")
with col2:
    path_input = st.text_input("File path (optional)", placeholder="C:/path/to/image_or_text")

if st.button("Send to /storeInfo"):
    if not url_input and not path_input:
        st.error("Provide at least a URL or a file path.")
    else:
        with st.spinner("Sending..."):
            try:
                resp = store_info(url_input.strip() or None, path_input.strip() or None)
                if resp.ok:
                    st.success(f"Stored successfully: {resp.json()}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")

st.header("Search Info")
query_input = st.text_input("Query", placeholder="search keywords")
if st.button("Search"):
    if not query_input.strip():
        st.error("Enter a query.")
    else:
        with st.spinner("Searching..."):
            try:
                resp = search_info(query_input.strip())
                if resp.ok:
                    results = resp.json().get("results", [])
                    if not results:
                        st.info("No results.")
                    else:
                        st.subheader("Results")
                        for idx, item in enumerate(results, start=1):
                            with st.expander(f"Result {idx} (distance: {item.get('distance')})", expanded=False):
                                st.write(f"Summary: {item.get('summary')}")
                                st.json(item.get("metadata", {}))
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
