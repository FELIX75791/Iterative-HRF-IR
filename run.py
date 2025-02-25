#!/usr/bin/env python3

"""
Advanced Relevance Feedback with TF-IDF-based Query Expansion

Usage:
    python run.py <google_api_key> <google_engine_id> <target_precision> "<query>"

Example:
    python run.py ABC123XYZ abc456def 0.9 "milky way"

Author: Ziyue Jin, Ken Deng
Date:   2025-02-23

Notes:
 - Demonstrates a basic TF-IDF approach to pick expansion terms.
 - Shows a small-scale permutation strategy for query reordering if query <= 5 terms.
 - Ignores non-HTML results in both indexing and precision calculation.
 - Exits if fewer than 10 results are returned in the first iteration, as per instructions.
 - Only uses 'title' and 'snippet' for text analysis. You can optionally fetch full HTML pages.

"""

import sys
import pprint
import itertools
import math
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup

# Trying to use nltk library for tokenize and specify stopwords
# other than manually listing stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ensure nltk data are correctly downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# # A simple English stopword list.
# STOPWORDS = {
#     "the", "and", "of", "to", "a", "in", "is", "it", "you", "that", "he", "was",
#     "for", "on", "are", "as", "with", "his", "they", "i", "be", "at", "one",
#     "have", "this", "from", "or", "had", "by", "not", "word", "but", "what",
#     "some", "we", "can", "out", "other", "were", "all", "there", "when", "up",
#     "use", "your", "how", "said", "an", "each", "she", "which", "their", 
#     "will", "also", "do"
# }

# Simple set of file extensions that we consider "non-HTML."
NON_HTML_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}

def parse_args():
    if len(sys.argv) != 5:
        print("Usage: python advanced_proj1.py <google_api_key> <google_engine_id> <target_precision> \"<query>\"")
        sys.exit(1)

    google_api_key = sys.argv[1]
    google_engine_id = sys.argv[2]
    try:
        target_precision = float(sys.argv[3])
    except ValueError:
        print("Error: target_precision must be a float (e.g., 0.9).")
        sys.exit(1)
    initial_query = sys.argv[4].strip("\"")

    return google_api_key, google_engine_id, target_precision, initial_query


def build_service(google_api_key):
    """
    Build the Google API service object for Custom Search.
    """
    return build("customsearch", "v1", developerKey=google_api_key)


def is_likely_html(url):
    """
    A crude way to detect if a URL is likely an HTML page by checking its file extension.
    """
    url = url.lower()
    for ext in NON_HTML_EXTENSIONS:
        if url.endswith(ext):
            return False
    return True

def fetch_full_text(url):
    """
    Fetch the full text from a URL by downloading and parsing the HTML.
    Returns the extracted text or an empty string on failure.
    """
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return ""
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        # Get text and join paragraphs
        texts = soup.stripped_strings
        full_text = " ".join(texts)
        return full_text
    except Exception as e:
        print(f"Error fetching full text from {url}: {e}")
        return ""


def search_query(service, engine_id, query, num_results=10):
    """
    Execute the query via the Custom Search API, return up to num_results = 10 documents.
    Each doc is a tuple: (title, link, snippet, is_html).
    """
    try:
        res = service.cse().list(
            q=query,
            cx=engine_id,
            num=num_results
        ).execute()

        items = res.get("items", [])
        results = []
        for item in items:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            # Check if the doc is HTML or not
            html_flag = is_likely_html(link)
            results.append((title, link, snippet, html_flag))

        return results
    except Exception as e:
        print(f"Error calling Google API: {e}")
        return []


def display_results(results):
    """
    Print the top-10 results in a user-friendly manner.
    """
    print("\n==================== Retrieved Results ====================")
    for i, (title, link, snippet, is_html) in enumerate(results, start=1):
        print(f"Result {i}:")
        print(f"  Title:   {title}")
        print(f"  URL:     {link}")
        print(f"  Summary: {snippet}")
        print(f"  (HTML?:  {is_html})\n")
    print("===========================================================")


def get_relevance_feedback(results):
    """
    Interactively ask user for relevance (y/n).
    Return a list of booleans (True if relevant, False otherwise).
    """
    relevance = []
    print("\n==================== RELEVANCE FEEDBACK ====================")
    for i, (title, link, snippet, is_html) in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Title: {title}")
        print(f"URL:   {link}")
        print(f"Summary: {snippet}")
        user_input = input("Relevant (y/n)? ").strip().lower()
        while user_input not in ["y", "n"]:
            user_input = input("Please enter 'y' or 'n': ").strip().lower()
        relevance.append(user_input == "y")
    print("===========================================================\n")
    return relevance


def compute_precision(results, relevance):
    """
    Compute precision among HTML docs only.
    If 10 documents were returned but only 8 are HTML, we compute
    (# relevant_html) / (number_of_html_docs).
    """
    html_docs = [(res, rel) for (res, rel) in zip(results, relevance) if res[3] is True]
    if not html_docs:
        return 0.0  # No HTML docs => precision = 0

    # Count how many are relevant among the HTML docs
    relevant_html = sum(rel for (_, rel) in html_docs)
    return relevant_html / len(html_docs)


def tokenize(text):
    ## old way using manual listed stopwords and REGEX tokenization
    # """
    # Very basic tokenizer: lowercases, splits on non-alpha, filters out stopwords.
    # TODO: maybe use some lib for tokenization
    # """
    # import re
    # tokens = re.split(r"[^a-zA-Z]+", text.lower())
    # tokens = [t for t in tokens if t and t not in STOPWORDS]
    # return tokens
    '''
    UPDATE: now used nltk library
    '''
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens


def build_tfidf_index(results, use_full_text = False):
    """
    Build a small TF-IDF index for the top-10 documents (title+snippet).
    Return:
      - docs_tokens: a list of dictionaries [ {term: tf_in_doc}, ... ] 
                     for each document in order
      - df: a dictionary {term: number_of_docs_that_have_it}
    By the project hint, we only consider documents that are likely HTML, so we skip non-HTML docs here.
    """
    df = {}
    docs_tokens = []

    for (title, link, snippet, is_html) in results:
        if not is_html:
            # Skip non-HTML docs for indexing
            docs_tokens.append({})
            continue

        if use_full_text:
            full_text = fetch_full_text(link)
            # Combine title and full text
            text = title + " " + full_text
        else:
            text = title + " " + snippet

        tokens = tokenize(text)
        tf_map = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        # Update DF
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

        docs_tokens.append(tf_map)

    return docs_tokens, df


def compute_sum_tfidf(docs_tokens, df, relevance):
    """
    Compute the sum of TF-IDF scores per term, but only over the relevant docs.
    If doc i is relevant, we add doc i's tf-idf for each term to the global sum.
    
    Returns a dict: {term: sum_of_tfidf_over_relevant_docs}
    """
    N = len(docs_tokens)  # Typically 10
    sum_tfidf = {}

    for doc_idx, tf_map in enumerate(docs_tokens):
        if doc_idx >= len(relevance):
            break
        if not relevance[doc_idx]:
            continue  # Only sum from relevant docs

        # For each term in this doc
        for term, tf in tf_map.items():
            # Document frequency
            df_t = df.get(term, 0)
            if df_t == 0:
                # Should not happen if the term is in the doc, but just in case
                continue

            # TF-IDF weighting
            # We use raw TF (no normalization by doc length), 
            #   multiplied by log(N / df_t) same as lecture
            tfidf_val = tf * math.log(float(N) / df_t, 2) 
            sum_tfidf[term] = sum_tfidf.get(term, 0.0) + tfidf_val

    return sum_tfidf


def pick_new_terms(current_query_terms, sum_tfidf, max_new_terms=2):
    """
    Pick up to 'max_new_terms' terms with highest sum_tfidf scores that 
    are not already in the query.
    """
    # Sort terms by descending TF-IDF
    sorted_terms = sorted(sum_tfidf.items(), key=lambda x: x[1], reverse=True)

    current_set = set(t.lower() for t in current_query_terms)
    new_terms = []
    for term, score in sorted_terms:
        if term not in current_set:
            new_terms.append(term)
        if len(new_terms) == max_new_terms:
            break
    return new_terms


def reorder_query(terms):
    """
    TODO: I have not think of a good way to implement this yet
    """
    return terms


def main():
    google_api_key, google_engine_id, target_precision, initial_query = parse_args()
    service = build_service(google_api_key)

    # Current query terms
    current_query_terms = initial_query.split()
    iteration = 1

    while True:
        print(f"\n==================== Iteration {iteration} ====================")
        # Reorder terms (if you have a more advanced method, adapt here)
        current_query_terms = reorder_query(current_query_terms)

        query_str = " ".join(current_query_terms)
        print(f"Current query: {query_str}")

        # 1. Retrieve top-10 results
        results = search_query(service, google_engine_id, query_str, num_results=10)
        if len(results) < 10 and iteration == 1:
            # As instructions say, if fewer than 10 results in first iteration, just stop
            print("Fewer than 10 results returned in the first iteration. Stopping.")
            break

        if not results:
            print("No results retrieved. Stopping.")
            break

        # Display the results
        display_results(results)

        # 2. Get user feedback (y/n)
        relevance = get_relevance_feedback(results)

        # 3. Compute precision (HTML docs only)
        precision = compute_precision(results, relevance)
        print(f"Precision = {precision:.4f}")

        # 4. Check stopping conditions
        if precision >= target_precision:
            print(f"Desired precision {target_precision} reached or exceeded. Stopping.")
            break
        if precision == 0.0:
            print("Precision is 0 => no relevant results among top-10. Stopping.")
            break

        # 5. Build TF-IDF index for these 10 results
        docs_tokens, df = build_tfidf_index(results, True)

        # 6. Compute sum of TF-IDF for each term across relevant docs only
        sum_tfidf = compute_sum_tfidf(docs_tokens, df, relevance)

        # 7. Pick up to 2 new terms not already in the query
        new_terms = pick_new_terms(current_query_terms, sum_tfidf, max_new_terms=2)
        if not new_terms:
            print("No new terms to add. Stopping.")
            break

        print(f"Expanding query with new terms: {new_terms}")
        # 8. Expand current query
        current_query_terms.extend(new_terms)

        iteration += 1

    print("\n==================== Finished ====================")
    print(f"Final Query: {' '.join(current_query_terms)}")
    print("Goodbye!")


if __name__ == "__main__":
    main()
