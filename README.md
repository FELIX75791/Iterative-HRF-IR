# Iterative Human-Relevance-Feedback Information Retrieval (Project 1)

## Author

Ziyue Jin, UNI: zj2393 (zj2393@columbia.edu)

Ken Deng, UNI: kd3005 (kd3005@columbia.edu)

## List of Files
```
run.py
requirements.txt
README.md
```


## Quick Start

In a VM that strictly follows the set-up instructions and already activate the venv:

Install required dependencies:
```
pip install -r requirements.txt
```
Usage of the main script:
```
python run.py <google_api_key> <google_engine_id> <target_precision> "<query>"
```

For example:

```
python run.py ABC123XYZ abc456def 0.9 "milky way"
```

## Engine ID and API Key

will update when submitting


## Detailed Description of the Query-Modification Method

This project uses a **Rocchio-based** approach to pick new terms for query expansion and a **bigram-based** heuristic to reorder them:

---

### 1. Building TF-IDF Vectors

1. For each (HTML) document in the retrieved set, we construct a term-frequency map, `tf_map`, by tokenizing the document’s text.
2. We maintain a global document-frequency map, `df`, that counts how many documents contain each term.
3. The **normalized TF-IDF** weight for a term \( t \) in document \( d \) is computed as:

$$TF-IDF(t, d) = ( TF(t, d) / TotalTerms(d) ) * log_2( N / DF(t) )$$

where:
- `TF(t, d)` is the frequency of term \( t \) in document \( d \),
- `TotalTerms(d)` is the sum of frequencies of all terms in \( d \),
- `N` is the total number of indexed documents (typically 10 in this context),
- `DF(t)` is the number of documents that contain term \( t \).

---

### 2. Rocchio-Based Term Selection

We use the **Rocchio formula** to construct a new query vector based on user feedback. Let:

- \( Q_0 \) = current query vector (terms in the query, each having some initial weight).
- \( \alpha, \beta, \gamma \) = Rocchio parameters (e.g., \(\alpha = 1.0\), \(\beta = 0.75\), \(\gamma = 0.15\)).
- \( \text{relevant\_vec} \) = average TF-IDF vector over all **relevant** documents.
- \( \text{non\_relevant\_vec} \) = average TF-IDF vector over all **non-relevant** documents.

The updated query vector \( Q_{\text{new}} \) is computed by:

$$
Q_{new} = \alpha * Q_0 + \beta * (\textbf{relevant vector}) - \gamma * (\textbf{non-relevant vector})
$$

**Step-by-step**:

1. **Sum TF-IDF Vectors**:
   - For each relevant document, compute its TF-IDF vector and add it to `relevant_vec`.
   - Do the same for non-relevant documents, building `non_relevant_vec`.

2. **Average Out Contributions**:
   - If there are `R` relevant documents, each term weight in `relevant_vec` is divided by `R`.
   - If there are `NR` non-relevant documents, each term weight in `non_relevant_vec` is divided by `NR`.

3. **Apply Rocchio**:
   - Initialize `new_query_vec` to `α * Q_0`.
   - Add `β * (relevant_vec)` to `new_query_vec`.
   - Subtract `γ * (non_relevant_vec)` from `new_query_vec`.

4. **Pick Top Terms**:
   - Sort all terms in `new_query_vec` by their Rocchio score (descending).
   - Exclude any terms that are already in the current query.
   - Take up to `max_new_terms` (e.g., 2) terms with **positive** scores.

### 3. Bigram-Based Term Reordering

1. We load the **Brown corpus** from **NLTK** to build a bigram frequency distribution, called `bigram_freq`.
2. After selecting new terms from the Rocchio method, we look at their pairwise ordering:

```(term_1, term_2) vs. (term_2, term_1)```

We compare:

```bigram_freq[(term1, term2)] vs. bigram_freq[(term2, term1)]```

3. We choose the ordering that has the **higher** bigram frequency in the Brown corpus. This is a simple heuristic to produce a more natural or commonly occurring phrase.

---

### 4. Integration into the Main Loop

Each iteration proceeds as follows:

1. **Build/Update the TF-IDF Index** for the retrieved documents.
2. **Apply Rocchio** to identify which terms to add:
- Compute relevant and non-relevant vectors based on user feedback.
- Combine them with the current query vector using the Rocchio formula.
3. **Reorder** the newly added terms (if any) using the **bigram frequency** heuristic.
4. **Update the Query** by appending these new terms.
5. **Search** again using the updated query and collect new feedback in the next round.

We repeat until:
- The **target precision** is met.
- No new terms can be added (scores are not positive).
- Precision is zero (no relevant documents at all).
- Or there are fewer than 10 results on the first iteration.

---

**In summary**, the query-modification method combines:
- **Rocchio** to select the most relevant new terms,
- **Bigram frequency** to order them,
- and **user feedback** (relevant vs. non-relevant) to iteratively refine the query in subsequent search rounds.


## Internal Code Design

### High-Level Overview

- **Initialization & Configuration:**  
  The program begins by parsing command-line arguments and initializing required services. This includes setting up the Google Custom Search API service, which is essential for executing search queries.  
  *Key functions:* `parse_args()`, `build_service()`

- **Search and Retrieval:**  
  The system performs a search using the provided query and retrieves a set of results from Google. Each result is processed to determine whether it is an HTML document (which is important for further text processing).  
  *Key functions:* `search_query()`, `is_likely_html()`, `fetch_full_text()`

- **Text Processing and TF-IDF Index Construction:**  
  The retrieved documents are tokenized using the NLTK library, which handles splitting text into words while filtering out common stopwords. A TF-IDF index is built from these tokens.  
  *Key functions:* `tokenize()`, `build_tfidf_index()`, `compute_doc_vector()`  
  *Structure:*  
  - Each document (if HTML) is converted into a term frequency map (`tf_map`).  
  - A global document frequency dictionary (`df`) is built across all documents.  
  - Normalized TF-IDF vectors are computed for each document to account for document length differences.

- **User Relevance Feedback and Precision Calculation:**  
  The system interacts with the user by displaying search results and collecting feedback on the relevance of each result. It then calculates the precision of the results, considering only HTML documents.  
  *Key functions:* `get_relevance_feedback()`, `compute_precision()`

- **User Relevance Feedback and Precision Calculation:**  
  The system interacts with the user by displaying search results and collecting feedback on the relevance of each result. It then calculates the precision of the results, considering only HTML documents.  
  *Key functions:* `get_relevance_feedback()`, `compute_precision()`

- **Query Expansion via the Rocchio Algorithm:**  
  Based on the user’s feedback, the system employs the Rocchio algorithm to modify the current query. This involves:
  - Creating a vector representation of the current query.
  - Calculating the average TF-IDF vectors for relevant and non-relevant documents.
  - Applying the Rocchio formula to generate an updated query vector.
  - Selecting new terms (with positive scores) that are not already in the query.
  *Key functions:* `pick_new_terms_rocchio()`

- **Bigram-Based Term Reordering:**  
  After new terms are selected, the system reorders them using a heuristic based on bigram frequencies from the Brown corpus. This step is designed to improve the naturalness of the query phrase.  
  *Key function:* `reorder_query()`

- **Main Iterative Loop:**  
  The main function (`main()`) ties together all the components into an iterative process. In each iteration:
  1. The current query is executed.
  2. Search results are retrieved and displayed.
  3. User relevance feedback is collected.
  4. Precision is computed to determine if the target is met.
  5. If not, the system updates the query using the Rocchio algorithm and bigram-based reordering.
  6. The process repeats until a stopping condition is met (target precision, no new terms, or zero precision).

---

### External Libraries and Their Roles

- **googleapiclient.discovery:**  
  Used to interact with the Google Custom Search API for executing search queries and retrieving results.

- **requests:**  
  Utilized to download full HTML content from web pages when a more comprehensive text analysis is needed.

- **BeautifulSoup (from bs4):**  
  Employed for parsing HTML content to extract the full text of a webpage.

- **NLTK (Natural Language Toolkit):**  
  Provides essential tools for natural language processing:
  - **Tokenization:** Using `word_tokenize` for splitting text into words.
  - **Stopwords:** Accessing a standard set of English stopwords to filter out common, non-informative words.
  - **Corpora and Bigrams:** Using the Brown corpus to generate bigram frequencies (via `FreqDist` and `bigrams`), which are then used to reorder new query terms.

---

### Summary of the Code Structure

1. **Initialization & Configuration:**  
   Sets up command-line argument parsing and the Google API service.

2. **Search & Retrieval:**  
   Executes the query, retrieves results, and determines which documents are HTML.

3. **Text Processing:**  
   Tokenizes document content and constructs a TF-IDF index with normalized weights.

4. **User Feedback & Precision Measurement:**  
   Displays results, collects user feedback, and computes precision over the relevant HTML documents.

5. **Query Expansion:**  
   Applies the Rocchio algorithm to derive new query terms based on user feedback and reorders them using bigram frequency heuristics.

6. **Iterative Loop:**  
   Integrates all components into an iterative process that continues refining the query until the stopping criteria are met.

This modular design ensures that each component is focused on a specific task, making the system both robust and easy to maintain or extend. External libraries are seamlessly integrated to handle tasks such as API communication, web content extraction, and natural language processing, which significantly enhances the overall functionality and reliability of the project.
