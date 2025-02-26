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