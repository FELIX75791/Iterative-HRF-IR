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

---

# Demo Test Transcript:

---

## Test Case 1: Look for information on the Per Se restaurant in New York City, starting with the query [per se]

## sample prompt: `python run.py <ABC123XYZ> <abc456def> 0.9 "per se"`

<pre>
==================== Iteration 1 ====================
Current query: per se

==================== RELEVANCE FEEDBACK ====================
Result 1
Title: Per Se | Thomas Keller Restaurant Group
URL:   https://www.thomaskeller.com/perseny
Summary: Per Se. Per Se Front Door. center. Michelin; Les Grandes; Relias. About. About; Restaurant · Team · Info & Directions · Gift Experiences · Reservations; Menus & ...
Relevant (y/n)? y
Result 2
Title: Perse Definition & Meaning - Merriam-Webster
URL:   https://www.merriam-webster.com/dictionary/perse
Summary: The meaning of PERSE is of a dark grayish blue resembling indigo. How to use perse in a sentence. Did you know?
Relevant (y/n)? n
Result 3
Title: Today's Menus | Thomas Keller Restaurant Group
URL:   https://www.thomaskeller.com/new-york-new-york/per-se/todays-menus
Summary: Per Se. Daily Menus. Two tasting menus are offered daily: a nine-course chef's tasting menu, as well as a ... Per Se . Complete your gift to make an impact.
Relevant (y/n)? y
Result 4
Title: How does one use 'per se'? : r/grammar
URL:   https://www.reddit.com/r/grammar/comments/otxmm7/how_does_one_use_per_se/
Summary: Jul 29, 2021 ... Comments Section ... It roughly means "by itself”, “in itself” or “of itself” and is often used with a negative statement followed by a but to ...
Relevant (y/n)? n
Result 5
Title: Per se - Wikipedia
URL:   https://en.wikipedia.org/wiki/Per_se
Summary: a primary topic, and an article needs to be written about it. It is believed to qualify as a broad-concept article.
Relevant (y/n)? n
Result 6
Title: The Magic of Napa With Urban Polish - The New York Times
URL:   https://www.nytimes.com/2004/09/08/dining/the-magic-of-napa-with-urban-polish.html
Summary: Sep 8, 2004 ... Frank Bruni reviews Per Se, Thomas Keller's restaurant at Time Warner Center; photos (L)
Relevant (y/n)? n
Result 7
Title: Per Se (@perseny) • Instagram photos and videos
URL:   https://www.instagram.com/perseny/?hl=en
Summary: 283K Followers, 202 Following, 1020 Posts - Per Se (@perseny) on Instagram: "Chef Thomas Keller's 3-Star Michelin restaurant overlooking Columbus Circle ...
Relevant (y/n)? y
Result 8
Title: James Perse Los Angeles
URL:   https://www.jamesperse.com/
Summary: James Perse. Women. Women; Apparel. Apparel Main Menu; New Arrivals · At Home Essentials · Classics · Dresses/Jumpsuits · Sweaters/Cashmere · T-shirts.
Relevant (y/n)? n
Result 9
Title: Per se meaning, how to use per se in a sentence | Readable ...
URL:   https://readable.com/blog/how-to-correctly-use-per-se/
Summary: Aug 11, 2017 ... 'Per se' is a Latin term which literally means, “by itself”, “in itself” or “of itself”. This means you're taking something out of its context to describe it ...
Relevant (y/n)? n
Result 10
Title: Per Se – New York - a MICHELIN Guide Restaurant
URL:   https://guide.michelin.com/us/en/new-york-state/new-york/restaurant/per-se
Summary: Per Se – a Three Stars: Exceptional cuisine restaurant in the 2024 MICHELIN Guide USA. The MICHELIN inspectors' point of view, information on prices, ...
Relevant (y/n)? y
===========================================================

Precision = 0.4000
Expanding query with new terms: ['perseny', 'photos']

==================== Iteration 2 ====================
Current query: per se perseny photos

==================== RELEVANCE FEEDBACK ====================
Result 1
Title: Per Se (@perseny) • Instagram photos and videos
URL:   https://www.instagram.com/perseny/?hl=en
Summary: 283K Followers, 202 Following, 1020 Posts - Per Se (@perseny) on Instagram: "Chef Thomas Keller's 3-Star Michelin restaurant overlooking Columbus Circle ...
Relevant (y/n)? y
Result 2
Title: Per Se | Thomas Keller Restaurant Group
URL:   https://www.thomaskeller.com/perseny
Summary: Per Se. Per Se Front Door. center. Michelin; Les Grandes; Relias. About. About; Restaurant · Team · Info & Directions · Gift Experiences · Reservations; Menus & ...
Relevant (y/n)? y 
Result 3
Title: Per Se (@perseny) • Instagram photos and videos
URL:   https://www.instagram.com/perseny/reels/
Summary: 283K Followers, 202 Following, 1019 Posts - Per Se (@perseny) on Instagram: "Chef Thomas Keller's 3-Star Michelin restaurant overlooking Columbus Circle ...
Relevant (y/n)? y
Result 4
Title: Per Se (@PerSeNY) / X
URL:   https://x.com/perseny?lang=en
Summary: ... of #NYCWFF Goes Virtual, presented by. @CapitalOne . TIX still available: https://nycwff.org/thomas-keller-corey-chow/… Image. Per Se and 5 others.
Relevant (y/n)? y
Result 5
Title: Per Se is 20 this week! So, as we celebrate two decades of making ...
URL:   https://www.instagram.com/chefthomaskeller/p/C3V-hamuAvV/
Summary: Feb 14, 2024 ... ... perseny, as we look back at Per Se's evolution. #PerSe20th ... Photo shared by Chef Thomas Keller on February 14, 2024 tagging @perseny.
Relevant (y/n)? y
Result 6
Title: Per Se, NYC - an extraordinary dining experience | The Foodie World
URL:   https://thefoodieworld.com.au/2014/05/per-se-nyc-an-extraordinary-dining-experience/
Summary: May 24, 2014 ... thomaskeller.com/perseny · Per Se Menu, Reviews, Photos, Location and Info - Zomato. Post Views: 4,064. Tagged under: degustation, fine dining ...
Relevant (y/n)? y
Result 7
Title: PER SE - Updated February 2025 - 8130 Photos & 1977 Reviews ...
URL:   https://www.yelp.com/biz/per-se-new-york
Summary: PER SE, 10 Columbus Cir, Fl 4, New York, NY 10019, 8130 Photos, Mon - 4:30 pm - 8:30 pm, Tue - 4:30 pm - 8:30 pm, Wed - 4:30 pm - 8:30 pm, Thu - 4:30 pm ...
Relevant (y/n)? y
Result 8
Title: Per Se - Updated 2025, American Restaurant in New York, NY
URL:   https://www.opentable.com/r/per-se
Summary: Get menu, photos and location information for Per Se in New York, NY. Or book now at one of our other 17643 great restaurants in New York.
Relevant (y/n)? y
Result 9
Title: Per Se
URL:   https://www.facebook.com/@perse/
Summary: Photos · Videos · Mentions. Details. 󱛐. Page · New American Restaurant. 󱤂. thomaskeller.com/new-york-new-york/per-se. 󱡍. 88% recommend (5,669 Reviews). Per Se ...
Relevant (y/n)? y
Result 10
Title: Arctic Sea Ice Minimum Extent | Vital Signs – Climate Change: Vital ...
URL:   https://climate.nasa.gov/vital-signs/arctic-sea-ice/
Summary: ANNUAL SEPTEMBER MINIMUM EXTENT. Data source: Satellite observations. Credit: NSIDC/NASA. Rate of Change. 12.2. percent per decade. 1980 1990 2000 2010 2020 ...
Relevant (y/n)? n
===========================================================

Precision = 0.9000
Desired precision 0.9 reached or exceeded. Stopping.

==================== Finished ====================
Final Query: per se perseny photos
Goodbye!
</pre>
---

## Test Case 2: Look for information on 23andMe cofounder Anne Wojcicki, starting with the query [wojcicki]

## sample prompt: `python run.py <ABC123XYZ> <abc456def> 0.9 "wojcicki"`

<pre>
==================== Iteration 1 ====================
Current query: wojcicki

==================== RELEVANCE FEEDBACK ====================
Result 1
Title: Susan Wojcicki - Wikipedia
URL:   https://en.wikipedia.org/wiki/Susan_Wojcicki
Summary: an American business executive who was the chief executive officer of YouTube from 2014 to 2023. Her net worth was estimated at $765 million in 2022.
Relevant (y/n)? n
Result 2
Title: Susan Wojcicki (@SusanWojcicki) / X
URL:   https://x.com/susanwojcicki?lang=en
Summary: Susan Wojcicki's posts ... 's team and innovative solutions in sustainability and security to enable transparency and accountability for countries, communities ...
Relevant (y/n)? n
Result 3
Title: Susan Wojcicki (@susanwojcicki) • Instagram photos and videos
URL:   https://www.instagram.com/susanwojcicki/?hl=en
Summary: YouTube CEO · Photo shared by YouTube for Families on May 27, 2022 tagging @susanwojcicki. Say hello to @susanwojcicki, the CEO of @YouTube. · Photo shared by ...
Relevant (y/n)? n
Result 4
Title: Anne Wojcicki (@annewoj23) / X
URL:   https://x.com/annewoj23?lang=en
Summary: CEO and Co-Founder, 23andMe.
Relevant (y/n)? y
Result 5
Title: A personal update from Susan - YouTube Blog
URL:   https://blog.youtube/inside-youtube/a-personal-update-from-susan/
Summary: A personal update from Susan. By Susan Wojcicki. Feb 16, 2023 – 3 minute read. Copy link. A personal update from Susan on stepping back from her role as CEO ...
Relevant (y/n)? n
Result 6
Title: Esther Wojcicki - It is with soul crushing grief that I... | Facebook
URL:   https://www.facebook.com/esther.wojcicki/posts/it-is-with-soul-crushing-grief-that-i-announce-the-passing-of-my-daughter-susan-/10162259101335763/
Summary: Aug 10, 2024 ... It is with soul crushing grief that I announce the passing of my daughter Susan. I am eternally grateful she was in my life.
Relevant (y/n)? n
Result 7
Title: Professor Stanley Wojcicki has died at age 86 | Physics Department
URL:   https://physics.stanford.edu/news/professor-stanley-wojcicki-has-died-age-86
Summary: Jun 5, 2023 ... Stanley G. Wojcicki died on May 31, 2023, at his condo in Los Altos at age 86. He still maintained his home on the Stanford campus.
Relevant (y/n)? n
Result 8
Title: Janet Wojcicki | UCSF Profiles
URL:   https://profiles.ucsf.edu/janet.wojcicki
Summary: Janet Wojcicki, PhD, MPH. Collapse Education and Training. Collapse Websites. Collapse Global Health Equity. Collapse Research Activities and Funding.
Relevant (y/n)? n
Result 9
Title: Esther Wojcicki on LinkedIn: I am sad to announce that my husband ...
URL:   https://www.linkedin.com/posts/estherwojcicki_i-am-sad-to-announce-that-my-husband-of-62-activity-7071514017499918336-ldK4
Summary: Jun 5, 2023 ... I am sad to announce that my husband of 62 years Stanley Wojcicki passed away on Wednesday, May 31 after a long battle with congestive heart ...
Relevant (y/n)? n 
Result 10
Title: Susan Wojcicki ON: How to Avoi - On Purpose with Jay Shetty ...
URL:   https://podcasts.apple.com/us/podcast/susan-wojcicki-on-how-to-avoid-burnout-in-a/id1450994021?i=1000584459639
Summary: Oct 31, 2022 ... Susan Wojcicki is CEO of YouTube, the world's most popular digital video platform used by more than a billion people across the globe.
Relevant (y/n)? n
===========================================================

Precision = 0.1000
Expanding query with new terms: ['anne', 'x']

==================== Iteration 2 ====================
Current query: wojcicki anne x

==================== RELEVANCE FEEDBACK ====================
Result 1
Title: Anne Wojcicki (@annewoj23) / X
URL:   https://x.com/annewoj23?lang=en
Summary: CEO and Co-Founder, 23andMe.
Relevant (y/n)? y
Result 2
Title: Anne Wojcicki (@annewoj23) • Instagram photos and videos
URL:   https://www.instagram.com/annewoj23/?hl=en
Summary: 15K Followers, 238 Following, 83 Posts - Anne Wojcicki (@annewoj23) on Instagram: "@23andme CEO Co-Founder | Think Big | Lead With Science | I ❤️ DNA| ...
Relevant (y/n)? y
Result 3
Title: Susan Wojcicki (YouTube CEO) teaches how to pronounce her ...
URL:   https://www.reddit.com/r/poland/comments/3bihro/susan_wojcicki_youtube_ceo_teaches_how_to/
Summary: Jun 29, 2015 ... ... and it was changed to the americanized Wojcicki. So many Polish ... This made me wonder how another well-known tech CEO, Anne Wojcicki ...
Relevant (y/n)? n
Result 4
Title: Anne Wojcicki on X: "Genetics in action! Love my sisters ...
URL:   https://twitter.com/annewoj23/status/1059331120582381573
Summary: Nov 5, 2018 ... Conversation. Anne Wojcicki. @annewoj23. Genetics in action! Love my sisters! @brkthroughprize. Image. 6:26 AM · Nov 5, 2018.
Relevant (y/n)? y
Result 5
Title: Anne Wojcicki - Wikipedia
URL:   https://en.wikipedia.org/wiki/Anne_Wojcicki
Summary: Anne E. Wojcicki is an American entrepreneur who co-founded and is CEO of the personal genomics company 23andMe. She founded the company in 2006 with Linda ...
Relevant (y/n)? y
Result 6
Title: CEO Anne Wojcicki on turning 23andMe into a 'full-fledged biotech ...
URL:   https://www.biopharmadive.com/news/23andme-ceo-anne-wojcicki-pharma-drug-development/698995/
Summary: Nov 7, 2023 ... 23andMe always had larger aims than at-home genetics testing. In this in-depth interview, Wojcicki explains the company's foray into drug R&D.
Relevant (y/n)? y
Result 7
Title: Anne Wojcicki - XPRIZE Foundation Bio
URL:   https://www.xprize.org/about/people/anne-wojcicki
Summary: Anne Wojcicki. Anne co-founded 23andMe in 2006 after a decade ... and academic researchers could better understand and develop new drugs and diagnostics.
Relevant (y/n)? y
Result 8
Title: Esther Wojcicki - Working out with my daughter Anne and... | Facebook
URL:   https://www.facebook.com/esther.wojcicki/posts/working-out-with-my-daughter-anne-and-her-hip-trainer-yeslandan/10162571368445763/
Summary: Nov 10, 2024 ... Working out with my daughter Anne and her hip trainer " YesLandan ... Please give him a hug for me, ok, Esther Wojcicki. 4 mos. Rajan ...
Relevant (y/n)? y
Result 9
Title: From the Innovator's Workbench with Anne Wojcicki | Stanford ...
URL:   https://biodesign.stanford.edu/our-impact/stories/From-the-Innovators-Workbench-Anne-Wojcicki-CEO-23andme.html
Summary: Catching up with Anne Wojcicki, CEO and co-founder, 23andMe ... Anne Wojcicki spent time discussing digital health technology with the Stanford Biodesign ...
Relevant (y/n)? y
Result 10
Title: A Day in the Life of the Assistant to the 23andMe CEO Anne Wojcicki ...
URL:   https://www.businessinsider.com/day-in-the-life-executive-assistant-ceo-23andme-anne-wojcicki-2023-3
Summary: Mar 16, 2023 ... Kristen Quint has worked for Anne Wojcicki for six years. They sometimes work from Wojcicki's house, and Quint's commute can be up to three ...
Relevant (y/n)? y
===========================================================

Precision = 0.9000
Desired precision 0.9 reached or exceeded. Stopping.

==================== Finished ====================
Final Query: wojcicki anne x
Goodbye!
</pre>
---

## Test Case 3: Look for information on Milky Way chocolate bars, starting with the query [milky way]

## sample prompt: `python run.py <ABC123XYZ> <abc456def> 0.9 "milky way"`
<pre>
==================== Iteration 1 ====================
Current query: milky way

==================== RELEVANCE FEEDBACK ====================
Result 1
Title: Milky Way - Wikipedia
URL:   https://en.wikipedia.org/wiki/Milky_Way
Summary: The Milky Way is a barred spiral galaxy with a D25 isophotal diameter estimated at 26.8 ± 1.1 kiloparsecs (87,400 ± 3,600 light-years), but only about 1,000 ...
Relevant (y/n)? n
Result 2
Title: Explore MILKY WAY® Official Website | Chocolate Bars
URL:   https://www.milkywaybar.com/
Summary: Explore MILKY WAY® Bar products and nutrition information, fun facts about the oh so stretchy caramel chocolate bar, social media channels, and much more!
Relevant (y/n)? y
Result 3
Title: Milky Way LA Restaurant Los Angeles
URL:   https://www.milkywayla.com/
Summary: Opened in 1977 by Bernie & Leah (Spielberg) Adler, The Milky Way is a storied, family-owned restaurant located in the heart of the Pico-Robertson ...
Relevant (y/n)? n
Result 4
Title: The Milky Way Galaxy - NASA Science
URL:   https://science.nasa.gov/resource/the-milky-way-galaxy/
Summary: Nov 8, 2017 ... The Milky Way's elegant spiral structure is dominated by just two arms wrapping off the ends of a central bar of stars.
Relevant (y/n)? n
Result 5
Title: Milky Way galaxy: Facts about our cosmic neighborhood | Space
URL:   https://www.space.com/19915-milky-way-galaxy.html
Summary: Apr 18, 2023 ... Everything else in the galaxy revolves around this powerful gateway to nothingness. In its immediate surroundings is a tightly packed region of ...
Relevant (y/n)? n
Result 6
Title: NASA - The Milky Way Galaxy
URL:   https://imagine.gsfc.nasa.gov/science/objects/milkyway1.html
Summary: The Milky Way is a large barred spiral galaxy. All the stars we see in the night sky are in our own Milky Way Galaxy.
Relevant (y/n)? n
Result 7
Title: The Milky Way Galaxy | AMNH
URL:   https://www.amnh.org/explore/ology/astronomy/the-milky-way-galaxy2
Summary: The Milky Way is a huge collection of stars, dust and gas. It's called a spiral galaxy because if you could view it from the top or bottom, it would look like a ...
Relevant (y/n)? n
Result 8
Title: Milky Way Pittsburgh
URL:   https://www.milkywaypgh.com/
Summary: Here at Milky Way, we pride ourselves on offering delicious Kosher alternatives to the essential pizzeria foods normally sold in a standard pizzeria.
Relevant (y/n)? n
Result 9
Title: Milky Way – Rowster Coffee
URL:   https://rowstercoffee.com/products/milky-way
Summary: Roast Level: Medium - Dark Tasting Notes: Rich, Walnut, Sweet Description: Our flagship blend, showcases our range of roast profiles and ability to source ...
Relevant (y/n)? n
Result 10
Title: Milky Way Farm: Dairy Farm, Ice Cream | Chester County, PA
URL:   https://www.milkywayfarm.com/
Summary: Enjoy a day with family and friends and enjoy some ice cream at our dairy farm in Chester County, PA. Milky Way Farm offers a fun and educational experience ...
Relevant (y/n)? n
===========================================================

Precision = 0.1000
Expanding query with new terms: ['chocolate', 'bars']

==================== Iteration 2 ====================
Current query: milky way chocolate bars

==================== RELEVANCE FEEDBACK ====================
Result 1
Title: Milky Way (chocolate bar) - Wikipedia
URL:   https://en.wikipedia.org/wiki/Milky_Way_(chocolate_bar)
Summary: Milky Way (chocolate bar) ... Milky Way is a brand of chocolate-covered confectionery bar manufactured and marketed by Mars Inc.. There are two varieties: ...
Relevant (y/n)? y
Result 2
Title: Explore MILKY WAY® Official Website | Chocolate Bars
URL:   https://www.milkywaybar.com/
Summary: Explore MILKY WAY® Bar products and nutrition information, fun facts about the oh so stretchy caramel chocolate bar, social media channels, and much more!
Relevant (y/n)? y
Result 3
Title: MilkyWay Candy Milk Chocolate Bars Bulk Pack, Full ... - Amazon.com
URL:   https://www.amazon.com/MilkyWay-Candy-Milk-Chocolate-Bars/dp/B0029JFOVK
Summary: Ingredients: Milk Chocolate (Sugar, Cocoa Butter, Skim Milk, Chocolate, Lactose, Milkfat, Soy Lecithin, Artificial Flavor), Corn Syrup, Sugar, Hydrogenated ...
Relevant (y/n)? y
Result 4
Title: Our Chocolate Candy Bar Products | MILKY WAY®
URL:   https://www.milkywaybar.com/our-products
Summary: So. Much. Caramel. From classic favorites to latest releases, discover every MILKY WAY®chocolate candy product.
Relevant (y/n)? y
Result 5
Title: Milky Way is the best chocolate bar. : r/unpopularopinion
URL:   https://www.reddit.com/r/unpopularopinion/comments/qhxa8h/milky_way_is_the_best_chocolate_bar/
Summary: Oct 28, 2021 ... Milky Way is the best chocolate bar. Call me a child, but the smoothness of the milk chocolate, and the caramel within it gives it a taste of pure bliss and ...
Relevant (y/n)? y
Result 6
Title: Nougat With Chocolate Malt - Like in Milky Way Bars - Cupcake ...
URL:   https://www.cupcakeproject.com/nougat-with-chocolate-malt-like-in/
Summary: Jan 24, 2011 ... Ingredients · ▢ 2 cups granulated sugar · ▢ 2/3 cup light corn syrup · ▢ 2/3 cup water · ▢ 2 large egg whites · ▢ 2 ounces unsweetened chocolate
Relevant (y/n)? n
Result 7
Title: TIL The chocolate bar Milky Way is called Mars Bar by the non-US ...
URL:   https://www.reddit.com/r/todayilearned/comments/17f88kl/til_the_chocolate_bar_milky_way_is_called_mars/
Summary: Oct 24, 2023 ... Aus - Milky way is a tiny bar with just whipped nougat. A Mars bar is full size with a layer of caramel. The advertising campaigns are totally ...
Relevant (y/n)? y
Result 8
Title: Why the (heck) do people buy Milky Way candy bars??? | Fantasy ...
URL:   https://forums.footballguys.com/threads/why-the-heck-do-people-buy-milky-way-candy-bars.761847/
Summary: Oct 19, 2017 ... It's 98% puffed-up, tasteless sugar, surrounded by the thinnest layer of milk chocolate... Minus the milkyness and chocolateyness. Other than ...
Relevant (y/n)? y
Result 9
Title: Food Network - More than 12 million Milky Way chocolate... | Facebook
URL:   https://www.facebook.com/FoodNetwork/posts/more-than-12-million-milky-way-chocolate-bars-leave-the-chicago-factory-on-a-dai/10154011941901727/
Summary: Dec 7, 2016 ... My most favorite candy bar is the Milky Way. It's the stringing of the caramel when I take that first bite and then slowly pull it away from ...
Relevant (y/n)? y
Result 10
Title: MilkyWay Cupcakes
URL:   https://www.thefoodieskitchen.com/2011/09/28/milkyway-cupcakes/
Summary: Sep 28, 2011 ... These are not ordinary cupcakes from a box, these have mini MilkyWay candy bars baked in the middle. Just follow the instructions to make cupcakes from your ...
Relevant (y/n)? y
===========================================================

Precision = 0.9000
Desired precision 0.9 reached or exceeded. Stopping.

==================== Finished ====================
Final Query: milky way chocolate bars
Goodbye!
</pre>