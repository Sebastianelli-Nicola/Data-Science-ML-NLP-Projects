# Data Science & Artificial Intelligence — End-to-End Portfolio

[![Institution: UNIVPM](https://img.shields.io/badge/Institution-UNIVPM-003366.svg)](#authors--academic-context)
[![Degree: M.Sc. Computer & Automation Eng.](https://img.shields.io/badge/Degree-M.Sc._Computer_%26_Automation_Engineering-blue.svg)](#authors--academic-context)
[![Python: 3.8 | 3.10+](https://img.shields.io/badge/Python-3.8_%7C_3.10%2B-3776AB.svg)](#consolidated-tech-stack)
[![Projects: 5 Complete Modules](https://img.shields.io/badge/Modules-5_End--to--End_Projects-brightgreen.svg)](#repository-structure--documentation)

A multi-disciplinary **Data Science and Artificial Intelligence** repository developed during the Master's Degree in **Computer and Automation Engineering** at **Università Politecnica delle Marche (UNIVPM)** (A.Y. 2025/2026).

This repository collects five end-to-end projects spanning **Supervised Classification**, **Unsupervised Clustering**, **Financial Time Series Forecasting**, **Complex Network & Graph Analysis (SNA)**, **Natural Language Processing (NLP)**, **Conversational AI (Rasa)**, and **Generative AI / Prompt Engineering**.

---

## Repository Structure & Documentation

Each module is self-contained in its dedicated directory along with source code, notebooks, and the full technical report in PDF format:

1. **[Module 1 — Healthcare ML (Classification & Clustering) & Financial Time Series Forecasting](./01_ML_Classification_Clustering_Forecasting/Progetto_1_ML_Forecasting.pdf)** *(78-page Report)*
2. **[Module 2 — Social Network Analysis (SNA) of the Global Airline Route Network](./02_Social_Network_Analysis_Flight_Routes/Progetto_2_SNA_Rotte_Aeree.pdf)** *(52-page Report)*
3. **[Module 3 — ChronoBot: Interactive Historical Conversational Agent with Rasa & LLMs](./03_ChronoBot_Rasa_Conversational_AI/Progetto_3_ChronoBot_Rasa.pdf)** *(39-page Report)*
4. **[Module 4 — Natural Language Processing on E-Commerce Clothing Reviews](./04_NLP_Ecommerce_Reviews/Progetto_4_NLP_Ecommerce.pdf)** *(20-page Report)*
5. **[Module 5 — Prompt Engineering: Smart Greenhouse BI & Epidemic Simulator GUI](./05_Prompt_Engineering_GenAI/Progetto_5_Prompt_Engineering.pdf)** *(34-page Report)*

---

## Module 1: Healthcare ML & Financial Time Series Forecasting

** Technical Report:** [Progetto_1_ML_Forecasting.pdf](./01_ML_Classification_Clustering_Forecasting/Progetto_1_ML_Forecasting.pdf)

### Part A — Cancer Risk Factors: Multi-Class Classification & Clustering
Analysis of a clinical and behavioral dataset (**2,000 patients, 21 attributes**) linking demographic, lifestyle (`Smoking`, `Alcohol_Use`, `BMI`, `Diet`, `Physical_Activity`, `Air_Pollution`), and genetic markers (`BRCA_Mutation`, `Family_History`, `H_Pylori`) to five cancer types (*Lung, Breast, Colon, Prostate, Skin*) and three ordinal risk levels (`Low`: 16.2%, `Medium`: 78.8%, `High`: 5.0%).

* **Supervised Classification Protocol:**
  * Built leakage-free `Scikit-Learn` pipelines (`ColumnTransformer` with `StandardScaler` and `One-Hot Encoding`) under an 80/20 stratified train-test split.
  * Evaluated two experimental scenarios: **Scenario A (Optimistic)** including derived summary variables (`Overall_Risk_Score`), and **Scenario B (Realistic)** restricted strictly to observable patient risk factors.
  * Addressed severe class imbalance (`High` risk at 5%) using custom class weights (`High = 2.0` in Logistic Regression, `balanced_subsample` in Random Forest) and evaluated models via **Balanced Accuracy**, **Macro F1**, **One-vs-Rest ROC-AUC**, and **High-Risk Recall**.

| Model | Scenario A (Balanced Acc / Macro F1) | Scenario B Realistic (Balanced Acc) | Scenario B (Macro F1) | Scenario B (ROC-AUC OvR) | Scenario B (`High` Class Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multi-Class Logistic Regression (Weighted)** | 0.967 / 0.973 | **0.692** | **0.711** | **0.953** | **0.350** |
| **Gradient Boosting** | 0.999 / 0.997 | 0.537 | 0.576 | 0.912 | 0.150 |
| **Random Forest (`balanced_subsample`)** | 1.000 / 1.000 | 0.435 | 0.457 | 0.917 | 0.000 |

* **Unsupervised Clustering & Manifold Learning:**
  * Standardized 18 numerical/boolean features; analyzed cumulative explained variance via **PCA** (requiring 10 components for 70% variance and 14 for 90% variance) and projected local neighborhood structures into 2D using **t-SNE** (`perplexity=30`).
  * **K-Means ($K=7$ selected via SSE Elbow Method):** Identified 7 distinct clinical profiles (e.g., low-risk young non-smokers in Clusters 0–1; heavy smokers/drinkers in Cluster 4; a specialized high-obesity `BRCA_Mutation` cohort in Cluster 6), achieving a mean Silhouette score of `0.075` reflective of continuous clinical risk gradients.
  * **DBSCAN Grid Search:** Explored $\epsilon \in [3.0, 5.0]$ and `min_samples` $\in [3, 15]$ across cluster count, noise ratio, and Silhouette heatmaps (optimal configuration at $\epsilon=5.0, \text{min\_samples}=3$, yielding 2 macro-clusters and a non-noise Silhouette of `0.289`).

### Part B — Google Stock Data (`GOOGL`): Time Series Forecasting (2004–2022)
Historical analysis and forecasting across **4,431 trading days** (August 2004 – March 2022), including a dedicated study of volatility clustering, return kurtosis (`10.39`), and the V-shaped COVID-19 market crash and recovery in 2020.

* **Adjusted Close Price Forecasting — `ARIMA(3, 2, 1)`:**
  * Trained on a 10-year modern window (2012–2022: 2,231 training days, 286 test days).
  * Performed **STL Decomposition** and **Augmented Dickey-Fuller (ADF)** stationarity tests, applying second-order differencing ($d=2$, reducing ADF $p$-value from `0.998` to $1.21 \times 10^{-28}$).
  * Selected `ARIMA(3,2,1)` via AIC Grid Search (`AIC = 18779.64`), achieving a test **MAPE of 12.66%** (`MAE = $344.99`, `RMSE = $399.31`).
* **Monthly Trading Volume Forecasting — `SARIMAX(1, 1, 1)x(1, 0, 2, 12)`:**
  * Resampled daily volume to monthly averages (169 train months, 43 test months) to capture annual trading cycles ($m=12$).
  * Validated stationarity via first-order differencing ($d=1$, ADF $p$-value = `0.0079`) and optimized via Auto-ARIMA (`AIC = 5517.18`), achieving a test **MAPE of 29.04%** (`MAE = 469,935` shares).
* **Intraday Volatility (`High - Low` Range) Forecasting — Meta `Prophet`:**
  * Modeled daily price excursion (`3,544` train days, `887` test days) integrating weekly/yearly Fourier seasonalities and **US Market Holidays**, followed by residual Q-Q Plot and ACF diagnostics (`MAE = $15.34`, `RMSE = $23.39`).

---

## Module 2: Social Network Analysis (SNA) of Global Flight Routes

** Technical Report:** [Progetto_2_SNA_Rotte_Aeree.pdf](./02_Social_Network_Analysis_Flight_Routes/Progetto_2_SNA_Rotte_Aeree.pdf)

Graph-theoretical modeling and structural analysis of the global aviation network built from **67,663 international flight routes** operated by **568 airlines** across **3,425 airports**.

* **Graph Construction (`NetworkX`):**
  * Engineered both a **Weighted Directed Graph (`DiGraph`)** (3,425 nodes, 37,595 directed edges, density `0.00321`) and a **Derived Undirected Graph** (3,425 nodes, 19,257 edges, density `0.00328`), where edge weights represent the number of distinct airlines operating each route.
  * Extracted connected components: Giant Weakly Connected Component of **3,397 nodes** (99.2% of the network) and Giant Strongly Connected Component of **3,354 nodes**.
* **Topological & Degree Distribution Analysis:**
  * Demonstrated a heavy-tailed, scale-free **hub-and-spoke** topology on log-log degree distributions: mean in/out-degree of `10.98` vs. median of `3.0` (max in-degree `238`, max out-degree `239`), with a near-linear correlation between in-degree and out-degree across major hubs (`FRA`, `CDG`, `AMS`, `IST`, `ATL`).
* **Multi-Dimensional Centrality Ranking:**
  * **Betweenness Centrality** (sampled $k=200$): Highlighted intercontinental bridge hubs led by **SEA** (`0.0718`), **LAX** (`0.0692`), **DXB** (`0.0691`), and **ANC** (`0.0622` — key transpolar/cargo relay).
  * **Closeness Centrality:** Identified the most globally accessible hubs led by **FRA** (`0.3956`), **CDG** (`0.3932`), **LHR** (`0.3915`), **DXB** (`0.3873`), and **AMS** (`0.3861`).
  * **Weighted PageRank:** Ranked **ATL** first (`0.00934`) with a wide margin over **ORD** (`0.00588`), **LAX** (`0.00570`), and **DFW** (`0.00540`).
  * **Eigenvector Centrality:** Confirmed `ATL`, `LHR`, `ORD`, `JFK`, and `LAX` as the most influential core nodes.
* **Mesoscale & Local Network Structures:**
  * **Louvain Community Detection:** Partitioned the main component into **19 geopolitical/regional communities** (largest community: 659 North American airports; second: 524 airports).
  * **k-Core Decomposition:** Extracted the innermost structural nucleus at **maximum core number $k = 31$**, comprising **93 densely interconnected airports** and **2,257 edges**.
  * **Ego-Network & Clustering Coefficient Analysis:** Contrasted **Frankfurt (`FRA`)** (245 nodes, 4,601 edges, local clustering `0.1470`) and **Dubai (`DXB`)** (189 nodes, 2,613 edges, local clustering `0.1380`) against the global average clustering coefficient (`0.4883`).
  * **Maximal Cliques:** Enumerated **22,763 maximal cliques**, identifying **2 maximum cliques of size 24** (fully connected 24-airport subgraphs).

---

## Module 3: ChronoBot — Historical Conversational AI (Rasa + Telegram)

** Technical Report:** [Progetto_3_ChronoBot_Rasa.pdf](./03_ChronoBot_Rasa_Conversational_AI/Progetto_3_ChronoBot_Rasa.pdf)

An interactive educational chatbot built with the open-source **Rasa Framework (Rasa NLU + Rasa Core)** and deployed on **Telegram** via **Ngrok** webhooks, allowing users to converse in first person with **7 iconic historical figures**: *Julius Caesar, Cleopatra, Leonardo da Vinci, Napoleon Bonaparte, Pythagoras, Marie Curie,* and *Albert Einstein*.

* **LLM-Driven Synthetic Knowledge Base (`knowledge_base.py`):**
  * Designed persona-specific system prompts to generate a human-verified, hallucination-free nested dictionary (`CHARACTERS_DB`) featuring historical metadata, gender-aware inflection rules, first-person thematic overviews, granular sub-topics, and character-authentic fallback responses.
* **Conversational Architecture & Intent Engineering:**
  * Configured **16 specialized intents** including `greet`, `goodbye`, `help`, `select_character`, `change_character`, `ask_specific_topic`, and **9 thematic inquiry intents** (`ask_vita`, `ask_studio`, `ask_scoperte`, `ask_epoca`, `ask_curiosita`, `ask_citazioni`, `ask_relazioni`, `ask_eredita`, `ask_futuro`).
  * Implemented custom Python actions (`actions.py`) featuring accent/case-insensitive keyword matching on sub-topics, dynamic follow-up topic suggestions (`get_random_suggestions`), slot resetting on character switches, and interactive Telegram inline button menus.

---

## Module 4: NLP on Women's E-Commerce Clothing Reviews

** Technical Report:** [Progetto_4_NLP_Ecommerce.pdf](./04_NLP_Ecommerce_Reviews/Progetto_4_NLP_Ecommerce.pdf)

End-to-end **Natural Language Processing** pipeline analyzing **23,486 customer reviews** across 10 attributes from a women's clothing e-commerce platform.

* **Text Pre-Processing & Exploratory Analysis:**
  * Cleaned raw text via regex punctuation/number removal, lowercasing, English stop-word filtering (`NLTK`), and morphological lemmatization (`WordNetLemmatizer`).
  * Performed demographic and product-class EDA (identifying *Layering* and *Jeans* as top-rated categories $>4.3/5$, and *Trend* / *Casual bottoms* as critical classes) and contrasted 5-star vs. 1-star vocabulary via **WordClouds**.
* **Information Extraction (`spaCy` & `CountVectorizer`):**
  * Extracted top 20 **Bi-grams** (`"true size"`, `"fit perfectly"`, `"look great"`, `"run large"`).
  * Applied **Part-of-Speech (POS) Tagging** (`en_core_web_sm`) to decouple tangible product **Aspects (Nouns)** (*dress, size, color, fabric, fit, length, waist, material*) from customer **Qualifiers (Adjectives)** (*great, small, large, perfect, soft, comfortable, flattering*) and lexicon-based emotional subjectivity markers.
* **Supervised Text Classification:**
  * **Binary Recommendation Prediction (`Recommended IND` — `FastText`):** Addressed positive class imbalance via training-set upsampling (`resample`) and trained a trigram-aware **FastText** model (`wordNgrams=3`, `lr=0.5`, `epochs=25`), achieving **80% overall accuracy**, **0.98 precision on positive reviews**, and a **92% recall on negative reviews** (minimizing false positives to `1.52%` to catch dissatisfied customers).
  * **Multi-Class Star Rating Prediction (1–5 Stars — `TF-IDF` + `Logistic Regression`):** Vectorized reviews into a 5,000-feature `TF-IDF` matrix and trained a balanced multi-class classifier (`class_weight='balanced'`), achieving **58% accuracy** across the 5 fine-grained classes (`72.9%` accuracy on 5-star reviews and `38%` recall on 1-star reviews, with errors confined to adjacent rating levels).

---

## Module 5: Prompt Engineering — Generative BI & Epidemic Simulator

** Technical Report:** [Progetto_5_Prompt_Engineering.pdf](./05_Prompt_Engineering_GenAI/Progetto_5_Prompt_Engineering.pdf)

An experimental study on **Iterative Prompt Engineering** across visual and code-generation Foundation Models, structured into two applied use cases:

1. **Automated BI Dashboard Design from E-R Schemas (`Gemini Nano Banana`):**
   * Starting from a formal **Entity-Relationship (E-R) schema** of a hydroponic Smart Greenhouse (`SENSORE`, `PIANTA`, `LOG_CONSUMI`), executed an 8-step iterative prompt refinement protocol to generate high-resolution maintenance and agronomic dashboards, systematically resolving data-ink inconsistencies, pie-chart percentage sums, sensor count mismatches (14 active / 4 maintenance / 2 offline), gauge alerts, and 24-hour environmental telemetry curves.
2. **Interactive Epidemic Simulator Desktop Software (`Gemini 1.5 Flash` + `Python Tkinter`):**
   * Conducted an **11-stage iterative code-generation workflow** to build a complete agent-based epidemiological simulator (`Sani`, `Infetti`, `Guariti`, `Vaccinati`, `Deceduti`).
   * Iteratively engineered real-time `Matplotlib` epidemic curves embedded in `Tkinter`, interactive parameter sliders (population size, contagion probability, disease duration, base mortality rate, vaccine coverage/efficacy, and healthcare treatment efficacy reducing mortality via $M_{\text{eff}} = M \times (1 - E_{\text{cure}})$), simulation playback controls (*Start, Pause, Resume, Reset*), and automated post-epidemic summary reports, while diagnosing and fixing LLM-generated runtime exceptions (`KeyError`, `AttributeError`).

---

## Consolidated Tech Stack

| Category | Libraries, Frameworks & Tools |
| :--- | :--- |
| **Core Programming & Data** | Python (3.8 / 3.10+), `pandas`, `numpy`, `scipy` |
| **Machine Learning & Clustering** | `scikit-learn` (*LogisticRegression, RandomForest, GradientBoosting, KMeans, DBSCAN, PCA, TSNE, Pipeline, ColumnTransformer*) |
| **Time Series & Econometrics** | `statsmodels` (*ARIMA, SARIMAX, STL, ADF Test, ACF/PACF*), `prophet`, `pmdarima` |
| **Graph & Network Science** | `networkx` (*DiGraph, Centrality Measures, Louvain Community Detection, k-Core, Ego-Networks, Cliques*) |
| **Natural Language Processing** | `spacy` (`en_core_web_sm` POS Tagging), `nltk` (`WordNetLemmatizer`), `fasttext`, `wordcloud` |
| **Conversational AI & Bots** | `Rasa Open Source` (NLU & Core), `Telegram Bot API`, `aiogram`, `Ngrok` |
| **Generative AI & GUI** | `Gemini 1.5 Flash`, `Gemini Nano Banana`, `Tkinter` / `CustomTkinter`, `matplotlib`, `seaborn` |

---

## Authors & Academic Context

Developed for the **Data Science** course (A.Y. 2025/2026) — M.Sc. in **Computer and Automation Engineering** at **Università Politecnica delle Marche (UNIVPM)**.

* **Nicola Sebastianelli** — [GitHub](https://github.com/Sebastianelli-Nicola)
* **Daniel Diocis Calero**
* **Ettore Ricci**
* **Pietro Salvatore**
