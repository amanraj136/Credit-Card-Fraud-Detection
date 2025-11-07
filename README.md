\# Credit Card Fraud Detection — Streamlit App + Scikit-learn



<!-- Badges (add CI/CD, Streamlit, or license badges here when available) -->

<!-- \\\[!\\\[Streamlit App](https://img.shields.io/badge/Streamlit-Demo-orange)](https://streamlit.io) -->



<!-- Live demo link placeholder -->

<!-- 🔗 \\\*\\\*Live Demo:\\\*\\\* \\\[Add your Streamlit deployment URL here once published] -->





\## Overview



This project detects fraudulent credit card transactions using classical machine learning models trained on the \*\*Kaggle Credit Card Fraud Detection dataset\*\*.

The dataset is \*\*highly imbalanced\*\* (~0.173% fraud cases). The workflow applies \*\*StandardScaler\*\* for normalization and \*\*SMOTE\*\* for balancing during training.



Multiple algorithms were evaluated — \*\*Logistic Regression\*\*, \*\*Naive Bayes\*\*, \*\*Random Forest\*\*, and \*\*XGBoost\*\* — and the \*\*Gradient Boosting (XGBoost-based)\*\* model was selected for deployment, achieving a \*\*ROC-AUC ≈ 0.979\*\* with strong recall on minority (fraud) detection under higher F-beta settings.



The Streamlit app enables:

\- \*\*Single transaction prediction\*\* via comma-separated input

\- \*\*Batch CSV upload\*\* for large-scale inference with downloadable results





\## Dataset Summary



\- \*\*Size:\*\* 284,807 transactions × 31 columns

\- \*\*Columns:\*\*

  - \*\*Features:\*\* `Time`, `V1`, `V2`, …, `V28`, `Amount`

  - \*\*Target:\*\* `Class` (0 = Normal, 1 = Fraud)

\- \*\*Challenge:\*\* Only ~0.173% of all transactions are fraudulent (extreme imbalance).





\## Methodology



\- \*\*Preprocessing:\*\* Standard scaling for all numeric columns

\- \*\*Imbalance Handling:\*\* SMOTE applied on training set only (to prevent data leakage)

\- \*\*Models Evaluated:\*\*

  - Logistic Regression → ROC-AUC ≈ 0.971

  - Naive Bayes → ROC-AUC ≈ 0.965

  - Random Forest → ROC-AUC ≈ 0.968

  - \*\*XGBoost (final)\*\* → ROC-AUC ≈ 0.979

\- \*\*Metrics Used:\*\*

  ROC-AUC, PR-AUC, F1-score, and \*\*F-beta (β ∈ {0.5, 1, 2, 3})\*\* for recall–precision trade-offs

\- \*\*Chosen Model:\*\* XGBoost for strong minority recall and balanced F-beta performance





\## Key Results



\- \*\*Best Model:\*\* Gradient Boosting (XGBoost)

\- \*\*ROC-AUC:\*\* ~0.979 on test data

\- \*\*Recall (Fraud Class):\*\* Highest under F-beta > 1 emphasis

\- \*\*Trade-off:\*\* Slight precision loss for improved fraud recall

\- \*\*Figures:\*\*

  - `figures/class\\\_distribution.png`

  - `figures/temporal\\\_analysis\\\_by\\\_day.png`

  - `figures/confusion\\\_matrices.png`

  - `figures/fbeta\\\_comparison.png`

  - `figures/fbeta\\\_grouped\\\_comparison.png`





\## How to Run Locally

1. Python 3.10+
2. Install dependencies:

&nbsp;	pip install -r requirements.txt

3\. Ensure the model file `best\_model\_gradient\_boosting.pkl` is in the project root.

4\. Start app:

&nbsp;	Option A: Double-Click `app.bat` to launch the app in your browser.

&nbsp;	Option B: `streamlit run app.py` in the terminal.





\## Files

\- `app.py` — Streamlit UI and inference.

\- `app.bat` — Launch the app in the default browser.

\- `best\_model\_gradient\_boosting.pkl` — trained pkl model.

\- `dataset/` — use `creditcard.csv` at project root.

\- `figures/` — ROC/PR Curves, Confusion Matrces, Class Distribution, etc. 

\- `notebook/` — training notebook.

\- `test\_samples/` — Batch test samples in `.csv` format for downloading and testing purposes.

