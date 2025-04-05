import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Step 1: Load articles and filter for published ones
articles = pd.read_csv("data/shared_articles.csv")
articles = articles[articles["eventType"] == "CONTENT SHARED"].reset_index(drop=True)

# Step 2: Fill missing values in the text field
articles["text"] = articles["text"].fillna("")

# Step 3: Build TF-IDF matrix from the text column
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(articles["text"])

# Step 4: Compute cosine similarity between all articles
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Step 5: Generate recommendations for all articles
rows = []

for idx in range(len(articles)):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Skip itself

    for rec_idx, score in sim_scores:
        rows.append({
            "baseContentId": articles.iloc[idx]["contentId"],
            "recommendedContentId": articles.iloc[rec_idx]["contentId"],
            "similarity": score
        })

# Step 6: Save all recommendations to CSV
output_df = pd.DataFrame(rows)
output_df.to_csv("models/content_recs_full.csv", index=False)

print("✅ Content-based recommendations for ALL articles saved to models/content_recs_full.csv")
