from flask import Flask, render_template, request
import pandas as pd
import requests
import json

app = Flask(__name__)

# Load article and collab data
collab_df = pd.read_csv("models/collab_recs_full.csv")
articles_df = pd.read_csv("data/shared_articles.csv")
articles_df = articles_df[articles_df["eventType"] == "CONTENT SHARED"]

# Filter for valid contentIds with collab recommendations
valid_collab_ids = collab_df["baseContentId"].unique().tolist()
sample_contents = articles_df[articles_df["contentId"].isin(valid_collab_ids)]["contentId"].drop_duplicates().head(10).tolist()

# Azure ML inference function
def get_azure_recommendations(user_id):
    url = 'http://c2226c7d-8e63-4cba-a4dc-f01e0fc99484.eastus2.azurecontainer.io/score'
    key = 'ZILBksIkG2shhQBrCOIYAinjjgKY9iBi'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {key}"
    }
    data = {
        "Inputs": {
            "input1": [{"personId": user_id}]
        }
    }
    try:
        res = requests.post(url, headers=headers, data=str.encode(json.dumps(data)))
        res.raise_for_status()
        result = res.json()
        return [entry.get("contentId", "Unknown") for entry in result.get("Results", {}).get("output1", [])][:5]
    except Exception as e:
        print("Azure request failed:", e)
        return ["results: {}"]

# Fixed user for Azure test
fixed_user_id = -6.824891492208524e+18

@app.route("/", methods=["GET", "POST"])
def index():
    collab_recs = None
    content_recs = None
    azure_recs = None
    content_id = ""

    if request.method == "POST":
        content_id = int(request.form.get("contentId"))

        collab_df_full = pd.read_csv("models/collab_recs_full.csv")
        content_df = pd.read_csv("models/content_recs_full.csv")

        # Collaborative filtering
        collab_top = collab_df_full[collab_df_full["baseContentId"] == content_id]
        collab_recs = collab_top.nlargest(5, "similarity")["recommendedContentId"].tolist() if not collab_top.empty else []

        # Content filtering
        content_top = content_df[content_df["baseContentId"] == content_id]
        content_recs = content_top.nlargest(5, "similarity")["recommendedContentId"].tolist() if not content_top.empty else []

        # Azure ML
        azure_recs = get_azure_recommendations(fixed_user_id)

    return render_template(
        "index.html",
        content_id=content_id,
        content_ids=sample_contents,
        collab_recs=collab_recs,
        content_recs=content_recs,
        azure_recs=azure_recs
    )

if __name__ == "__main__":
    app.run(debug=True)
