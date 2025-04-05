from flask import Flask, render_template, request
import pandas as pd
import requests
import json

app = Flask(__name__)

# Load CSVs and extract available titles for dropdown
collab_df = pd.read_csv("models/collab_recs_full.csv")
content_df = pd.read_csv("models/content_recs_full.csv")
titles = collab_df["title"].drop_duplicates().head(10).tolist()

# Azure ML API Call
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

fixed_user_id = -6.824891492208524e+18

@app.route("/", methods=["GET", "POST"])
def index():
    selected_title = ""
    collab_recs = []
    content_recs = []
    azure_recs = []

    if request.method == "POST":
        selected_title = request.form.get("title")

        # Collaborative
        collab_row = collab_df[collab_df["title"] == selected_title]
        if not collab_row.empty:
            collab_recs = collab_row.iloc[0, 1:].dropna().tolist()

        # Content
        content_row = content_df[content_df["Title"] == selected_title]
        if not content_row.empty:
            content_recs = content_row.iloc[0, 1:].dropna().tolist()

        # Azure ML
        azure_recs = get_azure_recommendations(fixed_user_id)

    return render_template(
        "index.html",
        titles=titles,
        selected_title=selected_title,
        collab_recs=collab_recs,
        content_recs=content_recs,
        azure_recs=azure_recs
    )

if __name__ == "__main__":
    app.run(debug=True)
