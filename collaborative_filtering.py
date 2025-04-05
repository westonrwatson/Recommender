import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Step 1: Load data
df = pd.read_csv("data/users_interactions.csv")

# Step 2: Map eventType to numeric ratings
event_type_map = {
    "VIEW": 1,
    "LIKE": 2,
    "FOLLOW": 3,
    "BOOKMARK": 4,
    "COMMENT CREATED": 5
}
df = df[df.eventType.isin(event_type_map)]
df["rating"] = df["eventType"].map(event_type_map)

# Step 3: Create item-user matrix
U = df['personId'].nunique()
I = df['contentId'].nunique()

user_mapper = dict(zip(np.unique(df['personId']), list(range(U))))
item_mapper = dict(zip(np.unique(df['contentId']), list(range(I))))
item_inv_mapper = dict(zip(list(range(I)), np.unique(df['contentId'])))

user_index = [user_mapper[i] for i in df['personId']]
item_index = [item_mapper[i] for i in df['contentId']]
X = csr_matrix((df["rating"], (item_index, user_index)), shape=(I, U))  # Item x User

# Step 4: Train the KNN model
k = 6  # Include self + 5 recommendations
knn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="cosine")
knn.fit(X)

# Step 5: Generate recommendations for all items
rows = []
for item_idx in range(I):
    item_vector = X[item_idx]
    distances, indices = knn.kneighbors(item_vector.reshape(1, -1))

    # Exclude the first index (it's the item itself)
    for neighbor_idx, dist in zip(indices.flatten()[1:], distances.flatten()[1:]):
        rows.append({
            "baseContentId": item_inv_mapper[item_idx],
            "recommendedContentId": item_inv_mapper[neighbor_idx],
            "similarity": 1 - dist  # Convert distance to similarity
        })

# Step 6: Save results to CSV
output_df = pd.DataFrame(rows)
output_df.to_csv("models/collab_recs_full.csv", index=False)

print("✅ Collaborative filtering recommendations for ALL items saved to models/collab_recs_full.csv")
