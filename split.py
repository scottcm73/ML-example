
### We did this ###
# You won't be able to run this because you don't have the original Fish.csv
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("Fish.csv")

# I'll give you df_participant and hold out df_holdout for evaluation
df_participant, df_holdout = train_test_split(df, random_state=42, test_size=.3, stratify=df["Species"])

# I'll give you df_demo_holdout which is just s subset of df_participant
# the point of this is so that you can test if your code works
_, df_demo_holdout = train_test_split(df_participant)

# You'll get:
df_participant.to_csv("fish_participant.csv", index=False)
df_demo_holdout.to_csv("fish_holdout_demo.csv", index=False)

# We'll keep:
df_holdout.to_csv("fish_holdout.csv", index=False)
######


