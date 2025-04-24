# %%

import wandb

# %%
api = wandb.Api()
# %%

run = api.run("gradient-routing-fwe/l5glivx6")

# %%
# Get history of metrics we care about
history = run.scan_history(keys=["step", "top_routed_tokens"])

# Convert to dataframe
import pandas as pd

df = pd.DataFrame(history)

# %%
# Split the list column into separate columns
df = pd.DataFrame(df["top_routed_tokens"].tolist())


# %%
# Create a function that will identify changes
def highlight_changes(df):
    # Create an empty DataFrame with the same shape as the input
    styled = pd.DataFrame("", index=df.index, columns=df.columns)

    # For each column, compare with previous row and mark changes
    for col in df.columns:
        # Find where values change
        changes = df[col].ne(df[col].shift())
        # Set styling for those cells
        styled.loc[changes, col] = "background-color: yellow"

    return styled


styled_df = df.style.apply(highlight_changes, axis=None)
# %%
styled_df
# %%
