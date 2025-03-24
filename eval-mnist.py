# %%
import matplotlib.pyplot as plt
import torch as t

from data import DataManager
from mlp import MNISTClassifier, MNISTConfig
from train import Checkpoint

# %%
# Load configuration and data
config = MNISTConfig()
data_manager = DataManager()
data_manager.prepare_data(recipe=["mnist"], val_split=0.0, batch_size=10)

# %%
# Load the model
model = MNISTClassifier(config)
manager = Checkpoint(config, data_manager, model)
manager.load_checkpoint(load_best=True)
# Sample 10 random images from test set
test_loader = data_manager.test_loader
# %%
assert test_loader is not None
images, labels = next(iter(test_loader))
images, labels = images[:10], labels[:10]

# Get predictions
with t.no_grad():
    predictions = model(images)
    predicted_labels = t.argmax(predictions, dim=1)

# Display images and predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for idx, (image, true_label, pred_label) in enumerate(
    zip(images, labels, predicted_labels)
):
    axes[idx].imshow(image.squeeze(), cmap="gray")
    color = "green" if true_label == pred_label else "red"
    axes[idx].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
    axes[idx].axis("off")

plt.tight_layout()
plt.show()

# %%
