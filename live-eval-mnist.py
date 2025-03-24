# %%
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from scipy.stats import entropy
from torchvision import transforms

from data import DataManager
from mlp import MNISTClassifier, MLPConfig
from checkpoint import Checkpoint


class DrawingInterface:
    def __init__(self, model: MNISTClassifier, data_manager: DataManager):
        self.model = model
        self.model.eval()
        self.data_manager = data_manager

        # Drawing window setup
        self.window_size = 280  # 10x the MNIST size for better drawing
        self.drawing = np.zeros((self.window_size, self.window_size), dtype=np.uint8)
        self.last_point = None
        self.drawing_thickness = 20

        # Create window and set up mouse callback
        cv2.namedWindow('Draw a digit (Press "q" to quit)')
        cv2.setMouseCallback('Draw a digit (Press "q" to quit)', self.mouse_callback)

        # Initialize prediction display
        self.prediction = None
        self.last_prediction_time = 0
        self.prediction_interval_MS = 0.3

        # Create the same transform as used in training
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.28,), (0.35,))]
        )

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.last_point is not None:
            cv2.line(
                self.drawing,
                self.last_point,
                (x, y),
                (255, 255, 255),
                self.drawing_thickness,
            )
            self.last_point = (x, y)
            # Only update prediction if enough time has passed
            current_time = time.time()
            if current_time - self.last_prediction_time >= self.prediction_interval_MS:
                self.update_prediction()
                self.last_prediction_time = current_time
        elif event == cv2.EVENT_LBUTTONUP:
            self.last_point = None
            # Always update prediction when mouse is released
            self.update_prediction()
            self.compare_to_test()

    def preprocess_drawing(self):
        # Resize to MNIST size (28x28)
        resized = cv2.resize(self.drawing, (28, 28), interpolation=cv2.INTER_AREA)
        # Convert to PIL Image for consistent preprocessing
        from PIL import Image

        pil_image = Image.fromarray(resized)

        # Apply the same transform as training
        tensor = self.transform(pil_image)
        # Ensure we have a torch tensor and add batch dimension
        if not isinstance(tensor, t.Tensor):
            tensor = t.from_numpy(tensor)
        tensor = tensor.unsqueeze(0)
        return tensor

    def update_prediction(self):
        with t.no_grad():
            input_tensor = self.preprocess_drawing()

            output = self.model(input_tensor)
            prediction = output.argmax(dim=1).item()
            self.prediction = prediction

    def plot_distributions_with_kl(
        self, drawing_pixels, test_pixels, ax1, ax2, bins=50
    ):
        """Plot distributions with KL divergence comparison."""
        # Calculate histograms
        drawing_hist, drawing_bins = np.histogram(
            drawing_pixels, bins=bins, density=True
        )
        test_hist, test_bins = np.histogram(test_pixels, bins=bins, density=True)

        # Calculate KL divergence using scipy's entropy
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        test_hist = np.clip(test_hist, epsilon, 1)
        drawing_hist = np.clip(drawing_hist, epsilon, 1)

        # Normalize the distributions
        drawing_hist = drawing_hist / np.sum(drawing_hist)
        test_hist = test_hist / np.sum(test_hist)

        # Calculate KL divergence
        kl_div = entropy(drawing_hist, test_hist)

        # Plot drawing distribution
        ax1.hist(drawing_pixels, bins=bins, alpha=0.7, color="blue", density=True)
        drawing_mean = drawing_pixels.mean()
        drawing_std = drawing_pixels.std()
        ax1.axvline(
            drawing_mean,
            color="red",
            linestyle="dashed",
            label=f"Mean: {drawing_mean:.2f}",
        )
        ax1.text(0.02, 0.95, f"Std Dev: {drawing_std:.2f}", transform=ax1.transAxes)
        ax1.set_title("Your Drawing Distribution")
        ax1.legend()

        # Plot test distribution
        ax2.hist(test_pixels, bins=bins, alpha=0.7, color="blue", density=True)
        test_mean = test_pixels.mean()
        test_std = test_pixels.std()
        ax2.axvline(
            test_mean, color="red", linestyle="dashed", label=f"Mean: {test_mean:.2f}"
        )
        ax2.text(0.02, 0.95, f"Std Dev: {test_std:.2f}", transform=ax2.transAxes)
        ax2.set_title("Test Image Distribution")
        ax2.legend()

        return kl_div

    def compare_to_test(self):
        test_loader = self.data_manager.test_loader
        assert test_loader is not None
        x, y = next(iter(test_loader))

        # Create a figure with subplots - images on top, distributions below
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Plot preprocessed drawing
        input_tensor = self.preprocess_drawing()
        with t.no_grad():
            pred1 = self.model(input_tensor).argmax(dim=1).item()
        ax1.imshow(input_tensor.squeeze(), cmap="gray")
        ax1.set_title(f"Your Drawing (Prediction: {pred1})")
        ax1.axis("off")

        # Find first test image with label matching pred1
        matching_indices = (y == pred1).nonzero().squeeze()
        if len(matching_indices) > 0:
            first_match_idx = matching_indices[0].item()
            with t.no_grad():
                pred2 = (
                    self.model(x[first_match_idx : first_match_idx + 1])
                    .argmax(dim=1)
                    .item()
                )
            ax2.imshow(x[first_match_idx].squeeze(), cmap="gray")
            ax2.set_title(
                f"Test Image (Label: {y[first_match_idx].item()}, Prediction: {pred2})"
            )
        else:
            ax2.text(
                0.5,
                0.5,
                f"No test image with label {pred1}",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax2.set_title("No matching test image found")
        ax2.axis("off")

        # Get pixel distributions
        drawing_pixels = input_tensor.squeeze().numpy().flatten()
        if len(matching_indices) > 0:
            test_pixels = x[first_match_idx].squeeze().numpy().flatten()
        else:
            test_pixels = np.zeros_like(drawing_pixels)  # Use zeros if no match found

        # Plot distributions with KL divergence
        kl_div = self.plot_distributions_with_kl(drawing_pixels, test_pixels, ax3, ax4)
        plt.suptitle(f"KL Divergence: {kl_div:.4f}")
        plt.tight_layout()
        plt.show()

    def run(self):
        while True:
            # Create a copy of the drawing for display
            display = self.drawing.copy()

            # Add prediction text
            if self.prediction is not None:
                cv2.putText(
                    display,
                    f"Prediction: {self.prediction}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            # Add instructions
            cv2.putText(
                display,
                "Press 'c' to clear",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "Press 'q' to quit",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow('Draw a digit (Press "q" to quit)', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                self.drawing = np.zeros(
                    (self.window_size, self.window_size), dtype=np.uint8
                )
                self.prediction = None

        cv2.destroyAllWindows()


def main():
    # Initialize the model and load the best checkpoint
    config = MLPConfig()
    data_manager = DataManager()
    data_manager.prepare_data(recipe=["mnist"], val_split=0.0, batch_size=100)
    model = MNISTClassifier(config)
    checkpoint = Checkpoint(
        run_name="mnist_25_03_24_14_30",
        model=model,
        optimizer=t.optim.AdamW(model.parameters(), lr=config.lr),
    )
    checkpoint.load_checkpoint(load_best=True)

    # Create and run the drawing interface
    interface = DrawingInterface(model, data_manager)
    interface.run()


if __name__ == "__main__":
    main()

# %%
