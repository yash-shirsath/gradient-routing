import cv2
import numpy as np
import torch
from mnist import MNISTConfig, MNISTClassifier, ModelManager, DataManager
from torchvision import transforms
import time
import matplotlib.pyplot as plt


class DrawingInterface:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.model_manager.model.eval()

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
        self.prediction_interval = 0.1  # Update prediction every 100ms

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
            if current_time - self.last_prediction_time >= self.prediction_interval:
                # self.update_prediction()
                self.last_prediction_time = current_time
        elif event == cv2.EVENT_LBUTTONUP:
            self.last_point = None
            # Always update prediction when mouse is released
            self.update_prediction()

    def preprocess_drawing(self):
        # Resize to MNIST size (28x28)
        resized = cv2.resize(self.drawing, (28, 28), interpolation=cv2.INTER_AREA)
        # Convert to PIL Image for consistent preprocessing
        from PIL import Image

        pil_image = Image.fromarray(resized)
        # Display PIL image for debugging
        pil_image.show()
        # Apply the same transform as training
        tensor = self.transform(pil_image)
        # Ensure we have a torch tensor and add batch dimension
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor)
        tensor = tensor.unsqueeze(0)
        return tensor

    def update_prediction(self):
        with torch.no_grad():
            input_tensor = self.preprocess_drawing()

            output = self.model_manager.model(input_tensor)
            prediction = output.argmax(dim=1).item()
            self.prediction = prediction
            self.compare_to_test()

    def compare_to_test(self):
        self.model_manager.data_manager.load_mnist()
        test_loader = self.model_manager.data_manager.test_loader
        assert test_loader is not None
        x, y = next(iter(test_loader))
        # Create a figure with subplots - images on top, distributions below
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Plot preprocessed drawing
        input_tensor = self.preprocess_drawing()
        with torch.no_grad():
            pred1 = self.model_manager.model(input_tensor).argmax(dim=1).item()
        ax1.imshow(input_tensor.squeeze(), cmap="gray")
        ax1.set_title(f"Your Drawing (Prediction: {pred1})")
        ax1.axis("off")

        # Plot first test image
        with torch.no_grad():
            pred2 = self.model_manager.model(x[0:1]).argmax(dim=1).item()
        ax2.imshow(x[0].squeeze(), cmap="gray")
        ax2.set_title(f"Test Image (Label: {y[0].item()}, Prediction: {pred2})")
        ax2.axis("off")

        # Plot pixel intensity distributions
        drawing_pixels = input_tensor.squeeze().numpy().flatten()
        test_pixels = x[0].squeeze().numpy().flatten()

        # Drawing distribution
        ax3.hist(drawing_pixels, bins=50, alpha=0.7, color="blue")
        drawing_mean = drawing_pixels.mean()
        drawing_median = np.median(drawing_pixels)
        drawing_std = drawing_pixels.std()
        ax3.axvline(
            drawing_mean,
            color="red",
            linestyle="dashed",
            label=f"Mean: {drawing_mean:.2f}",
        )
        ax3.axvline(
            drawing_median,
            color="green",
            linestyle="dashed",
            label=f"Median: {drawing_median:.2f}",
        )
        ax3.text(0.02, 0.95, f"Std Dev: {drawing_std:.2f}", transform=ax3.transAxes)
        ax3.set_title("Your Drawing Pixel Distribution")
        ax3.legend()

        # Test image distribution
        ax4.hist(test_pixels, bins=50, alpha=0.7, color="blue")
        test_mean = test_pixels.mean()
        test_median = np.median(test_pixels)
        test_std = test_pixels.std()
        ax4.axvline(
            test_mean, color="red", linestyle="dashed", label=f"Mean: {test_mean:.2f}"
        )
        ax4.axvline(
            test_median,
            color="green",
            linestyle="dashed",
            label=f"Median: {test_median:.2f}",
        )
        ax4.text(0.02, 0.95, f"Std Dev: {test_std:.2f}", transform=ax4.transAxes)
        ax4.set_title("Test Image Pixel Distribution")
        ax4.legend()

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
    config = MNISTConfig()
    data_manager = DataManager(config)
    model = MNISTClassifier(config)
    model_manager = ModelManager(config, data_manager, model)
    model_manager.load_checkpoint(load_best=True)

    # Create and run the drawing interface
    interface = DrawingInterface(model_manager)
    interface.run()


if __name__ == "__main__":
    main()
