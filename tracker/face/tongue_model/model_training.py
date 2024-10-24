import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


class AddRandomNoise:
    """Adds random Gaussian noise to a tensor."""

    def __init__(self, mean=0.0, std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size(), dtype=tensor.dtype, device=tensor.device)
        return tensor + noise * self.std + self.mean


class KeypointDataset(Dataset):
    def __init__(
        self,
        image_dir,
        keypoints_dir,
        transform=None,
        output_size=(32, 32),
        num_copies=10,
    ):
        self.image_dir = image_dir
        self.keypoints_dir = keypoints_dir
        self.transform = transform
        self.output_size = output_size
        self.image_paths = [
            os.path.join(image_dir, img) for img in os.listdir(image_dir)
        ] * num_copies
        self.keypoints_paths = [
            os.path.join(keypoints_dir, f"{os.path.splitext(img)[0]}.txt")
            for img in os.listdir(image_dir)
        ] * num_copies

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        keypoints_path = self.keypoints_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints = np.loadtxt(keypoints_path, delimiter=",")[0]
        heatmap = np.zeros(self.output_size)
        if keypoints[0] == -1 or keypoints[1] == -1:
            pass
        else:
            x, y = int(keypoints[0] * self.output_size[1]), int(
                keypoints[1] * self.output_size[0]
            )
            heatmap[y, x] = 100  # Simple binary heatmap
            sigma = 1.5  # Reduced standard deviation
            kernel_size = (7, 7)  # Smaller kernel size
            heatmap = cv2.GaussianBlur(heatmap, kernel_size, sigma)
            # for i in heatmap:
            # print(i)
            # exit()
        # image = cv2.resize(image, self.output_size) / 255.0
        image = image / 255.0

        if self.transform:
            image = self.transform(image)

        # Correctly format the tensors
        image_tensor = torch.tensor(
            image, dtype=torch.float32
        )  # Shape should be [1, H, W]
        heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(
            0
        )  # Shape should be [1, H, W]

        return image_tensor, heatmap_tensor


class KeypointCNN(nn.Module):
    def __init__(self, num_keypoints=1):
        super(KeypointCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(
            64 * 8 * 8, 256
        )  # Adjust the input size according to pooling and input dimensions

        # Keypoint heatmaps output layer: Outputs heatmaps instead of direct coordinates
        # Assuming the output size to be 32x32, which should be adjusted according to your needs
        self.heatmap_conv = nn.Conv2d(64, num_keypoints, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False)

        self.fc_classifier = nn.Linear(256, 1)  # Classification layer

    def forward(self, x):
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        x3 = F.relu(self.bn3(self.conv3(x2)))

        # Keypoint heatmap output
        keypoints_heatmap = self.upsample(self.heatmap_conv(x3))

        x_flat = x3.view(x3.size(0), -1)
        features = F.relu(self.fc1(x_flat))

        classification = torch.sigmoid(self.fc_classifier(features))
        return keypoints_heatmap, classification


def train(
    model,
    data_loader,
    optimizer,
    criterion_regression,
    criterion_classification,
    epochs,
    device,
):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, heatmaps in data_loader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            out_keypoints, out_classification = model(images)
            labels = (
                (heatmaps.sum(dim=[1, 2, 3]) > 0).float().reshape(-1, 1)
            )  # Assuming heatmap dimensions are [B, 1, H, W]
            # out_keypoints = out_keypoints*out_classification.reshape(-1,1,1,1)
            loss_1 = criterion_regression(
                out_keypoints * out_classification.reshape(-1, 1, 1, 1), heatmaps
            )
            loss_2 = criterion_classification(out_classification, labels)
            # binary_reg = torch.mean(out_classification * (1 - out_classification))
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(data_loader)}")
        # Save model after each epoch
        torch.save(model.state_dict(), f"./model_output_3/model_epoch_{epoch+1}.pth")


def max_average_point(heatmap, window_size=3):
    # 对热图应用均匀滤波
    filtered_heatmap = uniform_filter(heatmap, size=window_size)
    # 找到均值最大的点
    best_point = np.unravel_index(np.argmax(filtered_heatmap, axis=None), heatmap.shape)
    return best_point


def test(model, data_loader, device, output_folder="./result_3", window_size=3):
    model.eval()
    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        total_loss = 0.0
        for batch_idx, (images, heatmaps) in enumerate(data_loader):
            images, heatmaps = images.to(device), heatmaps.to(device)
            out_keypoints, out_classification = model(images)
            outputs = out_keypoints * out_classification.reshape(-1, 1, 1, 1)
            loss = F.mse_loss(outputs, heatmaps)
            total_loss += loss.item()

            for i in range(len(images)):  # Save up to 5 images per batch
                p = out_classification[i].item()
                fig, axs = plt.subplots(
                    1, 4, figsize=(16, 4)
                )  # Increase subplot for max point visualization

                # Show original image
                original_image = images[i].squeeze().cpu().numpy()
                axs[0].imshow(original_image, cmap="gray")
                axs[0].set_title("Original Image")

                # Show ground truth heatmap
                ground_truth = heatmaps[i].squeeze().cpu().numpy()
                axs[1].imshow(ground_truth, cmap="jet")
                axs[1].set_title("Ground Truth Heatmap")

                # Show predicted heatmap
                predicted_heatmap = outputs[i].squeeze().cpu().numpy()
                axs[2].imshow(predicted_heatmap, cmap="jet")
                axs[2].set_title(f"Predicted Heatmap P={p}")

                # Find and plot the maximum average point
                max_point = max_average_point(predicted_heatmap, window_size)
                axs[3].imshow(original_image, cmap="gray")
                axs[3].scatter(
                    max_point[1], max_point[0], color="red", s=50
                )  # Plot on original image
                axs[3].set_title("Max Average Point")

                for ax in axs:
                    ax.axis("off")

                plt.savefig(
                    os.path.join(output_folder, f"batch_{batch_idx}_image_{i}.png")
                )
                plt.close(fig)

        print(f"Test Loss: {total_loss / len(data_loader)}")


def main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            AddRandomNoise(0.0, 0.01),
        ]
    )
    no_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = KeypointDataset(
        "./train_dataset/img", "./train_dataset/lab", transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = KeypointDataset(
        "./test_dataset/img", "./test_dataset/lab", no_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_regression = nn.MSELoss()
    criterion_classification = nn.BCELoss()

    train(
        model,
        train_loader,
        optimizer,
        criterion_regression,
        criterion_classification,
        epochs=200,
        device=device,
    )
    model.load_state_dict(torch.load("./model_output_3/model_epoch_200.pth"))
    test(model, test_loader, device)


if __name__ == "__main__":
    main()
