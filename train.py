import torch
import torchvision
import cv2
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import random

# Bước 4: Định nghĩa hàm để đọc và xử lý video
def load_video(video_path, target_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Đảm bảo số lượng frame là bội số của 8 (yêu cầu của R3D_18)
    target_frames = (target_frames // 8) * 8

    # Lấy mẫu hoặc lặp lại các frame để đạt được số lượng frame mục tiêu
    if len(frames) > target_frames:
        step = len(frames) // target_frames
        frames = frames[::step][:target_frames]
    else:
        frames = frames + [frames[-1]] * (target_frames - len(frames))

    # Chuyển đổi kích thước frame thành 112x112 (yêu cầu của R3D_18)
    frames = [cv2.resize(frame, (112, 112)) for frame in frames]

    return np.array(frames)

# Hàm mới để điều chỉnh độ sáng của video
def adjust_brightness(video, factor):
    return np.clip(video * factor, 0, 255).astype(np.uint8)

# Hàm mới để tạo các phiên bản video với độ sáng khác nhau
def create_brightness_variations(video_path, output_dir):
    video = load_video(video_path)
    
    # Tạo phiên bản tối hơn
    darker = adjust_brightness(video, 0.7)
    
    # Tạo phiên bản sáng hơn
    brighter = adjust_brightness(video, 1.3)
    
    # Lưu các phiên bản mới
    filename = os.path.basename(video_path)
    base_name, _ = os.path.splitext(filename)
    
    for i, (dark_frame, bright_frame) in enumerate(zip(darker, brighter)):
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_dark_{i:03d}.jpg"), cv2.cvtColor(dark_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_bright_{i:03d}.jpg"), cv2.cvtColor(bright_frame, cv2.COLOR_RGB2BGR))

# Bước 3: Định nghĩa các đường dẫn đến dữ liệu
data_dir = './pickpocket_dataset'  # Thay đổi đường dẫn này theo thư mục của bạn
pickpocket_dir = os.path.join(data_dir, 'pickpocket')
non_pickpocket_dir = os.path.join(data_dir, 'non_pickpocket')

# Tạo thư mục mới để lưu các phiên bản video đã được tăng cường
augmented_pickpocket_dir = os.path.join(data_dir, 'augmented_pickpocket')
augmented_non_pickpocket_dir = os.path.join(data_dir, 'augmented_non_pickpocket')

os.makedirs(augmented_pickpocket_dir, exist_ok=True)
os.makedirs(augmented_non_pickpocket_dir, exist_ok=True)

# Tạo các phiên bản video với độ sáng khác nhau
print("Creating brightness variations for pickpocket videos...")
for video in tqdm(os.listdir(pickpocket_dir)):
    create_brightness_variations(os.path.join(pickpocket_dir, video), augmented_pickpocket_dir)

print("Creating brightness variations for non-pickpocket videos...")
for video in tqdm(os.listdir(non_pickpocket_dir)):
    create_brightness_variations(os.path.join(non_pickpocket_dir, video), augmented_non_pickpocket_dir)

# Bước 5: Định nghĩa Dataset
class PickpocketDataset(Dataset):
    def __init__(self, pickpocket_dir, non_pickpocket_dir, augmented_pickpocket_dir, augmented_non_pickpocket_dir, transform=None):
        self.pickpocket_videos = [os.path.join(pickpocket_dir, f) for f in os.listdir(pickpocket_dir)] + \
                                 [os.path.join(augmented_pickpocket_dir, f) for f in os.listdir(augmented_pickpocket_dir)]
        self.non_pickpocket_videos = [os.path.join(non_pickpocket_dir, f) for f in os.listdir(non_pickpocket_dir)] + \
                                     [os.path.join(augmented_non_pickpocket_dir, f) for f in os.listdir(augmented_non_pickpocket_dir)]
        self.all_videos = self.pickpocket_videos + self.non_pickpocket_videos
        self.labels = [1] * len(self.pickpocket_videos) + [0] * len(self.non_pickpocket_videos)
        self.transform = transform

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        video_path = self.all_videos[idx]
        label = self.labels[idx]

        frames = load_video(video_path)

        if self.transform:
            # Áp dụng transform cho từng frame
            transformed_frames = []
            for frame in frames:
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)

            # Ghép các frame đã được transform lại thành một tensor
            frames = torch.stack(transformed_frames)

        # Chuyển đổi thứ tự các chiều từ (T, C, H, W) sang (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)

        return frames, label

# Bước 6: Định nghĩa các transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
])

# Bước 7: Tạo dataset và dataloader
dataset = PickpocketDataset(pickpocket_dir, non_pickpocket_dir, augmented_pickpocket_dir, augmented_non_pickpocket_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Bước 8: Tải mô hình R3D_18 pretrained
def load_pretrained_r3d():
    model = torchvision.models.video.r3d_18(pretrained=True)
    # Thay đổi lớp cuối cùng cho binary classification
    model.fc = nn.Linear(in_features=512, out_features=2)
    return model

model = load_pretrained_r3d()
model = model.cuda()  # Chuyển mô hình sang GPU

# Bước 9: Định nghĩa loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Bước 10: Huấn luyện mô hình với early stopping
num_epochs = 50
patience = 5
best_val_acc = 0
counter = 0
best_model = None

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for videos, labels in tqdm(train_loader):
        videos = videos.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * videos.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.cuda()
            labels = labels.cuda()

            outputs = model(videos)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        best_model = model.state_dict()
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# Bước 11: Lưu mô hình tốt nhất
if best_model is not None:
    torch.save(best_model, 'pickpocket_model.pth')
    print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")
else:
    torch.save(model.state_dict(), 'pickpocket_model.pth')
    print("Final model saved")
