import torch
import torchvision
import cv2
import numpy as np
from torchvision import transforms
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def prepare_video(frames):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])

    # Chuyển đổi các frame và xếp chúng thành một tensor 5D
    video_tensor = torch.stack([transform(frame) for frame in frames])

    # Chuyển đổi từ (T, C, H, W) sang (C, T, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)

    # Thêm chiều batch
    return video_tensor.unsqueeze(0)

def load_model(model_path):
    model = torchvision.models.video.r3d_18(pretrained=False)
    num_classes = 2  # Thay đổi từ 1 thành 2
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load state dict
    #state_dict = torch.load(model_path, map_location='cpu')
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def predict(model, video_tensor):
    with torch.no_grad():
        output = model(video_tensor)
        probabilities = torch.softmax(output, dim=1)
        pickpocket_probability = probabilities[0, 1].item()  # Xác suất cho lớp "móc túi"
        
        # Nếu xác suất móc túi > 0.60, dự đoán là móc túi
        if pickpocket_probability > 0.60:
            prediction = 1
        else:
            # Nếu không, so sánh xác suất như trước
            prediction = torch.argmax(probabilities, dim=1).item()
        
        # print(f"Raw output: {output.squeeze().tolist()}, Probabilities: {probabilities.squeeze().tolist()}, Prediction: {prediction}")
    return prediction, pickpocket_probability

def process_video(input_path, output_path, model):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_buffer = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_buffer.append(frame)

            if len(frames_buffer) == 16:
                frames_np = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_buffer])
                video_tensor = prepare_video(frames_np)
                if torch.cuda.is_available():
                    video_tensor = video_tensor.cuda()

                prediction, probability = predict(model, video_tensor)
                label = f"Pickpocket ({probability:.2f})" if prediction == 1 else f"Normal ({1-probability:.2f})"

                for f in frames_buffer:
                    cv2.putText(f, label, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255) if prediction == 1 else (0, 255, 0), 2)
                    out.write(f)

                frames_buffer = []  # Reset buffer after processing

        # Xử lý các frame còn lại (nếu có)
        if frames_buffer:
            while len(frames_buffer) < 16:
                frames_buffer.append(frames_buffer[-1])
            frames_np = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_buffer])
            video_tensor = prepare_video(frames_np)
            if torch.cuda.is_available():
                video_tensor = video_tensor.cuda()

            prediction, probability = predict(model, video_tensor)
            label = f"Pickpocket ({probability:.2f})" if prediction == 1 else f"Normal ({1-probability:.2f})"

            for f in frames_buffer:
                cv2.putText(f, label, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255) if prediction == 1 else (0, 255, 0), 2)
                out.write(f)

    finally:
        cap.release()
        out.release()

    print(f"Video processing completed for: {input_path}")
    print(f"Output video saved at: {output_path}")

def process_folder(input_folder, output_folder, model_path):
    model = load_model(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Lấy danh sách các file video trong thư mục đầu vào
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"processed_{video_file}")
        
        print(f"Processing video: {video_file}")
        process_video(input_path, output_path, model)

    print("All videos processed.")

# Sử dụng hàm
input_video_path = './input_test_videos'
output_video_path = './output_test_videos'
model_path = "pickpocket_model_110video_8125.pth"

process_folder(input_video_path, output_video_path, model_path)