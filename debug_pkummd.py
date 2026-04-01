import os
import random
import numpy as np

data_dir = 'C:/Users/user/Desktop/PKUMMD'
data_path = os.path.join(data_dir, 'Data')
label_path = os.path.join(data_dir, 'Label')
num_frames = 16
num_nodes = 25
in_channels = 3
stride = num_frames // 2

split_path = os.path.join(data_dir, 'Split', 'cross-subject.txt')
with open(split_path, 'r') as f:
    content = f.read()

train_part = content.split("Training videos:")[1].split("Validataion videos:")[0]
file_list = [f.strip() + ".txt" for f in train_part.split(",") if f.strip()]  # 전체

total_action_frames = 0
total_frames = 0
action_windows_with_all_bg = 0
total_action_windows = 0

for file_name in file_list:
    skeleton_file = os.path.join(data_path, file_name)
    lfile = os.path.join(label_path, file_name)

    if not os.path.exists(skeleton_file):
        continue

    data = np.loadtxt(skeleton_file).astype(np.float32)
    if data.ndim == 1:
        data = data[None, :]
    T = int(data.shape[0])

    labels_p1 = np.zeros(T, dtype=np.int64)
    labels_p2 = np.zeros(T, dtype=np.int64)

    if os.path.exists(lfile):
        with open(lfile, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 4:
                    continue
                cls, start, end, pid = map(int, parts)
                start_f = max(start - 1, 0)
                end_f = min(end, T)
                if pid == 1:
                    labels_p1[start_f:end_f] = cls
                elif pid == 2:
                    labels_p2[start_f:end_f] = cls

    # person 선택
    candidates = []
    if labels_p1.any(): candidates.append(0)
    if labels_p2.any(): candidates.append(1)
    person = candidates[0] if candidates else 0
    labels = labels_p1 if person == 0 else labels_p2

    total_frames += T
    total_action_frames += (labels > 0).sum()

    # window 확인
    for start in range(0, T - num_frames + 1, stride):
        clip_labels = labels[start:start + num_frames]
        if (clip_labels > 0).any():
            total_action_windows += 1
            action_ratio = (clip_labels > 0).sum() / num_frames
            if action_ratio < 0.1:  # action이 10% 미만인 window
                action_windows_with_all_bg += 1
            if total_action_windows <= 3:  # 처음 3개 출력
                print(f"[{file_name}] start={start} | clip_labels: {clip_labels}")
                print(f"  action frames: {(clip_labels > 0).sum()}/{num_frames}")

print(f"\n=== 요약 ===")
print(f"전체 프레임: {total_frames} | action: {total_action_frames} ({total_action_frames/total_frames*100:.1f}%)")
print(f"action windows: {total_action_windows}")
print(f"action window 중 action 비율 10% 미만: {action_windows_with_all_bg} ({action_windows_with_all_bg/max(total_action_windows,1)*100:.1f}%)")