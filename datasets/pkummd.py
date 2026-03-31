import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset

class PKUMMD(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.data_dir = os.path.normpath(config.data_dir)
        self.num_frames = int(config.num_frames)   # 윈도우 크기 (e.g., 32 or 64)
        self.num_nodes = int(config.num_nodes)     # 25
        self.num_classes = int(config.num_classes)
        self.in_channels = int(config.in_channels) # 6 (xyz + vel)

        self.data_path = os.path.join(self.data_dir, "Data")
        self.label_path = os.path.join(self.data_dir, "Label")

        # Split 설정
        split_path = os.path.join(self.data_dir, "Split", "cross-subject.txt")
        with open(split_path, "r") as f:
            content = f.read()

        train_part = content.split("Training videos:")[1].split("Validataion videos:")[0]
        val_part = content.split("Validataion videos:")[1]
        raw = train_part if split == "train" else val_part
        file_list = [f.strip() + ".txt" for f in raw.split(",") if f.strip()]

        # ✅ [수정] OAD 프로토콜용 Stride 설정
        # Test/Val은 매 프레임 예측해야 하므로 1, Train은 효율을 위해 조절
        if self.split == "train":
            self.stride = max(1, self.num_frames // 8) 
        else:
            self.stride = 1

        self._data = []
        self._labels = []
        self._lengths = []
        self._files = []

        for file_name in file_list:
            skeleton_file = os.path.join(self.data_path, file_name)
            label_file = os.path.join(self.label_path, file_name)

            if not os.path.exists(skeleton_file):
                continue

            data = np.loadtxt(skeleton_file).astype(np.float32)
            if data.ndim == 1:
                data = data[None, :]
            
            T = int(data.shape[0])
            # (T, 150) -> (T, 2, 25, 3)
            data = data.reshape(T, 2, self.num_nodes, 3)

            labels = self._load_labels(label_file, T)

            # ✅ 1명 선택 (Main Subject Selection)
            person = self._select_active_person(data)
            clip_data = data[:, person]  # (T, 25, 3)

            self._data.append(clip_data)
            self._labels.append(labels)
            self._lengths.append(T)
            self._files.append(file_name)

        # 윈도우 빌드
        self.windows = []
        action_count = 0
        bg_count = 0

        for file_id, T in enumerate(self._lengths):
            if T >= self.num_frames:
                for start in range(0, T - self.num_frames + 1, self.stride):
                    # ✅ 마지막 프레임의 라벨 기준 분류
                    last_label = self._labels[file_id][start + self.num_frames - 1]
                    self.windows.append((file_id, start))
                    if last_label > 0: action_count += 1
                    else: bg_count += 1
            else:
                # 데이터가 윈도우보다 짧을 경우 (0번 인덱스부터 시작)
                self.windows.append((file_id, 0))
                last_label = self._labels[file_id][-1]
                if last_label > 0: action_count += 1
                else: bg_count += 1

        if self.split == "train":
            random.shuffle(self.windows)

        print(f"[DEBUG] PKUMMD {split} | Windows: {len(self.windows)} (Action: {action_count}, BG: {bg_count}) | Stride: {self.stride}")

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def _select_active_person(data):
        motion_scores = []
        for m in range(2):
            person = data[:, m]
            valid = np.linalg.norm(person, axis=-1) > 1e-6
            motion = np.diff(person, axis=0)
            motion = np.linalg.norm(motion, axis=-1)
            motion = motion * valid[1:]
            motion_scores.append(motion.sum())
        return int(np.argmax(motion_scores))

    def _load_labels(self, label_file, T):
        labels = np.zeros(T, dtype=np.int64)
        if not os.path.exists(label_file):
            return labels
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 4: continue
                cls, start, end, _ = map(int, parts)
                start_f = max(start - 1, 0)
                end_f = min(end, T)
                labels[start_f:end_f] = cls
        return labels

    @staticmethod
    def _normalize_skeleton(clip):
        # SpineBase(0) 기준 원점 이동
        spine_base = clip[:, 0:1, :]
        clip = clip - spine_base
        # 어깨 너비 기준 스케일링
        shoulder_vec = clip[:, 5, :] - clip[:, 9, :]
        shoulder_dist = np.linalg.norm(shoulder_vec, axis=1)
        mean_dist = shoulder_dist.mean() + 1e-6
        clip = clip / mean_dist
        return clip

    @staticmethod
    def _compute_velocity(clip):
        vel = np.zeros_like(clip)
        vel[1:] = clip[1:] - clip[:-1]
        return vel

    def __getitem__(self, idx):
        file_id, start_idx = self.windows[idx]
        data = self._data[file_id]
        labels = self._labels[file_id]
        T = self._lengths[file_id]

        # 윈도우 슬라이싱 및 패딩
        if T >= self.num_frames:
            clip = data[start_idx:start_idx + self.num_frames]
            clip_labels = labels[start_idx:start_idx + self.num_frames]
        else:
            pad_len = self.num_frames - T
            clip = np.concatenate(
                [data, np.zeros((pad_len, self.num_nodes, 3), dtype=np.float32)],
                axis=0
            )
            # 패딩 부분은 배경(0)으로 채움
            clip_labels = np.concatenate(
                [labels, np.zeros((pad_len,), dtype=np.int64)],
                axis=0
            )

        # 전처리
        clip = self._normalize_skeleton(clip)
        vel = self._compute_velocity(clip)
        
        # Concat (T, N, 3) + (T, N, 3) -> (T, N, 6)
        clip = np.concatenate([clip, vel], axis=-1)
        
        # (T, N, 6) -> (T, 6, N)
        clip = torch.from_numpy(clip).permute(0, 2, 1).contiguous()
        
        # ✅ [핵심 수정] 마지막 프레임의 라벨만 Scalar로 반환
        target_label = int(clip_labels[-1])
        target_label = torch.tensor(target_label, dtype=torch.long)

        return clip, target_label