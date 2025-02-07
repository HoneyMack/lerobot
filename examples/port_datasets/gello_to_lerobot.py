#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gello 形式の pickle ファイル群から，LeRobotDatasetV2 形式に変換するサンプルコード
"""

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image

# LeRobotDatasetV2 の作成用 API（zarr 形式の変換例を参考）
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import encode_video_frames, save_images_concurrently

# ここでは，新たなキー構成に合わせた features を定義
PUSHT_TASK = "Pick up the red square block and Put it onto the white plate."
GELLO_FEATURES = {
    "observation.images.base.rgb": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 10.0,
            "video.codec": "libx265",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": "false",
            "has_audio": "false",
        },
    },
    # "observation.images.base.depth": {
    #     "dtype": "video",
    #     "shape": (256, 256),  # 1 チャンネル画像
    #     "names": ["height", "width"],
    #     "video_info": {
    #         "video.fps": 10.0,
    #         "video.codec": "libx265",
    #         "video.pix_fmt": "gray",
    #         "video.is_depth_map": "true",
    #         "has_audio": "false",
    #     },
    # },
    "observation.images.wrist.rgb": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 10.0,
            "video.codec": "libx265",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": "false",
            "has_audio": "false",
        },
    },
    # "observation.images.wrist.depth": {
    #     "dtype": "video",
    #     "shape": (256, 256),
    #     "names": ["height", "width"],
    #     "video_info": {
    #         "video.fps": 10.0,
    #         "video.codec": "libx265",
    #         "video.pix_fmt": "gray",
    #         "video.is_depth_map": "true",
    #         "has_audio": "false",
    #     },
    # },
    "observation.state": {
        "dtype": "float32",
        "shape": (22,),  # 7 (joint_positions) + 7 (joint_velocities) + 7 (ee_pos_quat) + 1 (gripper_position)
        "names": {
            "joint_positions": [f"joint_pos_{i}" for i in range(7)],
            "joint_velocities": [f"joint_vel_{i}" for i in range(7)],
            "ee_pos_quat": [f"ee_pos_quat_{i}" for i in range(7)],
            "gripper_position": ["gripper_position"],
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),  # control (7 要素)
        "names": [f"control_{i}" for i in range(7)],
    },
    "next.done": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.success": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "episode_index": {
        "dtype": "int32",
        "shape": (1,),
        "names": None,
    },
}


def resize_image(np_array, size=(256, 256)):
    """PIL を使って画像をリサイズする（RGB 向け）"""
    img = Image.fromarray(np_array)
    img = img.resize(size, Image.Resampling.BICUBIC)
    return np.array(img)


def process_depth_image(np_array, size=(256, 256)):
    """
    深度画像 (uint16, shape=(H, W, 1)) を，
    1 チャンネルにして2で割って正規化し，uint8 化した上でリサイズする
    """
    # 1 チャンネル抽出
    depth = np_array[..., 0]
    # 2で割って正規化
    depth = (depth / 2).astype(np.uint8)

    # リサイズ
    img = Image.fromarray(depth)
    img = img.resize(size, Image.Resampling.BICUBIC)
    return np.array(img)


def load_gello_episode(episode_path: Path, fps: int, video: bool, out_dir: Path):
    """
    指定されたエピソードディレクトリ内の pickle ファイルを読み込み，
    各ステップのデータを前処理して，フレームごとの辞書リストとして返す
    """
    step_paths = sorted(list(episode_path.glob("*.pkl")))
    frames = []
    for step_idx, step_path in enumerate(step_paths):
        with open(step_path, "rb") as f:
            data = pickle.load(f)

        frame = {}
        # --- 画像データの変換 ---
        if "base_rgb" in data:
            # base カメラ RGB 画像 → observation.images.base.rgb
            rgb = data["base_rgb"]
            rgb = resize_image(rgb, (256, 256))
            frame["observation.images.base.rgb"] = Image.fromarray(rgb)
        # if "base_depth" in data:
        #     depth = data["base_depth"]
        #     depth = process_depth_image(depth, (256, 256))
        #     frame["observation.images.base.depth"] = Image.fromarray(depth)
        if "wrist_rgb" in data:
            rgb = data["wrist_rgb"]
            rgb = resize_image(rgb, (256, 256))
            frame["observation.images.wrist.rgb"] = Image.fromarray(rgb)
        # if "wrist_depth" in data:
        #     depth = data["wrist_depth"]
        #     depth = process_depth_image(depth, (256, 256))
        #     frame["observation.images.wrist.depth"] = Image.fromarray(depth)

        # --- 行動データ ---
        if "control" in data:
            frame["action"] = torch.tensor(data["control"], dtype=torch.float32)

        # --- 状態データの構築 ---
        # state を joint_positions, joint_velocities, ee_pos_quat, gripper_position の連結とする
        state_parts = []
        for key in ["joint_positions", "joint_velocities", "ee_pos_quat"]:
            if key in data:
                arr = np.array(data[key])
                state_parts.append(torch.tensor(arr, dtype=torch.float32))
            else:
                # 要素数 7 のゼロベクトル
                state_parts.append(torch.zeros(7, dtype=torch.float32))
        # gripper_position はスカラー
        if "gripper_position" in data:
            gp = torch.tensor(np.array([data["gripper_position"]]), dtype=torch.float32)
        else:
            gp = torch.zeros(1, dtype=torch.float32)
        state_parts.append(gp)
        # 連結して (7+7+7+1)=22 次元とする
        frame["observation.state"] = torch.cat(state_parts, dim=0)

        # --- 補助情報 ---
        # frame["frame_index"] = torch.tensor(step_idx, dtype=torch.int64)
        # frame["index"] = torch.tensor(step_idx, dtype=torch.int64)
        # frame["timestamp"] = torch.tensor(step_idx / fps, dtype=torch.float32)
        # 最終フレームのみ done=True とする
        is_last = step_idx == len(step_paths) - 1
        frame["next.done"] = torch.tensor(is_last, dtype=torch.bool)
        frame["next.success"] = torch.tensor(is_last, dtype=torch.bool)
        # 報酬が存在しない場合は 0.0 を設定
        frame["next.reward"] = torch.tensor(0.0, dtype=torch.float32)

        frames.append(frame)
    return frames


def convert_gello_to_lerobot_format(
    raw_dir: Path, out_dir: Path, fps: int = 10, video: bool = True, debug: bool = False
):
    """
    gello 形式のデータ（pickle ファイル群）を読み込み，
    LeRobotDatasetV2 形式に変換して out_dir に保存する関数
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    # LeRobotDatasetV2 のインスタンスを作成
    dataset = LeRobotDataset.create(
        repo_id="gello_dataset",
        fps=fps,
        root=out_dir,
        robot_type="lite6",
        features=GELLO_FEATURES,
        tolerance_s=0.5,
        image_writer_processes=4,
        image_writer_threads=4,
    )

    # raw_dir 内の各エピソードフォルダを処理
    episode_paths = sorted([p for p in raw_dir.glob("*") if p.is_dir()])
    for episode_idx, episode_path in enumerate(episode_paths):
        print(f"[INFO]: Processing episode {episode_path}")
        frames = load_gello_episode(episode_path, fps, video, out_dir)
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode(task=PUSHT_TASK)
        if debug:
            break

    # 最終的にデータセットを確定（不要な画像ファイルは削除）
    dataset.consolidate(keep_image_files=True)
    print(f"[INFO]: Dataset saved to {out_dir}")
    return dataset

# python examples/port_datasets/gello_to_lerobot.py --raw-dir ./data/gello_datasets/lite6 --out-dir ./data/converted_datasets/lite6 --fps 10 --video
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert gello dataset (pickle files) to LeRobotDatasetV2 format")
    parser.add_argument("--raw-dir", type=str, required=True, help="gello データが格納されているディレクトリ")
    parser.add_argument("--out-dir", type=str, required=True, help="変換後の LeRobotDataset を保存するディレクトリ")
    parser.add_argument("--fps", type=int, default=20, help="フレームレート")
    parser.add_argument("--video", action="store_true", help="画像データを動画形式としてエンコードする場合に指定")
    parser.add_argument("--debug", action="store_true", help="デバッグモード（1 エピソードのみ処理）")

    args = parser.parse_args()
    convert_gello_to_lerobot_format(
        Path(args.raw_dir), Path(args.out_dir), fps=args.fps, video=args.video, debug=args.debug
    )
