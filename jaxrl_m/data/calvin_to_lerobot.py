import typer
from typing import List, Dict
from pathlib import Path
import shutil
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from tqdm import tqdm
from scripts.tfrecord_inspection import tfrecord_to_dict
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    get_default_encoding,
)
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from lerobot.common.datasets.push_dataset_to_hub.robobuf_hdf5_format import (
    to_hf_dataset,
)
from lerobot.common.datasets.utils import calculate_episode_data_index, create_branch

PROTO_TYPE_SPEC = {
    "observations/images0": tf.uint8,
    "observations/state": tf.float32,
    # "next_observations/images0": tf.uint8,
    # "next_observations/state": tf.float32,
    "actions": tf.float32,
    # "mc_returns": tf.float32,
    "rewards": tf.float32,
}

app = typer.Typer(pretty_exceptions_show_locals=False)


def save_image(
    image: np.ndarray,
    frame_index: int,
    episode_index: int,
    videos_dir: Path,
    key: str = "observation.image",
):
    img = Image.fromarray(image)
    path = (
        videos_dir
        / f"{key}_episode_{episode_index:06d}"
        / f"frame_{frame_index:06d}.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


@app.command()
def calvin_to_lerobot(
    calvin_tfrecord_regexp: str,
    output_path: str,
    fps: int = 30,
    debug: bool = False,
):
    calvin_tfrecord_paths = tf.io.gfile.glob(calvin_tfrecord_regexp)
    if debug:
        calvin_tfrecord_paths = calvin_tfrecord_paths[:10]
    trajectories: List[Dict[str, np.ndarray]] = []
    print(f"Found {len(calvin_tfrecord_paths)} tfrecords. Starting to parse them.")
    for calvin_tfrecord_path in tqdm(calvin_tfrecord_paths):
        parsed_tensors = tfrecord_to_dict(calvin_tfrecord_path)
        trajectories.append(parsed_tensors)

    print(f"Finished parsing {len(calvin_tfrecord_paths)} tfrecords.")

    output_path = Path(output_path)
    assert not output_path.exists(), f"{output_path} already exists"

    episodes_path = output_path / "episodes"
    episodes_path.mkdir(parents=True, exist_ok=True)

    videos_dir = output_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    print("Processing trajectories.")
    for trajectoy_index, trajectory in tqdm(enumerate(trajectories)):
        episode_dict = {
            "observation.state": [],
            "action": [],
        }

        for frame_index in range(trajectory["rewards"].shape[0]):
            image = trajectory["observations/images0"][frame_index].numpy().copy()
            save_image(
                image=image,
                frame_index=frame_index,
                episode_index=trajectoy_index,
                videos_dir=videos_dir,
            )
            proprio = torch.tensor(
                trajectory["observations/state"][frame_index].numpy().copy()
            )
            episode_dict["observation.state"].append(proprio)
            action = torch.tensor(trajectory["actions"][frame_index].numpy().copy())
            episode_dict["action"].append(action)

        episode_dict["observation.state"] = torch.stack(
            episode_dict["observation.state"]
        )
        episode_dict["action"] = torch.stack(episode_dict["action"])

        num_frames = episode_dict["observation.state"].shape[0]

        filename = f"observation.image_episode_{trajectoy_index:06d}.mp4"
        video_path = videos_dir / filename
        if video_path.exists():
            video_path.unlink()
        episode_dict["observation.image"] = []
        for frame_index in range(num_frames):
            episode_dict["observation.image"].append(
                {"path": f"videos/{filename}", "timestamp": frame_index / fps}
            )

        episode_dict["episode_index"] = torch.tensor([trajectoy_index] * num_frames)
        episode_dict["frame_index"] = torch.arange(num_frames)
        episode_dict["timestamp"] = torch.arange(num_frames) / fps
        episode_dict["next.done"] = torch.zeros(num_frames, dtype=torch.bool)
        episode_dict["next.done"][-1] = True

        episode_path = episodes_path / f"episode_{trajectoy_index}.pth"
        torch.save(episode_dict, episode_path)

    print("Finished processing trajectories.")
    print("Encoding videos...")

    # Use ffmpeg to convert frames stored as png into mp4 videos
    for episode_index in tqdm(range(len(trajectories))):
        tmp_images_dir = videos_dir / f"observation.image_episode_{episode_index:06d}"
        filename = f"observation.image_episode_{episode_index:06d}.mp4"
        video_path = videos_dir / filename
        if video_path.exists():
            continue
        encode_video_frames(tmp_images_dir, video_path, fps=fps)
        shutil.rmtree(tmp_images_dir)

    print("Finished encoding videos.")
    print("Concatenating episodes...")
    episode_dicts = []
    for episode_index in tqdm(range(len(trajectories))):
        episode_path = episodes_path / f"episode_{episode_index}.pth"
        episode_dict = torch.load(episode_path)
        episode_dicts.append(episode_dict)

    data_dict = concatenate_episodes(episode_dicts)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    hf_dataset = to_hf_dataset(data_dict, True)
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": True,
    }
    info["encoding"] = get_default_encoding()
    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id="lerobot/calvin",
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )

    print("Computing dataset statistics")
    stats = compute_stats(lerobot_dataset)
    lerobot_dataset.stats = stats

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(output_path / "train"))

    metadata_dir = output_path / "meta_data"
    save_meta_data(info, stats, episode_data_index, metadata_dir)

    print("Finished.")

    return lerobot_dataset


if __name__ == "__main__":
    app()
