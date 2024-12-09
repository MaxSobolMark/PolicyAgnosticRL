# From RoboFuME codebase

import cv2
import click
import numpy as np
import torch
from tqdm import tqdm


def combine_images_into_grid(images, grid_size=(10, 10), fill_value=0):
    num_images = len(images)
    total_cells = grid_size[0] * grid_size[1]
    num_missing_cells = total_cells - (num_images % total_cells)

    if num_missing_cells != total_cells:
        # Append black images to fill missing cells
        for _ in range(num_missing_cells):
            images.append(np.zeros_like(images[0]))

    # Combine images into a grid
    rows = []
    for i in range(0, len(images), grid_size[0]):
        row = np.hstack(images[i : i + grid_size[0]])
        rows.append(row)

    grid_image = np.vstack(rows)
    return grid_image


def mask_image(image, in_min, in_max, out_min=None, out_max=None, color_space="rgb"):
    """Mask an image with inclusive and exclusive color based mask.

    Args:
        image: input image with value range in (0, 255)
        in_min, in_max: min and max allowed pixel value
        out_min, out_max: min and max pixel values to be excluded
        color_space: color space to work on (rgb or hsv)
    """
    assert image.dtype in [int, np.uint8]
    if len(in_min.shape) == 1:
        in_min, in_max = in_min[None], in_max[None]
    if out_min is not None and len(out_min.shape) == 1:
        out_min, out_max = out_min[None], out_max[None]

    if color_space == "hsv":
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    elif color_space == "rgb":
        pass
    else:
        raise ValueError("Color space not found.")

    masks = []
    for imin, imax in zip(in_min, in_max):
        mask = np.zeros_like(image)
        mask[np.logical_and(image >= imin, image <= imax)] = 1
        masks.append(mask.sum(-1) == 3)
    combined_mask = np.clip(np.stack(masks).sum(0), 0, 1)
    if out_min is not None:
        exclude_masks = []
        for imin, imax in zip(out_min, out_max):
            exclude_mask = np.zeros_like(image)
            exclude_mask[np.logical_and(image >= imin, image <= imax)] = 1
            exclude_mask = exclude_mask.sum(-1) == 3
            exclude_masks.append(exclude_mask)
        combined_exclude_mask = np.clip(np.stack(exclude_masks).sum(0), 0, 1)
    if out_max is not None:
        combined_mask = np.logical_and(combined_mask, 1 - combined_exclude_mask)

    return combined_mask[..., None]


def mask_image_by_query(image, query="pot"):
    use_raw_mask = False
    color_space = "hsv"
    in_min, in_max = None, None
    out_min, out_max = None, None
    if query == "orange_pot":
        color_space = "rgb"
        in_min = np.array([110, 29, 10])
        in_max = np.array([170, 70, 50])
    elif query == "pink_plate":
        color_space = "rgb"
        in_min = np.array([114, 30, 31])
        in_max = np.array([204, 96, 96])
        use_raw_mask = True
    else:
        raise ValueError("Unsupported query " + query)

    mask = mask_image(image, in_min, in_max, out_min, out_max, color_space=color_space)
    mask = mask[..., [0, 0, 0]].copy().astype(np.uint8)
    if not use_raw_mask:
        mask = cv2.erode(mask, np.ones((3, 3), "uint8"), iterations=1)
        mask = cv2.dilate(mask, np.ones((3, 3), "uint8"), iterations=1)
    return mask[..., [0]]


class ScriptedDiscriminator(object):
    def __init__(self, task_info):
        self._init_model()
        self.task_id2name = {}
        if task_info is not None:
            self.task_names = task_info["task_names"]
            self.task_ids = task_info["task_ids"]
            self.goal_frames = task_info["goal_frames"]
            for task_id, task_name in zip(self.task_ids, self.task_names):
                str_task_id = ",".join(f"{x:.3f}" for x in task_id[:20])
                self.task_id2name[str_task_id] = task_name

    def _reverse_search_task_names(self, queries):
        assert hasattr(self, "task_ids")
        task_names = [
            self.task_id2name[",".join(f"{x:.3f}" for x in q[:20])] for q in queries
        ]
        return task_names

    def _reverse_search_task_ids(self, queries):
        task_names = self._reverse_search_task_names(queries)
        return [x.replace(" ", "_") for x in task_names]

    def _make_prompts(self, task_names):
        prompts = []
        for task_name in task_names:
            if task_name == "put_orange_pot_on_sink":
                prompt = "orange_pot"
            elif "pink_plate" in task_name:
                prompt = "pink_plate"
            else:
                raise ValueError(f"No preset prompt for task {task_name}")
            prompts.append(prompt)
        return prompts


class CVDiscriminator(ScriptedDiscriminator):
    def _init_model(self):
        pass

    def __call__(self, images, states, vis=False, get_info=False):

        device = images.device
        images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        if torch.is_tensor(states):
            states = states.detach().cpu().numpy()
        assert len(states[0]) == 513
        gripper_states = states[:, 0]
        task_ids = states[:, 1:]
        task_names = self._reverse_search_task_ids(task_ids)
        prompts = self._make_prompts(task_names)
        boxes, vis_boxes, masks = [], [], []
        for query, image in zip(prompts, images):
            mask = mask_image_by_query(image.astype(int), query)
            pts = np.array(np.where(mask[..., 0] > 0)).T
            if len(pts) > 0:
                box_min, box_max = np.min(pts, axis=0), np.max(pts, axis=0)
                boxes.append(np.concatenate([np.mean(pts, axis=0), box_max - box_min]))
                vis_boxes.append(
                    (
                        image.astype(float) - (1 - mask) * image.astype(float) * 0.8
                    ).astype(int)
                )
                masks.append(mask)
            else:
                boxes.append([])
                vis_boxes.append((image.astype(float) * 0.2).astype(int))
                masks.append(np.zeros_like(image[..., [0]]))
        rewards = []
        pos_codes = []
        for task_name, box, mask, grip in zip(task_names, boxes, masks, gripper_states):
            if len(box) == 0:
                reward = 0.0
            if task_name == "put_orange_pot_on_sink":
                # breakpoint()
                # test_image = images[0].copy()
                # test_image[min_height:max_height, min_width:max_width] = (255, 255, 255)
                # from PIL import Image

                # Image.fromarray(test_image).save("test.png")
                sink_region = np.array([[45, 53], [78, 85]])
                pot_in_sink = np.any(
                    [
                        (pt[0] >= sink_region[0][0])
                        and (pt[0] <= sink_region[1][0])
                        and (pt[1] > sink_region[0][1])
                        and (pt[1] < sink_region[1][1])
                        for pt in pts
                    ]
                )
                reward = float(pot_in_sink) * float(grip > 0.5)
            elif task_name == "pink_plate_not_visible":
                assert images.shape[1:] == (224, 224, 3)

                plate_region = np.array([[111, 118], [158, 183]])
                plate_visible = np.any(
                    [
                        (pt[0] >= plate_region[0][0])
                        and (pt[0] <= plate_region[1][0])
                        and (pt[1] > plate_region[0][1])
                        and (pt[1] < plate_region[1][1])
                        for pt in pts
                    ]
                )
                reward = float(not plate_visible) * float(grip > 0.1)
            else:
                raise ValueError(f"No preset prompt for task {task_name}")
            rewards.append(reward)
        if get_info:
            info = dict()
            return (
                torch.tensor(rewards).view(-1, 1).to(device),
                np.array(vis_boxes),
                info,
            )
        else:
            return torch.tensor(rewards).view(-1, 1).to(device), np.array(vis_boxes)
