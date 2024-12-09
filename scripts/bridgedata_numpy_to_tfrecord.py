"""
Converts data from the BridgeData numpy format to TFRecord format.

Consider the following directory structure for the input data:

    bridgedata_numpy/
        rss/
            toykitchen2/
                set_table/
                    00/
                        train/
                            out.npy
                        val/
                            out.npy
        icra/
            ...

The --depth parameter controls how much of the data to process at the
--input_path; for example, if --depth=5, then --input_path should be
"bridgedata_numpy", and all data will be processed. If --depth=3, then
--input_path should be "bridgedata_numpy/rss/toykitchen2", and only data
under "toykitchen2" will be processed.

The same directory structure will be replicated under --output_path.  For
example, in the second case, the output will be written to
"{output_path}/set_table/00/...".

Can read/write directly from/to Google Cloud Storage.

Written by Kevin Black (kvablack@berkeley.edu).
"""

import os
from multiprocessing import Pool
from glob import glob

import numpy as np
import tqdm
from transformers import AutoTokenizer, CLIPModel
from absl import app, flags, logging

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse. Looks for {input_path}/dir_1/dir_2/.../dir_{depth-1}/train/out.npy",
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_bool("append_suffix", False, "Append train and val suffix")
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")


def get_task_embeds(task_names):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    inputs = tokenizer(task_names, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    return text_features


def tensor_feature(value, tf):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def process(path):
    import tensorflow as tf

    with tf.io.gfile.GFile(path, "rb") as f:
        arr = np.load(f, allow_pickle=True)
    dirname = os.path.dirname(os.path.abspath(path))
    # outpath = os.path.join(FLAGS.output_path, *dirname.split(os.sep)[-FLAGS.depth :])
    outpath = os.path.join(FLAGS.output_path, *dirname.split(os.sep)[-1:])
    filename = path.split("/")[-1].split(".")[0]
    outpath = f"{outpath}/{filename}.tfrecord"

    if tf.io.gfile.exists(outpath):
        if FLAGS.overwrite:
            logging.info(f"Deleting {outpath}")
            tf.io.gfile.rmtree(outpath)
        else:
            logging.info(f"Skipping {outpath}")
            return

    if len(arr) == 0:
        logging.info(f"Skipping {path}, empty")
        return

    tf.io.gfile.makedirs(os.path.dirname(outpath))

    with tf.io.TFRecordWriter(outpath) as writer:
        for traj in arr:
            truncates = np.zeros(len(traj["actions"]), dtype=np.bool_)
            truncates[-1] = True
            task_embed = get_task_embeds(traj["language"])[0].detach().numpy()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "observations/images0": tensor_feature(
                            np.array(
                                [o["images0"] for o in traj["observations"]],
                                dtype=np.uint8,
                            ),
                            tf,
                        ),
                        "observations/state": tensor_feature(
                            np.array(
                                [
                                    np.concatenate([o["state"], task_embed])
                                    for o in traj["observations"]
                                ],
                                dtype=np.float32,
                            ),
                            tf,
                        ),
                        "next_observations/images0": tensor_feature(
                            np.array(
                                [o["images0"] for o in traj["next_observations"]],
                                dtype=np.uint8,
                            ),
                            tf,
                        ),
                        "next_observations/state": tensor_feature(
                            np.array(
                                [
                                    np.concatenate([o["state"], task_embed])
                                    for o in traj["next_observations"]
                                ],
                                dtype=np.float32,
                            ),
                            tf,
                        ),
                        "language": tensor_feature(traj["language"], tf),
                        "actions": tensor_feature(
                            np.array(traj["actions"], dtype=np.float32), tf
                        ),
                        "rewards": tensor_feature(
                            np.array(traj["rewards"], dtype=np.float32), tf
                        ),
                        "terminals": tensor_feature(
                            np.zeros(len(traj["actions"]), dtype=np.bool_), tf
                        ),
                        "truncates": tensor_feature(truncates, tf),
                    }
                )
            )
            writer.write(example.SerializeToString())


def main(_):
    assert FLAGS.depth >= 1

    paths = glob(os.path.join(FLAGS.input_path, *("*" * (FLAGS.depth - 2))))
    if FLAGS.append_suffix:
        paths = [f"{p}/train/out.npy" for p in paths] + [
            f"{p}/val/out.npy" for p in paths
        ]
    else:
        paths = list(sorted(paths))
    with Pool(FLAGS.num_workers) as p:
        list(tqdm.tqdm(p.imap(process, paths), total=len(paths)))


if __name__ == "__main__":
    app.run(main)
