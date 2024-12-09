# Based on https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py

import copy
import os
import pickle
import time

# OpenVLA imports
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Set, Union

import jax
import numpy as np
from jaxrl_m.agents.continuous.base_policy import BasePolicy
from jaxrl_m.common.typing import Batch
from jaxrl_m.utils.timer_utils import Timer

try:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from PIL import Image
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform
    from torch.optim import AdamW
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ModuleNotFoundError:
    print(
        "OpenVLA requirements are not met. If you're  going to use OpenVLA, run:\npip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt"
    )

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class OpenVLAAgent(BasePolicy):
    processor: "PrismaticProcessor"
    vla_model: "OpenVLAForActionPrediction"
    instruction: str
    action_std: np.ndarray
    prompt: str = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
    pytorch = True

    # Training variables
    # distributed_state: Optional[PartialState] = None
    device_id: str = "cuda:0"
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    learning_rate: float = 2e-5
    optimizer: Optional["torch.optim.Optimizer"] = None
    action_tokenizer: Optional["ActionTokenizer"] = None
    batch_transform: Optional["RLDSBatchTransform"] = None
    collator: Optional["PaddedCollatorForActionPrediction"] = None
    adapter_tmp_dir: str = "openvla-adapter-tmp"
    vla_tmp_dir: str = "openvla-vla-tmp"
    gradient_accumulation_steps: int = 1

    action_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    ingested_episodes: Set[str] = field(default_factory=set)
    sample_actions_call_counter: int = 0

    def _hash_observations(self, observations: np.ndarray) -> List[str]:
        # """Convert observations to hashes."""
        # This is an operation that is called many times during critic training, and observations
        # are 224x224 images, so speed is critical. Since we only support the case of training with
        # at most a couple of hours of data, we can afford to use the simple hash function of taking
        # the mean of the image pixels without collisions.
        # Calculate means across dimensions [C, H, W], but not across batch dimension
        assert (
            len(observations.shape) == 4
        ), f"Expected (B, C, H, W) shape. Got {observations.shape}"

        img_patch_size = 100
        img_H, img_W = observations.shape[1:3]
        img_patch = observations[
            :,
            img_H // 2 - img_patch_size // 2 : img_H // 2 + img_patch_size // 2,
            img_W // 2 - img_patch_size // 2 : img_W // 2 + img_patch_size // 2,
            :,
        ]
        assert img_patch.shape == (
            observations.shape[0],
            img_patch_size,
            img_patch_size,
            observations.shape[-1],
        )
        means = observations.mean(axis=(1, 2, 3))
        # stds = observations.std(axis=(1, 2, 3))
        # hashes = [hash((mean, std)) for mean, std in zip(means, stds)]
        hashes = [hash(mean) for mean in means]

        return hashes

    def clear_cache(self):
        print("Clearing OpenVLA cache")
        self.action_cache = {}
        self.ingested_episodes = set()

    def load_cache_from_filesystem(self, cache_dir: str):
        cache_files = glob(os.path.join(cache_dir, "*.pkl"))
        for cache_file in cache_files:
            if cache_file not in self.ingested_episodes:
                with open(cache_file, "rb") as f:
                    cache = pickle.load(f)
                self.action_cache.update(cache)
                self.ingested_episodes.add(cache_file)
                print(f"Loaded cache from {cache_file}")

    def openvla_normalize_actions(
        self, actions: np.ndarray, norm_key="bridge_orig"
    ) -> np.ndarray:
        assert len(actions.shape) == 2, f"Expected (B, 7) shape. Got {actions.shape}"
        # Check that actions are not normalized already
        original_action_space_low = np.array(
            [-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.0]
        )
        original_action_space_high = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0])
        # assert np.all(
        #     np.logical_and(
        #         actions >= original_action_space_low,
        #         actions <= original_action_space_high,
        #     )
        # ), "Actions might already be normalized. Please check."

        action_norm_stats = self.vla_model.get_action_stats(norm_key)
        mask = action_norm_stats["mask"]
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )

        # Normalize actions
        actions = np.where(
            mask,
            np.clip(
                2 * (actions - action_low) / (action_high - action_low + 1e-8) - 1,
                -1,
                1,
            ),
            actions,
        )
        return actions

    def sample_actions(
        self,
        observations: Union[np.ndarray, Dict[str, np.ndarray]],
        *args,
        argmax: bool = False,
        repeat: int = 1,
        cache: bool = True,
        cache_dir: str = None,
        timer: Timer = None,
        wait_for_cache: bool = True,
        **kwargs,
    ) -> np.ndarray:
        if (
            cache
            and self.sample_actions_call_counter % 1 == 0
            and cache_dir is not None
        ):
            if timer is not None:
                timer.tick("load_cache_from_filesystem")
            self.load_cache_from_filesystem(cache_dir)
            if timer is not None:
                timer.tock("load_cache_from_filesystem")
        if isinstance(observations, dict):
            observations = observations["image"]

        if len(observations.shape) == 3:
            observations = observations[None]
        assert (
            len(observations.shape) == 4
        ), f"Expected (B, H, W, C) shape. Got {observations.shape}"
        batch_size = observations.shape[0]
        assert observations.shape[1:] == (
            224,
            224,
            3,
        ), f"Expected (224, 224, 3) shape. Got {observations.shape}"

        if timer is not None:
            timer.tick("hash_observation")

        observation_hashes = self._hash_observations(observations)
        if timer is not None:
            timer.tock("hash_observation")

        # Check cache and store results
        actions = np.zeros((batch_size, repeat, 7))  # Placeholder for actions
        cache_hits = 0

        if cache:
            for i, observation in enumerate(observations):
                observation_hash = observation_hashes[i]

                while wait_for_cache and observation_hash not in self.action_cache:
                    print(f"Waiting for cache, sleeping for 1s.")
                    self.load_cache_from_filesystem(cache_dir)
                    time.sleep(1)

                if observation_hash in self.action_cache:
                    assert self.action_cache[observation_hash].shape == (
                        repeat,
                        7,
                    )
                    actions[i] = self.action_cache[observation_hash]
                    cache_hits += 1

                else:
                    if timer is not None:
                        timer.tick("openvla_process_images")
                    images: List[Image.Image] = [Image.fromarray(observation)] * repeat
                    inputs = self.processor([self.prompt] * repeat, images).to(
                        self.device_id, dtype=torch.bfloat16
                    )
                    if timer is not None:
                        timer.tock("openvla_process_images")
                    if timer is not None:
                        timer.tick("openvla_predict_action")
                    if hasattr(self.vla_model, "predict_action"):
                        action = self.vla_model.predict_action(
                            **inputs, unnorm_key="bridge_orig", do_sample=(not argmax)
                        ).reshape(repeat, 7)
                    else:
                        action = self.vla_model.predict_action(
                            **inputs, unnorm_key="bridge_orig", do_sample=(not argmax)
                        ).reshape(repeat, 7)

                    if timer is not None:
                        timer.tock("openvla_predict_action")
                    self.action_cache[observation_hash] = action
                    actions[i] = action
        else:
            if timer is not None:
                timer.tick("openvla_process_images")

            repeated_images = []
            for observation in observations:
                repeated_images.extend([Image.fromarray(observation)] * repeat)

            assert len(repeated_images) == batch_size * repeat

            inputs = self.processor(
                [self.prompt] * batch_size * repeat, repeated_images
            ).to(self.device_id, dtype=torch.bfloat16)
            if timer is not None:
                timer.tock("openvla_process_images")

            if timer is not None:
                timer.tick("openvla_predict_action")
            if hasattr(self.vla_model, "predict_action"):
                actions = self.vla_model.predict_action(
                    **inputs, unnorm_key="bridge_orig", do_sample=(not argmax)
                ).reshape(batch_size, repeat, 7)
            else:
                actions = self.vla_model.predict_action(
                    **inputs, unnorm_key="bridge_orig", do_sample=(not argmax)
                ).reshape(batch_size, repeat, 7)

            if timer is not None:
                timer.tock("openvla_predict_action")

        if cache_hits != batch_size and cache:
            print(f"OpenVLA Cache hit rate: {cache_hits / (batch_size):.2f}")
        if (
            cache_hits == batch_size
            and cache
            and self.sample_actions_call_counter % 10 == 0
        ):
            print("100% cache hits.")

        # Actions are unnormalized, but the critic and environment expect normalized actions
        actions[..., :6] = actions[..., :6] / self.action_std[:6]

        self.sample_actions_call_counter += 1
        return actions.reshape(batch_size, repeat, 7)

    def update(
        self,
        batch: Batch,
        timer: Timer,
        pmap_axis: str = None,
    ):
        if timer is not None:
            timer.tick("openvla_update_batch_processing")
        self.optimizer.zero_grad()
        # Batch processing
        batch_size = batch["actions"].shape[0]
        batch_actions = jax.device_get(batch["actions"]).copy()
        batch_images = jax.device_get(batch["observations"]["image"])
        assert len(batch_images.shape) == 5 and batch_images.shape[1] == 1
        batch_images = batch_images.reshape(batch_size, 224, 224, 3)

        # Unnormalize jaxrl actions
        assert (
            len(batch_actions.shape) == 3 and batch_actions.shape[1] == 1
        ), f"Expected (B, 1, 7) shape. Got {batch_actions.shape}"
        batch_actions = batch_actions.reshape(batch_size, 7)
        batch_actions[:, :6] = batch_actions[:, :6] * self.action_std[:6]
        # Normalize actions with OpenVLA normalization
        batch_actions = self.openvla_normalize_actions(
            batch_actions, norm_key="bridge_orig"
        )

        batch = [
            self.batch_transform(
                {
                    "dataset_name": "PA-RL",
                    "action": batch_actions[i][None],
                    "observation": {"image_primary": batch_images[i][None]},
                    "task": {"language_instruction_text": self.instruction},
                }
            )
            for i in range(batch_size)
        ]
        batch = self.collator(batch)
        if timer is not None:
            timer.tock("openvla_update_batch_processing")
        for gradient_accumuplation_step in range(self.gradient_accumulation_steps):
            if timer is not None:
                timer.tick("openvla_update_forward_pass")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = self.vla_model(
                    input_ids=batch["input_ids"].to(self.device_id),
                    attention_mask=batch["attention_mask"].to(self.device_id),
                    pixel_values=batch["pixel_values"]
                    .to(torch.bfloat16)
                    .to(self.device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / self.gradient_accumulation_steps
            if timer is not None:
                timer.tock("openvla_update_forward_pass")

            # Backward pass
            if timer is not None:
                timer.tick("openvla_update_backward_pass")
            normalized_loss.backward()
            if timer is not None:
                timer.tock("openvla_update_backward_pass")

            if timer is not None:
                timer.tick("openvla_update_metrics")
            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[
                :,
                # self.vla_model.module.vision_backbone.featurizer.patch_embed.num_patches : -1,
                self.vla_model.vision_backbone.featurizer.patch_embed.num_patches : -1,
            ]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > self.action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                self.action_tokenizer.decode_token_ids_to_actions(
                    action_preds[mask].cpu().numpy()
                )
            )
            continuous_actions_gt = torch.tensor(
                self.action_tokenizer.decode_token_ids_to_actions(
                    action_gt[mask].cpu().numpy()
                )
            )
            action_l1_loss = torch.nn.functional.l1_loss(
                continuous_actions_pred, continuous_actions_gt
            )
            if timer is not None:
                timer.tock("openvla_update_metrics")

        # Optimizer Step
        if timer is not None:
            timer.tick("openvla_update_optimizer_step")
        self.optimizer.step()
        if timer is not None:
            timer.tock("openvla_update_optimizer_step")
        return self, {
            "loss": loss.item(),
            "action_accuracy": action_accuracy.item(),
            "action_l1_loss": action_l1_loss.item(),
        }

    def prepare_for_finetuning(self):
        self.vla_model.train()

    def prepare_for_inference(self):
        self.vla_model.eval()

    def save_checkpoint(self, output_dir: str):
        # Save the model checkpoint
        processor_saving_path = os.path.join(output_dir, "openvla_processor")
        os.makedirs(processor_saving_path, exist_ok=True)
        self.processor.save_pretrained(processor_saving_path)
        model_path = os.path.join(output_dir, "openvla_model")
        os.makedirs(model_path, exist_ok=True)
        print(f"Saving model to {model_path}...")
        self.vla_model.save_pretrained(model_path)

    def restore_checkpoint(self, checkpoint_path: str):
        # Load the model checkpoint
        model_path = os.path.join(checkpoint_path, "openvla_model")
        del self.vla_model
        torch.cuda.empty_cache()
        base_vla = OpenVLAForActionPrediction.from_pretrained(
            "/scr/maxsobolmark/openvla_base_checkpoint",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.vla_model = PeftModel.from_pretrained(
            base_vla, model_path, is_trainable=True
        ).to(self.device_id)
        self.vla_model.print_trainable_parameters()

        print(f"Loaded model from {model_path}...")
        # Clear cache because the model has changed
        self.clear_cache()

    def set_language_instruction(self, new_instruction: str):
        self.prompt = "In: What action should the robot take to {}?\nOut:".format(
            new_instruction
        )
        self.instruction = new_instruction

    def __init__(
        self,
        *args,
        action_std: List[float],
        # Fine-tuning parameters
        learning_rate: float = 2e-5,
        gradient_accumulation_steps: int = 1,
        image_aug: bool = True,
        # LoRA parameters
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_dropout: float = 0.0,
        adapter_tmp_dir: str = "openvla-adapter-tmp",
        instruction: str = "",
        **kwargs,
    ):
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        self.processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        )

        vla_model = OpenVLAForActionPrediction.from_pretrained(
            "openvla/openvla-7b",
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        assert (
            torch.cuda.is_available()
        ), "OpenVLA requires a GPU to run, but no GPU was found"
        self.device_id = "cuda:0"
        torch.cuda.empty_cache()
        self.vla_model = vla_model.to(self.device_id)

        # Create Action Tokenizer
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)

        self.batch_transform = RLDSBatchTransform(
            self.action_tokenizer,
            self.processor.tokenizer,
            image_transform=self.processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
        )

        # Create Collator
        self.collator = PaddedCollatorForActionPrediction(
            self.processor.tokenizer.model_max_length,
            self.processor.tokenizer.pad_token_id,
            padding_side="right",
        )

        if use_lora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=min(lora_rank, 16),
                lora_dropout=lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            self.vla_model = get_peft_model(self.vla_model, lora_config)
            self.vla_model.print_trainable_parameters()

        # Create Optimizer =>> note that we default to a simple constant learning rate!
        trainable_params = [
            param for param in vla_model.parameters() if param.requires_grad
        ]
        self.optimizer = AdamW(trainable_params, lr=learning_rate)

        self.use_lora = use_lora
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.adapter_tmp_dir = adapter_tmp_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.instruction = instruction
        self.action_std = np.array(action_std)
        self.prompt = "In: What action should the robot take to {}?\nOut:".format(
            instruction
        )
