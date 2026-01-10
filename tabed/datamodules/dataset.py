"""Dataset classes and utilities for MLLM training and evaluation."""

import os
import pickle
from copy import deepcopy
from typing import Dict, List, Optional

import PIL
import torch
from absl import logging
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BatchFeature

from ..modules.load_pretrained import load_tokenizer


# =============================================================================
# Path Configuration
# =============================================================================

def _get_llava_data_dir() -> str:
    """Get the LLaVA data directory from environment or default."""
    return os.environ.get(
        'LLAVA_DATA_DIR',
        os.path.join(os.path.expanduser('~'), 'data', 'llava-next', 'llava_instruct')
    )


class MLLMDataset(Dataset):
    """Dataset class for Multimodal Large Language Model training/evaluation.

    Handles various dataset formats including single-image, multi-image,
    and multi-turn conversation datasets.

    Args:
        dataset: The underlying Hugging Face dataset.
        config: Configuration dictionary.
        tokenizer: Tokenizer for text processing.
        drf_aux_tokenizer: Auxiliary tokenizer for draft model.
        drf_image_processor: Image processor for draft model.
    """

    def __init__(
        self,
        dataset,
        config: Dict,
        tokenizer,
        drf_aux_tokenizer,
        drf_image_processor,
    ):
        """Initialize the MLLM dataset."""
        self.dataset = dataset
        self._config = config
        self.tokenizer = tokenizer
        self.drf_aux_tokenizer = drf_aux_tokenizer
        self.drf_image_processor = drf_image_processor

        if 'caption' in config['drafting']:
            self.captioning_image_processor, _, _ = load_tokenizer(
                self._config, None, 'captioning_model'
            )

        self.image_token = "<image>"
        self.example = self._get_example_data()

        if self._config['multi_turn_task'] is not None:
            self._init_multi_turn(dataset, config, tokenizer,
                                  drf_aux_tokenizer, drf_image_processor)

    def _init_multi_turn(
        self,
        dataset,
        config,
        tokenizer,
        drf_aux_tokenizer,
        drf_image_processor,
    ):
        """Initialize multi-turn conversation settings."""
        self.multi_turn_tasks = self._config['multi_turn_task'].split('-')
        self.num_turns = len(self.multi_turn_tasks)
        self.current_task = self.multi_turn_tasks[-1]

        if self.current_task == 'nq':
            self.current_task_ds = load_dataset(
                "google-research-datasets/nq_open", split='validation'
            )['question'][:500]
        elif self.current_task == 'gsm8k':
            self.current_task_ds = load_dataset(
                "openai/gsm8k", "main"
            )['test']['question'][:500]
        elif 'convbench' in self.current_task:
            assert all(
                'convbench' in task for task in self.multi_turn_tasks[1:]
            )
            self.current_task_ds = self.dataset
        elif self.current_task == 'qaorg':
            _config_temp = deepcopy(self._config)
            _config_temp['multi_turn_task'] = None
            self.current_task_ds = MLLMDataset(
                dataset, _config_temp, tokenizer,
                drf_aux_tokenizer, drf_image_processor
            )
            print(f"[TURN {self.num_turns}]: {config['dataset']} - To be added")
        elif self.current_task == 'eval':
            _config_temp = deepcopy(self._config)
            _config_temp['multi_turn_task'] = None
            self.current_task_ds = MLLMDataset(
                dataset, _config_temp, tokenizer,
                drf_aux_tokenizer, drf_image_processor
            )

        self.prev_qa = self.load_prev_qa()

        self.task_map = {
            'detection': (
                "List 10 objects in the figure in order from "
                "left to right and top to bottom."
            ),
            'captioning': "Provide a detailed caption for the provided image.",
            'translation': "Translate the previous answer into German.",
            'summary': "Summarize the previous answer.",
            'story': (
                "Can you use the previous responses to write "
                "a long story for children?"
            ),
            'eval': (
                "Evaluate how well the questions in the previous "
                "conversation were answered."
            ),
        }

    def load_prev_qa(self):
        """Load previous QA data for multi-turn conversations."""
        prev_turns = '-'.join(self.multi_turn_tasks[:-1])

        try:
            prev_filename = f"{self._config['dataset']}_{prev_turns}"
            np_file_path = os.path.join(
                self._config['multi_turn_data_root'],
                prev_filename,
                "sequences.npy"
            )
            import numpy as np
            np_file = np.load(np_file_path, allow_pickle=True)
        except Exception:
            prev_filename = f"{self._config['dataset']}-{prev_turns}"
            np_file_path = os.path.join(
                self._config['multi_turn_data_root'],
                prev_filename,
                "sequences.npy"
            )
            import numpy as np
            np_file = np.load(np_file_path, allow_pickle=True)

        return np_file

    def get_prev_qa_prompt(self, sample_idx: int) -> str:
        """Get the previous QA prompt for a given sample index."""
        return self.tokenizer.decode(
            self.prev_qa[sample_idx], skip_special_tokens=False
        )

    def create_prompt_multi_turn_task(self, sample_idx: int) -> str:
        """Create prompt for multi-turn task based on current task type."""
        if self.current_task == 'nq':
            question = self.current_task_ds[sample_idx].capitalize()
            return (
                f"{question}. Provide a comprehensive and detailed "
                "explanation, including the reasoning and background "
                "information."
            )

        elif self.current_task == 'gsm8k':
            question = self.current_task_ds[sample_idx]
            return (
                f"{question}. Provide a detailed explanation.\n"
                "Let's think step by step."
            )

        elif 'convbench' in self.current_task:
            col_map = {
                2: "The_second_turn_instruction",
                3: "The_third_turn_instruction",
            }
            col = col_map[self.num_turns]
            question = self.current_task_ds[sample_idx][col]
            prompt = question
            if 'long' in self.current_task:
                prompt += " Provide a detailed explanation."
            return prompt

        return self.task_map[self.current_task]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, sample_idx: int) -> Dict:
        """Get a processed sample by index."""
        sample = self.dataset[sample_idx]
        return self._process_input(sample, sample_idx)

    def collate_fn(self, processed_samples: List[Dict[str, torch.Tensor]]):
        """Collate function for batching samples."""
        if len(processed_samples) == 1:
            return BatchFeature(processed_samples[0])

        text_list = [sample['prompt'] for sample in processed_samples]
        batched_raw_input = {
            'text': text_list,
            'padding': True,
            'return_tensors': "pt",
        }

        if self._config['is_drf_from_mllm'] or self._config['eval_datasets'] is None:
            image_list = sum(
                [sample['images'] for sample in processed_samples], []
            )
            batched_raw_input['images'] = image_list

        batched_samples = self.tokenizer(batched_raw_input)
        return BatchFeature(batched_samples)

    def _process_input(self, sample, sample_idx: int) -> Dict:
        """Process a single input sample."""
        processed_input = {}

        images = self._get_image_data(sample)
        is_no_img = (
            self._config['is_drf_text_only'] and self._config['is_tgt_text_only']
        )

        if self._config['is_drf_from_mllm'] or self._config['eval_datasets'] is None:
            pixel_values = None
            if not is_no_img:
                pixel_values = self.drf_image_processor(
                    images, return_tensors="pt"
                )["pixel_values"]
            processed_input['pixel_values'] = pixel_values

            if hasattr(self, 'captioning_image_processor'):
                self._add_captioning_inputs(processed_input, images)

            if self._config['batch_size'] > 1:
                processed_input['images'] = images

        outputs_tune_prompt = self._tune_prompt(sample, sample_idx)
        prompt = outputs_tune_prompt['prompt']

        processed_input.update(self.tokenizer(prompt, return_tensors="pt"))

        if outputs_tune_prompt['pixel_values_second_turn'] is not None:
            self._handle_second_turn_pixels(processed_input, outputs_tune_prompt)

        if not self._config['is_drf_from_mllm']:
            aux_inputs = {
                f"aux_{k}": v
                for k, v in self.drf_aux_tokenizer(
                    prompt, return_tensors="pt"
                ).items()
            }
            processed_input.update(aux_inputs)

        if self._config['batch_size'] > 1:
            processed_input['prompt'] = prompt

        return processed_input

    def _add_captioning_inputs(self, processed_input: Dict, images: List):
        """Add captioning model inputs to processed input."""
        inputs_caption = {}
        if (
            'caption' in self._config['drafting']
            and "lorence-2" in self._config['captioning_model']
        ):
            inputs_caption_processed = self.captioning_image_processor(
                text=self._config['caption_type'],
                images=images,
                return_tensors="pt"
            )
        else:
            inputs_caption_processed = self.captioning_image_processor(
                images=images, return_tensors="pt"
            )

        for key in inputs_caption_processed.keys():
            if key in ['pixel_values', 'input_ids']:
                inputs_caption[key + '_caption'] = inputs_caption_processed[key]

        processed_input.update(inputs_caption)

    def _handle_second_turn_pixels(self, processed_input: Dict, outputs: Dict):
        """Handle pixel values for second turn in multi-turn conversations."""
        if self.current_task == 'eval':
            processed_input['pixel_values'] = outputs['pixel_values_second_turn']
        else:
            processed_input['pixel_values'] = torch.cat(
                [processed_input['pixel_values'],
                 outputs['pixel_values_second_turn']],
                dim=0
            )

        if (
            self._config['multi_turn_task'] == 'qaorg-qaorg'
            and 'caption' in self._config['drafting']
        ):
            processed_input['pixel_values_caption'] = torch.cat(
                [processed_input['pixel_values_caption'],
                 outputs['pixel_values_caption_second_turn']],
                dim=0
            )

    def _get_example_data(self) -> Optional[Dict]:
        """Load example data for few-shot learning if available."""
        if not hasattr(self, 'batch_example'):
            if self._config['dataset'] == "ScienceQA":
                logging.info(
                    "[Dataset] Loading batch_example from disk for a single shot"
                )
                path_example = os.path.join(
                    self._config['input_datasets_dir'],
                    "example/batch_example.pkl"
                )
                with open(path_example, 'rb') as f:
                    return pickle.load(f)
        return None

    def _get_image_data(self, batch: Dict) -> List:
        """Extract and process images from a batch."""
        images = []

        if self.example is not None:
            images.append(self.example['image'].convert('RGB'))

        if self._config['ensemble_train_split']:
            return self._get_ensemble_images(batch)

        dataset = self._config['dataset']

        multi_image_datasets = [
            'Spot-the-Diff', 'Birds-to-Words', 'CLEVR-Change', 'HQ-Edit',
            'MagicBrush', 'IEdit', 'AESOP', 'FlintstonesSV', 'PororoSV',
            'VIST', 'WebQA', 'QBench', 'NLVR2_Mantis', 'OCR-VQA'
        ]

        if dataset in multi_image_datasets:
            cols_img = [col for col in batch.keys() if col.startswith('image_')]
            images = [
                batch[col].convert('RGB')
                for col in cols_img if batch[col] is not None
            ]
        elif dataset == 'LiveBench':
            images = [image.convert('RGB') for image in batch['images']]
        else:
            images = self._get_single_image(batch)

        return images

    def _get_ensemble_images(self, batch: Dict) -> List:
        """Get images for ensemble train split datasets."""
        dataset = self._config['dataset']

        if dataset in [
            "Spot-the-Diff", "PororoSV", "IEdit", "VIST", "WebQA", "MagicBrush"
        ]:
            image_dir = os.path.join(_get_llava_data_dir(), "M4-Instruct-Data")
            path_image_list = [
                os.path.join(image_dir, img_file)
                for img_file in batch['image']
            ]
            return [
                PIL.Image.open(path).convert('RGB')
                for path in path_image_list
            ]
        elif dataset in ["HallusionBench", "textvqa_val", "vqav2_val", "chartqa"]:
            return [batch['image'].convert('RGB')]
        else:
            raise ValueError(
                f"Dataset not supported for ensemble train split: {dataset}"
            )

    def _get_single_image(self, batch: Dict) -> List:
        """Get single image from batch."""
        images = []

        if isinstance(batch['image'], str):
            if self._config['dataset'] == "convbench":
                path_image = os.path.join(
                    self._config['input_datasets_dir'] + "_repo",
                    'visit_bench_images',
                    batch['image']
                )
            else:
                path_image = os.path.join(
                    self._config['input_datasets_dir'],
                    'images',
                    batch['image']
                )
            image = PIL.Image.open(path_image)
        else:
            image = batch['image']

        images.append(image.convert('RGB'))
        return images

    def _get_text_data(self, batch: Dict, sample_idx: int = None) -> str:
        """Extract text data from batch based on dataset type."""
        dataset = self._config['dataset']

        if self._config['ensemble_train_split']:
            return self._get_ensemble_text(batch)

        dataset_to_key = {
            "VibeEval": 'prompt',
            "DC100_EN": 'question',
            "LLaVA-Bench-Wilder": 'Question',
        }

        if dataset == "LLaVA-Instruct-150K":
            return batch["conversations"][0]['value']
        elif dataset == "COCO2014":
            return "Provide a detailed description of the given image."
        elif dataset in [
            'chartqa', 'docvqa_val', 'infovqa_val', 'ok_vqa_val2014',
            'textvqa_val', 'vizwiz_vqa_val', 'vqav2_val', 'QBench',
            'NLVR2_Mantis', 'OCR-VQA', 'MMVet', 'POPE', 'HallusionBench'
        ]:
            return (
                "For the following question, provide a detailed "
                f"explanation of your reasoning leading to the answer.\n"
                f"{batch['question']}"
            )
        elif dataset == "ScienceQA":
            pass
        elif dataset in dataset_to_key:
            return batch[dataset_to_key[dataset]]
        elif dataset in [
            'llava-bench-in-the-wild', 'Spot-the-Diff', 'Birds-to-Words',
            'CLEVR-Change', 'HQ-Edit', 'MagicBrush', 'IEdit', 'AESOP',
            'FlintstonesSV', 'PororoSV', 'VIST', 'WebQA', 'LiveBench',
            'convbench'
        ]:
            return batch['question']
        else:
            raise ValueError(f"Dataset not supported: {dataset}")

    def _get_ensemble_text(self, batch: Dict) -> str:
        """Get text data for ensemble train split datasets."""
        dataset = self._config['dataset']

        if dataset in [
            "Spot-the-Diff", "PororoSV", "IEdit", "VIST", "WebQA", "MagicBrush"
        ]:
            return batch['conversations'][0]['value']
        elif dataset in [
            "HallusionBench", "textvqa_val", "vqav2_val", "chartqa"
        ]:
            return batch['question']
        else:
            raise ValueError(
                f"Dataset not supported for ensemble train split: {dataset}"
            )

    def _tune_prompt(self, batch: Dict, sample_idx: int) -> Dict:
        """Apply prompt tuning based on dataset and configuration."""
        pixel_values_second_turn = None
        pixel_values_caption_second_turn = None

        if hasattr(self.tokenizer, 'image_processor'):
            conversation = self._build_conversation(batch, sample_idx)

            if self.tokenizer.chat_template is None:
                if 'llava-hf' not in self._config['drf']:
                    self.tokenizer.chat_template = self._get_default_chat_template()

            resulting_prompt = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            if self._config['multi_turn_task'] is not None:
                resulting_prompt, pixel_values_second_turn, pixel_values_caption_second_turn = (
                    self._apply_multi_turn_prompt(
                        sample_idx, resulting_prompt
                    )
                )
        else:
            resulting_prompt = self._get_text_data(batch, sample_idx)

        output = {
            'prompt': resulting_prompt,
            'pixel_values_second_turn': pixel_values_second_turn,
        }
        if pixel_values_caption_second_turn is not None:
            output['pixel_values_caption_second_turn'] = pixel_values_caption_second_turn

        return output

    def _build_conversation(self, batch: Dict, sample_idx: int) -> List[Dict]:
        """Build conversation structure based on dataset type."""
        dataset = self._config['dataset']

        if dataset == 'LLaVA-Instruct-150K':
            prompt_raw = self._get_text_data(batch, sample_idx)
            return [{
                "role": "user",
                "content": [{"type": "text", "text": prompt_raw}],
            }]

        single_image_datasets = [
            "llava-bench-in-the-wild", "COCO2014", "VibeEval", "DC100_EN",
            "LLaVA-Bench-Wilder", "LiveBench", "chartqa", "docvqa_val",
            "infovqa_val", "ok_vqa_val2014", "textvqa_val", "vizwiz_vqa_val",
            "vqav2_val", "MMVet", "POPE", "HallusionBench", "convbench"
        ]

        multi_image_datasets = [
            'Spot-the-Diff', 'Birds-to-Words', 'CLEVR-Change', 'HQ-Edit',
            'MagicBrush', 'IEdit', 'AESOP', 'FlintstonesSV', 'PororoSV',
            'VIST', 'WebQA', 'QBench', 'NLVR2_Mantis', 'OCR-VQA'
        ]

        if dataset in single_image_datasets:
            prompt_raw = self._get_text_data(batch, sample_idx)
            return [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_raw}
                ],
            }]
        elif dataset in multi_image_datasets:
            prompt_raw = self._get_text_data(batch, sample_idx)
            return [{
                "role": "user",
                "content": [{"type": "text", "text": prompt_raw}],
            }]
        elif dataset == 'ScienceQA':
            return self._get_conversation_scienceqa(batch)
        else:
            raise ValueError(f"Dataset not supported: {dataset}")

    def _get_default_chat_template(self) -> str:
        """Get the default chat template for LLaVA models."""
        return (
            "{% for message in messages %}"
            "{% if message['role'] != 'system' %}"
            "{{ message['role'].upper() + ': '}}"
            "{% endif %}"
            "{# Render all images first #}"
            "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
            "{{ '<image>\n' }}"
            "{% endfor %}"
            "{# Render all text next #}"
            "{% if message['role'] != 'assistant' %}"
            "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
            "{{ content['text'] + ' '}}"
            "{% endfor %}"
            "{% else %}"
            "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
            "{% generation %}{{ content['text'] + ' '}}{% endgeneration %}"
            "{% endfor %}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
        )

    def _apply_multi_turn_prompt(self, sample_idx: int, resulting_prompt: str):
        """Apply multi-turn conversation formatting."""
        pixel_values_second_turn = None
        pixel_values_caption_second_turn = None

        prev_qa_prompt = self.get_prev_qa_prompt(sample_idx)

        if self.current_task == 'qaorg':
            sample_idx_current = (
                sample_idx + 1
                if len(self.current_task_ds) > sample_idx + 1 else 0
            )
            resulting_prompt_current = clean_prompt(
                self.tokenizer.batch_decode(
                    self.current_task_ds[sample_idx_current]['input_ids']
                )[0]
            )
            pixel_values_second_turn = (
                self.current_task_ds[sample_idx_current]['pixel_values']
            )
            if (
                self._config['multi_turn_task'] == 'qaorg-qaorg'
                and 'caption' in self._config['drafting']
            ):
                pixel_values_caption_second_turn = (
                    self.current_task_ds[sample_idx_current]['pixel_values_caption']
                )
        else:
            current_q_prompt = self.create_prompt_multi_turn_task(sample_idx)

            if self.current_task == 'eval':
                pixel_values_second_turn = self._get_eval_pixel_values(sample_idx)

            conversation_current = [{
                "role": "user",
                "content": [{"type": "text", "text": current_q_prompt}],
            }]
            resulting_prompt_current = self.tokenizer.apply_chat_template(
                conversation_current, add_generation_prompt=True
            )

        resulting_prompt = clean_prompt(prev_qa_prompt) + ' ' + resulting_prompt_current

        return resulting_prompt, pixel_values_second_turn, pixel_values_caption_second_turn

    def _get_eval_pixel_values(self, sample_idx: int) -> torch.Tensor:
        """Get pixel values for evaluation multi-turn task."""
        n = sum(1 for task in self.multi_turn_tasks if task == 'qaorg')

        pixel_values_list = []
        current_idx = sample_idx + 1 if n > 1 else sample_idx

        for _ in range(n):
            if current_idx >= len(self.current_task_ds):
                current_idx = 0
            pixel_values_list.append(
                self.current_task_ds[current_idx]['pixel_values']
            )
            current_idx += 1

        return torch.cat(pixel_values_list, dim=0)

    def _get_conversation_scienceqa(self, batch: Dict) -> List[Dict]:
        """Build conversation for ScienceQA dataset."""
        example_shot_1, example_shot_2 = self._build_common_prompt(
            self.example, include_answer=True, split_example=True
        )
        subsequence_prompt_1, _ = self._build_common_prompt(
            batch, include_answer=False, split_example=True
        )

        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": example_shot_1}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example_shot_2}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": subsequence_prompt_1}],
            },
        ]

    def _build_common_prompt(
        self,
        batch: Dict,
        include_answer: bool = True,
        split_example: bool = False,
    ) -> tuple:
        """Build common prompt for ScienceQA."""
        question = batch['question']
        choices = batch['choices']
        answer = batch['choices'][batch['answer']]

        options = ' '.join([
            f"({chr(65 + i)}) {choice}"
            for i, choice in enumerate(choices)
        ])

        instruction = (
            "Based on the image, provide Reasoning and detailed "
            "Explanation behind the provided Answer to the Question."
        )

        common_prompt_1 = (
            f"{self.image_token}\n"
            f"Question: {question}\n"
            f"Options: {options}\n"
            f"Answer: The answer is {answer}\n"
            f"{instruction}\n"
        )

        common_prompt_2 = ""
        if include_answer:
            lecture = batch['lecture']
            solution = batch['solution']
            common_prompt_2 = (
                f"Reasoning: {lecture}\n"
                f"Explanation: {solution}"
            )

        return common_prompt_1, common_prompt_2


def load_datasets(
    config: Dict,
    tokenizer,
    drf_aux_tokenizer,
    drf_image_processor,
) -> Dict[str, Dataset]:
    """Load and prepare datasets based on configuration.

    Args:
        config: Configuration dictionary.
        tokenizer: Tokenizer for text processing.
        drf_aux_tokenizer: Auxiliary tokenizer for draft model.
        drf_image_processor: Image processor for draft model.

    Returns:
        Dictionary mapping split names to MLLMDataset instances.
    """
    input_datasets_dir = config['input_datasets_dir']
    if config['ensemble_train_split']:
        input_datasets_dir = os.path.join(
            config['ensemble_train_split_dir'], config['dataset']
        )

    if config['dataset'] == "LLaVA-Instruct-150K":
        path_dataset = os.path.join(input_datasets_dir, 'meta.json')
        map_datasets = load_dataset("json", data_files=path_dataset)
    else:
        path_dataset = input_datasets_dir
        map_datasets = load_from_disk(path_dataset)

    # Apply dataset-specific processing
    if config['ensemble_train_split']:
        reduce_map = {'train': 100, 'validation': 100, 'test': 200}
        map_datasets = _apply_tiny_data_filter(map_datasets, reduce_map)

    elif config['dataset'] == "LLaVA-Instruct-150K":
        split_single = list(map_datasets.keys())[0]
        full_train_dataset = map_datasets[split_single]
        map_datasets = _split_dataset(full_train_dataset, config)

    elif config['dataset'] == "ScienceQA":
        reduce_map = {'train': 100, 'validation': 100, 'test': 100}
        map_datasets = _apply_tiny_data_filter(map_datasets, reduce_map)

    elif config['dataset'] == 'llava-bench-in-the-wild':
        reduce_map = {'train': 100, 'validation': 100, 'test': 60}
        map_datasets = _apply_tiny_data_filter(map_datasets, reduce_map)

    else:
        reduce_map = {'train': 100, 'validation': 100, 'test': 100}
        map_datasets = _apply_tiny_data_filter(map_datasets, reduce_map)

    # Apply tiny_data or reduce_data filters
    if config['tiny_data']:
        tiny_map = {'train': 80, 'validation': 10, 'test': 3}
        map_datasets = _apply_tiny_data_filter(map_datasets, tiny_map)

    if config['reduce_data'] is not None:
        logging.info(f"[Dataset] Using reduced data to {config['reduce_data']} rows")
        for split in map_datasets.keys():
            map_datasets[split] = map_datasets[split].select(
                range(config['reduce_data'])
            )

    for split, dataset in map_datasets.items():
        logging.info(f"[Dataset] {split} dataset: {len(dataset)} samples")

    map_datasets = _wrap_with_mllm_dataset(
        map_datasets, config, tokenizer, drf_aux_tokenizer, drf_image_processor
    )

    return map_datasets


def _split_dataset_for_multiprocessing(
    dataset,
    rank: int,
    world_size: int,
    seed: int = 42,
):
    """Split dataset for multiprocessing based on rank and world_size.

    Args:
        dataset: Dataset to split.
        rank: Current process rank.
        world_size: Total number of processes.
        seed: Random seed (unused, kept for API compatibility).

    Returns:
        Subset of the dataset for the current rank.
    """
    if world_size <= 1:
        return dataset

    dataset_size = len(dataset)
    chunk_size = dataset_size // world_size
    start_idx = rank * chunk_size

    if rank == world_size - 1:
        end_idx = dataset_size
    else:
        end_idx = (rank + 1) * chunk_size

    indices = list(range(start_idx, end_idx))
    return Subset(dataset, indices)


def create_data_loaders(
    config: Dict,
    tokenizer,
    drf_image_processor,
    drf_aux_tokenizer=None,
) -> Dict[str, DataLoader]:
    """Create data loaders for all dataset splits.

    Args:
        config: Configuration dictionary.
        tokenizer: Tokenizer for text processing.
        drf_image_processor: Image processor for draft model.
        drf_aux_tokenizer: Auxiliary tokenizer for draft model.

    Returns:
        Dictionary mapping split names to DataLoader instances.
    """
    data_loaders = {}
    map_datasets = load_datasets(
        config, tokenizer, drf_aux_tokenizer, drf_image_processor
    )

    rank = config.get('rank', 0)
    world_size = config.get('world_size', 1)

    for split, dataset in map_datasets.items():
        original_dataset = dataset

        if world_size > 1:
            dataset = _split_dataset_for_multiprocessing(
                dataset, rank, world_size, config.get('seed', 42)
            )
            logging.info(
                f"[Dataset] Rank {rank}: Processing {len(dataset)} "
                f"samples from {split} split"
            )

        shuffle = split == "train"

        collate_fn = None
        if hasattr(original_dataset, 'collate_fn'):
            collate_fn = original_dataset.collate_fn
        elif (
            hasattr(dataset, 'dataset')
            and hasattr(dataset.dataset, 'collate_fn')
        ):
            collate_fn = dataset.dataset.collate_fn

        data_loaders[split] = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    return data_loaders


def _split_dataset(dataset, config: Dict) -> DatasetDict:
    """Split dataset into train/validation/test splits.

    Args:
        dataset: Dataset to split.
        config: Configuration dictionary containing seed.

    Returns:
        DatasetDict with train, validation, and test splits.
    """
    train_valid_test = dataset.train_test_split(
        test_size=0.2, shuffle=True, seed=config['seed']
    )
    test_valid_split = train_valid_test['test'].train_test_split(
        test_size=0.5, shuffle=True, seed=config['seed']
    )

    return DatasetDict({
        'train': train_valid_test['train'],
        'validation': test_valid_split['train'],
        'test': test_valid_split['test'],
    })


def _apply_tiny_data_filter(map_datasets, tiny_map: Dict[str, int]):
    """Apply tiny data filter to reduce dataset sizes.

    Args:
        map_datasets: Dictionary of datasets by split.
        tiny_map: Dictionary mapping splits to maximum sizes.

    Returns:
        Filtered datasets dictionary.
    """
    logging.info("[Dataset] Using tiny data")
    for split in map_datasets.keys():
        map_datasets[split] = map_datasets[split].select(range(tiny_map[split]))
    return map_datasets


def _wrap_with_mllm_dataset(
    map_datasets,
    config: Dict,
    tokenizer,
    drf_aux_tokenizer,
    drf_image_processor,
):
    """Wrap raw datasets with MLLMDataset class.

    Args:
        map_datasets: Dictionary of raw datasets by split.
        config: Configuration dictionary.
        tokenizer: Tokenizer for text processing.
        drf_aux_tokenizer: Auxiliary tokenizer for draft model.
        drf_image_processor: Image processor for draft model.

    Returns:
        Dictionary of MLLMDataset instances by split.
    """
    for split in map_datasets.keys():
        map_datasets[split] = MLLMDataset(
            map_datasets[split],
            config,
            tokenizer,
            drf_aux_tokenizer,
            drf_image_processor,
        )
    return map_datasets


def clean_prompt(text: str) -> str:
    """Remove special tokens from prompt text.

    Args:
        text: Input text potentially containing special tokens.

    Returns:
        Cleaned text without special tokens.
    """
    return text.replace('<s>', '').replace('</s>', '').strip()
