import torch
from pathlib import Path
from typing import List, Dict, Union
from argparse import ArgumentParser

from mobileposer.config import *


def load_file(file_path: Path):
    return torch.load(file_path)

def process_tensor(tensor: torch.Tensor, key: str): 
    processing_rules = {
        'acc': lambda t: t.view(-1, 5, 3)[:, [1, 4, 3, 0, 2]],
        'raw_acc': lambda t: t.view(-1, 5, 3)[:, [1, 4, 3, 0, 2]],
        'ori': lambda t: t.view(-1, 5, 3, 3)[:, [1, 4, 3, 0, 2]],
        'raw_ori': lambda t: t.view(-1, 5, 3, 3)[:, [1, 4, 3, 0, 2]],
        'pose': lambda t: t.view(-1, 24, 3, 3),
        'tran': lambda t: t.view(-1, 3)
    }
    return processing_rules.get(key, lambda t: t)(tensor)

def generate_dataset(file_paths: List[Path]):
    combined_data = {'calibration': []}
    keys_to_process = {'acc', 'raw_acc', 'raw_ori', 'ori', 'pose', 'tran'}

    for file_path in file_paths:
        data = load_file(file_path)
        for key, value in data.items():
            if key == 'calibration':
                combined_data['calibration'].append(value)
            else:
                if key not in combined_data:
                    combined_data[key] = []
                processed_value = process_tensor(value, key) if key in keys_to_process else value
                combined_data[key].append(processed_value)
    return combined_data

def save_dataset(dataset: Dict[str, Union[List[torch.Tensor], List[Dict]]], output_path: Path):
    torch.save(dataset, output_path)

def get_data_files(folder_path: Path, output_path: Path):
    return [f for f in folder_path.glob('*.pt') if f != output_path]

def print_data_shapes(dataset: Dict[str, Union[List[torch.Tensor], List[Dict]]]):
    for key, value in dataset.items():
        if key != 'calibration':
            print(f"{key}: {[tensor.shape for tensor in value]}")
        else:
            print(f"{key}: {len(value)} items")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--data-folder', type=str, default=paths.dev_data)
    args = args.parse_args()

    data_folder = Path(args.data_folder)
    output_path = Path(f'{data_folder}/dev.pt')
    file_paths = get_data_files(data_folder, output_path)
    
    if not file_paths:
        print(f"No .pt files found in {data_folder}.")
        exit(0)

    print(f"Found {len(file_paths)} .pt files in {data_folder}")
    dataset = generate_dataset(file_paths)
    save_dataset(dataset, output_path)
    print(f"Dataset saved to {output_path}")
    print_data_shapes(dataset)
