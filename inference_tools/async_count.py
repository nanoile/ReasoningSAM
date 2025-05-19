import os
import openai
import json
import sys
from instruct_sam.counting import encode_image_as_base64, extract_dict_from_string, save_count
import aiohttp
import asyncio
from instruct_sam.prompts import *
import time
import argparse


SEND_INTERVAL = 0.25

failed_requests = []  # record all failed images
failed_analysis = []  # record all failed analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run asynchronous counting on a dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset (e.g., 'dior_mini'). Must match a key in the config file.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="datasets/config.json",
        help="Path to the dataset config file (default: datasets/config.json).",
    )
    # base URL
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.gpt.ge/v1/",
        help="Base URL of the API in OpenAI format.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-RWrbcDyifkghniRiE1FeC79a9c1244F19063AaC7266c3dA2",
        help="API key.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Name of the model (e.g., 'gpt-4o-2024-11-20', 'gemini-2.5-flash-preview-04-17').",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default='prompts/dior/open_vocabulary.txt',
        required=True,
        help="Path to the prompt file.",
    )
    parser.add_argument(
        "--json_output",
        type=bool,
        default=False,
        help="Whether to output the results in JSON format.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to the save directory (default: ./object_counts).",
    )
    args = parser.parse_args()

    return args


async def create_completion(session, prompt, img_path, model_name, json_output=False, save_dir=None):
    try:
        img_name = os.path.basename(img_path)
        encoded_image = encode_image_as_base64(img_path)
        message = {
            "model": model_name,
            "max_tokens": 2000,
            "temperature": 0.01,
            "top_p": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encoded_image
                            }
                        }
                    ],
                }
            ],
        }
        if json_output:
            message["response_format"] = {"type": "json_object"}

        async with session.post(
            url=f"{openai.base_url}chat/completions",
            json=message,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                response = result['choices'][0]['message']['content']
                print(f'get response for {img_name}')
                pred_counts = extract_dict_from_string(response)
                if pred_counts == -1:
                    failed_analysis.append(img_name)
                else:
                    save_count(pred_counts, save_dir=save_dir,
                               img_name=img_name)
            else:
                print(f"Request failed, status code: {response.status}")
                failed_requests.append(img_name)  # record failed requests
    except Exception as e:
        print(f"Request failed: {e}")
        failed_requests.append(img_name)  # record failed requests


async def asinc_count(prompt, img_list, model_name, max_limits=10, json_output=False, save_dir='./defaut_save'):
    """
    Asynchronous processing of image lists, exit after traversal.
    """
    print(f'Processing {len(img_list)} images...')
    async with aiohttp.ClientSession() as session:
        # batch processing
        for i in range(0, len(img_list), max_limits):
            batch = img_list[i:i + max_limits]  # process a batch
            tasks = [create_completion(
                session, prompt, img_path, model_name, json_output, save_dir) for img_path in batch]
            # concurrent execution of the current batch
            await asyncio.gather(*tasks)
            print(
                f"Processed batch {i // max_limits + 1} / {((len(img_list) - 1) // max_limits) + 1}")
            # wait 0.5 seconds after each batch
            await asyncio.sleep(SEND_INTERVAL)
        print("All images processed.")
    # print failed images
    if failed_requests:
        print("\nFailed images:")
        for img_name in failed_requests:
            print(img_name)
        with open("failed_requests.txt", "w") as f:
            f.write("\n".join(failed_requests))
    if failed_analysis:
        print("\nFailed images:")
        for img_name in failed_analysis:
            print(img_name)
        with open("failed_analysis.txt", "w") as f:
            f.write("\n".join(failed_analysis))


if __name__ == '__main__':

    args = parse_args()

    model = args.model
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    openai.api_key = args.api_key
    openai.base_url = args.base_url
    json_output = args.json_output

    dataset = args.dataset_name
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)
    dataset_info = dataset_config[dataset]
    ann_path = dataset_info['ann_path']
    img_dir = dataset_info['img_dir']

    with open(args.prompt_path, 'r') as f:
        prompt = f.read()

    save_dir = args.save_dir
    if save_dir is None:
        base_name = os.path.basename(args.prompt_path).split('.')[0]
        save_dir = f'./object_counts/{dataset}/{model}_{base_name}'

    os.makedirs(save_dir, exist_ok=True)
    with open(ann_path, 'r') as f:
        print(f'Loading annotations from {ann_path}')
        annotations = json.load(f)
    file_names = [image['file_name'] for image in annotations['images']]

    # skip existing files in save_dir
    existing_files = set(os.listdir(save_dir))
    file_names = [
        file for file in file_names if f"{os.path.splitext(file)[0]}.json" not in existing_files]

    print(f"\nProcessing {len(file_names)} images...\n")
    img_path_list = [os.path.join(img_dir, image_name)
                     for image_name in file_names]

    print(f'Using model {model}\nSaving Results to {save_dir}\n')
    print(f'Prompt: {prompt}')
    time.sleep(6)
    tic = time.time()
    asyncio.run(asinc_count(prompt, img_list=img_path_list, model_name=model,
                max_limits=3000, json_output=json_output, save_dir=save_dir))
    toc = time.time()
    print(f'Time taken: {toc - tic} seconds')
