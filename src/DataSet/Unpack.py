from datasets import load_dataset, disable_caching
import os
import argparse
from utils import get_logger
from tqdm import tqdm
import re

logger = get_logger(__name__)

DESCRIPTION = """This script takes a huggingface dataset and unpacks it into a directory structure that can be used by
tools which work on a file level"""


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dataset', required=True, help='The name or path of the dataset to unpack')
    parser.add_argument('--split', default='train', help='The split to unpack')
    parser.add_argument('--contentfield', default='content', help='The name of the field containing the file content')
    parser.add_argument('--filenamefield', help='The name of the field containing the file name')
    parser.add_argument('--streaming', action='store_true', help='Whether to stream the dataset')
    parser.add_argument('--tqdm', action='store_true', help='Whether to show a progress bar')
    parser.add_argument('--add_to_existing', action='store_true', help='Whether to add to an existing dataset')
    parser.add_argument('-o', '--output', required=True, help='The directory to unpack the dataset into')
    args = parser.parse_args()

    disable_caching()
    logger.info(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    os.makedirs(args.output, exist_ok=True)

    # if dir is not empty, find the last file and start from there
    offset = 0
    if os.listdir(args.output):
        if not args.add_to_existing:
            logger.error(f"Directory {args.output} is not empty, aborting")
            exit(1)
        regex = re.compile(r"(\d+).py$")
        last_file = sorted(os.listdir(args.output), key=lambda x: int(regex.search(x).group(1)))[-1]
        last_file = int(regex.search(last_file).group(1))
        offset = last_file + 1
        logger.info(f"Found {offset} files in {args.output}, starting from {offset}")

    kwargs = {'total': len(dataset)} if not args.streaming else {}
    for i, sample in tqdm(enumerate(dataset), disable=not args.tqdm, **kwargs):
        index = i + offset
        prefix = "file"
        content = sample[args.contentfield]
        if not content:
            logger.warning(f"Sample {i} has no content")
            continue
        if args.filenamefield in sample:
            prefix = sample[args.filenamefield]
        virtual_filename = f"{prefix}_{index:010d}.py"
        filename = os.path.join(args.output, virtual_filename)
        with open(filename, 'w') as f:
            f.write(sample[args.contentfield])
    logger.info("Done")


if __name__ == '__main__':
    main()
