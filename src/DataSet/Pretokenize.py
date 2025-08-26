from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import argparse
from typing import Union, Optional, Iterable
from utils import get_logger

logger = get_logger(name=__name__)

DESCRIPTION = """
This script takes a HuggingFace tokenizer and dataset
and outputs a pretokenized dataset as serialized numpy arrays.

The PretokenizedDataset class can be used to load the dataset.

Example Usage:
python -m DataSet.Pretokenize --add_eos -m 64 -t 'Salesforce/codegen-350M-multi' -d $EXPERIMENT_ROOT/dataset/train -o $EXPERIMENT_ROOT/dataset/train_tokenized.bin
"""


class PretokenizedMagic(object):
    """
    Helper class to simplify the interfaces.
    We write some magic/metadata at the start of the binary file which describe the format.
    """
    MAGIC = b"PTDS"
    VERSION = 1
    OFFSET = 16

    def __init__(self, dtype: np.dtype, isPadded: bool, version: Optional[int] = None):
        self.dtype = dtype
        self.isPadded = isPadded
        self.version = PretokenizedMagic.VERSION if version is None else version

    def write(self, memmap: np.memmap):
        """
        Write magic to beginning of opened memmap.
        """

        # memmap needs a length of at least 16 bytes
        assert memmap.size * memmap.dtype.itemsize >= PretokenizedMagic.OFFSET
        a = np.ndarray(shape=(4,), dtype=np.uint32, buffer=memmap)
        a[0] = int.from_bytes(PretokenizedMagic.MAGIC, byteorder="little")
        a[1] = self.version
        a[2] = int(self.isPadded)
        a[3] = int.from_bytes(bytes(self.dtype.str, encoding='ascii'), byteorder="big")

    @classmethod
    def from_memmap(cls, memmap: Union[np.memmap, str]):
        """
        Read magic from loaded memmap or binary file.
        Returns None if no match.
        """
        if isinstance(memmap, str):
            memmap = np.memmap(memmap, dtype=np.uint32, mode="r", shape=(4,))
        assert memmap.size * memmap.dtype.itemsize >= cls.OFFSET
        a = np.ndarray(shape=(4,), dtype=np.uint32, buffer=memmap)
        if not a[0] == int.from_bytes(PretokenizedMagic.MAGIC, byteorder="little"):
            logger.warning("Did not find magic bytes in pretokenized file.")
            return None
        version = a[1]
        assert version == PretokenizedMagic.VERSION, f"Unsupported version {a[1]}"
        isPadded = bool(a[2])
        dtype = np.dtype(int(a[3]).to_bytes(length=4, byteorder="big").lstrip(b"\x00"))
        logger.debug("Magic: dtype={dtype}, isPadded={isPadded}")
        return cls(dtype=dtype, isPadded=isPadded, version=version)


class PretokenizedDataset(Dataset):
    """
    This class loads a pretokenized dataset binary files.
    By default, it does not perform any explicit buffering and instead relies on the
    operating system's file cache to keep the file in memory (mmap).

    The stride allows to use overlapping samples to get more samples from the same file.
    Parameters:
        path: path or list of paths to the binary file(s)
        dtype: numpy dtype of the binary file
        context_length: number of tokens per sample
        pad_token_id: id of the padding token
        stride: stride between samples (default: context_length)
        load_in_memory: load the entire dataset into memory (default: False)
        isPadded: whether the dataset contains padded sequences (default: False)
    """

    class DataContainer(object):
        # helper class which manages a single pretokenized file for us
        def __init__(self,
                     path: str,
                     context_length: int,
                     dtype: Optional[np.dtype] = None,
                     stride: int = None,
                     pad_token_id: int = None,
                     load_in_memory: bool = False,
                     isPadded: Optional[bool] = None):

            if stride is not None:
                assert stride <= context_length
            assert os.path.isfile(path), f"File {path} does not exist."

            self.byte_offset = 0
            magic = PretokenizedMagic.from_memmap(path)
            if magic is not None:
                logger.debug(f"Detected magic for {path}: dtype={magic.dtype}, isPadded={magic.isPadded}")
                isPadded = magic.isPadded
                dtype = magic.dtype
                self.byte_offset = PretokenizedMagic.OFFSET
            else:
                assert dtype is not None, "dtype neither in magic nor parameters"
                assert isPadded is not None, "isPadded neither in magic nor parameters"
                logger.debug(f"{path} does not contain magic")
            self.data = np.memmap(path, dtype=dtype, mode="r", offset=self.byte_offset)
            if isPadded:
                assert self.data.shape[0] % context_length == 0, f"File {path} is not a multiple of context length {context_length}."

            if stride is None:
                stride = context_length
            self.length = (self.data.shape[0] - context_length) // stride + 1
            assert self.length > 0, f"File {path} is too short for context length {context_length}."
            if (self.data.shape[0] - context_length) % stride != 0:
                # we fix this later on item retrieval by padding the last sample
                self.length += 1
            if load_in_memory:
                self._load_into_memory()
            self.pad_token_id = pad_token_id
            self.stride = stride
            self.context_length = context_length
            self.isPadded = isPadded

        def _load_into_memory(self):
            # TODO: is this really sane?
            self.data = np.copy(self.data)

        def __getitem__(self, idx):
            assert idx < self.length
            start = idx * self.stride
            end = start + self.context_length
            if self.isPadded:
                input_ids = torch.from_numpy(self.data[start:end].astype(np.int64))
                assert input_ids.shape[0] == self.context_length
                # attention mask is 0 for padding tokens
                attention_mask = torch.ones_like(input_ids)
                # For all trailing padding tokens, we set the attention mask to 0
                # This implementation is pretty wasteful, but all vectorized attempts were pretty much
                # just as inefficient (and much harder to read)
                for i in range(self.context_length - 1, -1, -1):
                    if input_ids[i] != self.pad_token_id:
                        break
                    attention_mask[i] = 0

                return {"input_ids": input_ids, "attention_mask": attention_mask}

            if idx < self.length - 1:
                # regular sample without padding being necessary
                # we still need an attention mask so the dataloader can stack the samples
                attention_mask = torch.ones(self.context_length, dtype=torch.long)
                return {"input_ids": torch.from_numpy(self.data[start:end].astype(np.int64)),
                        "attention_mask": attention_mask}
            else:
                # the very last sample might need padding
                out = torch.full((self.context_length,), fill_value=self.pad_token_id, dtype=torch.long)
            out[:self.data.shape[0] - start] = torch.from_numpy(self.data[start:].astype(np.int64))
            attention_mask = torch.zeros(self.context_length, dtype=torch.long)
            attention_mask[:self.data.shape[0] - start] = 1
            return {"input_ids": out, "attention_mask": attention_mask}

        def __len__(self):
            return self.length

    def __init__(self,
                 path: Union[str, list[str]],
                 context_length,
                 pad_token_id: int,
                 dtype: Optional[Union[np.dtype, str]] = None,
                 stride=None,
                 load_in_memory=False,
                 isPadded: Optional[bool] = None):
        if isinstance(path, str):
            path = [path]
        for f in path:
            assert os.path.isfile(f), f"File {f} does not exist."

        if isinstance(dtype, str):
            # cast to numpy dtype
            # (this option is nice because now the client does not need to import numpy)
            dtype = np.dtype(dtype)

        # We keep context_length around solely for assertions
        self.context_length = context_length
        self._load_data(path,
                        context_length=context_length,
                        load_in_memory=load_in_memory,
                        dtype=dtype,
                        stride=stride,
                        pad_token_id=pad_token_id,
                        isPadded=isPadded)

    def _load_data(self,
                   path: list[str],
                   context_length: int,
                   load_in_memory: bool,
                   dtype: Optional[np.dtype],
                   stride: int,
                   isPadded: Optional[bool],
                   pad_token_id: int = None):
        logger.debug(f"Loading data from {len(path)} files.")
        self.data = [PretokenizedDataset.DataContainer(path=f,
                                                       dtype=dtype,
                                                       context_length=context_length,
                                                       stride=stride,
                                                       pad_token_id=pad_token_id,
                                                       load_in_memory=load_in_memory,
                                                       isPadded=isPadded) for f in path]
        self.isPadded = self.data[0].isPadded
        self.pad_token_id = pad_token_id
        self.stride = stride
        self.context_length = context_length
        self.length = sum([len(d) for d in self.data])
        logger.debug(f"Loaded {self.length} samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert idx < self.length and idx >= 0

        # in practise we will load 2-3 files max, so this loop is fine in terms of overhead
        # if we had many files, we would create a running sum array over the container lengths
        # and use binary search to find the correct container
        for d in self.data:
            if idx < d.length:
                sample = d[idx]
                assert sample["input_ids"].shape[0] == self.context_length, f"Unexpected sample length: {sample['input_ids'].shape[0]} instead of {self.context_length}"
                return sample
            else:
                idx -= d.length
            assert idx >= 0


class BinaryPretokenizer(object):
    """
    This class accepts samples (strings) and serializes them to a binary file

    target_file: str - the file to write to
    tokenizer: transformers.AutoTokenizer - the tokenizer to use
    buffersamples: int - the number of samples to buffer before writing to disk
    add_eos: bool - whether to add an EOS token to the end of each sample
    pad_samples: bool - whether to pad samples to the maximum length of the buffer (implies add_eos = False)
    return_overflowing_tokens: bool - Allows to draw multiple tokenized samples from a single input sample
    min_sample_length: int - the minimum length of a sample (shorter samples will be discarded)
    """

    def __init__(self,
                 target_file: str,
                 tokenizer: AutoTokenizer,
                 buffersamples: int,
                 add_eos: bool,
                 pad_samples: bool,
                 return_overflowing_tokens: bool = True,
                 min_sample_length: int = 0,
                 output_sample_offsets: bool = False):
        assert os.path.isdir(os.path.dirname(target_file)), \
            f"Target file directory {os.path.dirname(target_file)} does not exist."
        if add_eos:
            assert tokenizer.eos_token_id is not None, "Tokenizer does not have an EOS token."
        self.target_file = target_file
        self.tokenizer = tokenizer
        self.np_dtype = np.dtype(np.uint16 if tokenizer.vocab_size < 2**16 else np.uint32)
        logger.info(f"Using dtype {self.np_dtype} for {tokenizer.vocab_size} tokens")
        self.sample_buffer = []
        self.offset_bytes = 0  # Keep track of current offset at which we can start appending new samples
        self.add_eos = add_eos
        self.buffersamples = buffersamples
        self.pad_samples = pad_samples
        self.min_sample_length = min_sample_length
        self.return_overflowing_tokens = return_overflowing_tokens
        self.output_sample_offsets = output_sample_offsets
        if self.output_sample_offsets:
            self.sample_lengths = [0]  # we prepend 0 so cumsum does what we want later
        assert not self.pad_samples or tokenizer.pad_token_id is not None, "Tokenizer does not have a pad token."
        if self.pad_samples and self.add_eos:
            logger.warning("pad_samples and add_eos are both set to True. This is not supported. "
                           "Setting add_eos to False.")
            self.add_eos = False
        self.__writeMagic()

    def __writeMagic(self):
        # Write magic bytes at the start of the file so we can load it more easily later

        magic = PretokenizedMagic(self.np_dtype, self.pad_samples)
        # dtype doesn't matter, we use byte for convencience so OFFSET is the correct size
        assert not os.path.isfile(self.target_file), f"Target file {self.target_file} already exists."
        m = np.memmap(self.target_file, dtype=np.int8, mode="w+", shape=(PretokenizedMagic.OFFSET,))
        magic.write(m)
        m.flush()

        # We need to adjust offset so the magic does not get overwritten by samples
        self.offset_bytes += PretokenizedMagic.OFFSET

    def addSample(self, sample: str) -> None:
        self.sample_buffer.append(sample)
        if len(self.sample_buffer) >= self.buffersamples:
            self._flushBuffer()
            assert len(self.sample_buffer) == 0

    def _flushBuffer(self):
        logger.debug(f"Tokenizing buffer of {len(self.sample_buffer)} samples to {self.target_file}")

        # We cannot directly use numpy arrays because the tokenized sequences have different lengths.
        # Also: Tokenizer__call__ automatically parallelizes if given more than 1 sample
        tokenized = self.tokenizer(self.sample_buffer,
                                   padding=False,
                                   add_special_tokens=False,
                                   truncation=False if self.return_overflowing_tokens else True)["input_ids"]
        sampleLen = self.tokenizer.model_max_length

        def sampleGen():
            # apply necessary transformations to each sample before yielding
            for v in tokenized:
                if len(v) < self.min_sample_length:
                    continue
                if self.add_eos:
                    v.append(self.tokenizer.eos_token_id)
                elif self.pad_samples:
                    lastSampleLen = len(v) % sampleLen
                    if lastSampleLen >= self.min_sample_length:
                        required_padding = (sampleLen - (lastSampleLen)) % sampleLen
                        v.extend([self.tokenizer.pad_token_id] * required_padding)
                    else:
                        # when padding, we need to remove the last sample if it is too short
                        # (consider a sample with 2050 tokens; when context length is 2048,
                        # it is no use to create a sample with 2 tokens + 2022 padding)
                        v = v[:-lastSampleLen]
                        if len(v) == 0:
                            continue
                        assert len(v) % sampleLen == 0
                assert len(v) > 0
                if self.output_sample_offsets:
                    self.sample_lengths.append(len(v))
                yield v

        tokenized = [np.array(t, dtype=self.np_dtype) for t in sampleGen()]

        # empty lists naturally disappear in np.concatenate
        tokenized = np.concatenate(tokenized)
        if self.pad_samples:
            assert tokenized.shape[0] % self.tokenizer.model_max_length == 0

        logger.debug(f"Tokenized {tokenized.shape[0]} tokens, writing to {self.target_file}")
        writeOffset = self.offset_bytes
        mode = "w+" if self.offset_bytes == 0 else "r+"
        serialized = np.memmap(self.target_file,
                               dtype=self.np_dtype,
                               mode=mode,
                               offset=writeOffset,
                               shape=(tokenized.shape[0]))
        serialized[:] = tokenized[:]
        # close memmap
        serialized.flush()
        del serialized
        logger.debug(f"Flushed {tokenized.shape[0]} tokens to {self.target_file} at offset {writeOffset}")
        self.offset_bytes += tokenized.shape[0] * np.dtype(self.np_dtype).itemsize
        self.sample_buffer.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug(f"Closing pretokenizer, serializing {len(self.sample_buffer)} leftover samples")
        if len(self.sample_buffer) > 0:
            self._flushBuffer()
        if self.output_sample_offsets:
            logger.info(f"Writing sample ids to {self.target_file}.offsets")
            sampleoffsets = np.cumsum(self.sample_lengths, dtype=np.uint64)
            np.save(f"{self.target_file}.offsets", sampleoffsets)


def detokenize(pretokenizedpath: str,
               outputpath: str,
               tokenizer: PreTrainedTokenizerBase,
               compress: bool = False):
    from tqdm import tqdm
    # revert the pretokenization process
    assert os.path.isfile(pretokenizedpath), f"File {pretokenizedpath} does not exist."
    # memmap the pretokenized file
    pretokenized = np.memmap(pretokenizedpath, dtype=np.int8, mode="r")  # just so we can read magic
    # read magic
    magic = PretokenizedMagic.from_memmap(pretokenized)
    # close memmap
    del pretokenized
    # memmap the pretokenized file with correct params
    pretokenized = np.memmap(pretokenizedpath, dtype=magic.dtype, mode="r", offset=magic.OFFSET)
    # open output serializer
    from DataSet.Serialization import SerializationBuffer

    def addsample(serializer, samplestart, sampleend):
        decoded = tokenizer.decode(pretokenized[samplestart:i], skip_special_tokens=True)
        logger.debug(f"Decoded sample from {samplestart} to {i} with length {len(decoded)} characters")
        serializer.addSample(decoded)

    seperatorTokens = [t for t in [tokenizer.sep_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id] if t is not None]

    def isSeparator(token_id):
        return token_id in seperatorTokens

    # check if any token in pretokenized is a separator token
    if not any((isSeparator(t) for t in pretokenized)):
        raise ValueError(f"File {pretokenizedpath} does not contain any separator tokens. Will not detokenize.")

    with SerializationBuffer(directory=outputpath, compress=compress) as serializer:
        # iterate of pretokenized samples until pad or eos token is encountered, then decode and serialize
        samplestart = 0
        for i, tokenid in enumerate(tqdm(pretokenized)):
            assert i >= samplestart, f"invariant violated: i={i}, samplestart={samplestart}"
            issep = isSeparator(tokenid)
            if not issep:
                if i < pretokenized.shape[0] - 1:
                    continue
                else:
                    # last sample, decode and serialize
                    addsample(serializer, samplestart, i)
            elif issep and samplestart < i:
                # eos/pad token encountered, end of current sample
                addsample(serializer, samplestart, i)
                samplestart = i + 1
            elif issep and samplestart == i:
                # fast forward through padding
                samplestart = i + 1


def pretokenize(dataset: Iterable[str],
                tokenizer: AutoTokenizer,
                target_file: str,
                add_eos: bool,
                pad_samples: bool,
                key: Optional[str] = None,
                buffersamples: int = 10**5,
                return_overflowing_tokens: bool = True,
                min_sample_length: int = 0,
                output_sample_offsets: bool = False):
    with BinaryPretokenizer(target_file=target_file,
                            tokenizer=tokenizer,
                            buffersamples=buffersamples,
                            add_eos=add_eos,
                            pad_samples=pad_samples,
                            min_sample_length=min_sample_length,
                            return_overflowing_tokens=return_overflowing_tokens,
                            output_sample_offsets=output_sample_offsets) as pretokenizer:
        def sampleGen():
            if key is None:
                for sample in dataset:
                    yield sample
            else:
                for sample in dataset:
                    yield sample[key]
        for sample in sampleGen():
            pretokenizer.addSample(sample)


def detokenize_main(args):
    assert os.path.isfile(args.dataset), f"File {args.dataset} does not exist."
    if os.path.isdir(args.out):
        args.out = os.path.join(args.out, os.path.basename(args.dataset))
        logger.info(f"Output path is a directory, using {args.out} as output dir")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        logger.warning("Tokenizer does not have a pad token, setting to eos token id")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    detokenize(pretokenizedpath=args.dataset,
               outputpath=args.out,
               tokenizer=tokenizer,
               compress=True)
    print("Done.")
    exit(0)


def main():
    global logger
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-t", "--tokenizer", type=str, required=True, help="HuggingFace tokenizer name or path")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="HuggingFace dataset name or path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument("-k", "--key", type=str, default="content", help="Dataset key to use (default: content)")
    parser.add_argument("-o", "--out", type=str, required=True, help="Output file")
    parser.add_argument("-b", "--buffersamples", type=int, default=10000, help="Number of samples to buffer before writing")
    parser.add_argument("-p", "--pad_samples", action=argparse.BooleanOptionalAction, default=False, help="Pad samples to the maximum length of the buffer (implies add_eos = False)")
    parser.add_argument("-m", "--min_sample_length", type=int, default=0, help="Minimum length of a sample (default: 0)")
    parser.add_argument("--return_overflowing_tokens", type=bool, action=argparse.BooleanOptionalAction, default=True, help="Return overflowing tokens (default: True)")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Use dataset streaming (default: True)")
    parser.add_argument("--add_eos", action=argparse.BooleanOptionalAction, default=True, help="Add EOS token to each sequence (default: True)")
    parser.add_argument("--loglevel", type=str, default="INFO", help="Log level")
    parser.add_argument("--invert", action="store_true", help="Invert the script function (detokenize pretokenized dataset)")
    parser.add_argument("--output_sample_offsets", action="store_true", help="Output sample offsets to map pretokenized samples to original samples")
    args = parser.parse_args()

    if args.invert:
        # we added this function after the fact, so we factor this out here to not make it
        # a complete maintenance nightmare
        detokenize_main(args)
        exit(0)

    logger = get_logger(name=__name__, localLevel=args.loglevel.upper())

    assert os.path.isdir(os.path.dirname(os.path.abspath(args.out)))
    assert not os.path.isfile(args.out), f"Output file {args.out} already exists."
    if args.add_eos and args.pad_samples:
        logger.warning("pad_samples and add_eos are both set to True. This is not supported. "
                       "Setting add_eos to False.")
        args.add_eos = False

    # if dataset has no explicit split, "train" always gets loaded. This is the case for CodeParrot and all our custom splits.
    dataset = load_dataset(args.dataset, streaming=args.streaming, split=args.split)

    # dataset.column_names is a list of all keys in the dataset, but is not reliably available for streaming datasets
    sample = next(iter(dataset))
    assert args.key in sample, f"Dataset does not have key {args.key}"
    key = args.key

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.pad_samples and tokenizer.pad_token_id is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos token.")
        tokenizer.pad_token = tokenizer.eos_token

    pretokenize(dataset=dataset, key=key,
                tokenizer=tokenizer,
                target_file=args.out,
                buffersamples=args.buffersamples,
                add_eos=args.add_eos,
                pad_samples=args.pad_samples,
                min_sample_length=args.min_sample_length,
                return_overflowing_tokens=args.return_overflowing_tokens,
                output_sample_offsets=args.output_sample_offsets)

    # sanity check
    logger.info(f"Done, pretokenized dataset saved to {args.out}")


if __name__ == "__main__":
    main()
