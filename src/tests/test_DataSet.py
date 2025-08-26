import unittest
from DataSet.Pretokenize import PretokenizedDataset, PretokenizedMagic
import numpy as np
import os

from utils import get_logger

logger = get_logger(name=__name__)

ASSETDIR = "tests/assets"
ARANGE_FILENAME = "pretokenized_arange.bin"
ARANGE2_FILENAME = "pretokenized_arange2.bin"
UNEVEN_FILENAME = "pretokenized_uneven.bin"
PADDED_FILENAME = "pretokenized_padded.bin"
MAGIC_FILENAME = "pretokenized_magic.bin"
ROWS = 10
CONTEXT_LENGTH = 5
UNEVEN_OFFSET = 2


samplefiles = {}


def get_testfile(filename: str) -> str:
    return os.path.join(ASSETDIR, filename)


def getsampledata(samplename: str) -> np.ndarray:
    return samplefiles[get_testfile(samplename)]


class TestPretokenizedDataset(unittest.TestCase):
    """
    Tests whether PretokenizedDataset works as intended.
    For this we write np.aranges to a file and then read them back in.

    The most critical part is probably retrieval of the last sample, which may not be a full sample.
    """

    @classmethod
    def setUpClass(cls):
        os.makedirs(ASSETDIR, exist_ok=True)
        cls._create_testfiles()

    @classmethod
    def _create_testfiles(cls):
        file = get_testfile(ARANGE_FILENAME)
        sampleFile = np.memmap(file, dtype=np.uint16, mode='w+', shape=(ROWS, CONTEXT_LENGTH))
        for i in range(ROWS):
            sampleFile[i] = np.arange(i * CONTEXT_LENGTH, (i + 1) * CONTEXT_LENGTH, dtype=np.uint16)
        sampleFile.flush()
        samplefiles[file] = np.copy(sampleFile)  # store np array for later assertions
        del sampleFile

        file = get_testfile(UNEVEN_FILENAME)
        uneven_file = np.memmap(file, dtype=np.uint16, mode='w+', shape=(ROWS * CONTEXT_LENGTH - UNEVEN_OFFSET,))
        for i in range(ROWS * CONTEXT_LENGTH - 2):
            uneven_file[i] = i
        uneven_file.flush()
        samplefiles[file] = np.copy(uneven_file)  # store np array for later assertions
        del uneven_file

        file = get_testfile(ARANGE2_FILENAME)
        sampleFile = np.memmap(file, dtype=np.uint16, mode='w+', shape=(ROWS // 2, CONTEXT_LENGTH))
        for i in range(ROWS // 2):
            sampleFile[i] = np.arange((i + 17) * CONTEXT_LENGTH, (i + 17 + 1) * CONTEXT_LENGTH, dtype=np.uint16)
        sampleFile.flush()
        samplefiles[file] = np.copy(sampleFile)  # store np array for later assertions
        del sampleFile

        file = get_testfile(PADDED_FILENAME)
        sampleFile = np.memmap(file, dtype=np.uint16, mode='w+', shape=(ROWS * CONTEXT_LENGTH))
        for i in range(ROWS * CONTEXT_LENGTH):
            sampleFile[i] = 0 if i > ROWS*CONTEXT_LENGTH - 5 else i  # 4 padding tokens (0)
        sampleFile.flush()
        samplefiles[file] = np.copy(sampleFile)  # store np array for later assertions
        del sampleFile

        file = get_testfile(MAGIC_FILENAME)
        sampleFile = np.memmap(file, dtype=np.int64, mode='w+', shape=(32,))
        sampleFile[PretokenizedMagic.OFFSET // 8:] = np.arange(32 - PretokenizedMagic.OFFSET // 8, dtype=np.int64)
        magic = PretokenizedMagic(dtype=sampleFile.dtype, isPadded=True)
        magic.write(sampleFile)
        sampleFile.flush()
        samplefiles[file] = np.copy(sampleFile[PretokenizedMagic.OFFSET // 8:])  # store np array for later assertions
        del sampleFile

    @classmethod
    def tearDownClass(cls):
        for filename in samplefiles.keys():
            if not os.path.commonpath([filename, ASSETDIR]) == ASSETDIR:
                logger.warning(f"Refusing to delete file {filename} outside of asset directory {ASSETDIR}")
            if os.path.exists(filename):
                os.unlink(filename)
        samplefiles.clear()

    def test_pretokenized_dataset(self):

        dataset = PretokenizedDataset(get_testfile(ARANGE_FILENAME),
                                      dtype=np.uint16,
                                      context_length=CONTEXT_LENGTH,
                                      pad_token_id=0,
                                      load_in_memory=False,
                                      isPadded=False)
        self.assertEqual(len(dataset), ROWS)
        # Dataset iteration never terminates by design, so we do not enumerate
        for i in range(len(dataset)):
            sample = dataset[i]["input_ids"]
            self.assertTrue(np.array_equal(sample, np.arange(i * CONTEXT_LENGTH, (i + 1) * CONTEXT_LENGTH, dtype=np.uint16)),
                            f"Sample {i} is not correct: {sample}")
            self.assertTrue(np.array_equal(dataset[i]["attention_mask"], np.ones(CONTEXT_LENGTH, dtype=np.uint8)),
                            f"Attention mask for sample {i} is not correct: {sample}")

    def test_pretokenized_dataset_inmemory(self):

        dataset = PretokenizedDataset(get_testfile(ARANGE_FILENAME),
                                      dtype=np.uint16,
                                      pad_token_id=0,
                                      context_length=CONTEXT_LENGTH,
                                      load_in_memory=True,
                                      isPadded=False)
        self.assertEqual(len(dataset), ROWS)
        # Dataset iteration never terminates by design, so we do not enumerate
        for i in range(len(dataset)):
            sample = dataset[i]["input_ids"]
            self.assertTrue(np.array_equal(sample, np.arange(i * CONTEXT_LENGTH, (i + 1) * CONTEXT_LENGTH, dtype=np.uint16)),
                            f"Sample {i} is not correct: {sample}")
            self.assertTrue(np.array_equal(dataset[i]["attention_mask"], np.ones(CONTEXT_LENGTH, dtype=np.uint8)),
                            f"Attention mask for sample {i} is not correct: {sample}")

    def test_lastSample(self):
        # check if last sample is correctly padded when necessary
        PAD_TOKEN_ID = 1337
        dataset = PretokenizedDataset(get_testfile(UNEVEN_FILENAME),
                                      dtype=np.uint16,
                                      pad_token_id=PAD_TOKEN_ID,
                                      context_length=CONTEXT_LENGTH,
                                      load_in_memory=True,
                                      isPadded=False)
        self.assertEqual(len(dataset), ROWS)
        for i in range(len(dataset) - 1):
            sample = dataset[i]
            self.assertEqual(sample["input_ids"].shape[0], CONTEXT_LENGTH)
            self.assertTrue(np.array_equal(sample["input_ids"], np.arange(i * CONTEXT_LENGTH, (i + 1) * CONTEXT_LENGTH, dtype=np.uint16)),
                            f"Sample {i} is not correct: {sample}")
            self.assertTrue(np.array_equal(sample["attention_mask"], np.ones(CONTEXT_LENGTH, dtype=np.uint8)),
                            f"Attention mask for sample {i} is not correct: {sample}")
        # check last sample
        # should contain CONTEXT_LENGTH - UNEVEN_OFFSET elements from the arange and then UNEVEN_OFFSET padding elements
        # attention_mask should appropriately reflect this
        lastsample = dataset[len(dataset) - 1]
        self.assertEqual(lastsample["input_ids"].shape[0], CONTEXT_LENGTH)
        self.assertTrue(np.array_equal(lastsample["input_ids"][:-UNEVEN_OFFSET], np.arange((ROWS - 1) * CONTEXT_LENGTH, ROWS * CONTEXT_LENGTH - UNEVEN_OFFSET, dtype=np.uint16)),
                        f"Last sample is not correct: {lastsample}")
        self.assertTrue(np.array_equal(lastsample["input_ids"][-UNEVEN_OFFSET:], np.full(UNEVEN_OFFSET, PAD_TOKEN_ID, dtype=np.uint16)),
                        f"Last sample is not correctly padded: {lastsample}")
        self.assertTrue(np.array_equal(lastsample["attention_mask"][:-UNEVEN_OFFSET], np.ones(CONTEXT_LENGTH - UNEVEN_OFFSET, dtype=np.uint8)),
                        f"Attention mask for last sample is not correct (0 where it should be 1): {lastsample}")
        self.assertTrue(np.array_equal(lastsample["attention_mask"][-UNEVEN_OFFSET:], np.zeros(UNEVEN_OFFSET, dtype=np.uint8)),
                        f"Attention mask for last sample is not correct (1 where it should be 0): {lastsample}")

    def test_multifile(self):
        PAD_TOKEN_ID = 1337
        dataset = PretokenizedDataset([get_testfile(ARANGE_FILENAME), get_testfile(ARANGE2_FILENAME)],
                                      dtype="uint16",
                                      pad_token_id=PAD_TOKEN_ID,
                                      context_length=CONTEXT_LENGTH,
                                      isPadded=False)
        actualdata_1 = getsampledata(ARANGE_FILENAME)
        actualdata_2 = getsampledata(ARANGE2_FILENAME)
        self.assertEqual(len(dataset), actualdata_1.shape[0] + actualdata_2.shape[0])

        # Currently this is just the arange test two times in a row
        for i in range(ROWS):
            sample = dataset[i]["input_ids"]
            self.assertTrue(np.array_equal(sample, actualdata_1[i]),
                            f"Sample {i} is not correct: {sample}")
            self.assertTrue(np.array_equal(dataset[i]["attention_mask"], np.ones(CONTEXT_LENGTH, dtype=np.uint8)),
                            f"Attention mask for sample {i} is not correct: {sample}")

        for i in range(ROWS, ROWS + actualdata_2.shape[0]):
            sample = dataset[i]["input_ids"]
            self.assertTrue(np.array_equal(sample, actualdata_2[i - ROWS]),
                            f"Sample {i} is not correct: {sample}")
            self.assertTrue(np.array_equal(dataset[i]["attention_mask"], np.ones(CONTEXT_LENGTH, dtype=np.uint8)),
                            f"Attention mask for sample {i} is not correct: {sample}")

    def test_padded(self):
        PAD_TOKEN_ID = 0
        dataset = PretokenizedDataset(get_testfile(PADDED_FILENAME),
                                      dtype="uint16",
                                      pad_token_id=PAD_TOKEN_ID,
                                      context_length=CONTEXT_LENGTH,
                                      isPadded=True)
        actualdata = getsampledata(PADDED_FILENAME)
        self.assertEqual(len(dataset), actualdata.shape[0] // CONTEXT_LENGTH)
        for i in range(ROWS * CONTEXT_LENGTH):
            self.assertEqual(dataset[i // CONTEXT_LENGTH]["input_ids"][i % CONTEXT_LENGTH], actualdata[i])
            self.assertEqual(dataset[i // CONTEXT_LENGTH]["attention_mask"][i % CONTEXT_LENGTH], 1 if i <= ROWS * CONTEXT_LENGTH - 5 else 0)

    def test_magic(self):
        actualdata = getsampledata(MAGIC_FILENAME)
        dataset = PretokenizedDataset(get_testfile(MAGIC_FILENAME),
                                      pad_token_id=actualdata[-1], # we act as if the last token was padding to test if magic set isPadding correctly
                                      context_length=1)
        self.assertEqual(len(dataset), actualdata.shape[0])
        for i in range(actualdata.shape[0]):
            self.assertEqual(dataset[i]["input_ids"], actualdata[i])
            self.assertEqual(dataset[i]["attention_mask"], 1 if i < actualdata.shape[0] - 1 else 0)


if __name__ == '__main__':
    unittest.main()
