import random
import typing as tp

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class GReaTDataset(Dataset):
    """GReaT Dataset

    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """

    def set_tokenizer(self, tokenizer):
        """Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

    def _getitem(
        self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs
    ) -> tp.Union[tp.Dict, tp.List]:
        """Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        # If the key is an int, it is used as an index to retrieve a single row of data from the _data attribute. For example, if key is 3, it will retrieve the fourth row of data.
        # If the key is a slice, it is used to retrieve a range of rows of data from the _data attribute. For example, if key is slice(2, 5), it will retrieve rows from the third to the fifth (exclusive) row.
        # If the key is a str, it is used to retrieve a row of data based on a specific condition. For example, if key is "column_name == 'value'", it will retrieve the rows where the value of the column named "column_name" is equal to "value".

        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)

        shuffled_text = ", ".join(
            [
                "%s is %s"
                % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip())
                for i in shuffle_idx
            ]
        )
        tokenized_text = self.tokenizer(shuffled_text, padding=True)
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)



# class GReaTDatasetVRUOC(Dataset):
#     """GReaT DatasetVRUOC
#     The GReaTDatasetVRUOC, like the GReaTDataset, overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.
#     """

#     def set_tokenizer(self, tokenizer):
#         self.tokenizer = tokenizer

#     def _getitem(
#         self, 
#         # key: tp.Union[int, slice, str], 
#         key: str, 
#         decoded: bool = True, 
#         **kwargs
#     ) -> tp.Union[tp.Dict, tp.List]:
        
#         # Get Item from Tabular Data accorint to the key in string form, e.g. "column_name == 'value'".
#         rows = self._data.fast_slice(key)
#         row0 = rows[0]

#         shuffle_idx = list(range(row0.num_columns))
#         random.shuffle(shuffle_idx)

#         shuffled_text = ", ".join(
#             [
#                 "%s is %s"
#                 % (row0.column_names[i], str(row0.columns[i].to_pylist()[0]).strip())
#                 for i in shuffle_idx
#             ]
#         )
#         tokenized_text = self.tokenizer(shuffled_text, padding=True)
#         return tokenized_text

#     def __getitems__(self, keys: tp.Union[int, slice, str, list]):
#         if isinstance(keys, list):
#             return [self._getitem(key) for key in keys]
#         else:
#             return self._getitem(keys)







@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """

    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
