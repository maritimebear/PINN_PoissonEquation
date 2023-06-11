import pandas as pd
import torch
import numpy  # Required only to check return types?
from typing import Union


class PINN_Dataset(torch.utils.data.Dataset):
    """
    Reads data (for data loss) from .csv files, intended for use
    with torch.utils.data.Dataloader.

    Returns data as (input array, output array), where inputs and outputs are
    wrt the PINN model.
    """
    def __init__(self,
                 filename: str,
                 input_cols: Union[list[str], list[int]],
                 output_cols: Union[list[str], list[int]]):
        """
        filename: .csv file containing data
        input_cols, output_cols: column names or column indices of input and
        output data in the .csv file

        input_cols, output_cols must be lists to guarantee numpy.ndarray is returned
        """
        data = pd.read_csv(filename)
        # try-catch block to access columns in .csv by either names or indices
        try:
            # data.loc inputs string labels (column names)
            self._inputs, self._outputs = [data.loc[:, labels].to_numpy() for
                                           labels in (input_cols, output_cols)]
        except KeyError:
            # data.iloc expects int indices
            self._inputs, self._outputs = [data.iloc[:, labels].to_numpy() for
                                           labels in (input_cols, output_cols)]

        assert len(self._inputs) == len(self._outputs)

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        return (self._inputs[idx], self._outputs[idx])
