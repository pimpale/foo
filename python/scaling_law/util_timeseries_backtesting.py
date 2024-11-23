import pandas as pd
import numpy as np


class BacktestSplitter:
    pass

class ExpandingWindowBacktestSplitter(BacktestSplitter):
    def __init__(
        self,
        min_train_size: int,
        test_size: int,
        increment: int,
        key: str,
    ):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.increment = increment
        self.key = key
        
    def split(self, df: pd.DataFrame):
        df = df.sort_values(self.key)
        for i in range(self.min_train_size, len(df) - self.test_size, self.increment):
            train = df.iloc[:i]
            test = df.iloc[i:i + self.test_size]
            yield train.copy(), test.copy()
        
class RollingWindowBacktestSplitter(BacktestSplitter):
    def __init__(
        self,
        train_size: int,
        test_size: int,
        increment: int,
        key: str,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.increment = increment
        self.key = key
        
    def split(self, df: pd.DataFrame):
        df = df.sort_values(self.key)
        for i in range(0, len(df) - self.train_size - self.test_size, self.increment):
            train = df.iloc[i:i + self.train_size]
            test = df.iloc[i + self.train_size:i + self.train_size + self.test_size]
            yield train.copy(), test.copy()
        
