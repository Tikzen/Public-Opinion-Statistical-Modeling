from abc import ABC, abstractmethod


class BaseDataSource(ABC):
    SOURCE_NAME = "base"

    @abstractmethod
    def load_data(self, *args, **kwargs) -> dict:
        raise NotImplementedError
