# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

###############################################################################
# Reference From PyTorch Core
###############################################################################
from torch.utils.data.datapipes.iter import (
    Batcher,
    Collator,
    Concater,
    Demultiplexer,
    FileLister,
    FileOpener,
    Filter,
    Forker,
    Grouper,
    IterableWrapper,
    Mapper,
    Multiplexer,
    RoutedDecoder,
    Sampler,
    ShardingFilter,
    Shuffler,
    StreamReader,
    UnBatcher,
    Zipper,
)
from torchdata.datapipes.iter.load.aisio import (
    AISFileListerIterDataPipe as AISFileLister,
    AISFileLoaderIterDataPipe as AISFileLoader,
)

###############################################################################
# TorchData
###############################################################################
from torchdata.datapipes.iter.load.fsspec import (
    FSSpecFileListerIterDataPipe as FSSpecFileLister,
    FSSpecFileOpenerIterDataPipe as FSSpecFileOpener,
    FSSpecSaverIterDataPipe as FSSpecSaver,
)

from torchdata.datapipes.iter.load.huggingface import HuggingFaceHubReaderIterDataPipe as HuggingFaceHubReader

from torchdata.datapipes.iter.load.iopath import (
    IoPathFileListerIterDataPipe as IoPathFileLister,
    IoPathFileOpenerIterDataPipe as IoPathFileOpener,
    IoPathSaverIterDataPipe as IoPathSaver,
)

from torchdata.datapipes.iter.load.online import (
    GDriveReaderDataPipe as GDriveReader,
    HTTPReaderIterDataPipe as HttpReader,
    OnlineReaderIterDataPipe as OnlineReader,
)
from torchdata.datapipes.iter.load.s3io import (
    S3FileListerIterDataPipe as S3FileLister,
    S3FileLoaderIterDataPipe as S3FileLoader,
)
from torchdata.datapipes.iter.transform.bucketbatcher import (
    BucketBatcherIterDataPipe as BucketBatcher,
    InBatchShufflerIterDataPipe as InBatchShuffler,
    MaxTokenBucketizerIterDataPipe as MaxTokenBucketizer,
)
from torchdata.datapipes.iter.transform.callable import (
    BatchAsyncMapperIterDataPipe as BatchAsyncMapper,
    BatchMapperIterDataPipe as BatchMapper,
    DropperIterDataPipe as Dropper,
    FlatMapperIterDataPipe as FlatMapper,
    FlattenIterDataPipe as Flattener,
    ShuffledFlatMapperIterDataPipe as ShuffledFlatMapper,
    SliceIterDataPipe as Slicer,
    ThreadPoolMapperIterDataPipe as ThreadPoolMapper,
)
from torchdata.datapipes.iter.util.bz2fileloader import Bz2FileLoaderIterDataPipe as Bz2FileLoader
from torchdata.datapipes.iter.util.cacheholder import (
    EndOnDiskCacheHolderIterDataPipe as EndOnDiskCacheHolder,
    InMemoryCacheHolderIterDataPipe as InMemoryCacheHolder,
    OnDiskCacheHolderIterDataPipe as OnDiskCacheHolder,
)
from torchdata.datapipes.iter.util.combining import (
    IterKeyZipperIterDataPipe as IterKeyZipper,
    MapKeyZipperIterDataPipe as MapKeyZipper,
    RoundRobinDemultiplexerIterDataPipe as RoundRobinDemultiplexer,
    UnZipperIterDataPipe as UnZipper,
)
from torchdata.datapipes.iter.util.cycler import CyclerIterDataPipe as Cycler, RepeaterIterDataPipe as Repeater
from torchdata.datapipes.iter.util.dataframemaker import (
    DataFrameMakerIterDataPipe as DataFrameMaker,
    ParquetDFLoaderIterDataPipe as ParquetDataFrameLoader,
)
from torchdata.datapipes.iter.util.decompressor import (
    DecompressorIterDataPipe as Decompressor,
    ExtractorIterDataPipe as Extractor,
)
from torchdata.datapipes.iter.util.distributed import FullSyncIterDataPipe as FullSync
from torchdata.datapipes.iter.util.hashchecker import HashCheckerIterDataPipe as HashChecker
from torchdata.datapipes.iter.util.header import HeaderIterDataPipe as Header, LengthSetterIterDataPipe as LengthSetter
from torchdata.datapipes.iter.util.indexadder import (
    EnumeratorIterDataPipe as Enumerator,
    IndexAdderIterDataPipe as IndexAdder,
)
from torchdata.datapipes.iter.util.jsonparser import JsonParserIterDataPipe as JsonParser
from torchdata.datapipes.iter.util.mux_longest import MultiplexerLongestIterDataPipe as MultiplexerLongest
from torchdata.datapipes.iter.util.paragraphaggregator import ParagraphAggregatorIterDataPipe as ParagraphAggregator
from torchdata.datapipes.iter.util.plain_text_reader import (
    CSVDictParserIterDataPipe as CSVDictParser,
    CSVParserIterDataPipe as CSVParser,
    LineReaderIterDataPipe as LineReader,
)
from torchdata.datapipes.iter.util.prefetcher import (
    PinMemoryIterDataPipe as PinMemory,
    PrefetcherIterDataPipe as Prefetcher,
)
from torchdata.datapipes.iter.util.randomsplitter import RandomSplitterIterDataPipe as RandomSplitter
from torchdata.datapipes.iter.util.rararchiveloader import RarArchiveLoaderIterDataPipe as RarArchiveLoader
from torchdata.datapipes.iter.util.rows2columnar import Rows2ColumnarIterDataPipe as Rows2Columnar
from torchdata.datapipes.iter.util.samplemultiplexer import SampleMultiplexerDataPipe as SampleMultiplexer
from torchdata.datapipes.iter.util.saver import SaverIterDataPipe as Saver
from torchdata.datapipes.iter.util.shardexpander import ShardExpanderIterDataPipe as ShardExpander
from torchdata.datapipes.iter.util.sharding import (
    ShardingRoundRobinDispatcherIterDataPipe as ShardingRoundRobinDispatcher,
)
from torchdata.datapipes.iter.util.tararchiveloader import TarArchiveLoaderIterDataPipe as TarArchiveLoader
from torchdata.datapipes.iter.util.tfrecordloader import (
    TFRecordExample,
    TFRecordExampleSpec,
    TFRecordLoaderIterDataPipe as TFRecordLoader,
)
from torchdata.datapipes.iter.util.webdataset import WebDatasetIterDataPipe as WebDataset
from torchdata.datapipes.iter.util.xzfileloader import XzFileLoaderIterDataPipe as XzFileLoader
from torchdata.datapipes.iter.util.zip_longest import ZipperLongestIterDataPipe as ZipperLongest
from torchdata.datapipes.iter.util.ziparchiveloader import ZipArchiveLoaderIterDataPipe as ZipArchiveLoader
from torchdata.datapipes.map.util.converter import MapToIterConverterIterDataPipe as MapToIterConverter

__all__ = [
    "AISFileLister",
    "AISFileLoader",
    "BatchAsyncMapper",
    "BatchMapper",
    "Batcher",
    "BucketBatcher",
    "Bz2FileLoader",
    "CSVDictParser",
    "CSVParser",
    "Collator",
    "Concater",
    "Cycler",
    "DataFrameMaker",
    "Decompressor",
    "Demultiplexer",
    "Dropper",
    "EndOnDiskCacheHolder",
    "Enumerator",
    "Extractor",
    "FSSpecFileLister",
    "FSSpecFileOpener",
    "FSSpecSaver",
    "FileLister",
    "FileOpener",
    "Filter",
    "FlatMapper",
    "Flattener",
    "Forker",
    "FullSync",
    "GDriveReader",
    "Grouper",
    "HashChecker",
    "Header",
    "HttpReader",
    "HuggingFaceHubReader",
    "InBatchShuffler",
    "InMemoryCacheHolder",
    "IndexAdder",
    "IoPathFileLister",
    "IoPathFileOpener",
    "IoPathSaver",
    "IterDataPipe",
    "IterKeyZipper",
    "IterableWrapper",
    "JsonParser",
    "LengthSetter",
    "LineReader",
    "MapKeyZipper",
    "MapToIterConverter",
    "Mapper",
    "MaxTokenBucketizer",
    "Multiplexer",
    "MultiplexerLongest",
    "OnDiskCacheHolder",
    "OnlineReader",
    "ParagraphAggregator",
    "ParquetDataFrameLoader",
    "PinMemory",
    "Prefetcher",
    "RandomSplitter",
    "RarArchiveLoader",
    "Repeater",
    "RoundRobinDemultiplexer",
    "RoutedDecoder",
    "Rows2Columnar",
    "S3FileLister",
    "S3FileLoader",
    "SampleMultiplexer",
    "Sampler",
    "Saver",
    "ShardExpander",
    "ShardingFilter",
    "ShardingRoundRobinDispatcher",
    "ShuffledFlatMapper",
    "Shuffler",
    "Slicer",
    "StreamReader",
    "TFRecordLoader",
    "TarArchiveLoader",
    "ThreadPoolMapper",
    "UnBatcher",
    "UnZipper",
    "WebDataset",
    "XzFileLoader",
    "ZipArchiveLoader",
    "Zipper",
    "ZipperLongest",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)

########################################################################################################################
# The part below is generated by parsing through the Python files where IterDataPipes are defined.
# This base template ("__init__.pyi.in") is generated from mypy stubgen with minimal editing for code injection
# The output file will be "__init__.pyi". The generation function is called by "setup.py".
# Note that, for mypy, .pyi file takes precedent over .py file, such that we must define the interface for other
# classes/objects here, even though we are not injecting extra code into them at the moment.

from .util.decompressor import CompressionType
from torchdata._constants import default_timeout_in_s
from torchdata.datapipes.map import MapDataPipe
from torchdata.datapipes.utils import pin_memory_fn
from torch.utils.data import DataChunk, IterableDataset, default_collate
from torch.utils.data.datapipes._typing import _DataPipeMeta
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union, Hashable

try:
    import torcharrow
except ImportError:
    torcharrow = None

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

class IterDataPipe(IterableDataset[T_co], metaclass=_DataPipeMeta):
    functions: Dict[str, Callable] = ...
    reduce_ex_hook: Optional[Callable] = ...
    getstate_hook: Optional[Callable] = ...
    def __getattr__(self, attribute_name: Any): ...
    @classmethod
    def register_function(cls, function_name: Any, function: Any) -> None: ...
    @classmethod
    def register_datapipe_as_function(
        cls, function_name: Any, cls_to_register: Any, enable_df_api_tracing: bool = ...
    ): ...
    def __getstate__(self): ...
    def __reduce_ex__(self, *args: Any, **kwargs: Any): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn: Any) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn: Any) -> None: ...
    # Functional form of 'BatcherIterDataPipe'
    def batch(self, batch_size: int, drop_last: bool = False, wrapper_class=DataChunk) -> IterDataPipe:
        r"""
        Creates mini-batches of data (functional name: ``batch``). An outer dimension will be added as
        ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size`` for the
        last batch if ``drop_last`` is set to ``False``.
    
        Args:
            datapipe: Iterable DataPipe being batched
            batch_size: The size of each batch
            drop_last: Option to drop the last batch if it's not full
            wrapper_class: wrapper to apply onto each batch (type ``List``) before yielding,
                defaults to ``DataChunk``
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(range(10))
            >>> dp = dp.batch(batch_size=3, drop_last=True)
            >>> list(dp)
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        """
    
    # Functional form of 'CollatorIterDataPipe'
    def collate(self, conversion: Optional[Union[Callable[..., Any],Dict[Union[str, Any], Union[Callable, Any]],]] = default_collate, collate_fn: Optional[Callable] = None) -> IterDataPipe:
        r"""
        Collates samples from DataPipe to Tensor(s) by a custom collate function (functional name: ``collate``).
        By default, it uses :func:`torch.utils.data.default_collate`.
    
        .. note::
            While writing a custom collate function, you can import :func:`torch.utils.data.default_collate` for the
            default behavior and `functools.partial` to specify any additional arguments.
    
        Args:
            datapipe: Iterable DataPipe being collated
            collate_fn: Customized collate function to collect and combine data or a batch of data.
                Default function collates to Tensor(s) based on data type.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> # Convert integer data to float Tensor
            >>> class MyIterDataPipe(torch.utils.data.IterDataPipe):
            ...     def __init__(self, start, end):
            ...         super(MyIterDataPipe).__init__()
            ...         assert end > start, "this example code only works with end >= start"
            ...         self.start = start
            ...         self.end = end
            ...
            ...     def __iter__(self):
            ...         return iter(range(self.start, self.end))
            ...
            ...     def __len__(self):
            ...         return self.end - self.start
            ...
            >>> ds = MyIterDataPipe(start=3, end=7)
            >>> print(list(ds))
            [3, 4, 5, 6]
            >>> def collate_fn(batch):
            ...     return torch.tensor(batch, dtype=torch.float)
            ...
            >>> collated_ds = CollateIterDataPipe(ds, collate_fn=collate_fn)
            >>> print(list(collated_ds))
            [tensor(3.), tensor(4.), tensor(5.), tensor(6.)]
        """
    
    # Functional form of 'ConcaterIterDataPipe'
    def concat(self, *datapipes: IterDataPipe) -> IterDataPipe:
        r"""
        Concatenates multiple Iterable DataPipes (functional name: ``concat``). The resulting DataPipe will
        yield all the elements from the first input DataPipe, before yielding from the subsequent ones.
    
        Args:
            datapipes: Iterable DataPipes being concatenated
    
        Example:
            >>> # xdoctest: +REQUIRES(module:torchdata)
            >>> import random
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp1 = IterableWrapper(range(3))
            >>> dp2 = IterableWrapper(range(5))
            >>> list(dp1.concat(dp2))
            [0, 1, 2, 0, 1, 2, 3, 4]
        """
    
    # Functional form of 'DemultiplexerIterDataPipe'
    def demux(self, num_instances: int, classifier_fn: Callable[[T_co], Optional[int]], drop_none: bool = False, buffer_size: int = 1000) -> List[IterDataPipe]:
        r"""
        Splits the input DataPipe into multiple child DataPipes, using the given
        classification function (functional name: ``demux``). A list of the child DataPipes is returned from this operation.
    
        Args:
            datapipe: Iterable DataPipe being filtered
            num_instances: number of instances of the DataPipe to create
            classifier_fn: a function that maps values to an integer within the range ``[0, num_instances - 1]`` or ``None``
            drop_none: defaults to ``False``, if ``True``, the function will skip over elements classified as ``None``
            buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
                DataPipes while waiting for their values to be yielded.
                Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
    
        Examples:
            >>> # xdoctest: +REQUIRES(module:torchdata)
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> def odd_or_even(n):
            ...     return n % 2
            >>> source_dp = IterableWrapper(range(5))
            >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even)
            >>> list(dp1)
            [0, 2, 4]
            >>> list(dp2)
            [1, 3]
            >>> # It can also filter out any element that gets `None` from the `classifier_fn`
            >>> def odd_or_even_no_zero(n):
            ...     return n % 2 if n != 0 else None
            >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even_no_zero, drop_none=True)
            >>> list(dp1)
            [2, 4]
            >>> list(dp2)
            [1, 3]
        """
    
    # Functional form of 'FilterIterDataPipe'
    def filter(self, filter_fn: Callable, input_col=None) -> IterDataPipe:
        r"""
        Filters out elements from the source datapipe according to input ``filter_fn`` (functional name: ``filter``).
    
        Args:
            datapipe: Iterable DataPipe being filtered
            filter_fn: Customized function mapping an element to a boolean.
            input_col: Index or indices of data which ``filter_fn`` is applied, such as:
    
                - ``None`` as default to apply ``filter_fn`` to the data directly.
                - Integer(s) is used for list/tuple.
                - Key(s) is used for dict.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> def is_even(n):
            ...     return n % 2 == 0
            >>> dp = IterableWrapper(range(5))
            >>> filter_dp = dp.filter(filter_fn=is_even)
            >>> list(filter_dp)
            [0, 2, 4]
        """
    
    # Functional form of 'ForkerIterDataPipe'
    def fork(self, num_instances: int, buffer_size: int = 1000, copy: Optional[Literal["shallow", "deep"]] = None) -> List[IterDataPipe]:
        r"""
        Creates multiple instances of the same Iterable DataPipe (functional name: ``fork``).
    
        Args:
            datapipe: Iterable DataPipe being copied
            num_instances: number of instances of the datapipe to create
            buffer_size: this restricts how far ahead the leading child DataPipe
               can read relative to the slowest child DataPipe.
               Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
            copy: copy strategy to use for items yielded by each branch. Supported
                options are ``None`` for no copying, ``"shallow"`` for shallow object
                copies, and ``"deep"`` for deep object copies. Defaults to ``None``.
    
        Note:
            All branches of the forked pipeline return the identical object unless
            the copy parameter is supplied. If the object is mutable or contains
            mutable objects, changing them in one branch will affect all others.
    
        Example:
            >>> # xdoctest: +REQUIRES(module:torchdata)
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper(range(5))
            >>> dp1, dp2 = source_dp.fork(num_instances=2)
            >>> list(dp1)
            [0, 1, 2, 3, 4]
            >>> list(dp2)
            [0, 1, 2, 3, 4]
        """
    
    # Functional form of 'GrouperIterDataPipe'
    def groupby(self, group_key_fn: Callable[[T_co], Any], *, keep_key: bool = False, buffer_size: int = 10000, group_size: Optional[int] = None, guaranteed_group_size: Optional[int] = None, drop_remaining: bool = False) -> IterDataPipe:
        r"""
        Groups data from input IterDataPipe by keys which are generated from ``group_key_fn``,
        and yields a ``DataChunk`` with batch size up to ``group_size`` if defined (functional name: ``groupby``).
    
        The samples are read sequentially from the source ``datapipe``, and a batch of samples belonging to the same group
        will be yielded as soon as the size of the batch reaches ``group_size``. When the buffer is full,
        the DataPipe will yield the largest batch with the same key, provided that its size is larger
        than ``guaranteed_group_size``. If its size is smaller, it will be dropped if ``drop_remaining=True``.
    
        After iterating through the entirety of source ``datapipe``, everything not dropped due to the buffer capacity
        will be yielded from the buffer, even if the group sizes are smaller than ``guaranteed_group_size``.
    
        Args:
            datapipe: Iterable datapipe to be grouped
            group_key_fn: Function used to generate group key from the data of the source datapipe
            keep_key: Option to yield the matching key along with the items in a tuple,
                resulting in `(key, [items])` otherwise returning [items]
            buffer_size: The size of buffer for ungrouped data
            group_size: The max size of each group, a batch is yielded as soon as it reaches this size
            guaranteed_group_size: The guaranteed minimum group size to be yielded in case the buffer is full
            drop_remaining: Specifies if the group smaller than ``guaranteed_group_size`` will be dropped from buffer
                when the buffer is full
    
        Example:
            >>> import os
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> def group_fn(file):
            ...     return os.path.basename(file).split(".")[0]
            >>> source_dp = IterableWrapper(["a.png", "b.png", "a.json", "b.json", "a.jpg", "c.json"])
            >>> dp0 = source_dp.groupby(group_key_fn=group_fn)
            >>> list(dp0)
            [['a.png', 'a.json', 'a.jpg'], ['b.png', 'b.json'], ['c.json']]
            >>> # A group is yielded as soon as its size equals to `group_size`
            >>> dp1 = source_dp.groupby(group_key_fn=group_fn, group_size=2)
            >>> list(dp1)
            [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
            >>> # Scenario where `buffer` is full, and group 'a' needs to be yielded since its size > `guaranteed_group_size`
            >>> dp2 = source_dp.groupby(group_key_fn=group_fn, buffer_size=3, group_size=3, guaranteed_group_size=2)
            >>> list(dp2)
            [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
        """
    
    # Functional form of 'FileListerIterDataPipe'
    def list_files(self, masks: Union[str, List[str]] = '', *, recursive: bool = False, abspath: bool = False, non_deterministic: bool = False, length: int = -1) -> IterDataPipe:
        r"""
        Given path(s) to the root directory, yields file pathname(s) (path + filename) of files within the root directory.
        Multiple root directories can be provided (functional name: ``list_files``).
    
        Args:
            root: Root directory or a sequence of root directories
            masks: Unix style filter string or string list for filtering file name(s)
            recursive: Whether to return pathname from nested directories or not
            abspath: Whether to return relative pathname or absolute pathname
            non_deterministic: Whether to return pathname in sorted order or not.
                If ``False``, the results yielded from each root directory will be sorted
            length: Nominal length of the datapipe
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import FileLister
            >>> dp = FileLister(root=".", recursive=True)
            >>> list(dp)
            ['example.py', './data/data.tar']
        """
    
    # Functional form of 'MapperIterDataPipe'
    def map(self, fn: Callable, input_col=None, output_col=None) -> IterDataPipe:
        r"""
        Applies a function over each item from the source DataPipe (functional name: ``map``).
        The function can be any regular Python function or partial object. Lambda
        function is not recommended as it is not supported by pickle.
    
        Args:
            datapipe: Source Iterable DataPipe
            fn: Function being applied over each item
            input_col: Index or indices of data which ``fn`` is applied, such as:
    
                - ``None`` as default to apply ``fn`` to the data directly.
                - Integer(s) is used for list/tuple.
                - Key(s) is used for dict.
    
            output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
                only when ``input_col`` is not ``None``
    
                - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
                  multiple indices, the left-most one is used, and other indices will be removed.
                - Integer is used for list/tuple. ``-1`` represents to append result at the end.
                - Key is used for dict. New key is acceptable.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> def add_one(x):
            ...     return x + 1
            >>> dp = IterableWrapper(range(10))
            >>> map_dp_1 = dp.map(add_one)  # Invocation via functional form is preferred
            >>> list(map_dp_1)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> # We discourage the usage of `lambda` functions as they are not serializable with `pickle`
            >>> # Use `functools.partial` or explicitly define the function instead
            >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
            >>> list(map_dp_2)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        """
    
    # Functional form of 'MultiplexerIterDataPipe'
    def mux(self, *datapipes) -> IterDataPipe:
        r"""
        Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux``). As in,
        one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
        and so on. It ends when the shortest input DataPipe is exhausted.
    
        Args:
            datapipes: Iterable DataPipes that will take turn to yield their elements, until the shortest DataPipe is exhausted
    
        Example:
            >>> # xdoctest: +REQUIRES(module:torchdata)
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp1, dp2, dp3 = IterableWrapper(range(3)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
            >>> list(dp1.mux(dp2, dp3))
            [0, 10, 20, 1, 11, 21, 2, 12, 22]
        """
    
    # Functional form of 'FileOpenerIterDataPipe'
    def open_files(self, mode: str = 'r', encoding: Optional[str] = None, length: int = -1) -> IterDataPipe:
        r"""
        Given pathnames, opens files and yield pathname and file stream
        in a tuple (functional name: ``open_files``).
    
        Args:
            datapipe: Iterable datapipe that provides pathnames
            mode: An optional string that specifies the mode in which
                the file is opened by ``open()``. It defaults to ``r``, other options are
                ``b`` for reading in binary mode and ``t`` for text mode.
            encoding: An optional string that specifies the encoding of the
                underlying file. It defaults to ``None`` to match the default encoding of ``open``.
            length: Nominal length of the datapipe
    
        Note:
            The opened file handles will be closed by Python's GC periodically. Users can choose
            to close them explicitly.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
            >>> dp = FileLister(root=".").filter(lambda fname: fname.endswith('.txt'))
            >>> dp = FileOpener(dp)
            >>> dp = StreamReader(dp)
            >>> list(dp)
            [('./abc.txt', 'abc')]
        """
    
    # Functional form of 'StreamReaderIterDataPipe'
    def read_from_stream(self, chunk=None) -> IterDataPipe:
        r"""
        Given IO streams and their label names, yields bytes with label
        name in a tuple (functional name: ``read_from_stream``).
    
        Args:
            datapipe: Iterable DataPipe provides label/URL and byte stream
            chunk: Number of bytes to be read from stream per iteration.
                If ``None``, all bytes will be read until the EOF.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper, StreamReader
            >>> from io import StringIO
            >>> dp = IterableWrapper([("alphabet", StringIO("abcde"))])
            >>> list(StreamReader(dp, chunk=1))
            [('alphabet', 'a'), ('alphabet', 'b'), ('alphabet', 'c'), ('alphabet', 'd'), ('alphabet', 'e')]
        """
    
    # Functional form of 'RoutedDecoderIterDataPipe'
    def routed_decode(self, *handlers: Callable, key_fn: Callable= ...) -> IterDataPipe:
        r"""
        Decodes binary streams from input DataPipe, yields pathname and decoded data
        in a tuple (functional name: ``routed_decode``).
    
        Args:
            datapipe: Iterable datapipe that provides pathname and binary stream in tuples
            handlers: Optional user defined decoder handlers. If ``None``, basic and image decoder
                handlers will be set as default. If multiple handles are provided, the priority
                order follows the order of handlers (the first handler has the top priority)
            key_fn: Function for decoder to extract key from pathname to dispatch handlers.
                Default is set to extract file extension from pathname
    
        Note:
            When ``key_fn`` is specified returning anything other than extension, the default
            handler will not work and users need to specify custom handler. Custom handler
            could use regex to determine the eligibility to handle data.
        """
    
    # Functional form of 'ShardingFilterIterDataPipe'
    def sharding_filter(self, sharding_group_filter=None) -> IterDataPipe:
        r"""
        Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``). After ``apply_sharding`` is
        called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
        original DataPipe, where `n` equals to the number of instances.
    
        Args:
            source_datapipe: Iterable DataPipe that will be sharded
        """
    
    # Functional form of 'ShufflerIterDataPipe'
    def shuffle(self, *, buffer_size: int = 10000, unbatch_level: int = 0) -> IterDataPipe:
        r"""
        Shuffles the input DataPipe with a buffer (functional name: ``shuffle``). The buffer
        with ``buffer_size`` is filled with elements from the datapipe first. Then,
        each item will be yielded from the buffer by reservoir sampling via iterator.
    
        ``buffer_size`` is required to be larger than ``0``. For ``buffer_size == 1``, the
        datapipe is not shuffled. In order to fully shuffle all elements from datapipe,
        ``buffer_size`` is required to be greater than or equal to the size of datapipe.
    
        When it is used with :class:`torch.utils.data.DataLoader`, the methods to
        set up random seed are different based on :attr:`num_workers`.
    
        For single-process mode (:attr:`num_workers == 0`), the random seed is set before
        the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
        mode (:attr:`num_worker > 0`), `worker_init_fn` is used to set up a random seed
        for each worker process.
    
        Args:
            datapipe: The IterDataPipe being shuffled
            buffer_size: The buffer size for shuffling (default to ``10000``)
            unbatch_level: Specifies if it is necessary to unbatch source data before
                applying the shuffle
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(range(10))
            >>> shuffle_dp = dp.shuffle()
            >>> list(shuffle_dp)
            [0, 4, 1, 6, 3, 2, 9, 5, 7, 8]
        """
    
    # Functional form of 'UnBatcherIterDataPipe'
    def unbatch(self, unbatch_level: int = 1) -> IterDataPipe:
        r"""
        Undoes batching of data (functional name: ``unbatch``). In other words, it flattens the data up to the specified level
        within a batched DataPipe.
    
        Args:
            datapipe: Iterable DataPipe being un-batched
            unbatch_level: Defaults to ``1`` (only flattening the top level). If set to ``2``,
                it will flatten the top two levels, and ``-1`` will flatten the entire DataPipe.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper([[[0, 1], [2]], [[3, 4], [5]], [[6]]])
            >>> dp1 = source_dp.unbatch()
            >>> list(dp1)
            [[0, 1], [2], [3, 4], [5], [6]]
            >>> dp2 = source_dp.unbatch(unbatch_level=2)
            >>> list(dp2)
            [0, 1, 2, 3, 4, 5, 6]
        """
    
    # Functional form of 'ZipperIterDataPipe'
    def zip(self, *datapipes: IterDataPipe) -> IterDataPipe:
        r"""
        Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).
        The output is stopped as soon as the shortest input DataPipe is exhausted.
    
        Args:
            *datapipes: Iterable DataPipes being aggregated
    
        Example:
            >>> # xdoctest: +REQUIRES(module:torchdata)
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
            >>> list(dp1.zip(dp2, dp3))
            [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
        """
    
    # Functional form of 'IndexAdderIterDataPipe'
    def add_index(self, index_name: str = "index") -> IterDataPipe:
        r"""
        Adds an index to an existing Iterable DataPipe with (functional name: ``add_index``). The row or batch
        within the DataPipe must have the type `Dict`; otherwise, a `NotImplementedError` will be thrown. The index
        of the data is set to the provided ``index_name``.
    
        Args:
            source_datapipe: Iterable DataPipe being indexed, its row/batch must be of type `Dict`
            index_name: Name of the key to store data index
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper([{'a': 1, 'b': 2}, {'c': 3, 'a': 1}])
            >>> index_dp = dp.add_index("order")
            >>> list(index_dp)
            [{'a': 1, 'b': 2, 'order': 0}, {'c': 3, 'a': 1, 'order': 1}]
        """
    
    # Functional form of 'BatchAsyncMapperIterDataPipe'
    def async_map_batches(self, async_fn: Callable, batch_size: int, input_col=None, output_col=None, max_concurrency: int = 32, flatten: bool = True) -> IterDataPipe:
        r"""
        Combines elements from the source DataPipe to batches and applies a coroutine function
        over each element within the batch concurrently, then flattens the outpus to a
        single, unnested IterDataPipe (functional name: ``async_map_batches``).
    
        Args:
            source_datapipe: Source IterDataPipe
            async_fn: The coroutine function to be applied to each batch of data
            batch_size: The size of batch to be aggregated from ``source_datapipe``
            input_col: Index or indices of data which ``fn`` is applied, such as:
    
                - ``None`` as default to apply ``fn`` to the data directly.
                - Integer(s) is used for list/tuple.
                - Key(s) is used for dict.
    
            output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
                only when ``input_col`` is not ``None``
    
                - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
                  multiple indices, the left-most one is used, and other indices will be removed.
                - Integer is used for list/tuple. ``-1`` represents to append result at the end.
                - Key is used for dict. New key is acceptable.
    
            max_concurrency: Maximum concurrency to call async functions. (Default: ``32``)
            flatten: Determine if the batches get flatten in the end (Default: ``True``)
                     If ``False``, outputs will be in batches of size ``batch_size``
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> async def mul_ten(x):
            ...     await asyncio.sleep(1)
            ...     return x * 10
            >>> dp = IterableWrapper(range(50))
            >>> dp = dp.async_map_batches(mul_ten, 16)
            >>> list(dp)
            [0, 10, 20, 30, ...]
            >>> dp = IterableWrapper([(i, i) for i in range(50)])
            >>> dp = dp.async_map_batches(mul_ten, 16, input_col=1)
            >>> list(dp)
            [(0, 0), (1, 10), (2, 20), (3, 30), ...]
            >>> dp = IterableWrapper([(i, i) for i in range(50)])
            >>> dp = dp.async_map_batches(mul_ten, 16, input_col=1, output_col=-1)
            >>> list(dp)
            [(0, 0, 0), (1, 1, 10), (2, 2, 20), (3, 3, 30), ...]
            # Async fetching html from remote
            >>> from aiohttp import ClientSession
            >>> async def fetch_html(url: str, **kwargs):
            ...     async with ClientSession() as session:
            ...         resp = await session.request(method="GET", url=url, **kwargs)
            ...         resp.raise_for_status()
            ...         html = await resp.text()
            ...     return html
            >>> dp = IterableWrapper(urls)
            >>> dp = dp.async_map_batches(fetch_html, 16)
        """
    
    # Functional form of 'BucketBatcherIterDataPipe'
    def bucketbatch(self, batch_size: int, drop_last: bool = False, batch_num: int = 100, bucket_num: int = 1, sort_key: Optional[Callable] = None, use_in_batch_shuffle: bool = True) -> IterDataPipe:
        r"""
        Creates mini-batches of data from sorted bucket (functional name: ``bucketbatch``). An outer
        dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``,
        or ``length % batch_size`` for the last batch if ``drop_last`` is set to ``False``.
    
        The purpose of this DataPipe is to batch samples with some similarity according to the sorting function
        being passed. For an example in the text domain, it may be batching examples with similar number of tokens
        to minimize padding and to increase throughput.
    
        Args:
            datapipe: Iterable DataPipe being batched
            batch_size: The size of each batch
            drop_last: Option to drop the last batch if it's not full
            batch_num: Number of batches within a bucket (i.e. `bucket_size = batch_size * batch_num`)
            bucket_num: Number of buckets to consist a pool for shuffling (i.e. `pool_size = bucket_size * bucket_num`)
            sort_key: Callable to sort a bucket (list)
            use_in_batch_shuffle: if True, do in-batch shuffle; if False, buffer shuffle
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper(range(10))
            >>> batch_dp = source_dp.bucketbatch(batch_size=3, drop_last=True)
            >>> list(batch_dp)
            [[5, 6, 7], [9, 0, 1], [4, 3, 2]]
            >>> def sort_bucket(bucket):
            >>>     return sorted(bucket)
            >>> batch_dp = source_dp.bucketbatch(
            >>>     batch_size=3, drop_last=True, batch_num=100,
            >>>     bucket_num=1, use_in_batch_shuffle=False, sort_key=sort_bucket
            >>> )
            >>> list(batch_dp)
            [[3, 4, 5], [6, 7, 8], [0, 1, 2]]
        """
    
    # Functional form of 'HashCheckerIterDataPipe'
    def check_hash(self, hash_dict: Dict[str, str], hash_type: str = "sha256", rewind: bool = True) -> IterDataPipe:
        r"""
        Computes and checks the hash of each file, from an input DataPipe of tuples of file name and
        data/stream (functional name: ``check_hash``). If the hashes match the given hash
        in the dictionary, it yields a tuple of file name and data/stream. Otherwise, it will raise an error.
    
        Args:
            source_datapipe: IterDataPipe with tuples of file name and data/stream
            hash_dict: Dictionary that maps file names to their corresponding hashes
            hash_type: The type of hash function to apply
            rewind: Rewind the stream after using the stream to compute the hash (this
                does not work with non-seekable stream, e.g. HTTP)
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, FileOpener
            >>> expected_MD5_hash = "bb9675028dd39d2dd2bf71002b93e66c"
            File is from "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
            >>> file_dp = FileOpener(IterableWrapper(["LICENSE.txt"]), mode='rb')
            >>> # An exception is only raised when the hash doesn't match, otherwise (path, stream) is returned
            >>> check_hash_dp = file_dp.check_hash({"LICENSE.txt": expected_MD5_hash}, "md5", rewind=True)
            >>> reader_dp = check_hash_dp.readlines()
            >>> it = iter(reader_dp)
            >>> path, line = next(it)
            >>> path
            LICENSE.txt
            >>> line
            b'BSD 3-Clause License'
        """
    
    # Functional form of 'CyclerIterDataPipe'
    def cycle(self, count: Optional[int] = None) -> IterDataPipe:
        """
        Cycles the specified input in perpetuity by default, or for the specified number
        of times (functional name: ``cycle``).
    
        If the ordering does not matter (e.g. because you plan to ``shuffle`` later) or if you would like to
        repeat an element multiple times before moving onto the next element, use :class:`.Repeater`.
    
        Args:
            source_datapipe: source DataPipe that will be cycled through
            count: the number of times to read through ``source_datapipe` (if ``None``, it will cycle in perpetuity)
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(range(3))
            >>> dp = dp.cycle(2)
            >>> list(dp)
            [0, 1, 2, 0, 1, 2]
        """
    
    # Functional form of 'DataFrameMakerIterDataPipe'
    def dataframe(self, dataframe_size: int = 1000, dtype=None, dtype_generator=None, columns: Optional[List[str]] = None, device: str = "") -> torcharrow.DataFrame:
        r"""
        Takes rows of data, batches a number of them together and creates `TorchArrow`
        DataFrames (functional name: ``dataframe``).
    
        Note:
            There is a trade-off between having a large number of rows within a DataFrame and usage of memory. Please
            choose a value carefully.
    
        Args:
            source_dp: IterDataPipe containing rows of data
            dataframe_size: number of rows of data within each DataFrame, page size can be option
            dtype: specify the `TorchArrow` dtype for the DataFrame, use ``torcharrow.dtypes.DType``
            dtype_generator: function with no input argument that generates a torcharrow.dtypes.DType,
                which overrides dtype if both are given. This is useful for when the desired dtype is
                not serializable.
            columns: List of str that specifies the column names of the DataFrame
            device: specify the device on which the DataFrame will be stored
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> import torcharrow.dtypes as dt
            >>> source_data = [(i,) for i in range(3)]
            >>> source_dp = IterableWrapper(source_data)
            >>> DTYPE = dt.Struct([dt.Field("Values", dt.int32)])
            >>> df_dp = source_dp.dataframe(dtype=DTYPE)
            >>> list(df_dp)[0]
              index    Values
            -------  --------
                  0         0
                  1         1
                  2         2
            dtype: Struct([Field('Values', int32)]), count: 3, null_count: 0
        """
    
    # Functional form of 'DecompressorIterDataPipe'
    def decompress(self, file_type: Optional[Union[str, CompressionType]] = None) -> IterDataPipe:
        r"""
        Takes tuples of path and compressed stream of data, and returns tuples of
        path and decompressed stream of data (functional name: ``decompress``). The input compression format can be specified
        or automatically detected based on the files' file extensions.
    
        Args:
            source_datapipe: IterDataPipe containing tuples of path and compressed stream of data
            file_type: Optional `string` or ``CompressionType`` that represents what compression format of the inputs
    
        Example:
            >>> from torchdata.datapipes.iter import FileLister, FileOpener
            >>> tar_file_dp = FileLister(self.temp_dir.name, "*.tar")
            >>> tar_load_dp = FileOpener(tar_file_dp, mode="b")
            >>> tar_decompress_dp = Decompressor(tar_load_dp, file_type="tar")
            >>> for _, stream in tar_decompress_dp:
            >>>     print(stream.read())
            b'0123456789abcdef'
        """
    
    # Functional form of 'DropperIterDataPipe'
    def drop(self, indices: Union[Hashable, List[Hashable]]) -> IterDataPipe:
        r"""
        Drop columns/elements in input DataPipe via its indices (functional name: ``drop``).
    
        Args:
            datapipe: IterDataPipe with columns to be dropped
            indices: a single column index to be dropped or a list of indices
    
                - Integer(s) is/are used for list/tuple.
                - Key(s) is/are used for dict.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, ZipperMapDataPipe
            >>> dp1 = IterableWrapper(range(5))
            >>> dp2 = IterableWrapper(range(10, 15))
            >>> dp = dp1.zip(dp2)
            >>> list(dp)
            [(0, 10), (1, 11), (2, 12), (3, 13), (4, 14)]
            >>> drop_dp = dp.drop(1)
            >>> list(drop_dp)
            [(0), (1), (2), (3), (4)]
        """
    
    # Functional form of 'EndOnDiskCacheHolderIterDataPipe'
    def end_caching(self, mode="wb", filepath_fn=None, *, same_filepath_fn=False, skip_read=False, timeout=300) -> IterDataPipe:
        """
        Indicates when the result of prior DataPipe will be saved local files specified
        by ``filepath_fn`` (functional name: ``end_caching``). Moreover, the result of source DataPipe
        is required to be a tuple of metadata and data, or a tuple of metadata and file handle.
    
        Args:
            datapipe: IterDataPipe with at least one ``OnDiskCacheHolder`` in the graph.
            mode: Mode in which the cached files are opened to write the data on disk. This is needed
                to be aligned with the type of data or file handle from ``datapipe``. ``"wb"`` is used by default.
            filepath_fn: Optional function to extract filepath from the metadata from ``datapipe``.
                By default, it would directly use the ?metadata? as file path.
            same_filepath_fn: Set to ``True`` to use same ``filepath_fn`` from the ``OnDiskCacheHolder``.
            skip_read: Boolean value to skip reading the file handle from ``datapipe``.
                By default, reading is enabled and reading function is created based on the ``mode``.
            timeout: Integer value of seconds to wait for uncached item to be written to disk
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, HttpReader
            >>> url = IterableWrapper(["https://path/to/filename", ])
            >>> def _filepath_fn(url):
            >>>     temp_dir = tempfile.gettempdir()
            >>>     return os.path.join(temp_dir, os.path.basename(url))
            >>> hash_dict = {"expected_filepath": expected_MD5_hash}
            >>> # You must call ``.on_disk_cache`` at some point before ``.end_caching``
            >>> cache_dp = url.on_disk_cache(filepath_fn=_filepath_fn, hash_dict=_hash_dict, hash_type="md5")
            >>> # You must call ``.end_caching`` at a later point to stop tracing and save the results to local files.
            >>> cache_dp = HttpReader(cache_dp).end_caching(mode="wb", filepath_fn=_filepath_fn)
        """
    
    # Functional form of 'EnumeratorIterDataPipe'
    def enumerate(self, starting_index: int = 0) -> IterDataPipe:
        r"""
        Adds an index to an existing DataPipe through enumeration, with
        the index starting from 0 by default (functional name: ``enumerate``).
    
        Args:
            source_datapipe: Iterable DataPipe being indexed
            starting_index: Index from which enumeration will start
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(['a', 'b', 'c'])
            >>> enum_dp = dp.enumerate()
            >>> list(enum_dp)
            [(0, 'a'), (1, 'b'), (2, 'c')]
        """
    
    # Functional form of 'FlatMapperIterDataPipe'
    def flatmap(self, fn: Optional[Callable] = None, input_col=None) -> IterDataPipe:
        r"""
        Applies a function over each item from the source DataPipe, then
        flattens the outputs to a single, unnested IterDataPipe (functional name: ``flatmap``).
    
        Note:
            The output from ``fn`` must be a Sequence. Otherwise, an error will be raised.
            If ``fn`` is ``None``, source DataPipe will be just flattened vertically, provided that items can be unpacked.
    
        Args:
            datapipe: Source IterDataPipe
            fn: the function to be applied to each element in the DataPipe, the output must be a Sequence
            input_col: Index or indices of data which ``fn`` is applied, such as:
    
                - ``None`` as default to apply ``fn`` to the data directly.
                - Integer(s) is/are used for list/tuple.
                - Key(s) is/are used for dict.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> def fn(e):
            >>>     return [e, e * 10]
            >>> source_dp = IterableWrapper(list(range(5)))
            >>> flatmapped_dp = source_dp.flatmap(fn)
            >>> list(flatmapped_dp)
            [0, 0, 1, 10, 2, 20, 3, 30, 4, 40]
            >>>
            >>> source_dp = IterableWrapper([[1, 2, 3], [4, 5, 6]])
            >>> flatmapped_dp = source_dp.flatmap()
            >>> list(flatmapped_dp)
            [1, 2, 3, 4, 5, 6]
        """
    
    # Functional form of 'FlattenIterDataPipe'
    def flatten(self, indices: Optional[Union[Hashable, List[Hashable]]] = None) -> IterDataPipe:
        r"""
        returns a flattened copy of the input DataPipe at the per sample/element level based on provided indices (functional name: ``flatten``).
    
        Note:
            no args will flatten each item in the datapipe 1 level
    
        Args:
            datapipe: IterDataPipe with iterable elements
            indices: a single index/key for the item to flatten from an iterator item or a list of indices/keys to be flattened
    
                - Integer(s) is/are used for list/tuple.
                - Key(s) is/are used for dict.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper([(0, 10, (100, 1000)), (1, 11, (111, 1001)), (2, 12, (122, 1002)), (3, 13, (133, 1003)), (4, 14, (144, 1004))])
            >>> flatten_dp = dp.flatten(2)
            >>> list(flatten_dp)
            [(0, 10, 100, 1000), (1, 11, 111, 1001), (2, 12, 122, 1002), (3, 13, 133, 1003), (4, 14, 144, 1004)]
            >>>
            >>> dp = IterableWrapper([(0, (1, 2)), (3, (4, 5)), (6, (7, 8))])
            >>> flatten_dp = dp.flatten()
            >>> list(flatten_dp)
            [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        """
    
    # Functional form of 'FullSyncIterDataPipe'
    def fullsync(self, timeout=default_timeout_in_s) -> IterDataPipe:
        r"""
        Synchronizes data across distributed processes to prevent hanging during training,
        which is caused by uneven sharded data (functional name: ``fullsync``). It stops
        when the shortest distributed shard is exhausted. It would be appended at the end
        of the graph of ``DataPipe`` by ``DistributedReadingService`` automatically.
    
        Args:
            datapipe: IterDataPipe that needs to be synchronized
            timeout: Timeout for prefetching data in seconds. Default value equals to 30 minutes
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> # Distributed training with world size 2
            >>> world_size = 2
            >>> dp = IterableWrapper(list(range(23))).sharding_filter()
            >>> torch.utils.data.graph_settings.apply_sharding(dp, world_size, rank)
            >>> # Rank 0 has 12 elements; Rank 1 has 11 elements
            >>> for d in dp:
            ...     model(d)  # Hanging at the end of epoch due to uneven sharding
            >>> dp = dp.fullsync()
            >>> # Both ranks have 11 elements
            >>> for d in dp:
            ...     model(d)  # Not hanging anymore
        """
    
    # Functional form of 'HeaderIterDataPipe'
    def header(self, limit: Optional[int] = 10) -> IterDataPipe:
        r"""
        Yields elements from the source DataPipe from the start, up to the specfied limit (functional name: ``header``).
    
        If you would like to manually set the length of a DataPipe to a certain value; we recommend you to
        use :class:`.LengthSetter`.
    
        Args:
            source_datapipe: the DataPipe from which elements will be yielded
            limit: the number of elements to yield before stopping
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(range(10))
            >>> header_dp = dp.header(3)
            >>> list(header_dp)
            [0, 1, 2]
        """
    
    # Functional form of 'InBatchShufflerIterDataPipe'
    def in_batch_shuffle(self) -> IterDataPipe:
        r"""
        Shuffles each mini-batch from the prior DataPipe (functional name: ``in_batch_shuffle``).
    
        Args:
            datapipe: Iterable DataPipe with batched data
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper(range(10))
            >>> batch_dp = source_dp.batch(batch_size=3, drop_last=True)
            >>> list(batch_dp)
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            >>> in_batch_shuffle_dp = batch_dp.in_batch_shuffle()
            >>> list(in_batch_shuffle_dp)
            [[2, 0, 1], [3, 5, 4], [7, 8, 6]]
        """
    
    # Functional form of 'InMemoryCacheHolderIterDataPipe'
    def in_memory_cache(self, size: Optional[int] = None) -> IterDataPipe:
        r"""
        Stores elements from the source DataPipe in memory, up to a size limit
        if specified (functional name: ``in_memory_cache``). This cache is FIFO - once the cache is full,
        further elements will not be added to the cache until the previous ones are yielded and popped off from the cache.
    
        Args:
            source_dp: source DataPipe from which elements are read and stored in memory
            size: The maximum size (in megabytes) that this DataPipe can hold in memory. This defaults to unlimited.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper(range(10))
            >>> cache_dp = source_dp.in_memory_cache(size=5)
            >>> list(cache_dp)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
    
    # Functional form of 'ParagraphAggregatorIterDataPipe'
    def lines_to_paragraphs(self, joiner: Callable= ...) -> IterDataPipe:
        r"""
        Aggregates lines of text from the same file into a single paragraph (functional name: ``lines_to_paragraphs``).
        Specifically, this accepts a DataPipe consisting of tuples of a file name and a line. For each tuple,
        it checks if the file name matches the file name from the previous tuple. If yes, it joins the current line
        with existing paragraph. If the file names do not match, the existing paragraph is yielded and a new
        paragraph starts.
    
        Args:
            source_datapipe: a DataPipe with tuples of a file name and a line
            joiner: a function that joins a list of lines together
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper(
            >>>                 [("file1", "Line1"), ("file1", "Line2"), ("file2", "Line2,1"), ("file2", "Line2,2"), ("file2", "Line2,3")]
            >>>             )
            >>> para_agg_dp = source_dp.lines_to_paragraphs(joiner=lambda ls: " ".join(ls))
            >>> list(para_agg_dp)
            [('file1', 'Line1 Line2'), ('file2', 'Line2,1 Line2,2 Line2,3')]
        """
    
    # Functional form of 'AISFileListerIterDataPipe'
    def list_files_by_ais(self, url: str, length: int = -1) -> IterDataPipe:
        """
        Iterable Datapipe that lists files from the AIStore backends with the given URL prefixes
        (functional name: ``list_files_by_ais``).
        Acceptable prefixes include but not limited to - `ais://bucket-name`, `ais://bucket-name/`
    
        Note:
            - This function also supports files from multiple backends (`aws://..`, `gcp://..`, `azure://..`, etc)
            - Input must be a list and direct URLs are not supported.
            - length is -1 by default, all calls to len() are invalid as
                not all items are iterated at the start.
            - This internally uses AIStore Python SDK.
    
        Args:
            source_datapipe(IterDataPipe[str]): a DataPipe that contains URLs/URL
                prefixes to objects on AIS
            url(str): AIStore endpoint
            length(int): length of the datapipe
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, AISFileLister
            >>> ais_prefixes = IterableWrapper(['gcp://bucket-name/folder/', 'aws:bucket-name/folder/', 'ais://bucket-name/folder/', ...])
            >>> dp_ais_urls = AISFileLister(url='localhost:8080', source_datapipe=ais_prefixes)
            >>> for url in dp_ais_urls:
            ...     pass
            >>> # Functional API
            >>> dp_ais_urls = ais_prefixes.list_files_by_ais(url='localhost:8080')
            >>> for url in dp_ais_urls:
            ...     pass
        """
    
    # Functional form of 'FSSpecFileListerIterDataPipe'
    def list_files_by_fsspec(self, masks: Union[str, List[str]] = "", **kwargs) -> IterDataPipe:
        r"""
        Lists the contents of the directory at the provided ``root`` pathname or URL,
        and yields the full pathname or URL for each file within the
        directory (functional name: ``list_files_by_fsspec``).
    
        Args:
            root: The root `fsspec` path directory or list of path directories to list files from
            masks: Unix style filter string or string list for filtering file name(s)
            kwargs: Extra options that make sense to a particular storage connection,
                e.g. host, port, username, password, etc.
    
        Example:
    
        .. testsetup::
    
            dir_path = "path"
    
        .. testcode::
    
            from torchdata.datapipes.iter import FSSpecFileLister
    
            datapipe = FSSpecFileLister(root=dir_path)
        """
    
    # Functional form of 'IoPathFileListerIterDataPipe'
    def list_files_by_iopath(self, masks: Union[str, List[str]] = "", *, pathmgr=None, handler=None) -> IterDataPipe:
        r"""
        Lists the contents of the directory at the provided ``root`` pathname or URL,
        and yields the full pathname or URL for each file within the directory (functional name: ``list_files_by_iopath``).
    
        Args:
            root: The root local filepath or URL directory or list of roots to list files from
            masks: Unix style filter string or string list for filtering file name(s)
            pathmgr: Custom ``iopath.PathManager``. If not specified, a default ``PathManager`` is created.
    
        Note:
            Default ``PathManager`` currently supports local file path, normal HTTP URL and OneDrive URL.
            S3 URL is supported only with ``iopath``>=0.1.9.
    
        Example:
    
        .. testsetup::
    
            s3_url = "path"
    
        .. testcode::
    
            from torchdata.datapipes.iter import IoPathFileLister
    
            datapipe = IoPathFileLister(root=s3_url)
        """
    
    # Functional form of 'S3FileListerIterDataPipe'
    def list_files_by_s3(self, length: int = -1, request_timeout_ms=-1, region="", masks: Union[str, List[str]] = "") -> IterDataPipe:
        r"""
        Iterable DataPipe that lists Amazon S3 file URLs with the given prefixes (functional name: ``list_files_by_s3``).
        Acceptable prefixes include ``s3://bucket-name``, ``s3://bucket-name/``, ``s3://bucket-name/folder``.
    
        Note:
            1. ``source_datapipe`` **must** contain a list of valid S3 URLs
            2. ``length`` is `-1` by default, and any call to ``__len__()`` is invalid, because the length is unknown
               until all files are iterated.
            3. ``request_timeout_ms`` and ``region`` will overwrite settings in the configuration file or
               environment variables.
            4. The lack of AWS proper configuration can lead empty response. For more details related to S3 IO DataPipe
               setup and AWS config, please see the `README file`_.
    
        .. _README file:
            https://github.com/pytorch/data/tree/main/torchdata/datapipes/iter/load#s3-io-datapipe-documentation
    
        Args:
            source_datapipe: a DataPipe that contains URLs/URL prefixes to s3 files
            length: Nominal length of the datapipe
            request_timeout_ms: timeout setting for each reqeust (3,000ms by default)
            region: region for access files (inferred from credentials by default)
    
        Example:
    
        .. testsetup::
    
            from unittest import mock
            from torchdata.datapipes.iter import IterableWrapper, S3FileLister
    
            file_lister_patch = mock.patch.object(S3FileLister, "__iter__", return_value=iter([]))
            file_lister_patch.start()
    
        .. testcode::
    
            from torchdata.datapipes.iter import IterableWrapper, S3FileLister
    
            s3_prefixes = IterableWrapper(['s3://bucket-name/folder/', ...])
    
            dp_s3_urls = S3FileLister(s3_prefixes)
            for d in dp_s3_urls:
                pass
    
            # Functional API
            dp_s3_urls = s3_prefixes.list_files_by_s3(request_timeout_ms=100)
            for d in dp_s3_urls:
                pass
    
        .. testcleanup::
    
            file_lister_patch.stop()
        """
    
    # Functional form of 'AISFileLoaderIterDataPipe'
    def load_files_by_ais(self, url: str, length: int = -1) -> IterDataPipe:
        """
        Iterable DataPipe that loads files from AIStore with the given URLs (functional name: ``load_files_by_ais``).
        Iterates all files in BytesIO format and returns a tuple (url, BytesIO).
    
        Note:
        -   This function also supports files from multiple backends (`aws://..`, `gcp://..`, `azure://..`, etc)
        -   Input must be a list and direct URLs are not supported.
        -   This internally uses AIStore Python SDK.
    
        Args:
            source_datapipe(IterDataPipe[str]): a DataPipe that contains URLs/URL prefixes to objects
            url(str): AIStore endpoint
            length(int): length of the datapipe
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, AISFileLister,AISFileLoader
            >>> ais_prefixes = IterableWrapper(['gcp://bucket-name/folder/', 'aws:bucket-name/folder/', 'ais://bucket-name/folder/', ...])
            >>> dp_ais_urls = AISFileLister(url='localhost:8080', source_datapipe=ais_prefixes)
            >>> dp_cloud_files = AISFileLoader(url='localhost:8080', source_datapipe=dp_ais_urls)
            >>> for url, file in dp_cloud_files:
            ...     pass
            >>> # Functional API
            >>> dp_cloud_files = dp_ais_urls.load_files_by_ais(url='localhost:8080')
            >>> for url, file in dp_cloud_files:
            ...     pass
        """
    
    # Functional form of 'S3FileLoaderIterDataPipe'
    def load_files_by_s3(self, request_timeout_ms=-1, region="", buffer_size=None, multi_part_download=None) -> IterDataPipe:
        r"""
        Iterable DataPipe that loads Amazon S3 files from the given S3 URLs (functional name: ``load_files_by_s3``).
        ``S3FileLoader`` iterates all given S3 URLs in ``BytesIO`` format with ``(url, BytesIO)`` tuples.
    
        Note:
            1. ``source_datapipe`` **must** contain a list of valid S3 URLs.
            2. ``request_timeout_ms`` and ``region`` will overwrite settings in the
               configuration file or environment variables.
            3. The lack of AWS proper configuration can lead empty response. For more details related to S3 IO DataPipe
               setup and AWS config, please see the `README file`_.
    
        .. _README file:
            https://github.com/pytorch/data/tree/main/torchdata/datapipes/iter/load#s3-io-datapipe-documentation
    
        Args:
            source_datapipe: a DataPipe that contains URLs to s3 files
            request_timeout_ms: timeout setting for each reqeust (3,000ms by default)
            region: region for access files (inferred from credentials by default)
            buffer_size: buffer size of each chunk to download large files progressively (128Mb by default)
            multi_part_download: flag to split each chunk into small packets and download those packets in parallel (enabled by default)
    
        Example:
    
        .. testsetup::
    
            from unittest import mock
            from torchdata.datapipes.iter import S3FileLister
    
            file_lister_patch = mock.patch.object(S3FileLister, "__iter__", return_value=iter([]))
            file_lister_patch.start()
    
        .. testcode::
    
            from torchdata.datapipes.iter import IterableWrapper, S3FileLoader
    
            dp_s3_urls = IterableWrapper(['s3://bucket-name/folder/', ...]).list_files_by_s3()
            # In order to make sure data are shuffled and sharded in the
            # distributed environment, `shuffle`  and `sharding_filter`
            # are required. For detail, please check our tutorial in:
            # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
            sharded_s3_urls = dp_s3_urls.shuffle().sharding_filter()
    
            dp_s3_files = S3FileLoader(sharded_s3_urls)
            for url, fd in dp_s3_files: # Start loading data
                data = fd.read()
    
            # Functional API
            dp_s3_files = sharded_s3_urls.load_files_by_s3(buffer_size=256)
            for url, fd in dp_s3_files:
                data = fd.read()
    
        .. testcleanup::
    
            file_lister_patch.stop()
        """
    
    # Functional form of 'Bz2FileLoaderIterDataPipe'
    def load_from_bz2(self, length: int = -1) -> IterDataPipe:
        r"""
        Decompresses bz2 binary streams from an Iterable DataPipe which contains tuples of
        path name and bz2 binary streams, and yields a tuple of path name and extracted binary
        stream (functional name: ``load_from_bz2``).
    
        Args:
            datapipe: Iterable DataPipe that provides tuples of path name and bz2 binary stream
            length: Nominal length of the DataPipe
    
        Note:
            The opened file handles will be closed automatically if the default ``DecoderDataPipe``
            is attached. Otherwise, user should be responsible to close file handles explicitly
            or let Python's GC close them periodically.
    
        Example:
            >>> from torchdata.datapipes.iter import FileLister, FileOpener
            >>> datapipe1 = FileLister(".", "*.bz2")
            >>> datapipe2 = FileOpener(datapipe1, mode="b")
            >>> bz2_loader_dp = datapipe2.load_from_bz2()
            >>> for _, stream in bz2_loader_dp:
            >>>     print(stream.read())
            b'0123456789abcdef'
        """
    
    # Functional form of 'RarArchiveLoaderIterDataPipe'
    def load_from_rar(self, *, length: int = -1) -> IterDataPipe:
        r"""
        Decompresses rar binary streams from input Iterable Datapipes which contains tuples of path name and rar
        binary stream, and yields  a tuple of path name and extracted binary stream (functional name: ``load_from_rar``).
    
        Note:
            The nested RAR archive is not supported by this DataPipe
            due to the limitation of the archive type. Please extract
            outer RAR archive before reading the inner archive.
    
        Args:
            datapipe: Iterable DataPipe that provides tuples of path name and rar binary stream
            length: Nominal length of the DataPipe
    
        Example:
            >>> from torchdata.datapipes.iter import FileLister, FileOpener
            >>> datapipe1 = FileLister(".", "*.rar")
            >>> datapipe2 = FileOpener(datapipe1, mode="b")
            >>> rar_loader_dp = datapipe2.load_from_rar()
            >>> for _, stream in rar_loader_dp:
            >>>     print(stream.read())
            b'0123456789abcdef'
        """
    
    # Functional form of 'TarArchiveLoaderIterDataPipe'
    def load_from_tar(self, mode: str = "r:*", length: int = -1) -> IterDataPipe:
        r"""
        Opens/decompresses tar binary streams from an Iterable DataPipe which contains tuples of path name and
        tar binary stream, and yields a tuple of path name and extracted binary stream (functional name: ``load_from_tar``).
    
        Args:
            datapipe: Iterable DataPipe that provides tuples of path name and tar binary stream
            mode: File mode used by `tarfile.open` to read file object.
                Mode has to be a string of the form `'filemode[:compression]'`
            length: a nominal length of the DataPipe
    
        Note:
            The opened file handles will be closed automatically if the default ``DecoderDataPipe``
            is attached. Otherwise, user should be responsible to close file handles explicitly
            or let Python's GC close them periodically.
    
        Example:
            >>> from torchdata.datapipes.iter import FileLister, FileOpener
            >>> datapipe1 = FileLister(".", "*.tar")
            >>> datapipe2 = FileOpener(datapipe1, mode="b")
            >>> tar_loader_dp = datapipe2.load_from_tar()
            >>> for _, stream in tar_loader_dp:
            >>>     print(stream.read())
            b'0123456789abcdef'
        """
    
    # Functional form of 'TFRecordLoaderIterDataPipe'
    def load_from_tfrecord(self, spec: Optional[TFRecordExampleSpec] = None, length: int = -1) -> IterDataPipe:
        r"""
        Opens/decompresses tfrecord binary streams from an Iterable DataPipe which contains tuples of path name and
        tfrecord binary stream, and yields the stored records (functional name: ``load_from_tfrecord``).
    
        Args:
            datapipe: Iterable DataPipe that provides tuples of path name and tfrecord binary stream
            length: a nominal length of the DataPipe
    
        Note:
            The opened file handles will be closed automatically if the default ``DecoderDataPipe``
            is attached. Otherwise, user should be responsible to close file handles explicitly
            or let Python's GC close them periodically.
    
        Example:
            >>> from torchdata.datapipes.iter import FileLister, FileOpener
            >>> datapipe1 = FileLister(".", "*.tfrecord")
            >>> datapipe2 = FileOpener(datapipe1, mode="b")
            >>> tfrecord_loader_dp = datapipe2.load_from_tfrecord()
            >>> for example in tfrecord_loader_dp:
            >>>     print(example)
        """
    
    # Functional form of 'XzFileLoaderIterDataPipe'
    def load_from_xz(self, length: int = -1) -> IterDataPipe:
        r"""
        Decompresses xz (lzma) binary streams from an Iterable DataPipe which contains tuples of
        path name and xy binary streams, and yields a tuple of path name and extracted binary
        stream (functional name: ``load_from_xz``).
    
        Args:
            datapipe: Iterable DataPipe that provides tuples of path name and xy binary stream
            length: Nominal length of the DataPipe
    
        Note:
            The opened file handles will be closed automatically if the default ``DecoderDataPipe``
            is attached. Otherwise, user should be responsible to close file handles explicitly
            or let Python's GC close them periodically.
    
        Example:
            >>> from torchdata.datapipes.iter import FileLister, FileOpener
            >>> datapipe1 = FileLister(".", "*.xz")
            >>> datapipe2 = FileOpener(datapipe1, mode="b")
            >>> xz_loader_dp = datapipe2.load_from_xz()
            >>> for _, stream in xz_loader_dp:
            >>>     print(stream.read())
            b'0123456789abcdef'
        """
    
    # Functional form of 'ZipArchiveLoaderIterDataPipe'
    def load_from_zip(self, length: int = -1) -> IterDataPipe:
        r"""
        Opens/decompresses zip binary streams from an Iterable DataPipe which contains a tuple of path name and
        zip binary stream, and yields a tuple of path name and extracted binary stream (functional name: ``load_from_zip``).
    
        Args:
            datapipe: Iterable DataPipe that provides tuples of path name and zip binary stream
            length: Nominal length of the DataPipe
    
        Note:
            The opened file handles will be closed automatically if the default ``DecoderDataPipe``
            is attached. Otherwise, user should be responsible to close file handles explicitly
            or let Python's GC close them periodically. Due to how `zipfiles` implements its ``open()`` method,
            the data_stream variable below cannot be closed within the scope of this function.
    
        Example:
            >>> from torchdata.datapipes.iter import FileLister, FileOpener
            >>> datapipe1 = FileLister(".", "*.zip")
            >>> datapipe2 = FileOpener(datapipe1, mode="b")
            >>> zip_loader_dp = datapipe2.load_from_zip()
            >>> for _, stream in zip_loader_dp:
            >>>     print(stream.read())
            b'0123456789abcdef'
        """
    
    # Functional form of 'ParquetDFLoaderIterDataPipe'
    def load_parquet_as_df(self, dtype=None, columns: Optional[List[str]] = None, device: str = "", use_threads: bool = False) -> IterDataPipe:
        r"""
        Takes in paths to Parquet files and return a `TorchArrow` DataFrame for each row group
        within a Parquet file (functional name: ``load_parquet_as_df``).
    
        Args:
            source_dp: source DataPipe containing paths to the Parquet files
            columns: List of `str` that specifies the column names of the DataFrame
            use_threads: if ``True``, Parquet reader will perform multi-threaded column reads
            dtype: specify the `TorchArrow` dtype for the DataFrame, use ``torcharrow.dtypes.DType``
            device: specify the device on which the DataFrame will be stored
    
        Example:
            >>> from torchdata.datapipes.iter import FileLister
            >>> import torcharrow.dtypes as dt
            >>> DTYPE = dt.Struct([dt.Field("Values", dt.int32)])
            >>> source_dp = FileLister(".", masks="df*.parquet")
            >>> parquet_df_dp = source_dp.load_parquet_as_df(dtype=DTYPE)
            >>> list(parquet_df_dp)[0]
              index    Values
            -------  --------
                  0         0
                  1         1
                  2         2
            dtype: Struct([Field('Values', int32)]), count: 3, null_count: 0
        """
    
    # Functional form of 'BatchMapperIterDataPipe'
    def map_batches(self, fn: Callable, batch_size: int, input_col=None) -> IterDataPipe:
        r"""
        Combines elements from the source DataPipe to batches and applies a function
        over each batch, then flattens the outputs to a single, unnested IterDataPipe
        (functional name: ``map_batches``).
    
        Args:
            datapipe: Source IterDataPipe
            fn: The function to be applied to each batch of data
            batch_size: The size of batch to be aggregated from ``datapipe``
            input_col: Index or indices of data which ``fn`` is applied, such as:
    
                - ``None`` as default to apply ``fn`` to the data directly.
                - Integer(s) is used for list/tuple.
                - Key(s) is used for dict.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> def fn(batch):
            >>>     return [d + 1 for d in batch]
            >>> source_dp = IterableWrapper(list(range(5)))
            >>> mapped_dp = source_dp.map_batches(fn, batch_size=3)
            >>> list(mapped_dp)
            [1, 2, 3, 4, 5]
    
        Notes:
            Compared with ``map``, the reason that ``map_batches`` doesn't take
            ``output_col`` argument is the size of ``fn`` output is not guaranteed
            to be the same as input batch. With different size, this operation cannot
            assign data back to original data structure.
    
            And, this operation is introduced based on the use case from `TorchText`.
            A pybinded C++ vectorized function can be applied for efficiency.
        """
    
    # Functional form of 'MaxTokenBucketizerIterDataPipe'
    def max_token_bucketize(self, max_token_count: int, len_fn: Callable= ..., min_len: int = 0, max_len: Optional[int] = None, buffer_size: int = 1000, include_padding: bool = False) -> IterDataPipe:
        r"""
        Creates mini-batches of data from a min-heap with limited size, and the total length of samples
        returned by ``len_fn`` within each batch will be limited by ``max_token_count``
        (functional name: ``max_token_bucketize``). If ``min_len`` or ``max_len`` is set, the samples with
        length that is out of ``[min_len, max_len]`` will be filtered out.
    
        The purpose of this DataPipe is to batch samples with similar length according to ``len_fn``.
        Min-heap is used here to make sure the samples are sorted incrementally based on the length. And,
        the total length of samples in each batch is guaranteed to be smaller than ``max_token_count``.
        For an example in the audio domain, it may be batching samples with similar length. Then, given the
        ``max_token_count``, each batch may be concatenated to a Tensor with the same size and minimum padding.
    
        If ``include_padding`` is set to ``True``, the token count of each batch includes the padding a succeeding
        DataPipe could add. This guarentees that even after the batch is padded, ``max_token_count`` will not be exceeded.
        This can prevent out-of-memory issues for data with large variations in length.
    
        Note that batches are bucketized starting from the smallest size in a buffer.
        This can limit the variablity of batches if ``buffer_size`` is large.
        To increase variablity, apply ``torchdata.datapipes.iter.Shuffler`` before and after this DataPipe,
        and keep ``buffer_size`` small.
    
    
        Args:
            datapipe: Iterable DataPipe being batched
            max_token_count: Maximum length of total length of data in each batch
            len_fn: Function to be applied to each element to get lengths. ``len(data)`` is used by default.
            min_len: Optional minimum length to be included into each batch
            max_len: Optional maximum length to be included into each batch.
            buffer_size: This restricts how many samples are taken from prior DataPipe to bucketize
            include_padding: If True, the size of each batch includes the extra padding to the largest length in the batch.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper(['1', '11', '1', '1111', '111', '1', '11', '11', '111'])
            >>> # Using default len_fn to sort samples based on length (string length in this case)
            >>> batch_dp = source_dp.max_token_bucketize(max_token_count=5)
            >>> list(batch_dp)
            [['1', '1', '1', '11'], ['11', '11'], ['111'], ['111'], ['1111']]
            >>> batch_dp = source_dp.max_token_bucketize(max_token_count=4, buffer_size=4)
            >>> list(batch_dp)
            [['1', '1', '1'], ['11', '11'], ['11'], ['111'], ['111'], ['1111']]
        """
    
    # Functional form of '_MemoryCellIterDataPipe'
    def memory_cell(self, remember_elements=1000) -> IterDataPipe:
        ...
    
    # Functional form of 'MultiplexerLongestIterDataPipe'
    def mux_longest(self, *datapipes) -> IterDataPipe:
        r"""
        Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux_longest``). As in,
        one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
        and so on. It skips over DataPipes that are exhausted, and ends when all input DataPipes are exhausted.
    
        Args:
            datapipes: Iterable DataPipes that will take turn to yield their elements, until they are all exhausted
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
            >>> list(dp1.mux_longest(dp2, dp3))
            [0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24]
        """
    
    # Functional form of 'OnDiskCacheHolderIterDataPipe'
    def on_disk_cache(self, filepath_fn: Optional[Callable] = None, hash_dict: Dict[str, str] = None, hash_type: str = "sha256", extra_check_fn: Optional[Callable[[str], bool]] = None) -> IterDataPipe:
        """
        Caches the outputs of multiple DataPipe operations to local files, which are
        typically performance bottleneck such download, decompress, and etc (functional name: ``on_disk_cache``).
    
        Must use ``.end_caching()`` to stop tracing the sequence of DataPipe operations and save the results to local files.
    
        Args:
            source_datapipe: IterDataPipe
            filepath_fn: Given data from ``source_datapipe``, returns file path(s) on local file system.
                Single file path is only allowed as output of the function.
                If resulted file name is different from the filename generated by the filename function of the end_cache
                original file name used to store list of yield files (and as cached items availability check)
            hash_dict: A Dictionary mapping file names to their corresponding hashes. If ``hash_dict`` is specified,
                the extra hash check will be attached before saving data to local file system. If the data
                doesn't meet the hash, the pipeline will raise an Error.
            hash_type: The type of hash function to apply
            extra_check_fn: Optional function to carry out extra validation on
                the given file path from ``filepath_fn``.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, HttpReader
            >>> url = IterableWrapper(["https://path/to/filename", ])
            >>> def _filepath_fn(url):
            >>>     temp_dir = tempfile.gettempdir()
            >>>     return os.path.join(temp_dir, os.path.basename(url))
            >>> hash_dict = {"expected_filepath": expected_MD5_hash}
            >>> cache_dp = url.on_disk_cache(filepath_fn=_filepath_fn, hash_dict=_hash_dict, hash_type="md5")
            >>> # You must call ``.end_caching`` at a later point to stop tracing and save the results to local files.
            >>> cache_dp = HttpReader(cache_dp).end_caching(mode="wb", filepath_fn=_filepath_fn)
        """
    
    # Functional form of 'FSSpecFileOpenerIterDataPipe'
    def open_files_by_fsspec(self, mode: str = "r", *, kwargs_for_open: Optional[Dict] = None, **kwargs) -> IterDataPipe:
        r"""
        Opens files from input datapipe which contains `fsspec` paths and yields a tuple of
        pathname and opened file stream (functional name: ``open_files_by_fsspec``).
    
        Args:
            source_datapipe: Iterable DataPipe that provides the pathnames or URLs
            mode: An optional string that specifies the mode in which the file is opened (``"r"`` by default)
            kwargs_for_open: Optional Dict to specify kwargs for opening files (``fs.open()``)
            kwargs: Extra options that are used to establish a particular storage connection,
                e.g. host, port, username, password, etc.
    
        Example:
    
        .. testsetup::
    
            dir_path = "path"
    
        .. testcode::
    
            from torchdata.datapipes.iter import FSSpecFileLister
    
            datapipe = FSSpecFileLister(root=dir_path)
            file_dp = datapipe.open_files_by_fsspec()
        """
    
    # Functional form of 'IoPathFileOpenerIterDataPipe'
    def open_files_by_iopath(self, mode: str = "r", pathmgr=None, handler=None) -> IterDataPipe:
        r"""
        Opens files from input datapipe which contains pathnames or URLs,
        and yields a tuple of pathname and opened file stream (functional name: ``open_files_by_iopath``).
    
        Args:
            source_datapipe: Iterable DataPipe that provides the pathnames or URLs
            mode: An optional string that specifies the mode in which the file is opened (``"r"`` by default)
            pathmgr: Custom ``iopath.PathManager``. If not specified, a default ``PathManager`` is created.
    
        Note:
            Default ``PathManager`` currently supports local file path, normal HTTP URL and OneDrive URL.
            S3 URL is supported only with `iopath`>=0.1.9.
    
        Example:
    
        .. testsetup::
    
            s3_url = "path"
    
        .. testcode::
    
            from torchdata.datapipes.iter import IoPathFileLister
    
            datapipe = IoPathFileLister(root=s3_url)
            file_dp = datapipe.open_files_by_iopath()
        """
    
    # Functional form of 'CSVParserIterDataPipe'
    def parse_csv(self, *, skip_lines: int = 0, decode: bool = True, encoding: str = "utf-8", errors: str = "ignore", return_path: bool = False, as_tuple: bool = False, **fmtparams) -> IterDataPipe:
        r"""
        Accepts a DataPipe consists of tuples of file name and CSV data stream,
        reads and returns the contents within the CSV files one row at a time (functional name: ``parse_csv``).
        Each output is a `List` by default, but it depends on ``fmtparams``.
    
        Args:
            source_datapipe: source DataPipe with tuples of file name and CSV data stream
            skip_lines: number of lines to skip at the beginning of each file
            strip_newline: if ``True``, the new line character will be stripped
            decode: if ``True``, this will decode the contents of the file based on the specified ``encoding``
            encoding: the character encoding of the files (`default='utf-8'`)
            errors: the error handling scheme used while decoding
            return_path: if ``True``, each line will return a tuple of path and contents, rather
                than just the contents
            as_tuple: if ``True``, each line will return a tuple instead of a list
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, FileOpener
            >>> import os
            >>> def get_name(path_and_stream):
            >>>     return os.path.basename(path_and_stream[0]), path_and_stream[1]
            >>> datapipe1 = IterableWrapper(["1.csv", "empty.csv", "empty2.csv"])
            >>> datapipe2 = FileOpener(datapipe1, mode="b")
            >>> datapipe3 = datapipe2.map(get_name)
            >>> csv_parser_dp = datapipe3.parse_csv()
            >>> list(csv_parser_dp)
            [['key', 'item'], ['a', '1'], ['b', '2'], []]
        """
    
    # Functional form of 'CSVDictParserIterDataPipe'
    def parse_csv_as_dict(self, *, skip_lines: int = 0, decode: bool = True, encoding: str = "utf-8", errors: str = "ignore", return_path: bool = False, **fmtparams) -> IterDataPipe:
        r"""
        Accepts a DataPipe consists of tuples of file name and CSV data stream, reads and returns the contents
        within the CSV files one row at a time (functional name: ``parse_csv_as_dict``).
    
        Each output is a `Dict` by default, but it depends on ``fmtparams``. The first row of each file, unless skipped,
        will be used as the header; the contents of the header row will be used as keys for the `Dict`\s
        generated from the remaining rows.
    
        Args:
            source_datapipe: source DataPipe with tuples of file name and CSV data stream
            skip_lines: number of lines to skip at the beginning of each file
            strip_newline: if ``True``, the new line character will be stripped
            decode: if ``True``, this will decode the contents of the file based on the specified ``encoding``
            encoding: the character encoding of the files (`default='utf-8'`)
            errors: the error handling scheme used while decoding
            return_path: if ``True``, each line will return a tuple of path and contents, rather
                than just the contents
    
        Example:
            >>> from torchdata.datapipes.iter import FileLister, FileOpener
            >>> import os
            >>> def get_name(path_and_stream):
            >>>     return os.path.basename(path_and_stream[0]), path_and_stream[1]
            >>> datapipe1 = FileLister(".", "*.csv")
            >>> datapipe2 = FileOpener(datapipe1, mode="b")
            >>> datapipe3 = datapipe2.map(get_name)
            >>> csv_dict_parser_dp = datapipe3.parse_csv_as_dict()
            >>> list(csv_dict_parser_dp)
            [{'key': 'a', 'item': '1'}, {'key': 'b', 'item': '2'}]
        """
    
    # Functional form of 'JsonParserIterDataPipe'
    def parse_json_files(self, **kwargs) -> IterDataPipe:
        r"""
        Reads from JSON data streams and yields a tuple of file name and JSON data (functional name: ``parse_json_files``).
    
        Args:
            source_datapipe: a DataPipe with tuples of file name and JSON data stream
            kwargs: keyword arguments that will be passed through to ``json.loads``
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, FileOpener
            >>> import os
            >>> def get_name(path_and_stream):
            >>>     return os.path.basename(path_and_stream[0]), path_and_stream[1]
            >>> datapipe1 = IterableWrapper(["empty.json", "1.json", "2.json"])
            >>> datapipe2 = FileOpener(datapipe1, mode="b")
            >>> datapipe3 = datapipe2.map(get_name)
            >>> json_dp = datapipe3.parse_json_files()
            >>> list(json_dp)
            [('1.json', ['foo', {'bar': ['baz', None, 1.0, 2]}]), ('2.json', {'__complex__': True, 'real': 1, 'imag': 2})]
        """
    
    # Functional form of 'PinMemoryIterDataPipe'
    def pin_memory(self, device=None, pin_memory_fn=pin_memory_fn) -> IterDataPipe:
        r"""
        Prefetches one element from the source DataPipe and moves it to pinned memory (functional name: ``pin_memory``).
        When used with ``MultiProcessingReadingService``, this DataPipe would be kept in the main process to prevent
        duplicated CUDA context creation.
    
        Args:
            source_datapipe: IterDataPipe from which samples are moved to pinned memory.
            device: The device to pin samples.
            pin_memory_fn: Optional callable function to move data to pinned memory.
                A ``pin_memory_fn`` to handle general objects is provided by default.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(file_paths).open_files().readlines().map(tokenize_fn).pin_memory()
        """
    
    # Functional form of 'PrefetcherIterDataPipe'
    def prefetch(self, buffer_size: int = 10) -> IterDataPipe:
        r"""
        Prefetches elements from the source DataPipe and puts them into a buffer (functional name: ``prefetch``).
        Prefetching performs the operations (e.g. I/O, computations) of the DataPipes up to this one ahead of time
        and stores the result in the buffer, ready to be consumed by the subsequent DataPipe. It has no effect aside
        from getting the sample ready ahead of time.
    
        This is used by ``MultiProcessingReadingService`` when the arguments
        ``worker_prefetch_cnt`` (for prefetching at each worker process) or
        ``main_prefetch_cnt`` (for prefetching at the main loop) are greater than 0.
    
        Beyond the built-in use cases, this can be useful to put after I/O DataPipes that have
        expensive I/O operations (e.g. takes a long time to request a file from a remote server).
    
        Args:
            source_datapipe: IterDataPipe from which samples are prefetched
            buffer_size: the size of the buffer which stores the prefetched samples
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(file_paths).open_files().prefetch(5)
        """
    
    # Functional form of 'RandomSplitterIterDataPipe'
    def random_split(self, weights: Dict[T, Union[int, float]], seed, total_length: Optional[int] = None, target: Optional[T] = None) -> Union[IterDataPipe, List[IterDataPipe]]:
        r"""
        Randomly split samples from a source DataPipe into groups (functional name: ``random_split``).
        Since there is no buffer, only ONE group of samples (i.e. one child DataPipe) can be iterated through
        at any time. Attempts to iterate through multiple of them simultaneously will fail.
    
        Note that by default, multiple iterations of this DataPipe will yield the same split for consistency across epochs.
        You can invoke ``override_seed`` on the output(s) to update the seed whenever needed (such as per epoch to
        get a different split per epoch).
    
        Args:
            source_datapipe: Iterable DataPipe being split
            weights: Dict of weights; the length of this list determines how many output DataPipes there will be.
                It is recommended to provide integer weights that sum up to ``total_length``, which allows
                resulting DataPipes' length values to be known in advance.
            seed: random _seed used to determine the randomness of the split
            total_length: Length of the ``source_datapipe``, optional but providing an integer is highly encouraged,
                because not all ``IterDataPipe`` has ``len``, espeically ones that can be easily known in advance.
            target: Optional key (that must exist in ``weights``) to indicate the specific group to return.
                If set to the default ``None``, returns ``List[IterDataPipe]``.
                If target is specified, returns ``IterDataPipe``.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(range(10))
            >>> train, valid = dp.random_split(total_length=10, weights={"train": 0.5, "valid": 0.5}, seed=0)
            >>> list(train)
            [2, 3, 5, 7, 8]
            >>> list(valid)
            [0, 1, 4, 6, 9]
            >>> # You can also specify a target key if you only need a specific group of samples
            >>> train = dp.random_split(total_length=10, weights={"train": 0.5, "valid": 0.5}, seed=0, target='train')
            >>> list(train)
            [2, 3, 5, 7, 8]
            >>> # Be careful to use the same seed as before when specifying `target` to get the correct split.
            >>> valid = dp.random_split(total_length=10, weights={"train": 0.5, "valid": 0.5}, seed=0, target='valid')
            >>> list(valid)
            [0, 1, 4, 6, 9]
        """
    
    # Functional form of 'GDriveReaderDataPipe'
    def read_from_gdrive(self, *, timeout: Optional[float] = None, skip_on_error: bool = False, **kwargs: Optional[Dict[str, Any]]) -> IterDataPipe:
        r"""
        Takes URLs pointing at GDrive files, and yields tuples of file name and
        IO stream (functional name: ``read_from_gdrive``).
    
        Args:
            source_datapipe: a DataPipe that contains URLs to GDrive files
            timeout: timeout in seconds for HTTP request
            skip_on_error: whether to skip over urls causing problems, otherwise an exception is raised
            **kwargs: a Dictionary to pass optional arguments that requests takes. For the full list check out https://docs.python-requests.org/en/master/api/
    
        Example:
    
        .. testsetup::
    
            from torchdata.datapipes.iter import GDriveReader
    
            GDriveReader.readlines = lambda self: [
                ("https://drive.google.com/uc?export=download&id=SomeIDToAGDriveFile", b"<First line from the GDrive File>")
            ]
    
        .. testcode::
    
            from torchdata.datapipes.iter import IterableWrapper, GDriveReader
    
            gdrive_file_url = "https://drive.google.com/uc?export=download&id=SomeIDToAGDriveFile"
            gdrive_reader_dp = GDriveReader(IterableWrapper([gdrive_file_url]))
            reader_dp = gdrive_reader_dp.readlines()
            it = iter(reader_dp)
            path, line = next(it)
            print((path, line))
    
        Output:
    
        .. testoutput::
    
            ('https://drive.google.com/uc?export=download&id=SomeIDToAGDriveFile', b'<First line from the GDrive File>')
        """
    
    # Functional form of 'HTTPReaderIterDataPipe'
    def read_from_http(self, timeout: Optional[float] = None, skip_on_error: bool = False, **kwargs: Optional[Dict[str, Any]]) -> IterDataPipe:
        r"""
        Takes file URLs (HTTP URLs pointing to files), and yields tuples of file URL and
        IO stream (functional name: ``read_from_http``).
    
        Args:
            source_datapipe: a DataPipe that contains URLs
            timeout: timeout in seconds for HTTP request
            skip_on_error: whether to skip over urls causing problems, otherwise an exception is raised
            **kwargs: a Dictionary to pass optional arguments that requests takes. For the full list check out https://docs.python-requests.org/en/master/api/
    
        Example:
    
        .. testcode::
    
            from torchdata.datapipes.iter import IterableWrapper, HttpReader
    
            file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
            query_params = {"auth" : ("fake_username", "fake_password"), "allow_redirects" : True}
            timeout = 120
            http_reader_dp = HttpReader(IterableWrapper([file_url]), timeout=timeout, **query_params)
            reader_dp = http_reader_dp.readlines()
            it = iter(reader_dp)
            path, line = next(it)
            print((path, line))
    
        Output:
    
        .. testoutput::
    
            ('https://raw.githubusercontent.com/pytorch/data/main/LICENSE', b'BSD 3-Clause License')
        """
    
    # Functional form of 'OnlineReaderIterDataPipe'
    def read_from_remote(self, *, timeout: Optional[float] = None, skip_on_error: bool = False, **kwargs: Optional[Dict[str, Any]]) -> IterDataPipe:
        r"""
        Takes file URLs (can be HTTP URLs pointing to files or URLs to GDrive files), and
        yields tuples of file URL and IO stream (functional name: ``read_from_remote``).
    
        Args:
            source_datapipe: a DataPipe that contains URLs
            timeout: timeout in seconds for HTTP request
            skip_on_error: whether to skip over urls causing problems, otherwise an exception is raised
            **kwargs: a Dictionary to pass optional arguments that requests takes. For the full list check out https://docs.python-requests.org/en/master/api/
    
        Example:
    
        .. testcode::
    
            from torchdata.datapipes.iter import IterableWrapper, OnlineReader
    
            file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
            online_reader_dp = OnlineReader(IterableWrapper([file_url]))
            reader_dp = online_reader_dp.readlines()
            it = iter(reader_dp)
            path, line = next(it)
            print((path, line))
    
        Output:
    
        .. testoutput::
    
            ('https://raw.githubusercontent.com/pytorch/data/main/LICENSE', b'BSD 3-Clause License')
        """
    
    # Functional form of 'LineReaderIterDataPipe'
    def readlines(self, *, skip_lines: int = 0, strip_newline: bool = True, decode: bool = False, encoding="utf-8", errors: str = "ignore", return_path: bool = True) -> IterDataPipe:
        r"""
        Accepts a DataPipe consisting of tuples of file name and string data stream, and for each line in the
        stream, yields a tuple of file name and the line (functional name: ``readlines``).
    
        Args:
            source_datapipe: a DataPipe with tuples of file name and string data stream
            skip_lines: number of lines to skip at the beginning of each file
            strip_newline: if ``True``, the new line character will be stripped
            decode: if ``True``, this will decode the contents of the file based on the specified ``encoding``
            encoding: the character encoding of the files (`default='utf-8'`)
            errors: the error handling scheme used while decoding
            return_path: if ``True``, each line will return a tuple of path and contents, rather
                than just the contents
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> import io
            >>> text1 = "Line1\nLine2"
            >>> text2 = "Line2,1\r\nLine2,2\r\nLine2,3"
            >>> source_dp = IterableWrapper([("file1", io.StringIO(text1)), ("file2", io.StringIO(text2))])
            >>> line_reader_dp = source_dp.readlines()
            >>> list(line_reader_dp)
            [('file1', 'Line1'), ('file1', 'Line2'), ('file2', 'Line2,1'), ('file2', 'Line2,2'), ('file2', 'Line2,3')]
        """
    
    # Functional form of 'RepeaterIterDataPipe'
    def repeat(self, times: int) -> IterDataPipe:
        """
        Repeatedly yield each element of source DataPipe for the specified number of times before
        moving onto the next element (functional name: ``repeat``). Note that no copy is made in this DataPipe,
        the same element is yielded repeatedly.
    
        If you would like to yield the whole DataPipe in order multiple times, use :class:`.Cycler`.
    
        Args:
            source_datapipe: source DataPipe that will be iterated through
            times: the number of times an element of ``source_datapipe`` will be yielded before moving onto the next element
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(range(3))
            >>> dp = dp.repeat(2)
            >>> list(dp)
            [0, 0, 1, 1, 2, 2]
        """
    
    # Functional form of 'RoundRobinDemultiplexerIterDataPipe'
    def round_robin_demux(self, num_instances: int, buffer_size: int = 1000) -> List[IterDataPipe]:
        r"""
        Splits the input DataPipe into multiple child DataPipes in the round-robin order (functional name: ``round_robin_demux``).
        A list of the child DataPipes is returned from this operation.
    
        Args:
            datapipe: Iterable DataPipe being filtered
            num_instances: number of instances of the DataPipe to create
            buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
                DataPipes while waiting for their values to be yielded.
                Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
    
        Examples:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper(range(5))
            >>> dp1, dp2 = source_dp.round_robin_demux(2)
            >>> list(dp1)
            [0, 2, 4]
            >>> len(dp1)
            3
            >>> list(dp2)
            [1, 3]
            >>> len(dp2)
            2
        """
    
    # Functional form of 'Rows2ColumnarIterDataPipe'
    def rows2columnar(self, column_names: List[str] = None) -> IterDataPipe:
        r"""
        Accepts an input DataPipe with batches of data, and processes one batch
        at a time and yields a Dict for each batch, with ``column_names`` as keys and lists of
        corresponding values from each row as values (functional name: ``rows2columnar``).
    
        Within the input DataPipe, each row within a batch must either be a `Dict` or a `List`
    
        Note:
            If ``column_names`` are not given and each row is a `Dict`, the keys of that Dict will be used as column names.
    
        Args:
            source_datapipe: a DataPipe where each item is a batch. Within each batch,
                there are rows and each row is a `List` or `Dict`
            column_names: if each element in a batch contains `Dict`, ``column_names`` act as a filter for matching keys;
                otherwise, these are used as keys to for the generated `Dict` of each batch
    
        Example:
            >>> # Each element in a batch is a `Dict`
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper([[{'a': 1}, {'b': 2, 'a': 1}], [{'a': 1, 'b': 200}, {'b': 2, 'c': 3, 'a': 100}]])
            >>> row2col_dp = dp.rows2columnar()
            >>> list(row2col_dp)
            [defaultdict(<class 'list'>, {'a': [1, 1], 'b': [2]}),
             defaultdict(<class 'list'>, {'a': [1, 100], 'b': [200, 2], 'c': [3]})]
            >>> row2col_dp = dp.rows2columnar(column_names=['a'])
            >>> list(row2col_dp)
            [defaultdict(<class 'list'>, {'a': [1, 1]}),
             defaultdict(<class 'list'>, {'a': [1, 100]})]
            >>> # Each element in a batch is a `List`
            >>> dp = IterableWrapper([[[0, 1, 2, 3], [4, 5, 6, 7]]])
            >>> row2col_dp = dp.rows2columnar(column_names=["1st_in_batch", "2nd_in_batch", "3rd_in_batch", "4th_in_batch"])
            >>> list(row2col_dp)
            [defaultdict(<class 'list'>, {'1st_in_batch': [0, 4], '2nd_in_batch': [1, 5],
                                          '3rd_in_batch': [2, 6], '4th_in_batch': [3, 7]})]
        """
    
    # Functional form of 'FSSpecSaverIterDataPipe'
    def save_by_fsspec(self, mode: str = "w", filepath_fn: Optional[Callable] = None, *, kwargs_for_open: Optional[Dict] = None, **kwargs) -> IterDataPipe:
        r"""
        Takes in a DataPipe of tuples of metadata and data, saves the data to the target
        path (generated by the filepath_fn and metadata), and yields the resulting `fsspec`
        path (functional name: ``save_by_fsspec``).
    
        Args:
            source_datapipe: Iterable DataPipe with tuples of metadata and data
            mode: Mode in which the file will be opened for write the data (``"w"`` by default)
            filepath_fn: Function that takes in metadata and returns the target path of the new file
            kwargs_for_open: Optional Dict to specify kwargs for opening files (``fs.open()``)
            kwargs: Extra options that are used to establish a particular storage connection,
                e.g. host, port, username, password, etc.
    
    
        Example:
    
        .. testsetup::
    
            file_prefix = "file"
    
        .. testcode::
    
            from torchdata.datapipes.iter import IterableWrapper
    
    
            def filepath_fn(name: str) -> str:
                return file_prefix + name
    
    
            name_to_data = {"1.txt": b"DATA1", "2.txt": b"DATA2", "3.txt": b"DATA3"}
            source_dp = IterableWrapper(sorted(name_to_data.items()))
            fsspec_saver_dp = source_dp.save_by_fsspec(filepath_fn=filepath_fn, mode="wb")
            res_file_paths = list(fsspec_saver_dp)
    
        .. testcleanup::
    
            import os
    
            for name in name_to_data.keys():
                os.remove(file_prefix + name)
        """
    
    # Functional form of 'IoPathSaverIterDataPipe'
    def save_by_iopath(self, mode: str = "w", filepath_fn: Optional[Callable] = None, *, pathmgr=None, handler=None) -> IterDataPipe:
        r"""
        Takes in a DataPipe of tuples of metadata and data, saves the data
        to the target path which is generated by the ``filepath_fn`` and metadata, and yields the resulting path
        in `iopath` format (functional name: ``save_by_iopath``).
    
        Args:
            source_datapipe: Iterable DataPipe with tuples of metadata and data
            mode: Mode in which the file will be opened for write the data (``"w"`` by default)
            filepath_fn: Function that takes in metadata and returns the target path of the new file
            pathmgr: Custom ``iopath.PathManager``. If not specified, a default ``PathManager`` is created.
    
        Note:
            Default ``PathManager`` currently supports local file path, normal HTTP URL and OneDrive URL.
            S3 URL is supported only with `iopath`>=0.1.9.
    
        Example:
    
        .. testsetup::
    
            s3_url = "url"
    
        .. testcode::
    
            from torchdata.datapipes.iter import IterableWrapper
    
    
            def filepath_fn(name: str) -> str:
                return s3_url + name
    
    
            name_to_data = {"1.txt": b"DATA1", "2.txt": b"DATA2", "3.txt": b"DATA3"}
            source_dp = IterableWrapper(sorted(name_to_data.items()))
            iopath_saver_dp = source_dp.save_by_iopath(filepath_fn=filepath_fn, mode="wb")
            res_file_paths = list(iopath_saver_dp)
    
        .. testcleanup::
    
            import os
    
            for file in ["1.txt", "1.txt.lock", "2.txt", "2.txt.lock", "3.txt", "3.txt.lock"]:
                os.remove(s3_url + file)
        """
    
    # Functional form of 'SaverIterDataPipe'
    def save_to_disk(self, mode: str = "w", filepath_fn: Optional[Callable] = None) -> IterDataPipe:
        r"""
        Takes in a DataPipe of tuples of metadata and data, saves the data
        to the target path generated by the ``filepath_fn`` and metadata, and yields file path on local file
        system (functional name: ``save_to_disk``).
    
        Args:
            source_datapipe: Iterable DataPipe with tuples of metadata and data
            mode: Node in which the file will be opened for write the data (``"w"`` by default)
            filepath_fn: Function that takes in metadata and returns the target path of the new file
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> import os
            >>> def filepath_fn(name: str) -> str:
            >>>     return os.path.join(".", os.path.basename(name))
            >>> name_to_data = {"1.txt": b"DATA1", "2.txt": b"DATA2", "3.txt": b"DATA3"}
            >>> source_dp = IterableWrapper(sorted(name_to_data.items()))
            >>> saver_dp = source_dp.save_to_disk(filepath_fn=filepath_fn, mode="wb")
            >>> res_file_paths = list(saver_dp)
            >>> res_file_paths
            ['./1.txt', './2.txt', './3.txt']
        """
    
    # Functional form of 'LengthSetterIterDataPipe'
    def set_length(self, length: int) -> IterDataPipe:
        r"""
        Set the length attribute of the DataPipe, which is returned by ``__len__`` (functional name: ``set_length``).
        This can be used after DataPipes whose final length cannot be known in advance (e.g. ``filter``). If you
        know the final length with certainty, you can manually set it, which can then be used by
        DataLoader or other DataPipes.
    
        Note:
            This DataPipe differs from :class:`.Header` in that this doesn't restrict the number of elements that
            can be yielded from the DataPipe; this is strictly used for setting an attribute so that it can be used later.
    
        Args:
            source_datapipe: a DataPipe
            length: the integer value that will be set as the length
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(range(10)).filter(lambda x: x < 5).set_length(3)
            >>> list(dp)  # Notice that the number of elements yielded is unchanged
            [0, 1, 2, 3, 4]
            >>> len(dp)
            3
            >>> header_dp = IterableWrapper(range(10)).filter(lambda x: x < 5).header(3)
            >>> list(header_dp)  # Use `.header()` if you want to limit the number of elements yielded
            [0, 1, 2]
            >>> len(header_dp)
            3
        """
    
    # Functional form of 'ShardExpanderIterDataPipe'
    def shard_expand(self) -> IterDataPipe:
        r"""
        Expands incoming shard strings into shards.
    
        Sharded data files are named using shell-like brace notation. For example,
        an ImageNet dataset sharded into 1200 shards and stored on a web server
        might be named `imagenet-{000000..001199}.tar`.
    
        Note that shard names can be expanded without any server transactions;
        this makes `shard_expand` reproducible and storage system independent
        (unlike :class `.FileLister` etc.).
    
        Args:
            source_datapipe: a DataPipe yielding a stream of  pairs
    
        Returns:
            a DataPipe yielding a stream of expanded pathnames.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper(["ds-{00..05}.tar"])
            >>> expand_dp = source_dp.shard_expand()
            >>> list(expand_dp)
            ['ds-00.tar', 'ds-01.tar', 'ds-02.tar', 'ds-03.tar', 'ds-04.tar', 'ds-05.tar']
            >>> source_dp = IterableWrapper(["imgs_{00..05}.tar", "labels_{00..05}.tar"])
            >>> expand_dp = source_dp.shard_expand()
            >>> list(expand_dp)
            ['imgs_00.tar', 'imgs_01.tar', 'imgs_02.tar', 'labels_00.tar', 'labels_01.tar', 'labels_02.tar']
        """
    
    # Functional form of 'ShardingRoundRobinDispatcherIterDataPipe'
    def sharding_round_robin_dispatch(self, sharding_group_filter: Optional[SHARDING_PRIORITIES] = None) -> IterDataPipe:
        r"""
        Wrapper that indicates the prior section of ``DataPipe`` graph is non-replicable and will be
        iterated in a separate, single dispatching process to distribute data to worker processes
        in a round-robin manner when multiprocessing is being used.
        (functional name: ``sharding_round_robin_dispatch``).
    
        Args:
            source_datapipe: Iterable DataPipe that will be sharded
            sharding_group_filter: Optional ``SHARDING_PRIORITIES`` value
    
        Note:
            - ``sharding_group_filter`` only accepts ``SHARDING_PRIORITIES.MULTIPROCESSING`` for now
            - When using distributed training, you can add a ``sharding_filter()`` prior to this DataPipe
              to distribute samples among worker nodes.
    
        Examples:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
            >>> dp = IterableWrapper(range(10))
            >>> # `.shuffle()` will be executed in a single dispatching processing, then the samples are distributed
            >>> # to worker processes
            >>> dp = dp.shuffle().sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
            >>> # `.map()` will be executed within each worker process
            >>> dp = dp.map(lambda x: x + 1)
            >>> # Distributed case: the 10 samples will be distributed among the nodes
            >>> dp = IterableWrapper(range(10)).sharding_filter()
            >>> # `.map()` will be executed in a single dispatching processing in each node
            >>> # You may apply further transformation after within each worker process
            >>> dp = dp.map(lambda x: x + 1).sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
        """
    
    # Functional form of 'ShuffledFlatMapperIterDataPipe'
    def shuffled_flatmap(self, fn: Optional[Callable] = None, input_col=None, buffer_size: int = 100) -> IterDataPipe:
        r"""
        Applies a function over each item from the source DataPipe,
        then collects the iterables returned in a buffer,
        then, at every iteration, chooses at random one of the iterables in the buffer
        and yields one item from this iterable (functional name: ``shuffled_flatmap``).
    
        When the buffer is full, the DataPipe will begin to yield elements from iterables within the buffer.
        New iterables will be added to the buffer once the existing ones run out of elements.
        Note:
            The output from ``fn`` must be an Iterable. Otherwise, an error will be raised.
            If ``fn`` is ``None``, source DataPipe will be just flattened vertically, provided that items can be unpacked.
    
        Args:
            datapipe: Source IterDataPipe
            fn: the function to be applied to each element in the DataPipe, the output must be a Sequence
            input_col: Index or indices of data which ``fn`` is applied, such as:
    
                - ``None`` as default to apply ``fn`` to the data directly.
                - Integer(s) is/are used for list/tuple.
                - Key(s) is/are used for dict.
            buffer_size: the max number of iterables this DataPipe can hold at a time (default to ``100``)
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper([[1, 2, 3, 4], 'abcd', 'ABCD'])
            >>> shuffled_flatmapped_dp = source_dp.shuffled_flatmap(buffer_size=2)
            >>> list(shuffled_flatmapped_dp)
            ['a', 'b', 'c', 1, 'd', 'A', 'B', 'C', 2, 'D', 3, 4]
            >>>
            >>> # To shuffle all the elements, you can combine `shuffled_flatmap` with `in_batch_shuffle` like this:
            >>> fully_shuffled_flatmapped_dp = source_dp.in_batch_shuffle()
            >>> fully_shuffled_flatmapped_dp = fully_shuffled_flatmapped_dp.shuffled_flatmap()
            >>> list(fully_shuffled_flatmapped_dp)
            ['b', 3, 'c', 'd', 'C', 'A', 'a', 2, 'B', 'D', 4, 1]
        """
    
    # Functional form of 'SliceIterDataPipe'
    def slice(self, index: Union[int, List[Hashable]], stop: Optional[int] = None, step: Optional[int] = None) -> IterDataPipe:
        r"""
        returns a slice of elements in input DataPipe via start/stop/step or indices (functional name: ``slice``).
    
        Args:
            datapipe: IterDataPipe with iterable elements
            index: a single start index for the slice or a list of indices to be returned instead of a start/stop slice
    
                - Integer(s) is/are used for list/tuple.
                - Key(s) is/are used for dict.
    
    
            stop: the slice stop. ignored if index is a list or if element is a dict
            step: step to be taken from start to stop. ignored if index is a list or if element is a dict
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper([(0, 10, 100), (1, 11, 111), (2, 12, 122), (3, 13, 133), (4, 14, 144)])
            >>> slice_dp = dp.slice(0, 2)
            >>> list(slice_dp)
            [(0, 10), (1, 11), (2, 12), (3, 13), (4, 14)]
        """
    
    # Functional form of 'ThreadPoolMapperIterDataPipe'
    def threadpool_map(self, fn: Callable, input_col=None, output_col=None, scheduled_tasks: int = 128, max_workers: Optional[int] = None, **threadpool_kwargs) -> IterDataPipe:
        r"""
        Applies a function over each item from the source DataPipe concurrently
        using ``ThreadPoolExecutor`` (functional name: ``threadpool_map``).
        The function can be any regular Python function or partial object. Lambda
        function is not recommended as it is not supported by pickle.
    
        Args:
            source_datapipe: Source IterDataPipe
            fn: Function being applied over each item
            input_col: Index or indices of data which ``fn`` is applied, such as:
    
                - ``None`` as default to apply ``fn`` to the data directly.
                - Integer(s) is used for list/tuple.
                - Key(s) is used for dict.
    
            output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
                only when ``input_col`` is not ``None``
    
                - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
                  multiple indices, the left-most one is used, and other indices will be removed.
                - Integer is used for list/tuple. ``-1`` represents to append result at the end.
                - Key is used for dict. New key is acceptable.
    
            scheduled_tasks: How many tasks will be scheduled at any given time (Default value: 128)
            max_workers: Maximum number of threads to execute function calls
            **threadpool_kwargs: additional arguments to be given to the ``ThreadPoolExecutor``
    
        Note:
             For more information about ``max_workers`` and additional arguments for the ``ThreadPoolExecutor``
             please refer to: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
    
        Note:
            For optimal use of all threads, ``scheduled_tasks`` > ``max_workers`` is strongly recommended. The higher the
            variance of the time needed to finish execution of the given ``fn`` is, the higher the value
            of ``scheduled_tasks`` needs to be to avoid threads sitting idle while waiting
            for the next result (as results are returned in correct order).
    
            However, too high value of ``scheduled_tasks`` might lead to long waiting period until the first element is yielded
            as ``next`` is called ``scheduled_tasks`` many times on ``source_datapipe`` before yielding.
    
            We encourage you to try out different values of ``max_workers`` and ``scheduled_tasks``
            in search for optimal values for your use-case.
    
        Example:
    
        .. testsetup::
    
            from torchdata.datapipes.iter import IterableWrapper
            import requests
            import time
            from unittest.mock import MagicMock
    
            requests.get = MagicMock()
            urls = []
    
        .. testcode::
    
            # fetching html from remote
            def fetch_html(url: str, **kwargs):
                r = requests.get(url, **kwargs)
                r.raise_for_status()
                return r.content
            dp = IterableWrapper(urls)
            dp = dp.threadpool_map(fetch_html,max_workers=16)
    
        .. testcode::
    
            def mul_ten(x):
                time.sleep(0.1)
                return x * 10
    
            dp = IterableWrapper([(i, i) for i in range(50)])
            dp = dp.threadpool_map(mul_ten, input_col=1)
            print(list(dp))
    
        .. testoutput::
    
            [(0, 0), (1, 10), (2, 20), (3, 30), ...]
    
        .. testcode::
    
            dp = IterableWrapper([(i, i) for i in range(50)])
            dp = dp.threadpool_map(mul_ten, input_col=1, output_col=-1)
            print(list(dp))
    
        .. testoutput::
    
            [(0, 0, 0), (1, 1, 10), (2, 2, 20), (3, 3, 30), ...]
    
        """
    
    # Functional form of 'IterToMapConverterMapDataPipe'
    def to_map_datapipe(self, key_value_fn: Optional[Callable] = None) -> MapDataPipe:
        r"""
        Lazily load data from ``IterDataPipe`` to construct a ``MapDataPipe`` with
        the key-value pair generated by ``key_value_fn`` (functional name: ``to_map_datapipe``).
        If ``key_value_fn`` is not given, each data from the source IterDataPipe must itself be an iterable
        with exactly two objects. The first object of each item becomes a key in
        the new dictionary, and the second object the corresponding value.
    
        For the opposite converter, use :class:`.MapToIterConverter`.
    
        Args:
            datapipe: Source IterDataPipe
            key_value_fn: Function being applied over each data to generate key-value pair
    
        Note:
            If a key being added is already present, the corresponding value
            will be replaced by the new value.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper([(i, i) for i in range(10)])
            >>> map_dp = source_dp.to_map_datapipe()
            >>> list(map_dp)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> source_dp2 = IterableWrapper([('a', 1), ('b', 2), ('c', 1)])
            >>> map_dp2 = source_dp2.to_map_datapipe()
            >>> map_dp2['a']
            1
            >>> def row_to_tuple(row):
            >>>     label = row[0]
            >>>     data = row[1:]
            >>>     return label, data
            >>> source_dp3 = IterableWrapper([('a', 1, 1, 1, 1, 1, 1), ('b', 2, 2, 2, 2, 2, 2), ('c', 3, 3, 3, 3, 3, 3)])
            >>> map_dp3 = source_dp3.to_map_datapipe(key_value_fn=row_to_tuple)
            >>> map_dp3['a']
            (1, 1, 1, 1, 1, 1)
        """
    
    # Functional form of 'UnZipperIterDataPipe'
    def unzip(self, sequence_length: int, buffer_size: int = 1000, columns_to_skip: Optional[Sequence[int]] = None) -> List[IterDataPipe]:
        r"""
        Takes in a DataPipe of Sequences, unpacks each Sequence, and return the elements in separate DataPipes
        based on their position in the Sequence (functional name: ``unzip``). The number of instances produced equals to
        the sequence length minus the number of columns to skip.
    
        Note:
            Each sequence within the DataPipe should have the same length, specified by
            the input argument `sequence_length`.
    
        Args:
            source_datapipe: Iterable DataPipe with sequences of data
            sequence_length: Length of the sequence within the source_datapipe. All elements should have the same length.
            buffer_size: this restricts how far ahead the leading child DataPipe can read relative
                to the slowest child DataPipe. Use -1 for the unlimited buffer.
            columns_to_skip: optional indices of columns that the DataPipe should skip (each index should be
                an integer from 0 to sequence_length - 1)
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper([(i, i + 10, i + 20) for i in range(3)])
            >>> dp1, dp2, dp3 = source_dp.unzip(sequence_length=3)
            >>> list(dp1)
            [0, 1, 2]
            >>> list(dp2)
            [10, 11, 12]
            >>> list(dp3)
            [20, 21, 22]
        """
    
    # Functional form of 'WebDatasetIterDataPipe'
    def webdataset(self) -> IterDataPipe:
        r"""
        Iterable DataPipe that accepts stream of (path, data) tuples, usually,
        representing the pathnames and files of a tar archive (functional name:
        ``webdataset``). This aggregates consecutive items with the same basename
        into a single dictionary, using the extensions as keys (WebDataset file
        convention). Any text after the first "." in the filename is used as
        a key/extension.
    
        File names that do not have an extension are ignored.
    
        Args:
            source_datapipe: a DataPipe yielding a stream of (path, data) pairs
    
        Returns:
            a DataPipe yielding a stream of dictionaries
    
        Examples:
            >>> from torchdata.datapipes.iter import FileLister, FileOpener
            >>>
            >>> def decode(item):
            >>>     key, value = item
            >>>     if key.endswith(".txt"):
            >>>         return key, value.read().decode("utf-8")
            >>>     if key.endswith(".bin"):
            >>>         return key, value.read().decode("utf-8")
            >>>
            >>> datapipe1 = FileLister("test/_fakedata", "wds*.tar")
            >>> datapipe2 = FileOpener(datapipe1, mode="b")
            >>> dataset = datapipe2.load_from_tar().map(decode).webdataset()
            >>> for obj in dataset:
            >>>     print(obj)
        """
    
    # Functional form of 'ZipperLongestIterDataPipe'
    def zip_longest(self, *datapipes: IterDataPipe, fill_value: Any = None) -> IterDataPipe:
        r"""
        Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip_longest``).
        The output is stopped until all input DataPipes are exhausted. If any input DataPipe is exhausted,
        missing values are filled-in with `fill_value` (default value is None).
    
        Args:
            *datapipes: Iterable DataPipes being aggregated
            *fill_value: Value that user input to fill in the missing values from DataPipe. Default value is None.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp1, dp2, dp3 = IterableWrapper(range(3)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
            >>> list(dp1.zip_longest(dp2, dp3))
            [(0, 10, 20), (1, 11, 21), (2, 12, 22), (None, 13, 23), (None, 14, 24)]
            >>> list(dp1.zip_longest(dp2, dp3, -1))
            [(0, 10, 20), (1, 11, 21), (2, 12, 22), (-1, 13, 23), (-1, 14, 24)]
        """
    
    # Functional form of 'IterKeyZipperIterDataPipe'
    def zip_with_iter(self, ref_datapipe: IterDataPipe, key_fn: Callable, ref_key_fn: Optional[Callable] = None, keep_key: bool = False, buffer_size: int = 10000, merge_fn: Optional[Callable] = None) -> IterDataPipe:
        r"""
        Zips two IterDataPipes together based on the matching key (functional name: ``zip_with_iter``). The keys
        are computed by ``key_fn`` and ``ref_key_fn`` for the two IterDataPipes, respectively. When there isn't a match
        between the elements of the two IterDataPipes, the element from ``ref_datapipe`` is stored in a buffer. Then, the
        next element from ``ref_datapipe`` is tried. After a match is found, the ``merge_fn`` determines how they will
        be combined and returned (a tuple is generated by default).
    
        Args:
            source_datapipe: IterKeyZipper will yield data based on the order of this IterDataPipe
            ref_datapipe: Reference IterDataPipe from which IterKeyZipper will find items
                with matching key for ``source_datapipe``
            key_fn: Callable function that will compute keys using elements from ``source_datapipe``
            ref_key_fn: Callable function that will compute keys using elements from ``ref_datapipe``
                If it's not specified, the ``key_fn`` will also be applied to elements from ``ref_datapipe``
            keep_key: Option to yield the matching key along with the items in a tuple,
                resulting in `(key, merge_fn(item1, item2))`.
            buffer_size: The size of buffer used to hold key-data pairs from reference DataPipe until a match is found.
                If it's specified as ``None``, the buffer size is set as infinite.
            merge_fn: Function that combines the item from ``source_datapipe`` and the item from ``ref_datapipe``,
                by default a tuple is created
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> from operator import itemgetter
            >>> def merge_fn(t1, t2):
            >>>     return t1[1] + t2[1]
            >>> dp1 = IterableWrapper([('a', 100), ('b', 200), ('c', 300)])
            >>> dp2 = IterableWrapper([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
            >>> res_dp = dp1.zip_with_iter(dp2, key_fn=itemgetter(0),
            >>>                            ref_key_fn=itemgetter(0), keep_key=True, merge_fn=merge_fn)
            >>> list(res_dp)
            [('a', 101), ('b', 202), ('c', 303)]
        """
    
    # Functional form of 'MapKeyZipperIterDataPipe'
    def zip_with_map(self, map_datapipe: MapDataPipe, key_fn: Callable, merge_fn: Optional[Callable] = None, keep_key: bool = False) -> IterDataPipe:
        r"""
        Joins the items from the source IterDataPipe with items from a MapDataPipe (functional name: ``zip_with_map``).
        The matching is done by the provided ``key_fn``, which maps an item from ``source_iterdatapipe`` to
        a key that should exist in the ``map_datapipe``. The return value is created by the ``merge_fn``, which returns
        a tuple of the two items by default.
    
        Args:
            source_iterdatapipe: IterDataPipe from which items are yield and will be combined with an item
                from ``map_datapipe``
            map_datapipe: MapDataPipe that takes a key from ``key_fn``, and returns an item
            key_fn: Function that maps each item from ``source_iterdatapipe`` to a key that exists in ``map_datapipe``
            keep_key: Option to yield the matching key along with the items in a tuple,
                resulting in ``(key, merge_fn(item1, item2))``.
            merge_fn: Function that combines the item from ``source_iterdatapipe`` and the matching item
                from ``map_datapipe``, by default a tuple is created
    
        Example:
    
        .. testsetup::
    
            from operator import itemgetter
    
        .. testcode::
    
            from torchdata.datapipes.iter import IterableWrapper
            from torchdata.datapipes.map import SequenceWrapper
    
    
            def merge_fn(tuple_from_iter, value_from_map):
                return tuple_from_iter[0], tuple_from_iter[1] + value_from_map
    
    
            dp1 = IterableWrapper([('a', 1), ('b', 2), ('c', 3)])
            mapdp = SequenceWrapper({'a': 100, 'b': 200, 'c': 300, 'd': 400})
            res_dp = dp1.zip_with_map(map_datapipe=mapdp, key_fn=itemgetter(0), merge_fn=merge_fn)
    
            print(list(res_dp))
    
        Output:
    
        .. testoutput::
    
            [('a', 101), ('b', 202), ('c', 303)]
    
        """
    
