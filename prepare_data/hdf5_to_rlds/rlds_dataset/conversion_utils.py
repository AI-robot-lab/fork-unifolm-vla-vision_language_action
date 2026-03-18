from typing import Tuple, Any, Dict, Union, Callable, Iterable
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import itertools
from multiprocessing import Pool
from functools import partial
from tensorflow_datasets.core import download
from tensorflow_datasets.core import split_builder as split_builder_lib
from tensorflow_datasets.core import naming
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import utils
from tensorflow_datasets.core import writer as writer_lib
from tensorflow_datasets.core import example_serializer
from tensorflow_datasets.core import dataset_builder
from tensorflow_datasets.core import file_adapters

Key = Union[str, int]
# The nested example dict passed to `features.encode_example`
Example = Dict[str, Any]
KeyExample = Tuple[Key, Example]


class MultiThreadedDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """Konstruktor zbioru danych z wielowątkowym przetwarzaniem."""
    N_WORKERS = 10                  # liczba równoległych workerów do konwersji danych
    MAX_PATHS_IN_MEMORY = 100       # liczba ścieżek konwertowanych i przechowywanych w pamięci przed zapisem na dysk
                                    # -> im wyższa wartość, tym szybsza / bardziej równoległa konwersja, dostosuj do dostępnej pamięci RAM
                                    # uwaga: jedna ścieżka może generować wiele epizodów – dostosuj odpowiednio
    PARSE_FCN = None                # należy uzupełnić funkcją parsowania ścieżki do nagranego epizodu

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Definiuje podziały danych."""
        split_paths = self._split_paths()
        return {split: type(self).PARSE_FCN(paths=split_paths[split]) for split in split_paths}

    def _generate_examples(self):
        pass  # this is implemented in global method to enable multiprocessing

    def _download_and_prepare(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
            self,
            dl_manager: download.DownloadManager,
            download_config: download.DownloadConfig,
    ) -> None:
        """Generuje wszystkie podziały danych i zwraca obliczone informacje o podziale."""
        assert self.PARSE_FCN is not None       # konieczne nadpisanie funkcji parsowania
        split_builder = ParallelSplitBuilder(
            split_dict=self.info.splits,
            features=self.info.features,
            dataset_size=self.info.dataset_size,
            max_examples_per_split=download_config.max_examples_per_split,
            beam_options=download_config.beam_options,
            beam_runner=download_config.beam_runner,
            file_format=self.info.file_format,
            shard_config=download_config.get_shard_config(),
            split_paths=self._split_paths(),
            parse_function=type(self).PARSE_FCN,
            n_workers=self.N_WORKERS,
            max_paths_in_memory=self.MAX_PATHS_IN_MEMORY,
        )
        split_generators = self._split_generators(dl_manager)
        split_generators = split_builder.normalize_legacy_split_generators(
            split_generators=split_generators,
            generator_fn=self._generate_examples,
            is_beam=False,
        )
        dataset_builder._check_split_names(split_generators.keys())

        # Rozpoczęcie generowania danych dla wszystkich podziałów
        path_suffix = file_adapters.ADAPTER_FOR_FORMAT[
            self.info.file_format
        ].FILE_SUFFIX

        split_info_futures = []
        for split_name, generator in utils.tqdm(
                split_generators.items(),
                desc="Generowanie podziałów...",
                unit=" podziały",
                leave=False,
        ):
            filename_template = naming.ShardedFileTemplate(
                split=split_name,
                dataset_name=self.name,
                data_dir=self.data_path,
                filetype_suffix=path_suffix,
            )
            future = split_builder.submit_split_generation(
                split_name=split_name,
                generator=generator,
                filename_template=filename_template,
                disable_shuffling=self.info.disable_shuffling,
            )
            split_info_futures.append(future)

        # Finalizacja podziałów (po zakończeniu apache beam, jeśli był używany)
        split_infos = [future.result() for future in split_info_futures]

        # Aktualizacja obiektu info o informacje o podziałach.
        split_dict = splits_lib.SplitDict(split_infos)
        self.info.set_splits(split_dict)


class _SplitInfoFuture:
    """Future zawierający wynik `tfds.core.SplitInfo`."""

    def __init__(self, callback: Callable[[], splits_lib.SplitInfo]):
        self._callback = callback

    def result(self) -> splits_lib.SplitInfo:
        return self._callback()


def parse_examples_from_generator(paths, fcn, split_name, total_num_examples, features, serializer):
    generator = fcn(paths)
    outputs = []
    for sample in utils.tqdm(
            generator,
            desc=f'Generowanie przykładów dla {split_name}...',
            unit=' przykłady',
            total=total_num_examples,
            leave=False,
            mininterval=1.0,
    ):
        if sample is None: continue
        key, example = sample
        try:
            example = features.encode_example(example)
        except Exception as e:  # pylint: disable=broad-except
            utils.reraise(e, prefix=f'Nie udało się zakodować przykładu:\n{example}\n')
        outputs.append((key, serializer.serialize_example(example)))
    return outputs


class ParallelSplitBuilder(split_builder_lib.SplitBuilder):
    def __init__(self, *args, split_paths, parse_function, n_workers, max_paths_in_memory, **kwargs):
        super().__init__(*args, **kwargs)
        self._split_paths = split_paths
        self._parse_function = parse_function
        self._n_workers = n_workers
        self._max_paths_in_memory = max_paths_in_memory

    def _build_from_generator(
            self,
            split_name: str,
            generator: Iterable[KeyExample],
            filename_template: naming.ShardedFileTemplate,
            disable_shuffling: bool,
    ) -> _SplitInfoFuture:
        """Generator podziału dla generatorów przykładów.

        Args:
          split_name: str,
          generator: Iterable[KeyExample],
          filename_template: Szablon do formatowania nazwy pliku dla fragmentu (shard).
          disable_shuffling: Określa, czy mieszać przykłady.

        Returns:
          future: Future zawierający `tfds.core.SplitInfo`.
        """
        total_num_examples = None
        serialized_info = self._features.get_serialized_info()
        writer = writer_lib.Writer(
            serializer=example_serializer.ExampleSerializer(serialized_info),
            filename_template=filename_template,
            hash_salt=split_name,
            disable_shuffling=disable_shuffling,
            file_format=self._file_format,
            shard_config=self._shard_config,
        )

        del generator  # zamiast tego używaj równoległych generatorów
        paths = self._split_paths[split_name]
        path_lists = chunk_max(paths, self._n_workers, self._max_paths_in_memory)  # generuj N list plików
        print(f"Generowanie z użyciem {self._n_workers} workerów!")
        pool = Pool(processes=self._n_workers)
        for i, paths in enumerate(path_lists):
            print(f"Przetwarzanie fragmentu {i + 1} z {len(path_lists)}.")
            results = pool.map(
                partial(
                    parse_examples_from_generator,
                    fcn=self._parse_function,
                    split_name=split_name,
                    total_num_examples=total_num_examples,
                    serializer=writer._serializer,
                    features=self._features
                ),
                paths
            )
            # zapis wyników do shufflera --> w razie potrzeby zostanie automatycznie przeniesiony na dysk
            print("Zapisywanie wyników konwersji...")
            for result in itertools.chain(*results):
                key, serialized_example = result
                writer._shuffler.add(key, serialized_example)
                writer._num_examples += 1
        pool.close()

        print("Finalizacja konwersji podziału...")
        shard_lengths, total_size = writer.finalize()

        split_info = splits_lib.SplitInfo(
            name=split_name,
            shard_lengths=shard_lengths,
            num_bytes=total_size,
            filename_template=filename_template,
        )
        return _SplitInfoFuture(lambda: split_info)


def dictlist2listdict(DL):
    " Konwertuje słownik list na listę słowników "
    return [dict(zip(DL, t)) for t in zip(*DL.values())]

def chunks(l, n):
    """Generuje n kolejnych fragmentów listy l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si:si + (d + 1 if i < r else d)]

def chunk_max(l, n, max_chunk_sum):
    out = []
    for _ in range(int(np.ceil(len(l) / max_chunk_sum))):
        out.append(list(chunks(l[:max_chunk_sum], n)))
        l = l[max_chunk_sum:]
    return out