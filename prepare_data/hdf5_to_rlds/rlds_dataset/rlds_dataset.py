from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
# Dodanie katalogu zawierającego ten plik do sys.path na potrzeby importów
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
from conversion_utils import MultiThreadedDatasetBuilder

def batch_pose17_to_pose23(actions):
    """
    Konwersja wsadu pozy 17D do 23D.
    akcje: (T, 17)
    wyjście: (T, 23)
    """
    actions = np.asarray(actions, dtype=float)
    T = actions.shape[0]

    # Podział
    L_xyz = actions[:, 0:3]
    L_rpy = actions[:, 3:6]
    R_xyz = actions[:, 6:9]
    R_rpy = actions[:, 9:12]
    waist5 = actions[:, 12:17]

    # Konwersja RPY→6D (wektoryzowana)
    def rpy_to_6d_batch(rpy):
        roll, pitch, yaw = rpy[:,0], rpy[:,1], rpy[:,2]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Kolumna 1
        col1 = np.stack([cy*cp,
                         sy*cp,
                         -sp], axis=1)

        # Kolumna 2
        col2 = np.stack([
            cy*sp*sr - sy*cr,
            sy*sp*sr + cy*cr,
            cp*sr
        ], axis=1)

        return np.concatenate([col1, col2], axis=1)

    L_6d = rpy_to_6d_batch(L_rpy)
    R_6d = rpy_to_6d_batch(R_rpy)

    # Wyjście
    return np.concatenate([L_xyz, L_6d, R_xyz, R_6d, waist5], axis=1)

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Generuje epizody dla listy ścieżek do danych."""
    # poniższa linia musi znajdować się *wewnątrz* generate_examples,
    # aby każdy worker tworzył własny model –
    # stworzenie jednego wspólnego modelu poza tą funkcją spowodowałoby zakleszczenie

    def _parse_example(episode_path):
        # Wczytanie surowych danych
        with h5py.File(episode_path, "r") as F:

            actions = F['action'][:]
            states = F['observations']["qpos"][:]
            if "ee_qpos" in F['observations']:
                ee_states = F['observations']["ee_qpos"][:]
                ee_states_6d = batch_pose17_to_pose23(ee_states)
            if "ee_action" in F:
                ee_actions = F["ee_action"][:]
                ee_actions_6d = batch_pose17_to_pose23(ee_actions)
            images_left_top = F['observations']["images"]["cam_left_high"][:]  
            images_right_top = F['observations']["images"]["cam_right_high"][:]  
            images_left_wrist = F['observations']["images"]["cam_left_wrist"][:]  
            images_right_wrist = F['observations']["images"]["cam_right_wrist"][:]  
            
            language_raw_data = F['language_raw']
            if language_raw_data.shape == (): 
                language_instruction = str(language_raw_data[()])
            else:  
                language_instruction = str(language_raw_data[0])

        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'image_left_top': images_left_top[i],
                    'image_right_top': images_right_top[i],
                    'image_left_wrist': images_left_wrist[i],
                    'image_right_wrist': images_right_wrist[i],
                    'state': np.asarray(states[i], np.float32),
                    'ee_state': np.asarray(ee_states[i], np.float32),
                    'ee_state_6d': np.asarray(ee_states_6d[i], np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'ee_action': np.asarray(ee_actions[i], dtype=np.float32),
                'ee_action_6d': np.asarray(ee_actions_6d[i], dtype=np.float32),
                'discount': 1.0,
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': language_instruction,
            })

        # Tworzenie próbki wyjściowej
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # Aby pominąć przykład z dowolnego powodu, wystarczy zwrócić None
        return episode_path, sample

    # Dla małych zbiorów danych używamy parsowania jednowątkowego
    for sample in paths:
        ret = _parse_example(sample)
        yield ret


class rlds_dataset(MultiThreadedDatasetBuilder):
    """Konstruktor zbioru danych RLDS."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Pierwsze wydanie.',
    }
    N_WORKERS = 8            # liczba równoległych workerów do konwersji danych
    MAX_PATHS_IN_MEMORY = 8  # liczba ścieżek konwertowanych i przechowywanych w pamięci przed zapisem na dysk
                               # -> im wyższa wartość, tym szybsza / bardziej równoległa konwersja, dostosuj do dostępnej pamięci RAM
                               # uwaga: jedna ścieżka może generować wiele epizodów – dostosuj odpowiednio
    PARSE_FCN = _generate_examples      # odniesienie do funkcji parsowania ze ścieżek pliku do epizodów RLDS

    def _info(self) -> tfds.core.DatasetInfo:
        """Metadane zbioru danych (strona główna, cytowanie itp.)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_left_top': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Obserwacja RGB z lewej górnej kamery.',
                        ),
                        'image_right_top': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Obserwacja RGB z prawej górnej kamery.',
                        ),
                        'image_left_wrist': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Obserwacja RGB z lewej kamery na nadgarstku.',
                        ),
                        'image_right_wrist': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Obserwacja RGB z lewej kamery na nadgarstku.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(19,),
                            dtype=np.float32,
                            doc='Stan stawów robota (7D lewe ramię + 7D prawe ramię + 1D lewy chwytak + 1D prawy chwytak + 3D talia).',
                        ),
                        'ee_state': tfds.features.Tensor(
                            shape=(17,),
                            dtype=np.float32,
                            doc='Stan end-effektora robota (6D EEF lewe ramię + 6D EEF prawe ramię + 1D lewy chwytak + 1D prawy chwytak + 3D talia).',
                        ),
                        'ee_state_6d': tfds.features.Tensor(
                            shape=(23,),
                            dtype=np.float32,
                            doc='Stan end-effektora robota (9D EEF lewe ramię + 9D EEF prawe ramię + 1D lewy chwytak + 1D prawy chwytak + 3D talia).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(19,),
                        dtype=np.float32,
                        doc='Akcja stawów robota (7D lewe ramię + 7D prawe ramię + 1D lewy chwytak + 1D prawy chwytak + 3D talia).',
                    ),
                    'ee_action': tfds.features.Tensor(
                        shape=(17,),
                        dtype=np.float32,
                        doc='Akcja end-effektora robota (6D EEF lewe ramię + 6D EEF prawe ramię + 1D lewy chwytak + 1D prawy chwytak + 3D talia).',
                    ),
                    'ee_action_6d': tfds.features.Tensor(
                        shape=(23,),
                        dtype=np.float32,
                        doc='Akcja end-effektora robota (9D EEF lewe ramię + 9D EEF prawe ramię + 1D lewy chwytak + 1D prawy chwytak + 3D talia).',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Współczynnik dyskontowania, jeśli podany; domyślnie 1.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='Wartość True dla pierwszego kroku epizodu.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='Wartość True dla ostatniego kroku epizodu.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='Wartość True dla ostatniego kroku epizodu, jeśli jest krokiem terminalnym; True dla demonstracji.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Instrukcja językowa.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({ 
                    'file_path': tfds.features.Text(
                        doc='Ścieżka do oryginalnego pliku danych.'
                    ),
                }),
            }))
        
        
        
    def _split_paths(self):
        """Definiuje ścieżki do plików dla podziałów danych."""
        return {
            'train': glob.glob("/path/to/save/the/converted/data/directory/*.hdf5"),
        }