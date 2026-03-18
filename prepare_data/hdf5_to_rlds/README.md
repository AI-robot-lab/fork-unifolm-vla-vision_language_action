To jest adapter zbioru danych, który konwertuje dane w formacie LeRobot do formatu RLDS, umożliwiając wykorzystanie zbioru danych LeRobot do dostrajania modelu openVLA (https://github.com/moojink/openvla-oft).
Kod inspirowany jest projektem https://github.com/moojink/rlds_dataset_builder/tree/main.
Oryginalny kod wymaga jednak Pythona w wersji <=3.9, co jest niezgodne z LeRobot (Python >= 3.10).

Kroki:
(1) Uruchom `python convert_lerobot_to_hdf5.py` w środowisku conda z zainstalowanym LeRobot.


(2) Przełącz się na środowisko rlds_dataset (zob. https://github.com/moojink/rlds_dataset_builder/tree/main) i uruchom `tfds build --overwrite`.

(3) Aby sprawdzić, czy konwersja przebiegła pomyślnie, uruchom `python3 visualize_dataset.py <nazwa_twojego_zbioru_danych>` z repozytorium rlds_dataset_builder.
