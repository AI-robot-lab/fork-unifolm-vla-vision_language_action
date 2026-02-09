# UnifoLM-VLA-0: Framework Vision-Language-Action (VLA) z rodziny UnifoLM
<p style="font-size: 1.2em;">
  <a href="https://unigen-x.github.io/unifolm-vla.github.io"><strong>Strona Projektu</strong></a> | 
  <a href="https://huggingface.co/unitreerobotics/models"><strong>Modele</strong></a> |
  <a href="https://huggingface.co/unitreerobotics/datasets"><strong>Zbiory Danych</strong></a> 
</p>

<div align="center">
  <p align="right">
    <a href="README.md"> 🌎English </a> | <a href="README_cn.md"> 🇨🇳中文 </a> | <span> 🇵🇱Polski </span>
  </p>
</div>

## 📚 Wprowadzenie dla Studentów Politechniki Rzeszowskiej

**UnifoLM-VLA-0** to zaawansowany model Vision–Language–Action (VLA) z serii UnifoLM, zaprojektowany specjalnie do manipulacji obiektami przez roboty humanoidalne ogólnego przeznaczenia. Model ten wykracza poza ograniczenia konwencjonalnych modeli Vision–Language (VLM) poprzez dodanie możliwości fizycznej interakcji ze światem.

### Co to jest VLA (Vision-Language-Action)?

**Vision-Language-Action** to architektura łącząca trzy kluczowe komponenty:
- **Vision (Wizja)**: Przetwarzanie obrazów z kamer robota w celu zrozumienia sceny
- **Language (Język)**: Rozumienie instrukcji tekstowych podanych przez użytkownika
- **Action (Akcja)**: Generowanie konkretnych ruchów i działań robota

Dzięki ciągłemu pre-trenowaniu na danych manipulacyjnych robotów, model ewoluuje od prostego "rozumienia wizji i języka" do "ucieleśnionego mózgu" wyposażonego w fizyczny zmysł praktyczny.

## 🤖 Kontekst: Robot Unitree G1 EDU-U6

Ten framework został opracowany i przetestowany głównie na robocie humanoidalnym **Unitree G1**, w tym wersji edukacyjnej **G1 EDU-U6**. Robot ten:

- Jest humanoidalnym robotem ogólnego przeznaczenia z dwoma ramionami manipulacyjnymi
- Posiada zaawansowane systemy wizyjne (kamery RGB)
- Może wykonywać złożone zadania manipulacyjne wymagające koordynacji wzrokowo-ruchowej
- Jest używany w badaniach nad uczeniem robotów poprzez demonstracje

### Praktyczne zastosowania w projekcie z robotem G1:

**UnifoLM-VLA-0** umożliwia robotowi wykonywanie 12 kategorii złożonych zadań manipulacyjnych przy użyciu pojedynczej polityki (single policy):

1. **G1_Stack_Block** - Układanie klocków
2. **G1_Bag_Insert** - Wkładanie przedmiotów do torby
3. **G1_Erase_Board** - Ścieranie tablicy
4. **G1_Clean_Table** - Czyszczenie stołu
5. **G1_Pack_PencilBox** - Pakowanie piórnika
6. **G1_Pour_Medicine** - Nalewanie leków
7. **G1_Pack_PingPong** - Pakowanie piłeczek pingpongowych
8. **G1_Prepare_Fruit** - Przygotowywanie owoców
9. **G1_Organize_Tools** - Organizowanie narzędzi
10. **G1_Fold_Towel** - Składanie ręcznika
11. **G1_Wipe_Table** - Wycieranie stołu
12. **G1_DualRobot_Clean_Table** - Czyszczenie stołu przez dwa roboty współpracujące

<table width="100%">
  <tr>
    <th width="50%">Wzmocnienie Semantyki Przestrzennej</th>
    <th width="50%">Generalizacja Manipulacji</th>
  </tr>
  <tr>
    <td valign="top">
      Aby sprostać wymaganiom dotyczącym rozumienia instrukcji i przestrzeni w zadaniach manipulacyjnych, model głęboko integruje instrukcje tekstowe ze szczegółami przestrzennymi 2D/3D poprzez kontynuowane pre-trenowanie, <strong>znacząco wzmacniając swoje zdolności percepcji przestrzennej i rozumienia geometrycznego.</strong>
    </td>
    <td valign="top">
      Wykorzystując dane pełnej predykcji dynamiki, model osiąga silną generalizację w różnorodnych zadaniach manipulacyjnych. W walidacji na prawdziwym robocie <strong>może wykonywać 12 kategorii złożonych zadań manipulacyjnych wysokiej jakości używając tylko jednej polityki.</strong>
    </td>
  </tr>
</table>

<div align="center">
  <img 
    src="assets/gif/UnifoLM-VLA-0.gif"
    style="width:100%; max-width:1000px; height:auto;"
  />
</div>

## 🔥 Aktualności
* 29 stycznia 2026: 🚀 Opublikowaliśmy kod treningowy i inferencyjny wraz z wagami modelu [**UnifoLM-VLA-0**](https://huggingface.co/collections/unitreerobotics/unifolm-wma-0-68ca23027310c0ca0f34959c).

## 📑 Plan Open-Source
- [x] Kod treningowy (Training)
- [x] Kod inferencyjny (Inference)
- [x] Punkty kontrolne modelu (Checkpoints)

## ⚙️ Instalacja

### Wymagania systemowe
Ten projekt został zbudowany na **CUDA 12.4**. Zdecydowanie zalecamy używanie tej samej wersji, aby zapewnić kompatybilność.

### Kroki instalacji:

```bash
# Krok 1: Utworzenie środowiska conda z Pythonem 3.10.18
conda create -n unifolm-vla python==3.10.18
conda activate unifolm-vla

# Krok 2: Sklonowanie repozytorium (jeśli jeszcze nie zostało pobrane)
git clone https://github.com/unitreerobotics/unifolm-vla.git

# Krok 3: Przejście do katalogu projektu
cd unifolm-vla

# Krok 4: Instalacja LeRobot (biblioteka do uczenia robotów)
# Używamy konkretnej wersji z commita 0878c68
pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@0878c68"

# Krok 5: Instalacja pakietu unifolm-vla w trybie edytowalnym
# Flaga -e pozwala na modyfikację kodu bez reinstalacji
pip install -e .

# Krok 6: Instalacja FlashAttention2 (przyspiesza mechanizm attention w transformerach)
pip install "flash-attn==2.5.6" --no-build-isolation
```

### Wyjaśnienie kroków instalacji:

1. **Środowisko conda**: Izoluje zależności projektu od innych projektów Pythona
2. **LeRobot**: Framework do uczenia robotów, dostarcza narzędzia do obsługi danych robotycznych
3. **Tryb edytowalny (-e)**: Pozwala modyfikować kod źródłowy bez konieczności reinstalacji pakietu
4. **FlashAttention2**: Zoptymalizowana implementacja mechanizmu attention, znacząco przyspiesza trenowanie i inferencję

## 🧰 Punkty Kontrolne Modelu (Checkpoints)

Dostępne są trzy główne modele, każdy wytrenowany na różnych danych:

| Model | Opis | Link|
|---------|-------|------|
|`UnifoLM-VLM-Base` | Model bazowy - fine-tunowany na ogólnych danych VQA (Visual Question Answering) obrazowo-tekstowych oraz open-source'owych zbiorach danych robotycznych. **Punkt startowy** dla dalszego treningu. | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLM-Base)|
|`UnifoLM-VLA-Base` | Model VLA - fine-tunowany na zbiorach danych [Unitree opensource](https://huggingface.co/collections/unitreerobotics/g1-dex1-datasets-68bae98bf0a26d617f9983ab). **Główny model** do użycia z robotem G1. | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLA-Base)|
|`UnifoLM-VLA-LIBERO`| Model specjalistyczny - fine-tunowany na zbiorze danych symulacyjnych [Libero](https://huggingface.co/collections/unitreerobotics/g1-dex1-datasets-68bae98bf0a26d617f9983ab). Do użycia w **symulacji**. | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLA-Libero)|

### Który model wybrać?

- **Dla pracy z prawdziwym robotem G1**: Użyj `UnifoLM-VLA-Base`
- **Dla eksperymentów w symulacji LIBERO**: Użyj `UnifoLM-VLA-LIBERO`
- **Dla trenowania własnego modelu od podstaw**: Zacznij od `UnifoLM-VLM-Base`

## 🛢️ Zbiory Danych

W naszych eksperymentach wykorzystujemy dwanaście zbiorów danych open-source, wszystkie zebrane z robota **Unitree G1**:

| Zbiór Danych | Robot | Link | Opis zadania |
|---------|-------|------|--------------|
|G1_Stack_Block| [Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Stack_Block)| Układanie kolorowych klocków w stosy |
|G1_Bag_Insert|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Bag_Insert)| Wkładanie różnych przedmiotów do torby |
|G1_Erase_Board|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Erase_Board)| Ścieranie napisów z tablicy gąbką |
|G1_Clean_Table|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Clean_Table)| Zbieranie śmieci i czyszczenie powierzchni stołu |
|G1_Pack_PencilBox|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Pack_PencilBox)| Pakowanie przyborów piśmienniczych do piórnika |
|G1_Pour_Medicine|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Pour_Medicine)| Precyzyjne nalewanie płynnych leków |
|G1_Pack_PingPong|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Pack_PingPong)| Zbieranie i pakowanie piłeczek pingpongowych |
|G1_Prepare_Fruit|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Prepare_Fruit)| Przygotowywanie owoców (mycie, obieranie) |
|G1_Organize_Tools|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Organize_Tools)| Organizowanie i układanie narzędzi w szufladzie |
|G1_Fold_Towel|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Fold_Towel)| Składanie ręcznika w precyzyjny sposób |
|G1_Wipe_Table|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Wipe_Table)| Wycieranie stołu ściereczką |
|G1_DualRobot_Clean_Table|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_DualRobot_Clean_Table)| Współpraca dwóch robotów przy czyszczeniu stołu |

### Format danych: RLDS (Reinforcement Learning Datasets)

Wszystkie dane muszą być w formacie **RLDS** - standardzie HuggingFace LeRobot V2.1 dla zbiorów uczenia ze wzmocnieniem.

### Przygotowanie własnych danych

Aby trenować na własnym zbiorze danych, upewnij się, że dane są w formacie [Huggingface LeRobot V2.1](https://github.com/huggingface/lerobot). 

Zakładając następującą strukturę katalogów źródłowych:
```
source_dir/
    ├── dataset1_name
    ├── dataset2_name
    ├── dataset3_name
    └── ...
```

#### Krok 1: Konwersja z LeRobot do HDF5

```bash
cd prepare_data
python convert_lerobot_to_hdf5.py \
    --data_path /ścieżka/do/source_dir/dataset1_name \
    --target_path /ścieżka/do/zapisu/przekonwertowanych/danych
```

**Dlaczego HDF5?** Format HDF5 (Hierarchical Data Format) efektywnie przechowuje duże ilości danych hierarchicznych, idealny dla sekwencji obrazów i działań robota.

#### Krok 2: Konwersja z HDF5 do RLDS

Najpierw zaktualizuj ścieżkę ([tutaj](prepare_data/hdf5_to_rlds/rlds_dataset/rlds_dataset.py#L232)) do poprawnej lokalizacji danych HDF5.

```bash
cd prepare_data/hdf5_to_rlds/rlds_dataset
tfds build --data_dir /ścieżka/do/zapisu/przekonwertowanych/danych
```

Struktura przekonwertowanego zbioru danych RLDS:
```
source_dir/
├── downloads
├── rlds_dataset
         └── 1.0.0
```

Katalog `1.0.0` to finalna wersja zbioru RLDS gotowa do użycia w treningu. Końcowa ścieżka powinna być zachowana jako `source_dir/1.0.0` (np. `g1_stack_block/1.0.0`).

## 🚴‍♂️ Trenowanie (Training)

### Proces trenowania krok po kroku

Aby trenować na jednym lub wielu zbiorach danych, wykonaj następujące kroki:

#### **Krok 1: Rejestracja zbioru danych**

Zakładając, że masz już przygotowany zbiór RLDS, zarejestruj go (np. zbiór open-source Unitree `G1_StackBox`) w naszym dataloader poprzez dodanie wpisu w następujących plikach:

1. **`configs.py`** ([tutaj](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/configs.py#L58)) - Konfiguracja podstawowa zbioru
2. **`transforms.py`** ([tutaj](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/transforms.py#L948)) - Transformacje danych (normalizacja, augmentacja)
3. **`mixtures.py`** ([tutaj](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/mixtures.py#L366)) - Mieszanki zbiorów danych
4. **`datasets.py`** ([tutaj](src/unifolm_vla/rlds_dataloader/datasets/datasets.py#L106)) - Główna konfiguracja datasetu

Dla odniesienia, w każdym z tych plików znajdują się przykładowe wpisy dla zbiorów G1, które wykorzystaliśmy w eksperymentach.

#### **Krok 2: Konfiguracja stałych akcji i stanów**

Przed rozpoczęciem fine-tuningu skonfiguruj w `constants.py` ([tutaj](src/unifolm_vla/rlds_dataloader/constants.py#L70)):

- **`NUM_ACTIONS_CHUNK`**: Rozmiar chunka (porcji) akcji przewidywanych przez model
- **`ACTION_DIM`**: Wymiar akcji (liczba stopni swobody akcji)
- **`PROPRIO_DIM`**: Wymiar propriocepcji (czujniki wewnętrzne robota - pozycje stawów itp.)
- **`ACTION_PROPRIO_NORMALIZATION_TYPE`**: Schemat normalizacji danych

Zobacz `G1_CONSTANTS` dla przykładu konfiguracji robota G1.

**Dlaczego to ważne?** Różne roboty mają różną liczbę stawów i czujników. Prawidłowa konfiguracja wymiarów jest kluczowa dla poprawnego działania modelu.

#### **Krok 3: Konfiguracja skryptu treningowego**

Skonfiguruj parametry w następującej kolejności (zobacz [tutaj](scripts/run_scripts/run_unifolm_vla_train.sh)):

1. **Inicjalizacja modelu (`base_vlm`)**: 
   - Ustaw na lokalną ścieżkę lub URL wag modelu **UnifoLM-VLM-Base**
   - Ten model inicjalizuje szkielet wizyjno-językowy (vision-language backbone)

2. **Ścieżka zbioru danych (`oxe_data_root`)**: 
   - Katalog główny zbioru danych
   - Zapewnia, że skrypt treningowy poprawnie załaduje dane RLDS

3. **Specyfikacja mieszanki zbiorów (`data_mix`)**: 
   - Nazwa zbioru lub mieszanki zbiorów do użycia w treningu
   - Możesz trenować na jednym zbiorze lub kombinacji wielu

4. **Zapis punktów kontrolnych modelu**: 
   - Ścieżki do zapisywania checkpoint'ów i logów
   - Przechowują wagi i stany treningowe dla późniejszego użycia

5. **Konfiguracja równoległości (`num_processes`)**: 
   - Dostosuj według liczby dostępnych GPU
   - Określa skalę trenowania rozproszonego (distributed training)

#### **Krok 4: Uruchomienie treningu**

Przed uruchomieniem skryptu [`run_unifolm_vla_train.sh`](scripts/run_scripts/run_unifolm_vla_train.sh), upewnij się, że wszystkie powyższe konfiguracje są poprawnie ustawione.

```bash
conda activate unifolm-vla
cd unifolm-vla
bash scripts/run_scripts/run_unifolm_vla_train.sh
```

### Monitorowanie treningu

- **TensorBoard**: Logi treningowe są zapisywane do TensorBoard, możesz je wizualizować
- **Wandb**: Jeśli skonfigurowane, można śledzić eksperymenty online
- **Checkpoints**: Regularnie zapisywane punkty kontrolne pozwalają na wznowienie treningu

## 🌏 Ewaluacja Inferencji w Symulacji

### Testowanie w środowisku LIBERO

Aby ocenić model **UnifoLM-VLA-Libero** w środowisku symulacyjnym `LIBERO` ([tutaj](https://huggingface.co/datasets/openvla/modified_libero_rlds)), wykonaj następujące kroki:

#### **Krok 1: Instalacja środowiska LIBERO**

```bash
# Sklonowanie repozytorium LIBERO (symulator robotyczny)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO

# Instalacja dodatkowych zależności dla LIBERO
# Uruchom z głównego katalogu projektu UnifoLM-VLA
pip install -r experiments/LIBERO/libero_requirements.txt
```

**Co to jest LIBERO?** To środowisko symulacyjne do benchmarkingu zadań manipulacyjnych robotów. Pozwala testować modele bez potrzeby dostępu do prawdziwego robota.

#### **Krok 2: Konfiguracja skryptu ewaluacji**

W `run_eval_libero.sh` ([tutaj](scripts/eval_scripts/run_eval_libero.sh)), zmodyfikuj następujące pola:

- **`your_ckpt`**: Ścieżka do wytrenowanego checkpointu modelu
- **`task_suite_name`**: Nazwa zestawu zadań do ewaluacji
- **`unnorm_key`**: Klucz do denormalizacji akcji
- **`LIBERO_HOME`**: Ścieżka do instalacji LIBERO
- **`vlm_pretrained_path`**: Ścieżka do pretrenowanego modelu VLM

#### **Krok 3: Uruchomienie ewaluacji**

```bash
conda activate unifolm-vla
cd unifolm-vla
bash scripts/eval_scripts/run_eval_libero.sh
```

**Wyniki**: Skrypt wyprodukuje metryki sukcesu dla różnych zadań symulacyjnych, pozwalając ocenić jakość modelu.

## 🤖 Ewaluacja Inferencji na Prawdziwym Robocie

### Architektura klient-serwer

W naszym systemie inferencja jest wykonywana **po stronie serwera**. Klient robota zbiera obserwacje z prawdziwego robota i wysyła je na serwer w celu wnioskowania akcji. 

**Dlaczego taka architektura?**
- **Obliczenia na GPU**: Serwer ma mocne GPU do szybkiej inferencji
- **Lekki klient**: Robot nie musi posiadać drogiego sprzętu obliczeniowego
- **Elastyczność**: Łatwa aktualizacja modelu bez zmiany oprogramowania robota

Pełny pipeline można zrealizować wykonując poniższe kroki:

### Konfiguracja Serwera

#### **Krok 1: Konfiguracja skryptu serwera**

W `run_real_eval_server.sh` ([tutaj](scripts/eval_scripts/run_real_eval_server.sh)), zmodyfikuj:

- **`ckpt_path`**: Ścieżka do checkpointu modelu
- **`port`**: Port, na którym serwer będzie nasłuchiwał
- **`unnorm_key`**: Klucz denormalizacji akcji
- **`vlm_pretrained_path`**: Ścieżka do pretrenowanego VLM

#### **Krok 2: Uruchomienie serwera**

```bash
conda activate unifolm-vla
cd unifolm-vla
bash scripts/eval_scripts/run_real_eval_server.sh
```

Serwer rozpocznie nasłuchiwanie na określonym porcie, gotowy do przyjmowania obserwacji od klienta robota.

### Konfiguracja Klienta (Robot)

#### **Krok 1: Przygotowanie środowiska klienta**

Zapoznaj się z [unitree_deploy/README.md](https://github.com/unitreerobotics/unifolm-world-model-action/blob/main/unitree_deploy/README.md) aby:
- Utworzyć środowisko conda `unitree_deploy`
- Zainstalować wymagane zależności
- Uruchomić kontroler lub serwis na prawdziwym robocie

#### **Krok 2: Ustanowienie tunelu SSH**

Otwórz nowy terminal i ustanów połączenie tunelowe od klienta do serwera:

```bash
ssh nazwa_użytkownika@IP_serwera -CNg -L port:127.0.0.1:port
```

**Co robi ta komenda?**
- **SSH tunnel**: Bezpieczne połączenie sieciowe
- **Port forwarding**: Przekierowuje komunikację na lokalny port
- **-CNg**: Flagi SSH (Compression, No command, background)

#### **Krok 3: Uruchomienie klienta robota**

Zmodyfikuj i uruchom skrypt `unitree_deploy/robot_client.py` jako odniesienie.

**Przepływ danych:**
1. Robot zbiera obserwacje (obrazy z kamery, stan stawów)
2. Klient wysyła obserwacje na serwer przez tunel SSH
3. Serwer wykonuje inferencję modelu VLA
4. Przewidywane akcje są odsyłane do klienta
5. Robot wykonuje akcje

## 📝 Architektura Kodu

Oto przegląd wysokopoziomowy struktury projektu i głównych komponentów:

```
unifolm-vla/
    ├── assets/                     # Zasoby medialne (GIF-y, obrazy demonstracyjne)
    │
    ├── experiments/                # Eksperymenty i zbiory danych
    │   └── LIBERO/                 # Kod do eksperymentów w symulatorze LIBERO
    │       ├── eval_libero.py      # Główny skrypt ewaluacji
    │       ├── libero_utils.py     # Funkcje pomocnicze dla LIBERO
    │       └── unifolm_vla_inference.py  # Interfejs inferencji dla LIBERO
    │
    ├── deployment/                 # Kod do wdrożenia produkcyjnego
    │   └── model_server/           # Serwer modelu dla inferencji na robotach
    │       └── run_real_eval_server.py  # Serwer dla prawdziwych robotów
    │
    ├── prepare_data/               # Skrypty przetwarzania i konwersji danych
    │   ├── convert_lerobot_to_hdf5.py   # LeRobot → HDF5
    │   └── hdf5_to_rlds/           # HDF5 → RLDS (TensorFlow Datasets)
    │       └── rlds_dataset/
    │
    ├── scripts/                    # Główne skrypty uruchomieniowe
    │   ├── run_scripts/            # Skrypty treningowe
    │   │   └── run_unifolm_vla_train.sh
    │   └── eval_scripts/           # Skrypty ewaluacyjne
    │       ├── run_eval_libero.sh
    │       └── run_real_eval_server.sh
    │
    └── src/unifolm_vla/            # Główny pakiet Python
        ├── config/                 # Pliki konfiguracyjne treningu
        │
        ├── model/                  # Architektury modeli
        │   ├── framework/          # Główne frameworki modelowe
        │   │   ├── unifolm_vla.py  # Klasa główna UnifoLM-VLA
        │   │   └── base_framework.py
        │   ├── modules/            # Moduły składowe
        │   │   ├── vlm/            # Moduł Vision-Language (Qwen2.5-VL)
        │   │   └── action_model/   # Moduł predykcji akcji (DiT)
        │   └── utils/              # Narzędzia pomocnicze
        │
        ├── rlds_dataloader/        # Ładowanie i przetwarzanie danych
        │   ├── datasets/           # Definicje zbiorów danych
        │   │   └── rlds/oxe/       # Konfiguracje zbiorów OXE/RLDS
        │   │       ├── configs.py  # Konfiguracje zbiorów
        │   │       ├── transforms.py  # Transformacje danych
        │   │       └── mixtures.py # Mieszanki zbiorów
        │   └── constants.py        # Stałe (wymiary akcji, normalizacja)
        │
        └── training/               # Kod treningowy
            ├── train_unifolm_vla.py  # Główny skrypt treningu
            └── trainer_utils/      # Narzędzia treningowe
                ├── metrics.py      # Metryki ewaluacyjne
                ├── overwatch.py    # Monitoring treningu
                └── trainer_tools.py
```

### Kluczowe komponenty dla studentów:

1. **`src/unifolm_vla/model/`**: 
   - Zrozumienie architektury modelu VLA
   - Integracja Vision-Language-Action

2. **`src/unifolm_vla/rlds_dataloader/`**: 
   - Jak dane robotyczne są ładowane i przetwarzane
   - Transformacje i normalizacja

3. **`scripts/`**: 
   - Praktyczne skrypty do uruchomienia treningu i ewaluacji
   - Wzorce konfiguracji

4. **`experiments/LIBERO/`**: 
   - Przykład integracji z symulatorem
   - Interfejs inferencji

## 🎓 Wskazówki dla Studentów

### Zaczynanie pracy z projektem:

1. **Najpierw symulacja**: Zacznij od eksperymentów w LIBERO, zanim przejdziesz do prawdziwego robota
2. **Czytaj kod od góry**: Zacznij od `unifolm_vla.py`, potem przejdź do modułów składowych
3. **Debugowanie**: Użyj małych zbiorów danych do szybkiego testowania zmian
4. **Wizualizacja**: Zawsze wizualizuj predykcje modelu przed wdrożeniem na robocie

### Typowy workflow projektu:

```
1. Zbieranie danych demonstracyjnych z robota G1
   ↓
2. Konwersja danych (LeRobot → HDF5 → RLDS)
   ↓
3. Rejestracja zbioru danych w dataloader
   ↓
4. Konfiguracja parametrów treningu
   ↓
5. Trenowanie modelu (start: UnifoLM-VLM-Base)
   ↓
6. Ewaluacja w symulacji (LIBERO)
   ↓
7. Ewaluacja na prawdziwym robocie
   ↓
8. Iteracja: zbieranie więcej danych, fine-tuning
```

## 🙏 Podziękowania

Znaczna część kodu została odziedziczona z projektów:
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) - Model Vision-Language
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) - NVIDIA robotics
- [Open-X](https://robotics-transformer-x.github.io/) - Open-X-Embodiment dataset
- [openvla-oft](https://github.com/moojink/openvla-oft) - VLA framework
- [InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1) - VLA architecture

## 📚 Dodatkowe Zasoby

- [GUIDE_FOR_STUDENTS_PL.md](GUIDE_FOR_STUDENTS_PL.md) - Szczegółowy przewodnik dla studentów
- [UNITREE_G1_PRACTICAL_GUIDE_PL.md](UNITREE_G1_PRACTICAL_GUIDE_PL.md) - Praktyczny przewodnik pracy z robotem G1

## 📝 Cytowanie

```bibtex
@misc{unifolm-vla-0,
  author       = {Unitree},
  title        = {UnifoLM-VLA-0: A Vision-Language-Action (VLA) Framework under UnifoLM Family},
  year         = {2026},
}
```

---

**Opracowane dla studentów Politechniki Rzeszowskiej**  
*Tłumaczenie i adaptacja edukacyjna - 2026*
