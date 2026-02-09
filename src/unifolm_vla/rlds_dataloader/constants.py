"""
constants.py - Stałe konfiguracyjne dla różnych platform robotycznych

OPIS PO POLSKU:
Ten plik zawiera KLUCZOWE PARAMETRY definiujące wymiary akcji i stanów
dla różnych robotów i zadań. To jest JEDEN Z NAJWAŻNIEJSZYCH PLIKÓW
do zrozumienia przed rozpoczęciem trenowania własnych modeli.

DLACZEGO TO WAŻNE?
Różne roboty mają:
- Różną liczbę stawów (degrees of freedom - DOF)
- Różne zakresy ruchów
- Różne czujniki proprioceptywne

Przed treningiem na nowych danych MUSISZ:
1. Wybrać odpowiednie stałe dla swojego robota
2. Lub zdefiniować nowe stałe jeśli używasz innego robota
3. Sprawdzić czy wymiary pasują do twoich danych

STRUKTURA:
- Definicje NormalizationType: Jak normalizować dane
- Stałe dla różnych robotów: G1, LIBERO, ALOHA, etc.
- Automatyczna detekcja platformy na podstawie argumentów CLI

Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
# (Nie używane w UnifoLM-VLA, pozostawione dla kompatybilności)
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'

# lisa method
ACTION_TOKEN_IDX = 32001

# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    """
    Typy normalizacji dla akcji i stanu proprioceptywnego.
    
    OPIS PO POLSKU:
    Normalizacja jest kluczowa dla stabilnego treningu sieci neuronowych.
    Różne zakresy wartości (np. -π do π vs 0 do 1) utrudniają uczenie.
    
    Dostępne metody:
    
    1. NORMAL (Gaussian Normalization):
       - Normalizuje do Mean = 0, Std = 1
       - Wzór: x_norm = (x - mean) / std
       - Najlepsze dla: Rozkłady zbliżone do normalnego
       - Przykład: Pozycje stawów z symetrycznym ruchem
    
    2. BOUNDS (Min-Max Normalization):
       - Normalizuje do przedziału [-1, 1]
       - Wzór: x_norm = 2 * (x - min) / (max - min) - 1
       - Najlepsze dla: Znane sztywne granice wartości
       - Przykład: Gripper (zawsze 0.0 do 1.0)
    
    3. BOUNDS_Q99 (Quantile-based Normalization):
       - Normalizuje [percentyl_1, ..., percentyl_99] → [-1, ..., 1]
       - Ignoruje outliers (skrajne 1% wartości)
       - Najlepsze dla: Dane z outliers lub długimi ogonami rozkładu
       - Przykład: Prędkości mogą mieć sporadyczne skoki
    
    JAK WYBRAĆ?
    - Jeśli znasz dokładne zakresy → BOUNDS
    - Jeśli dane mają outliers → BOUNDS_Q99
    - W pozostałych przypadkach → NORMAL
    """
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
# ============================================
# STAŁE DLA RÓŻNYCH PLATFORM ROBOTYCZNYCH
# ============================================

LIBERO_CONSTANTS = {
    """
    Stałe dla środowiska symulacyjnego LIBERO.
    
    LIBERO: Long-horizon manipulation benchmark w symulacji (Mujoco)
    Robot: Franka Emika Panda (7 DOF ramię + 1 DOF gripper)
    
    Konfiguracja:
    - NUM_ACTIONS_CHUNK: 8 akcji przewidywanych naraz (action chunking)
      → Robot planuje 8 kroków do przodu
    - ACTION_DIM: 7 (7 stawów ramienia, gripper osobno w propriocepcji)
    - PROPRIO_DIM: 8 (7 pozycji stawów + 1 pozycja grippera)
    - Normalizacja: BOUNDS_Q99 (ignoruje outliers w symulacji)
    """
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    """
    Stałe dla robota ALOHA (bi-manual manipulation).
    
    ALOHA: Robot z dwoma ramionami do zadań wymagających obu rąk
    
    Konfiguracja:
    - NUM_ACTIONS_CHUNK: 25 (dłuższy horyzont planowania)
    - ACTION_DIM: 14 (7 DOF x 2 ramiona)
    - PROPRIO_DIM: 14 (pozycje 14 stawów)
    - Normalizacja: BOUNDS (znane sztywne granice)
    """
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    """
    Stałe dla robota z datasetu BridgeData.
    
    BridgeData: Duży zbiór danych manipulation z WidowX robot
    
    Konfiguracja:
    - NUM_ACTIONS_CHUNK: 5 (krótszy horyzont)
    - ACTION_DIM: 7 (6 DOF + gripper)
    - PROPRIO_DIM: 7
    - Normalizacja: BOUNDS_Q99
    """
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

FRACTAL_CONSTANTS = {
    """
    Stałe dla robota Fractal (Google Robotics).
    """
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

G1_CONSTANTS = {
    """
    Stałe dla robota Unitree G1 (kontrola przestrzeni stawów - joint space).
    
    UNITREE G1: Humanoidalny robot do manipulacji ogólnego przeznaczenia
    
    !!! UŻYJ TYCH STAŁYCH DLA PROJEKTÓW Z ROBOTEM G1 !!!
    
    Konfiguracja:
    - NUM_ACTIONS_CHUNK: 25 akcji (planowanie ~2.5s przy 10Hz)
      → Pozwala na gładkie, długie trajektorie
    - ACTION_DIM: 16
      • 7 stawów lewego ramienia
      • 7 stawów prawego ramienia
      • 1 lewy gripper
      • 1 prawy gripper
      = 16 wymiarów akcji
    
    - PROPRIO_DIM: 16 (identyczne jak ACTION_DIM dla G1)
      • 7 pozycji stawów lewego ramienia
      • 7 pozycji stawów prawego ramienia  
      • 1 pozycja lewego grippera
      • 1 pozycja prawego grippera
    
    - Normalizacja: BOUNDS (wszystkie stawy mają znane limity)
      • Stawy: typowo -π do π
      • Gripper: 0.0 (zamknięty) do 1.0 (otwarty)
    
    KIEDY UŻYĆ?
    - Kontrola w przestrzeni stawów (joint space control)
    - Standardowe zadania manipulacyjne jedną lub dwiema rękami
    - Większość przypadków użycia robota G1
    """
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 16,
    "PROPRIO_DIM": 16,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

G1_EE_6D_CONSTANTS = {
    """
    Stałe dla robota Unitree G1 (kontrola end-effector w przestrzeni kartezjańskiej).
    
    EE (End-Effector) 6D Control: Kontrola pozycji i orientacji chwytaka
    zamiast bezpośrednich pozycji stawów.
    
    Konfiguracja:
    - NUM_ACTIONS_CHUNK: 25
    - ACTION_DIM: 23
      • 6 DOF lewy end-effector (x, y, z, roll, pitch, yaw)
      • 6 DOF prawy end-effector
      • 1 lewy gripper
      • 1 prawy gripper
      • 9 dodatkowych (prawdopodobnie konfiguracja korpusu/nóg)
      = 23 wymiary
    
    - PROPRIO_DIM: 23 (pozycje i orientacje EE + gripperów)
    - Normalizacja: BOUNDS_Q99 (EE może mieć outliers)
    
    KIEDY UŻYĆ?
    - Zadania wymagające precyzyjnej kontroli pozycji w przestrzeni
    - Gdy trudno zaplanować trajektorię w przestrzeni stawów
    - Task-space control (np. "przesuń 10cm w prawo")
    """
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 23,
    "PROPRIO_DIM": 23,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

G1_STACK_BLOCK_CONSTANTS = {
    """
    Stałe specyficzne dla zadania G1_Stack_Block.
    
    To pokazuje jak można mieć task-specific constants.
    Dla zadania stack_block używamy tych samych wymiarów co G1_EE_6D.
    
    UWAGA: W praktyce możesz definiować różne stałe dla różnych zadań
    jeśli mają różne requirements (np. różna długość action chunks).
    """
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 23,
    "PROPRIO_DIM": 23,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Function to detect robot platform from command line arguments
def detect_robot_platform():
    """
    Automatyczna detekcja platformy robotycznej z argumentów linii poleceń.
    
    OPIS PO POLSKU:
    Ta funkcja analizuje argumenty z jakimi został uruchomiony skrypt
    (np. python train.py --data libero_10) i próbuje automatycznie
    wykryć, której platformy robotycznej dotyczy trening.
    
    Przykłady detekcji:
    - "libero" w argumentach → LIBERO_CONSTANTS
    - "aloha" w argumentach → ALOHA_CONSTANTS
    - "joint" w argumentach → G1_CONSTANTS (kontrola stawów)
    - "ee_6d" w argumentach → G1_EE_6D_CONSTANTS (kontrola EE)
    
    UWAGA DLA STUDENTÓW:
    Jeśli detekcja zawodzi, możesz:
    1. Użyć bardziej specyficznych nazw w argumentach
    2. Ręcznie ustawić ROBOT_PLATFORM poniżej
    3. Zdefiniować własne stałe i dodać detekcję tutaj
    
    Returns:
        str: Nazwa platformy ("LIBERO", "G1", "G1_EE_6D", etc.)
    """
    cmd_args = " ".join(sys.argv).lower()
    print(cmd_args)
    
    # Sprawdź kolejno różne platformy
    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "fractal" in cmd_args:
        return "FRACTAL"
    elif "ee_6d" in cmd_args:
        return "G1_EE_6D"
    elif "joint" in cmd_args:
        return "G1"
    elif "stack_block" in cmd_args:
        return "G1_STACK_BLOCK"
    else:
        # Domyślnie: G1 z kontrolą EE
        return "G1_EE_6D"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
# OPIS PO POLSKU:
# Na podstawie wykrytej platformy, wybieramy odpowiednie stałe.
# Te stałe będą używane przez cały framework podczas treningu i inferencji.
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "FRACTAL":
    constants = FRACTAL_CONSTANTS
elif ROBOT_PLATFORM == "G1_EE_6D":
    constants = G1_EE_6D_CONSTANTS
elif ROBOT_PLATFORM == "G1":
    constants = G1_CONSTANTS
elif ROBOT_PLATFORM == "G1_STACK_BLOCK":
    constants = G1_STACK_BLOCK_CONSTANTS


# Assign constants to global variables
# OPIS PO POLSKU:
# Eksportujemy wybrane stałe jako zmienne globalne.
# Te zmienne są importowane w innych częściach kodu:
# - dataloader używa ich do walidacji wymiarów danych
# - model używa ich do konfiguracji wyjściowego rozmiaru
# - training loop używa ich do sprawdzania spójności

NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]  # Ile akcji przewidywać naraz
ACTION_DIM = constants["ACTION_DIM"]  # Wymiar pojedynczej akcji (liczba stawów + grippery)
PROPRIO_DIM = constants["PROPRIO_DIM"]  # Wymiar stanu proprioceptywnego
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]  # Typ normalizacji

# Print which robot platform constants are being used (for debugging)
# OPIS PO POLSKU:
# Wypisanie jakie stałe zostały wybrane - BARDZO WAŻNE dla debugowania!
# Jeśli widzisz nieoczekiwane wartości, prawdopodobnie:
# 1. Źle wykryto platformę (sprawdź nazwę w argumentach CLI)
# 2. Użyto złych stałych (ustaw ręcznie poniżej)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f" in constants.py NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `training/vla/constants.py`!")

# UWAGA DLA STUDENTÓW:
# =====================
# Jeśli automatyczna detekcja nie działa, możesz ręcznie ustawić stałe:
#
# Przykład dla robota G1 z kontrolą stawów:
# NUM_ACTIONS_CHUNK = 25
# ACTION_DIM = 16
# PROPRIO_DIM = 16
# ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS
#
# Albo po prostu odkomentuj poniższe linie i ustaw swoje wartości:
# NUM_ACTIONS_CHUNK = ...
# ACTION_DIM = ...
# PROPRIO_DIM = ...
# ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType....
