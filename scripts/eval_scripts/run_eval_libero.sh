#!/bin/bash
# =============================================================================
# run_eval_libero.sh - Skrypt Ewaluacji w Symulatorze LIBERO
# =============================================================================
#
# OPIS PO POLSKU:
# Ten skrypt służy do testowania wytrenowanego modelu UnifoLM-VLA
# w środowisku symulacyjnym LIBERO (benchmark manipulacji robotycznych).
#
# LIBERO: Long-horizon manipulation benchmark
# - 130 zadań w 4 kategoriach trudności
# - Symulacja w Mujoco (fizyka wysokiej jakości)
# - Standardowy benchmark dla VLA models
#
# DLACZEGO EWALUACJA W SYMULACJI?
# 1. Szybkie (setki prób w godzinę)
# 2. Bezpieczne (brak ryzyka uszkodzenia robota)
# 3. Powtarzalne (identyczne warunki początkowe)
# 4. Benchmark (porównywalne z innymi pracami)
#
# PRZED URUCHOMIENIEM:
# 1. Zainstaluj LIBERO: git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
# 2. Ustaw LIBERO_HOME na ścieżkę instalacji
# 3. Wytrenuj lub pobierz checkpoint modelu
# 4. Skonfiguruj ścieżki poniżej
#
# =============================================================================

# -----------------------------------------------------------------------------
# KONFIGURACJA ŚRODOWISKA LIBERO
# -----------------------------------------------------------------------------

# Główny katalog instalacji LIBERO
# ZMIEŃ NA SWOJĄ ŚCIEŻKĘ!
export LIBERO_HOME=/jfs/jiang/code/unitree/LIBERO
# export LIBERO_HOME=/path/to/your/LIBERO  # <-- Odkomentuj i ustaw swoją ścieżkę

# Ścieżka do plików konfiguracyjnych LIBERO (zazwyczaj ${LIBERO_HOME}/libero)
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

# Dodanie LIBERO do PYTHONPATH (żeby Python mógł go zaimportować)
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}  # Dodanie obecnego katalogu

# -----------------------------------------------------------------------------
# KONFIGURACJA MODELU I CHECKPOINTU
# -----------------------------------------------------------------------------

# Ścieżka do wytrenowanego checkpointu modelu
# Użyj checkpointu z najlepszym validation loss lub ostatniego
# PRZYKŁADY WARTOŚCI (odkomentuj i ustaw):
# your_ckpt=/path/to/your/Unifolm-VLA-Libero/checkpoints/pytorch_model.pt
# vlm_pretrained_path=/path/to/your/Unifolm-VLM-Base
your_ckpt=/DATA/disk2/unitree_vla/unitreevla_libero_4_task_window_size_2/checkpoints/pytorch_model.pt
vlm_pretrained_path=/root/Unifolm-VLM-0

# Automatyczne wydobycie nazwy folderu i kroku z ścieżki checkpointu
# (używane do organizacji wyników)
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $5}')
step_name=$(echo "$your_ckpt" | awk -F'/' '{print $6}')

# -----------------------------------------------------------------------------
# KONFIGURACJA ZADAŃ I EWALUACJI
# -----------------------------------------------------------------------------

# Wybór zestawu zadań (task suite) do ewaluacji
# Dostępne opcje:
# - libero_spatial: Zadania wymagające rozumienia przestrzennego (10 tasków)
# - libero_goal: Różne cele, ta sama scena (10 tasków)
# - libero_object: Różne obiekty, podobne cele (10 tasków)
# - libero_10: 10 diverse tasków (benchmark baseline)
# - libero_90: 90 diverse tasków (pełny benchmark, bardzo trudny)
task_suite_name=libero_spatial   # ZMIEŃ aby testować inne zestawy

# Liczba prób dla każdego zadania
# PRZYKŁAD: 50 prób × 10 zadań = 500 łącznych ewaluacji
# Zalecane: min 20 dla wiarygodnych statystyk, 50+ dla publikacji
num_trials_per_task=50

# Rozmiar okna temporalnego (musi pasować do treningu!)
# window_size=1: Tylko obecna klatka
# window_size=2: 2 ostatnie klatki (temporal context)
window_size=2

# Klucz denormalizacji akcji
# KRYTYCZNE: Musi pasować do danych użytych w treningu!
# Format: {dataset_name}_no_noops
# "no_noops" oznacza, że usunięto no-op akcje (pauzę) z danych
unnorm_key="libero_spatial_no_noops"
# Opcje dla innych task suites:
# - "libero_goal_no_noops"
# - "libero_object_no_noops"
# - "libero_10_no_noops"
# - "libero_90_no_noops"

# -----------------------------------------------------------------------------
# KONFIGURACJA ZAPISU WYNIKÓW
# -----------------------------------------------------------------------------

# Ścieżka do zapisu video i wyników
# Struktura: results/{task_suite}/{folder_name}/{step_name}/
# Przykład: results/libero_spatial/exp001_baseline/checkpoint-10000/
video_out_path="results/${task_suite_name}/${folder_name}/${step_name}"
# Folder zostanie utworzony automatycznie, będzie zawierał:
# - Videos (mp4) każdej próby
# - JSON z wynikami (success rate per task)
# - Agregowane statystyki

# Które GPU użyć (0 = pierwsze GPU, 1 = drugie, etc.)
# Jeśli masz tylko jedno GPU, zostaw 0
DEVICE=0

# -----------------------------------------------------------------------------
# URUCHOMIENIE EWALUACJI
# -----------------------------------------------------------------------------

echo "=========================================="
echo "  LIBERO EVALUATION"
echo "=========================================="
echo "Task Suite: ${task_suite_name}"
echo "Checkpoint: ${your_ckpt}"
echo "Trials per task: ${num_trials_per_task}"
echo "Output path: ${video_out_path}"
echo "=========================================="
echo ""

# Uruchomienie skryptu ewaluacyjnego
CUDA_VISIBLE_DEVICES=${DEVICE} python ./experiments/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    `# Ścieżka do wytrenowanego checkpointu VLA` \
    \
    --args.vlm-pretrained-path ${vlm_pretrained_path} \
    `# Ścieżka do pretrenowanego VLM (Qwen2.5-VL)` \
    \
    --args.task-suite-name "$task_suite_name" \
    `# Zestaw zadań do ewaluacji` \
    \
    --args.num-trials-per-task "$num_trials_per_task" \
    `# Liczba prób na zadanie (więcej = lepsze statystyki)` \
    \
    --args.video-out-path "$video_out_path" \
    `# Gdzie zapisać video i wyniki` \
    \
    --args.unnorm-key "$unnorm_key" \
    `# Klucz denormalizacji (musi pasować do treningu!)` \
    \
    --args.window-size "$window_size"
    `# Rozmiar okna temporalnego (musi pasować do treningu!)`

# -----------------------------------------------------------------------------
# PO ZAKOŃCZENIU EWALUACJI
# -----------------------------------------------------------------------------
#
# Wyniki znajdziesz w: ${video_out_path}/
#
# STRUKTURA WYNIKÓW:
# results/{task_suite}/{model}/
# ├── task_0/
# │   ├── trial_0_success.mp4
# │   ├── trial_1_failure.mp4
# │   └── ...
# ├── task_1/
# │   └── ...
# ├── results.json  # Success rate per task
# └── summary.txt   # Agregowane statystyki
#
# INTERPRETACJA WYNIKÓW:
# - Success Rate ≥ 80% → Doskonały wynik
# - Success Rate 60-80% → Dobry wynik
# - Success Rate 40-60% → Przeciętny wynik
# - Success Rate < 40% → Słaby wynik, wymaga poprawy
#
# NASTĘPNE KROKI:
# 1. Przejrzyj failure videos (co poszło nie tak?)
# 2. Porównaj z baseline results
# 3. Jeśli wyniki dobre → deployment na prawdziwym robocie
# 4. Jeśli wyniki słabe → analiza błędów i fine-tuning
#
# TROUBLESHOOTING:
# - "LIBERO not found" → Sprawdź LIBERO_HOME i instalację
# - "Checkpoint not found" → Sprawdź ścieżkę your_ckpt
# - "CUDA Out of Memory" → Użyj mniejszego batch size (w eval_libero.py)
# - Success rate 0% → Sprawdź unnorm_key (prawdopodobnie źle)
#
# =============================================================================

echo ""
echo "Ewaluacja zakończona!"
echo "Wyniki dostępne w: ${video_out_path}"