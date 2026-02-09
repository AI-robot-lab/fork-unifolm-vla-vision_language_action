#!/bin/bash
# =============================================================================
# run_unifolm_vla_train.sh - Skrypt Treningowy dla UnifoLM-VLA
# =============================================================================
#
# OPIS PO POLSKU:
# Ten skrypt uruchamia trening modelu UnifoLM-VLA na danych robotycznych.
# Jest to GŁÓWNY skrypt do trenowania - przeczytaj go uważnie przed użyciem!
#
# ARCHITEKTURA TRENINGU:
# - Wykorzystuje Accelerate (HuggingFace) do distributed training
# - DeepSpeed Zero-2 dla optymalizacji pamięci GPU
# - Możliwość treningu na wielu GPU równocześnie
#
# PRZED URUCHOMIENIEM:
# 1. Upewnij się że dane są w formacie RLDS w oxe_data_root
# 2. Pobierz pretrenowany model VLM (base_vlm)
# 3. Skonfiguruj wszystkie ścieżki poniżej (oznaczone jako /path/to/your/...)
# 4. Dostosuj liczbę GPU (--num_processes)
#
# URUCHOMIENIE:
# bash scripts/run_scripts/run_unifolm_vla_train.sh
#
# =============================================================================

# -----------------------------------------------------------------------------
# KONFIGURACJA NCCL (NVIDIA Collective Communications Library)
# -----------------------------------------------------------------------------
# NCCL służy do komunikacji między GPU podczas distributed training.
# Te ustawienia optymalizują komunikację dla multi-GPU training.

export NCCL_SOCKET_IFNAME=bond0  # Interfejs sieciowy (zmień jeśli inny)
export NCCL_IB_HCA=mlx5_2,mlx5_3  # InfiniBand adapters (dla szybkich sieci)
export NCCL_BLOCKING_WAIT=1  # Czekaj na zakończenie operacji (stabilność)
export NCCL_ASYNC_ERROR_HANDLING=1  # Asynchroniczna obsługa błędów
export NCCL_TIMEOUT=1000  # Timeout w sekundach

# UWAGA: Jeśli nie masz InfiniBand, możesz usunąć NCCL_IB_HCA

# -----------------------------------------------------------------------------
# SEKCJA 1: KONFIGURACJA MODELU
# -----------------------------------------------------------------------------

# Nazwa frameworku (nie zmieniaj - to identyfikator w registry)
Framework_name=unifolm_vla

# Ścieżka do pretrenowanego Vision-Language Model
# WAŻNE: To jest punkt startowy dla treningu!
# Opcje:
# 1. Lokalna ścieżka: /path/to/downloaded/Unifolm-VLM-Base
# 2. HuggingFace Hub: unitreerobotics/Unifolm-VLM-Base (pobierze automatycznie)
base_vlm=/path/to/your/Unifolm-VLM-0

# Typ modelu VLM (Qwen2.5-VL - nie zmieniaj)
model_type=qwen2_5_vl

# Lista modułów do zamrożenia podczas treningu (optional)
# Przykłady:
# - 'vlm' - zamroź cały Vision-Language Model (tylko trenuj Action Model)
# - 'vlm.vision_encoder' - zamroź tylko encoder wizyjny
# - '' (pusty) - trenuj wszystko (domyślnie)
freeze_module_list=''

# Rozmiar okna temporalnego (context window)
# window_size=1 oznacza: używamy tylko obecnej klatki (single-frame)
# window_size=3 oznacza: używamy 3 ostatnich klatek (temporal context)
window_size=1

# -----------------------------------------------------------------------------
# SEKCJA 2: KONFIGURACJA DANYCH
# -----------------------------------------------------------------------------

# Główny katalog z danymi RLDS
# Struktura powinna być:
# oxe_data_root/
#   ├── dataset1_name/1.0.0/
#   ├── dataset2_name/1.0.0/
#   └── ...
oxe_data_root=/path/to/your/data

# Mieszanka zbiorów danych do użycia w treningu
# Opcje:
# 1. Pojedynczy dataset: 'g1_stack_block'
# 2. Mieszanka wielu: 'Unitree_all_task' (wszystkie zadania G1)
# 3. Custom mixture zdefiniowana w mixtures.py
data_mix=your_data_mix   # Przykłady: Unitree_all_task, g1_stack_block

# UWAGA DLA STUDENTÓW:
# data_mix musi być zarejestrowana w:
# src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/mixtures.py

# -----------------------------------------------------------------------------
# SEKCJA 3: ŚCIEŻKI ZAPISU
# -----------------------------------------------------------------------------

# Główny katalog dla wszystkich eksperymentów
run_root_dir=/path/to/your/run_root_dir

# Unikalny identyfikator tego konkretnego treningu
# Przykłady:
# - 'exp001_baseline_g1_stack'
# - 'exp002_multiTask_20240210'
# - 'student_jan_kowalski_proj1'
run_id=your_run_id

# Katalog wyjściowy (automatycznie tworzony)
# Będzie zawierał:
# - Checkpointy modelu (.pth files)
# - Logi treningowe (TensorBoard)
# - Kopię tego skryptu (dla reprodukowalności)
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/  # Kopia skryptu dla dokumentacji

# -----------------------------------------------------------------------------
# SEKCJA 4: URUCHOMIENIE TRENINGU
# -----------------------------------------------------------------------------

# Accelerate launch - narzędzie HuggingFace do distributed training
# --config_file: Konfiguracja DeepSpeed (optymalizacja pamięci)
# --num_processes: Liczba GPU do użycia (DOSTOSUJ DO TWOJEGO SPRZĘTU!)

accelerate launch \
  --config_file src/unifolm_vla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  src/unifolm_vla/training/train_unifolm_vla.py \
  --config_yaml ./src/unifolm_vla/config/training/unifolm_vla_train.yaml \
  \
  `# --- Parametry Modelu ---` \
  --framework.framework_py ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.model_type ${model_type} \
  \
  `# --- Parametry Danych ---` \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.window_size ${window_size} \
  --datasets.vla_data.per_device_batch_size 6 \
  `# per_device_batch_size: Batch size na JEDNO GPU` \
  `# Efektywny batch = per_device_batch_size × num_processes` \
  `# Przykład: 6 × 8 = 48 samples per training step` \
  \
  `# --- Parametry Treningu ---` \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 150000 \
  `# max_train_steps: Łączna liczba kroków treningu` \
  `# Oszacowanie czasu: ~2-7 dni na 8xA100 GPU` \
  \
  --trainer.shuffle_buffer_size 10000 \
  `# shuffle_buffer_size: Rozmiar bufora do shuffling` \
  `# Większy = lepsza randomizacja, więcej pamięci RAM` \
  \
  --trainer.save_interval 10000 \
  `# save_interval: Co ile kroków zapisywać checkpoint` \
  `# 10000 kroków = ~15 checkpointów dla 150k steps` \
  \
  --trainer.use_wrist_image True \
  `# use_wrist_image: Czy używać kamery na nadgarstku` \
  `# True jeśli masz wrist camera w danych` \
  `# False jeśli tylko główna kamera` \
  \
  --trainer.use_proprio True \
  `# use_proprio: Czy używać propriocepcji (stan stawów)` \
  `# ZALECANE: True (closed-loop control)` \
  \
  --trainer.logging_frequency 500 \
  `# logging_frequency: Co ile kroków logować metryki` \
  `# Mniejsza wartość = więcej logów = lepszy monitoring` \
  \
  --trainer.eval_interval 500 \
  `# eval_interval: Co ile kroków ewaluować na validation set` \
  `# WAŻNE: Pozwala wykryć overfitting wcześnie` \
  \
  --trainer.learning_rate.base 4e-5 \
  `# learning_rate: Bazowy learning rate (LR)` \
  `# 4e-5 (0.00004) to dobra wartość dla fine-tuning` \
  `# Jeśli loss nie spada: zwiększ LR (np. 1e-4)` \
  `# Jeśli loss oscyluje: zmniejsz LR (np. 1e-5)` \
  \
  `# --- Parametry Zapisywania ---` \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  \
  `# --- Weights & Biases (Tracking Eksperymentów) ---` \
  --wandb_project vla_jiang \
  `# wandb_project: Nazwa projektu w wandb.ai` \
  `# ZMIEŃ NA SWOJĄ: np. 'politechnika_rzeszow_g1'` \
  \
  --wandb_entity zbdz
  `# wandb_entity: Twoja nazwa użytkownika lub zespołu w wandb` \
  `# ZMIEŃ NA SWOJĄ: np. 'jan_kowalski'` \
  `# Lub zostaw puste jeśli używasz tylko TensorBoard`

# -----------------------------------------------------------------------------
# PO ZAKOŃCZENIU TRENINGU
# -----------------------------------------------------------------------------
# 
# 1. Checkpointy znajdziesz w: ${output_dir}/checkpoints/
# 2. Logi TensorBoard: tensorboard --logdir ${output_dir}/logs/
# 3. Logi Wandb: https://wandb.ai/${wandb_entity}/${wandb_project}
#
# NASTĘPNE KROKI:
# 1. Ewaluacja w symulacji (run_eval_libero.sh)
# 2. Deployment na robocie (run_real_eval_server.sh)
# 3. Analiza wyników i iteracja
#
# TROUBLESHOOTING:
# - CUDA Out of Memory → Zmniejsz per_device_batch_size
# - Loss = NaN → Zmniejsz learning_rate
# - Trening za wolny → Zwiększ num_processes (więcej GPU)
#
# =============================================================================
