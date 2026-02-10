#!/bin/bash
# =============================================================================
# run_real_eval_server.sh - Serwer Inferencji dla Prawdziwego Robota
# =============================================================================
#
# OPIS PO POLSKU:
# Ten skrypt uruchamia serwer HTTP do inferencji modelu VLA.
# Robot (klient) wysyła obserwacje, serwer zwraca przewidziane akcje.
#
# ARCHITEKTURA KLIENT-SERWER:
#
#   ┌─────────────┐                    ┌─────────────────┐
#   │   ROBOT G1  │  ←── Ethernet ───→ │  GPU Server     │
#   │   (Klient)  │                    │  (Ten skrypt)   │
#   └─────────────┘                    └─────────────────┘
#        │                                      │
#        │ 1. Wyślij obserwację                │
#        │    (obraz + proprio)                │
#        │ ─────────────────────────────────→  │
#        │                                      │
#        │                                      │ 2. Inferencja VLA
#        │                                      │    (~40-100ms)
#        │                                      │
#        │ 3. Odbierz akcje                    │
#        │ ←─────────────────────────────────  │
#        │                                      │
#        ↓                                      │
#   Wykonaj akcje                               │
#
# DLACZEGO SERWER NA GPU?
# - Model VLA wymaga GPU do szybkiej inferencji
# - Robot często ma tylko CPU (np. Raspberry Pi, Jetson Nano)
# - Centralizacja: łatwa aktualizacja modelu
#
# PRZED URUCHOMIENIEM:
# 1. Wytrenuj lub pobierz checkpoint modelu
# 2. Skonfiguruj ścieżki poniżej
# 3. Upewnij się że port jest otwarty (firewall)
# 4. Sprawdź połączenie sieciowe robot ↔ serwer
#
# URUCHOMIENIE:
# bash scripts/eval_scripts/run_real_eval_server.sh
#
# =============================================================================

# -----------------------------------------------------------------------------
# KONFIGURACJA SERWERA
# -----------------------------------------------------------------------------

# Ścieżka do wytrenowanego checkpointu
# To jest model, który będzie używany do przewidywania akcji
# ZMIEŃ NA SWOJĄ ŚCIEŻKĘ!
ckpt_path="/path/to/your/Unifolm-VLA-Base/checkpoints/pytorch_model.pt"

# Port na którym serwer będzie nasłuchiwał
# Standardowe porty: 8000-9000
# Upewnij się że:
# 1. Port nie jest używany przez inną aplikację
# 2. Firewall pozwala na połączenia na tym porcie
# 3. Klient robota łączy się na TEN SAM port
port=8777

# Klucz denormalizacji akcji
# KRYTYCZNE: Musi pasować do danych użytych w treningu!
# Dla robota G1:
# - "g1_stack_block" - dla zadania stack block
# - "g1_clean_table" - dla zadania clean table
# - "g1_joint" - dla ogólnej kontroli stawów G1
# - "g1_ee_6d" - dla kontroli end-effector
unnorm_key="g1_stack_block"

# Ścieżka do pretrenowanego VLM
# Qwen2.5-VL model (używany jako backbone)
vlm_pretrained_path="/path/to/your/Unifolm-VLM-Base"

# -----------------------------------------------------------------------------
# INFORMACJE DLA UŻYTKOWNIKA
# -----------------------------------------------------------------------------

echo "=========================================="
echo "  UnifoLM-VLA INFERENCE SERVER"
echo "=========================================="
echo "Checkpoint: ${ckpt_path}"
echo "Port: ${port}"
echo "Unnorm key: ${unnorm_key}"
echo "VLM: ${vlm_pretrained_path}"
echo "=========================================="
echo ""
echo "Serwer będzie dostępny na:"
echo "  http://localhost:${port}"
echo "  http://<YOUR_IP>:${port}"
echo ""
echo "Oczekiwanie na połączenia od klienta robota..."
echo "Naciśnij Ctrl+C aby zatrzymać serwer"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# URUCHOMIENIE SERWERA
# -----------------------------------------------------------------------------

python deployment/model_server/run_real_eval_server.py \
    --ckpt_path ${ckpt_path} \
    `# Ścieżka do wytrenowanego checkpointu modelu VLA` \
    \
    --port ${port} \
    `# Port HTTP na którym serwer nasłuchuje` \
    `# Klient robota musi łączyć się na TEN SAM port` \
    \
    --unnorm_key ${unnorm_key} \
    `# Klucz do denormalizacji akcji` \
    `# MUSI pasować do danych treningowych!` \
    `# Niepoprawny klucz → robot będzie wykonywał dziwne ruchy!` \
    \
    --vlm_pretrained_path ${vlm_pretrained_path}
    `# Ścieżka do pretrenowanego Qwen2.5-VL` \
    `# Używany jako vision-language backbone`

# -----------------------------------------------------------------------------
# INSTRUKCJE DLA KLIENTA ROBOTA
# -----------------------------------------------------------------------------
#
# KONFIGURACJA KLIENTA:
#
# 1. Na komputerze robota, utwórz tunel SSH (jeśli serwer zdalny):
#    ssh -L ${port}:localhost:${port} user@server_ip -CNg
#
# 2. W kodzie klienta, ustaw:
#    server_url = f"http://localhost:${port}"
#
# 3. Format requestu (POST /predict):
#    {
#      "image": [[...]],  # Numpy array jako lista (H, W, 3)
#      "proprio": [...],   # Numpy array jako lista (PROPRIO_DIM,)
#      "instruction": "pick up the red cube"
#    }
#
# 4. Format response:
#    {
#      "actions": [[...]],  # Numpy array jako lista (NUM_CHUNKS, ACTION_DIM)
#      "inference_time_ms": 45.2
#    }
#
# PRZYKŁADOWY KOD KLIENTA (Python):
#
# import requests
# import numpy as np
#
# # Zbierz obserwacje
# image = robot.get_camera_image()  # (720, 1280, 3)
# proprio = robot.get_joint_states()  # (16,) dla G1
#
# # Wyślij do serwera
# response = requests.post(
#     f"http://localhost:${port}/predict",
#     json={
#         "image": image.tolist(),
#         "proprio": proprio.tolist(),
#         "instruction": "pick up the red cube"
#     },
#     timeout=5.0
# )
#
# # Odbierz akcje
# actions = np.array(response.json()["actions"])
#
# # Wykonaj pierwszą akcję
# robot.execute_action(actions[0])
#
# -----------------------------------------------------------------------------
# MONITOROWANIE I DEBUGGING
# -----------------------------------------------------------------------------
#
# Logi serwera pokazują:
# - Otrzymane requesty (timestamp, client IP)
# - Czas inferencji (ms)
# - Błędy (jeśli wystąpią)
#
# Typowe czasy inferencji:
# - GPU RTX 3090: ~40-60ms
# - GPU A100: ~30-40ms
# - CPU (nie zalecane): ~2000-5000ms
#
# TROUBLESHOOTING:
# - "Connection refused" → Sprawdź czy port otwarty, firewall
# - "Model not found" → Sprawdź ścieżkę checkpointu
# - "CUDA Out of Memory" → Użyj tylko jednego klienta naraz
# - "Invalid action dimensions" → Sprawdź unnorm_key
# - Dziwne ruchy robota → Prawdopodobnie zły unnorm_key!
#
# BEZPIECZEŃSTWO:
# - Serwer NIE ma wbudowanych safety checks!
# - Klient MUSI sprawdzać safety limits przed wykonaniem akcji
# - Zawsze miej przycisk EMERGENCY STOP w zasięgu ręki
# - Testuj najpierw na małych ruchach bez obiektów
#
# =============================================================================