# 🤖 Praktyczny Przewodnik: Praca z Robotem Unitree G1 EDU-U6

## Unitree G1 + UnifoLM-VLA Framework

---

## 📑 Spis Treści

1. [Wprowadzenie do Robota Unitree G1](#wprowadzenie-do-robota-unitree-g1)
2. [Architektura Systemu](#architektura-systemu)
3. [Pierwszy Kontakt z Robotem](#pierwszy-kontakt-z-robotem)
4. [Zbieranie Danych Treningowych](#zbieranie-danych-treningowych)
5. [Pipeline: Od Demonstracji do Wytrenowanego Modelu](#pipeline)
6. [Deployment na Prawdziwym Robocie](#deployment)
7. [Bezpieczeństwo i Best Practices](#bezpieczeństwo)
8. [Przykładowe Projekty](#przykładowe-projekty)

---

## 🤖 Wprowadzenie do Robota Unitree G1

### Specyfikacja Techniczna G1 EDU-U6

**Unitree G1 EDU-U6** to humanoidalny robot przeznaczony do badań i edukacji:

```
┌─────────────────────────────────────────────────────┐
│  ROBOT UNITREE G1 EDU-U6                            │
├─────────────────────────────────────────────────────┤
│  Wysokość: ~1.3m                                    │
│  Ciężar: ~35kg                                      │
│  Ramiona: 2x (7 stopni swobody każde)              │
│  Chwytaki: 2x (1 DOF - otwarcie/zamknięcie)        │
│  Mobilność: Nogi (opcjonalnie, dla U6 limited)     │
│  Kamery: RGB (główna kamera manipulacyjna)         │
│  Czujniki: Encodery w stawach, force/torque        │
│  Zasilanie: Bateria litowa (~2h pracy)             │
└─────────────────────────────────────────────────────┘
```

### Anatomia Ramienia G1

Każde ramię ma **7 stopni swobody (DOF)**:

```
        Bark (Shoulder)
             │
    ┌────────┴────────┐
    │   Joint 1       │  Rotacja ramienia (wokół osi pionowej)
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   Joint 2       │  Podnoszenie/opuszczanie ramienia
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   Joint 3       │  Rotacja przedramienia
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   Joint 4       │  Zgięcie łokcia
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   Joint 5       │  Rotacja nadgarstka (roll)
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   Joint 6       │  Pitch nadgarstka
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   Joint 7       │  Yaw nadgarstka
    └────────┬────────┘
             │
        ┌────┴────┐
        │ Gripper │  Chwytak (1 DOF: 0=zamknięty, 1=otwarty)
        └─────────┘
```

**Zakres ruchu stawów:**

| Staw | Zakres | Maksymalna prędkość |
|------|--------|---------------------|
| Joint 1 | -π do π | 2.0 rad/s |
| Joint 2 | -π/2 do π/2 | 2.0 rad/s |
| Joint 3 | -π do π | 2.5 rad/s |
| Joint 4 | 0 do π | 2.5 rad/s |
| Joint 5 | -π do π | 3.0 rad/s |
| Joint 6 | -π/2 do π/2 | 3.0 rad/s |
| Joint 7 | -π do π | 3.0 rad/s |
| Gripper | 0.0 do 1.0 | 0.5 units/s |

### System Wizyjny

**Kamera główna:**
- Rozdzielczość: 1280x720 pikseli (720p)
- FPS: 30 frames per second
- Typ: RGB (color)
- Położenie: Mounted on head/torso (skierowana do workspace)
- FOV (Field of View): ~60-80 stopni

**Opcjonalne kamery:**
- Kamery wrist-mounted (na nadgarstkach) - dla precyzyjnych zadań
- Depth camera (opcjonalnie) - dla percepcji 3D

---

## 🏗️ Architektura Systemu

### Kompletny Stack Technologiczny

```
┌─────────────────────────────────────────────────────────────────┐
│                        WARSTWA APLIKACJI                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  UnifoLM-VLA Model (Inferencja)                           │  │
│  │  • Vision-Language Understanding (Qwen2.5-VL)             │  │
│  │  • Action Prediction (DiT)                                │  │
│  │  • Flow Matching                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Predicted Actions
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    WARSTWA KOMUNIKACJI                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Model Server (FastAPI/gRPC)                              │  │
│  │  • Przyjmowanie obserwacji od klienta                     │  │
│  │  • Uruchomienie inferencji modelu                         │  │
│  │  • Zwracanie przewidzianych akcji                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↕                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Robot Client                                             │  │
│  │  • Zbieranie obserwacji (kamera, stato stawów)            │  │
│  │  • Wysyłanie do serwera                                   │  │
│  │  • Odbieranie i wykonanie akcji                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Joint Commands
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      WARSTWA KONTROLERA                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Unitree Controller SDK                                   │  │
│  │  • Low-level motor control                                │  │
│  │  • Safety limits enforcement                              │  │
│  │  • Joint trajectory execution                             │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Motor Commands
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                          HARDWARE                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │  Kamera    │  │  Encodery  │  │  Silniki   │                │
│  │  RGB       │  │  Stawów    │  │  Stawów    │                │
│  └────────────┘  └────────────┘  └────────────┘                │
│         ROBOT UNITREE G1 EDU-U6                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Przepływ Danych w Czasie Rzeczywistym

```
Cykl kontrolny (control loop) - 10Hz (100ms per cycle):

1. KROK: Odczyt Sensorów [~10ms]
   ├─ Kamera → Obraz RGB 1280x720
   ├─ Encodery → Pozycje 7 stawów
   └─ Force sensors → Siły kontaktu (opcjonalnie)

2. KROK: Preprocessing [~5ms]
   ├─ Resize obrazu jeśli potrzebne
   ├─ Konwersja do tensora
   └─ Normalizacja proprio

3. KROK: Komunikacja z Serwerem [~20ms]
   ├─ Serializacja danych (JSON/Protocol Buffers)
   ├─ Wysłanie przez sieć (LAN/Localhost)
   └─ Oczekiwanie na odpowiedź

4. KROK: Inferencja Modelu (Server-side) [~40ms]
   ├─ VLM forward pass → scene features
   ├─ DiT forward pass → action predictions
   └─ Denormalizacja akcji

5. KROK: Odbior i Wykonanie [~20ms]
   ├─ Otrzymanie predicted actions
   ├─ Wysłanie do kontrolera
   └─ Interpolacja trajektorii

6. KROK: Safety Checks [~5ms]
   ├─ Collision detection
   ├─ Joint limit checking
   └─ Emergency stop ready

TOTAL: ~100ms → 10Hz control frequency
```

---

## 🎮 Pierwszy Kontakt z Robotem

### Przygotowanie Stanowiska Robota

**Lista kontrolna przed uruchomieniem:**

```
□ Workspace:
  □ Stół roboczy oczyszczony (min 1m x 1m)
  □ Brak przeszkód w zasięgu robota
  □ Mocowanie robota stabilne
  □ Podłoże równe i antypoślizgowe

□ Zasilanie:
  □ Bateria naładowana (>50%)
  □ Lub zasilacz podłączony i działający
  □ Wyświetlacz status LED: zielony

□ Sieć:
  □ Robot podłączony do sieci LAN
  □ IP robota known: ____.____.____.____
  □ Ping do robota działa: ping <ROBOT_IP>

□ Bezpieczeństwo:
  □ Emergency stop button accessible
  □ Operator przeszkolony
  □ Okulary ochronne (jeśli manipulacja z narzędziami)
  □ Strefa bezpieczeństwa oznaczona
```

### Pierwsze Uruchomienie

#### Krok 1: Połączenie z Robotem

```bash
# Terminal 1: SSH do robota
ssh unitree@<ROBOT_IP>
# Hasło: (podane przez administratora)

# Sprawdź status robota
unitree-status
# Output powinien pokazać:
# - Battery: XX%
# - Motors: All OK
# - Sensors: All OK
# - Temperature: Normal
```

#### Krok 2: Uruchomienie Kontrolera Bazowego

```bash
# Na robocie (przez SSH)
cd ~/unitree_controller
./start_controller.sh

# Logi powinny pokazać:
# [INFO] Controller initialized
# [INFO] All joints calibrated
# [INFO] Ready to receive commands
```

#### Krok 3: Test Podstawowych Ruchów

```python
# test_robot_basic.py
# Uruchom na swoim komputerze (nie na robocie!)

from unitree_deploy import RobotClient

# Połącz z robotem
robot = RobotClient(ip="<ROBOT_IP>", port=8080)

# Sprawdź połączenie
status = robot.get_status()
print(f"Robot ready: {status['ready']}")
print(f"Joint positions: {status['joint_positions']}")

# Test 1: Pozycja zero (home position)
print("Moving to home position...")
robot.move_to_home()  # Bezpieczna pozycja startowa
time.sleep(3)

# Test 2: Otwarcie chwytaka
print("Opening gripper...")
robot.set_gripper(1.0)  # 1.0 = fully open
time.sleep(2)

# Test 3: Zamknięcie chwytaka
print("Closing gripper...")
robot.set_gripper(0.0)  # 0.0 = fully closed
time.sleep(2)

# Test 4: Prosty ruch ramienia (delta move)
print("Small arm movement...")
robot.move_joint_delta(joint_idx=2, delta=0.1)  # Ruch o 0.1 rad
time.sleep(2)

# Powrót do home
robot.move_to_home()
print("Test completed!")
```

**Oczekiwane zachowanie:**
- Robot płynnie przesuwa się do pozycji home
- Chwytak otwiera i zamyka się bez problemów
- Ramię wykonuje mały ruch i wraca
- Brak alarm sounds lub emergency stops

⚠️ **Jeśli coś poszło nie tak:**
- Naciśnij EMERGENCY STOP
- Sprawdź logi kontrolera
- Sprawdź czy robot nie ma mechanicznych blokad

---

## 📊 Zbieranie Danych Treningowych

### Metody Zbierania Demonstracji

#### Metoda 1: Kinesthetic Teaching (Fizyczne Prowadzenie)

**Najlepsza dla:** Zadania wymagające precyzji i intuicyjnych ruchów

```python
# collect_kinesthetic.py

from unitree_deploy import KinestheticRecorder

# Inicjalizacja recordera
recorder = KinestheticRecorder(
    robot_ip="<ROBOT_IP>",
    save_path="/data/demonstrations/task_name"
)

# Przełącz robot w tryb gravity compensation
# (silniki nie trzymają pozycji, można swobodnie poruszać)
recorder.enable_gravity_compensation()

print("=" * 50)
print("KINESTHETIC TEACHING MODE")
print("=" * 50)
print("Robot is now in gravity compensation mode.")
print("You can freely move the robot arms.")
print()
print("Press ENTER to start recording...")
input()

# Start nagrywania
recorder.start_recording()
print("🔴 RECORDING STARTED")
print("Perform the task demonstration...")
print("Press ENTER when done.")

# Nagrywanie w tle (zbiera joint states + kamera)
input()

# Stop nagrywania
recorder.stop_recording()
print("✅ Recording saved!")

# Odzyskaj control (motors hold position)
recorder.disable_gravity_compensation()

# Metadata
metadata = {
    'task_name': 'stack_red_cube',
    'task_description': 'Pick red cube and place on blue platform',
    'operator': 'Jan Kowalski',
    'success': True,  # Czy demonstracja udana?
}
recorder.save_metadata(metadata)

print(f"Demonstration saved to: {recorder.get_last_recording_path()}")
```

**Struktura nagranego pliku:**

```
/data/demonstrations/task_name/
├── episode_001.hdf5
│   ├── /observation
│   │   ├── /image              # [T, H, W, 3] uint8
│   │   └── /state              # [T, 15] float32 (7 pos + 7 vel + 1 gripper)
│   ├── /action                 # [T, 8] float32 (7 joints + 1 gripper)
│   └── /metadata
│       ├── task_name: "stack_red_cube"
│       ├── timestamp: "2026-02-10T14:30:00"
│       └── success: True
└── episode_002.hdf5
    └── ...
```

#### Metoda 2: Teleoperation (Zdalne Sterowanie)

**Najlepsza dla:** Zadania wymagające szybkich reakcji lub z przeszkodami

```python
# collect_teleoperation.py

from unitree_deploy import TeleoperationInterface
import pygame  # For joystick/keyboard control

# Inicjalizacja interfejsu
teleop = TeleoperationInterface(
    robot_ip="<ROBOT_IP>",
    control_mode="joystick",  # lub "keyboard", "vr_controller"
    save_path="/data/demonstrations/task_name"
)

# Konfiguracja mappingu (które przyciski co robią)
teleop.configure_controls({
    'left_stick_x': 'joint_1',     # Lewy stick: rotacja ramienia
    'left_stick_y': 'joint_2',     # Podnoszenie/opuszczanie
    'right_stick_x': 'joint_5',    # Nadgarstek roll
    'right_stick_y': 'joint_6',    # Nadgarstek pitch
    'trigger_right': 'gripper_close',
    'trigger_left': 'gripper_open',
    'button_A': 'record_start',
    'button_B': 'record_stop',
})

print("Teleoperation Interface Ready!")
print("Controls:")
print("  Left Stick: Arm position")
print("  Right Stick: Wrist orientation")
print("  Triggers: Gripper open/close")
print("  Button A: Start recording")
print("  Button B: Stop recording")
print()
print("Move robot to starting position and press A...")

# Main control loop
running = True
while running:
    # Odczyt kontrolera
    commands = teleop.read_controller()
    
    # Wyślij komendy do robota
    teleop.send_commands(commands)
    
    # Jeśli nagrywanie aktywne, zapisuj dane
    if teleop.is_recording():
        teleop.record_step()
    
    # Visualization (opcjonalnie)
    teleop.render_camera_view()  # Pokazuje co widzi robot
    
    # Exit condition
    if commands.get('button_X'):
        running = False

teleop.cleanup()
print("Teleoperation session ended.")
```

### Best Practices dla Demonstracji

**Jakość demonstracji > Ilość demonstracji**

✅ **Dobra demonstracja:**
- Płynne ruchy (bez gwałtownych skoków)
- Zakończona sukcesem
- Różnorodne pozycje początkowe obiektów
- Czyste tło (minimalne distractors)
- Dobre oświetlenie

❌ **Zła demonstracja:**
- Robot uderza w obiekty
- Zadanie niekompletne
- Obiekt upuszczony
- Zła kalibracja kamery (obraz rozmazany)

**Ile demonstracji potrzeba?**

| Typ zadania | Minimalne | Zalecane | Optymalne |
|-------------|-----------|----------|-----------|
| **Proste pick-and-place** | 50 | 100-200 | 500+ |
| **Manipulacja z orientacją** | 100 | 200-400 | 1000+ |
| **Zadania wieloetapowe** | 200 | 500-1000 | 2000+ |
| **Zadania z precyzją** | 300 | 1000+ | 3000+ |

**Strategie zbierania:**

```python
# collect_diverse_demonstrations.py

# Strategia 1: Randomizacja pozycji obiektów
object_positions = generate_random_positions(
    num_positions=100,
    workspace_bounds=[(0.3, 0.7), (-0.3, 0.3), (0.0, 0.2)]
)

for pos in object_positions:
    place_object_at(pos)
    print(f"Demonstrate task with object at {pos}")
    input("Press ENTER when ready...")
    recorder.record_demonstration()

# Strategia 2: Różne orientacje obiektów
orientations = [0, 45, 90, 135, 180, 225, 270, 315]  # stopnie
for angle in orientations:
    rotate_object_to(angle)
    recorder.record_demonstration()

# Strategia 3: Różne warunki oświetlenia
lighting_conditions = ['normal', 'bright', 'dim', 'side_light']
for lighting in lighting_conditions:
    set_lighting(lighting)
    # Zbierz 10-20 demonstracji w każdych warunkach
    for _ in range(20):
        recorder.record_demonstration()
```

---

## 🔄 Pipeline: Od Demonstracji do Wytrenowanego Modelu

### Kompletny Workflow

```
┌────────────────────────────────────────────────────────────┐
│  FAZA 1: Zbieranie Danych (1-2 tygodnie)                   │
├────────────────────────────────────────────────────────────┤
│  • Zbierz 200-500 demonstracji zadania                     │
│  • Każda demonstracja: kinesthetic/teleoperation           │
│  • Output: *.hdf5 files w formacie LeRobot                │
│                                                            │
│  Lokalizacja: /data/raw_demonstrations/task_name/         │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  FAZA 2: Konwersja Danych (1 dzień)                        │
├────────────────────────────────────────────────────────────┤
│  Krok 2.1: LeRobot → HDF5 (unified format)                │
│    cd prepare_data                                         │
│    python convert_lerobot_to_hdf5.py \                    │
│      --data_path /data/raw_demonstrations/task_name \     │
│      --target_path /data/processed/task_name_hdf5         │
│                                                            │
│  Krok 2.2: HDF5 → RLDS (TensorFlow Datasets)              │
│    cd prepare_data/hdf5_to_rlds/rlds_dataset              │
│    # Edit rlds_dataset.py: update HDF5 path               │
│    tfds build --data_dir /data/processed/task_name_rlds   │
│                                                            │
│  Output: /data/processed/task_name_rlds/1.0.0/            │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  FAZA 3: Rejestracja Datasetu (30 min)                    │
├────────────────────────────────────────────────────────────┤
│  Edytuj 4 pliki:                                           │
│                                                            │
│  1. src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/    │
│     configs.py:                                            │
│       'my_task': DatasetConfig(...)                       │
│                                                            │
│  2. transforms.py:                                         │
│       def my_task_transform(trajectory): ...              │
│                                                            │
│  3. mixtures.py:                                           │
│       'my_mixture': {'my_task': 1.0}                      │
│                                                            │
│  4. datasets.py:                                           │
│       register_dataset('my_task', ...)                    │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  FAZA 4: Konfiguracja Treningu (1 godzina)                │
├────────────────────────────────────────────────────────────┤
│  Edytuj scripts/run_scripts/run_unifolm_vla_train.sh:    │
│                                                            │
│  # Model initialization                                   │
│  base_vlm="unitreerobotics/Unifolm-VLM-Base"             │
│                                                            │
│  # Dataset                                                 │
│  oxe_data_root="/data/processed"                          │
│  data_mix="my_task"                                       │
│                                                            │
│  # Training hyperparams                                    │
│  batch_size=8                                              │
│  learning_rate=1e-4                                        │
│  num_train_steps=10000                                     │
│                                                            │
│  # Hardware                                                │
│  num_processes=2  # Number of GPUs                        │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  FAZA 5: Trenowanie (2-7 dni, w zależności od GPU)        │
├────────────────────────────────────────────────────────────┤
│  bash scripts/run_scripts/run_unifolm_vla_train.sh       │
│                                                            │
│  Monitoring:                                               │
│  • TensorBoard: tensorboard --logdir logs/                │
│  • Wandb: https://wandb.ai/your-project                   │
│                                                            │
│  Checkpoints zapisywane co 1000 kroków:                   │
│  checkpoints/                                              │
│  ├── checkpoint-1000.pth                                   │
│  ├── checkpoint-2000.pth                                   │
│  └── ...                                                   │
│  └── checkpoint-10000.pth  ← Final model                  │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  FAZA 6: Ewaluacja w Symulacji (1-2 dni)                  │
├────────────────────────────────────────────────────────────┤
│  • Test w LIBERO (jeśli applicable)                       │
│  • Lub custom simulation environment                       │
│  • Metryka: Success rate na 50-100 epizodach              │
│                                                            │
│  Jeśli success rate < 70%:                                 │
│    → Zbierz więcej danych                                  │
│    → Sprawdź preprocessing                                 │
│    → Dostosuj hyperparametry                               │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  FAZA 7: Deployment na Prawdziwym Robocie                 │
├────────────────────────────────────────────────────────────┤
│  1. Uruchom serwer inferencji                              │
│  2. Połącz klienta robota                                  │
│  3. Testuj ostrożnie (start with safety limits)           │
│  4. Iteruj: zbieraj failure cases → fine-tune             │
└────────────────────────────────────────────────────────────┘
```

### Automatyzacja Pipeline'u

Przykładowy skrypt automatyzujący cały proces:

```bash
#!/bin/bash
# full_pipeline.sh - Kompletny pipeline od raw data do trained model

set -e  # Exit on error

TASK_NAME="g1_my_new_task"
RAW_DATA_DIR="/data/raw_demonstrations/${TASK_NAME}"
PROCESSED_DIR="/data/processed/${TASK_NAME}"

echo "=========================================="
echo "  UNITREE G1 - TRAINING PIPELINE"
echo "  Task: ${TASK_NAME}"
echo "=========================================="

# FAZA 1: Sprawdź czy dane są zebrane
echo "[1/6] Checking raw data..."
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "ERROR: Raw data not found at $RAW_DATA_DIR"
    echo "Please collect demonstrations first!"
    exit 1
fi

NUM_EPISODES=$(ls -1 ${RAW_DATA_DIR}/*.hdf5 | wc -l)
echo "Found $NUM_EPISODES episodes"

if [ $NUM_EPISODES -lt 50 ]; then
    echo "WARNING: Only $NUM_EPISODES episodes. Recommended: 100+"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# FAZA 2: Konwersja LeRobot → HDF5
echo "[2/6] Converting LeRobot to HDF5..."
python prepare_data/convert_lerobot_to_hdf5.py \
    --data_path ${RAW_DATA_DIR} \
    --target_path ${PROCESSED_DIR}_hdf5

# FAZA 3: Konwersja HDF5 → RLDS
echo "[3/6] Converting HDF5 to RLDS..."
cd prepare_data/hdf5_to_rlds/rlds_dataset

# Aktualizuj ścieżkę w rlds_dataset.py
sed -i "s|/path/to/hdf5/data|${PROCESSED_DIR}_hdf5|g" rlds_dataset.py

tfds build --data_dir ${PROCESSED_DIR}_rlds
cd -

# FAZA 4: Rejestracja datasetu (manual - pomiń na razie)
echo "[4/6] Dataset registration (manual step - skipping)"
echo "TODO: Edit configs.py, transforms.py, mixtures.py, datasets.py"
read -p "Press ENTER when done..."

# FAZA 5: Trenowanie
echo "[5/6] Starting training..."
bash scripts/run_scripts/run_unifolm_vla_train.sh

# FAZA 6: Ewaluacja
echo "[6/6] Evaluation..."
echo "Checkpoints saved in: checkpoints/"
echo "To evaluate, run:"
echo "  bash scripts/eval_scripts/run_eval_libero.sh"
echo "  (or deploy on real robot)"

echo "=========================================="
echo "  PIPELINE COMPLETED!"
echo "=========================================="
```

---

## 🚀 Deployment na Prawdziwym Robocie

### Krok 1: Przygotowanie Serwera Inferencji

```bash
# Terminal 1: Serwer (komputer z GPU)
conda activate unifolm-vla
cd /path/to/unifolm-vla

# Edytuj konfigurację serwera
nano scripts/eval_scripts/run_real_eval_server.sh

# Kluczowe parametry:
# ckpt_path="/path/to/your/checkpoint-10000.pth"
# port=8000
# unnorm_key="g1_my_task"
# vlm_pretrained_path="unitreerobotics/Unifolm-VLM-Base"

# Uruchom serwer
bash scripts/eval_scripts/run_real_eval_server.sh

# Output:
# [INFO] Loading model from checkpoint-10000.pth...
# [INFO] Model loaded successfully
# [INFO] Server started on http://0.0.0.0:8000
# [INFO] Waiting for connections...
```

### Krok 2: Konfiguracja Klienta Robota

```bash
# Terminal 2: Klient (może być ten sam komputer lub raspberry pi na robocie)

# Jeśli serwer jest na innym komputerze, utwórz tunel SSH
ssh -L 8000:localhost:8000 user@server_ip -CNg
# To przekierowuje port 8000 serwera na lokalny port 8000

# Sprawdź połączenie
curl http://localhost:8000/health
# Output: {"status": "ok", "model": "loaded"}
```

```python
# robot_client_realtime.py
# Klient do komunikacji z serwerem i robotem

import numpy as np
import cv2
import time
import requests
from unitree_deploy import RobotController

class RealtimeVLAClient:
    def __init__(self, 
                 robot_ip, 
                 server_url="http://localhost:8000",
                 control_frequency=10):  # Hz
        
        self.robot = RobotController(robot_ip)
        self.server_url = server_url
        self.dt = 1.0 / control_frequency
        
        # Action buffer (dla action chunking)
        self.action_buffer = []
        
    def run_task(self, instruction, max_steps=1000):
        """
        Uruchom zadanie na prawdziwym robocie
        
        Args:
            instruction: Instrukcja tekstowa (np. "pick up the red cube")
            max_steps: Maksymalna liczba kroków
        """
        print(f"Starting task: {instruction}")
        print(f"Max steps: {max_steps}")
        
        # Przenieś robota do pozycji startowej
        self.robot.move_to_home()
        time.sleep(2)
        
        step = 0
        while step < max_steps:
            step_start_time = time.time()
            
            # 1. Zbierz obserwacje
            observation = self.get_observation()
            
            # 2. Jeśli action buffer pusty, zapytaj serwer
            if len(self.action_buffer) == 0:
                self.query_server(observation, instruction)
            
            # 3. Wykonaj pierwszą akcję z bufora
            if len(self.action_buffer) > 0:
                action = self.action_buffer.pop(0)
                self.execute_action(action)
            else:
                print("WARNING: No actions in buffer!")
                break
            
            # 4. Sprawdź warunek zakończenia (opcjonalnie)
            if self.check_termination_condition():
                print(f"Task completed at step {step}")
                break
            
            # 5. Utrzymuj stałą częstotliwość
            elapsed = time.time() - step_start_time
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
            
            step += 1
            
            # Logging
            if step % 10 == 0:
                print(f"Step {step}/{max_steps}")
        
        # Powrót do home
        self.robot.move_to_home()
        print("Task execution finished")
    
    def get_observation(self):
        """Zbierz obserwacje z robota"""
        # Obraz z kamery
        image = self.robot.get_camera_image()  # numpy array [H, W, 3]
        
        # Stan proprioceptywny
        joint_pos = self.robot.get_joint_positions()  # [7]
        joint_vel = self.robot.get_joint_velocities()  # [7]
        gripper_pos = self.robot.get_gripper_position()  # scalar
        
        proprio = np.concatenate([joint_pos, joint_vel, [gripper_pos]])
        
        return {
            'image': image,
            'proprio': proprio,
        }
    
    def query_server(self, observation, instruction):
        """Zapytaj serwer o predykcję akcji"""
        # Przygotuj dane
        payload = {
            'image': observation['image'].tolist(),  # Konwersja numpy → list
            'proprio': observation['proprio'].tolist(),
            'instruction': instruction,
        }
        
        # Wyślij request
        try:
            response = requests.post(
                f"{self.server_url}/predict",
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()
            
            # Odbierz predicted actions (action chunk)
            result = response.json()
            predicted_actions = np.array(result['actions'])  # [num_chunks, 8]
            
            # Zapisz do bufora
            self.action_buffer = list(predicted_actions)
            
            print(f"Received {len(self.action_buffer)} actions from server")
            
        except Exception as e:
            print(f"ERROR querying server: {e}")
            # W przypadku błędu, zatrzymaj robota
            self.action_buffer = []
    
    def execute_action(self, action):
        """Wykonaj akcję na robocie"""
        # action = [joint1, joint2, ..., joint7, gripper]
        joint_positions = action[:7]
        gripper_position = action[7]
        
        # Wyślij komendy do robota
        self.robot.set_joint_positions(joint_positions)
        self.robot.set_gripper_position(gripper_position)
    
    def check_termination_condition(self):
        """
        Sprawdź czy zadanie zakończone
        (opcjonalnie - możesz zaimplementować task-specific logic)
        """
        # Przykład: jeśli chwytak zamknięty i ramię w górze
        # → obiekt złapany i podniesiony
        
        # gripper_pos = self.robot.get_gripper_position()
        # joint_pos = self.robot.get_joint_positions()
        # 
        # if gripper_pos < 0.3 and joint_pos[2] > 0.5:
        #     return True
        
        return False  # Nigdy nie kończymy (rely on max_steps)


# Główny skrypt uruchomieniowy
if __name__ == "__main__":
    # Inicjalizacja klienta
    client = RealtimeVLAClient(
        robot_ip="192.168.1.100",  # IP robota G1
        server_url="http://localhost:8000",
        control_frequency=10  # 10 Hz
    )
    
    # Definicja zadania
    instruction = "pick up the red cube and place it on the blue platform"
    
    # Uruchomienie
    try:
        client.run_task(instruction, max_steps=500)
    except KeyboardInterrupt:
        print("\nTask interrupted by user")
        client.robot.emergency_stop()
    except Exception as e:
        print(f"\nERROR: {e}")
        client.robot.emergency_stop()
```

### Krok 3: Pierwsze Testy

**Procedura bezpiecznego testowania:**

```
1. Test w powietrzu (no object interaction):
   □ Robot w home position
   □ Daj instrukcję "move arm forward slightly"
   □ Obserwuj czy ruch jest płynny i sensowny
   □ Emergency stop ready!

2. Test z pustym workspace:
   □ Workspace pusty (brak obiektów)
   □ Daj instrukcję pick-and-place
   □ Robot powinien wykonać charakterystyczny ruch
     (reach, close gripper, retract)
   □ Sprawdź czy nie uderza w stół

3. Test z obiektem (soft object first!):
   □ Umieść miękki obiekt (np. gąbka, pluszak)
   □ Wykonaj pełne zadanie
   □ Sprawdź success rate

4. Test z docelowym obiektem:
   □ Użyj właściwych obiektów z demonstracji
   □ Wykonaj 10-20 prób
   □ Zapisuj success/failure cases

5. Stress testing:
   □ Różne pozycje obiektów
   □ Różne oświetlenie
   □ Distractors na stole
```

---

## 🛡️ Bezpieczeństwo i Best Practices

### Zasady Bezpieczeństwa

**❗ ZAWSZE:**
- Miej przycisk EMERGENCY STOP w zasięgu ręki
- Trzymaj bezpieczną odległość od robota (min 1m)
- Testuj nowe zachowania najpierw w symulacji
- Używaj software safety limits (joint ranges, velocities)
- Miej plan awaryjny (co zrobić gdy robot źle działa)

**❌ NIGDY:**
- Nie wkładaj rąk/głowy do workspace podczas ruchu robota
- Nie blokuj fizycznie ruchu robota (może uszkodzić motory)
- Nie uruchamiaj nieznanego kodu bez przeglądu
- Nie zostawiaj robota bez nadzoru podczas autonomicznego działania

### Software Safety Limits

```python
# safety_config.py
# Konfiguracja limitów bezpieczeństwa

SAFETY_CONFIG = {
    # Limity pozycji stawów (rad)
    'joint_position_limits': {
        'min': [-3.14, -1.57, -3.14, 0.0, -3.14, -1.57, -3.14],
        'max': [3.14, 1.57, 3.14, 3.14, 3.14, 1.57, 3.14],
    },
    
    # Maksymalne prędkości (rad/s)
    'joint_velocity_limits': [2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 3.0],
    
    # Maksymalna siła kontaktu (N)
    # Jeśli wykryjesz większą siłę → emergency stop
    'max_contact_force': 50.0,
    
    # Workspace limits (kartezjańskie, metry)
    'workspace_limits': {
        'x': [0.2, 0.8],   # Do przodu
        'y': [-0.4, 0.4],  # Na boki
        'z': [0.0, 0.6],   # Wysokość
    },
    
    # Timeout dla pojedynczego ruchu
    'action_timeout': 5.0,  # sekundy
}

class SafetyController:
    """Wrapper dodający safety checks do kontrolera robota"""
    
    def __init__(self, robot, safety_config):
        self.robot = robot
        self.config = safety_config
    
    def safe_set_joint_positions(self, positions):
        """Ustaw pozycje stawów z safety checks"""
        # Check 1: Pozycje w dozwolonym zakresie
        for i, (pos, min_pos, max_pos) in enumerate(zip(
            positions,
            self.config['joint_position_limits']['min'],
            self.config['joint_position_limits']['max']
        )):
            if not (min_pos <= pos <= max_pos):
                print(f"SAFETY: Joint {i} position {pos} out of range [{min_pos}, {max_pos}]")
                return False
        
        # Check 2: Sprawdź czy ruch nie jest zbyt duży (nagły skok)
        current = self.robot.get_joint_positions()
        max_delta = 0.5  # rad
        for i, (new_pos, cur_pos) in enumerate(zip(positions, current)):
            if abs(new_pos - cur_pos) > max_delta:
                print(f"SAFETY: Joint {i} delta too large: {abs(new_pos - cur_pos)} > {max_delta}")
                return False
        
        # Check 3: End-effector w workspace?
        ee_position = self.robot.forward_kinematics(positions)
        if not self.is_in_workspace(ee_position):
            print(f"SAFETY: End-effector {ee_position} outside workspace")
            return False
        
        # All checks passed → execute
        self.robot.set_joint_positions(positions)
        return True
    
    def is_in_workspace(self, position):
        """Sprawdź czy pozycja kartezjańska w workspace"""
        x, y, z = position
        limits = self.config['workspace_limits']
        
        return (limits['x'][0] <= x <= limits['x'][1] and
                limits['y'][0] <= y <= limits['y'][1] and
                limits['z'][0] <= z <= limits['z'][1])
```

### Hardware Maintenance

**Cotygodniowa konserwacja:**
```
□ Sprawdź stan baterii (cycles, degradacja)
□ Oczyść kamery (delikatną szmatką)
□ Sprawdź luzy w stawach (czy silniki dobrze trzymają)
□ Sprawdź kable (czy nie są przecierane)
□ Soft reset kontrolera (reboot)
```

**Comiesięczna konserwacja:**
```
□ Kalibracja kamer (jeśli drift)
□ Aktualizacja firmware (jeśli dostępne)
□ Backup danych i konfiguracji
□ Sprawdzenie momentów obrotowych silników
```

---

## 🎯 Przykładowe Projekty

### Projekt 1: Sortowanie Kolorowych Obiektów

**Zadanie:** Robot sortuje klocki według kolorów do odpowiednich pojemników

**Setup:**
- 3 kolory klocków (czerwony, niebieski, zielony)
- 3 pojemniki oznaczone kolorami
- Początkowa pozycja: klocki rozsypane na stole

**Demonstracje:**
```python
# Zbierz ~150 demonstracji:
# - 50 demonstracji: sortowanie czerwonych
# - 50 demonstracji: sortowanie niebieskich
# - 50 demonstracji: sortowanie zielonych

instructions = [
    "sort the blocks by color",
    "put red blocks in the red container",
    "organize the colored blocks",
]
```

**Wyzwania:**
- Detekcja koloru w różnym oświetleniu
- Precyzyjne umieszczanie w pojemnikach
- Unikanie kolizji z już ułożonymi klockami

**Metryka sukcesu:**
- 100% klocków w odpowiednich pojemnikach
- Czas wykonania < 60 sekund dla 9 klocków

### Projekt 2: Przygotowanie Prostego Posiłku

**Zadanie:** Robot przygotowuje kanapkę (przykład złożonego zadania wieloetapowego)

**Etapy:**
1. Weź kromkę chleba z opakowania
2. Połóż chleb na desce
3. Weź nóż
4. Weź słoik z masłem orzechowym
5. Otwórz słoik (twist-off)
6. Nanieś masło na chleb
7. Odłóż nóż
8. Zamknij słoik
9. Przykryj drugą kromką

**Demonstracje:**
- ~500 demonstracji pełnego zadania
- Lub 100 demonstracji na każdy sub-task (modular approach)

**Wyzwania:**
- Długi horyzont czasowy
- Manipulacja deformowalnymi obiektami (chleb)
- Precyzyjne ruchy (smarowanie)
- Multi-object interaction

### Projekt 3: Współpraca Dwóch Robotów

**Zadanie:** Dwa roboty G1 współpracują przy czyszczeniu stołu

**Roles:**
- Robot A: Zbiera śmieci do worka
- Robot B: Trzyma worek otwarty

**Demonstracje:**
- Zbierz demonstracje z perspektywy każdego robota
- Synchronizuj timestamp'y
- Use dataset: G1_DualRobot_Clean_Table

**Implementacja:**
```python
# Dual robot client

class DualRobotVLAClient:
    def __init__(self):
        self.robot_a = RealtimeVLAClient("192.168.1.100")
        self.robot_b = RealtimeVLAClient("192.168.1.101")
    
    def run_collaborative_task(self):
        instruction_a = "pick up trash and put in the bag"
        instruction_b = "hold the bag open"
        
        # Run both robots simultaneously (multi-threading)
        thread_a = threading.Thread(
            target=self.robot_a.run_task,
            args=(instruction_a,)
        )
        thread_b = threading.Thread(
            target=self.robot_b.run_task,
            args=(instruction_b,)
        )
        
        thread_a.start()
        thread_b.start()
        
        thread_a.join()
        thread_b.join()
```

---

## 📊 Oczekiwane Wyniki i Benchmarki

### Typowe Success Rates

**Dla zadań z Unitree G1:**

| Zadanie | Oczekiwany Success Rate | Komentarz |
|---------|-------------------------|-----------|
| **Stack Block** (proste) | 85-95% | Zadanie benchmark |
| **Bag Insert** | 75-85% | Wymaga precyzji orientacji |
| **Erase Board** | 80-90% | Wymaga coverage planning |
| **Pour Medicine** | 60-75% | Bardzo precyzyjne |
| **Fold Towel** | 50-70% | Deformowalne obiekty, trudne |

### Troubleshooting Niskiej Success Rate

```python
# Analiza failure cases

failures = collect_failure_cases(num_attempts=100)

# Kategoryzacja błędów
error_types = {
    'perception': 0,      # Robot nie widzi obiektu
    'grasp_failure': 0,   # Nie złapał obiektu
    'collision': 0,       # Uderzył w coś
    'dropped': 0,         # Upuścił obiekt
    'misplacement': 0,    # Złe miejsce docelowe
    'timeout': 0,         # Zadanie trwało zbyt długo
}

for failure in failures:
    error_type = categorize_error(failure)
    error_types[error_type] += 1

# Analiza
print("Error Analysis:")
for error, count in error_types.items():
    pct = 100 * count / len(failures)
    print(f"  {error}: {count} ({pct:.1f}%)")

# Action plan based on most common error:
if error_types['perception'] > 30:
    print("→ Improve lighting or add more camera angles")
elif error_types['grasp_failure'] > 30:
    print("→ Collect more grasp demonstrations")
elif error_types['collision'] > 30:
    print("→ Add collision avoidance or adjust workspace")
```

---

## 🎓 Podsumowanie

**Kluczowe punkty dla pracy z Unitree G1:**

1. **Bezpieczeństwo przede wszystkim** - zawsze miej emergency stop gotowy
2. **Jakość demonstracji** - lepiej 100 dobrych niż 500 złych
3. **Iteracyjny proces** - zbierz dane → trenuj → testuj → popraw
4. **Monitoring** - używaj TensorBoard/Wandb do śledzenia treningu
5. **Start simple** - zacznij od prostych zadań, potem zwiększaj złożoność

**Typowy timeline projektu:**
- **Tydzień 1-2:** Setup, pierwsze demonstracje, konwersja danych
- **Tydzień 3:** Trening pierwszego modelu
- **Tydzień 4:** Ewaluacja, iteracja
- **Tydzień 5-6:** Deployment, testy na robocie
- **Tydzień 7-8:** Fine-tuning, dokumentacja

---

**Powodzenia w pracy z robotem Unitree G1! 🤖🚀**

*Przewodnik opracowany dla studentów Politechniki Rzeszowskiej, 2026*

**Dodatkowe zasoby:**
- README_pl.md - Główna dokumentacja po polsku
- GUIDE_FOR_STUDENTS_PL.md - Szczegółowy przewodnik frameworku
- [Oficjalna dokumentacja Unitree G1](https://www.unitree.com/g1)
