# 📚 Przewodnik dla Studentów: UnifoLM-VLA Framework

## Politechnika Rzeszowska - Projekt z Robotem Humanoidalnym Unitree G1 EDU-U6

---

## 📑 Spis Treści

1. [Wprowadzenie i Motywacja](#wprowadzenie-i-motywacja)
2. [Podstawowe Pojęcia](#podstawowe-pojęcia)
3. [Jak Działa UnifoLM-VLA](#jak-działa-unifolm-vla)
4. [Struktura Projektu Krok po Kroku](#struktura-projektu-krok-po-kroku)
5. [Praktyczne Ćwiczenia](#praktyczne-ćwiczenia)
6. [Troubleshooting - Najczęstsze Problemy](#troubleshooting)
7. [Zaawansowane Tematy](#zaawansowane-tematy)

---

## 🎯 Wprowadzenie i Motywacja

### Dlaczego ten projekt jest ważny?

Tradycyjne podejścia do programowania robotów wymagały:
- Ręcznego pisania kodu dla każdego zadania
- Dokładnego planowania trajektorii
- Skomplikowanych systemów percepcji wizualnej

**UnifoLM-VLA** zmienia to podejście poprzez:
- **Uczenie z demonstracji**: Robot uczy się obserwując przykłady
- **Instrukcje w języku naturalnym**: "Połóż jabłko do pudełka"
- **Jedna polityka, wiele zadań**: Jeden model wykonuje różnorodne zadania

### Co osiągniecie w tym projekcie?

Po ukończeniu pracy z tym frameworkiem będziecie potrafili:

1. ✅ Zrozumieć architekturę modeli Vision-Language-Action
2. ✅ Przygotować dane treningowe z demonstracji robota
3. ✅ Trenować model VLA na własnych danych
4. ✅ Ewaluować model w symulacji i na prawdziwym robocie
5. ✅ Dostosować model do nowych zadań manipulacyjnych

---

## 📖 Podstawowe Pojęcia

### 1. Vision-Language-Action (VLA)

**VLA** to architektura łącząca trzy modalności:

```
┌─────────────┐
│   VISION    │  ← Obrazy z kamer robota (RGB, depth)
│  (Wizja)    │
└─────┬───────┘
      │
      ↓
┌─────────────┐
│  LANGUAGE   │  ← Instrukcje tekstowe ("połóż kubek na stole")
│   (Język)   │
└─────┬───────┘
      │
      ↓
┌─────────────┐
│   ACTION    │  → Konkretne ruchy robota (pozycje stawów, chwytak)
│   (Akcja)   │
└─────────────┘
```

**Dlaczego to działa?**
- Model uczy się **mapowania** od percepcji (wizja + język) do działania
- **Transformery** (architektura z mechanizmem attention) pozwalają na wielomodalną fuzję
- **Pre-trenowanie** na dużych zbiorach danych daje wiedzę ogólną

### 2. Kluczowe Komponenty Techniczne

#### A. Vision-Language Model (VLM)

To **"oczy i uszy"** systemu:
- Bazuje na **Qwen2.5-VL** (zaawansowany model multimodalny)
- Przetwarza obrazy i tekst równocześnie
- Generuje **embeddingi** (reprezentacje numeryczne) sceny

```python
# Pseudo-kod: VLM w akcji
image = camera.capture()              # Obraz z kamery
instruction = "pick up the red cube"  # Instrukcja

# VLM łączy oba wejścia
scene_embedding = vlm(image, instruction)
# scene_embedding zawiera semantyczne zrozumienie zadania
```

#### B. Action Model (DiT - Diffusion Transformer)

To **"mózg decyzyjny"** systemu:
- Używa **Diffusion Transformer** do przewidywania akcji
- Generuje sekwencje akcji (action chunks) zamiast pojedynczych kroków
- **Flow Matching**: nowoczesna technika generatywna

```python
# Pseudo-kod: Generowanie akcji
scene_embedding = vlm(image, instruction)

# DiT generuje sekwencję akcji
predicted_actions = dit(scene_embedding, proprio_state)
# predicted_actions = [action_t, action_t+1, ..., action_t+n]
```

**Dlaczego chunki akcji?**
- Pojedyncze kroki są niestabilne (drgania, wahania)
- Sekwencje dają gładkie, naturalne ruchy
- Zwiększona efektywność (mniej wywołań modelu)

#### C. Propriocepcja (Proprio)

**Czujniki wewnętrzne robota:**
- Pozycje stawów (joint positions)
- Prędkości stawów (joint velocities)
- Stan chwytaka (gripper state)

**Dlaczego to ważne?**
- Robot musi "czuć" swoje ciało
- Pozwala na zamkniętą pętlę kontroli
- Zapobiega kolizjom i niebezpiecznym ruchom

```python
# Przykład proprio_state dla robota G1
proprio_state = {
    'joint_positions': [0.1, -0.5, 0.3, ...],  # 7 stawów ramienia
    'joint_velocities': [0.0, 0.0, 0.0, ...],  
    'gripper_position': 0.8,  # 0.0 = zamknięty, 1.0 = otwarty
}
```

### 3. RLDS (Reinforcement Learning Datasets)

**Standard formatu danych** dla uczenia robotów:

```
Episode (Epizod) = Pełna demonstracja jednego zadania
    ├── Step 0: {image, action, state, reward}
    ├── Step 1: {image, action, state, reward}
    ├── ...
    └── Step N: {image, action, state, reward}
```

**Struktura kroku (step):**
```python
step = {
    'observation': {
        'image': np.array([720, 1280, 3]),  # Obraz RGB
        'state': np.array([14]),             # Stan stawów
    },
    'action': np.array([7]),                 # Akcja (pozycje docelowe)
    'reward': 1.0,                           # Nagroda (opcjonalnie)
    'is_terminal': False,                    # Czy koniec epizodu?
}
```

---

## 🔧 Jak Działa UnifoLM-VLA

### Pipeline Inferencji (Krok po Kroku)

Oto co się dzieje, gdy robot otrzymuje instrukcję:

```
┌─────────────────────────────────────────────────────────────┐
│  KROK 1: Pobranie Obserwacji                                │
├─────────────────────────────────────────────────────────────┤
│  • Kamera RGB: obraz sceny (1280x720 pikseli)              │
│  • Propriocepcja: obecne pozycje 7 stawów ramienia         │
│  • Instrukcja: "połóż czerwony kubek na niebieskim talerzu"│
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  KROK 2: Przetwarzanie przez VLM (Qwen2.5-VL)              │
├─────────────────────────────────────────────────────────────┤
│  • Tokenizacja obrazu (image patches)                       │
│  • Tokenizacja tekstu instrukcji                            │
│  • Self-attention między modalami                           │
│  • Output: embedding sceny (4096-wymiarowy wektor)          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  KROK 3: Generowanie Akcji przez DiT                        │
├─────────────────────────────────────────────────────────────┤
│  • Input: scene_embedding + proprio_state                    │
│  • Diffusion process (denoising)                             │
│  • Flow matching dla gładkich trajektorii                    │
│  • Output: chunk 16 akcji [a_t, a_t+1, ..., a_t+15]        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  KROK 4: Denormalizacja i Wykonanie                         │
├─────────────────────────────────────────────────────────────┤
│  • Denormalizacja akcji do rzeczywistych wartości           │
│  • Wysłanie akcji do kontrolera robota                      │
│  • Robot wykonuje pierwsze 1-2 akcje z chunka               │
│  • Pozostałe akcje są buforowane                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  KROK 5: Pętla Kontrolna                                    │
├─────────────────────────────────────────────────────────────┤
│  • Pobranie nowej obserwacji                                │
│  • Powrót do KROKU 1 (closed-loop control)                  │
│  • Kontynuacja aż zadanie zakończone                        │
└─────────────────────────────────────────────────────────────┘
```

### Proces Treningu (Training Pipeline)

```
┌────────────────────────────────────────────────────────────┐
│  FAZA 1: Przygotowanie Danych                              │
├────────────────────────────────────────────────────────────┤
│  1. Zbieranie demonstracji (teleoperation/kinesthetic)     │
│  2. Konwersja LeRobot → HDF5                               │
│  3. Konwersja HDF5 → RLDS                                  │
│  4. Rejestracja w dataloader                               │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  FAZA 2: Inicjalizacja Modelu                             │
├────────────────────────────────────────────────────────────┤
│  1. Załadowanie UnifoLM-VLM-Base (pretrained)              │
│  2. Inicjalizacja Action Model (DiT)                       │
│  3. Konfiguracja wymiarów (ACTION_DIM, PROPRIO_DIM)        │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  FAZA 3: Fine-tuning                                       │
├────────────────────────────────────────────────────────────┤
│  For each batch:                                           │
│    1. Załaduj (image, text, proprio, actions)              │
│    2. Forward pass: predictions = model(image, text, proprio)│
│    3. Oblicz loss: MSE(predictions, ground_truth_actions)  │
│    4. Backward pass: gradients                             │
│    5. Optimizer step: update weights                       │
│  ─────────────────────────────────────────────             │
│  • Checkpoint co N kroków                                  │
│  • Logowanie metryk do TensorBoard/Wandb                   │
│  • Early stopping jeśli loss przestaje maleć               │
└────────────────────────────────────────────────────────────┘
```

### Dlaczego Fine-tuning a nie Training from Scratch?

**Pre-trenowany VLM (Qwen2.5-VL)** już posiada:
- Rozumienie języka naturalnego
- Zdolności wizyjne (detekcja obiektów, scen)
- Wiedzę o świecie (fizyka, geometria)

**Fine-tuning dodaje:**
- Specyficzne umiejętności manipulacyjne
- Zrozumienie dynamiki konkretnego robota
- Mapowanie "zamiar → akcja"

**Korzyści:**
- ⚡ Szybszy trening (dni zamiast tygodni)
- 📊 Mniej danych potrzebnych (setki vs. miliony demonstracji)
- 🎯 Lepsza generalizacja

---

## 🛠️ Struktura Projektu Krok po Kroku

### Krok 1: Zrozumienie Architektury Kodu

Kod jest zorganizowany w logiczne moduły:

```
src/unifolm_vla/
│
├── model/                          # 🧠 Definicje modeli
│   ├── framework/
│   │   ├── unifolm_vla.py         # GŁÓWNA KLASA - zacznij tutaj!
│   │   ├── base_framework.py      # Klasa bazowa
│   │   └── share_tools.py         # Narzędzia współdzielone
│   │
│   ├── modules/
│   │   ├── vlm/
│   │   │   └── QWen2_5.py         # Vision-Language Model (Qwen2.5-VL)
│   │   │
│   │   └── action_model/
│   │       ├── DiT_ActionHeader.py           # Główna klasa DiT
│   │       └── flow_matching_modules/        # Moduły flow matching
│   │           ├── cross_attention_dit.py    # Cross-attention DiT blocks
│   │           └── action_encoder.py         # Encoder akcji
│   │
│   └── utils/
│       └── pooling_utils.py        # Pooling dla wizyjnych features
│
├── rlds_dataloader/                # 📦 Ładowanie i przetwarzanie danych
│   ├── datasets/
│   │   ├── datasets.py             # Główna klasa datasetu
│   │   └── rlds/oxe/
│   │       ├── configs.py          # Konfiguracje zbiorów RLDS
│   │       ├── transforms.py       # Transformacje danych
│   │       └── mixtures.py         # Mieszanki zbiorów
│   │
│   └── constants.py                # ⚙️ STAŁE - wymiary, normalizacja
│
└── training/                       # 🏋️ Kod treningowy
    ├── train_unifolm_vla.py        # GŁÓWNY SKRYPT TRENINGU
    └── trainer_utils/
        ├── metrics.py              # Metryki (loss, accuracy)
        ├── overwatch.py            # Monitoring i logowanie
        └── trainer_tools.py        # Narzędzia pomocnicze
```

### Krok 2: Czytanie Kodu - Strategia Top-Down

**Zalecana kolejność czytania:**

#### A. Start: `src/unifolm_vla/model/framework/unifolm_vla.py`

To **główna klasa** całego modelu. Kluczowe metody:

```python
class UnifoLMVLA(BaseFramework):
    """
    Główna klasa UnifoLM-VLA łącząca VLM i Action Model.
    
    Kluczowe komponenty:
    - self.vlm: Vision-Language Model (Qwen2.5-VL)
    - self.action_model: DiT Action Model
    - self.flow_matching: Flow matching dla generacji akcji
    """
    
    def __init__(self, config):
        """Inicjalizacja modelu z konfiguracji"""
        # Załaduj VLM (Vision-Language Model)
        # Załaduj Action Model (DiT)
        # Skonfiguruj flow matching
    
    def forward(self, images, text, proprio_state):
        """
        Forward pass - główna metoda inferencji
        
        Args:
            images: Tensor [B, C, H, W] - obrazy z kamer
            text: List[str] - instrukcje tekstowe
            proprio_state: Tensor [B, PROPRIO_DIM] - stan stawów
            
        Returns:
            predicted_actions: Tensor [B, NUM_CHUNKS, ACTION_DIM]
        """
        # 1. VLM processing
        vlm_features = self.vlm(images, text)
        
        # 2. Action generation
        actions = self.action_model(vlm_features, proprio_state)
        
        return actions
```

**Zadanie dla Was:** 
- Otwórz ten plik
- Znajdź metodę `forward()`
- Prześledzić przepływ danych od obrazu do akcji

#### B. VLM: `src/unifolm_vla/model/modules/vlm/QWen2_5.py`

```python
class Qwen2VLModel:
    """
    Qwen2.5-VL - Vision-Language Model
    
    To jest serce rozumienia sceny. Model:
    1. Dzieli obraz na patches (fragmenty)
    2. Tokenizuje tekst
    3. Przetwarza oba przez transformer
    4. Zwraca unified embedding sceny
    """
    
    def forward(self, images, text_tokens):
        # Przetwarzanie obrazu przez ViT (Vision Transformer)
        vision_tokens = self.vision_encoder(images)
        
        # Połączenie vision i language tokens
        combined = self.merge_modalities(vision_tokens, text_tokens)
        
        # Transformer layers z cross-attention
        output = self.transformer(combined)
        
        return output
```

#### C. Action Model: `src/unifolm_vla/model/modules/action_model/DiT_ActionHeader.py`

```python
class DiTActionHeader:
    """
    Diffusion Transformer dla generacji akcji
    
    Używa flow matching do generowania gładkich
    sekwencji akcji (action chunks).
    """
    
    def forward(self, condition, proprio):
        """
        Generuje akcje kondycjonowane na VLM features
        
        Args:
            condition: Features z VLM [B, D]
            proprio: Obecny stan robota [B, PROPRIO_DIM]
            
        Returns:
            actions: Przewidziane akcje [B, NUM_CHUNKS, ACTION_DIM]
        """
        # 1. Encode proprio state
        proprio_emb = self.proprio_encoder(proprio)
        
        # 2. Cross-attention: condition + proprio
        fused = self.cross_attention(condition, proprio_emb)
        
        # 3. Flow matching denoising
        actions = self.flow_matching_process(fused)
        
        return actions
```

### Krok 3: Dataloader - Jak Dane Stają Się Użyteczne

#### `src/unifolm_vla/rlds_dataloader/constants.py`

**Najważniejszy plik konfiguracyjny!**

```python
# Przykład dla robota Unitree G1
G1_CONSTANTS = {
    # Liczba akcji przewidywanych naraz (action chunking)
    'NUM_ACTIONS_CHUNK': 16,
    
    # Wymiar akcji - dla G1:
    # 7 stawów ramienia + 1 chwytak = 8
    'ACTION_DIM': 8,
    
    # Wymiar propriocepcji:
    # 7 pozycji stawów + 7 prędkości + 1 chwytak = 15
    'PROPRIO_DIM': 15,
    
    # Typ normalizacji
    'ACTION_PROPRIO_NORMALIZATION_TYPE': 'normal',  # Gaussian normalization
}
```

**Dlaczego normalizacja?**
- Różne stawy mają różne zakresy (np. -π do π, 0 do 1)
- Normalizacja do ~N(0,1) stabilizuje trening
- Model uczy się łatwiej na znormalizowanych danych

#### `src/unifolm_vla/rlds_dataloader/datasets/datasets.py`

```python
class RLDSDataset:
    """
    Główna klasa datasetu - ładuje dane RLDS
    """
    
    def __getitem__(self, idx):
        """
        Zwraca jeden sample treningowy
        
        Returns:
            {
                'image': Tensor [3, H, W],
                'instruction': str,
                'proprio': Tensor [PROPRIO_DIM],
                'actions': Tensor [NUM_CHUNKS, ACTION_DIM],
            }
        """
        # 1. Załaduj epizod z RLDS
        episode = self.load_episode(idx)
        
        # 2. Wybierz losowy krok z epizodu
        step = random.choice(episode.steps)
        
        # 3. Zastosuj transformacje (augmentacja)
        image = self.transform(step.image)
        
        # 4. Normalizuj akcje i proprio
        actions = self.normalize_actions(step.actions)
        proprio = self.normalize_proprio(step.proprio)
        
        return {
            'image': image,
            'instruction': step.instruction,
            'proprio': proprio,
            'actions': actions,
        }
```

### Krok 4: Training Loop - Jak Model Się Uczy

#### `src/unifolm_vla/training/train_unifolm_vla.py`

```python
def train_epoch(model, dataloader, optimizer):
    """
    Jedna epoka treningu
    """
    model.train()  # Tryb treningowy (włącz dropout, batch norm)
    
    for batch_idx, batch in enumerate(dataloader):
        # 1. Pobierz dane z batcha
        images = batch['image']          # [B, 3, H, W]
        instructions = batch['instruction']  # List[str]
        proprio = batch['proprio']       # [B, PROPRIO_DIM]
        gt_actions = batch['actions']    # [B, NUM_CHUNKS, ACTION_DIM]
        
        # 2. Forward pass - predykcja akcji
        pred_actions = model(images, instructions, proprio)
        
        # 3. Oblicz loss (Mean Squared Error)
        loss = F.mse_loss(pred_actions, gt_actions)
        
        # 4. Backward pass - oblicz gradienty
        optimizer.zero_grad()  # Wyzeruj stare gradienty
        loss.backward()        # Oblicz nowe gradienty
        
        # 5. Optimizer step - zaktualizuj wagi
        optimizer.step()
        
        # 6. Logowanie
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
```

**Co to znaczy "gradient"?**
- Gradient pokazuje, w którą stronę zmienić wagi, aby zmniejszyć loss
- Backpropagation oblicza gradienty dla wszystkich parametrów
- Optimizer (np. AdamW) używa gradientów do update'u wag

**Loss Function (MSE):**
```
MSE = (1/N) * Σ (predicted_action - ground_truth_action)²

Przykład:
predicted = [0.5, 0.3, 0.1]
ground_truth = [0.6, 0.2, 0.15]
MSE = ((0.5-0.6)² + (0.3-0.2)² + (0.1-0.15)²) / 3
    = (0.01 + 0.01 + 0.0025) / 3
    = 0.0075
```

---

## 🎮 Praktyczne Ćwiczenia

### Ćwiczenie 1: Uruchomienie Ewaluacji w Symulacji

**Cel:** Zrozumienie pełnego pipeline'u inferencji

**Krok po kroku:**

```bash
# 1. Instalacja środowiska LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/LIBERO/libero_requirements.txt

# 2. Pobranie modelu
# (Symulacja lokalna - możesz użyć CPU jeśli nie masz GPU)
huggingface-cli download unitreerobotics/Unifolm-VLA-Libero

# 3. Konfiguracja skryptu
cd scripts/eval_scripts
nano run_eval_libero.sh  # lub vim/gedit

# Edytuj:
# your_ckpt="/ścieżka/do/pobranego/modelu"
# LIBERO_HOME="/ścieżka/do/LIBERO"

# 4. Uruchomienie
bash run_eval_libero.sh
```

**Co obserwować:**
- Wizualizacja zadania w symulatorze
- Success rate dla różnych zadań
- Czas inferencji na step

**Pytania do refleksji:**
1. Jak model radzi sobie z różnymi wariantami tego samego zadania?
2. Które zadania są trudniejsze? Dlaczego?
3. Jak wpływa jakość instrukcji na sukces?

### Ćwiczenie 2: Analiza Predykcji Akcji

**Cel:** Zrozumienie, co model faktycznie przewiduje

Utwórz skrypt `analyze_predictions.py`:

```python
import torch
import matplotlib.pyplot as plt
from unifolm_vla.model.framework.unifolm_vla import UnifoLMVLA

# Załaduj model
model = UnifoLMVLA.from_pretrained("path/to/checkpoint")
model.eval()

# Przygotuj dane testowe
image = load_image("test_scene.jpg")
instruction = "pick up the red cube"
proprio = torch.zeros(15)  # Przykładowy stan

# Predykcja
with torch.no_grad():
    actions = model(image, instruction, proprio)
    
# Wizualizacja action chunk
plt.figure(figsize=(12, 6))
for i in range(8):  # 8 stawów (7 + chwytak)
    plt.subplot(2, 4, i+1)
    plt.plot(actions[0, :, i].cpu().numpy())
    plt.title(f"Joint {i}")
    plt.xlabel("Time step")
    plt.ylabel("Action value")
plt.tight_layout()
plt.savefig("predicted_actions.png")
```

**Analiza:**
- Czy akcje są gładkie?
- Czy widać logiczny pattern (np. otwarcie chwytaka przed złapaniem)?
- Jak zmienia się predykcja dla różnych instrukcji?

### Ćwiczenie 3: Dodanie Nowego Zbioru Danych

**Scenariusz:** Zbieraliście demonstracje nowego zadania "G1_Sort_Colors"

**Kroki:**

#### 1. Rejestracja w `configs.py`

```python
# src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/configs.py

OXE_DATASET_CONFIGS = {
    # ... existing datasets ...
    
    # NOWY ZBIÓR DANYCH
    'g1_sort_colors': DatasetConfig(
        name='g1_sort_colors',
        data_dir='/path/to/g1_sort_colors/1.0.0',
        image_obs_keys={
            'primary': 'observation/image',
        },
        state_obs_keys=['observation/state'],
        action_keys=['action'],
        language_key='observation/instruction',
    ),
}
```

#### 2. Dodanie transformacji w `transforms.py`

```python
# src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/transforms.py

def g1_sort_colors_dataset_transform(trajectory):
    """
    Transformacja specyficzna dla zadania sortowania kolorów
    
    Args:
        trajectory: RLDS trajectory
        
    Returns:
        Transformed trajectory z odpowiednią strukturą
    """
    # Normalizacja obrazu do [0, 1]
    trajectory['observation']['image'] = (
        trajectory['observation']['image'] / 255.0
    )
    
    # Normalizacja akcji
    # (zakres akcji G1: każdy staw -π do π, chwytak 0 do 1)
    actions = trajectory['action']
    actions[:, :7] = actions[:, :7] / np.pi  # Stawy ramienia
    actions[:, 7] = actions[:, 7]            # Chwytak już w [0,1]
    
    trajectory['action'] = actions
    
    return trajectory

# Rejestracja transformacji
DATASET_TRANSFORMS = {
    # ... existing transforms ...
    'g1_sort_colors': g1_sort_colors_dataset_transform,
}
```

#### 3. Dodanie do mieszanki w `mixtures.py`

```python
# src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/mixtures.py

MIXTURE_CONFIGS = {
    # Mieszanka wszystkich zadań G1 + nowe zadanie
    'g1_all_tasks_with_sorting': {
        'g1_stack_block': 0.1,
        'g1_bag_insert': 0.1,
        # ... other tasks ...
        'g1_sort_colors': 0.15,  # Większa waga dla nowego zadania
    },
}
```

#### 4. Aktualizacja stałych w `constants.py`

```python
# src/unifolm_vla/rlds_dataloader/constants.py

# Jeśli nowe zadanie używa tych samych wymiarów co G1
# nie musicie nic zmieniać

# Jeśli używa specyficznej normalizacji:
DATASET_NORMALIZATION = {
    'g1_sort_colors': {
        'type': 'bounds',  # Normalizacja min-max zamiast Gaussian
        'action_bounds': {
            'low': [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0],
            'high': [3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 1.0],
        },
    },
}
```

#### 5. Trening na nowym zbiorze

```bash
# scripts/run_scripts/run_unifolm_vla_train.sh

# Ustaw data_mix na nową mieszankę
data_mix="g1_all_tasks_with_sorting"

# Uruchom trening
bash scripts/run_scripts/run_unifolm_vla_train.sh
```

### Ćwiczenie 4: Debugging Treningu

**Problem:** Loss nie spada, model się nie uczy

**Checklist debugowania:**

```python
# 1. Sprawdź zakresy danych
def check_data_ranges(dataloader):
    """Sprawdź czy dane są poprawnie znormalizowane"""
    batch = next(iter(dataloader))
    
    print("Image range:", batch['image'].min(), batch['image'].max())
    # Powinno być ~[0, 1] lub ~[-1, 1]
    
    print("Action range:", batch['actions'].min(), batch['actions'].max())
    # Powinno być ~[-3, 3] dla normalizacji Gaussian
    
    print("Proprio range:", batch['proprio'].min(), batch['proprio'].max())
    
check_data_ranges(train_dataloader)
```

```python
# 2. Sprawdź gradienty
def check_gradients(model):
    """Sprawdź czy gradienty przepływają"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm = {grad_norm:.4f}")
            
            # Problem: grad_norm = 0.0 → martwe neurony
            # Problem: grad_norm > 100.0 → exploding gradients
```

```python
# 3. Uproszczone overfitting test
def overfit_single_batch(model, batch, num_steps=1000):
    """
    Spróbuj zoverfit'ować jeden batch
    Jeśli się nie uda, problem z modelem/danymi
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(num_steps):
        pred = model(batch['image'], batch['instruction'], batch['proprio'])
        loss = F.mse_loss(pred, batch['actions'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
    
    # Po 1000 krokach loss powinien być bardzo bliski 0
```

---

## 🔧 Troubleshooting - Najczęstsze Problemy

### Problem 1: CUDA Out of Memory

**Objaw:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Rozwiązania:**

```python
# Opcja 1: Zmniejsz batch size
# W skrypcie treningowym:
batch_size = 4  # Zamiast 8 lub 16

# Opcja 2: Gradient accumulation (symuluje większy batch)
accumulation_steps = 4

for batch_idx, batch in enumerate(dataloader):
    loss = compute_loss(model, batch)
    loss = loss / accumulation_steps  # Skaluj loss
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Opcja 3: Mixed precision training (FP16)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    pred = model(images, text, proprio)
    loss = compute_loss(pred, gt_actions)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Problem 2: Model Przewiduje Tylko Zerowe Akcje

**Diagnoza:**
```python
# Sprawdź predykcje
pred_actions = model(test_image, test_instruction, test_proprio)
print("Predicted actions std:", pred_actions.std().item())
# Jeśli std ≈ 0, model przewiduje konstantę
```

**Prawdopodobne przyczyny:**

1. **Błędna normalizacja danych**
```python
# Sprawdź statystyki datasetu
actions = []
for batch in dataloader:
    actions.append(batch['actions'])
actions = torch.cat(actions)

print("Mean:", actions.mean(dim=(0, 1)))
print("Std:", actions.std(dim=(0, 1)))

# Jeśli mean ≈ 0 dla wszystkich, problem z normalizacją!
```

2. **Learning rate zbyt mała**
```python
# Zwiększ learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Zamiast 1e-5
```

3. **Zamrożone warstwy**
```python
# Sprawdź czy wszystkie parametry są treningowe
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"FROZEN: {name}")
        # Odblokuj jeśli to błąd:
        param.requires_grad = True
```

### Problem 3: Konwersja Danych Się Nie Udaje

**Objaw:**
```
tfds build fails with "Invalid trajectory format"
```

**Rozwiązanie krok po kroku:**

```python
# 1. Sprawdź strukturę HDF5
import h5py

with h5py.File('your_data.hdf5', 'r') as f:
    print("Keys:", list(f.keys()))
    
    # Powinno być:
    # ['episode_0', 'episode_1', ...]
    
    ep = f['episode_0']
    print("Episode keys:", list(ep.keys()))
    
    # Powinno być:
    # ['observation', 'action', 'instruction', ...]

# 2. Zweryfikuj wymiary
print("Image shape:", ep['observation']['image'].shape)
# Powinno być [T, H, W, 3] gdzie T = kroki

print("Action shape:", ep['action'].shape)
# Powinno być [T, ACTION_DIM]

# 3. Sprawdź typy danych
print("Image dtype:", ep['observation']['image'].dtype)
# Powinno być uint8 lub float32

print("Action dtype:", ep['action'].dtype)
# Powinno być float32
```

### Problem 4: Serwer Inferencji Nie Odpowiada

**Diagnoza:**

```bash
# 1. Sprawdź czy port jest otwarty
netstat -tuln | grep PORT_NUMBER

# 2. Sprawdź logi serwera
tail -f server_logs.txt

# 3. Test prostym klientem
python -c "
import requests
response = requests.post(
    'http://localhost:PORT/predict',
    json={'test': 'data'}
)
print(response.status_code)
"
```

**Częste błędy:**

```python
# Błąd 1: Model nie załadowany do GPU
model = model.cuda()  # Nie zapomnij!

# Błąd 2: Preprocessing obrazu niepoprawny
# Serwer oczekuje uint8 [0, 255], a dostaje float [0, 1]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)  # uint8
    # NIE normalizuj tutaj - model to zrobi
    return image

# Błąd 3: Timeout zbyt krótki
# Ustaw dłuższy timeout dla inferencji
client = ModelClient(timeout=30)  # 30 sekund
```

---

## 🚀 Zaawansowane Tematy

### 1. Action Chunking - Dlaczego To Działa?

**Intuicja:**
Wyobraź sobie, że chcesz złapać piłkę:
- Pojedyncze kroki: "ruch w lewo", "ruch w górę", "zamknij dłoń"
  → Niestabilne, reaktywne, spóźnione
- Chunk akcji: Cała trajektoria ruchu zaplanowana z wyprzedzeniem
  → Gładka, antycypująca, skuteczna

**Implementacja:**

```python
class ActionChunkPredictor:
    def __init__(self, chunk_size=16):
        self.chunk_size = chunk_size
        self.action_buffer = []
    
    def predict_and_execute(self, observation):
        # Jeśli bufor pusty lub prawie pusty, generuj nowy chunk
        if len(self.action_buffer) < 2:
            # Predykcja całego chunka naraz
            action_chunk = self.model(observation)  # [16, ACTION_DIM]
            self.action_buffer = list(action_chunk)
        
        # Wykonaj pierwszą akcję z bufora
        action_to_execute = self.action_buffer.pop(0)
        
        return action_to_execute
```

**Korzyści:**
- **Temporal consistency**: Akcje są spójne w czasie
- **Reduced latency**: Mniej wywołań modelu
- **Better generalization**: Model uczy się sekwencji, nie pojedynczych kroków

### 2. Flow Matching vs. Diffusion

**Diffusion Models (tradycyjne):**
```
Proces: Czysty szum → ... → Akcje
Kroki: Wiele (50-1000 kroków denoising)
Czas inferencji: Wolny
```

**Flow Matching (UnifoLM-VLA):**
```
Proces: Jednolity przepływ od szumu do akcji
Kroki: Mniej (10-20 kroków)
Czas inferencji: Szybszy
```

**Matematyka (uproszczona):**

```python
# Diffusion: iteracyjne odszumianie
x_t = x_{t-1} + √(β_t) * ε  # Dodaj szum
x_0 = denoise(x_T, T steps)  # Odszum w wielu krokach

# Flow Matching: bezpośrednia interpolacja
x(t) = (1-t) * x_noise + t * x_target
# Jeden krok: x(1) = x_target
```

**Dlaczego Flow Matching dla robotyki?**
- ⚡ Szybsza inferencja → wyższe control frequency
- 🎯 Deterministyczny → przewidywalne zachowanie
- 🔧 Prostsza implementacja

### 3. Multi-task Learning - Jedna Polityka, Wiele Zadań

**Jak to możliwe?**

**Kluczowy trick: Language Conditioning**

```python
# Tradycyjne podejście: osobna polityka na zadanie
policy_stack = StackingPolicy()
policy_insert = InsertionPolicy()
# ... 12 osobnych modeli!

# UnifoLM-VLA: jedna polityka
policy = UnifoLMVLA()

# Różne zadania przez różne instrukcje:
actions_stack = policy(image, "stack the blocks", proprio)
actions_insert = policy(image, "insert into the bag", proprio)
```

**Shared Representations:**
- VLM uczy się ogólnych umiejętności: chwytak, pick, place, manipulacja
- Instrukcja specyfikuje, które umiejętności użyć
- Action Model adaptuje się do kontekstu

**Transfer Learning:**
```
Zadanie A: Stack blocks → Uczy się: picking, precyzyjne umieszczanie
Zadanie B: Insert objects → Uczy się: orientacja obiektów, detekcja wejść
Zadanie C: Pour liquid → Transfer z A+B: picking + precyzja
```

### 4. Proprioceptive Feedback Loop

**Dlaczego proprio jest ważne?**

Bez proprio (open-loop):
```
Komenda: "Move to position X"
Robot: *próbuje* → ale nie wie gdzie faktycznie jest
Rezultat: Akumulacja błędów, drift
```

Z proprio (closed-loop):
```
Komenda: "Move to position X"
Robot: *próbuje* → czyta własną pozycję → koryguje
Rezultat: Stabilna, dokładna kontrola
```

**Implementacja w UnifoLM-VLA:**

```python
class ClosedLoopController:
    def control_loop(self, target_instruction):
        while not task_complete:
            # 1. Odczyt stanu (proprio)
            current_proprio = robot.get_joint_states()
            
            # 2. Obserwacja wizualna
            current_image = robot.get_camera_image()
            
            # 3. Predykcja z kondycjonowaniem na proprio
            predicted_actions = model(
                current_image, 
                target_instruction,
                current_proprio  # ← Kluczowe!
            )
            
            # 4. Wykonanie
            robot.execute_action(predicted_actions[0])
            
            # 5. Pętla się powtarza (closed-loop!)
```

**Proprio jako "sense of body":**
- Robot "czuje" gdzie są jego ramiona
- Pozwala na reaktywną korekcję błędów
- Umożliwia pracę w dynamicznych środowiskach

---

## 📊 Metryki i Ewaluacja

### Jak Ocenić Jakość Modelu?

#### 1. Success Rate (Wskaźnik Sukcesu)

```python
def evaluate_success_rate(model, test_episodes):
    """
    Główna metryka: Czy zadanie zostało wykonane poprawnie?
    """
    successes = 0
    
    for episode in test_episodes:
        # Uruchom epizod z modelem
        final_state = run_episode(model, episode.init_state, episode.instruction)
        
        # Sprawdź warunek sukcesu (task-specific)
        if check_success_condition(final_state, episode.goal):
            successes += 1
    
    return successes / len(test_episodes)

# Przykład: Dla zadania "stack blocks"
def check_success_condition(final_state, goal):
    # Sprawdź czy klocki są ułożone w stos
    blocks_stacked = all(
        abs(block.position.z - expected_z) < 0.02  # 2cm tolerancji
        for block, expected_z in zip(final_state.blocks, goal.heights)
    )
    return blocks_stacked
```

#### 2. Action Prediction Error

```python
def compute_action_error(model, val_dataloader):
    """
    Ile model się myli w predykcji akcji?
    """
    errors = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            pred = model(batch['image'], batch['instruction'], batch['proprio'])
            gt = batch['actions']
            
            # MSE per action dimension
            error = (pred - gt).pow(2).mean(dim=0)  # [ACTION_DIM]
            errors.append(error)
    
    mean_error = torch.stack(errors).mean(dim=0)
    
    # Raport per staw
    for i, err in enumerate(mean_error):
        print(f"Joint {i}: MSE = {err:.4f}")
    
    return mean_error
```

#### 3. Generalization Tests

```python
# Test 1: Nowe pozycje obiektów
def test_spatial_generalization():
    # Trenuj: obiekty w pozycjach A, B, C
    # Testuj: obiekty w pozycjach D, E, F
    pass

# Test 2: Nowe obiekty (ta sama kategoria)
def test_object_generalization():
    # Trenuj: czerwony kubek, niebieski kubek
    # Testuj: zielony kubek (nowy kolor!)
    pass

# Test 3: Nowe instrukcje (parafrazowanie)
def test_language_generalization():
    instructions_train = ["pick up the red cube"]
    instructions_test = [
        "grab the red cube",
        "take the red block",
        "lift the crimson cube"
    ]
    # Czy model rozumie synonimydescriptions?
    pass
```

### Benchmarking

**Standardowe benchmarki dla VLA:**

| Benchmark | Opis | Metryka |
|-----------|------|---------|
| **LIBERO** | 12 zadań manipulacyjnych w symulacji | Success rate (%) |
| **Calvin** | Long-horizon tasks (łańcuchy zadań) | Average success rate |
| **Real-world eval** | 12 zadań G1 na prawdziwym robocie | Success rate + execution time |

**Typowe wyniki dla UnifoLM-VLA:**
- LIBERO: ~85% success rate
- Real G1 tasks: ~75% success rate (trudniejsze warunki)

---

## 💡 Best Practices dla Projektu

### 1. Organizacja Eksperymentów

```bash
# Struktura katalogów dla eksperymentów
experiments/
├── exp_001_baseline/
│   ├── config.yaml
│   ├── checkpoints/
│   ├── logs/
│   └── results.json
├── exp_002_new_data/
│   ├── config.yaml
│   ├── checkpoints/
│   ├── logs/
│   └── results.json
└── exp_003_architecture_change/
    └── ...

# Każdy eksperyment ma:
# - Pełną konfigurację (reproducibility)
# - Checkpointy
# - Logi (TensorBoard/Wandb)
# - Wyniki ewaluacji
```

### 2. Version Control dla Modeli

```bash
# Git LFS dla dużych plików
git lfs install
git lfs track "*.pth"
git lfs track "*.h5"

# Tag dla każdego ważnego checkpointu
git tag -a v1.0-baseline -m "Baseline model trained on G1_stack_block"
git push origin v1.0-baseline
```

### 3. Dokumentacja Eksperymentów

```markdown
# experiments/exp_001_baseline/README.md

## Experiment 001: Baseline Model

**Date:** 2026-02-10  
**Author:** Jan Kowalski

### Hypothesis
Training UnifoLM-VLA on single task (G1_stack_block) should achieve >80% success rate.

### Configuration
- Model: UnifoLM-VLA-Base
- Dataset: G1_stack_block (500 episodes)
- Batch size: 8
- Learning rate: 1e-4
- Training steps: 10,000

### Results
- Final train loss: 0.0123
- Validation success rate: 82.5%
- Inference time: 45ms per step

### Conclusions
- ✅ Hypothesis confirmed
- Model generalizes well to unseen object positions
- Failure cases: mostly when blocks are partially occluded

### Next steps
- Experiment 002: Add more diverse block colors
- Experiment 003: Multi-task learning with G1_bag_insert
```

### 4. Code Quality

```python
# Zawsze dodawaj docstringi
def normalize_actions(actions, normalization_type='normal'):
    """
    Normalizuje akcje zgodnie ze specyfikowanym schematem.
    
    Args:
        actions (np.ndarray): Surowe akcje, shape [T, ACTION_DIM]
        normalization_type (str): Typ normalizacji ('normal', 'bounds', 'none')
        
    Returns:
        np.ndarray: Znormalizowane akcje, shape [T, ACTION_DIM]
        
    Raises:
        ValueError: Jeśli normalization_type jest nieznany
        
    Example:
        >>> actions = np.array([[1.57, 0.5], [-1.57, 0.8]])
        >>> normalized = normalize_actions(actions, 'normal')
        >>> print(normalized.mean(), normalized.std())
        0.0 1.0
    """
    if normalization_type == 'normal':
        mean = actions.mean(axis=0)
        std = actions.std(axis=0)
        return (actions - mean) / (std + 1e-8)
    elif normalization_type == 'bounds':
        # ... implementation ...
        pass
    else:
        raise ValueError(f"Unknown normalization: {normalization_type}")
```

### 5. Testing

```python
# tests/test_model.py
import unittest

class TestUnifoLMVLA(unittest.TestCase):
    def setUp(self):
        """Przygotowanie przed każdym testem"""
        self.model = UnifoLMVLA.from_config(test_config)
    
    def test_forward_shape(self):
        """Sprawdź czy output ma poprawny kształt"""
        batch = {
            'image': torch.randn(2, 3, 224, 224),
            'instruction': ["test1", "test2"],
            'proprio': torch.randn(2, 15),
        }
        
        output = self.model(**batch)
        
        # Expected shape: [batch, chunks, action_dim]
        self.assertEqual(output.shape, (2, 16, 8))
    
    def test_gradient_flow(self):
        """Sprawdź czy gradienty przepływają"""
        # ... implementation ...
    
    def test_normalization(self):
        """Sprawdź czy normalizacja działa poprawnie"""
        # ... implementation ...

if __name__ == '__main__':
    unittest.main()
```

---

## 🎓 Projekt Semestralny - Propozycje

### Projekt 1: Nowe Zadanie dla Robota G1

**Cel:** Nauczyć robota nowego zadania manipulacyjnego

**Kroki:**
1. Zaprojektuj zadanie (np. "G1_Water_Plant")
2. Zbierz 100-200 demonstracji (teleoperation)
3. Przetworz dane do formatu RLDS
4. Fine-tune model UnifoLM-VLA-Base
5. Ewaluuj w symulacji i na robocie

**Ocena:**
- Jakość zbioru danych (30%)
- Implementacja (30%)
- Success rate (25%)
- Raport i analiza (15%)

### Projekt 2: Improved Action Prediction

**Cel:** Ulepsz moduł predykcji akcji

**Możliwe kierunki:**
- Eksperymentuj z różnymi rozmiarami action chunks
- Wypróbuj alternatywne architektury (e.g., Mamba zamiast DiT)
- Dodaj trajectory optimization

**Ocena:**
- Nowatorskość podejścia (30%)
- Implementacja (30%)
- Wyniki eksperymentów (25%)
- Dokumentacja (15%)

### Projekt 3: Multi-Robot Collaboration

**Cel:** Koordynacja dwóch robotów G1

**Wyzwania:**
- Shared scene understanding
- Koordinacja działań
- Collision avoidance

**Wykorzystaj:** Dataset G1_DualRobot_Clean_Table

**Ocena:**
- Trudność problemu (extra punkty)
- Implementacja (30%)
- Wyniki (30%)
- Prezentacja (20%)

---

## 📚 Dodatkowe Zasoby

### Artykuły Naukowe (Podstawy)

1. **Vision-Language Models:**
   - "CLIP: Learning Transferable Visual Models From Natural Language Supervision"
   - "Flamingo: a Visual Language Model for Few-Shot Learning"

2. **Robotics Learning:**
   - "RT-1: Robotics Transformer for Real-World Control at Scale"
   - "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"

3. **Diffusion Models:**
   - "Denoising Diffusion Probabilistic Models"
   - "Flow Matching for Generative Modeling"

### Online Kursy

- **CS231n** (Stanford): Computer Vision
- **CS224n** (Stanford): NLP with Deep Learning
- **Deepmind x UCL**: Deep Learning Lectures

### Narzędzia i Biblioteki

```bash
# Wizualizacja danych robotycznych
pip install rosbag2 plotjuggler

# Debugging treningu
pip install tensorboard wandb

# 3D visualization
pip install open3d trimesh
```

---

## ✅ Checklist Projektu

Przed oddaniem projektu, sprawdź:

- [ ] **Kod:**
  - [ ] Wszystkie pliki mają docstringi
  - [ ] Kod jest sformatowany (black, ruff)
  - [ ] Testy jednostkowe przechodzą
  - [ ] Nie ma hardcoded paths

- [ ] **Eksperymenty:**
  - [ ] Checkpointy zapisane i wersjonowane
  - [ ] Logi treningowe dostępne (TensorBoard)
  - [ ] Wyniki ewaluacji udokumentowane

- [ ] **Dokumentacja:**
  - [ ] README.md opisuje projekt
  - [ ] Instrukcje reprodukcji wyników
  - [ ] Analiza błędów i ograniczeń

- [ ] **Prezentacja:**
  - [ ] Demo działającego modelu (wideo)
  - [ ] Prezentacja slajdów (PDF)
  - [ ] Źródła kodu na GitHub

---

## 🤝 Wsparcie i Pomoc

### Gdzie szukać pomocy?

1. **Dokumentacja:**
   - README_pl.md (ten plik!)
   - Docstringi w kodzie
   - Oficjalna dokumentacja zależności

2. **Issues:**
   - GitHub Issues projektu
   - Zadawaj konkretne pytania z przykładami kodu

3. **Konsultacje:**
   - Godziny dyżurów prowadzącego
   - Forum dyskusyjne kursu

### Jak zadawać dobre pytania?

❌ **Źle:**
"Model nie działa, co robić?"

✅ **Dobrze:**
```
Problem: Model przewiduje tylko zerowe akcje

Środowisko:
- UnifoLM-VLA commit: abc123
- CUDA 12.4, PyTorch 2.5.1
- GPU: RTX 3090

Kroki do reprodukcji:
1. Trenowałem na G1_stack_block (200 episodes)
2. Użyłem config z configs/baseline.yaml
3. Po 5000 krokach loss = 0.15 ale predykcje = zeros

Kod:
[wklej minimalne snippet]

Logi:
[wklej relevantne logi]

Co już próbowałem:
- Sprawdziłem normalizację danych → OK
- Zwiększyłem learning rate → bez zmian
```

---

## 🎯 Podsumowanie - Kluczowe Punkty

**Co musicie zapamiętać:**

1. **UnifoLM-VLA** = VLM (Qwen2.5-VL) + Action Model (DiT)
2. **Action chunking** = predykcja sekwencji akcji zamiast pojedynczych kroków
3. **RLDS format** = standard dla danych robotycznych
4. **Fine-tuning** zamiast training from scratch = wykorzystanie pretrenowanej wiedzy
5. **Closed-loop control** = kontinuous feedback z propriocepcji

**Workflow:**
```
Zbierz dane → RLDS → Rejestruj dataset → Skonfiguruj → Trenuj → Ewaluuj
```

**Debugging:**
```
Sprawdź dane → Sprawdź gradienty → Overfit single batch → Pełny trening
```

---

**Powodzenia w projekcie! 🚀🤖**

*Dokument stworzony dla studentów Politechniki Rzeszowskiej, 2026*