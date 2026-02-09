"""
unifolm_vla.py - Główna klasa modelu UnifoLM-VLA

Ten plik zawiera główną architekturę modelu łączącą:
- Vision-Language Model (VLM) - zrozumienie sceny i instrukcji
- Action Model (DiT) - generowanie akcji robota

UWAGA DLA STUDENTÓW:
To jest CENTRALNY plik całego frameworku. Zacznij czytanie kodu tutaj!

Kluczowe klasy:
- Unifolm_VLA: Główna klasa modelu łącząca VLM i Action Model

Przepływ danych:
1. forward(): Trenowanie - oblicza loss na podstawie demonstracji
2. predict_action(): Inferencja - generuje akcje dla nowych obserwacji
"""

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from unifolm_vla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.model.modules.vlm import get_vlm_model
from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model, FlowmatchingActionHead
from unifolm_vla.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("unifolm_vla")
class Unifolm_VLA(baseframework):
    """
    Multimodal vision-language-action model.
    
    OPIS PO POLSKU:
    Główna klasa modelu UnifoLM-VLA łącząca wizję, język i akcje.
    
    Model składa się z dwóch głównych komponentów:
    1. VLM (Vision-Language Model) - Qwen2.5-VL
       - Przetwarza obrazy z kamery robota
       - Rozumie instrukcje tekstowe w języku naturalnym
       - Generuje semantyczne reprezentacje sceny (embeddings)
    
    2. Action Model - Diffusion Transformer (DiT)
       - Przyjmuje embeddingi z VLM
       - Generuje sekwencje akcji (action chunks)
       - Używa flow matching dla gładkich trajektorii
    
    Zastosowanie:
    - Trenowanie: Wywołaj forward() z demonstracjami
    - Inferencja: Wywołaj predict_action() z nowymi obserwacjami
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Inicjalizacja modelu UnifoLM-VLA.
        
        OPIS PO POLSKU:
        Konstruktor inicjalizuje dwa główne komponenty modelu:
        
        Args:
            config: Słownik konfiguracyjny zawierający parametry modelu
                   - framework.vlm: Konfiguracja Vision-Language Model
                   - framework.action_model: Konfiguracja Action Model
        
        Kroki inicjalizacji:
        1. Załaduj Vision-Language Model (Qwen2.5-VL)
           - To jest pretrenowany model multimodalny
           - Rozumie obrazy i tekst równocześnie
        
        2. Skonfiguruj wymiary cross-attention w Action Model
           - Action Model potrzebuje wiedzieć rozmiar embeddingów z VLM
           - hidden_size z Qwen2.5-VL → cross_attention_dim w DiT
        
        3. Załaduj procesor dla VLM
           - Procesor konwertuje obrazy i tekst na format akceptowany przez VLM
           - Zawiera tokenizer (dla tekstu) i image processor (dla obrazów)
        
        4. Utwórz Action Model
           - FlowmatchingActionHead z architekturą DiT
           - Generuje akcje na podstawie embeddingów z VLM
        """

        super().__init__()
        self.config = config
        
        # Krok 1: Inicjalizacja Vision-Language Model (Qwen2.5-VL)
        # VLM jest sercem rozumienia sceny - łączy wizję i język
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        
        # Krok 2: Konfiguracja wymiarów dla Action Model
        # Action Model potrzebuje znać rozmiar embeddingów z VLM
        # aby poprawnie skonfigurować cross-attention
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
        
        # Krok 3: Procesor do preprocessing'u danych wejściowych
        # Konwertuje surowe obrazy i tekst na tensory PyTorch
        self.processor = self.qwen_vl_interface.processor
        
        # Krok 4: Inicjalizacja Action Model (DiT z flow matching)
        # Ten model generuje akcje robota na podstawie embeddingów sceny
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  
    
    def forward(
        self,
        qwen_inputs: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Forward pass podczas trenowania - oblicza loss na demonstracjach.
        
        OPIS PO POLSKU:
        Główna metoda treningowa. Wykonywana dla każdego batcha podczas treningu.
        
        Args:
            qwen_inputs: Słownik zawierający:
                - input_ids: Tokenizowane instrukcje tekstowe [B, seq_len]
                - attention_mask: Maska attention dla tokensów [B, seq_len]
                - pixel_values: Przetworzone obrazy z kamer [B, C, H, W]
                - image_grid_thw: Metadane o gridzie obrazu
                - action: Ground-truth akcje z demonstracji [B, num_chunks, action_dim]
                - state: Stan proprioceptywny (pozycje stawów) [B, proprio_dim]
        
        Returns:
            Słownik z:
                - action_loss: MSE loss między przewidzianymi a prawdziwymi akcjami
        
        Przepływ danych:
        1. VLM Processing (bfloat16 dla szybkości):
           obraz + tekst → VLM → scene embeddings [B, L, H]
           
        2. Action Generation (float32 dla precyzji):
           scene embeddings + proprio → Action Model → predicted actions
           
        3. Loss Calculation:
           MSE(predicted_actions, ground_truth_actions)
        
        UWAGA: repeated_diffusion_steps
        - Akcje są powtarzane N razy (domyślnie 4)
        - To technika augmentacji podczas treningu diffusion models
        - Pomaga stabilizować trening
        """
        
        # Ekstrakcja danych z batcha i konwersja do bfloat16
        actions = qwen_inputs["action"].to(torch.bfloat16)  # Ground-truth akcje
        state = qwen_inputs["state"].to(torch.bfloat16)  # Stan proprioceptywny
        state = state.unsqueeze(1) if state.dim() == 2 else state  # Zapewnienie wymiaru [B, 1, D]

        # FAZA 1: Vision-Language Model Processing
        # Używamy bfloat16 dla oszczędności pamięci i szybkości
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Przekaż obraz i tekst przez Qwen2.5-VL
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs["input_ids"],  # Tokenizowane instrukcje
                attention_mask=qwen_inputs["attention_mask"],  # Maska padding
                pixel_values=qwen_inputs["pixel_values"],  # Obrazy RGB
                image_grid_thw=qwen_inputs["image_grid_thw"],  # Metadane obrazu
                output_hidden_states = True,  # Zwróć hidden states ze wszystkich warstw
                return_dict=True,  
            )
            # Weź embedding z ostatniej warstwy transformera
            # To jest najbogatsze semantyczne zrozumienie sceny
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            # B = batch size, L = długość sekwencji (tokens), H = hidden size
            
        # FAZA 2: Action Model Processing
        # Używamy float32 dla większej precyzji w predykcji akcji
        with torch.autocast("cuda", dtype=torch.float32):
            # Powtórzenie dla techniki treningu diffusion (data augmentation)
            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            
            # Replikuj akcje, embeddingi i state N razy
            # To zwiększa efektywny batch size dla stabilności treningu
            actions_target_repeated = actions.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)
            
            # Oblicz loss akcji używając flow matching
            # Action Model porównuje przewidywane akcje z ground-truth
            action_loss = self.action_model(
                last_hidden_repeated,  # Embeddingi sceny z VLM
                actions_target_repeated,  # Ground-truth akcje (target)
                state_repeated,  # Obecny stan robota (propriocepcja)
            )
            
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        qwen_inputs,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Predykcja akcji podczas inferencji (deployment na robocie).
        
        OPIS PO POLSKU:
        Metoda używana podczas rzeczywistej pracy robota (nie treningu).
        
        @torch.inference_mode() - Dekorator wyłączający gradient computation:
        - Oszczędza pamięć GPU
        - Przyspiesza obliczenia (brak backprop)
        - Używany tylko podczas inferencji, nie treningu
        
        Args:
            qwen_inputs: Słownik z obserwacjami:
                - input_ids: Tokenizowana instrukcja (np. "pick up the red cube")
                - pixel_values: Obraz z kamery robota
                - state: Obecne pozycje stawów robota
        
        Returns:
            Słownik z:
                - normalized_actions: Przewidziane akcje [1, num_chunks, action_dim]
                  (znormalizowane - trzeba denormalizować przed wysłaniem do robota)
        
        Przepływ danych (identyczny jak w forward, ale bez obliczania loss):
        1. obraz + instrukcja → VLM → scene understanding
        2. scene understanding + proprio → Action Model → predicted actions
        3. Zwróć akcje jako numpy array
        
        UWAGA: Output jest znormalizowany!
        Przed wysłaniem do robota musisz:
        1. Denormalizować akcje (np. z [-3, 3] do rzeczywistych pozycji stawów)
        2. Zastosować safety limits (sprawdzić czy w dozwolonym zakresie)
        """

        # Ekstrakcja stanu proprioceptywnego
        state = qwen_inputs["state"]
        state = state.unsqueeze(1) if state.dim() == 2 else state  # Wymiar [B, 1, D]

        # FAZA 1: VLM Processing - zrozumienie sceny
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs["pixel_values"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                output_hidden_states = True,       
                return_dict=True,  
            )
            # Embedding sceny z ostatniej warstwy VLM
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            
        # Upewnij się że state jest na tym samym device i ma ten sam dtype co embeddingi
        state = state.to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        
        # FAZA 2: Action Prediction - generowanie akcji
        with torch.autocast("cuda", dtype=torch.float32):
            # Wywołaj metodę predict_action z Action Model
            # Flow matching generuje akcje poprzez denoising process
            pred_actions = self.action_model.predict_action(
                last_hidden,  # Scene embeddings z VLM
                state,  # Obecny stan robota (propriocepcja)
            )
        
        # Konwersja z tensora PyTorch do numpy array
        # .detach() - odłącz od grafu obliczeniowego
        # .cpu() - przenieś z GPU do CPU
        # .numpy() - konwertuj do numpy
        normalized_actions = pred_actions.detach().cpu().numpy()
        
        return {"normalized_actions": normalized_actions}

