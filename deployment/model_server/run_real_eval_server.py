import os
import sys
import logging
import argparse
import torch
import json
import time
import base64
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import json_numpy
json_numpy.patch()
import tensorflow as tf
from qwen_vl_utils import process_vision_info
import traceback
from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.rlds_dataloader.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
unifolm_vla_IMAGE_SIZE = 224  


def check_image_format(image: Any) -> None:
    """
    Weryfikacja formatu wejściowego obrazu.

    Args:
        image: Obraz do sprawdzenia

    Raises:
        AssertionError: Jeśli format obrazu jest nieprawidłowy
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Wykryto nieprawidłowy format obrazu! Upewnij się, że obraz wejściowy jest "
        "tablicą numpy o kształcie (H, W, 3) i typie danych np.uint8!"
    )


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Zmiana rozmiaru obrazu do oczekiwanego rozmiaru wejściowego polityki.

    Używa tego samego schematu zmiany rozmiaru co w pipeline danych treningowych,
    aby zachować zgodność rozkładów.

    Args:
        img: Tablica numpy zawierająca obraz
        resize_size: Docelowy rozmiar jako int (kwadrat) lub krotka (wysokość, szerokość)

    Returns:
        np.ndarray: Obraz po zmianie rozmiaru
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Zmiana rozmiaru przy użyciu tego samego pipeline co w RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Kodowanie jako JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Dekodowanie
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()

def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Kadrowanie środkowe obrazu i przywrócenie jego oryginalnych wymiarów.

    Używa tej samej logiki co w pipeline danych treningowych, aby zachować zgodność rozkładów.

    Args:
        image: Tensor TF o kształcie (batch_size, H, W, C) lub (H, W, C) z wartościami w [0, 1]
        crop_scale: Pole kadru centralnego względem oryginalnego obrazu
        batch_size: Rozmiar batcha

    Returns:
        tf.Tensor: Obraz po kadrowaniu i zmianie rozmiaru
    """
    # Obsługa wejść 3D poprzez dodanie wymiaru batcha, jeśli potrzebne
    assert image.shape.ndims in (3, 4), "Obraz musi być tensorem 3D lub 4D"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Obliczanie wymiarów kadru (uwaga: używamy sqrt(crop_scale) dla h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Tworzenie obwiedni dla kadru
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Zastosowanie kadrowania i zmiany rozmiaru
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (unifolm_vla_IMAGE_SIZE, unifolm_vla_IMAGE_SIZE)
    )

    # Usunięcie wymiaru batcha, jeśli został dodany
    if expanded_dims:
        image = image[0]

    return image

def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Kadrowanie środkowe obrazu w celu dopasowania do rozkładu danych treningowych.

    Args:
        image: Wejściowy obraz (PIL lub tablica numpy)

    Returns:
        Image.Image: Przycięty obraz PIL
    """
    batch_size = 1
    crop_scale = 0.9

    # Konwersja do tensora TF, jeśli potrzebne
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Konwersja do float32 w zakresie [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Zastosowanie kadrowania środkowego i zmiany rozmiaru
    image = crop_and_resize(image, crop_scale, batch_size)

    # Konwersja z powrotem do oryginalnego typu danych
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Konwersja do obrazu PIL
    return Image.fromarray(image.numpy()).convert("RGB")

def unnormalize_action(normalized_actions: np.ndarray, action_norm_stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])

    actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

    return actions

def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalizacja danych proprioceptywnych do rozkładu danych treningowych.

    Args:
        proprio: Surowe dane proprioceptywne
        norm_stats: Statystyki normalizacyjne

    Returns:
        np.ndarray: Znormalizowane dane proprioceptywne
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Wykryto nieobsługiwany typ normalizacji akcji/propriocepcji!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio

class Unifolm_VLA_Server:
    """Serwer FastAPI do inferencji modelu VLA"""
    
    def __init__(self, args):
        self.args = args
        logging.info("Ładowanie modelu VLA z: %s", args.ckpt_path)
        # TODO: should auto detect framework from model path
        vla = baseframework.from_pretrained(args.ckpt_path, vlm_pretrained_path=args.vlm_pretrained_path)

        if args.use_bf16:
            logging.info("Konwersja modelu do bfloat16")
            vla = vla.to(torch.bfloat16)
        
        vla = vla.to("cuda").eval()
        self.vla = vla        
        self.norm_stats_action = vla.norm_stats[self.args.unnorm_key]['action']
        self.norm_stats_proprio = vla.norm_stats[self.args.unnorm_key]['proprio']
        self.processor = vla.qwen_vl_interface.processor
        logging.info("Model załadowany pomyślnie")

    def prepare_images_for_vla(self, images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
        """
        Przygotowanie obrazów do wejścia modelu VLA poprzez zmianę rozmiaru i kadrowanie.

        Args:
            images: Lista wejściowych obrazów jako tablice numpy
            cfg: Obiekt konfiguracyjny z parametrami

        Returns:
            List[Image.Image]: Przetworzone obrazy gotowe dla modelu
        """
        processed_images = []

        for image in images:
            # Weryfikacja formatu
            check_image_format(image)

            # Zmiana rozmiaru, jeśli potrzebne
            if image.shape != (unifolm_vla_IMAGE_SIZE, unifolm_vla_IMAGE_SIZE, 3):
                image = resize_image_for_policy(image, unifolm_vla_IMAGE_SIZE)

            # Konwersja do obrazu PIL
            pil_image = Image.fromarray(image).convert("RGB")

            # Zastosowanie kadrowania środkowego, jeśli skonfigurowano
            if cfg.center_crop:
                pil_image = center_crop_image(pil_image)

            processed_images.append(pil_image)

        return processed_images
        
    
    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            t1 = time.time()
            if double_encode := "encoded" in payload:
                # Obsługa przypadków, gdy `json_numpy` jest trudne do zainstalowania,
                # a tablice numpy są "podwójnie kodowane" jako ciągi znaków
                assert len(payload.keys()) == 1, "Dozwolone jest tylko zakodowane payload!"
                payload = json.loads(payload["encoded"])

            observations = payload['observations']
            all_images = []
            for observation in observations:    
                all_images.append(observation["full_image"])
            for observation in observations:
                all_images.extend([observation[k] for k in observation.keys() if "wrist" in k])
            instruction = observations[0]["instruction"]
            
            if observations[0].get("task_name", None) is not None:
                task_name = observations[0].get("task_name", None)
                self.norm_stats_action = self.vla.norm_stats[task_name]['action']
                self.norm_stats_proprio = self.vla.norm_stats[task_name]['proprio']

            # Przetwarzanie obrazów
            all_images = self.prepare_images_for_vla(all_images, self.args)
            lang = instruction.lower()
            text = f"The task is \"{lang}\"."
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[
                            {"type": "image", "image": img}
                            for img in all_images
                        ],
                        {"type": "text", "text": text},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            batch_input = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            proprios = []
            for observation in observations:
                proprios.append(observation["state"])
                
            batch_input["state"] = torch.from_numpy(normalize_proprio(np.stack(proprios, axis=0), self.norm_stats_proprio)).unsqueeze(0).to(DEVICE)

            batch_input["input_ids"] = batch_input["input_ids"].to(DEVICE)
            batch_input["attention_mask"] = batch_input["attention_mask"].to(DEVICE)
            batch_input["pixel_values"] = batch_input["pixel_values"].to(DEVICE)
            batch_input["image_grid_thw"] = batch_input["image_grid_thw"].to(DEVICE)
            
            action = self.vla.predict_action(
                qwen_inputs=batch_input,
            )

            action = unnormalize_action(action['normalized_actions'][0], self.norm_stats_action)
            inference_time = time.time() - t1
            logging.info(f"VLA inference time: {inference_time:.3f}s")
            
            print(f"get_vla_action time: {time.time() - t1}")
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  
            logging.error(traceback.format_exc())
            logging.warning(
                "Twoje żądanie spowodowało błąd; upewnij się, że żądanie jest zgodne z oczekiwanym formatem:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"
        
    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        """Uruchomienie serwera FastAPI"""
        logging.info("Tworzenie serwera FastAPI...")
        self.app = FastAPI(
            title="VLA Model Server",
            description="API inferencji modelu VLA (Vision-Language-Action)",
            version="1.0.0"
        )

        self.app.post("/act")(self.get_server_action)
        
        logging.info(f"Uruchamianie serwera pod adresem http://{host}:{port}")
        logging.info(f"Punkt końcowy API: POST http://{host}:{port}/act")
        logging.info("Naciśnij Ctrl+C, aby zatrzymać serwer")

        uvicorn.run(self.app, host=host, port=port, log_level="info")


def deploy(args):
    """Wdrożenie serwera modelu VLA"""
    server = Unifolm_VLA_Server(args)
    server.run(host=args.host, port=args.port)

def build_argparser():
    """Budowanie parsera argumentów wiersza poleceń"""
    parser = argparse.ArgumentParser(
        description="Wdrożenie modelu VLA jako serwera FastAPI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/path/to/your/ckpt.pt",
        help="Ścieżka do punktu kontrolnego modelu lub nazwa modelu na HuggingFace"
    )
    parser.add_argument(
        "--vlm_pretrained_path",
        type=str,
        default=None,
        help="Ścieżka do punktu kontrolnego modelu VLM lub nazwa modelu na HuggingFace"
    )
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default="new_embodiment",
        help="Nazwa zbioru danych"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Adres nasłuchiwania serwera"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8777,
        help="Port nasłuchiwania serwera"
    )
    parser.add_argument(
        "--use_bf16", 
        action="store_true",
        default=True,
        help="Czy używać precyzji bfloat16"
    )
    parser.add_argument(
        "--center_crop", 
        action="store_true",
        help="Czy stosować podwójne kodowanie"
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    
    parser = build_argparser()
    args = parser.parse_args()
    
    deploy(args)
