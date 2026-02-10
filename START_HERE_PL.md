# 🚀 START HERE - Rozpocznij Tutaj

## Witajcie Studenci Politechniki Rzeszowskiej! 👋

To repozytorium zostało specjalnie przygotowane dla Was do nauki frameworku **UnifoLM-VLA** w kontekście pracy z robotem humanoidalnym **Unitree G1 EDU-U6**.

---

## 📚 Od Czego Zacząć?

### Krok 1: Przeczytaj Podstawową Dokumentację

1. **[README_pl.md](README_pl.md)** - Główna dokumentacja po polsku
   - Co to jest UnifoLM-VLA?
   - Jak działa Vision-Language-Action?
   - Instalacja i pierwsze kroki
   - Przegląd dostępnych modeli i danych

### Krok 2: Studiuj Przewodniki

2. **[GUIDE_FOR_STUDENTS_PL.md](GUIDE_FOR_STUDENTS_PL.md)** - Szczegółowy przewodnik dla studentów
   - Podstawowe pojęcia (VLA, VLM, Action Model, etc.)
   - Jak działa kod krok po kroku
   - Architektura i struktura projektu
   - Praktyczne ćwiczenia
   - Troubleshooting
   - Propozycje projektów semestralnych

3. **[UNITREE_G1_PRACTICAL_GUIDE_PL.md](UNITREE_G1_PRACTICAL_GUIDE_PL.md)** - Praktyczny przewodnik robota G1
   - Specyfikacja techniczna robota Unitree G1
   - Pierwszy kontakt z robotem
   - Zbieranie danych treningowych
   - Pipeline: od demonstracji do wytrenowanego modelu
   - Deployment na prawdziwym robocie
   - Bezpieczeństwo i best practices

---

## 🗺️ Mapa Repozytorium

```
unifolm-vla/
│
├── 📄 START_HERE_PL.md           ← JESTEŚ TUTAJ!
├── 📄 README_pl.md               ← Główna dokumentacja PL
├── 📄 GUIDE_FOR_STUDENTS_PL.md   ← Przewodnik dla studentów
├── 📄 UNITREE_G1_PRACTICAL_GUIDE_PL.md  ← Przewodnik robota G1
│
├── 📁 src/unifolm_vla/           ← Główny kod frameworku
│   ├── model/                    ← Definicje modeli (VLM + Action Model)
│   │   └── framework/
│   │       └── unifolm_vla.py    ← ⭐ GŁÓWNA KLASA - zacznij tutaj!
│   │
│   ├── rlds_dataloader/          ← Ładowanie i przetwarzanie danych
│   │   └── constants.py          ← ⭐ STAŁE - wymiary, normalizacja
│   │
│   └── training/                 ← Kod treningowy
│       └── train_unifolm_vla.py  ← Główny skrypt treningu
│
├── 📁 scripts/                   ← Skrypty uruchomieniowe
│   ├── run_scripts/
│   │   └── run_unifolm_vla_train.sh  ← ⭐ TRENING - użyj tego
│   │
│   └── eval_scripts/
│       ├── run_eval_libero.sh    ← ⭐ EWALUACJA w symulacji
│       └── run_real_eval_server.sh  ← ⭐ DEPLOYMENT na robocie
│
├── 📁 prepare_data/              ← Skrypty przygotowania danych
│   └── convert_lerobot_to_hdf5.py  ← ⭐ Konwersja danych
│
└── 📁 experiments/               ← Eksperymenty i ewaluacje
    └── LIBERO/                   ← Benchmark symulacyjny
```

---

## 🎯 Zalecana Ścieżka Nauki

### Tydzień 1-2: Zrozumienie Podstaw

1. **Przeczytaj dokumentację:**
   - README_pl.md (co to jest VLA?)
   - GUIDE_FOR_STUDENTS_PL.md (sekcje: Wprowadzenie, Podstawowe Pojęcia)

2. **Zrozum architekturę:**
   - Przeczytaj "Jak Działa UnifoLM-VLA" w przewodniku
   - Prześledzić przepływ danych w diagramach

3. **Eksploruj kod:**
   - Otwórz `src/unifolm_vla/model/framework/unifolm_vla.py`
   - Przeczytaj komentarze PL w klasie `Unifolm_VLA`
   - Zrozum metody `forward()` i `predict_action()`

### Tydzień 3-4: Pierwszy Projekt

4. **Instalacja środowiska:**
   - Wykonaj kroki z README_pl.md (Instalacja)
   - Zainstaluj zależności
   - Pobierz pretrenowane modele

5. **Ewaluacja w symulacji:**
   - Zainstaluj LIBERO
   - Uruchom `scripts/eval_scripts/run_eval_libero.sh`
   - Obserwuj jak model działa w symulacji

### Tydzień 5-6: Własne Dane

6. **Zbieranie demonstracji:**
   - UNITREE_G1_PRACTICAL_GUIDE_PL.md (sekcja: Zbieranie Danych)
   - Zbierz 50-100 demonstracji prostego zadania
   - Konwertuj dane: LeRobot → HDF5 → RLDS

7. **Trening modelu:**
   - Skonfiguruj `scripts/run_scripts/run_unifolm_vla_train.sh`
   - Uruchom trening na swoich danych
   - Monitoruj logi w TensorBoard

### Tydzień 7-8: Deployment

8. **Test na prawdziwym robocie:**
   - UNITREE_G1_PRACTICAL_GUIDE_PL.md (sekcja: Deployment)
   - Uruchom serwer inferencji
   - Testuj ostrożnie z safety limits

---

## 💡 Kluczowe Pliki z Komentarzami PL

Te pliki mają szczegółowe komentarze po polsku - czytaj je uważnie:

### Python Files:

1. **`src/unifolm_vla/model/framework/unifolm_vla.py`**
   - Główna klasa modelu VLA
   - Metoda `forward()` - jak trenować
   - Metoda `predict_action()` - jak używać

2. **`src/unifolm_vla/rlds_dataloader/constants.py`**
   - Stałe dla różnych robotów
   - Wymiary akcji i propriocepcji
   - Typy normalizacji

3. **`prepare_data/convert_lerobot_to_hdf5.py`**
   - Konwersja danych
   - Format LeRobot → HDF5

### Shell Scripts:

4. **`scripts/run_scripts/run_unifolm_vla_train.sh`**
   - Kompletna instrukcja treningu
   - Wszystkie parametry wyjaśnione

5. **`scripts/eval_scripts/run_eval_libero.sh`**
   - Ewaluacja w symulacji LIBERO
   - Konfiguracja testów

6. **`scripts/eval_scripts/run_real_eval_server.sh`**
   - Serwer do pracy z prawdziwym robotem
   - Architektura klient-serwer

---

## 🆘 Potrzebujesz Pomocy?

### Dokumentacja
1. **Podstawy:** README_pl.md
2. **Szczegóły techniczne:** GUIDE_FOR_STUDENTS_PL.md
3. **Robot G1:** UNITREE_G1_PRACTICAL_GUIDE_PL.md

### Troubleshooting
- GUIDE_FOR_STUDENTS_PL.md - sekcja "Troubleshooting"
- Typowe problemy i rozwiązania

### Pytania?
- Sprawdź FAQ w przewodnikach
- Przeczytaj komentarze w kodzie
- Konsultuj się z prowadzącym

---

## 📋 Checklist Przed Rozpoczęciem Projektu

- [ ] Przeczytałem README_pl.md
- [ ] Przeczytałem odpowiednie sekcje GUIDE_FOR_STUDENTS_PL.md
- [ ] Rozumiem czym jest VLA (Vision-Language-Action)
- [ ] Zainstalowałem środowisko (conda, dependencies)
- [ ] Pobrałem pretrenowane modele
- [ ] Uruchomiłem ewaluację w LIBERO (opcjonalnie)
- [ ] Zapoznałem się z bezpieczeństwem pracy z robotem G1
- [ ] Mam plan projektu semestralnego

---

## 🎓 Projekty Semestralne - Propozycje

Zobacz GUIDE_FOR_STUDENTS_PL.md, sekcja "Projekt Semestralny - Propozycje":

1. **Projekt 1:** Nowe Zadanie dla Robota G1
2. **Projekt 2:** Improved Action Prediction
3. **Projekt 3:** Multi-Robot Collaboration

Każdy projekt z:
- Opisem zadania
- Krokami realizacji
- Kryteriami oceny

---

## 🚦 Status Repozytorium

✅ **Gotowe do użycia przez studentów**

- ✅ Dokumentacja po polsku (133KB)
- ✅ Komentarze w kluczowym kodzie
- ✅ Instrukcje krok po kroku
- ✅ Przykłady i ćwiczenia
- ✅ Troubleshooting
- ✅ Projekty przykładowe
- ✅ Bezpieczeństwo i best practices

---

## 📞 Kontakt

**Pytania dotyczące:**
- **Frameworku UnifoLM-VLA:** Zobacz dokumentację lub Issues na GitHub
- **Robota Unitree G1:** UNITREE_G1_PRACTICAL_GUIDE_PL.md
- **Projektu/Kursu:** Konsultacje z prowadzącym

---

## 🎉 Powodzenia!

Życzymy owocnej nauki i sukcesów w projektach z robotem Unitree G1!

**Zespół przygotowujący materiały**  
*Politechnika Rzeszowska, 2026*

---

**Następny krok:** Przeczytaj [README_pl.md](README_pl.md) 📖
