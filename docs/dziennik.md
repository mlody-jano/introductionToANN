# Dziennik prac i problemów
## Systemowy Python 3.9

- **Problem:** na macOS domyślnie uruchamiał się systemowy Python 3.9.6. Najnowsze wersje PyTorch
  wymagają Pythona 3.10+, więc instalowała się starsza wersja biblioteki.
- **Rozwiązanie:** instalacja Pythona 3.13 i osobne środowisko wirtualne (venv) dla projektu.
- **Wniosek:** środowisko projektu powinno być odizolowane; wersje bibliotek zapisane w `requirements.txt`.

## Pamięć przy pełnej rozdzielczości

- **Problem:** 17 736 obrazów 256×256×3 to ok. 3,5 GB jako `uint8`, ale ok. 14 GB po zamianie na
  `float32`, co przekracza możliwości komputera z 16 GB współdzielonej pamięci.
- **Rozwiązanie:** wczytywanie pliku `.h5` porcjami, zapis obrazów jako `uint8`, zamiana na
  liczby zmiennoprzecinkowe dopiero dla pojedynczej partii w trakcie treningu.

## Wybór rozdzielczości wejścia

- **Problem:** pełna rozdzielczość 256×256 to 16 razy więcej pikseli niż 64×64, więc trening
  trwałby ok. 40 min zamiast 2–3 min, co uniemożliwia serię kilkudziesięciu eksperymentów.
- **Rozwiązanie:** główne eksperymenty na 64×64 (podobnie Dieleman i in., 2015), a wpływ
  rozdzielczości (64, 128, 256) zbadany w osobnym eksperymencie na najlepszej architekturze.

## Możliwe duplikaty galaktyk

- **Problem:** ta sama galaktyka mogła trafić do zbioru kilka razy (różne kampanie Galaxy Zoo).
  Gdyby kopie znalazły się w zbiorze treningowym i testowym, wynik na teście byłby zawyżony.
- **Rozwiązanie:** wykrywanie duplikatów po współrzędnych na niebie (ra, dec, promień 1″)
  i po identycznych pikselach.
- **Wynik na prawdziwych danych:** 60 galaktyk występuje w zbiorze 2–3 razy (121 z 17 736 obrazów;
  59 par o tych samych współrzędnych, 61 par o identycznych pikselach). We **wszystkich 60** grupach
  kopie mają **różne etykiety**: ten sam obraz jest opisany jako dwie różne klasy.
- **Decyzja:** usunięcie wszystkich 121 obrazów, bo nie da się ustalić, która etykieta jest poprawna;
  zostaje 17 615 obrazów (strata 0,7%). Lista usuniętych obrazów: `results/tables/duplicates.csv`,
  przykłady: `results/figures/duplicates_conflicts.png`. Wariant bez usuwania: `--keep-duplicates`.
- **Wniosek:** etykiety Galaxy10 to ocena ludzi (głosy wolontariuszy), a nie pewna prawda;
  dla części galaktyk granica między klasami jest niejednoznaczna.

## Nowy format pliku

- Dane surowe (plik HDF5 udostępniony przez autorów zbioru) pozostają niezmienione; 
dane przetworzone są zapisywane w formacie NPZ (natywny format NumPy), ponieważ po 
zmniejszeniu rozdzielczości mieszczą się w pamięci i są wczytywane w całości na 
początku treningu, a format ten nie wymaga dodatkowych bibliotek.