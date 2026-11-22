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


## Wstępne badania MLP I CNN
- dla seed 0 MLP +/- 53,3% a CNN 68,3% po 5 epokach
- CNN bazowa: najlepsza epoka 6 (val 68,1%), od epoki 7 przeuczenie: strata treningowa 0,05, walidacyjna 2,80 w epoce 30 

## Pierwsze wyniki tak dla porownania od startu sa w `results/noTune`
**Wnioski jakie wynikaja z tego:**
- Zaburzona to najsłabsza klasa i najczęściej zamienia się w spiralę luźną. Obie mają nieregularny kształt. To prawdopodobnie te same klasy, z którymi mieli problem ludzie z Galaxy Zoo. Przy usuwaniu duplikatów ze sprzecznymi etykietami najwięcej obrazów ubyło właśnie z klas Zaburzona (−30) i Spirala luźna (−33). Trzeba szobaczyc w `results/tables/duplicates.csv` czy to rzeczywiście te pary. Jak tak to spoko wniosek - sieć myli te same klasy, które mylą ludzie.
- Cygarowata gładka ma recall tylko 46%, a do tego tylko 50 obrazów walidacyjnych i 233 treningowe. 18% z nich trafia do klasy „z boku, ze zgrubieniem”, co ma sens, bo obie są wydłużone. Sieć widziała mało przykładów tej klasy. Hipoteza: pomogą wagi klas albo augmentacja.
- Mocne klasy to te o wyraźnym, prostym kształcie: okrągła gładka (89%), z boku bez zgrubienia (85%), z boku ze zgrubieniem (80%), pośrednia gładka (77%). Galaktyki widziane z boku mylą się czasem ze sobą (po 8% w obie strony), bo różni je tylko zgrubienie w środku.

## Wyniki po pierwszym usprawnienu
- sa w folderze `/results/runs/<model>_aug`

## Chyba odpusczenie sprawdzenia rozdzielczosci 128x128
- Za duzo czasu a nie tak istotne chyba na tym etapie
- kod jest gotowy zeby robic ale testy raczej nie beda robione 

## Serie
1. Współczynnik uczenia (Adam)

python src/run_series.py --param lr --values 0.01 0.003 0.001 0.0003 --augment

Zamiana wielkość kroku przy poprawianiu wag. Wartości rosną mniej więcej co trzy razy, bo liczy się rząd wielkości, a nie dokładna liczba.
2. Metoda gradientowa: SGD zamiast Adama

`python src/run_series.py --param lr --values 0.1 0.03 0.01 --series lr_sgd --optimizer sgd --augment`

Czysty spadek gradientowy ma jeden stały krok dla wszystkich wag, a Adam dobiera krok osobno dla każdej wagi. SGD potrzebuje zwykle większego lr, dlatego wartości są inne niż w serii 1. --series lr_sgd jest konieczne. Bez tego wyniki trafiłyby do folderu lr/, a przebieg lr/0.01_s0 z Adamem zostałby uznany za „już policzony” i pominięty.
3. Liczba bloków konwolucyjnych (głębokość)

`python src/run_series.py --param blocks --values 2 3 4 5 --augment`

4. Liczba filtrów (szerokość)

`python src/run_series.py --param filters --values 16 32 64 --augment`

Wartość dotyczy pierwszego bloku, a w kolejnych jest podwajana, więc mamy 16/32/64, 32/64/128 i 64/128/256.
5. Neurony w warstwie ukrytej klasyfikatora

`python src/run_series.py --param hidden --values 64 128 256 512 --aug`ment

To warstwa po spłaszczeniu, która z wykrytych cech składa decyzję.
6. Funkcja aktywacji

`python src/run_series.py --param activation --values relu leakyrelu elu tanh sigmoid --augment`

Porównanie klasycznych funkcji (sigmoid, tanh) z nowoczesnymi z rodziny ReLU.
7. Dropout

`python src/run_series.py --param dropout --values 0 0.3 0.5 --augment`

Regularyzacja: jaki ułamek neuronów wyłączamy losowo przy treningu, żeby sieć nie polegała na pojedynczych neuronach.
8. Normalizacja wsadowa

`python src/run_series.py --param batchnorm --values off on --augment`

Po każdej warstwie konwolucyjnej wartości są normalizowane w obrębie partii. Często przyspiesza i stabilizuje uczenie.
9. Kara za duże wagi (weight decay)

`python src/run_series.py --param weight-decay --values 0 0.0001 0.001 --augment`

Inny rodzaj regularyzacji: optymalizator „ściąga” wagi w stronę zera, więc sieć ma trudniej zapamiętać szczegóły zbioru treningowego.
10. Wagi klas

`python src/run_series.py --param class-weights --values off on --augment`

Pomyłka na rzadkiej klasie kosztuje więcej. Ta seria sprawdza twoją hipotezę z dziennika: cygarowata gładka ma tylko 233 obrazy treningowe i może jej pomóc ważenie.
11. Augmentacja

`python src/run_series.py --param augment --values off on`

Bez --augment na końcu, bo tu to on jest badanym parametrem. 
12. MLP: liczba warstw

`python src/run_series.py --param hidden --values "256" "512 256" "1024 512 256" --series mlp_hidden --model mlp`

Jedna, dwie i trzy warstwy ukryte. Cudzysłowy są potrzebne, bo jedna wartość to kilka liczb. MLP liczymy bez augmentacji, tak jak mlp_base.
13. Warunek stopu (nowa)

`python src/run_series.py --param patience --values 3 5 10 --augment`

Ile epok bez poprawy straty walidacyjnej czekanie zanim przerwie trening. Polecenie wymienia warunek stopu wprost.
