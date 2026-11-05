# Prova 3D: ala di Zerihan con endplate (kit per il PC fisso)

Preparato il 03/11/2026. È una **prova di fattibilità**, non produzione: serve a misurare quanto costa un caso 3D (celle, tempo di mesh, tempo per iterazione, RAM, iterazioni fino a convergenza) e se il risultato è sensato, per decidere se e quanto dataset 3D si può fare. Il codice del dataset 2D (`../dataset/`) non viene toccato: il kit ne legge solo la geometria e la versione di OpenFOAM fissata.

## Perché Zerihan e non un'ala F1 "vera"

- È **lo stesso modello** da cui nasce il dataset 2D: stessa sezione (contorni `smooth_te` congelati), stessa h/c, stesso Re. Ogni caso 3D si confronta direttamente con il caso 2D corrispondente.
- Esistono le **forze 3D misurate** (bilancia, ala intera con endplate): Fig. 36b della tesi, massimo CL = 3,028 a h/c = 0,079 (flap datum). Il 2D non può riprodurle per costruzione (D1, punto 6 del documento D2); il 3D sì.
- Si parametrizza come il 2D: le 80 ali del dataset (`../dataset/geometries/generated/`) si possono estrudere allo stesso modo.
- Seconda configurazione da provare dopo: **Imperial Front Wing** (McLaren MP4-17D, 3 elementi, endplate, footplate, ruota; open access, Imperial HPC repository, DOI 10.14469/hpc/6049). Più realistica, ma una sola forma fissa da CAD.

## Geometria

| Voce | Valore | Fonte |
|---|---|---|
| Sezione | main + flap, flap datum, α nominale 1° | `frontwing.geometry.zerihan_wing` (tabelle 3–4, contorni congelati) |
| Apertura | 1100 mm (semi-modello: 550 mm, simmetria a metà apertura) | tesi §2.2.1 |
| Endplate | rettangolari 400 × 170 × 4 mm, paralleli al suolo a α = 1° | tesi §2.2.2 e §2.7 |
| Posizione dell'endplate | bordo d'attacco 3,1 mm davanti al LE del main, bordo inferiore 12,5 mm sotto il punto più basso dell'ala | **ipotesi**: ricavata dalla Fig. 1c (vedi sotto) |
| h/c | definizione di Zerihan (punto più basso a α nominale 0), come il caso di riferimento 2D | `ground_y_zerihan` |

**Posizione dell'endplate.** Il testo dà le dimensioni ma non la posizione rispetto all'ala. La Fig. 1c è uno schema in scala: `zerihan3d/fit_endplate_from_fig1c.py` sovrappone il profilo vero (α = 1°) al disegno con scala orizzontale fissata dalla lunghezza dell'endplate (400 mm) e scala verticale libera. Residuo 0,87 px (circa 0,5 mm); l'altezza dell'endplate che ne esce è 170,3 mm contro i 170 mm del testo, una verifica indipendente. Incertezza stimata ±1 mm. Il risultato è in `zerihan3d/endplate_fig1c.json`; la sovrapposizione è in `endplate_fig1c_overlay.png`. Per α ≠ 1° l'endplate ruota con l'ala attorno allo stesso perno.

Alcune conseguenze: a h/c = 0,211 l'endplate sta a 67,5 mm dal suolo. A h/c = 0,053, la quota più bassa della tesi, scende a circa 7,5 mm. Il generatore si ferma se l'endplate tocca il suolo.

## Fisica e differenze rispetto al 2D

Restano uguali: Re = 7,5·10⁵ su c = 0,38 m, U = 30 m/s, k-ω SST, Tu 0,2%, ν_t/ν = 10 all'ingresso, suolo mobile a U∞, segno di CL positivo = deportanza.

Cambiano tre cose, da dichiarare:

1. **Leggi di parete.** Il 2D risolve lo strato limite (y⁺ ≈ 1). Il 3D usa `nutUSpaldingWallFunction` con prima cella a y⁺ di circa 30–150 e 3 layer. Risolvere lo strato limite in 3D costerebbe da 10 a 100 volte più celle. È un modello diverso: CL e Cd 3D e 2D non sono confrontabili al decimale.
2. **Solutore.** SIMPLE robusto (p 0,3, U 0,7, k e ω 0,5, convezione di k e ω al primo ordine) con partenza da campo uniforme. Il preset SIMPLEC del 2D con partenza da potentialFoam è divergito in 30 iterazioni sulla mesh smoke.
3. **Dominio.** Aria libera con contorni lontani in slip: x da −1,5 a 4,0 m, altezza 2,0 m, semi-apertura del dominio 1,75 m. Nessuna galleria e nessuna correzione di bloccaggio, come nel 2D.

`Aref = c · b/2`: per il semi-modello CL è direttamente confrontabile con il CL della bilancia (ala intera).

## Mesh (snappyHexMesh)

| Livello | d0 | Celle | Uso |
|---|---|---|---|
| smoke | 0,10 m | 1,04 M (misurate) | controllo della catena; numeri solo indicativi |
| L0 | 0,075 m | circa 2,5 M (stima ×2,4) | **la prova** |
| L1 | 0,05 m | circa 8–9 M (stima ×3,4) | facoltativo: primo numero di sensibilità alla mesh |

La superficie dell'ala è al livello 5 (d0/32), ma al livello 6 dove c'è curvatura e sul 25% posteriore di ogni elemento, che nell'STL è la regione `aft`: il flap ha gli ultimi circa 35 mm spessi 1 mm. Gli spigoli del bordo d'uscita sono al livello 7, l'endplate ai livelli 4 e 5, e ci sono tre box di scia. Le patch sono `main_fore`, `main_aft`, `flap_fore`, `flap_aft` ed `endplate`.

## Come lanciarlo (PC fisso, WSL2)

Il PC fisso è libero solo se la produzione 2D è ferma (lo script controlla `production.pid` e si rifiuta di partire).

```bash
# copiare la cartella dataset3d accanto a dataset/ nel pacchetto installato, poi:
cd ~/frontwing/F1-Front-Wing-Physics-Informed-NN/course/_instructor/dataset3d
chmod +x run_trial.sh zerihan3d/Allrun
./run_trial.sh smoke            # 1) controllo: 1 M celle, 300 iterazioni, circa 5–15 min
./run_trial.sh L0               # 2) la prova: fino a 3000 iterazioni
NP=6 ./run_trial.sh L1          # 3) facoltativo, di notte; serve memory=20GB in .wslconfig
```

- **Output.** Ogni prova crea una cartella nuova in `data/frontwing_runs3d/` e non sovrascrive nulla. L'avanzamento si segue con `tail -f <caso>/log.Allrun` e `log.simpleFoam`.
- **Al termine.** `report3d.json` e `report3d.png` contengono: celle, qualità della mesh e copertura dei layer, tempi di ogni fase, picco di RAM, s/iterazione, µs per cella e iterazione per rank, CL/CD medi con ampiezza e deriva sulle ultime 500 iterazioni, confronto con la Fig. 36b e con il 2D, e la proiezione in giorni per 20, 80, 160 e 400 casi.
- **ParaView.** `case.foam` con "Decomposed Case". La distribuzione del carico lungo l'apertura si ricava in ParaView dalla pressione di parete (un function object `binField` è stato lasciato fuori perché non era provabile qui).

## Risultati della prova smoke nel cloud (03/11, OpenFOAM v1912, 2 core Xeon, 8 GB)

Serve solo a verificare che la catena funzioni; i numeri veri vanno presi sul PC fisso con v2606.

| Voce | Valore |
|---|---|
| Celle | 1 035 401 (castellata 0,80 M, dopo i layer 1,04 M; i layer sono circa il 40% delle celle). Mesh identica cella per cella in due esecuzioni |
| Qualità | checkMesh "Mesh OK"; non ortogonalità massima 64,9°, skewness massima 3,9 |
| Layer | 3 strati, copertura 93–97% su main e flap, 87% sull'endplate |
| Tempo di mesh | snappyHexMesh 179–197 s su 2 rank |
| Solutore | 2,52 s/iterazione su 2 rank, cioè **4,9 µs per cella e iterazione per rank**; RAM circa 1,7 GB per 1 M celle (picco 2,3 GB con snappy) |
| Convergenza | 400 iterazioni: residui iniziali scesi di 4,5–5 ordini (p 2,6e-5, U ~3e-5); non è un criterio sufficiente da solo |
| CL di pressione (deportanza +), h/c 0,211 | 2,25 (iter. 100), 2,320 (300), 2,303 (400): ancora in assestamento (−0,7% tra 300 e 400) |
| CD di pressione | 0,259 (300), 0,258 (400) |
| Riferimenti | sperimentale ≈ 2,63 (Fig. 36b, letto a occhio, ±0,03); 2D del dataset 4,279 |

Il Debian OpenFOAM v1912 del cloud non riesce a eseguire nessun function object (errore "sha1" anche con un solo `yPlus`, difetto di quella build). Per questo CL e CD qui sono solo la parte di pressione, integrata dai campi esportati; y⁺ non è stato misurato. Sul PC fisso con v2606 gli stessi function object del 2D funzionano: `forceCoeffs` per ogni patch, `yPlus`, `solverInfo`.

Lettura: sulla mesh più grossa il 3D dà circa il 12% in meno del dato sperimentale, contro il +63% del 2D. L'effetto dell'ala finita è nell'ordine giusto. Non è una validazione: mesh grossa, non a convergenza, solo pressione e un solo punto.

**Stima per il PC fisso, da verificare con L0.** Se un rank del Ryzen 7600X va da 1 a 2 volte questo Xeon, cioè 2,5–5 µs per cella e iterazione:

| Livello | Celle | s/iterazione (6 rank) | Caso completo (mesh + 1500–3000 iterazioni) | RAM |
|---|---|---|---|---|
| L0 | circa 2,5 M | 1–2 | circa 1–2 h | circa 4–5 GB |
| L1 | circa 8,5 M | 3,5–7 | circa 3–6 h | circa 14–16 GB |

Con L0 sono circa 12–24 casi al giorno: 100 casi in 4–8 giorni, mentre un equivalente dei 1234 casi 2D richiederebbe 2–3 mesi. Un dataset 3D realistico è quindi un **sottoinsieme** (per esempio 20 geometrie × 5–6 condizioni), coerente con il principio "geometrie prima dei casi".

## Risultati L0 sul portatile (03/11, OpenFOAM v2606, i7-13705H, 6 rank)

Caso `data/frontwing_runs3d/zerihan3d_L0_h0.211_np6_20261103T150218Z`, h/c 0,211, flap datum, α 1°.

| Voce | Valore |
|---|---|
| Celle | 1 710 431, contro i circa 2,5 M stimati; i layer passano le celle da 1,04 M a 1,71 M |
| Qualità | non ortogonalità massima 64,8°; 4 facce con skewness 5,0 sopra la soglia di checkMesh (4), unico controllo fallito |
| Layer | copertura 93,7–95,6% su main e flap, 84,5% sull'endplate |
| Mesh | 202 s |
| Solutore | 2,08 s/iterazione, cioè 7,3 µs per cella e iterazione per rank; picco di RAM 3,3 GB |
| Iterazioni | 2504, fermato a mano con `stopAt writeNow` |
| Durata totale | 1 h 31 min |
| Convergenza | CL fermo dall'iterazione 900 circa (variazione < 0,01% ogni 100 iterazioni). Sulle ultime 500: ampiezza 0,007%, deriva 0,00005%. I residui iniziali si fermano a p ≈ 2e-5…4e-4 (oscillano fra i correttori), U ≈ 5e-5, Uz 1,5e-4: `residualControl` non scatta mai |
| CL totale (deportanza +) | **2,307**: main 1,734, flap 0,565, endplate 0,007 |
| CD totale | 0,268: main 0,029, flap 0,230, endplate 0,009 |
| y⁺ medio | main 33 (davanti) e 15 (dietro), flap 19 e 14, endplate 35; massimo 158 sul main, minimo 0,3. Molte celle nella zona di raccordo, accettabile con Spalding |
| Confronto | sperimentale ≈ 2,63 (Fig. 36b, a occhio): **−12,3%**; 2D del dataset 4,279: +85% rispetto al 3D |

**Costo per il dataset.**
- Come lanciato (fino a 2500 iterazioni) un caso costa 1,5 h, cioè circa 16 casi al giorno sul portatile.
- I coefficienti però sono fermi già intorno a 900–1000 iterazioni. Con un arresto automatico sulla stabilità di CL e CD (come `conv-v1` del 2D, al posto di `residualControl`) un caso costerebbe circa 3,5 min di mesh più 35 min di solutore, cioè circa 40 min: circa 35 casi al giorno, 100 casi in 3 giorni.
- Da fare prima di un dataset: arresto automatico, L1 per la sensibilità alla mesh, una curva in h/c (per esempio 0,079, 0,132, 0,211, 0,395) contro la Fig. 36b.

## Criteri per dire "il 3D è fattibile"

1. L0 converge: CL con ampiezza e deriva < 0,5% sulle ultime 500 iterazioni, residui scesi di almeno 3 ordini.
2. CL a h/c 0,211 nell'ordine del dato sperimentale (≈ 2,63 letto dalla Fig. 36b, ±0,03 di lettura). Non è una validazione.
3. Il costo per caso (mesh + soluzione) permette almeno circa 100 casi in 1–2 settimane di PC fisso.
4. L1 cambia CL di poco (ordine 2–3%). Altrimenti L0 non basta e il costo sale di un fattore 3–4.

## Limiti noti

- Posizione dell'endplate ricavata da uno schema (±1 mm), non da una quota.
- Il modello in galleria aveva supporti, una cavità per il trasduttore e ritagli negli endplate (tesi §2.3): qui non ci sono.
- Le leggi di parete cambiano il modello rispetto al 2D; il confronto 2D/3D è di andamento.
- RANS stazionario su un flusso con vortici d'estremità e separazione vicino al suolo: sotto h/c circa 0,1 è ancora meno affidabile che in 2D.
- La decomposizione `hierarchical` dà risultati che dipendono dal numero di rank, come nel 2D: non si mescolano rank diversi.

## File

- `run_trial.sh`: lancio sul PC fisso (podman e immagine v2606 fissata per digest in `../dataset/pipeline_versions.env`).
- `zerihan3d/geometry3d.py`: STL di main, flap ed endplate (regioni fore e aft), controllo di tenuta.
- `zerihan3d/case3d.py`: scrittura del caso (blockMesh, snappyHexMesh, campi, function object).
- `zerihan3d/Allrun`: mesh e soluzione dentro il container, con tempi (`timings.txt`) e RAM (`mem.log`).
- `zerihan3d/report3d.py`: riepilogo, grafici e proiezione dei costi.
- `zerihan3d/fit_endplate_from_fig1c.py`, `endplate_fig1c.json`, `endplate_fig1c_overlay.png`: provenienza della posizione dell'endplate.
