# Imperial Front Wing (IFW): infrastruttura e prima prova

Preparato il 04/11/2026. È un caso di **validazione** su una geometria F1 vera: tre elementi, canard, gurney, endplate, footplate, naso e supporto. Usa la stessa fisica, la stessa numerica e lo stesso arresto automatico del kit Zerihan (`../zerihan3d/`).

## Fonte e licenza

- **CAD:** Buscariolo F. F., "Imperial Front Wing (IFW) geometry", Imperial College London, DOI 10.14469/hpc/6049, licenza **CC-BY 4.0** (si può usare citando la fonte). Ricevuto da Imperial RDM il 04/11/2026 perché il repository HPC è in migrazione verso Helix.
- **File in `imperial/`:**
  - `ifw.stp`: CATIA, in mm, file originale `IFW_HC036_NOWHEEL_V4`, SHA-256 `6053849d…0ebc`;
  - `ifw_mesh.xml`: mesh di ordine elevato di Nektar++, non usata;
  - il PDF dei metadati.
- **Riferimento numerico:** Buscariolo et al., J. Wind Eng. Ind. Aerodyn. 2022 (arXiv 1909.06701), Nektar++ iLES/uDNS NM54 a Re 2,2·10⁵: **CL = −5,54, CD = 0,575**, area c·W_s = 0,25 × 0,70 m². Il paper non dice quali patch entrano nel coefficiente. Le forze sperimentali non sono pubbliche.

## Cosa c'è nello STEP (ispezione del 04/11)

209 facce tagliate e nessun solido:

- **Non usati:** il box del dominio CFD di Imperial (x da −7,5 a 12,0 m, y da −3,05 a 0, suolo a z = 4,066 mm) e due patch ausiliarie piane.
- **Il modello**, metà vettura a y ≤ 0:
  - l'**ala**: 170 facce in un unico guscio, con elementi, canard, gurney, endplate e footplate;
  - il **naso**: 9 facce;
  - il **supporto**: 22 facce.
- **Aperture:** i gusci sono aperti solo sul piano di simmetria e ai raccordi del supporto (contatto geometrico, non topologico, con fessure sotto il millimetro).
- **Posizione:** il modello è già in posizione di prova. Bordo d'uscita del footplate a 88,97 mm dal suolo, **h/c = 0,356** (paper: 0,36). **Beccheggio 1,091°** misurato sul bordo inferiore (paper: 1,094°). Semi-apertura 0,70 m.

## Catena

| Passo | Script | Uscita |
|---|---|---|
| STEP → STL | `prepare_ifw.py`: gmsh/OpenCASCADE, triangoli ≤ 2,5 mm con raffinamento sulla curvatura, mm → m, suolo a z = 0 | `imperial/stl/{wing,nose,hanger}.stl` (binari; ala 1,17 M triangoli) e `ifw_geometry.json` (misure) |
| Caso | `case_ifw.py`: blockMesh, snappyHexMesh, k-ω SST con funzioni di parete, U = 25 m/s, Re 2,2·10⁵, suolo mobile, simmetria y = 0 | caso OpenFOAM |
| Lancio | `../run_ifw.sh smoke\|L0\|L1`: container v2606, arresto automatico, report | `data/frontwing_runs3d/ifw_*` |

- **Trasferimento dell'ala:** `wing.stl` (58 MB) viaggia in tre parti (`wing.stl.part00…02`) per il limite di trasferimento. `run_ifw.sh` la ricompone e controlla `SHA256SUMS`. Se gli STL mancano, li rigenera dallo STEP (circa 5 minuti).
- **Livelli** (cella di fondo d0; superficie dell'ala ai livelli 5–6, spigoli al livello 7):
  - smoke: d0 0,14 m (spigoli 1,1 mm);
  - L0: 0,10 m (0,8 mm);
  - L1: 0,07 m (0,55 mm).
- **Box di raffinamento:** attorno al modello (livello 3), tra footplate e suolo (4), scia vicina (2) e scia lontana (1).
- **Coefficienti:** `coeffs_total` (ala + naso + supporto), `coeffs_wing`, `coeffs_nose`, `coeffs_hanger`, con CL positivo = deportanza.

## Collaudo nel cloud (04/11, mesh smoke, OpenFOAM v1912, 2 core)

- **Celle:** 1 248 588. Strati coperti al 79% sull'ala, all'89% sul naso e al 77% sul supporto.
- **Qualità:** non ortogonalità massima 64,9°; 2 facce con skewness 4,45, sopra la soglia di checkMesh.
- **Tempo di mesh:** 534 s su 2 core. Il solutore è stabile (errori di continuità circa 1e-4).
- **Iterazione 100** (non ancora a convergenza, solo pressione): CL 4,04 (ala 4,07, naso −0,02), CD 0,50. Riferimento: 5,54 e 0,575. L'ordine di grandezza è giusto e il valore sta ancora crescendo.

## Run L0 sul portatile (04/11, OpenFOAM v2606, i7-13705H, 6 rank)

Caso `data/frontwing_runs3d/ifw_L0_np6_20261104T180753Z`.

| Voce | Valore |
|---|---|
| Mesh | 2 170 474 celle in 5 min 16 s; strati coperti all'82% sull'ala, al 92% sul naso e all'81% sul supporto. Non ortogonalità massima 65,0°, 1 faccia con skewness 4,18 (soglia 4) |
| Solutore | 1518 iterazioni in 75 min (2,95 s/iterazione, 8,2 µs per cella e iterazione per rank), picco di RAM 4,9 GB; caso completo 1 h 21 min |
| Arresto | a mano con `stopAt writeNow`, secondo la regola `conv3d-v1`. Classe "oscillating" (media ferma, oscillazione limitata), che con la regola si sarebbe fermata a 1300 iterazioni. Il controllore del run aveva la regola v0, che sull'IFW non sarebbe mai scattata (vedi sotto) |
| **CL totale (deportanza +)** | **5,389 ± 0,07%**: ala 5,424, naso −0,029, supporto −0,005 |
| **CD totale** | **0,594 ± 0,34%**: ala 0,593 |
| Confronto con Nektar++ NM54 (5,54 / 0,575) | **CL −2,7%** (solo ala −2,1%), **CD +3,3%** |
| y⁺ medio | ala 14 (massimo 159), naso 38, supporto 15 |
| Residui | piatti da circa 400 iterazioni: p circa 9e-4 (3,0 ordini), U circa 1e-4, k circa 2e-5 |

**Lettura.** Una mesh da 2,2 M celle con leggi di parete riproduce la deportanza e la resistenza della simulazione di riferimento (iLES/uDNS ad alto ordine) entro il 3%. Non è una validazione sperimentale e non dice niente sulla distribuzione dei carichi. Il prossimo controllo è la posizione dei vortici in scia contro le figure PIV del paper (piani a X = −294, −150, −24 e +150 mm).

**Arresto automatico, versione `conv3d-v1`.** Sull'IFW CD oscilla di ±0,3–0,4%, per i bordi d'uscita tronchi, i gurney e i vortici. Le modifiche alla regola:
- le soglie su CD diventano relative;
- è stata aggiunta la classe "oscillating" (come la B del 2D): deriva di CL ≤ 5e-4 e di CD ≤ 1e-3, ampiezza di CL ≤ 0,25% e di CD ≤ 1%, residui ≥ 2,5 ordini, tenuta per 3 controlli di fila;
- lo Zerihan si ferma ancora a 1300 iterazioni come "converged".

## Fase 2b: parametri di forma (05/11)

`shape_modes.py`: **16 modi di forma**, ciascuno su un elemento (o sui due flap insieme), nel sistema locale della corda di ogni sezione in apertura:
- s va da 0 al bordo d'attacco a 1 al bordo d'uscita;
- n è la distanza dalla linea della corda, positiva verso l'alto.

Ogni modo si annulla:
- verso il raccordo con l'endplate (100 mm);
- sul main, anche attorno alla radice del supporto (60 mm). Senza questa protezione, un ispessimento del 20% del main apriva un gradino di 3 mm sotto il supporto, che resta fermo.

| Famiglia | Modi | Range provato | Cosa cambia |
|---|---|---|---|
| Incidenza | `rot_flap2`, `rot_flap1` (flap 1 + flap 2) | ±6° | angolo dei flap |
| Svergolamento | `twist_flap2`, `twist_flap1` | ±4° | incidenza alla radice rispetto all'estremità |
| Curvatura | `camber_main`, `camber_flap1`, `camber_flap2` | ±2% c | la linea media si gonfia verso il suolo, spessore invariato |
| Posizione della curvatura | `camber2_main`, `camber2_flap1`, `camber2_flap2` | ±1,5% c | curvatura spostata avanti o indietro (modo sin 2πs) |
| Spessore | `thick_main`, `thick_flap1`, `thick_flap2` | ×0,85…1,15 | spessore attorno alla linea media; bordo d'attacco e bordo d'uscita invariati |
| Corda | `chord_flap2` | ×0,9…1,1 | il bordo d'uscita si allunga lungo la corda |
| Fessura | `gap_flap2`, `overlap_flap2` | da −1,5 a +2 mm / ±4 mm | posizione del flap 2 rispetto al flap 1 |

Più i 2 parametri di condizione già presenti (altezza da terra e beccheggio), per **18 parametri** in totale.

**Controlli automatici** su ogni variante (`deform.json`):
- fessure main–flap1 e flap1–flap2 ≥ 1 mm;
- nessun triangolo ribaltato e nessuna rotazione delle normali oltre 60° (si ignorano i triangoli con un lato sotto 0,05 mm);
- radice del supporto ferma (≤ 0,05 mm);
- distanza dal suolo.

**Prove:**
- **Ogni modo agli estremi del range:** tutti validi. Rotazione massima delle normali 2–17° per i modi di forma (fino a 30° nelle combinazioni con rotazione e svergolamento insieme), nessun triangolo ribaltato.
- **32 combinazioni casuali (Sobol) con tutti i 16 modi insieme:** **32 valide su 32**. Fessure minime 2,1 mm (flap1–flap2) e 6,6 mm (main–flap1); spostamento massimo 39 mm (`check_shape_space.py`).
- **Mesh della combinazione casuale più deformata** (n. 5, spostamento 39 mm, mesh smoke nel cloud): 1 286 106 celle (originale 1 248 588), strati sull'ala al 79,4% (originale 79,3%), nessun errore di mesh, solutore stabile.
- **Catalogo dei modi:** `imperial/variants/ifw_shape_modes.png`, sezione a y = −300 mm.

**Lancio di una variante con modi di forma:**
`SHAPE='{"camber_main": 1.5, "thick_flap1": 1.1}' NP=6 bash run_ifw.sh L0`

**Cosa resta fuori:**
- **Endplate, footplate e canard** (118 facce, il punto a cui sono attaccati gli elementi): sono parametrizzabili con lo stesso metodo, ma servono raccordi propri.
- **Altezza dei gurney.**
- **Varianti di grande ampiezza:** sono deformazioni dell'ala originale e non nuovi progetti, con lo svergolamento forzato negli ultimi 100 mm verso l'endplate.

**Da decidere prima di un dataset.** 18 dimensioni con mille campioni sono uno spazio rado. Conviene scegliere 6–10 parametri in base alla sensibilità: una prova "un parametro alla volta" a ± l'ampiezza, cioè 2 casi L0 per modo, circa 32 casi, cioè 1–2 giorni di una macchina.

## Fase 3: test di sensibilità OAT sul cluster (preparato il 05/11)

L'obiettivo è misurare la classifica d'importanza dei parametri invece di stimarla.

- **Piano:** `hpc/oat_plan.csv`, 37 casi L0, cioè la baseline più ognuno dei 18 parametri (16 modi di forma, h/c e beccheggio) da solo, all'estremo basso e a quello alto del suo intervallo. Tutte le 37 geometrie sono valide (`oat_plan.py --check`).
- **Esecuzione:** cluster del Politecnico, Apptainer + SLURM, con gli script in `hpc/` (vedi `hpc/README.md` e `hpc/AGENT_INSTRUCTIONS.md`). Il kit arriva sul cluster con il branch `hpc-ifw`.
- **Costo atteso:** circa 37 × 7 core-h, cioè circa 260 core-h. Con 8 casi alla volta servono circa 6-8 h.
- **Uscita:** `oat_ranking.md/.png`, cioè l'effetto (estremo alto meno estremo basso) su CL, CD e L/D in % della baseline, con la soglia di rumore data dall'oscillazione residua.

### Risultati (cluster Polimi, coda PBS `cpu`, 11-13/11/2026)

**Esecuzione:**
- 37/37 casi `done`, ciascuno su 6 rank in Apptainer.
- Il container è l'immagine OpenFOAM v2606 nuda (rotta B), con Python sull'host.
- Due casi (`thick_flap1_lo`, `gap_flap2_lo`) erano falliti per una copia annidata in `restore0Dir` sul filesystem `/work`. Corretto in `Allrun` e rilanciati.

**Riproducibilità:** la baseline sul cluster è identica a quella del portatile: 2,170,474 celle, CL 5.3896, CD 0.5939. L'autostop si ferma come "oscillating" a 1300 iterazioni.

**Costo:**
- 92 min per caso su 6 core, cioè circa 9 core-h; 3.8 s/iterazione a 2.17 M celle.
- Picco di memoria 5.1 GB.
- Circa 330 core-h per l'intera campagna.

**Classifica per variazione massima di CL a uno dei due estremi** (in % della baseline; tra parentesi l'estremo basso / alto):

| # | parametro | intervallo | max \|dCL\| | basso / alto | note |
|---|---|---|---|---|---|
| 1 | h/c (`dz_mm`) | −40 .. +60 mm | 28.3 | −28.3 / −21.3 | massimo vicino alla baseline (h/c 0.36); h/c 0.2 instazionario |
| 2 | `camber2_main` | −1.5 .. 1.5 | 18.8 | −1.9 / −18.8 | estremo alto instazionario, probabile separazione |
| 3 | `rot_flap1` | ±6° | 16.8 | −16.8 / −6.3 | ottimo vicino a 0°; +6° instazionario |
| 4 | `rot_flap2` | ±6° | 12.7 | −12.7 / +9.1 | quasi lineare; i due estremi arrivano a 3000 iterazioni |
| 5 | beccheggio | ±1° | 7.7 | −7.7 / +6.5 | lineare |
| 6 | `twist_flap1` | ±4° | 4.9 | −4.9 / +4.0 | lineare |
| 7 | `twist_flap2` | ±4° | 3.6 | −3.6 / +3.0 | lineare |
| 8 | `camber_main` | ±2 | 2.7 | −2.7 / −0.2 | |
| 9 | `camber_flap2` | ±2 | 1.9 | −1.9 / +1.2 | |
| 10 | `chord_flap2` | 0.9 .. 1.1 | 1.6 | −1.6 / +1.6 | |
| 11 | `camber2_flap2` | ±1.5 | 1.4 | +1.3 / −1.4 | |
| 12–18 | `camber2_flap1`, `gap_flap2`, `camber_flap1`, `overlap_flap2`, `thick_main`, `thick_flap2`, `thick_flap1` | | 0.1–0.9 | | sotto o vicino al rumore |

**Lettura:**
- **Classifica attesa:** confermata nelle linee generali: h/c, angoli dei flap e beccheggio davanti, spessori trascurabili.
- **Sorprese:** lo svergolamento dei flap conta più di gap e overlap. La posizione della curvatura del main conta molto, ma solo perché all'estremo alto il flusso separa.
- **"Alto meno basso" non basta:** per h/c e `rot_flap1` la baseline sta vicino a un ottimo, quindi entrambi gli estremi perdono carico e la differenza alto meno basso sottostima il parametro. Per questo la classifica usa la variazione massima.
- **Rumore:** il rumore dovuto al rifare la mesh è circa 0.2–0.4 % su CL (`thick_main` dà −0.25 / −0.22). È più dell'oscillazione residua (0.15 %): effetti sotto lo 0.5 % non sono distinguibili.
- **Casi lenti o instazionari:** i casi arrivati a 3000 iterazioni o instazionari si concentrano agli estremi degli intervalli. Per il dataset gli intervalli vanno ristretti, altrimenti costano il doppio e danno etichette rumorose.

**Proposta per il dataset (8 parametri, intervalli da fissare dalle storie di convergenza):** h/c, beccheggio, `rot_flap1`, `rot_flap2`, `twist_flap1`, `twist_flap2`, `camber_main`, `camber2_main` con l'estremo alto ridotto. Si stimano circa 10–11 core-h per caso, cioè circa 11.000 core-h per 1000 casi.

## Limiti dichiarati

- Leggi di parete con y⁺ circa 30–150 contro lo strato limite risolto dell'iLES. Flusso a Re 2,2·10⁵ trattato come completamente turbolento.
- RANS stazionario su un flusso dominato da vortici.
- Il riferimento è un'altra simulazione, non un esperimento.

## Fase 2: parametrizzazione (indagine del 04/11)

**1. Separazione degli elementi** (`classify_elements.py`, regole deterministiche, nessun tag scelto a mano). Le facce dell'ala vengono unite in pezze lisce se l'angolo tra facce adiacenti è sotto i 40°. Le tre pezze che partono dal piano di simmetria e coprono più di 300 mm di apertura sono il main e i due flap. La pezza esterna più grande è il gruppo endplate. Le pezze piccole (basi dei bordi d'uscita, gurney, piastre) vanno all'elemento con cui condividono più bordo.

| Elemento | Facce CAD | Raccordo con l'endplate (y) |
|---|---|---|
| main | 12 | da −631 a −600 mm |
| flap 1 | 20 | da −601 a −585 mm |
| flap 2 (con gurney) | 20 | da −586 a −569 mm |
| endplate, footplate, canard, raccordi | 118 | – |

Nessun bordo è condiviso tra due elementi: sono attaccati **solo** all'endplate. Il risultato coincide con una classificazione fatta a mano.

**2. Deformazione** (`deform_ifw.py`, sugli STL, senza ricostruire il CAD). Ogni sezione in apertura ruota attorno al bordo d'attacco locale; la rotazione si annulla dolcemente verso il raccordo con l'endplate (smoothstep su 100 mm), così l'ala resta chiusa e i nodi dell'endplate non si muovono.

| Parametro | Significato | Prove |
|---|---|---|
| `D_FLAP2` | flap 2 da solo, attorno al suo bordo d'attacco | ±6°: fessura flap1–flap2 da 3,7 a 4,7 mm (nominale 4,5) |
| `D_FLAP1` | flap 1 e flap 2 insieme, attorno al bordo d'attacco del flap 1 | ±6°: fessura main–flap1 da 7,1 a 8,4 mm (nominale 8,2). Il flap 1 **da solo** a +6° chiude la fessura con il flap 2 (0,04 mm), per questo i flap ruotano insieme |
| `DZ_MM` | altezza da terra (corpo rigido, ala + naso + supporto) | +20 mm → h/c 0,436 |
| `DPITCH` | beccheggio attorno al bordo d'uscita del footplate | +0,5° → 1,59° |
| `d_main` (solo diagnostica) | main da solo | +3° chiude la fessura main–flap1: non è un parametro utilizzabile |

- **Spigoli dello spazio ±10°/±10°:** tutti validi per le fessure (main–flap1 ≥ 6,3 mm, flap1–flap2 ≥ 3,1 mm), con spostamenti fino a 67 mm. Propongo ±6°.
- **Prova di mesh:** la variante più spinta del collaudo (flap1 +6°, flap2 +6°, mesh smoke nel cloud) dà 1 231 740 celle contro 1 248 588 dell'originale, strati all'78,8% contro il 79,3%, nessun errore di mesh e solutore stabile. La catena va avanti senza modifiche.
- **Lancio di una variante:** `D_FLAP2=4 NP=6 bash run_ifw.sh L0`. La geometria va in `imperial/variants/` con `deform.json` (parametri e controlli); una variante non valida non viene lanciata.

**3. Fattibilità del dataset.**
- **Spazio proposto (v0 3D):** 4 parametri, cioè `D_FLAP1` e `D_FLAP2` in ±6° (forma) più h/c (circa 0,2–0,6, cioè `DZ_MM` da −40 a +60) e beccheggio (±1°) (condizione). Stessa struttura del dataset 2D: geometrie e condizioni.
- **Costo:** circa 1,1–1,4 h per caso sul portatile (L0 con arresto automatico), cioè circa 18–20 casi al giorno. 100 casi richiedono circa 5–6 giorni di una macchina.
- **Limiti:**
  - le varianti sono deformazioni del CAD originale, non progetti di ali;
  - l'ultimo tratto di 100 mm di ogni flap viene svergolato per restare attaccato all'endplate;
  - per ora si cambia solo l'angolo dei flap, non curvatura o spessore (servirebbe una deformazione libera per elemento, la strada 2 qui sotto);
  - il riferimento Nektar++ vale solo per il CAD originale.
- **Da fare prima di produrre:**
  1. due varianti L0 sul portatile (`D_FLAP2` = ±4°) per verificare che CL e CD si muovano in modo regolare e con il segno giusto;
  2. una misura del rumore di mesh: la stessa geometria con variazioni piccolissime (±0,5°) deve dare coefficienti che variano in modo liscio.

### Strade per ampliare lo spazio delle forme, dalla più semplice
1. Rotazioni rigide degli elementi sugli STL: fatto (`deform_ifw.py`).
2. Deformazione libera (FFD/RBF) per elemento: curvatura, spessore, corda. È il parallelo dei "modi lisci" del 2D.
3. Parametri nel CAD: richiede ricostruire le superfici B-spline; è la strada più costosa.
