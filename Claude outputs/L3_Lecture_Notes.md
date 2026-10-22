# Lesson 3 — Il machine learning del progetto

**F1 Front-Wing PINN · Polimi Data Scientists**
Note complete per chi tiene la lezione (un componente del gruppo) e da dare agli studenti dopo. Versione estesa del 22 ottobre 2026: ogni blocco ha cosa dire, cosa mostrare, i numeri verificati, le domande probabili e gli errori da evitare.

| | |
|---|---|
| **Data** | martedì 27 ottobre 2026 · 60 min (55 pianificati + 5 di margine) · seguita dal kickoff (30 min, separato) |
| **Chi la tiene** | un componente del gruppo (non Verri). Deck: da preparare (stesso template delle Lesson 1 e 2, se possibile) |
| **Prerequisiti degli studenti** | Lesson 1 e 2. Nessun compito da portare: la Lesson 2 ha lasciato quattro domande aperte (vedi §2.3) |
| **Script live** | `course/L3/demo_l3_slope.py` (vincolo sulla derivata), `course/L1/audit_replication.py --full` (leakage del paper) |
| **Materiale di supporto** | `shape_input/` (analisi delle codifiche della forma su AirfRANS, figure 01–05), `Project_Objectives.md` (T4, T5), `D2_Dataset_Proposal.md` (il dataset) |
| **Lingua** | slide in inglese, note e spiegazioni in italiano |

---

## Indice

0. Come usare queste note
1. La lezione in una pagina
2. Chi sono gli studenti e cosa sanno già
3. Il progetto e il dataset, quello che serve sapere
4. Preparazione (checklist)
5. Scaletta
6. Blocco 0 — Un run non prova niente
7. Blocco 1 — La forma come input
8. Blocco 2 — Input e normalizzazione
9. Blocco 3 — Il vincolo fisico su una derivata (live)
10. Blocco 4 — Valutare un surrogato senza ingannarsi (live)
11. Blocco 5 — Addestrare con pochi dati
12. Blocco 6 — Il vincolo fisico vicino al suolo, e il protocollo di T5
13. Chiusura e compiti
14. Domande probabili
15. Glossario
16. File e comandi
17. Letture per ruolo

---

## 0. Come usare queste note

- **Per preparare la lezione:** leggi §1–5 (un'ora), poi esegui una volta tutti i comandi di §16 sul portatile che userai in aula. I numeri attesi sono nelle tabelle di ogni blocco: se escono uguali, l'ambiente è a posto.
- **Durante la lezione:** ogni blocco ha una sezione **"Cosa dire"**. Non va letta: è il ragionamento da trasmettere, con le frasi chiave in grassetto. La sezione **"Cosa mostrare"** dice cosa mettere sulla slide.
- **Regola della lezione:** ogni numero mostrato viene da uno script che gli studenti possono rieseguire. Se un numero non viene da uno script, si dice "stima" o non si mostra.
- **Cosa non fare:** non promettere risultati sul nostro dataset (non esiste ancora: la v0 arriva a metà novembre). Oggi si insegnano **gli strumenti e il metodo**; i risultati li produrranno gli studenti in T4 e T5.

---

## 1. La lezione in una pagina

**Obiettivo.** Alla fine gli studenti hanno tutto quello che serve per iniziare **T4** (il surrogato: dai parametri dell'ala a $C_l$ e $C_d$) e **T5** (verificare se un vincolo fisico aiuta quando i dati sono pochi).

**Sei messaggi, uno per blocco.**

| Blocco | Messaggio da portare a casa |
|---|---|
| 0 | **Un run non prova niente**: media ± deviazione standard su più seed e più dimensioni del dataset. |
| 1 | **La forma va data alla rete come numeri**, e ci sono più modi (parametri, coordinate, PCA, SDF): si sceglie misurando. Le grandezze geometriche si calcolano dalla forma, mai dai nomi. |
| 2 | **Ingressi adimensionali e di ordine 1, normalizzati con statistiche del solo training.** Angolo e altezza da terra restano ingressi espliciti. |
| 3 | **Una legge fisica si può imporre sulla derivata della rete** (autograd, come in una PINN); attenzione alle unità: dimenticare un fattore 18/π rende il vincolo dannoso. |
| 4 | **Si valuta su ali mai viste**, mai su righe mescolate a caso; prima le baseline; più seed; curve di apprendimento; R² non basta. |
| 5 | **Con pochi dati la rete impara a memoria**: regolarizzazione, early stopping, iperparametri scelti sulla validazione, test usato una volta sola. |
| 6 | **Vicino al suolo la teoria del profilo sottile non vale così com'è**: restano vincoli più deboli ($C_d \ge 0$, limite lontano dal suolo, linearità a piccoli angoli). Se aiutano si misura, con un protocollo; "non aiuta" è un risultato. |

**Due demo live**: `demo_l3_slope.py` (blocco 3, circa 30 s a comando) e `audit_replication.py --full` (blocco 4, pochi secondi).

---

## 2. Chi sono gli studenti e cosa sanno già

### 2.1 Il profilo

Quattro studenti della triennale (informatica, ingegneria, matematica). Sanno programmare in Python, calcolo in una variabile, ODE semplici, algebra lineare e statistica di base. **Non** hanno fatto PDE né fluidodinamica. Il deep learning lo hanno visto solo nelle Lesson 1 e 2: sanno cos'è una rete, una loss, la discesa del gradiente, la validazione; **non** hanno visto strati, attivazioni, regolarizzazione, ottimizzatori nel dettaglio (nella Lesson 2 la rete era volutamente una scatola chiusa).

### 2.2 Cosa hanno visto

**Lesson 1 (20 ottobre) — Le reti e l'audit del paper.**

- Una rete è "una retta che può piegarsi": parametri, loss (errore quadratico medio), discesa del gradiente, training contro validazione.
- Demo `demo_l1_network.py`: su una curva di portanza inventata, retta 0.0671 contro rete 0.0002 di errore di validazione (40 punti). La rete riceve $x = \alpha/10$ (gradi), non $\alpha$.
- **Audit del paper** di riferimento (Shah, arXiv:2509.01963): i dati sono un'unica simulazione di un'unica geometria; le uscite sono gli ingressi divisi per una costante (spread relativo $1.1\times10^{-13}$); una regressione lineare con 26 parametri arriva a $R^2 = 1$ mentre la rete del paper fa 0.968; la "physics loss" è sbagliata di 11.5× e manca l'area; non c'è nessuna derivata rispetto agli ingressi, quindi non è una PINN; lo split è casuale su righe quasi identiche (difetto D6, lo riprendi tu nel blocco 4).
- **La nostra physics loss**: per un profilo sottile, ad angoli piccoli, $C_l \approx 2\pi(\alpha - \alpha_0)$. Con 6 punti di training la fisica aiuta in 4 seed su 5 (seed 1: 0.0098 senza, 0.0103 con).

**Lesson 2 (22 ottobre) — Il contesto della rete: equazioni, dataset, SDF.**

- **Il residuo come loss**: per $dy/dt = -ky$ il residuo $R = dy/dt + ky$ è zero solo per la soluzione giusta.
- **Due derivate diverse**: quella della loss rispetto ai pesi (`loss.backward()`, per addestrare) e quella dell'uscita della rete rispetto all'ingresso (`torch.autograd.grad`, per scrivere il residuo).
- **Demo `demo_l2_pinn_ode.py`**: PINN senza dati, errore 0.033% contro $e^{-t}$; senza condizione iniziale l'equazione è rispettata ma l'errore è circa 110% (**una loss piccola non è una risposta giusta**).
- **D5 chiuso**: una PINN ha un'equazione nella loss e derivate rispetto all'input; il paper non ha né l'una né l'altra. **La nostra rete non sarà una PINN ma un surrogato con un vincolo fisico.**
- **L'ala anteriore**: rovesciata (deportanza), due elementi (main plane + flap), vicino al suolo (effetto suolo). Simulazione OpenFOAM dal vivo della sezione 2D di Zerihan: $C_l$ = 3.44 con il suolo a 15 cm ($h/c$ = 0.395), 4.31 a 8 cm ($h/c$ = 0.211). Numeri preliminari.
- **Il dataset**: una simulazione = una riga (ala + altezza + angolo → $C_l$, $C_d$); circa 80 ali e 1,200 simulazioni; ogni caso convergente o segnato.
- **La SDF**: in ogni cella di una griglia, la distanza dalla parete più vicina, negativa dentro l'ala. Costo: su una griglia grossa il bordo d'uscita sottile sparisce (4.4% della corda persa a 96 × 48, 1.1% a 384 × 192).

**Cosa NON hanno visto** (tagliato dalla Lesson 2): l'equazione del calore e l'esempio ReLU (loss 14× più bassa, errore 175× peggiore). Lo script `course/L2/demo_l2_pinn_1d.py --activation relu` resta disponibile se vuoi usarlo per parlare di attivazioni (blocco 5).

### 2.3 Le quattro domande lasciate aperte

La Lesson 2 si è chiusa senza compiti, con quattro domande "Think about the problem", da discutere insieme nella Lesson 4:

1. Arriva un'ala nuova: cosa dareste in input alla rete, e perché?
2. Quali simulazioni terreste fuori dall'addestramento per provare la rete onestamente?
3. Molto vicino al suolo una simulazione non converge: cosa dovrebbe fare la rete lì?
4. Cosa può dire una legge fisica alla rete che i dati non dicono?

**Non rispondere al posto loro**: questa lezione dà gli strumenti per rispondere (domanda 1 → blocchi 1–2; domanda 2 → blocco 4; domanda 3 → blocchi 4 e 6; domanda 4 → blocchi 3 e 6). Puoi citarle all'inizio ("oggi vi do gli attrezzi per le quattro domande dell'altra volta") e alla fine.

---

## 3. Il progetto e il dataset, quello che serve sapere

### 3.1 Gli obiettivi che questa lezione prepara

Da `Project_Objectives.md`, fase 2 (dopo il freeze degli esami, da fine febbraio, sul dataset v1):

- **T4 — Surrogato onesto.** Prevedere $C_l$ e $C_d$ da geometria e configurazione, ingressi adimensionali. Split ufficiale **per geometria** (ali mai viste nel test) più uno split di estrapolazione (altezze fuori dal range di training). Scala di baseline completa: lineare → random forest → gradient boosting → MLP. Media ± deviazione standard su almeno 3 seed. Curve di apprendimento. Errore relativo accanto a $R^2$. Risultati con e senza i casi non convergenti, separati.
- **T5 — L'ipotesi del paper, fatta bene.** Un vincolo fisico *legittimo* migliora il surrogato quando i dati sono pochi? Stessa MLP di T4 più un termine fisico; $\lambda$ su almeno 4 ordini di grandezza; dimensioni del training da poche ali a tutte; almeno 3 seed. Si riporta la risposta qualunque sia. **È il cuore scientifico del progetto.**

### 3.2 Il dataset (traccia D, lo costruisce Verri)

| | |
|---|---|
| Cosa | sezioni 2D di ala a due elementi (main + flap), rovesciata, sopra un suolo che si muove alla velocità dell'aria |
| Solutore | OpenFOAM, RANS stazionaria, k-ω SST, $Re = 7.5\times10^5$ fisso (30 m/s, corda 0.38 m) |
| Dimensione prevista | circa 80 geometrie × 14 condizioni (12 ben distribuite in $h/c$ e $\alpha$ + 2 lontane dal suolo), più 80 casi di scansione in $\alpha$ su 20 geometrie: circa 1,200 casi (proposta D2) |
| Ingressi | 4 parametri di forma (spessore e curvatura del main, angolo e corda del flap) + 2 di condizione (altezza da terra $h/c$, incidenza $\alpha$). La geometria vera è disponibile anche come contorno (per SDF o coordinate) |
| Uscite | $C_l$ (positivo = deportanza), $C_d$, totali e per elemento |
| Intervalli indicativi | $h/c$ circa 0.08–1.0 (scala logaritmica) più 2–3 punti a 2–4; $\alpha$ circa −1° … +4°; angolo del flap circa −10° … +12°; spessore, curvatura e corda del flap ±10–15% attorno a Zerihan |
| Flag | ogni riga ha un flag: `converged`, `coeffs_stable`, `oscillating`, `failed` (più `invalid_geometry` per le forme scartate prima della mesh) |
| Split ufficiale | **per geometria**, circa 56/12/12 ali (train/validazione/test), fissato dal PM con un seme dichiarato; più due split di estrapolazione tenuti fuori dal training: E1 lontano dal suolo ($h/c \ge 2$), E2 flap agli estremi |
| Campionamento | geometrie con sequenza di Sobol: i prefissi (10, 20, 40, 80 ali) sono ben distribuiti, quindi le curve di apprendimento si fanno prendendo le prime N ali, senza rigenerare dati |
| Calendario | v0 pilota a metà novembre, v1 a fine dicembre |

**Simboli.** $h$ = distanza tra il suolo e il punto più basso dell'ala; $c$ = corda totale (380 mm); $h/c$ = altezza adimensionale (0.211 vuol dire 8 cm). $C_l = L/(\tfrac12\rho U^2 c)$ e $C_d = D/(\tfrac12\rho U^2 c)$: forza divisa per pressione dinamica e dimensione, quindi numeri puri che si confrontano tra velocità e dimensioni diverse.

### 3.3 Vincoli tecnici

Tutto gira su CPU del portatile. PyTorch puro con `torch.autograd.grad`; scikit-learn per le baseline. Niente GPU, niente librerie esotiche.

---

## 4. Preparazione (checklist)

**Una settimana prima**

- [ ] Leggere queste note e `Project_Objectives.md` §3, §6.
- [ ] Clonare la repo, attivare il venv del corso, `pip install -r course/requirements.txt`, poi `python course/L1/check_setup.py` → `ALL CHECKS PASSED`.
- [ ] Eseguire tutti i comandi di §16 e confrontare con i numeri attesi.
- [ ] Preparare il deck (in inglese). Suggerimento: una slide per messaggio, poco testo, numeri grandi.

**Il giorno prima**

- [ ] Copiare `demo_l3_slope.py` in `course/L3/` se non c'è già, fare commit e push, così gli studenti lo hanno con `git pull`.
- [ ] Provare i comandi live sul portatile dell'aula (tempi: circa 30 s a comando).

**In aula**

- [ ] Studenti: `git pull`, venv attivo.
- [ ] Piano B se una demo è lenta: le tabelle con i risultati attesi sono in queste note (e in `course/L3/expected_output/`).

---

## 5. Scaletta

| Min | Blocco | Modalità |
|---|---|---|
| 0–5 | 0 · Un run non prova niente | slide |
| 5–13 | 1 · La forma come input | slide (figure di `shape_input/`) |
| 13–17 | 2 · Input e normalizzazione | slide |
| 17–25 | 3 · Il vincolo fisico su una derivata | **live** `demo_l3_slope.py` |
| 25–35 | 4 · Valutare un surrogato senza ingannarsi | slide + **live** `audit_replication.py --full` |
| 35–43 | 5 · Addestrare con pochi dati | slide |
| 43–52 | 6 · Il vincolo fisico vicino al suolo, protocollo di T5 | slide |
| 52–55 | Chiusura e compiti | slide |

Se sei in ritardo: accorcia il blocco 1 (tieni solo la tabella delle codifiche) e il blocco 5 (tieni overfitting + validazione/test). Non tagliare i blocchi 3 e 4: sono le due cose che servono subito.

---

## 6. Blocco 0 — Un run non prova niente (0–5 min)

**Obiettivo.** Fissare la regola che vale per tutta la lezione e tutto il progetto.

**Cosa dire.**

Nella Lesson 1 gli studenti hanno riempito una tabella: con 6 punti di training, la fisica aiuta? In 4 seed su 5 sì, nel seed 1 no (0.0098 senza, 0.0103 con). **Quattro su cinque, su una sola dimensione del dataset, non è ancora un risultato.**

Un "seed" fissa tutte le scelte casuali di un addestramento. In `demo_l1_network.py` sono tre:

1. **quali punti** finiscono nel training (estratti a caso tra −5° e 20°);
2. **il rumore** aggiunto ai dati;
3. **i pesi iniziali** della rete.

Cambiando seed cambiano tutti e tre, e il risultato cambia. Quindi un numero solo può essere fortuna. La regola:

- **almeno 3 seed (meglio 5)**, si riporta **media ± deviazione standard**;
- **più dimensioni del training** (pochi dati, tanti dati): un metodo può aiutare con pochi dati e non servire con tanti;
- un confronto vale se la differenza è grande rispetto alla deviazione standard. Se due metodi fanno 0.0072 ± 0.0026 e 0.0065 ± 0.0030, non si può dire che uno sia migliore.

**Cosa mostrare.** La tabella dei seed della Lesson 1 e la frase "mean ± std over seeds and dataset sizes".

**Da dire esplicitamente.** Tutti gli script del corso girano già su più seed e stampano media ± deviazione standard (`demo_l3_slope.py` usa i seed 0–4).

---

## 7. Blocco 1 — La forma come input (5–13 min)

**Obiettivo.** Far capire che la forma dell'ala va trasformata in numeri, che ci sono più modi di farlo, e che la scelta si misura.

**Cosa dire.**

Una rete prende in ingresso un vettore di numeri di lunghezza fissa. Una forma non è un vettore: va **codificata**. Nella Lesson 2 hanno visto la SDF (una griglia di distanze). Oggi la mettiamo accanto alle alternative.

| Codifica | Quanti numeri | Rete che la legge | Pro | Contro |
|---|---|---|---|---|
| **Parametri** del generatore (le 4 forme del nostro dataset) | 4 | MLP | compattissima, interpretabile | vale solo per questa famiglia di ali; i parametri possono "mentire" (vedi sotto) |
| **Descrittori misurati** sulla forma (spessore massimo e posizione, curvatura massima e posizione, pendenza al bordo d'uscita) | ~5 | MLP | stesso significato per ogni ala | scelti a mano: possono perdere dettagli |
| **Coordinate** della superficie a stazioni fisse lungo la corda | ~64 per elemento | MLP | generale per qualunque profilo | tanti ingressi per pochi esempi |
| **PCA delle coordinate** | 4–8 | MLP | pochi numeri, generale | la PCA dipende dal dataset su cui la calcoli |
| **SDF su griglia** | migliaia (4,608 a 96 × 48) | CNN (o MLP su pochi punti campionati) | qualunque forma, più elementi e suolo; standard per prevedere campi | perde il bordo d'uscita se la griglia è grossa; tantissimi numeri per due uscite |

**Spiegare la PCA in due frasi.** Le coordinate di 1,000 profili sono 1,000 vettori da 64 numeri, ma i profili si somigliano: variano in pochi "modi" (più spesso, più curvo, curvatura più avanti...). La PCA trova questi modi; ogni profilo diventa la sua "ricetta" in quei modi. Su AirfRANS **4 modi spiegano il 99.9% della varianza** (6 modi il 99.99%).

**Il numero che conta: quanto bene ogni codifica "contiene" una grandezza fisica.** Nell'analisi su AirfRANS (1,000 profili veri) abbiamo provato a ricavare l'angolo di portanza nulla $\alpha_0$ (quello della physics loss della Lesson 1) da ogni codifica con un modello **lineare** (un "linear probe"). Se un modello lineare ci riesce, l'informazione è lì, facile da usare.

| Codifica | Errore su $\alpha_0$ (rmse) | $R^2$ |
|---|---|---|
| Cifre NACA grezze | **0.92°** | 0.81 |
| Descrittori (5) | 0.062° | 0.999 |
| PCA delle coordinate (6) | 0.016° | 0.9999 |
| SDF 96 × 48 (4,608 numeri) | 0.0015° | 1.0 |
| Coordinate (64) | 0.0002° | 1.0 |

Le coordinate sono esatte quasi per costruzione: $\alpha_0$ dipende linearmente dalla linea media, che è la media delle coordinate di dorso e ventre. Le cifre grezze sono le peggiori, per il motivo che segue.

**Due lezioni da AirfRANS (da raccontare, sono memorabili).**

1. **I nomi mentono: le grandezze si calcolano dalla forma.** 120 dei 489 profili NACA a 4 cifre di AirfRANS hanno nel nome una curvatura fino al 7% ma, per come li costruisce il generatore, sono **quasi simmetrici**. Calcolando $\alpha_0$ dalle cifre si sbaglia fino a **5.9°**; calcolandolo dalla geometria vera viene tra 0 e −0.14°, come deve essere. È la domanda D1 dell'audit ("cosa c'è davvero nel dataset?") in un dataset serio, pubblicato a NeurIPS.
2. **Traslare e scalare sì, ruotare no.** Se si "normalizza" un profilo ruotandolo sulla sua corda, l'angolo d'attacco cambia (fino a 0.4° su AirfRANS). La rotazione cambia la fisica.

**Il costo della SDF** (figura `04_sdf_trailing_edge.png`, già vista nella Lesson 2):

| Griglia | Passo (% corda) | Corda persa al bordo d'uscita |
|---|---|---|
| 48 × 24 | 3.2 | 9.1% |
| 96 × 48 | 1.6 | 4.4% |
| 192 × 96 | 0.8 | 2.1% |
| 384 × 192 | 0.4 | 1.1% |

E un dato sorprendente: anche le immagini SDF, viste come vettori, stanno in 8 modi PCA per il 99.9%. **4,608 pixel portano la stessa informazione di 4–8 numeri.** Con circa 56 ali di training, una CNN su 4,608 ingressi ha molti più parametri che esempi.

**Conclusione da dire.** Non c'è una codifica giusta in assoluto: dipende da cosa si vuole prevedere (due numeri o un campo intero) e da quanti dati ci sono. **Quale funziona meglio sul nostro dataset è un esperimento di T4**: stessa rete, codifiche diverse (parametri → descrittori → PCA → coordinate → SDF), stessi seed, si confronta. La proposta di partenza per prevedere solo $C_l$ e $C_d$: coordinate o loro PCA, più $\alpha$ e $h/c$ come ingressi separati. La SDF diventa la scelta naturale se si passa a prevedere campi (obiettivo stretch S5) o forme che i parametri non descrivono.

**Cosa mostrare.** La tabella delle codifiche; `figures/01_gallery.png` (i profili, con i 120 "falsi curvi" in rosa); `05_alpha0_probe.png`; `04_sdf_trailing_edge.png`.

**Domande probabili.**

- *"Allora la SDF della Lesson 2 era inutile?"* No: è la codifica più generale (più elementi, suolo, qualunque forma) e quella giusta per i campi. Per due numeri da 56 ali probabilmente è sovradimensionata. Si misura.
- *"Perché non usare direttamente i 4 parametri del dataset?"* Si può, ed è la baseline più semplice. Ma una rete addestrata sui parametri funziona solo per ali generate con quei parametri; con coordinate o SDF può, in principio, leggere un'ala qualunque.

**Errori da evitare.** Non dire che una codifica "è migliore" in generale. Non mostrare risultati sul nostro dataset: non ci sono ancora.

---

## 8. Blocco 2 — Input e normalizzazione (13–17 min)

**Obiettivo.** Tre regole pratiche, ciascuna con il suo perché.

**Cosa dire.**

**1. Ingressi adimensionali e di ordine 1.** Una rete con attivazione `tanh` lavora bene quando gli ingressi sono numeri tra circa −3 e 3. Se un ingresso vale 15 (gradi) e un altro 0.2 ($h/c$), i pesi iniziali li trattano in modo molto diverso e l'ottimizzatore fatica. Nella Lesson 1 la rete riceveva $x = \alpha/10$ proprio per questo. Due modi:

- **riscalare a mano** con una costante fisica sensata ($\alpha/10$ in gradi, $h/c$ così com'è);
- **standardizzare**: $x' = (x - \mu)/\sigma$, con media $\mu$ e deviazione standard $\sigma$ dell'ingresso. Anche le uscite si possono standardizzare ($C_l$ vale circa 2–5, $C_d$ circa 0.03–0.12: scale diverse; con una loss sulla somma degli errori quadratici, $C_l$ dominerebbe).

**2. Normalizzare con statistiche del solo training.** $\mu$ e $\sigma$ si calcolano **solo** sulle righe di training, poi si applicano identiche a validazione e test. Se si calcolano su tutto il dataset, un po' di informazione del test entra nell'addestramento: è una piccola fuga di informazione (leakage), lo stesso tipo di errore del paper che ha calibrato sul dataset intero. In scikit-learn: `scaler.fit(X_train)`, poi `scaler.transform(X_val)`.

**3. Angolo e altezza da terra restano ingressi espliciti, anche se la forma entra come SDF.** Motivo: il vincolo fisico del blocco 3 si scrive su $\partial C_l/\partial\alpha$, e autograd sa derivare l'uscita **solo rispetto a un ingresso della rete**. Se l'angolo fosse "nascosto" ruotando l'immagine SDF, la derivata rispetto ad $\alpha$ passerebbe per la costruzione della SDF e il vincolo diventerebbe molto più complicato.

**Attenzione alle unità quando si deriva** (aggancio al blocco 3): se la rete riceve $x = \alpha/10$ in gradi, autograd dà $\partial \hat C_l/\partial x$, non $\partial \hat C_l/\partial\alpha$. Lo vediamo subito.

**Cosa mostrare.** Le tre regole, ognuna con una riga di perché.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)          # statistiche del solo training
X_train_n, X_val_n = scaler.transform(X_train), scaler.transform(X_val)
```

---

## 9. Blocco 3 — Il vincolo fisico su una derivata (17–25 min, live)

**Obiettivo.** Mostrare che una legge fisica si può imporre non solo sul valore dell'uscita (Lesson 1) ma sulle sue derivate rispetto all'ingresso (la tecnica delle PINN della Lesson 2), e che le unità contano.

**Cosa dire.**

Nella Lesson 1 la physics loss confrontava il **valore** della rete con la teoria: $\hat C_l(\alpha) = 2\pi(\alpha - \alpha_0)$. Per usarla serve conoscere $\alpha_0$, che dipende dalla forma. La stessa legge si può dire in modi più deboli, che richiedono meno informazioni:

| Versione | Vincolo | Cosa serve sapere | In parole |
|---|---|---|---|
| Il valore (Lesson 1) | $\hat C_l(\alpha) = 2\pi(\alpha - \alpha_0)$ | $\alpha_0$ | la curva passa esattamente per la retta della teoria |
| La pendenza | $\dfrac{d\hat C_l}{d\alpha} = 2\pi$ per radiante | niente $\alpha_0$ | la curva sale con la pendenza giusta, ovunque sia |
| La linearità | $\dfrac{d^2\hat C_l}{d\alpha^2} = 0$ | né $\alpha_0$ né $2\pi$ | la curva è una retta, qualunque retta |

Le derivate si calcolano con lo stesso strumento della Lesson 2, `torch.autograd.grad`, sull'uscita della rete rispetto al suo ingresso. Il vincolo si controlla solo **a piccoli angoli** (da −4° a 6°), dove la teoria vale: 50 angoli estratti a caso a ogni passo, **senza dati** (sono punti di collocazione, come i 200 istanti della PINN della Lesson 2).

**La trappola della scala.** La rete non riceve $\alpha$ ma $x = \alpha/10$ in gradi. Autograd quindi dà $d\hat C_l/dx$, non $d\hat C_l/d\alpha$. La legge invece è per radiante. Regola della catena:

$$\frac{d\hat C_l}{d\alpha_{\text{rad}}} = \frac{d\hat C_l}{dx}\cdot\frac{dx}{d\alpha_{\text{rad}}}, \qquad x = \frac{\alpha_{\text{gradi}}}{10} = \frac{180}{10\pi}\,\alpha_{\text{rad}} \;\Rightarrow\; \frac{dx}{d\alpha_{\text{rad}}} = \frac{18}{\pi} \approx 5.73$$

Quindi, in unità della rete, la pendenza da imporre è $2\pi \cdot \pi/18 \approx 1.10$, non 6.28. Se il fattore si dimentica, si chiede alla rete una pendenza 5.73 volte troppo grande.

**Il codice (da mostrare, sono poche righe di `demo_l3_slope.py`).**

```python
a_phys = rng.uniform(-4, 6, 50)                                   # 50 angoli piccoli, nuovi a ogni passo
x = torch.tensor(a_phys / 10).reshape(-1, 1).requires_grad_(True) # l'ingresso della rete: x = alpha/10
cl = net(x)
dcl_dx = torch.autograd.grad(cl, x, torch.ones_like(cl), create_graph=True)[0]
factor = 18 / math.pi                                             # dx/dalpha_rad: la regola della catena
loss = loss_dati + lam * (dcl_dx * factor - 2 * math.pi).pow(2).mean()
```

Leggerlo riga per riga: punti di collocazione → derivata dell'uscita rispetto all'ingresso (con `create_graph=True` perché poi va derivata rispetto ai pesi) → fattore della regola della catena → residuo al quadrato, pesato con $\lambda$, sommato alla loss sui dati.

**Live** (stessi dati, stessa rete e stessi seed di `demo_l1_network.py`; ogni comando circa 30 s su CPU, gira sui seed 0–4):

```bash
python course/L3/demo_l3_slope.py --points 6
python course/L3/demo_l3_slope.py --points 6 --constraint slope
python course/L3/demo_l3_slope.py --points 6 --constraint slope --no-chain-rule
```

**Risultati attesi** (verificati il 22 ottobre; errore quadratico medio di validazione su 201 angoli tra −5° e 20°, media ± deviazione standard sui seed 0–4):

| 6 punti di training | Errore di validazione | Pendenza a 2° (per rad, teoria 6.28) |
|---|---|---|
| Rete da sola | 0.0072 ± 0.0026 | 6.75 |
| Con la pendenza | **0.0016 ± 0.0022** (meglio in 5 seed su 5) | **6.28** |
| Pendenza, fattore dimenticato | 2.02 ± 0.87 (circa 280× peggio della rete da sola) | 34.5 |

Per seed, con la pendenza: 0.0003, 0.0006, 0.0008, 0.0055, 0.0009 (rete da sola: 0.0083, 0.0098, 0.0060, 0.0087, 0.0033).

**Come commentare.**

1. Con 6 punti la rete da sola indovina la forma della curva solo in parte; il vincolo le dice "a piccoli angoli sali con pendenza $2\pi$" e l'errore scende di circa 4.5 volte, in tutti e 5 i seed.
2. La pendenza della rete addestrata a 2° è 6.28: esattamente la teoria. Il vincolo è rispettato.
3. **Con il fattore dimenticato la rete obbedisce benissimo al vincolo sbagliato** (pendenza 34.5 ≈ 6.28 × 5.73) e l'errore esplode. È lo stesso errore del paper (una costante sbagliata di 11.5× nella physics loss): **un vincolo fisico con le unità sbagliate è peggio di nessun vincolo.**

**Il peso $\lambda$** (`--lam`, di default 1): quanto conta il termine fisico rispetto ai dati. Con 6 punti, pendenza, seed 0–4:

| $\lambda$ | Errore di validazione (seed 0–4) | Rispetto alla rete da sola (0.0072 ± 0.0026) |
|---|---|---|
| 0.01 | 0.0038 ± 0.0035 | meglio, ma poco |
| 0.1 | 0.0023 ± 0.0023 | meglio |
| **1** (default) | **0.0016 ± 0.0022** | **il migliore** |
| 10 | 0.037 ± 0.038 | **peggio**: 5 volte la rete da sola |
| 100 | 0.099 ± 0.009 | **molto peggio**: 14 volte |

La pendenza a 2° è 6.28 in tutti i casi: il vincolo è sempre rispettato. Con $\lambda$ grande però la loss è quasi solo il termine fisico, e la rete "si dimentica" dei 6 punti di dati: impara la pendenza giusta ma non il valore giusto (né la curva fuori dai piccoli angoli). Con $\lambda$ piccolo il vincolo conta poco. **Il vincolo è giusto, eppure con il peso sbagliato fa danni.**

Da questa tabella viene la regola del protocollo di T5 (blocco 6): $\lambda$ va provato su più ordini di grandezza, perché l'effetto dipende da $\lambda$.

**Con tanti dati il vincolo serve meno.** Con 40 punti di training:

| 40 punti di training | Errore di validazione (seed 0–4) |
|---|---|
| Rete da sola | 0.0003 ± 0.0002 |
| Con la pendenza | 0.0003 ± 0.0001 |

Con 40 punti i dati bastano: il vincolo non migliora l'errore (riduce solo un po' la variabilità tra i seed). Con 6 punti lo riduceva di circa 4.5 volte. **È esattamente l'ipotesi di T5 in piccolo: la fisica aiuta quando i dati sono pochi.**

**Da non vendere troppo (dirlo chiaramente).** Qui la curva è inventata, e il suo tratto lineare *è* la teoria: il vincolo è giusto per costruzione. Sul nostro dataset (ala a due elementi vicino al suolo) la pendenza $2\pi$ non vale così com'è (blocco 6). La demo insegna il **meccanismo** e la **trappola delle unità**, non dimostra che la fisica aiuta sulla nostra ala.

**Domande probabili.**

- *"Perché i punti fisici solo tra −4° e 6°?"* Perché la teoria del profilo sottile vale solo a piccoli angoli: oltre, il flusso si stacca (stallo) e la curva piega. Imporre la pendenza fino a 20° sarebbe imporre una cosa falsa.
- *"Perché la linearità non vuole $2\pi$?"* Perché dice solo "retta": la pendenza la decide la rete dai dati. È il vincolo più debole e il più sicuro quando non si conosce la pendenza vera.
- *"E con ReLU?"* La derivata seconda di una rete ReLU è zero quasi ovunque: il vincolo di linearità è sempre soddisfatto e non dà nessun gradiente. Il compito 1 lo fa vedere (risultati identici alla rete da sola).

---

## 10. Blocco 4 — Valutare un surrogato senza ingannarsi (25–35 min)

**Obiettivo.** Il blocco più importante per T4. Cinque regole, ognuna con un esempio concreto.

### 10.1 Split per geometria, mai per righe (con la demo del paper)

**Cosa dire.** Il test deve simulare l'uso vero: **arriva un'ala nuova, mai vista**, e la rete deve prevederne $C_l$ e $C_d$. Se nel test ci sono righe "quasi uguali" a righe di training, il test misura la memoria, non la capacità di generalizzare. Si chiama **leakage** (fuga di informazione).

**Live:** `python course/L1/audit_replication.py --full` (pochi secondi), sezione "CHECK 5". Il paper divide a caso 5,000 righe che sono 5,000 iterazioni consecutive della stessa simulazione:

| Misura (dall'output dello script) | Valore |
|---|---|
| Autocorrelazione tra righe consecutive ($C_d$, lag 1) | 0.78 |
| Distanza media, in passi, da ogni riga di validazione alla riga di training più vicina | 1.045 |
| Righe di validazione con una riga di training a **un passo** di distanza | **96.1%** |
| Distanza massima | 4 passi |

In più: lo split dichiarato "stratificato" nel codice non lo è, e la calibrazione lineare finale è fatta su tutto il dataset, validazione inclusa. Ogni metrica calcolata dopo è addestrata sul proprio test.

**Collegare al nostro dataset.** Da noi la stessa ala compare in circa 14 condizioni (altezze e angoli diversi). Due righe della stessa ala a $h/c$ = 0.30 e 0.32 sono quasi la stessa riga. Se una va nel training e l'altra nel test, la rete "riconosce" l'ala. Per questo **lo split ufficiale è per geometria**: tutte le righe di un'ala stanno nello stesso insieme (circa 56 ali training, 12 validazione, 12 test).

```python
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
train_idx, test_idx = next(gss.split(X, y, groups=wing_id))   # nessuna ala in entrambi
```

**Gli split di estrapolazione** misurano una cosa diversa e più dura: altezze fuori dal range di training ($h/c \ge 2$, lontano dal suolo) e flap agli estremi. Qui le reti tipicamente sbagliano di più, ed è proprio dove un vincolo fisico potrebbe aiutare (blocco 6).

### 10.2 Baseline prima di tutto

**Cosa dire.** La lezione più dura del paper: una regressione lineare con 26 parametri faceva $R^2 = 1$, la rete con 12,066 parametri 0.968. **Prima di dire che una rete funziona, bisogna sapere quanto fa un modello semplice.** La scala per T4, dal più semplice al più complesso:

| Modello | Perché c'è | Cosa ci dice |
|---|---|---|
| Regressione lineare | il riferimento minimo | quanto del problema è "una retta" |
| Random forest | non lineare, quasi senza iperparametri | un riferimento non lineare robusto |
| Gradient boosting | spesso il migliore su tabelle piccole | il rivale vero della rete |
| MLP | la nostra rete | serve se batte il gradient boosting, o se serve per T5 (derivate) |

**Se la rete non batte il gradient boosting, si dice.** È un risultato. La rete resta comunque necessaria per T5, perché sul gradient boosting non si possono calcolare derivate con autograd.

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
for model in [LinearRegression(), RandomForestRegressor(random_state=s), HistGradientBoostingRegressor(random_state=s)]:
    model.fit(X_train, y_train); err = relative_error(model.predict(X_val), y_val)
```

### 10.3 Mai un numero solo

Almeno 3 seed (meglio 5) per ogni modello, media ± deviazione standard (blocco 0). Per i modelli di scikit-learn il seed è `random_state`; per la rete, `torch.manual_seed`. Con poche ali conviene anche ripetere lo split (validazione incrociata **per gruppi**, `GroupKFold`): ogni ala finisce una volta nella validazione.

### 10.4 Curve di apprendimento

**Cosa dire.** La domanda "bastano 80 ali?" ha una risposta misurabile: si addestra con 10, 20, 40, tutte le ali di training e si guarda l'errore sulla validazione. Se la curva scende ancora all'ultimo punto, più dati aiuterebbero; se è piatta, il collo di bottiglia è altrove (modello, codifica, rumore della CFD). Il dataset è campionato con Sobol apposta: le prime 10, 20, 40 ali sono già ben distribuite, quindi basta prendere i prefissi.

Le curve di apprendimento sono anche il cuore di T5: **l'ipotesi è che il vincolo fisico aiuti di più quando le ali sono poche.** Si vede confrontando due curve (con e senza vincolo). La demo del blocco 3 ne è un assaggio: con 6 punti il vincolo riduce l'errore di circa 4.5 volte, con 40 punti non lo cambia (0.0003 in entrambi i casi).

### 10.5 Metriche: $R^2$ non basta

**Cosa dire.** $R^2 = 1 - \dfrac{\sum (y - \hat y)^2}{\sum (y - \bar y)^2}$ misura quanta della variabilità dei dati il modello spiega. Problemi:

- **Dipende da quanto variano i dati.** Se nel test $C_l$ varia molto (da 2 a 5), anche un modello mediocre fa $R^2$ alto. Il paper aveva $R^2 = 1$ su un problema banale.
- **Non dice quanto si sbaglia in unità che contano.** Un ingegnere vuole sapere "sbaglio il $C_l$ del 2%" o "di 0.05".

Quindi accanto a $R^2$ si riporta sempre un **errore relativo** (ad esempio l'errore medio assoluto diviso il valore medio, in %) e l'errore assoluto, **per $C_l$ e $C_d$ separatamente** (e per elemento se si prevede per elemento). $C_d$ è più piccolo e più difficile: un errore di 0.005 su $C_d$ = 0.07 è il 7%.

### 10.6 I casi non convergenti

Ogni riga ha un flag. I casi `oscillating` o `failed` (tipicamente molto vicino al suolo o vicino allo stallo) hanno numeri meno affidabili. Regola di T4: **risultati con e senza quei casi, riportati separatamente**. Non si buttano in silenzio (la domanda 3 della Lesson 2): sono anche l'informazione "qui la CFD stessa non è sicura".

**Cosa mostrare per tutto il blocco.** Le cinque regole come lista; la tabella del CHECK 5; lo schema "un'ala = tutte le sue righe nello stesso insieme"; un esempio di tabella dei risultati come la vorremmo in T4:

| Modello | $C_l$: errore relativo | $C_d$: errore relativo | $R^2$ ($C_l$) | Seed |
|---|---|---|---|---|
| Lineare | ... ± ... | ... ± ... | ... | 5 |
| Random forest | ... | ... | ... | 5 |
| Gradient boosting | ... | ... | ... | 5 |
| MLP | ... | ... | ... | 5 |

(vuota di proposito: la riempiranno loro).

---

## 11. Blocco 5 — Addestrare con pochi dati (35–43 min)

**Obiettivo.** Dare gli strumenti per non ingannarsi con circa 56 ali di training.

**Cosa dire.**

**Overfitting, con un esempio che già conoscono.** Dalla Lesson 1, `demo_l1_network.py --hidden 4/32/128`:

| Neuroni per strato | 6 punti | 40 punti |
|---|---|---|
| 4 | 0.0200 | 0.0008 |
| 32 | 0.0083 | 0.0002 |
| 128 | 0.0057 | 0.0002 |

Qui la rete più grande non va peggio (la curva è liscia e `tanh` è "gentile"), ma in generale con pochi dati una rete grande può **imparare a memoria** i punti di training, rumore compreso, e sbagliare tra un punto e l'altro. Il segnale: l'errore di training continua a scendere mentre quello di validazione si ferma o risale. **Si guarda sempre la curva di validazione durante l'addestramento.**

**Gli strumenti contro l'overfitting** (spiegare cosa fa ognuno, in una frase):

| Strumento | Cosa fa | In PyTorch |
|---|---|---|
| Rete più piccola | meno parametri, meno capacità di memorizzare | meno neuroni/strati |
| Weight decay | penalizza pesi grandi: la funzione resta più liscia | `torch.optim.AdamW(..., weight_decay=1e-4)` |
| Early stopping | si ferma quando l'errore di validazione smette di scendere; si tengono i pesi migliori | un ciclo che salva il modello migliore |
| Dropout | spegne a caso dei neuroni durante il training: la rete non può dipendere da uno solo | `torch.nn.Dropout(p=0.1)` |
| Ensemble | media di più reti (seed diversi): riduce la varianza e dà una stima dell'incertezza | addestrare 5 reti e mediare |
| Un vincolo fisico | informazione in più dove non ci sono dati | blocchi 3 e 6: è l'ipotesi di T5 |

**Validazione e test: la regola d'oro.**

- **Training**: le ali su cui la rete impara.
- **Validazione**: le ali su cui si scelgono iperparametri (numero di neuroni, learning rate, weight decay, $\lambda$, quando fermarsi). Si guardano quante volte si vuole.
- **Test**: le ali che si guardano **una volta sola, alla fine**, per il numero da riportare. Se si guarda il test per scegliere qualcosa, il test diventa validazione e il numero finale è ottimista.

Con 12 ali di validazione la stima è rumorosa: per scelte importanti conviene la validazione incrociata per gruppi (`GroupKFold` con 5 gruppi di ali).

**Iperparametri senza toccare il test.** Pochi tentativi ragionati (learning rate 1e-3 o 3e-3; 32 o 64 neuroni; weight decay 0, 1e-5, 1e-4) valutati sulla validazione. Si annota tutto: ogni tentativo fatto è un tentativo "speso" sulla validazione.

**Attivazione (se vuoi agganciare la Lesson 2).** Se il modello deve essere derivato (T5), l'attivazione deve essere liscia: `tanh`, `softplus`, `GELU`, `SiLU`. Con ReLU la derivata seconda è nulla quasi ovunque e i vincoli sulle derivate seconde non funzionano (compito 1; oppure `course/L2/demo_l2_pinn_1d.py --activation relu`: loss 14× più bassa, errore 175× peggio).

**Cosa mostrare.** La tabella hidden 4/32/128; lo schema training / validazione / test; la tabella degli strumenti.

**Errori da evitare.** Non presentare una ricetta unica: il blocco serve a dare il vocabolario e la regola validazione/test. I dettagli li sceglieranno in T4.

---

## 12. Blocco 6 — Il vincolo fisico vicino al suolo, e il protocollo di T5 (43–52 min)

**Obiettivo.** Spiegare perché il vincolo della Lesson 1 non si può riusare così com'è sulla nostra ala, quali vincoli restano, e come si misura se aiutano.

**Cosa dire.**

**Perché $2\pi(\alpha - \alpha_0)$ non basta.** La teoria del profilo sottile vale per un profilo **sottile, da solo, in aria libera, ad angoli piccoli**. La nostra ala è:

- **vicino al suolo**: l'aria tra ala e suolo accelera, la deportanza cresce avvicinandosi (3.44 → 4.31 da 15 a 8 cm nella simulazione della Lesson 2), finché molto vicino crolla. La teoria in aria libera non lo sa;
- **a due elementi**: il flap e la fessura cambiano la pendenza e il valore;
- **viscosa**: anche in aria libera, la pendenza vera di un profilo reale è tipicamente il 5–10% sotto $2\pi$.

Quindi imporre "pendenza = $2\pi$" ovunque sarebbe imporre una cosa falsa: come il fattore dimenticato del blocco 3, la rete obbedirebbe al vincolo sbagliato.

**I vincoli che restano** (candidati, da scegliere sul dataset; da `Project_Objectives.md`):

| Vincolo | Cosa dice | Dove vale | Come si scrive nella loss |
|---|---|---|---|
| $C_d \ge 0$ | la resistenza non è mai negativa | sempre | penalità solo se violato: $\text{mean}(\max(0, -\hat C_d)^2)$ |
| Limite lontano dal suolo | per $h/c$ grande l'ala si comporta come in aria libera, e lì la teoria del profilo sottile (pendenza o linearità) torna valida | casi con $h/c$ grande, inclusi apposta nel dataset | vincolo di pendenza/linearità solo su punti di collocazione con $h/c$ grande |
| Linearità a piccoli angoli | $\partial^2 C_l/\partial\alpha^2 \approx 0$ con flusso attaccato | da verificare sui dati: l'effetto suolo può piegare la curva; le scansioni in $\alpha$ su 20 geometrie del dataset servono proprio a questo | come `--constraint linear` del blocco 3 |
| Simmetria | un profilo rovesciato in aria libera è l'immagine speculare di uno dritto ($C_l \to -C_l$ con $\alpha \to -\alpha$, $C_d$ uguale) | solo in aria libera | confronto tra la rete in $(\alpha)$ e in $(-\alpha)$ su forme specchiate |

Regole comuni: ogni vincolo è adimensionale; il termine fisico si valuta su **punti di collocazione** (configurazioni generate a caso, senza CFD), non solo sui punti di training: è proprio così che il vincolo porta informazione **dove non ci sono dati**; si confrontano versioni rigide e morbide (ad esempio pendenza esatta contro sola linearità).

```python
# schema della loss di T5 (da adattare): dati + vincolo su punti di collocazione
loss = mse(net(X_train), y_train)
X_col = sample_configurations(n=256)                      # forme e condizioni a caso, nessuna CFD
cd_pred = net(X_col)[:, 1]
loss = loss + lam * torch.relu(-cd_pred).pow(2).mean()    # Cd >= 0, penalità solo se violato
```

**Il protocollo di T5: come si misura se il vincolo aiuta.**

1. **Stessa rete di T4**, stessi ingressi, stesso split. L'unica differenza è il termine fisico.
2. **$\lambda$ su almeno 4 ordini di grandezza** (ad esempio 0.01, 0.1, 1, 10, 100): nella demo del blocco 3 lo stesso vincolo, giusto, passa da "riduce l'errore di 4.5 volte" ($\lambda$ = 1) a "lo aumenta di 14 volte" ($\lambda$ = 100). Scegliere un solo $\lambda$ è un modo per trovare il risultato che si vuole; $\lambda$ si sceglie sulla validazione, mai sul test.
3. **Diverse dimensioni del training**: 5, 10, 20, 40, 56 ali. L'ipotesi è che il vincolo aiuti con poche ali e serva meno con tante.
4. **Almeno 3 seed** per ogni combinazione: 5 valori di $\lambda$ × 5 dimensioni × 3 seed = 75 addestramenti per vincolo. Sono indipendenti: si parallelizzano facilmente (è il lavoro dell'infra lead).
5. **Si guarda soprattutto lo split di estrapolazione** (lontano dal suolo, flap estremi): è dove i dati mancano e la fisica può dare qualcosa.
6. **Si riporta comunque.** "Il vincolo non aiuta" o "aiuta solo con meno di 10 ali" sono risultati validi. Il paper ha affermato che la fisica aiuta senza misurarlo: noi lo misuriamo.

**Cosa mostrare.** La lista "perché non basta"; la tabella dei vincoli; il protocollo come sei punti; una griglia vuota $\lambda$ × numero di ali, da riempire in T5.

**Domande probabili.**

- *"E se nessun vincolo aiuta?"* È la risposta alla domanda del paper, fatta bene. Va nella presentazione finale come risultato.
- *"Perché non mettere direttamente le equazioni di Navier–Stokes nella loss, come una PINN?"* Perché a $Re \approx 10^6$, con strato limite e turbolenza, una PINN non è competitiva con la CFD (lo si spiega nella Lesson 4). Il nostro vincolo è più modesto e misurabile.

---

## 13. Chiusura e compiti (52–55 min)

**Riassunto in sei frasi** (una per blocco, vedi §1). Poi aggancio alle quattro domande della Lesson 2: "adesso avete gli strumenti; ne parliamo insieme nell'ultima lezione".

**Compiti (circa 1 h, prima del kickoff)**

1. **Valore, pendenza o retta? (20 min).** Con `course/L3/demo_l3_slope.py --points 6 --compare` provare `--constraint value`, `--constraint slope`, `--constraint linear`; poi `--constraint linear --activation relu`. Riempire una tabella: in quanti seed su 5 il vincolo aiuta? Risultati attesi (verificati):

   | Vincolo | Errore di validazione | Aiuta in |
   |---|---|---|
   | Rete da sola | 0.0072 ± 0.0026 | — |
   | Valore (Lesson 1) | 0.0029 ± 0.0041 | 4 seed su 5 |
   | Pendenza | 0.0016 ± 0.0022 | 5 su 5 |
   | Linearità | 0.0018 ± 0.0014 | 5 su 5 |
   | Linearità con ReLU | 0.0109 ± 0.0113, identico alla stessa rete ReLU senza vincolo | 0 (il vincolo non dà gradiente) |

   Domanda da portare: perché con ReLU il vincolo di linearità non fa niente?
2. **Leggere `Project_Objectives.md` (30 min)**, in particolare T4 e T5, e arrivare al kickoff con una preferenza di ruolo (audit & surrogate lead, physics lead, PINN lead, infra lead).
3. **Facoltativo:** provare `--constraint slope --lam 0.01` e `--lam 100` e spiegare la differenza (attesi: 0.0038 ± 0.0035 e 0.099 ± 0.009; con $\lambda$ = 100 il vincolo è rispettato ma i dati vengono ignorati).

---

## 14. Domande probabili

**"Quale codifica useremo?"** Si decide misurando, in T4. Proposta di partenza: coordinate o PCA, più $\alpha$ e $h/c$ espliciti.

**"Bastano 80 ali?"** È la domanda giusta, ed è il motivo delle curve di apprendimento. Con pochi dati il vincolo fisico ha più possibilità di aiutare: è l'ipotesi di T5.

**"Perché non una CNN direttamente sulla SDF?"** Si può provare. Con circa 56 ali di training una CNN ha molti più parametri che esempi; probabilmente una rete piccola su poche feature va meglio. Si misura.

**"Perché non usare solo il gradient boosting, se è più bravo?"** Per T4 può essere la risposta. Per T5 serve un modello derivabile: sul gradient boosting non si possono calcolare $\partial C_l/\partial\alpha$ con autograd.

**"Quando il dataset è pronto?"** v0 pilota a metà novembre (per costruire la pipeline di T4), v1 a fine dicembre. Il lavoro vero su T4 e T5 parte dopo il freeze degli esami.

**"Il vincolo di pendenza della demo vale sulla nostra ala?"** No, non così com'è: la demo è su una curva inventata in aria libera. Sulla nostra ala vale al più lontano dal suolo (blocco 6).

**"Che differenza c'è tra validazione e test?"** La validazione si usa per scegliere; il test si guarda una volta sola, alla fine.

**"Quanti seed?"** Almeno 3, meglio 5. Il costo è basso (CPU, reti piccole) e senza seed non si può dire niente.

---

## 15. Glossario

| Termine | Significato |
|---|---|
| **Surrogato** | modello veloce che imita una simulazione costosa: dai parametri dell'ala a $C_l$, $C_d$ in millisecondi invece che in minuti di CFD |
| **Seed** | numero che fissa tutte le scelte casuali di un addestramento (dati estratti, rumore, pesi iniziali) |
| **Split** | divisione dei dati in training, validazione, test |
| **Split per geometria (per gruppi)** | tutte le righe di un'ala stanno nello stesso insieme |
| **Leakage** | informazione del test che entra, anche indirettamente, nell'addestramento |
| **Baseline** | modello semplice di riferimento da battere |
| **Curva di apprendimento** | errore di validazione in funzione della quantità di dati di training |
| **$R^2$** | quota della variabilità dei dati spiegata dal modello; 1 = perfetto |
| **Errore relativo** | errore diviso il valore vero (o il suo ordine di grandezza), in % |
| **Overfitting** | il modello impara a memoria il training e generalizza male |
| **Regolarizzazione** | ogni tecnica che limita l'overfitting (weight decay, dropout, early stopping, rete più piccola) |
| **Iperparametro** | scelta non imparata dai dati: neuroni, learning rate, $\lambda$, epoche |
| **Punti di collocazione** | ingressi generati a caso dove si controlla il vincolo fisico; non servono simulazioni |
| **$\lambda$** | peso del termine fisico nella loss |
| **Codifica (encoding)** | il modo di trasformare la forma dell'ala in un vettore di numeri |
| **PCA** | trova le poche direzioni in cui i dati variano di più; riassume tanti numeri in pochi |
| **SDF** | signed distance function: in ogni punto, la distanza dalla parete più vicina, negativa dentro |
| **Linear probe** | un modello lineare usato per misurare se un'informazione è "facilmente leggibile" in una codifica |
| **$h/c$** | altezza da terra divisa per la corda |
| **$C_l$, $C_d$** | coefficienti di portanza (qui deportanza) e resistenza, adimensionali |

---

## 16. File e comandi

Tutti i comandi dalla root della repo, con il venv del corso attivo. Tempi misurati su CPU (2 core); su un portatile recente sono simili o più brevi.

| File | Dove | Cosa fa | Comando | Tempo | Numeri attesi |
|---|---|---|---|---|---|
| `demo_l3_slope.py` | `course/L3/` | vincolo su valore/pendenza/linearità, seed 0–4 | `python course/L3/demo_l3_slope.py --points 6 [--constraint slope] [--no-chain-rule] [--compare] [--lam X] [--activation relu]` | circa 30 s (60 s con `--compare`) | tabelle del blocco 3 e dei compiti |
| `demo_l1_network.py` | `course/L1/` | la demo della Lesson 1 (stessi dati) | `python course/L1/demo_l1_network.py --points 6 --hidden 4` | circa 10 s | tabella del blocco 5 |
| `audit_replication.py` | `course/L1/` | i difetti del paper; `--full` include il controllo di leakage | `python course/L1/audit_replication.py --full` | pochi s | tabella del blocco 4 |
| `analyse_shape_encodings.py` | `course/_instructor/shape_input/` | confronto delle codifiche su AirfRANS; scrive `results.json` e `figures/` | `python course/_instructor/shape_input/analyse_shape_encodings.py` | circa 30 s | tabelle del blocco 1 |
| `figures/01–05` | `course/_instructor/shape_input/figures/` | galleria dei profili, PCA/CST, esempi SDF, bordo d'uscita, probe di $\alpha_0$ | — | — | — |
| `demo_l2_pinn_1d.py` | `course/L2/` | facoltativo: PINN del calore, `--activation relu` | `python course/L2/demo_l2_pinn_1d.py --activation relu` | circa 30 s | loss 14× più bassa, errore 61% |
| Output attesi | `course/L3/expected_output/` | gli output dei comandi live, per il piano B | — | — | — |

Documenti da avere a portata: `Project_Objectives.md` (T4, T5, ruoli), `D2_Dataset_Proposal.md` (il dataset), `Shape_Input_and_F1_Data.md` (l'analisi delle codifiche).

---

## 17. Letture per ruolo (dopo il kickoff)

| Ruolo | Lettura |
|---|---|
| Audit & surrogate lead | README del dataset (dalla v0); Kapoor & Narayanan (2023), *Leakage and the reproducibility crisis in ML-based science* |
| Physics lead | Un capitolo sulla teoria del profilo sottile; Zhang, Toet & Zerihan (2006), *Ground effect aerodynamics of race cars* |
| PINN lead | Raissi et al. (2019) §1–3; *An Expert's Guide to Training PINNs* §2–4 |
| Infra lead | Documentazione DeepXDE; guida di Kaggle alle GPU |
