# PID vs. Sliding-Mode Control for Net-Distance Keeping in Aquaculture Cage Inspection

Progetto per Research Track II (Prof. Recchiuto, DIBRIS, Università degli Studi di
Genova). Confronto tra un controllore PID e uno Sliding Mode Control (SMC) sul task di
mantenimento della distanza di un BlueROV2 da una rete da acquacoltura, sotto disturbi
di corrente marina di ampiezza crescente, con lo scopo di individuare la soglia di
corrente oltre la quale ciascun controllore smette di mantenere il ROV nella finestra
operativa 70–150 cm richiesta dal detector di buchi (YOLOv8, da López-Barajas et al.,
JMSE 2024).

## Struttura del repository

```
report/report.tex              paper IEEE conference-style
src/simulation/simulation/
    core/                       fisica + controllori, NESSUNA dipendenza ROS2
        dynamics.py             modello 3-DOF (surge/sway/yaw), RK4
        controllers.py          PIDAxis, SMCAxis (per-asse, condivisi da PID/SMC)
        currents.py             profili di corrente (step / rampa)
        metrics.py              le 5 metriche di valutazione
        experiment.py           motore di simulazione headless + campagna
        storage.py              salvataggio/caricamento run su disco
    dynamics_node.py            nodo ROS2: integra la dinamica, pubblica /bluerov2/odom
    PID.py / SMC.py             nodi ROS2 di controllo (wrapper rclpy sul core)
    current_node.py             nodo ROS2 che pubblica il disturbo di corrente
notebooks/experiments.ipynb    notebook per lanciare esperimenti singoli e la campagna
results/                        output della campagna (una cartella per run + summary.csv)
figures/                        grafici esportati per il paper
```

La fisica e i controllori vivono in `core/` e non dipendono da ROS2: i nodi ROS2 sono
wrapper sottili sullo stesso codice (un'unica fonte di verità), mentre il notebook
importa `core/` direttamente per lanciare centinaia di run in pochi secondi senza dover
avviare nodi/topic.

## Cosa ho fatto

### Bug di segno trovati e corretti
1. **τ_d duplicato** (capitolo 2 del paper, eq. 11–13): la corrente entra nel modello
   già tramite la velocità relativa ν_r dentro il termine di damping; il τ_d generico
   ereditato dall'equazione 6-DOF era un doppio conteggio. Rimosso dalla LaTeX,
   aggiunta una nota che spiega perché.
2. **Switching term dello SMC instabile**: con la convenzione e(t) = d_ref − d(t) già
   usata per il PID, la legge scritta nel paper (τ = τ_eq − k·sat(s/φ)) è instabile
   (verificato via Lyapunov: V̇ ≥ 0 con quel segno). In simulazione lo SMC divergeva
   (da 2.5 m a 19.6 m invece di convergere a 1.1 m). Corretto in
   `core/controllers.py` in **τ = τ_eq + k·sat(s/φ)**; la LaTeX del paper va allineata
   quando scriviamo la sezione Control Methods definitiva.

### Estensione a 3-DOF
Su richiesta, i controllori agiscono su tutti e tre gli assi (non solo surge): un loop
SISO indipendente per surge (distanza dalla rete, riferimento d_ref), sway (station
keeping laterale, riferimento 0) e yaw (prua perpendicolare alla rete, riferimento 0),
speculare alla struttura disaccoppiata già assunta nel capitolo 2. Nota fisica
importante: con questo modello, sway e yaw **non hanno alcun effetto sulla distanza
surge** (nessun accoppiamento nel modello disaccoppiato), quindi i risultati sulla
metrica principale sono identici a quelli di un controllo surge-only — l'estensione a
3-DOF serve per completezza/realismo del task, non cambia la conclusione sulla
robustezza.

### Calibrazione in acqua calma (baseline, corrente = 0)
| Controllore | % tempo in finestra | RMSE | Sforzo di controllo |
|---|---|---|---|
| PID | 98.4% | 0.139 m | 381 |
| SMC | 98.3% | 0.135 m | 368 |

Guadagni finali: PID surge `Kp=18, Ki=2.5, Kd=14`; SMC surge `λ=1.0, k=18, φ=0.08`
(vedi `core/experiment.py::DEFAULT_GAINS`). Il guadagno di switching k dello SMC è
stato ritarato da 12 a 18: con k=12 lo SMC rompeva molto prima del PID (~0.8 m/s contro
~1.0 m/s) semplicemente perché aveva meno "spinta" disponibile a parità di limite
attuatore — non è una proprietà intrinseca dello SMC, ma un parametro di progetto che
va scelto in base alla capacità dell'attuatore.

### Campagna di robustezza
14 ampiezze di corrente (0.0–1.3 m/s, step 0.1) × 2 controllori = 28 run da 120 s,
corrente a gradino attivata a t=10s e diretta lungo l'asse che allontana il ROV dalla
rete (il caso peggiore per il mantenimento della distanza). Ogni run salvato in
`results/<controller>_step_amp<X.XX>/` (parametri, metriche, serie temporale
completa); riassunto in `results/summary.csv`.

**Risultato:**
- **SMC**: prestazioni pressoché piatte e ideali (~98% tempo in finestra, RMSE
  costante, zero overshoot) fino a **0.8 m/s**, poi collasso netto tra 0.8 e 0.9 m/s
  (il tempo in finestra crolla dal 98% al 47% e poi all'11%).
- **PID**: degrado già graduale a partire da **0.7 m/s** (95.7% → 93.1% → 90.8%, RMSE
  in crescita costante da 0.17 a 0.37 m), collasso completo a **1.0 m/s**.

Non è un semplice "SMC batte PID": il PID sopravvive nominalmente un po' più a lungo
sfruttando il windup del termine integrale contro il limite dell'attuatore, ma con un
errore già enorme prima del collasso vero (RMSE 0.37 m a 0.9 m/s, ben fuori dalla
tolleranza pratica). Lo SMC mantiene invece una qualità costante fino al proprio
limite, poi crolla di colpo. Entrambi i collassi sono bruschi, non graduali: è un
fenomeno fisico reale (una volta che la forza richiesta per contrastare la corrente
supera quella disponibile — dal limite dell'attuatore per il PID, dal guadagno di
switching k per lo SMC — il ROV va alla deriva senza possibilità di recupero), non un
artefatto numerico.

**Nota aperta**: la corrente testata spinge il ROV *lontano* dalla rete, quindi la
metrica di distanza minima (rischio collisione) resta identica a ogni ampiezza — riflette
solo il transitorio di avvicinamento iniziale, non il disturbo. Per testare il rischio
di collisione vero servirebbe una seconda campagna con corrente diretta *verso* la
rete.

### Figure prodotte (in `figures/`)
- `pid_vs_smc_single_current.png` — confronto diretto a parità di corrente (0.4 m/s)
- `metrics_vs_current.png` — le 5 metriche vs ampiezza di corrente, PID vs SMC
- `breakdown_summary.png` — grafico riassuntivo con le soglie di rottura annotate

## Come rilanciare

**Notebook (campagna headless, no ROS2):**
```bash
cd notebooks && jupyter lab experiments.ipynb
```

**Demo ROS2 live** (richiede build colcon + `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` in
alcuni ambienti sandbox, dove cyclonedds fallisce per assenza dell'interfaccia di
rete di default):
```bash
colcon build --packages-select simulation
source install/setup.bash
RMW_IMPLEMENTATION=rmw_fastrtps_cpp ros2 run simulation dynamics_node
RMW_IMPLEMENTATION=rmw_fastrtps_cpp ros2 run simulation pid_node   # o smc_node
RMW_IMPLEMENTATION=rmw_fastrtps_cpp ros2 run simulation current_node --ros-args -p amplitude:=0.4
```

## Cosa manca ancora
- Sezioni Experimental Setup / Results / Conclusion del paper (dati pronti in
  `results/summary.csv` e `figures/`)
- Placeholder "AGGIUNGI DESCRIZIONE CAPITOLI" nell'introduzione
- Allineare la LaTeX della sezione Control Methods al segno corretto dello switching
  term SMC (vedi sopra)
- Eventuale seconda campagna con corrente diretta verso la rete (rischio collisione)
- Controllo finale lunghezza (target 5–6 pagine IEEE conference)
