# End‑to‑End ML Trading Pipeline (BTC/USDT · 15m)

> **Propósito del repo:** demostrar, de forma clara y reproducible, cómo construir **un pipeline de ML de punta a punta** (features → entrenamiento con tuning → calibración → selección de umbral por EV → *walk‑forward backtest*). Este repositorio **no es** una promesa de rentabilidad; es un **proyecto de portafolio** orientado a ingeniería de datos/ML aplicado a mercados.

---

## Tabla de contenidos
- [Resumen ejecutivo](#resumen-ejecutivo)
- [Cómo funciona (pipeline ML)](#cómo-funciona-pipeline-ml)
- [Resultados del modelo](#resultados-del-modelo)
- [Backtest Walk‑Forward (WFA)](#backtest-walkforward-wfa)
- [Limitaciones y decisiones de diseño](#limitaciones-y-decisiones-de-diseño)
- [Próximos pasos](#próximos-pasos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Configuración (.env)](#configuración-env)
- [Reproducibilidad](#reproducibilidad)
- [Descargo de responsabilidad](#descargo-de-responsabilidad)

---

## Resumen ejecutivo

- **Algoritmo principal:** XGBoost (clasificación binaria).  
- **Objetivo:** predecir probabilidad de que una operación alcance el *take‑profit* antes del *stop* en un horizonte de **3 velas**.  
- **Métrica objetivo en tuning:** **PR‑AUC** (adecuada cuando hay desbalance).  
- **Calibración:** **Isotonic Regression** en *hold‑out* para convertir *scores* en probabilidades bien calibradas.  
- **Selección de umbral:** por **Expected Value (EV)** con costos de transacción y relación R:R explícita.  
- **Backtest:** análisis **Walk‑Forward** con costos, *position sizing* fraccional y *stops* globales de riesgo.

**Mejor PR‑AUC (tuning)**: `0.6455`  
**Mejores hiperparámetros (Optuna):**
```json
{
  "max_depth": 8,
  "learning_rate": 0.005277168981965397,
  "n_estimators": 350,
  "subsample": 0.6724851882573539,
  "colsample_bytree": 0.9078057797512761,
  "gamma": 1.791399496320974,
  "reg_alpha": 2.1738777626652293,
  "reg_lambda": 2.436326663284468,
  "min_child_weight": 11,
  "max_delta_step": 0
}
```
**Notas de entrenamiento final:** `scale_pos_weight=1.05` · Probabilidades calibradas con isotónica.

---

## Cómo funciona (pipeline ML)

1. **Procesamiento de datos & *feature engineering*** (`ML/data_processing.py`)  
   - OHLCV de **BTC/USDT** en marco **15m**.  
   - Indicadores: medias móviles (SMA/EMA), RSI, MACD, ATR, Bandas de Bollinger, ADX, volumen/OBV, entre otros.  
   - **Etiquetado dinámico** por volatilidad (ATR): horizonte de **3 velas**, con TP/SL derivados de la volatilidad.

2. **Entrenamiento y optimización** (`ML/model_training.py`)  
   - Tuning con **Optuna**, validación temporal purgada/embargada.  
   - **Calibración isotónica** en *hold‑out*.  
   - **Selección de umbral por EV**: el umbral no es 0.5 fijo; se busca maximizar EV considerando **costos** y **R:R**.

3. **Backtest Walk‑Forward** (`ML/backtest_improved.py`)  
   - Ventanas móviles (train → test) para evitar *look‑ahead bias*.  
   - Simulación con **comisiones**, **slippage** y **límite de operaciones** por bloque.  
   - *Position sizing* fraccional y **stop global** de pérdida.

<p align="center">
  <img src="ML/assets/ModelGif" alt="Proceso del modelo" width="700">
</p>

<p align="center">
  <img src="ML/assets/BackTest" alt="Proceso del modelo" width="700">
</p>


<p align="center">
  <img src="ML/assets/Graf" alt="Proceso del modelo" width="700">
</p>



## Resultados del modelo

### Métricas en *test* (probabilidades calibradas; umbral aplicado = **0.780**)

> Con este umbral alto, **no se generaron señales** en test (Prec/Recall/F1=0). Esto es informativo: la calibración + umbral conservador redujeron la cobertura. Más abajo se proponen ajustes de umbral/cobertura para producción.

| Métrica | Valor |
|---|---|
| AUC | **0.6342** |
| PR‑AUC | **0.6050** |
| Brier score | **0.2373** |
| Precisión | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| Balanced Accuracy | 0.5000 |
| MCC | 0.0000 |
| PosRate (y=1) | 0.502 |

**Figuras de evaluación** (mueve/guarda las imágenes en `ML/assets` y referencia en el README):  
- *Precision/Recall vs Threshold*  
  `ML/assets/threshold_precision_recall_test.png`  
- *Curva ROC*  
  `ML/assets/roc_curve_binary_test.png`  
- *Distribución de probabilidades (por clase) con umbral 0.780*  
  `ML/assets/probability_hist_test.png`

Para vista previa local rápida (no producirá imágenes en GitHub hasta que subas los archivos a `ML/assets`):  
![Precision/Recall vs Threshold](ML/assets/threshold_precision_recall_test.png)  
![ROC Curve](ML/assets/roc_curve_binary_test.png)  
![Distribución de probabilidades](ML/assets/probability_hist_test.png)

### Umbral y candidatos
Se evaluaron candidatos (extracto). Muchos fueron descartados por **n<min_signals** en *hold‑out*, por lo que el selector propuso un **umbral final = 0.780** que, en *test*, produjo 0 señales.

### EV por decil de probabilidad (*proxy*)
Resumen (extracto de *hold‑out/test*):  
```
dec  n     p_mean   hit     ev_proxy
0    1102  0.3388   0.3339  -0.0268
1    1495  0.3941   0.3666   0.0684
2    1269  0.4626   0.4555   0.3277
3    1320  0.5423   0.5258   0.5327
4    1031  0.5617   0.5606   0.6344
5    1904  0.6697   0.6381   0.8605
6     504  0.7742   0.6885   1.0074
```
**Lectura crítica:** el EV *proxy* crece con la probabilidad, lo cual es coherente. Sin embargo, fijar **umbral=0.78** desplaza cobertura hacia cero en *test*. Para uso práctico conviene **bajar el umbral** hasta lograr un **trade‑off cobertura/EV** estable (p.ej., *target* de 0.6–0.65 si los costos lo permiten) o imponer un **mínimo de señales** por ventana.

---

## Backtest Walk‑Forward (WFA)

> El **WFA está concebido para *ilustrar* el flujo de evaluación y stress del modelo**. En la corrida incluida se usan reglas deliberadamente conservadoras (costos, límites, *stop* global), por lo que **los resultados no reflejan una estrategia óptima**, sino la **robustez del andamiaje**.

**Extracto de logs:**
```
[EQUITY] ... equity=$509.28 dd=49.07%
Guardia holdout: note=holdout_pass | señales=27 | win=88.89% | ev=18.00bps | thr=0.8346
... límite de operaciones por bloque alcanzado (60) ...
WFA ... trades=60 | capital=$502.16
[EQUITY] ... equity=$495.87 dd=50.41%
[WARNING] Capital cayó a 495.87 (49.6% del inicial). Stop global activado; simulación detenida.
```
**Resumen WFA (corrida de ejemplo):**
- **Capital inicial:** 1,000 USD  
- **Capital final:** ~**495.87 USD** (stop global del 50% activado)  
- **Operaciones por bloque:** límite = 60  
- **Conclusión:** el WFA evidencia que con el **umbral/coverage actual** y parámetros de riesgo/costos, el sistema **no es rentable**. Es un buen *baseline* para iterar.

Para reportes visuales, sube `results_wfa/equity_curve.png` y enlázalo aquí:  
![Equity Curve](results_wfa/equity_curve.png)

---

## Limitaciones y decisiones de diseño

- **Cobertura vs. precisión:** el umbral 0.780 maximiza EV teórico con restricciones, pero **reduce señales a 0** en *test*.  
- **Costos y fricción:** se incluyen comisión y *slippage*; pequeñas diferencias impactan el EV.  
- **Etiquetado y *horizon bias*:** horizonte de 3 velas; otros horizontes pueden modificar la señal.  
- **No es HFT:** el motor de backtest simula latencia nula/orden perfecto; en vivo habrá más fricción.  
- **Datos & *non‑stationarity*:** BTC/USDT (15m) cambia de régimen; el WFA intenta capturar esto.

---

## Próximos pasos

1. **Ajuste de umbral con objetivo de cobertura**: fijar *target* de señales por ventana (p.ej., 10–30) o un rango de *recall* mínimo, p.ej., 0.55–0.65.  
2. **Relajar `min_signals` en *hold‑out*** para evitar descartar *thresholds* con buen EV por falta de muestras.  
3. **Selector EV con *guard‑rails***: penalizar umbrales que dejen `trades/día≈0` o imponer `thr_max` dinámico.  
4. **Re‑tuning con costos en la función objetivo** (cost‑aware PR‑AUC o EV directo).  
5. **Features y *regime filters***: hora del día, tendencia, volatilidad de régimen, *microstructure*.  
6. **Revisión de etiquetado** (TP/SL dinámicos y *look‑ahead*).  
7. **Experimentar con *focal loss*/`scale_pos_weight` y calibración por platt/isotónica comparada.  
8. **Monitoreo de calibración** (*reliability curve*) y *drift* entre *hold‑out* y *test*.

---

## Estructura del proyecto

```
/ModelMLTrading
├── ML/
│   ├── data/                 # Datos crudos/procesados
│   ├── logs/                 # Logs de entrenamiento/backtest
│   ├── results/              # Artefactos del modelo (pipeline, umbral, métricas)
│   ├── out/                  # Salidas intermedias de WFA
│   ├── data_processing.py
│   ├── model_training.py
│   └── backtest_improved.py
├── results_wfa/              # Curvas de equity, trades, métricas del WFA
├── ML/assets/                # Coloca aquí las figuras del README
├── scripts/
│   └── run_pipeline.sh
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
git clone <tu-repo>
cd ModelMLTrading
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

### 1) Procesar datos
```bash
python -m ML.data_processing
```

### 2) Entrenar y calibrar
```bash
python -m ML.model_training
# Artefactos en ML/results/: *.joblib, metrics_test.json, threshold_candidates.csv, etc.
```

### 3) Backtest Walk‑Forward
```bash
python -m ML.backtest_improved --out results_wfa
```

---

## Configuración (.env)

Ejemplo mínimo:
```dotenv
# === Datos & Paths ===
SYMBOL="BTCUSDT"
INTERVAL="15m"
START="2017-03-01"
DATA_DIR="ML/data"
RESULTS_DIR="ML/results"

# === Entrenamiento ===
TEST_SIZE=0.3
N_TRIALS=100
K_VOL=1.8
LOOK_AHEAD=3

# === Backtesting (WFA) ===
INITIAL_CAPITAL=1000.0
RISK_PER_TRADE=0.008
COMMISSION_RATE=0.0002
SLIPPAGE_PCT=0.0001
TRAIN_BARS=9000
TEST_BARS=2500
STEP_BARS=500
```

---

## Reproducibilidad

- Versiones fijadas en `requirements.txt`.  
- Semillas controladas (cuando aplica) y validación temporal con **purge/embargo**.  
- Se guardan: `metrics_test.json`, `reliability_test.csv`, `classification_report_test.txt`, `feature_importance_permutation_test.csv`, `threshold_candidates.csv`, `*_trained_pipeline.joblib`, `iso_cal.joblib`.

> **Tip:** sube a `ML/assets/` las imágenes `threshold_precision_recall_test.png`, `roc_curve_binary_test.png` y `probability_hist_test.png` para que el README las muestre en GitHub.

---

## Descargo de responsabilidad

Este proyecto es **formativo**. **No** constituye consejo financiero. Los criptoactivos conllevan riesgo elevado. Usa este repositorio para **aprender, experimentar y evaluar** un pipeline realista de ML aplicado a trading, manteniendo siempre un enfoque crítico sobre las limitaciones del modelo y del backtest.

---

**Autor:** Proyecto de ingeniería de ML (portafolio).  
**Contacto:** abre un *Issue* o *Pull Request* con dudas/mejoras.
