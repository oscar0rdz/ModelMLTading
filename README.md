# End‑to‑End ML Trading Pipeline (BTC/USDT · 15m)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB5F15?logo=xgboost&logoColor=white)](#)
[![Optuna](https://img.shields.io/badge/Tuning-Optuna-573E8C?logo=optuna&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Use-Portfolio-blue)](#)

> **Objetivo:** mostrar, de forma clara y reproducible, cómo construir **un pipeline de ML de punta a punta** (features → entrenamiento con tuning → calibración → selección de umbral por EV → *walk‑forward backtest*).  
> Este repositorio **no promete rentabilidad**; es un **proyecto de portafolio** de ingeniería de ML aplicado a mercados.

---

## Tabla de contenidos
- [Resumen ejecutivo](#resumen-ejecutivo)
- [Demo visual](#demo-visual)
- [Cómo funciona (pipeline ML)](#cómo-funciona-pipeline-ml)
- [Resultados del modelo](#resultados-del-modelo)
- [Backtest Walk‑Forward (WFA)](#backtest-walk-forward-wfa)
- [Limitaciones y decisiones de diseño](#limitaciones-y-decisiones-de-diseño)
- [Próximos pasos](#próximos-pasos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Configuración (.env)](#configuración-env)
- [Reproducibilidad](#reproducibilidad)
- [Cómo subir a GitHub](#cómo-subir-a-github)
- [Descargo de responsabilidad](#descargo-de-responsabilidad)

---

## Resumen ejecutivo

- **Algoritmo:** XGBoost (clasificación binaria).  
- **Objetivo del modelo:** probabilidad de que una operación alcance el *take‑profit* antes del *stop* en un horizonte de **3 velas**.  
- **Métrica del *tuning*:** **PR‑AUC** (apropiada con desbalance).  
- **Calibración:** **Isotonic Regression** en *hold‑out*.  
- **Selección de umbral:** por **Expected Value (EV)** con costos y R:R explícita.  
- **Backtest:** análisis **Walk‑Forward** con costos, *position sizing* fraccional y **stop global**.

**Mejor PR‑AUC (tuning)**: `0.6455`  
**Hiperparámetros (Optuna):**
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
**Entrenamiento final:** `scale_pos_weight=1.05` · Probabilidades calibradas con isotónica.

---

## Demo visual

> Coloca tus GIFs en `ML/assets/` y **asegúrate de que tengan extensión `.gif`**.  
> Si tus nombres son distintos, renómbralos o cambia el `src` abajo.

<p align="center">
  <img src="ML/assets/Model.gif" alt="Pipeline del modelo" width="760">
</p>

<p align="center">
  <img src="ML/assets/BTesting.gif" alt="Backtest Walk-Forward" width="760">
</p>

<p align="center">
  <img src="ML/assets/Graf.gif" alt="Métricas y gráficas" width="760">
</p>

---

## Cómo funciona (pipeline ML)

1. **Procesamiento de datos & *feature engineering*** (`ML/data_processing.py`)  
   - OHLCV de **BTC/USDT** (marco **15m**).  
   - Indicadores: SMA/EMA, RSI, MACD, ATR, Bandas de Bollinger, ADX, volumen/OBV, etc.  
   - **Etiquetado dinámico** por volatilidad (ATR): horizonte **3 velas**, con TP/SL derivados de volatilidad.

2. **Entrenamiento y optimización** (`ML/model_training.py`)  
   - Tuning con **Optuna** usando validación temporal purgada/embargada.  
   - **Calibración isotónica** en *hold‑out*.  
   - **Umbral por EV**: no se usa 0.5 fijo; se maximiza EV con **costos** + **R:R**.

3. **Backtest Walk‑Forward** (`ML/backtest_improved.py`)  
   - Ventanas móviles (train → test) para reducir *look‑ahead bias*.  
   - Simulación con **comisiones**, **slippage** y **límite de operaciones** por bloque.  
   - *Position sizing* fraccional y **stop global** de pérdida.

---

## Resultados del modelo

### Métricas en *test* (probabilidades calibradas; umbral aplicado = **0.780**)
> Con este umbral alto, **no se generaron señales** en test (Prec/Recall/F1=0). Es útil: la combinación calibración + umbral conservador redujo cobertura. Abajo se sugieren ajustes.

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

**Figuras:**  
![Precision/Recall vs Threshold](ML/assets/threshold_precision_recall_test.png)  
![Curva ROC](ML/assets/roc_curve_binary_test.png)  
![Distribución de probabilidades](ML/assets/probability_hist_test.png)

### Umbral y candidatos
Se evaluaron candidatos; varios fueron descartados por **n < min_signals** en *hold‑out*. El **umbral final = 0.780**, que en *test* produjo 0 señales.

### EV por decil de probabilidad (*proxy*)
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

---

## Backtest Walk‑Forward (WFA)

> El **WFA ilustra** el flujo de evaluación y *stress* del modelo. En la corrida de ejemplo se usan reglas conservadoras (costos, límites, *stop* global), por lo que **no busca optimizar PnL** sino demostrar **robustez del andamiaje**.

**Extracto de logs:**
```
[EQUITY] ... equity=$509.28 dd=49.07%
Guardia holdout: note=holdout_pass | señales=27 | win=88.89% | ev=18.00bps | thr=0.8346
... límite de operaciones por bloque alcanzado (60) ...
WFA ... trades=60 | capital=$502.16
[EQUITY] ... equity=$495.87 dd=50.41%
[WARNING] Capital cayó a 495.87 (49.6% del inicial). Stop global activado; simulación detenida.
```
**Resumen:** capital 1,000 → **495.87 USD** (stop del 50% activado). Es un **baseline** para iterar.

![Equity Curve](results_wfa/equity_curve.png)

---

## Limitaciones y decisiones de diseño

- **Cobertura vs. precisión:** el umbral 0.780 maximiza EV con restricciones, pero **reduce señales a 0** en *test*.  
- **Costos/fricción:** comisiones y *slippage* incluidos; pequeñas variaciones impactan EV.  
- **Etiquetado / horizonte:** 3 velas; otros horizontes cambian la señal.  
- **No HFT:** backtest con latencia cero; en vivo la fricción será mayor.  
- **Régimen de mercado:** BTC/USDT (15m) cambia de régimen; el WFA lo captura parcialmente.

---

## Próximos pasos

1. **Ajustar umbral con objetivo de cobertura:** fijar *target* de señales por ventana (p.ej., 10–30) o rango mínimo de *recall* (0.55–0.65).  
2. **Relajar `min_signals` en *hold‑out*** para evitar descartar *thresholds* con buen EV.  
3. **Selector EV con *guard‑rails***: penalizar umbrales que dejen `trades/día≈0` o imponer `thr_max` dinámico.  
4. **Re-tuning *cost-aware*** (función objetivo basada en EV/costos).  
5. **Features y *regime filters***: hora del día, tendencia, volatilidad de régimen, *microstructure*.  
6. **Revisión de etiquetado** (TP/SL dinámicos y *look‑ahead*).  
7. **Comparar calibraciones** (Platt vs. isotónica) y `scale_pos_weight`/focal loss.  
8. **Monitoreo de calibración** (curvas de confiabilidad) y *drift* entre *hold‑out* y *test*.

---

## Estructura del proyecto

```
/ModelMLTrading
├── ML/
│   ├── data/                 # Datos crudos/procesados
│   ├── logs/                 # Logs de entrenamiento/backtest
│   ├── results/              # Artefactos del modelo
│   ├── out/                  # Salidas intermedias del WFA
│   ├── assets/               # Figuras y GIFs para el README
│   ├── data_processing.py
│   ├── model_training.py
│   └── backtest_improved.py
├── results_wfa/              # Equity/trades/métricas del WFA
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
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
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
- Semillas controladas y validación temporal con **purge/embargo**.  
- Artefactos: `metrics_test.json`, `reliability_test.csv`, `classification_report_test.txt`, `feature_importance_permutation_test.csv`, `threshold_candidates.csv`, `*_trained_pipeline.joblib`, `iso_cal.joblib`.

> **Checklist de assets (para que el README se vea bien en GitHub):**
> - `ML/assets/Model.gif`, `ML/assets/BTesting.gif`, `ML/assets/Graf.gif`  
> - `ML/assets/threshold_precision_recall_test.png`, `ML/assets/roc_curve_binary_test.png`, `ML/assets/probability_hist_test.png`  
> - `results_wfa/equity_curve.png`

---

## Cómo subir a GitHub

```bash
# 1) Inicializa el repo (si aún no existe)
git init
git add .
git commit -m "Proyecto: pipeline ML trading (portafolio)"

# 2) Crea la rama principal
git branch -M main

# 3) Conecta tu repositorio remoto (reemplaza <usuario> y <repo>)
git remote add origin https://github.com/<usuario>/<repo>.git

# 4) Sube todo
git push -u origin main
```

> **Consejos de presentación del portafolio**
> - Coloca los GIFs al inicio (sección *Demo visual*) y mantén un tono claro/honesto.  
> - Añade una sección “Qué aprendí / Decisiones de ingeniería”.  
> - Si usas licencia, crea `LICENSE` (MIT es común para portafolio) y referencia en el README.

---

## Descargo de responsabilidad

Proyecto **formativo**. **No** constituye consejo financiero. Usa este repositorio para **aprender** y **evaluar** un pipeline realista de ML aplicado a trading, manteniendo un enfoque crítico sobre las limitaciones del modelo y del backtest.
