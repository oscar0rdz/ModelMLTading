# ModelMLTrading · Pipeline de clasificación, calibración y evaluación temporal (BTC/USDT · 15m)

Proyecto de **Machine Learning aplicado a mercados** orientado a portafolio. Muestra un pipeline completo para: (1) **estimar probabilidades** de que el *take‑profit* se alcance antes que el *stop‑loss* en un horizonte de **3 velas**, (2) **calibrar** esas probabilidades para que reflejen frecuencia observada y (3) **decidir** con un **umbral basado en valor esperado (EV)**, ilustrando su comportamiento mediante **Walk‑Forward Analysis (WFA)**.

> Uso principal: demostración educativa y de portafolio sobre buenas prácticas de ML en finanzas. No es asesoría financiera.

---

## 1) Qué problema resuelve y cómo

* **Tarea**: clasificación binaria «TP antes que SL» en 3 velas (BTC/USDT, 15m).
* **Modelo**: `XGBoostClassifier` con *tuning* (Optuna). La métrica objetivo de búsqueda es **PR‑AUC**, adecuada para clases desbalanceadas.
* **Calibración**: **Isotonic Regression** en *hold‑out* para que `p̂` sea interpretable (probabilidades confiables).
* **Decisión**: se escanean umbrales y se elige aquel cuyo **EV neto de costos** es positivo bajo una frecuencia mínima razonable.
* **Evaluación temporal**: **WFA** por ventanas deslizantes para observar estabilidad del criterio de decisión a lo largo del tiempo.

Razonamiento del diseño:

* PR‑AUC prioriza aciertos en la fracción minoritaria de eventos útiles para operar.
* La calibración hace que probabilidades y frecuencias observadas coincidan, requisito para convertir `p̂` en tamaños/umbrales.
* El EV integra **comisiones, spread y slippage**, alineando la decisión con el costo real de transacción.
* WFA evita conclusiones a partir de un único *split* y muestra sensibilidad de reglas a cambios de régimen.

---
## Cómo funciona y **por qué** se hace así

1. **Procesamiento & *feature engineering*** (`ML/data_processing.py`)  
   - **Qué:** OHLCV de **BTC/USDT** (15m) + indicadores técnicos (SMA/EMA, RSI, MACD, ATR, Bandas de Bollinger, ADX, OBV, etc.).  
   - **Por qué:** mejorar **señal‑ruido** y capturar **tendencia, momentum y volatilidad**; el modelo aprende patrones que por sí solos no están en el precio bruto.  
   - **Etiquetado dinámico (ATR, 3 velas):** TP/SL se escalan con volatilidad. **Por qué:** evita metas fijas desalineadas con el régimen de mercado.

2. **Entrenamiento & *tuning*** (`ML/model_training.py`)  
   - **Qué:** búsqueda con **Optuna** + validación temporal con **purge/embargo**.  
   - **Por qué:** evitar *leakage* por dependencia temporal y converger hacia **hiperparámetros estables** que maximizan **PR‑AUC**.

3. **Calibración** (isotónica, *hold‑out*)  
   - **Qué:** ajustar *scores* a **probabilidades fieles**.  
   - **Por qué:** si la probabilidad “70%” no corresponde a 70% observado, el control de riesgo y el **EV** se distorsionan.

4. **Selección de umbral por EV**  
   - **Qué:** elegir `thr` que **maximiza valor esperado** bajo **costos** y **R:R**.  
   - **Por qué:** pasar de clasificación a **decisión económica**; umbrales con alta *precision* pero **cobertura nula** no sirven para operar.

5. **Walk‑Forward Backtest** (`ML/backtest_improved.py`)  
   - **Qué:** ventanas *train→test* deslizantes con **comisiones**, **slippage**, **límite de operaciones** y **stop global**.  
   - **Por qué:** **realismo y robustez**; medir desempeño fuera de muestra y bajo fricción.

## 2) Resultados del modelo (11‑nov‑2025)

**Tuning (Optuna)**

* **Mejor PR‑AUC**: `0.6445`.
* Hiperparámetros destacados (ejemplo): `max_depth=5, learning_rate≈0.0056, n_estimators≈418, subsample≈0.747, colsample_bytree≈0.900`, regularizaciones y `min_child_weight=6`.
  (Ver artefactos en `ML/results/`.)

**Evaluación en test con probabilidades calibradas**

* **AUC**: `0.6349`
* **PR‑AUC**: `0.6070`
* **Brier score**: `0.2369`

Interpretación: el modelo **aprende patrones útiles** (por encima del azar) y sus probabilidades están **bien alineadas** con la frecuencia observada. En los deciles superiores de `p̂` aumentan tanto la tasa de aciertos como un *proxy* de EV, lo que respalda su uso para priorizar contextos con mejor expectativa.

**Regla de decisión ilustrativa**

* Umbral empleado a modo de ejemplo: `thr = 0.780`.
* Efecto: **reduce la cobertura** (menos señales) y **conserva alta precisión** en los casos que sí selecciona. Esto se utiliza para mostrar cómo cambia el equilibrio cobertura/precisión cuando el criterio exige mayor evidencia.

---

## 3) Qué se ve en los gráficos y GIFs

Las figuras y animaciones provienen de la ejecución del proyecto y están en `ML/assets/` (sin LFS):

* **Curva de equity (WFA)** → `ML/assets/equity_curve.png`
  Muestra la evolución del capital por ventanas. En el periodo ilustrado, la curva es **sensible a costos** y al **pacing** (frecuencia efectiva de operaciones), con *drawdowns* cercanos a 50%.

* **Curva ROC (test)** → `ML/assets/roc_curve_binary_test.png`
  Indica la capacidad discriminativa global del clasificador; complementa PR‑AUC cuando las clases están desbalanceadas.

* **Precisión/Recall vs Umbral (test)** → `ML/assets/threshold_precision_recall_test.png`
  Visualiza el intercambio cobertura/precisión al mover el umbral. Explica por qué umbrales altos concentran operaciones en contextos con mayor probabilidad estimada.

* **Demos en GIF/MP4**
* Entrenamiento *
* <img src="ML/assets/ModelGif.gif" alt="Pipeline (entrenamiento y evaluación)" width="720"><br>
* Backtestin*
* <img src="ML/assets/BackTest.gif" alt="Backtest WFA (flujo de señales)" width="720">



<p align="center">
  <img src="ML/assets/Graf.gif" alt="Métricas y gráficas" width="760">
</p>
e.

---

## 4) Qué muestra la simulación (WFA) y por qué es útil aquí

El **backtest** se usa como **instrumento de inspección**: permite ver cómo la regla se comporta bajo distintas ventanas, costos y densidad de señales. En la muestra mostrada, la combinación de umbral exigente, filtros y costos **reduce la frecuencia** y hace exigente sostener el rendimiento agregado.

Aun con esa cautela, el aprendizaje central es sólido: el modelo **clasifica por encima del azar**, sus probabilidades están **calibradas**, y los gráficos por bandas de `p̂` muestran una relación coherente entre probabilidad y aciertos. Este es el tipo de evidencia que **sí** traslada valor a una estrategia operativa una vez que se fijan costos, *risk‑reward* y ritmo de operación con parámetros consistentes.

La pregunta a la decisión operable

La meta inicial fue sencilla: **convertir probabilidades en decisiones que valgan la pena** en el marco de 15m. Tras construir *features* y fijar **PR‑AUC** como métrica, ejecuté **Optuna**. La búsqueda no fue lineal: *trials* 40–41 se movieron en `0.642–0.644`, el 42 fue *pruned* y el **trial 12** fijó el tope con `0.64437`. Esa región de hiperparámetros (profundidad 6, `min_child_weight≈10`, `colsample_bytree≈0.94`, `gamma≈1.7`) mostró **estabilidad**.  

## 5) Cómo reproducir

```bash
# Entorno (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Variables (costos, ritmo objetivo, etc.)
cp .env.example .env

# Entrenar + calibrar + exportar artefactos
python ML/model_training.py

# Simulación WFA (ilustrativa)
python ML/backtest_improved.py --out results_wfa
```

Artefactos principales en `ML/results/`:

* `btc15m_xgb_calibrated_trained_pipeline.joblib` (pipeline)
* `iso_cal.joblib` (calibrador)
* `metrics_test.json`, `training_report.md`, CSVs de fiabilidad/umbral
* Figuras en `ML/assets/` y resultados por ventanas en `results_wfa/`.

---

## 6) Estructura del repositorio

```
/ModelMLTrading
├── ML/
│   ├── data/
│   ├── logs/
│   ├── results/           # artefactos: *.joblib, métricas, CSV
│   ├── assets/            # GIFs y figuras del README (sin LFS)
│   ├── data_processing.py
│   ├── model_training.py
│   └── backtest_improved.py
├── results_wfa/
├── scripts/
│   └── run_pipeline.sh
├── requirements.txt
└── README.md
```

---

## 7) Licencia y aviso

Proyecto educativo y de portafolio. **No** constituye asesoramiento financiero ni recomendación de inversión. Las simulaciones son **ilustrativas** y están sujetas a supuestos de datos, costos y ejecución.

