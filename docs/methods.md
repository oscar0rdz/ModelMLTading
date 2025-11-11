# Métodos

**Etiqueta (h=3)**: y=1 si (close_{t+3}-close_t)/close_t > 0, con MIN_SPACING≥4.  
**Modelo**: XGBoost + calibración isotónica.  
**Selección por EV**: cutoff en bps (incluye costos).  
**Filtros**: ATR_PCT_MIN, BBW_MIN, VOL_FILTER_FRAC_MIN.  
**Salidas**: dynamic (TP/SL/TSL/BE) o label (cierre exacto t+3).  
**WFA**: 6000 train / 1000 test / step 1000 (sin fuga).
