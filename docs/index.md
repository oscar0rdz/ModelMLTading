# ModelMLTrading — WFA report

# ModelMLTrading — Walk-Forward Analysis (BTCUSDT 15m)

**Resumen**  
Señales ML para BTC/USDT (15m). Pipeline: preproceso → entrenamiento → selección por **EV (bps)** y filtros de régimen → WFA → backtest con salida **dynamic** o **label**.

## Resultados visuales
### Equity curve
![Equity curve](./figs/equity_curve.png)

### Distribución de PnL por trade
![PnL histogram](./figs/pnl_hist.png)

### Razones de salida
![Razones](./figs/reasons.png)

## Métricas clave
<!--METRICS:START-->
Pendiente de calcular.
<!--METRICS:END-->

## Cómo reproducir
```bash
python ML/backtest_improved.py > backtest.log
./make_pages.sh backtest.log "ML/out/**/trades_block_*.csv"
```
