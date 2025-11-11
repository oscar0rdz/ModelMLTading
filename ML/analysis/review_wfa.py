import os, json, math
import numpy as np
import pandas as pd

WFA_JSON = os.getenv("WFA_JSON", "ML/results_wfa/wfa_summary.json")


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1.0).fillna(0.0)
    return float(dd.min())  # negativo


def _concat_equities(wfa_results):
    frames = []
    for r in wfa_results:
        meta = r["meta"]
        # equity no se guardó; usamos resumen por ventana y reconstruimos por aproximación
        # Si guardas curvas en tu backtest, léelas aquí en vez de aproximar.
        final = float(r["summary"]["final"])
        frames.append(
            pd.DataFrame(
                {
                    "window": [f"{meta['test_from']}→{meta['test_to']}"],
                    "thr": [r["thr"]],
                    "n": [r["thr_stats"]["n"]],
                    "ev_mean": [r["thr_stats"]["ev_mean"]],
                    "ev_total": [r["thr_stats"]["ev_total"]],
                    "trades_per_day": [r["thr_stats"]["trades_per_day"]],
                    "final": [final],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main():
    if not os.path.exists(WFA_JSON):
        raise FileNotFoundError(f"No existe {WFA_JSON}")
    with open(WFA_JSON, "r") as f:
        data = json.load(f)
    results = data["results"]
    if not results:
        print("Sin resultados en WFA.")
        return

    dfw = _concat_equities(results)
    dfw["win_win"] = dfw["ev_mean"] > 0.0
    dfw["ok_tpd"] = (dfw["trades_per_day"] > 0.5) & (dfw["trades_per_day"] < 1.5)

    # Resumen
    print("\n=== RESUMEN POR VENTANA ===")
    print(dfw.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Indicadores “portfolio-grade”
    pct_win_windows = 100.0 * dfw["win_win"].mean()
    pct_ok_tpd = 100.0 * dfw["ok_tpd"].mean()
    n_ventanas = len(dfw)

    # Equity agregado (aprox con finales de cada ventana)
    # Si guardaste curva, reemplaza por concatenación real; aquí mostramos proxy de final.
    eq_total = dfw["final"]
    eq_total.index = range(len(eq_total))
    max_dd = _max_drawdown(eq_total)

    print("\n=== INDICADORES CLAVE ===")
    print(f"Ventanas ‘EV_mean>0’: {pct_win_windows:.1f}% de {n_ventanas}")
    print(f"Ventanas con trades/día≈1: {pct_ok_tpd:.1f}% de {n_ventanas}")
    print(f"Final capital (proxy últimas ventanas): {eq_total.iloc[-1]:.2f}")
    print(f"Max Drawdown (proxy): {max_dd:.2%}")

    # Reglas de decisión (heurísticas)
    flags = []
    if pct_win_windows < 60:
        flags.append("Menos del 60% de ventanas con EV_mean>0")
    if pct_ok_tpd < 70:
        flags.append("Ritmo de trades/día no consistente con objetivo")
    if max_dd < -0.20:
        flags.append("MaxDD peor que -20%")
    print("\n=== DICTAMEN ===")
    if flags:
        print("⚠️ Aún no es ‘portfolio-ready’ por:\n - " + "\n - ".join(flags))
    else:
        print("✅ Señales de ‘portfolio-ready’: estabilidad y riesgo controlado.")
    print("\nSugerencias: si falló alguno, aplica stress de costes, ajusta umbral para ritmo y revisa filtros ADX/range_c.")


if __name__ == "__main__":
    main()
