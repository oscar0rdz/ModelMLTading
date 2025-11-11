#!/usr/bin/env python3
import re, argparse, csv, sys
from datetime import datetime

EQUITY_RE = re.compile(
    r"""\[EQUITY\]\s+
        (?P<start>[\d\-:T\+\s]{19,32})\s+→\s+(?P<end>[\d\-:T\+\s]{19,32}).*?
        equity=\$(?P<equity>[\d\.]+)\s+dd=(?P<dd>[\d\.]+)%""",
    re.VERBOSE
)

def parse_log(path):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "[EQUITY]" not in line:
                continue
            m = EQUITY_RE.search(line)
            if m:
                rows.append({
                    "timestamp": m.group("start").strip(),
                    "equity_usd": float(m.group("equity")),
                    "dd_pct": float(m.group("dd"))
                })
            else:
                # Fallback si faltan dd% o flecha
                m2 = re.search(r"equity=\$(?P<equity>[\d\.]+)", line)
                s  = re.search(r"\[EQUITY\]\s+([0-9T:\-\+\s]{19,32})", line)
                if m2 and s:
                    rows.append({
                        "timestamp": s.group(1).strip(),
                        "equity_usd": float(m2.group("equity")),
                        "dd_pct": ""
                    })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Ruta al log con líneas [EQUITY]")
    ap.add_argument("--out", required=True, help="CSV de salida, ej. reports/equity.csv")
    args = ap.parse_args()

    rows = parse_log(args.log)
    if not rows:
        print("No se encontraron líneas [EQUITY].", file=sys.stderr)
        sys.exit(1)

    # Ordena por timestamp si posible
    for r in rows:
        try:
            r["_dt"] = datetime.fromisoformat(r["timestamp"].replace("Z",""))
        except Exception:
            r["_dt"] = None
    rows.sort(key=lambda x: x["_dt"] or datetime.min)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "equity_usd", "dd_pct"])
        for r in rows:
            w.writerow([r["timestamp"], r["equity_usd"], r["dd_pct"]])
    print(f"OK: escrito {len(rows)} filas -> {args.out}")

if __name__ == "__main__":
    main()
