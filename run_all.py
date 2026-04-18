# run_all.py
# ============================================================
# Point d'entrée unique — lance un ou plusieurs modèles
# Usage : python run_all.py --models cnn1d gru gcn
#         python run_all.py --models all
# ============================================================

import argparse
import importlib
import sys
import time

AVAILABLE = ["cnn1d", "cnn2d", "gru", "lstm", "gcn", "arima", "cnn_arima", "gcn_lstm"]          # ← 2 nouveaux modèles

def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Prediction — PEMS07"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn1d"],
        help=f"Modèles à lancer : {AVAILABLE} ou 'all'"
    )
    return parser.parse_args()


def run_model(name: str):
    print("\n" + "=" * 60)
    print(f"  MODÈLE : {name.upper()}")
    print("=" * 60)
    t0 = time.time()
    try:
        mod = importlib.import_module(f"mains.main_{name}")
        mod.run()
        elapsed = time.time() - t0
        print(f"\n✅ {name.upper()} terminé en {elapsed:.1f}s")
    except Exception as e:
        print(f"\n❌ Erreur sur {name} : {e}")
        raise


def main():
    args = parse_args()
    models = AVAILABLE if "all" in args.models else args.models

    # Vérification des noms
    for m in models:
        if m not in AVAILABLE:
            print(f"❌ Modèle inconnu : '{m}'. Disponibles : {AVAILABLE}")
            sys.exit(1)

    print(f"\nModèles sélectionnés : {models}")
    total_start = time.time()

    for m in models:
        run_model(m)

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"✅ Pipeline complet — {len(models)} modèle(s) en {total:.1f}s")
    print(f"   Résultats dans results/")
    print("=" * 60)


if __name__ == "__main__":
    main()