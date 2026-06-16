"""
check_install.py — Vérifie que l'environnement est correctement installé.
Lancer depuis la racine du repo : python utils/check_install.py
"""
import sys

errors = []
warnings = []

def check(label, fn):
    try:
        result = fn()
        print(f"  ✅ {label}" + (f" ({result})" if result else ""))
    except Exception as e:
        errors.append(label)
        print(f"  ❌ {label} — {e}")

def warn(label, fn):
    try:
        result = fn()
        print(f"  ✅ {label}" + (f" ({result})" if result else ""))
    except Exception as e:
        warnings.append(label)
        print(f"  ⚠️  {label} — {e} (optionnel)")

print("=" * 50)
print("Vérification de l'environnement NRLE822")
print("=" * 50)

print("\n📦 Packages Python :")
check("numpy",       lambda: __import__("numpy").__version__)
check("pandas",      lambda: __import__("pandas").__version__)
check("matplotlib",  lambda: __import__("matplotlib").__version__)
check("torch",       lambda: __import__("torch").__version__)
check("gymnasium",   lambda: __import__("gymnasium").__version__)
check("PIL",         lambda: __import__("PIL").__version__)
warn("pygame",       lambda: __import__("pygame").__version__)

print("\n🏎️  Environnement CarRacing-v2 :")
check("Création env", lambda: (
    __import__("gymnasium").make("CarRacing-v2", render_mode=None).__class__.__name__
))
check("Reset + step", lambda: (
    lambda env: (
        env.step(env.action_space.sample()),
        env.close(),
        "OK"
    )[-1]
)(__import__("gymnasium").make("CarRacing-v2", render_mode=None)))

print("\n💾 Utilitaires du repo :")
sys.path.insert(0, ".")
check("utils.env_wrappers", lambda: __import__("utils.env_wrappers", fromlist=["make_env"]) and "OK")
check("utils.replay_buffer", lambda: __import__("utils.replay_buffer", fromlist=["ReplayBuffer"]) and "OK")
check("utils.metrics",      lambda: __import__("utils.metrics", fromlist=["rmse"]) and "OK")

print("\n🔥 GPU (optionnel) :")
warn("CUDA disponible", lambda: (
    "OUI — " + __import__("torch").cuda.get_device_name(0)
    if __import__("torch").cuda.is_available() else None
) or (_ for _ in ()).throw(Exception("Non disponible — CPU sera utilisé")))

print("\n" + "=" * 50)
if errors:
    print(f"❌ {len(errors)} erreur(s) critique(s) : {', '.join(errors)}")
    print("   → Consultez docs/PREREQUIS.md pour l'installation")
    sys.exit(1)
elif warnings:
    print(f"⚠️  {len(warnings)} avertissement(s) non bloquant(s)")
    print("✅ Environnement fonctionnel pour les notebooks 01–04")
else:
    print("✅ Tout est prêt ! Bonne formation 🏎️")
print("=" * 50)
