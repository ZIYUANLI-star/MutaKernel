"""Build mk_stress.zip with the layout the orchestrator/worker expect."""
import os, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "外部Benchmark差分测试_RQ4", "MutaKernel-KGB",
                    "MutaKernel-KGB", "MutaKernel", "runs")
STRESS = os.path.join(RUNS, "kgb_ext_llmemd", "stress")
OUT = os.path.join(ROOT, "scripts", "mk_stress.zip")

z = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
# src
z.write(os.path.join(ROOT, "src", "stress", "policy_bank.py"), "src/stress/policy_bank.py")
z.writestr("src/stress/__init__.py", "")
z.writestr("src/__init__.py", "")
# scripts
z.write(os.path.join(ROOT, "scripts", "kgb_stress_worker.py"), "scripts/kgb_stress_worker.py")
z.write(os.path.join(ROOT, "scripts", "kgb_stress_orchestrator.py"), "scripts/kgb_stress_orchestrator.py")
# stress assets
z.write(os.path.join(STRESS, "stress_work.json"), "stress/stress_work.json")
refdir = os.path.join(STRESS, "refmods")
n = 0
for fn in sorted(os.listdir(refdir)):
    if fn.endswith(".py"):
        z.write(os.path.join(refdir, fn), f"stress/refmods/{fn}")
        n += 1
z.close()
print(f"wrote {OUT} | refmods={n} size={os.path.getsize(OUT)//1024}KB")
