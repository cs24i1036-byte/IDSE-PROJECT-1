# CarWise India — Phase 1

Recommended Python: 3.12

## Windows PowerShell setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Place the original CSV at `data/raw/car_details_v3.csv`.

Run:
```powershell
python -m src.check_environment
python -m src.data_audit
```

The audit script only reads the CSV; it does not modify it.
