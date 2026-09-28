"""Repair only installation prefixes after copying the pinned Beijing venv."""
from pathlib import Path
old = '/volume/pt-train/users/zhaoye'
new = '/volume/pt-test/users/zhaoye'
venv = Path(new)/'envs/opsd-py312-cu128'
paths = list((venv/'bin').glob('*')) + list((venv/'lib/python3.12/site-packages').glob('*.pth')) + list((venv/'lib/python3.12/site-packages').glob('*.dist-info/direct_url.json')) + [venv/'pyvenv.cfg']
for p in paths:
    if p.is_symlink() or not p.is_file(): continue
    b = p.read_bytes()
    if b'\x00' in b[:4096]: continue
    if old.encode() in b:
        p.write_bytes(b.replace(old.encode(),new.encode()))
        print('relocated',p.name)
