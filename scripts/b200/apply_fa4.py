"""Audited local backport enabling official FA4 decode in SGLang 0.5.6.

Decode gate removal follows upstream SGLang v0.5.8; retain all FA3 SM guards.
Only the B200 isolated venv is changed. Original files are saved once.
"""
import hashlib
import json
from pathlib import Path
import sysconfig
ROOT=Path('/volume/pt-test/users/zhaoye/OPSD-B200-runtime')
SITE=Path(sysconfig.get_paths()['purelib'])
assert str(SITE).startswith('/volume/pt-test/users/zhaoye/envs/opsd-py312-cu128/')
backup=ROOT/'migration/fa4-backport-originals';backup.mkdir(parents=True,exist_ok=True)
changes={
 'sglang/srt/server_args.py':[(
 '''        if self.attention_backend == "fa4" or self.decode_attention_backend == "fa4":
            raise ValueError(
                "FA4 backend is only supported for prefill. Please use `--prefill-attention-backend fa4` instead."
            )''',
 '''        if self.attention_backend == "fa4" or self.decode_attention_backend == "fa4":
            # OPSD FA4 backport: official kernels, validated paged BF16 decode.
            assert is_sm100_supported(), "OPSD FA4 profile requires SM100-class Blackwell"
            self.page_size = 128''')],
 'sglang/srt/layers/attention/flashattention_backend.py':[(
 '        assert self.fa_impl_ver in [3], "Only FA3 support decoding"',
 '        assert self.fa_impl_ver in [3, 4], "Only FA3/official FA4 support decoding"  # OPSD FA4 backport')],
 'sgl_kernel/flash_attn.py':[(
 '    from ._fa4_interface import flash_attn_varlen_func as flash_attn_varlen_func_v4',
 '    from opd.fa4_adapter import flash_attn_varlen_func as flash_attn_varlen_func_v4  # OPSD official FA4')],
 'sglang-0.5.6.dist-info/METADATA':[(
 'Requires-Dist: nvidia-cutlass-dsl==4.2.1',
 'Requires-Dist: nvidia-cutlass-dsl==4.7.1')],
}
report=[]
for name,replacements in changes.items():
 p=SITE/name;original=backup/name;original.parent.mkdir(parents=True,exist_ok=True)
 if not original.exists():original.write_bytes(p.read_bytes())
 old=original.read_text();new=old
 for a,b in replacements:
  assert new.count(a)==1,(name,a)
  new=new.replace(a,b)
 previous = new.replace('assert is_sm100_supported(), \"OPSD FA4 profile', 'assert torch.cuda.get_device_capability()[0] == 10, \"OPSD FA4 profile') if name == 'sglang/srt/server_args.py' else new
 assert p.read_text() in (old,new,previous),f'Unexpected third-party edits: {p}'
 p.write_text(new)
 report.append(dict(path=name,original_sha256=hashlib.sha256(old.encode()).hexdigest(),patched_sha256=hashlib.sha256(new.encode()).hexdigest()))
 print('FA4 backport:',name)
(ROOT/'migration/fa4-backport.json').write_text(json.dumps({'base_sglang':'0.5.6','patch_scope':'FA4 decode admission, official-kernel API bridge, Cutlass dependency','reference':'https://github.com/sgl-project/sglang/tree/v0.5.8','files':report},indent=2)+'\n')
