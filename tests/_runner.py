"""测试文件共用的 runner: 零依赖, 直接 python tests/test_x.py."""

import sys


def run(namespace: dict) -> None:
    failed = 0
    for name, fn in sorted(namespace.items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{failed} failed")
    sys.exit(1 if failed else 0)
