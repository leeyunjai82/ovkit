"""Every composed capability ovkit ships, and what each one answers.

python examples/capabilities.py
"""

from __future__ import annotations

from ovkit import Model, list_pipelines


def main() -> None:
    print("Model(name) — a capability, not just a network:\n")
    for name, description in list_pipelines().items():
        print(f"  {name:16s} {description}")
    from ovkit.pipelines import ALIASES

    print("\nAliases:")
    for alias, target in sorted(ALIASES.items()):
        print(f"  {alias:16s} -> {target}")
    print("\nOr click through them all:  ovkit gui")
    print("\nExample:")
    print("  from ovkit import Model")
    print('  r = Model("face_analyze")("group.jpg")[0]')
    print("  print(r.summary())")
    print("\nDevices and options pass straight through:")
    print('  Model("face_analyze", device="GPU", attributes=("age_gender",))')
    assert Model  # imported for the snippets above


if __name__ == "__main__":
    main()
