#!/usr/bin/env python3
"""Auditoria de escopo: verifica se apenas caminhos permitidos foram alterados."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_PREFIXES = (
  "src/webapp/",
  "docs/etapa5-site.md",
  "docs/img/etapa5/",
)
OPTIONAL_FILES = {
  "README.md",
  "presentation/README.md",
  ".gitignore",
}


def is_allowed(path: str) -> bool:
  normalized = path.replace("\\", "/")
  if normalized in OPTIONAL_FILES:
    return True
  return any(normalized.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def main() -> int:
  result = subprocess.run(
    ["git", "status", "--short"],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True,
    check=False,
  )

  if result.returncode != 0:
    print(result.stderr)
    return result.returncode

  lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
  violations = []

  for line in lines:
    parts = line.split()
    if len(parts) < 2:
      continue
    file_path = parts[-1]
    if not is_allowed(file_path):
      violations.append(file_path)

  if violations:
    print("Arquivos fora do escopo permitido:")
    for path in violations:
      print(f" - {path}")
    return 1

  print("Auditoria de escopo: nenhuma violação encontrada.")
  print(f"Arquivos rastreados/alterados: {len(lines)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
