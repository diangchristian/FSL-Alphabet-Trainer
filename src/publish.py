"""
Step 4: copy a release into fsl-pwa as the *candidate* alphabet model.

Writes server/model-assets/models/alphabet/<version>/ and models/alphabet/candidate.json
(read by the app's Recognition lab). It does not touch catalog.json or the database:
promote the candidate only after it wins the A/B on real devices (see README).
Usage: python src/publish.py [version]   (defaults to the newest release)
"""
import json
import shutil
import sys

import config


def main() -> None:
    releases = sorted(config.RELEASES.iterdir(), key=lambda p: p.stat().st_mtime)
    source = config.RELEASES / sys.argv[1] if len(sys.argv) > 1 else releases[-1]
    entry = json.loads((source / "catalog-entry.json").read_text())
    target = config.FSL_PWA / "server" / "model-assets" / "models" / "alphabet"
    if (target / entry["version"]).exists():
        # Release URLs are cached as immutable; changed content must get a new version (re-run export.py).
        sys.exit(f"release {entry['version']} is already published; not overwriting it")
    shutil.copytree(source, target / entry["version"], ignore=shutil.ignore_patterns("catalog-entry.json"))
    # Served release form: URLs as the API serves them (same rewrite registerModels.ts applies).
    candidate = {**entry}
    for field in ("modelUrl", "weightsUrl", "handUrl"):
        candidate[field] = "/api/models/files/" + entry[field].removeprefix("./")
    (target / "candidate.json").write_text(json.dumps(candidate, indent=2))
    fixture = config.FIXTURES / "normalize.json"
    shutil.copy(fixture, config.FSL_PWA / "client" / "tests" / "fixtures-normalize.json")
    print(f"published candidate {entry['version']} -> {target / entry['version']}")


if __name__ == "__main__":
    main()
