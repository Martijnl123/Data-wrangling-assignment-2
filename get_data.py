"""
Download the benchmark data sets into data/ and check their SHA-256 hashes.

The files are public copies of the ODDS benchmark versions (Rayana, 2016):
ten from the ADBench repository and Arrhythmia and Ionosphere from the PyOD
repository. Mulcross comes from OpenML (data set 40897) and has to be saved by
hand as data/mulcross.arff; this script only checks its hash.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ADBENCH = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"
PYOD = "https://raw.githubusercontent.com/yzhao062/pyod/master/notebooks/data/"

FILES = {
    "http.npz": (ADBENCH + "16_http.npz",
                 "9d98710091551587f4740cfc851a6b064fff6319a337c2dc77ee6da4bcc2e02c"),
    "forestcover.npz": (ADBENCH + "10_cover.npz",
                        "945a097c3542ae81a551bf1c06dcad78a755d94f84fcaf1d60687e42d1a656df"),
    "smtp.npz": (ADBENCH + "34_smtp.npz",
                 "d2569fc9bc12115419e3f5b37d19921e10e6c26e5ad1e61064cfb0fbc402dfdd"),
    "shuttle.npz": (ADBENCH + "32_shuttle.npz",
                    "26900b0f6be45d9ddb7a402c9b2f956ceda42d23a8ca4cf4406b8bc1352823d1"),
    "mammography.npz": (ADBENCH + "23_mammography.npz",
                        "a0308ac9712d78321073b1c2dc98dc37665c0f35c1cc33019416fe8f8d7de7ea"),
    "annthyroid.npz": (ADBENCH + "2_annthyroid.npz",
                       "e738f7c58a8740d2c4aac4c71ff49ce5e1d1aa16f44178135fc9c09df05fad1b"),
    "satellite.npz": (ADBENCH + "30_satellite.npz",
                      "1f048f5686f1b8c16a227fdfa3c81a62cd9a92715714a6b24c69b1c5ad96b46e"),
    "pima.npz": (ADBENCH + "29_Pima.npz",
                 "215454bb263a2f040945c9ea7bd7f06f1e89a61a3143bdb6f196e3887e801a7d"),
    "breastw.npz": (ADBENCH + "4_breastw.npz",
                    "4f37ce07dbc25ddad8178fdc6d664ab46d2d0000ee28c2787c3c7ff3c03ef9f6"),
    "arrhythmia.mat": (PYOD + "arrhythmia.mat",
                       "e4d47e499591ae31e5d7f0cd983367092353eac240f48e14c9b36457edb3460d"),
    "ionosphere.mat": (PYOD + "ionosphere.mat",
                       "8ac5dfa6f47db4acdfe10615489b7f7dd015d79d5b381ce641e72baca3cfefb0"),
    # Mulcross has no direct download link here: save the ARFF from
    # https://www.openml.org/d/40897 as data/mulcross.arff
    "mulcross.arff": (None,
                      "76ed253364a3bcbf26470059f9c1178e7b0a004421f34aebd15d177e1aba8c5c"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = Path(__file__).resolve().parent / "data"
    out.mkdir(exist_ok=True)
    ok = True
    for name, (url, expected) in FILES.items():
        path = out / name
        if not path.exists() and url is None:
            print(f"{name:18s} MISSING: download it from https://www.openml.org/d/40897")
            ok = False
            continue
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        digest = sha256(path)
        status = "ok" if digest == expected else "HASH MISMATCH"
        ok &= digest == expected
        print(f"{name:18s} {status}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
