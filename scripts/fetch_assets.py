"""家具（Poly Haven）とテクスチャ（ambientCG）の CC0 素材を取得し、MuJoCo で読める形にする
  python scripts/fetch_assets.py"""
from __future__ import annotations

import io
import json
import os
import struct
import sys
import urllib.request
import zipfile

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

MODELS = ["sofa_02", "modern_arm_chair_01", "modern_coffee_table_01", "side_table_01", "potted_plant_02",
          "modern_wooden_cabinet", "dining_chair_02"]
TEXTURES = {"floor": "Tiles078", "carpet": "Carpet012", "wall": "PaintedPlaster017",
            "wood_light": "Wood092", "wood_dark": "WoodFloor064"}
UA = {"User-Agent": "g1-gemini-agent asset fetcher (CC0 assets for a student project)"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _accessor(g, bins, i):
    a = g["accessors"][i]
    bv = g["bufferViews"][a["bufferView"]]
    dt, n = np.dtype(COMP[a["componentType"]]), NCOMP[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or dt.itemsize * n
    raw = bins[bv["buffer"]]
    if stride == dt.itemsize * n:
        out = np.frombuffer(raw, dt, a["count"] * n, start).reshape(a["count"], n)
    else:
        out = np.stack([np.frombuffer(raw, dt, n, start + k * stride) for k in range(a["count"])])
    if a.get("normalized"):
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out


def _node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    t = np.array(node.get("translation", [0, 0, 0]), dtype=float)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), dtype=float)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * s
    m[:3, 3] = t
    return m


def convert_gltf(gltf_path, out_dir, name):
    g = json.load(open(gltf_path, encoding="utf-8"))
    base = os.path.dirname(gltf_path)
    bins = [open(os.path.join(base, b["uri"]), "rb").read() for b in g["buffers"]]
    parts = {}

    def visit(ni, parent):
        node = g["nodes"][ni]
        m = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in g["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                pos = _accessor(g, bins, prim["attributes"]["POSITION"]).astype(float)
                pos = (m @ np.c_[pos, np.ones(len(pos))].T).T[:, :3]
                uv = _accessor(g, bins, prim["attributes"]["TEXCOORD_0"]).astype(float) \
                    if "TEXCOORD_0" in prim["attributes"] else np.zeros((len(pos), 2))
                idx = _accessor(g, bins, prim["indices"]).reshape(-1, 3).astype(np.int64) \
                    if "indices" in prim else np.arange(len(pos)).reshape(-1, 3)
                parts.setdefault(prim.get("material", -1), []).append((pos, uv, idx))
        for c in node.get("children", []):
            visit(c, m)

    for ni in g["scenes"][g.get("scene", 0)]["nodes"]:
        visit(ni, np.eye(4))

    # Y 上向き → Z 上向き
    allv = np.concatenate([p[0] for lst in parts.values() for p in lst])
    zup = lambda v: np.c_[v[:, 0], -v[:, 2], v[:, 1]]
    allv = zup(allv)
    lo, hi = allv.min(0), allv.max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    meta = {"name": name, "size": (hi - lo).round(4).tolist(), "parts": []}
    for k, (mat_i, lst) in enumerate(sorted(parts.items())):
        mat = g["materials"][mat_i] if mat_i >= 0 else {}
        pbr = mat.get("pbrMetallicRoughness", {})
        part = {"obj": f"{name}_{k}.obj", "rgba": pbr.get("baseColorFactor", [1, 1, 1, 1])}
        if "baseColorTexture" in pbr:
            img = g["images"][g["textures"][pbr["baseColorTexture"]["index"]]["source"]]
            im = Image.open(os.path.join(base, img["uri"])).convert("RGB")
            im.thumbnail((1024, 1024))
            im.save(os.path.join(out_dir, f"{name}_{k}.png"))
            part["png"] = f"{name}_{k}.png"
        with open(os.path.join(out_dir, part["obj"]), "w") as f:
            off = 0
            for pos, uv, idx in lst:
                v = zup(pos) + shift
                f.write("".join(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in v))
                f.write("".join(f"vt {u:.5f} {1 - w:.5f}\n" for u, w in uv))  # glTF は左上原点、OBJ は左下原点
                f.write("".join(f"f {a + off + 1}/{a + off + 1} {b + off + 1}/{b + off + 1} {c + off + 1}/{c + off + 1}\n"
                                for a, b, c in idx))
                off += len(v)
        meta["parts"].append(part)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


def fetch_model(name):
    out_dir = os.path.join(ROOT, "models", name)
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        return
    files = json.loads(get(f"https://api.polyhaven.com/files/{name}"))["gltf"]["1k"]["gltf"]
    src = os.path.join(ROOT, "_download", name)
    os.makedirs(src, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(src, f"{name}.gltf"), "wb").write(get(files["url"]))
    for rel, info in files.get("include", {}).items():
        if rel.endswith((".jpg", ".png")) and "_diff_" not in rel:
            continue
        p = os.path.join(src, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(get(info["url"]))
    meta = convert_gltf(os.path.join(src, f"{name}.gltf"), out_dir, name)
    print(f"  モデル {name}: 大きさ {meta['size']} m, 部品 {len(meta['parts'])}")


def fetch_texture(key, asset_id):
    out = os.path.join(ROOT, "textures", f"{key}.png")
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(get(f"https://ambientcg.com/get?file={asset_id}_1K-JPG.zip")))
    color = next(n for n in z.namelist() if n.endswith("_Color.jpg"))
    Image.open(z.open(color)).convert("RGB").save(out)
    print(f"  テクスチャ {key}: {asset_id}")


def main():
    os.makedirs(ROOT, exist_ok=True)
    print("Poly Haven（CC0）から家具のモデルを取得しています…")
    for m in MODELS:
        fetch_model(m)
    print("ambientCG（CC0）からテクスチャを取得しています…")
    for k, a in TEXTURES.items():
        fetch_texture(k, a)
    with open(os.path.join(ROOT, "CREDITS.md"), "w", encoding="utf-8") as f:
        f.write("# 素材のクレジット（すべて CC0）\n\n")
        f.write("".join(f"- {m}: Poly Haven https://polyhaven.com/a/{m}\n" for m in MODELS))
        f.write("".join(f"- {a}: ambientCG https://ambientcg.com/view?id={a}\n" for a in TEXTURES.values()))
    print(f"完了: {ROOT}")


if __name__ == "__main__":
    sys.exit(main())
