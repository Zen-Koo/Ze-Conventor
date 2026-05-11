"""
Ze-Conventor — SSB2 Map Converter
Конвертирует .obj 3D-модели в карты Simple Sandbox 2 (.clm.msg.gz)

Зависимости: pip install flet
Сборка APK:  flet build apk
"""

import flet as ft
import gzip
import io
import math
import os
import struct
import threading
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# MSGPACK ENCODER (без внешних зависимостей)
# ─────────────────────────────────────────────────────────────────────────────

def _pack(obj) -> bytes:
    if obj is None:
        return b'\xc0'
    if isinstance(obj, bool):
        return b'\xc3' if obj else b'\xc2'
    if isinstance(obj, int):
        if 0 <= obj <= 0x7f:              return bytes([obj])
        if -32 <= obj < 0:               return bytes([obj + 256])
        if 0 <= obj <= 0xff:             return b'\xcc' + bytes([obj])
        if -0x80 <= obj <= 0x7f:         return b'\xd0' + struct.pack('b', obj)
        if 0 <= obj <= 0xffff:           return b'\xcd' + struct.pack('>H', obj)
        if -0x8000 <= obj <= 0x7fff:     return b'\xd1' + struct.pack('>h', obj)
        if 0 <= obj <= 0xffffffff:       return b'\xce' + struct.pack('>I', obj)
        if -0x80000000 <= obj <= 0x7fffffff: return b'\xd2' + struct.pack('>i', obj)
        return b'\xd3' + struct.pack('>q', obj)
    if isinstance(obj, float):
        return b'\xca' + struct.pack('>f', obj)
    if isinstance(obj, (bytes, bytearray)):
        n = len(obj)
        if n <= 0xff:   return b'\xc4' + bytes([n]) + bytes(obj)
        if n <= 0xffff: return b'\xc5' + struct.pack('>H', n) + bytes(obj)
        return b'\xc6' + struct.pack('>I', n) + bytes(obj)
    if isinstance(obj, str):
        e = obj.encode('utf-8'); n = len(e)
        if n <= 31:     return bytes([0xa0 | n]) + e
        if n <= 0xff:   return b'\xd9' + bytes([n]) + e
        if n <= 0xffff: return b'\xda' + struct.pack('>H', n) + e
        return b'\xdb' + struct.pack('>I', n) + e
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        h = bytes([0x90|n]) if n<=15 else (b'\xdc'+struct.pack('>H',n) if n<=0xffff else b'\xdd'+struct.pack('>I',n))
        return h + b''.join(_pack(v) for v in obj)
    if isinstance(obj, dict):
        n = len(obj)
        h = bytes([0x80|n]) if n<=15 else (b'\xde'+struct.pack('>H',n) if n<=0xffff else b'\xdf'+struct.pack('>I',n))
        return h + b''.join(_pack(k)+_pack(v) for k,v in obj.items())
    raise TypeError(f'Cannot pack {type(obj)}')


# ─────────────────────────────────────────────────────────────────────────────
# РОТАЦИИ (кватернионы)
# ─────────────────────────────────────────────────────────────────────────────

SQ2 = math.sqrt(2) / 2

ROTATIONS = {
    "Нет":     {"x": 0.0,  "y": 0.0,  "z": 0.0,  "w": 1.0},
    "X +90°":  {"x": SQ2,  "y": 0.0,  "z": 0.0,  "w": SQ2},
    "X 180°":  {"x": 1.0,  "y": 0.0,  "z": 0.0,  "w": 0.0},
    "X −90°":  {"x":-SQ2,  "y": 0.0,  "z": 0.0,  "w": SQ2},
    "Y +90°":  {"x": 0.0,  "y": SQ2,  "z": 0.0,  "w": SQ2},
    "Y 180°":  {"x": 0.0,  "y": 1.0,  "z": 0.0,  "w": 0.0},
    "Y −90°":  {"x": 0.0,  "y":-SQ2,  "z": 0.0,  "w": SQ2},
    "Z +90°":  {"x": 0.0,  "y": 0.0,  "z": SQ2,  "w": SQ2},
    "Z 180°":  {"x": 0.0,  "y": 0.0,  "z": 1.0,  "w": 0.0},
    "Z −90°":  {"x": 0.0,  "y": 0.0,  "z":-SQ2,  "w": SQ2},
}


# ─────────────────────────────────────────────────────────────────────────────
# ПАРСИНГ ФОРМАТОВ
# Поддерживаемые: .obj  .glb  .gltf  .stl  .ply  .fbx (ASCII)
# .blend — нативно не поддерживается (проприетарный бинарный формат Blender),
#          нужен экспорт в GLB прямо из Blender: File → Export → glTF 2.0
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = [".obj", ".glb", ".gltf", ".stl", ".ply", ".fbx"]


def parse_obj(text: str):
    """Wavefront .obj — вершины v + грани f с триангуляцией."""
    vertices, triangles = [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            try:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except (IndexError, ValueError):
                pass
        elif parts[0] == "f":
            idxs = []
            for tok in parts[1:]:
                try:
                    idxs.append(int(tok.split("/")[0]) - 1)
                except ValueError:
                    pass
            for i in range(1, len(idxs) - 1):
                triangles.append((idxs[0], idxs[i], idxs[i + 1]))
    return vertices, triangles


def parse_glb_gltf(data: bytes):
    """
    GLB / GLTF — через pygltflib.
    pip install pygltflib
    """
    try:
        import pygltflib
    except ImportError:
        raise RuntimeError(
            "Для GLB/GLTF нужна библиотека pygltflib.\n"
            "Установи: pip install pygltflib"
        )

    gltf = pygltflib.GLTF2.load_from_bytes(data)
    blob = gltf.binary_blob()

    def get_buf(buf_obj):
        if buf_obj.uri is None:
            return blob
        import base64
        uri = buf_obj.uri
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        raise RuntimeError("Внешние буферы URI не поддерживаются при загрузке файла.")

    all_verts, all_tris = [], []
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            if prim.attributes.POSITION is None:
                continue
            acc = gltf.accessors[prim.attributes.POSITION]
            bv  = gltf.bufferViews[acc.bufferView]
            raw = get_buf(gltf.buffers[bv.buffer])
            off = (bv.byteOffset or 0) + (acc.byteOffset or 0)
            stride = bv.byteStride if bv.byteStride else 12
            base = len(all_verts)
            for i in range(acc.count):
                o = off + i * stride
                all_verts.append(struct.unpack_from("<fff", raw, o))

            if prim.indices is not None:
                iacc = gltf.accessors[prim.indices]
                ibv  = gltf.bufferViews[iacc.bufferView]
                iraw = get_buf(gltf.buffers[ibv.buffer])
                ioff = (ibv.byteOffset or 0) + (iacc.byteOffset or 0)
                ct = iacc.componentType
                fmt, sz = ("B", 1) if ct == 5121 else (("H", 2) if ct == 5123 else ("I", 4))
                idx = [struct.unpack_from("<" + fmt, iraw, ioff + j * sz)[0]
                       for j in range(iacc.count)]
                for j in range(0, len(idx) - 2, 3):
                    all_tris.append((base + idx[j], base + idx[j+1], base + idx[j+2]))
            else:
                for j in range(0, acc.count - 2, 3):
                    all_tris.append((base+j, base+j+1, base+j+2))

    return all_verts, all_tris


def parse_stl(data: bytes):
    """
    STL бинарный и ASCII.
    Бинарный: 80 байт заголовок, uint32 count, затем 50-байтные треугольники.
    ASCII:     solid … facet normal … vertex … endsolid
    """
    vertices, triangles = [], []

    # Попытка ASCII (если начинается с 'solid')
    try:
        text = data.decode("utf-8", errors="replace")
        if text.lstrip().startswith("solid"):
            # ASCII STL
            import re
            pts = re.findall(
                r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", text
            )
            for i, (x, y, z) in enumerate(pts):
                vertices.append((float(x), float(y), float(z)))
            # Каждые 3 вершины — треугольник
            for i in range(0, len(vertices) - 2, 3):
                triangles.append((i, i + 1, i + 2))
            if vertices:
                return vertices, triangles
    except Exception:
        pass

    # Бинарный STL
    if len(data) < 84:
        raise ValueError("STL файл слишком короткий")
    num_tris = struct.unpack_from("<I", data, 80)[0]
    off = 84
    for _ in range(num_tris):
        if off + 50 > len(data):
            break
        # пропускаем нормаль (12 байт), читаем 3 вершины
        base = len(vertices)
        for vi in range(3):
            x, y, z = struct.unpack_from("<fff", data, off + 12 + vi * 12)
            vertices.append((x, y, z))
        triangles.append((base, base + 1, base + 2))
        off += 50

    return vertices, triangles


def parse_ply(data: bytes):
    """
    PLY (Stanford Polygon Format) — ASCII и бинарный (little-endian).
    """
    # Парсим заголовок (всегда ASCII)
    try:
        header_end = data.index(b"end_header")
    except ValueError:
        raise ValueError("Не найден end_header в PLY файле")

    header = data[:header_end].decode("ascii", errors="replace")
    body   = data[header_end + len("end_header"):]
    # Убираем один перевод строки после end_header
    if body[:1] in (b"\n", b"\r"):
        body = body[1:]
    if body[:1] == b"\n":
        body = body[1:]

    lines_h = header.splitlines()

    # Определяем формат
    fmt = "ascii"
    for l in lines_h:
        if l.strip().startswith("format"):
            if "binary_little_endian" in l:
                fmt = "binary_le"
            elif "binary_big_endian" in l:
                fmt = "binary_be"
            break

    # Читаем описание элементов (vertex, face)
    elements = []   # [(name, count, [(prop_type, prop_name), ...]), ...]
    cur_elem = None
    for l in lines_h:
        l = l.strip()
        if l.startswith("element"):
            parts = l.split()
            cur_elem = [parts[1], int(parts[2]), []]
            elements.append(cur_elem)
        elif l.startswith("property") and cur_elem:
            parts = l.split()
            if parts[1] == "list":
                cur_elem[2].append(("list", parts[2], parts[3], parts[4]))
            else:
                cur_elem[2].append((parts[1], parts[2]))

    PLY_SIZES = {"char":1,"uchar":1,"short":2,"ushort":2,
                 "int":4,"uint":4,"float":4,"double":8,
                 "int8":1,"uint8":1,"int16":2,"uint16":2,
                 "int32":4,"uint32":4,"float32":4,"float64":8}
    PLY_FMT = {"char":"b","uchar":"B","short":"h","ushort":"H",
               "int":"i","uint":"I","float":"f","double":"d",
               "int8":"b","uint8":"B","int16":"h","uint16":"H",
               "int32":"i","uint32":"I","float32":"f","float64":"d"}
    endian = "<" if fmt != "binary_be" else ">"

    vertices, triangles = [], []

    if fmt == "ascii":
        body_lines = body.decode("ascii", errors="replace").splitlines()
        line_idx = 0
        for (ename, ecount, eprops) in elements:
            for _ in range(ecount):
                if line_idx >= len(body_lines):
                    break
                vals = body_lines[line_idx].split()
                line_idx += 1
                if ename == "vertex":
                    # ищем x y z среди пропертей
                    xi = yi = zi = -1
                    vi = 0
                    for pi, prop in enumerate(eprops):
                        if prop[1] == "x": xi = vi
                        if prop[1] == "y": yi = vi
                        if prop[1] == "z": zi = vi
                        vi += 1
                    if xi >= 0 and yi >= 0 and zi >= 0:
                        try:
                            vertices.append((float(vals[xi]), float(vals[yi]), float(vals[zi])))
                        except (IndexError, ValueError):
                            pass
                elif ename == "face":
                    try:
                        n = int(vals[0])
                        idxs = [int(vals[k+1]) for k in range(n)]
                        for i in range(1, len(idxs)-1):
                            triangles.append((idxs[0], idxs[i], idxs[i+1]))
                    except (IndexError, ValueError):
                        pass
    else:
        # Бинарный PLY
        pos = 0
        for (ename, ecount, eprops) in elements:
            for _ in range(ecount):
                row = {}
                for prop in eprops:
                    if prop[0] == "list":
                        _, cnt_type, val_type, pname = prop
                        csz = PLY_SIZES[cnt_type]
                        cfmt = endian + PLY_FMT[cnt_type]
                        n = struct.unpack_from(cfmt, body, pos)[0]; pos += csz
                        vsz = PLY_SIZES[val_type]
                        vfmt = endian + PLY_FMT[val_type]
                        row[pname] = [struct.unpack_from(vfmt, body, pos + i*vsz)[0] for i in range(n)]
                        pos += n * vsz
                    else:
                        ptype, pname = prop
                        sz = PLY_SIZES[ptype]
                        pfmt = endian + PLY_FMT[ptype]
                        row[pname] = struct.unpack_from(pfmt, body, pos)[0]; pos += sz

                if ename == "vertex":
                    try:
                        vertices.append((row["x"], row["y"], row["z"]))
                    except KeyError:
                        pass
                elif ename == "face":
                    idxs = row.get("vertex_indices") or row.get("vertex_index") or []
                    for i in range(1, len(idxs)-1):
                        triangles.append((idxs[0], idxs[i], idxs[i+1]))

    return vertices, triangles


def parse_fbx_ascii(text: str):
    """
    FBX ASCII — извлекает координаты вершин из блоков Vertices: { }.
    Поддерживает только ASCII FBX (бинарный FBX требует специализированной библиотеки).
    """
    import re
    vertices, triangles = [], []

    # Ищем все массивы Vertices
    vert_blocks = re.findall(r"Vertices:\s*\{[^}]*a:\s*([\d,.\-eE\s]+)", text, re.DOTALL)
    if not vert_blocks:
        # Альтернативный формат: Vertices: *N { ... }
        vert_blocks = re.findall(r"Vertices:\s*\*\d+\s*\{[^}]*a:\s*([\d,.\-eE\s]+)", text, re.DOTALL)

    base = 0
    for block in vert_blocks:
        nums = [float(x) for x in re.split(r"[,\s]+", block.strip()) if x]
        for i in range(0, len(nums) - 2, 3):
            vertices.append((nums[i], nums[i+1], nums[i+2]))

        # Ищем индексы полигонов (PolygonVertexIndex)
        idx_blocks = re.findall(
            r"PolygonVertexIndex:\s*\*\d+\s*\{[^}]*a:\s*([\d,.\-\s]+)", text, re.DOTALL
        )
        for ib in idx_blocks:
            raw_idx = [int(x) for x in re.split(r"[,\s]+", ib.strip()) if x]
            # FBX: отрицательный индекс = последний в полигоне (xor -1)
            poly = []
            for ri in raw_idx:
                if ri < 0:
                    poly.append(ri ^ -1)
                    # Триангулируем полигон
                    for pi in range(1, len(poly)-1):
                        triangles.append((base+poly[0], base+poly[pi], base+poly[pi+1]))
                    poly = []
                else:
                    poly.append(ri)

        base += len(nums) // 3

    if not vertices:
        raise ValueError(
            "Вершины не найдены. Возможно, это бинарный FBX.\n"
            "Экспортируй модель как ASCII FBX или GLB из Blender/3ds Max."
        )
    return vertices, triangles


def parse_file(path: str, data: bytes):
    """
    Универсальный парсер. Определяет формат по расширению.
    Возвращает (vertices, triangles).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".obj":
        return parse_obj(data.decode("utf-8", errors="replace"))

    elif ext in (".glb", ".gltf"):
        return parse_glb_gltf(data)

    elif ext == ".stl":
        return parse_stl(data)

    elif ext == ".ply":
        return parse_ply(data)

    elif ext == ".fbx":
        try:
            text = data.decode("utf-8", errors="replace")
            if "; FBX" in text[:200] or "FBXHeaderExtension" in text[:500]:
                return parse_fbx_ascii(text)
            else:
                raise ValueError("Похоже на бинарный FBX")
        except UnicodeDecodeError:
            raise ValueError(
                "Бинарный FBX не поддерживается напрямую.\n"
                "Открой в Blender и экспортируй как GLB: File → Export → glTF 2.0"
            )

    elif ext == ".blend":
        raise ValueError(
            ".blend — проприетарный бинарный формат Blender.\n"
            "Открой файл в Blender и экспортируй:\n"
            "File → Export → glTF 2.0 (.glb/.gltf)\n"
            "Затем загрузи полученный .glb в Ze-Conventor."
        )

    else:
        raise ValueError(
            f"Формат «{ext}» не поддерживается.\n"
            f"Поддерживаемые: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ВОКСИЛИЗАЦИЯ (solid fill через рейкастинг по оси Y)
# ─────────────────────────────────────────────────────────────────────────────

def snap(c: float, step: float = 5.0) -> float:
    return round(c / step) * step


def voxelize(vertices, triangles, scale: float, step: float = 5.0,
             progress_cb=None) -> set:
    """
    Заполняет модель воксами. Возвращает set из (x, y, z).
    progress_cb(float 0..1) — обратный вызов прогресса.
    """
    if not triangles:
        # Нет граней — только вершины
        out = set()
        for x, y, z in vertices:
            out.add((snap(x*scale,step), snap(y*scale,step), snap(z*scale,step)))
        return out

    sv = [(x*scale, y*scale, z*scale) for x, y, z in vertices]

    # Группируем треугольники по XZ-колонкам
    col_tris = defaultdict(list)
    for ti, (i, j, k) in enumerate(triangles):
        try:
            ax,ay,az = sv[i]; bx,by,bz = sv[j]; cx,cy,cz = sv[k]
        except IndexError:
            continue
        xlo = math.floor(min(ax,bx,cx)/step)
        xhi = math.ceil (max(ax,bx,cx)/step)
        zlo = math.floor(min(az,bz,cz)/step)
        zhi = math.ceil (max(az,bz,cz)/step)
        for gx in range(xlo, xhi+1):
            for gz in range(zlo, zhi+1):
                col_tris[(gx,gz)].append(ti)

    def ray_y(rx, rz, ti):
        try:
            ax,ay,az = sv[triangles[ti][0]]
            bx,by,bz = sv[triangles[ti][1]]
            cx,cy,cz = sv[triangles[ti][2]]
        except IndexError:
            return None
        def cross(p1x,p1z, p2x,p2z, p3x,p3z):
            return (p1x-p3x)*(p2z-p3z)-(p2x-p3x)*(p1z-p3z)
        d1 = cross(rx,rz, ax,az, bx,bz)
        d2 = cross(rx,rz, bx,bz, cx,cz)
        d3 = cross(rx,rz, cx,cz, ax,az)
        if (d1<0 or d2<0 or d3<0) and (d1>0 or d2>0 or d3>0):
            return None
        denom = (bz-cz)*(ax-cx)+(cx-bz)*(az-cz)
        if abs(denom) < 1e-9:
            return (ay+by+cy)/3.0
        w1 = ((bz-cz)*(rx-cx)+(cx-bz)*(rz-cz))/denom
        w2 = ((cz-az)*(rx-cx)+(ax-cz)*(rz-cz))/denom
        w3 = 1.0-w1-w2
        if w1<-0.01 or w2<-0.01 or w3<-0.01:
            return None
        return w1*ay + w2*by + w3*cy

    voxels = set()
    cols = list(col_tris.items())
    total = len(cols)
    for idx, ((gx,gz), tlist) in enumerate(cols):
        if progress_cb and idx % 50 == 0:
            progress_cb(idx / max(total, 1))
        rx, rz = gx*step, gz*step
        hits = []
        for ti in tlist:
            yv = ray_y(rx, rz, ti)
            if yv is not None:
                hits.append(yv)
        if not hits:
            continue
        hits.sort()
        merged = [hits[0]]
        for yv in hits[1:]:
            if yv - merged[-1] > step*0.4:
                merged.append(yv)
        if len(merged) % 2 == 1:
            for yv in merged:
                voxels.add((rx, snap(yv,step), rz))
        else:
            for k in range(0, len(merged), 2):
                y0 = int(round(merged[k]  /step))
                y1 = int(round(merged[k+1]/step))
                for gy in range(y0, y1+1):
                    voxels.add((rx, gy*step, rz))

    if not voxels:
        for x,y,z in vertices:
            voxels.add((snap(x*scale,step), snap(y*scale,step), snap(z*scale,step)))
    return voxels


# ─────────────────────────────────────────────────────────────────────────────
# СБОРКА КАРТЫ → .clm.msg.gz
# ─────────────────────────────────────────────────────────────────────────────

def build_clm(vertices, triangles, prop_name: str, scale: float,
              rotation_key: str, fill_solid: bool,
              progress_cb=None) -> tuple:
    """Возвращает (bytes_gz, block_count)."""
    step = 5.0
    rot  = ROTATIONS.get(rotation_key, ROTATIONS["Нет"])

    if fill_solid:
        voxels = voxelize(vertices, triangles, scale, step, progress_cb)
    else:
        voxels = set()
        for x,y,z in vertices:
            voxels.add((snap(x*scale,step), snap(y*scale,step), snap(z*scale,step)))

    if not voxels:
        raise ValueError("Нет вокселей — проверьте модель")

    pts  = list(voxels)
    xs   = [p[0] for p in pts]; ys=[p[1] for p in pts]; zs=[p[2] for p in pts]
    cx   = snap((max(xs)+min(xs))/2.0, step)
    cz   = snap((max(zs)+min(zs))/2.0, step)
    ymin = snap(min(ys), step)

    props = []
    for (x,y,z) in pts:
        props.append({
            "itemName":        prop_name,
            "vectorPosition":  {"x": float(x-cx), "y": float(y-ymin), "z": float(z-cz)},
            "vectorRotation":  {"x": float(rot["x"]), "y": float(rot["y"]),
                                "z": float(rot["z"]), "w": float(rot["w"])},
            "isKinematics":    True,
            "mass":            1.0,
            "maxFloor":        -1,
            "customParameters":{},
        })

    payload = {"mapSize":0,"version":1,"mapIcon":b"","list":props,"saveVersion":0}
    raw = _pack(payload)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(raw)
    return buf.getvalue(), len(props)


# ─────────────────────────────────────────────────────────────────────────────
# FLET UI
# ─────────────────────────────────────────────────────────────────────────────

# Цвета темы
C_BG       = "#F0F4FF"          # очень светлый лавандово-синий фон
C_GLASS    = "#FFFFFFB0"        # белый с 70% прозрачностью (glass)
C_GLASS2   = "#FFFFFF88"
C_BORDER   = "#D6DFFE"
C_PRIMARY  = "#5B6EF5"          # главный акцент — индиго
C_PRIMARY2 = "#7C8FFF"
C_TEXT     = "#1C2051"          # тёмно-синий текст
C_MUTED    = "#8890BB"
C_SUCCESS  = "#3DDC84"          # зелёный успех
C_ERROR    = "#FF5C72"
C_WARN     = "#FFB347"
C_WHITE    = "#FFFFFF"


def gradient_bg():
    return ft.Container(
        expand=True,
        gradient=ft.LinearGradient(
            begin=ft.alignment.top_left,
            end=ft.alignment.bottom_right,
            colors=["#E8EEFF", "#F5F0FF", "#EEF5FF"],
        ),
    )


def glass_card(*controls, padding=20, radius=24, border_color=C_BORDER, **kwargs):
    return ft.Container(
        content=ft.Column(controls, spacing=0),
        padding=ft.padding.all(padding),
        border_radius=radius,
        bgcolor=C_GLASS,
        border=ft.border.all(1, border_color),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=24,
            color="#1A2060" + "18",
            offset=ft.Offset(0, 6),
        ),
        **kwargs,
    )


def section_label(text: str):
    return ft.Text(text, size=11, weight=ft.FontWeight.W_600,
                   color=C_MUTED, letter_spacing=1.2)


def main(page: ft.Page):
    page.title        = "Ze-Conventor"
    page.bgcolor      = C_BG
    page.padding      = 0
    page.window_width = 400
    page.fonts        = {
        "Nunito": "https://fonts.gstatic.com/s/nunito/v26/XRXI3I6Li01BKofiOc5wtlZ2di8HDOUheKQ.woff2",
    }
    page.theme = ft.Theme(font_family="Nunito")

    # ── состояние ──
    state = {
        "obj_path":  None,
        "file_data": None,   # bytes — содержимое файла
        "busy":      False,
        "rot_key":   "Нет",
        "fill_solid": True,
    }

    # ── refs ──
    file_label      = ft.Ref[ft.Text]()
    prop_field      = ft.Ref[ft.TextField]()
    scale_field     = ft.Ref[ft.TextField]()
    status_text     = ft.Ref[ft.Text]()
    status_card     = ft.Ref[ft.Container]()
    progress_bar    = ft.Ref[ft.ProgressBar]()
    progress_card   = ft.Ref[ft.Container]()
    convert_btn     = ft.Ref[ft.ElevatedButton]()
    rot_row         = ft.Ref[ft.Row]()
    fill_solid_btn  = ft.Ref[ft.ElevatedButton]()
    fill_vertex_btn = ft.Ref[ft.ElevatedButton]()
    stats_card      = ft.Ref[ft.Container]()
    stat_blocks     = ft.Ref[ft.Text]()
    stat_scale      = ft.Ref[ft.Text]()
    stat_rot        = ft.Ref[ft.Text]()
    stat_fill       = ft.Ref[ft.Text]()

    # ── вспомогательные функции UI ──

    def set_status(msg: str, color: str = C_TEXT, show: bool = True):
        status_text.current.value = msg
        status_text.current.color = color
        status_card.current.visible = show
        page.update()

    def set_progress(val: float | None):
        """val=None → скрыть, 0..1 → показать"""
        if val is None:
            progress_card.current.visible = False
        else:
            progress_card.current.visible = True
            progress_bar.current.value    = val
        page.update()

    def show_stats(blocks: int, scale: float, rot: str, fill: str):
        stat_blocks.current.value = str(blocks)
        stat_scale.current.value  = f"{scale}×"
        stat_rot.current.value    = rot
        stat_fill.current.value   = fill
        stats_card.current.visible = True
        page.update()

    # ── файл-пикер ──
    def on_file_picked(e: ft.FilePickerResultEvent):
        if not e.files:
            return
        f = e.files[0]
        state["obj_path"] = f.path
        state["file_data"] = None
        try:
            with open(f.path, "rb") as fh:
                state["file_data"] = fh.read()
            ext = os.path.splitext(f.name)[1].lower()
            short = f.name if len(f.name) <= 30 else "…" + f.name[-27:]
            file_label.current.value = f"📄 {short}"
            file_label.current.color = C_TEXT
            # Показываем подсказку для .blend
            if ext == ".blend":
                set_status(
                    ".blend нельзя открыть напрямую.\n"
                    "В Blender: File → Export → glTF 2.0 → сохрани .glb\n"
                    "Затем загрузи .glb в Ze-Conventor.",
                    C_WARN, show=True,
                )
            else:
                set_status("", show=False)
            stats_card.current.visible = False
        except Exception as ex:
            set_status(f"Ошибка чтения: {ex}", C_ERROR)
        page.update()

    fp = ft.FilePicker(on_result=on_file_picked)
    page.overlay.append(fp)

    def pick_file(_):
        fp.pick_files(
            allowed_extensions=["obj", "glb", "gltf", "stl", "ply", "fbx", "blend"],
            allow_multiple=False,
        )

    # ── ротация ──
    rot_keys = list(ROTATIONS.keys())

    def build_rot_buttons():
        btns = []
        for rk in rot_keys:
            is_active = rk == state["rot_key"]
            b = ft.ElevatedButton(
                text=rk,
                style=ft.ButtonStyle(
                    bgcolor=C_PRIMARY if is_active else "#EEF0FF",
                    color=C_WHITE if is_active else C_MUTED,
                    side=ft.BorderSide(0, "transparent"),
                    padding=ft.padding.symmetric(horizontal=8, vertical=4),
                    shape=ft.RoundedRectangleBorder(radius=10),
                    elevation=2 if is_active else 0,
                ),
                on_click=lambda e, rk=rk: select_rot(rk),
            )
            btns.append(b)
        return btns

    def select_rot(rk: str):
        state["rot_key"] = rk
        rot_row.current.controls = build_rot_buttons()
        page.update()

    # ── режим заливки ──
    def set_fill(solid: bool):
        state["fill_solid"] = solid
        fill_solid_btn.current.style.bgcolor  = C_PRIMARY if solid else "#EEF0FF"
        fill_solid_btn.current.style.color    = C_WHITE   if solid else C_MUTED
        fill_vertex_btn.current.style.bgcolor = C_PRIMARY if not solid else "#EEF0FF"
        fill_vertex_btn.current.style.color   = C_WHITE   if not solid else C_MUTED
        page.update()

    # ── конвертация ──
    def do_convert(_):
        if state["busy"]:
            return
        if not state.get("file_data"):
            set_status("⚠ Сначала выберите файл модели", C_WARN)
            return

        prop_name = prop_field.current.value.strip() or "Props10"
        try:
            scale = float(scale_field.current.value.strip() or "30")
            if scale <= 0:
                raise ValueError
        except ValueError:
            set_status("⚠ Масштаб должен быть числом > 0", C_WARN)
            return

        state["busy"] = True
        convert_btn.current.disabled = True
        set_status("Парсинг модели…", C_MUTED)
        set_progress(0.0)
        stats_card.current.visible = False
        page.update()

        def worker():
            try:
                # Универсальный парсинг по расширению
                verts, tris = parse_file(state["obj_path"], state["file_data"])
                if not verts:
                    set_status("Ошибка: вершины не найдены в файле", C_ERROR)
                    return

                ext = os.path.splitext(state["obj_path"])[1].lower()
                set_status(f"Воксилизация ({ext})…", C_MUTED)

                def prog(v):
                    set_progress(v * 0.8)  # 0→80%

                gz_bytes, count = build_clm(
                    verts, tris,
                    prop_name=prop_name,
                    scale=scale,
                    rotation_key=state["rot_key"],
                    fill_solid=state["fill_solid"],
                    progress_cb=prog,
                )

                set_progress(0.95)
                set_status("Сохранение…", C_MUTED)

                # Определяем путь сохранения
                src_path = state["obj_path"] or ""
                if src_path:
                    base = os.path.splitext(src_path)[0]
                    out_path = base + ".clm.msg.gz"
                else:
                    import tempfile
                    out_path = os.path.join(tempfile.gettempdir(), "output.clm.msg.gz")

                with open(out_path, "wb") as fout:
                    fout.write(gz_bytes)

                set_progress(None)
                short_out = os.path.basename(out_path)
                set_status(f"✓ Готово! Сохранено: {short_out}", C_SUCCESS)
                show_stats(count, scale, state["rot_key"],
                           "Solid" if state["fill_solid"] else "Vertices")

            except Exception as ex:
                set_progress(None)
                set_status(f"Ошибка: {ex}", C_ERROR)
            finally:
                state["busy"] = False
                convert_btn.current.disabled = False
                page.update()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    # ═══════════════════════════════════════════════════════════════
    # КОМПОНОВКА UI
    # ═══════════════════════════════════════════════════════════════

    # ─ AppBar ─
    appbar = ft.AppBar(
        title=ft.Row([
            ft.Text("Ze", size=22, weight=ft.FontWeight.W_900, color=C_PRIMARY),
            ft.Text("-Conventor", size=22, weight=ft.FontWeight.W_300, color=C_TEXT),
        ], spacing=0),
        center_title=False,
        bgcolor=C_GLASS,
        shadow_color="#1A206010",
        elevation=2,
        actions=[
            ft.Container(
                content=ft.Text("SSB2", size=11, weight=ft.FontWeight.W_700,
                                color=C_WHITE),
                bgcolor=C_PRIMARY,
                border_radius=8,
                padding=ft.padding.symmetric(horizontal=10, vertical=4),
                margin=ft.margin.only(right=12, top=8, bottom=8),
            )
        ],
    )

    # ─ Секция 1: выбор файла ─
    file_section = glass_card(
        section_label("ИСХОДНЫЙ ФАЙЛ"),
        ft.Container(height=10),
        ft.Row([
            ft.Icon(ft.icons.INSERT_DRIVE_FILE_OUTLINED, color=C_PRIMARY, size=20),
            ft.Text(ref=file_label, value="Файл не выбран", color=C_MUTED,
                    size=14, expand=True, overflow=ft.TextOverflow.ELLIPSIS),
        ], spacing=10),
        ft.Container(height=14),
        ft.ElevatedButton(
            content=ft.Row([
                ft.Icon(ft.icons.FOLDER_OPEN_ROUNDED, color=C_WHITE, size=18),
                ft.Text("Выбрать модель", color=C_WHITE,
                        size=14, weight=ft.FontWeight.W_600),
            ], spacing=8, alignment=ft.MainAxisAlignment.CENTER),
            style=ft.ButtonStyle(
                bgcolor=C_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=14),
                padding=ft.padding.symmetric(horizontal=20, vertical=14),
                elevation=4,
                overlay_color=C_PRIMARY2,
            ),
            expand=True,
            on_click=pick_file,
        ),
        ft.Container(height=6),
        ft.Text(
            "OBJ · GLB · GLTF · STL · PLY · FBX",
            size=10, color=C_MUTED,
            text_align=ft.TextAlign.CENTER,
        ),
    )

    # ─ Секция 2: проп ─
    prop_section = glass_card(
        section_label("ОБЪЕКТ ИГРЫ (ПРОП)"),
        ft.Container(height=10),
        ft.TextField(
            ref=prop_field,
            value="Props10",
            hint_text="Например: Props10, Props2131…",
            hint_style=ft.TextStyle(color=C_MUTED),
            border_radius=14,
            border_color=C_BORDER,
            focused_border_color=C_PRIMARY,
            text_style=ft.TextStyle(color=C_TEXT, size=14,
                                    weight=ft.FontWeight.W_500),
            prefix_icon=ft.icons.WIDGETS_OUTLINED,
            filled=True,
            fill_color=C_GLASS2,
            content_padding=ft.padding.symmetric(horizontal=14, vertical=14),
        ),
    )

    # ─ Секция 3: масштаб ─
    scale_section = glass_card(
        section_label("МАСШТАБ"),
        ft.Container(height=10),
        ft.TextField(
            ref=scale_field,
            value="30",
            hint_text="Число > 0, например 30",
            hint_style=ft.TextStyle(color=C_MUTED),
            keyboard_type=ft.KeyboardType.NUMBER,
            border_radius=14,
            border_color=C_BORDER,
            focused_border_color=C_PRIMARY,
            text_style=ft.TextStyle(color=C_TEXT, size=14,
                                    weight=ft.FontWeight.W_500),
            prefix_icon=ft.icons.STRAIGHTEN_ROUNDED,
            filled=True,
            fill_color=C_GLASS2,
            content_padding=ft.padding.symmetric(horizontal=14, vertical=14),
        ),
        ft.Container(height=6),
        ft.Text("Увеличивает размер модели в игре (рекомендую 20–60)",
                size=11, color=C_MUTED),
    )

    # ─ Секция 4: режим заливки ─
    fill_section = glass_card(
        section_label("РЕЖИМ ЗАПОЛНЕНИЯ"),
        ft.Container(height=10),
        ft.Row([
            ft.ElevatedButton(
                ref=fill_solid_btn,
                content=ft.Row([
                    ft.Icon(ft.icons.CROP_DIN_ROUNDED, color=C_WHITE, size=16),
                    ft.Text("Solid", color=C_WHITE, size=13,
                            weight=ft.FontWeight.W_600),
                ], spacing=6),
                style=ft.ButtonStyle(
                    bgcolor=C_PRIMARY,
                    shape=ft.RoundedRectangleBorder(radius=12),
                    padding=ft.padding.symmetric(horizontal=14, vertical=12),
                    elevation=3,
                ),
                expand=True,
                on_click=lambda _: set_fill(True),
            ),
            ft.ElevatedButton(
                ref=fill_vertex_btn,
                content=ft.Row([
                    ft.Icon(ft.icons.GRAIN_ROUNDED, color=C_MUTED, size=16),
                    ft.Text("Vertices", color=C_MUTED, size=13,
                            weight=ft.FontWeight.W_600),
                ], spacing=6),
                style=ft.ButtonStyle(
                    bgcolor="#EEF0FF",
                    shape=ft.RoundedRectangleBorder(radius=12),
                    padding=ft.padding.symmetric(horizontal=14, vertical=12),
                    elevation=0,
                ),
                expand=True,
                on_click=lambda _: set_fill(False),
            ),
        ], spacing=10),
        ft.Container(height=6),
        ft.Text("Solid — полное заполнение объёма (рекомендуется)",
                size=11, color=C_MUTED),
    )

    # ─ Секция 5: ротация ─
    rotation_section = glass_card(
        section_label("ПОВОРОТ БЛОКОВ"),
        ft.Container(height=10),
        ft.Row(
            ref=rot_row,
            controls=build_rot_buttons(),
            wrap=True,
            spacing=6,
            run_spacing=6,
        ),
    )

    # ─ Прогресс ─
    progress_section = ft.Container(
        ref=progress_card,
        visible=False,
        content=glass_card(
            ft.Row([
                ft.Icon(ft.icons.AUTORENEW_ROUNDED, color=C_PRIMARY, size=18),
                ft.Text("Конвертация…", size=13, color=C_TEXT,
                        weight=ft.FontWeight.W_500),
            ], spacing=8),
            ft.Container(height=10),
            ft.ProgressBar(
                ref=progress_bar,
                value=0,
                bgcolor=C_BORDER,
                color=C_PRIMARY,
                border_radius=6,
                height=8,
            ),
        ),
    )

    # ─ Статус ─
    status_section = ft.Container(
        ref=status_card,
        visible=False,
        content=glass_card(
            ft.Row([
                ft.Icon(ft.icons.INFO_OUTLINE_ROUNDED, color=C_MUTED, size=16),
                ft.Text(ref=status_text, value="", size=13,
                        color=C_TEXT, expand=True),
            ], spacing=8),
        ),
    )

    # ─ Статистика ─
    def stat_pill(icon, label, ref_):
        return ft.Container(
            content=ft.Column([
                ft.Icon(icon, color=C_PRIMARY, size=18),
                ft.Text(ref=ref_, value="—", size=15, color=C_TEXT,
                        weight=ft.FontWeight.W_700),
                ft.Text(label, size=10, color=C_MUTED),
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            expand=True,
            bgcolor=C_GLASS2,
            border_radius=14,
            border=ft.border.all(1, C_BORDER),
            padding=ft.padding.symmetric(vertical=14, horizontal=8),
        )

    stats_section = ft.Container(
        ref=stats_card,
        visible=False,
        content=glass_card(
            section_label("РЕЗУЛЬТАТ"),
            ft.Container(height=10),
            ft.Row([
                stat_pill(ft.icons.LAYERS_ROUNDED,  "блоков",   stat_blocks),
                stat_pill(ft.icons.STRAIGHTEN,      "масштаб",  stat_scale),
                stat_pill(ft.icons.THREE_SIXTY,     "поворот",  stat_rot),
                stat_pill(ft.icons.GRID_ON_ROUNDED, "режим",    stat_fill),
            ], spacing=8),
        ),
    )

    # ─ Кнопка конвертации ─
    convert_section = ft.ElevatedButton(
        ref=convert_btn,
        content=ft.Row([
            ft.Icon(ft.icons.BOLT_ROUNDED, color=C_WHITE, size=22),
            ft.Text("Сконвертировать и сохранить",
                    color=C_WHITE, size=16, weight=ft.FontWeight.W_700),
        ], spacing=10, alignment=ft.MainAxisAlignment.CENTER),
        style=ft.ButtonStyle(
            bgcolor=C_PRIMARY,
            shape=ft.RoundedRectangleBorder(radius=18),
            padding=ft.padding.symmetric(horizontal=24, vertical=18),
            elevation=8,
            overlay_color=C_PRIMARY2,
            shadow_color="#3B4FDD40",
        ),
        on_click=do_convert,
    )

    # ─ Сборка страницы ─
    page.appbar = appbar

    body = ft.Column(
        controls=[
            file_section,
            ft.Container(height=10),
            ft.Row([
                ft.Container(prop_section,   expand=3),
                ft.Container(width=10),
                ft.Container(scale_section,  expand=2),
            ]),
            ft.Container(height=10),
            fill_section,
            ft.Container(height=10),
            rotation_section,
            ft.Container(height=16),
            convert_section,
            ft.Container(height=10),
            progress_section,
            status_section,
            stats_section,
            ft.Container(height=24),
        ],
        scroll=ft.ScrollMode.ADAPTIVE,
        spacing=0,
    )

    page.add(
        ft.Stack([
            # Градиентный фон
            ft.Container(
                expand=True,
                gradient=ft.LinearGradient(
                    begin=ft.alignment.top_left,
                    end=ft.alignment.bottom_right,
                    colors=["#E8EEFF", "#F5F0FF", "#EEF5FF"],
                ),
            ),
            # Декоративные круги
            ft.Container(
                width=280, height=280,
                top=-60, right=-80,
                border_radius=140,
                gradient=ft.RadialGradient(
                    colors=["#5B6EF520", "#5B6EF500"],
                    center=ft.alignment.center,
                    radius=0.5,
                ),
            ),
            ft.Container(
                width=200, height=200,
                bottom=100, left=-60,
                border_radius=100,
                gradient=ft.RadialGradient(
                    colors=["#7C8FFF18", "#7C8FFF00"],
                    center=ft.alignment.center,
                    radius=0.5,
                ),
            ),
            # Основной контент
            ft.Container(
                content=body,
                padding=ft.padding.symmetric(horizontal=16, vertical=16),
                expand=True,
            ),
        ], expand=True),
    )


ft.app(target=main)
