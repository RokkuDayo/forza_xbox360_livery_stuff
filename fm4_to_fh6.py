#!/usr/bin/env python3
"""
--Forza Motorsport 4 to Forza Horizon 6 livery converter--
--by RokkuDayo--

Converts a decompressed Xbox 360 FM4 livery to an FH6 C_livery.
This is more of a silly experiment and does not try to properly reverse the format at all.

Do note that the final result won't load up in-game properly.
You must use Forza Livery Studio to open the final result and extract each region as an entire vinyl group.

Python 3.10+; standard library only. A current FH6 livery from the target
car supplies the metadata and paint/descriptor table.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import os
import zlib

REGIONS = ("Front", "Back", "Top", "Left", "Right", "Spoiler",
           "FrontWindshield", "BackWindshield", "TopWindow", "LeftWindow", "RightWindow")
# The FM4 root rotations describe the incoming panel frames. They are also
# present in FH6's section scaffolds, rather than in its artwork coordinates.
PANEL_ROTATIONS = (0., 0., 0., 0., 180., 90., -90., 90., 0., 0., 180.)
MAX_DEPTH = 64
MAX_FILE_SIZE = 128 * 1024 * 1024
DEFAULT_LOGO_PLACEHOLDER = 0x8000 | 378  # Turn 10 logo in the FH6 logo list.
KNOWN_VECTOR_STARTS = (
    101, 201, 301, 401, 501, 601, 701, 801, 901,
    1301, 1401, 1501, 1601, 1701, 1801, 1901, 2001,
    2101, 2201, 2301, 2401, 2501, 2601, 2701, 2801,
    2901, 3001, 3101, 3201, 3301, 3401, 3501, 3601, 3701, 3801,
)


class ConversionError(ValueError):
    """A file or an option cannot be converted without guessing its layout."""


@dataclass(frozen=True)
class Shape:
    shape_id: int
    transform: tuple[float, ...]  # rotation, X, Y, scale X, scale Y, skew
    color: bytes                # FM4 ARGB on input; FH6 BGRA after conversion


@dataclass(frozen=True)
class Group:
    transform: tuple[float, ...]  # X, Y, uniform scale, rotation
    children: tuple[Shape | Group, ...]


@dataclass(frozen=True)
class FM4Livery:
    roots: tuple[Group, ...]
    counts: tuple[int, ...]
    car_id: int
    creator_id: int
    header: bytes
    tail: bytes


@dataclass(frozen=True)
class FH6Template:
    prefix: bytes
    gyvl_header: bytes
    footer: bytes
    car_id: int
    original_counts: tuple[int, ...]


class Reader:
    def __init__(self, data: bytes, start: int, end: int, endian: str):
        if not 0 <= start <= end <= len(data):
            raise ConversionError("Invalid reader bounds.")
        self.data, self.pos, self.end, self.endian = data, start, end, endian

    def take(self, count: int) -> bytes:
        if count < 0 or self.pos + count > self.end:
            raise ConversionError(f"Truncated record at 0x{self.pos:X} ({count} bytes required).")
        result = self.data[self.pos:self.pos + count]
        self.pos += count
        return result

    def unpack(self, fmt: str) -> tuple:
        codec = struct.Struct(self.endian + fmt)
        return codec.unpack(self.take(codec.size))


def finite(values: tuple, address: int) -> None:
    if not all(math.isfinite(x) for x in values):
        raise ConversionError(f"Nonfinite transform at 0x{address:X}.")


def read_file(path: Path) -> bytes:
    if path.stat().st_size > MAX_FILE_SIZE:
        raise ConversionError(f"File exceeds the {MAX_FILE_SIZE // 1024 // 1024} MiB limit: {path}")
    return path.read_bytes()


def leaf_count(node: Shape | Group) -> int:
    return 1 if isinstance(node, Shape) else sum(leaf_count(child) for child in node.children)


def group_count(node: Shape | Group) -> int:
    return 0 if isinstance(node, Shape) else 1 + sum(group_count(child) for child in node.children)


def shapes_in(node: Shape | Group):
    if isinstance(node, Shape):
        yield node
    else:
        for child in node.children:
            yield from shapes_in(child)


def parse_fm4(data: bytes) -> FM4Livery:
    if len(data) < 49 or data[:4] != b"crlv":
        raise ConversionError("FM4 input must be decompressed and start with 'crlv' (big-endian).")
    if struct.unpack_from(">I", data, 4)[0] != 3 or data[25:29] != b"lvry":
        raise ConversionError("Unsupported FM4 header; expected crlv version 3 with lvry at 0x19.")
    if struct.unpack_from(">I", data, 29)[0] != 14 or data[45:49] != b"lvyg":
        raise ConversionError("Unsupported FM4 layout; expected lvry version 14 and lvyg at 0x2D.")
    artwork_end = 45 + struct.unpack_from(">I", data, 41)[0]
    if artwork_end < 49 + 6 * 23 or artwork_end + 28 > len(data):
        raise ConversionError("FM4 artwork length or section counters exceed the file bounds.")
    counters = struct.unpack_from(">7I", data, artwork_end)
    if counters[6] != 0:
        raise ConversionError("Unsupported nonzero FM4 extension counter.")
    reader = Reader(data, 49, artwork_end, ">")

    def read_group(depth: int) -> Group:
        start = reader.pos
        if depth > MAX_DEPTH:
            raise ConversionError(f"Group nesting exceeds {MAX_DEPTH} at 0x{start:X}.")
        transform = reader.unpack("4f")
        finite(transform, start)
        control, count, bitmap_size = reader.unpack("BHI")
        if control != 0:
            raise ConversionError(f"Unknown FM4 group control 0x{control:02X} at 0x{start + 16:X}.")
        if bitmap_size != (count + 7) // 8:
            raise ConversionError(f"Invalid bitmap at 0x{start:X}: {count} children, {bitmap_size} bytes.")
        if reader.pos + bitmap_size + count * 23 > artwork_end:
            raise ConversionError(f"Group children exceed the artwork boundary at 0x{start:X}.")
        bitmap = reader.take(bitmap_size)
        children = []
        for index in range(count):
            if bitmap[index // 8] & (1 << (index % 8)):
                children.append(read_group(depth + 1))
            else:
                address = reader.pos
                fields = reader.unpack("H6f4s")
                finite(fields[1:7], address)
                children.append(Shape(fields[0], fields[1:7], fields[7]))
        return Group(transform, tuple(children))

    roots = tuple(read_group(0) for _ in range(6))
    if reader.pos != artwork_end:
        raise ConversionError(f"Unparsed FM4 artwork at 0x{reader.pos:X}; expected end 0x{artwork_end:X}.")
    actual = tuple(leaf_count(root) for root in roots)
    if actual != counters[:6]:
        raise ConversionError(f"FM4 counters disagree with the tree: {counters[:6]} versus {actual}.")
    return FM4Livery(roots, actual, struct.unpack_from(">I", data, 16)[0],
                     struct.unpack_from(">Q", data, 33)[0], data[:45], data[artwork_end + 28:])


def unwrap_fh6(data: bytes) -> bytes:
    if data.startswith(b"vlrc"):
        return data
    if len(data) < 8:
        raise ConversionError("FH6 template is shorter than the compression header.")
    compressed, expanded = struct.unpack_from("<II", data)
    if compressed != len(data) - 8 or expanded > MAX_FILE_SIZE:
        raise ConversionError("Invalid FH6 compression wrapper or excessive decompressed size.")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(data[8:], expanded + 1)
    except zlib.error as exc:
        raise ConversionError(f"FH6 template zlib data is invalid: {exc}") from exc
    if len(raw) != expanded or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ConversionError("FH6 template has incomplete zlib data or incorrect size fields.")
    return raw


def parse_template(data: bytes) -> FH6Template:
    raw = unwrap_fh6(data)
    if len(raw) < 72 or raw[:4] != b"vlrc" or struct.unpack_from("<I", raw, 4)[0] != 2:
        raise ConversionError("Expected a current FH6 template (vlrc version 2).")
    if raw[26:30] != b"yrvl" or struct.unpack_from("<I", raw, 30)[0] != 19:
        raise ConversionError("Unsupported FH6 livery-information layout (expected yrvl version 19).")
    if raw[51:55] != b"gyvl" or struct.unpack_from("<I", raw, 55)[0] != 0:
        raise ConversionError("Expected embedded gyvl version 0 at offset 0x33.")
    gyvl_size = struct.unpack_from("<I", raw, 47)[0]
    stats = 51 + gyvl_size
    if gyvl_size < 21 or stats + 52 + 14 > len(raw) or raw[stats:stats + 4] != b"yrvl":
        raise ConversionError("FH6 artwork length does not lead to the section-count tag.")
    counts = struct.unpack_from("<12I", raw, stats + 4)
    if counts[11] != 0:
        raise ConversionError("Unsupported nonzero FH6 extension counter.")
    footer_start = stats + 52
    footer = raw[footer_start:]
    if footer[:4] != b"yrvl" or footer[4] > 1 or footer[5] != 2:
        raise ConversionError("FH6 template has an unsupported paint/descriptor table.")
    materials = struct.unpack_from("<H", footer, 6)[0]
    panel_offset = 14 + materials * 27
    if panel_offset + 4 > len(footer) - 8:
        raise ConversionError("FH6 template material records extend beyond the descriptor table.")
    panels = struct.unpack_from("<I", footer, panel_offset)[0]
    if panel_offset + 4 + panels * 8 > len(footer) - 8 or footer[-8:] != b"yrvl\0\0\0\0":
        raise ConversionError("FH6 template has invalid panel descriptors or a missing terminator.")
    return FH6Template(raw[:51], raw[51:72], footer,
                       struct.unpack_from("<I", raw, 16)[0], counts[:11])


def is_known_vector(shape_id: int) -> bool:
    return shape_id in (2000, 3000) or any(start <= shape_id < start + 40 for start in KNOWN_VECTOR_STARTS)


def is_unmatched_vector(shape_id: int) -> bool:
    return shape_id < 0x8000 and not is_known_vector(shape_id)


def numeric_id(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ConversionError(f"An ID must be an integer or integer string: {value!r}")
    result = int(value, 16 if value.lower().startswith("0x") else 10) if isinstance(value, str) else int(value)
    if result < 0 or result > 0xFFFF:
        raise ConversionError(f"ID must be an unsigned 16-bit integer: {value!r}")
    return result


def load_id_map(path: Path | None) -> dict[int, int]:
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ConversionError("Shape mapping must be a JSON object of source ID: target ID pairs.")
    result = {}
    for key, value in document.items():
        source, target = numeric_id(key), numeric_id(value)
        if source in result:
            raise ConversionError(f"Duplicate mapping for source ID {source}.")
        if target == 0:
            raise ConversionError("A shape mapping cannot silently remove a shape (target ID zero).")
        if (source >= 0x8000) != (target >= 0x8000):
            raise ConversionError("Vector-to-raster mappings require different shape geometry and are unsupported.")
        result[source] = target
    return result


def mapped_id(source_id: int, id_map: dict[int, int], logo_placeholder: int | None) -> int:
    # Explicit mappings take precedence, including mappings for individual logos.
    if source_id in id_map:
        return id_map[source_id]
    if source_id >= 0x8000 and logo_placeholder is not None:
        return logo_placeholder
    return source_id


def prepare_regions(source: FM4Livery, id_map: dict[int, int],
                    logo_placeholder: int | None = DEFAULT_LOGO_PLACEHOLDER,
                    skip_unmatched_shapes: bool = False) -> tuple[Group, ...]:
    def convert(node: Shape | Group) -> Shape | Group | None:
        if isinstance(node, Shape):
            target_id = mapped_id(node.shape_id, id_map, logo_placeholder)
            if skip_unmatched_shapes and is_unmatched_vector(target_id):
                return None
            # The same packed AARRGGBB integer is stored in opposite endian order.
            return replace(node, shape_id=target_id, color=node.color[::-1])
        children = []
        for child in node.children:
            converted_child = convert(child)
            if converted_child is not None and (isinstance(converted_child, Shape) or converted_child.children):
                children.append(converted_child)
        return replace(node, children=tuple(children))

    roots = []
    for slot, root in enumerate(source.roots):
        converted = convert(root)
        x, y, scale, rotation = root.transform
        # Remove the incoming panel frame, which belongs to FH6's scaffolding.
        artwork_rotation = (rotation - PANEL_ROTATIONS[slot] + 180.) % 360. - 180.
        children = converted.children
        if children and (x != 0. or y != 0. or scale != 1. or artwork_rotation != 0.):
            children = (Group((x, y, scale, artwork_rotation), children),)
        roots.append(Group((0., 0., 1., 0.), children))
    roots.extend(Group((0., 0., 1., 0.), ()) for _ in range(5))
    return tuple(roots)


def f32_values(values: tuple[float, ...]) -> tuple[float, ...]:
    """Round composed transforms to the actual on-disk precision."""
    finite(values, 0)
    try:
        codec = struct.Struct("<" + "f" * len(values))
        return codec.unpack(codec.pack(*values))
    except (OverflowError, struct.error) as exc:
        raise ConversionError("A composed group transform exceeds float32 range.") from exc


def compose_group_into_child(group: Group, child: Shape | Group) -> Shape | Group:
    """Remove a wrapper, retaining its uniform affine transform on its child."""
    px, py, scale, rotation = group.transform
    radians = math.radians(rotation % 360.)
    cosine, sine = math.cos(radians), math.sin(radians)
    if isinstance(child, Group):
        x, y, child_scale, child_rotation = child.transform
        transform = (px + scale * (cosine * x - sine * y),
                     py + scale * (sine * x + cosine * y),
                     scale * child_scale, (rotation + child_rotation) % 360.)
    else:
        child_rotation, x, y, sx, sy, skew = child.transform
        transform = ((rotation + child_rotation) % 360.,
                     px + scale * (cosine * x - sine * y),
                     py + scale * (sine * x + cosine * y),
                     scale * sx, scale * sy, skew)
    return replace(child, transform=f32_values(transform))


def normalize_artwork_groups(roots: tuple[Group, ...]) -> tuple[Group, ...]:
    """Artwork groups need at least two children; region roots are exempt."""
    def normalize(node: Shape | Group) -> Shape | Group | None:
        if isinstance(node, Shape):
            rotation, *rest = node.transform
            return replace(node, transform=f32_values((rotation % 360., *rest)))
        children = tuple(child for original in node.children
                         if (child := normalize(original)) is not None)
        if not children:
            return None
        x, y, scale, rotation = node.transform
        group = replace(node, children=children,
                        transform=f32_values((x, y, scale, rotation % 360.)))
        if len(children) == 1:
            return compose_group_into_child(group, children[0])
        return group

    # A section with a single shape is valid. Only its artwork wrappers collapse.
    return tuple(replace(root, children=tuple(child for original in root.children
                                             if (child := normalize(original)) is not None))
                 for root in roots)


def child_bitmap(children: tuple[Shape | Group, ...]) -> bytes:
    count = len(children)
    if count > 0xFFFF:
        raise ConversionError("A group exceeds the 16-bit direct-child limit.")
    bitmap = bytearray((count + 7) // 8)
    for index, child in enumerate(children):
        if isinstance(child, Group):
            bitmap[index // 8] |= 1 << (index % 8)
    return bytes(bitmap)


def terminal_group_depth(group: Group) -> int:
    """Group stack closed by the next sibling, matching the FH6 encoder."""
    if group.children and isinstance(group.children[-1], Group):
        return 1 + terminal_group_depth(group.children[-1])
    return 1


def encode_artwork(roots: tuple[Group, ...]) -> bytes:
    def encode_children(children: tuple[Shape | Group, ...], counted: bool) -> bytes:
        out = bytearray()
        for index, child in enumerate(children):
            if isinstance(child, Group):
                if len(child.children) < 2:
                    raise ConversionError("An artwork group has fewer than two children after normalization.")
                # Match ForzaLiveryStudio's canonical structural writer. The
                # first root child has an unmarked transform; a first nested
                # child uses 01. Other group starts close the previous sibling.
                if index == 0:
                    marker = b"\x01" if counted else b""
                elif isinstance(children[index - 1], Group):
                    marker = b"\x00" + b"\x01" * (terminal_group_depth(children[index - 1]) + 1)
                else:
                    marker = b"\x00\x01"
                out += marker + struct.pack("<4f", *child.transform)
                bitmap = child_bitmap(child.children)
                out += b"\x20" + struct.pack("<HH", len(child.children), len(bitmap)) + b"\0\0" + bitmap
                out += encode_children(child.children, True)
            else:
                # Counted groups start with 01 02; other shapes use normal 00 02.
                # Root shapes are framed too, preserving occupancy of unknown IDs.
                out += bytes((1 if counted and index == 0 else 0, 2))
                out += struct.pack("<H6f4s", child.shape_id, *child.transform, child.color)
        return bytes(out)

    counts = tuple(leaf_count(root) for root in roots)
    last_populated = max((i for i, count in enumerate(counts) if count), default=-1)
    body = bytearray()
    for slot, root in enumerate(roots):
        if counts[slot] == 0:
            # The final empty slot uses only the six-byte empty header when
            # the entire livery is empty. Otherwise all empty frames are full.
            empty = bytes(8) + struct.pack("<ff", 1., PANEL_ROTATIONS[slot]) + bytes(7)
            body += empty[:6] if last_populated < 0 and slot == 10 else empty
            continue
        bitmap = child_bitmap(root.children)
        body += struct.pack("<HH", len(root.children), len(bitmap)) + b"\0\0" + bitmap
        body += encode_children(root.children, False)
        if slot < last_populated:
            body += bytes(9) + struct.pack("<ff", 1., PANEL_ROTATIONS[slot + 1]) + b"\0"
        else:
            # Only the terminal state precedes the remaining empty scaffolds.
            body += b"\0"
    return bytes(body)


def decode_emitted_artwork(body: bytes, counts: tuple[int, ...]) -> tuple[Group, ...]:
    """Strict independent walk of the emitted grammar; no scanning or ID guesses."""
    reader = Reader(body, 0, len(body), "<")

    def read_children(count: int, bitmap: bytes, depth: int, counted: bool) -> tuple[Shape | Group, ...]:
        if depth > MAX_DEPTH + 1:
            raise ConversionError("Generated artwork exceeds the nesting limit.")
        children = []
        for index in range(count):
            if bitmap[index // 8] & (1 << (index % 8)):
                if index == 0:
                    expected_marker = b"\x01" if counted else b""
                elif isinstance(children[-1], Group):
                    expected_marker = b"\x00" + b"\x01" * (terminal_group_depth(children[-1]) + 1)
                else:
                    expected_marker = b"\x00\x01"
                if reader.take(len(expected_marker)) != expected_marker:
                    raise ConversionError("Generated group transform has invalid sibling framing.")
                transform = reader.unpack("4f")
                if reader.take(1) != b"\x20":
                    raise ConversionError("Generated group has invalid marker.")
                child_count, bitmap_size = reader.unpack("HH")
                if bitmap_size != (child_count + 7) // 8 or reader.take(2) != b"\0\0":
                    raise ConversionError("Generated group has invalid bitmap fields.")
                child_bitmap_data = reader.take(bitmap_size)
                children.append(Group(transform, read_children(child_count, child_bitmap_data, depth + 1, True)))
            else:
                wanted = b"\x01\x02" if counted and index == 0 else b"\x00\x02"
                if reader.take(2) != wanted:
                    raise ConversionError("Generated shape has invalid framing.")
                fields = reader.unpack("H6f4s")
                children.append(Shape(fields[0], fields[1:7], fields[7]))
        return tuple(children)

    last_populated = max((i for i, count in enumerate(counts) if count), default=-1)
    roots = []
    for slot, expected in enumerate(counts):
        if expected == 0:
            size = 6 if last_populated < 0 and slot == 10 else 23
            frame = reader.take(size)
            wanted = bytes(8) + struct.pack("<ff", 1., PANEL_ROTATIONS[slot]) + bytes(7)
            if frame != wanted[:size]:
                raise ConversionError(f"Invalid generated empty frame in {REGIONS[slot]}.")
            roots.append(Group((0., 0., 1., 0.), ()))
            continue
        count, bitmap_size = reader.unpack("HH")
        if bitmap_size != (count + 7) // 8 or reader.take(2) != b"\0\0":
            raise ConversionError("Generated region root has invalid bitmap fields.")
        bitmap = reader.take(bitmap_size)
        root = Group((0., 0., 1., 0.), read_children(count, bitmap, 0, False))
        if leaf_count(root) != expected:
            raise ConversionError(f"Generated {REGIONS[slot]} shape count is incorrect.")
        roots.append(root)
        remnant = bytes(9) + struct.pack("<ff", 1., PANEL_ROTATIONS[slot + 1]) + b"\0" if slot < last_populated else b"\0"
        if reader.take(len(remnant)) != remnant:
            raise ConversionError("Generated section remnant is invalid.")
    if reader.pos != reader.end:
        raise ConversionError("Generated artwork contains unparsed bytes.")
    return tuple(roots)


def convert_livery(source: FM4Livery, template: FH6Template, id_map: dict[int, int],
                   car_id: int | None = None,
                   logo_placeholder: int | None = DEFAULT_LOGO_PLACEHOLDER,
                   skip_unmatched_shapes: bool = False) -> tuple[bytes, dict]:
    roots = prepare_regions(source, id_map, logo_placeholder, skip_unmatched_shapes)
    groups_before_normalization = sum(group_count(root) - 1 for root in roots)
    roots = normalize_artwork_groups(roots)
    collapsed_groups = groups_before_normalization - sum(group_count(root) - 1 for root in roots)
    counts = tuple(leaf_count(root) for root in roots)
    skipped_sources = Counter()
    skipped_targets = Counter()
    skipped_regions = []
    for root in source.roots:
        skipped = 0
        if skip_unmatched_shapes:
            for shape in shapes_in(root):
                target_id = mapped_id(shape.shape_id, id_map, logo_placeholder)
                if is_unmatched_vector(target_id):
                    skipped_sources[shape.shape_id] += 1
                    skipped_targets[target_id] += 1
                    skipped += 1
        skipped_regions.append(skipped)
    expected_counts = tuple(original - skipped for original, skipped in zip(source.counts, skipped_regions))
    if counts[:6] != expected_counts:
        raise ConversionError("Converted shape counts disagree with the source and skipped records.")
    body = encode_artwork(roots)
    if decode_emitted_artwork(body, counts) != roots:
        raise ConversionError("Binary round-trip changed a shape, color, transform, or group.")
    prefix = bytearray(template.prefix)
    if car_id is not None:
        struct.pack_into("<I", prefix, 16, car_id)
    gyvl = template.gyvl_header + body
    struct.pack_into("<I", prefix, 47, len(gyvl))
    payload = bytes(prefix) + gyvl + b"yrvl" + struct.pack("<12I", *counts, 0) + template.footer
    checked = parse_template(payload)
    if checked.original_counts != counts or checked.footer != template.footer:
        raise ConversionError("FH6 metadata or descriptor table failed verification.")
    histogram = Counter(shape.shape_id for root in roots for shape in shapes_in(root))
    unknown = {str(key): value for key, value in sorted(histogram.items()) if is_unmatched_vector(key)}
    source_histogram = Counter(shape.shape_id for root in source.roots for shape in shapes_in(root))
    placeholder_uses = {str(key): value for key, value in sorted(source_histogram.items())
                        if key >= 0x8000 and key not in id_map and logo_placeholder is not None}
    manual_logos = Counter()
    for key, value in source_histogram.items():
        if key >= 0x8000 and (key in id_map or logo_placeholder is None):
            manual_logos[mapped_id(key, id_map, logo_placeholder)] += value
    report = {
        "source_format": "Xbox 360 FM4 crlv v3 / lvry v14", "target_format": "FH6 vlrc v2 / yrvl v19",
        "source_car_id": source.car_id, "target_car_id": checked.car_id,
        "source_total_shapes": sum(source.counts), "source_region_counts": dict(zip(REGIONS[:6], source.counts)),
        "total_shapes": sum(counts), "region_counts": dict(zip(REGIONS, counts)),
        "skip_unmatched_shapes": skip_unmatched_shapes,
        "skipped_shapes": sum(skipped_regions), "skipped_region_counts": dict(zip(REGIONS[:6], skipped_regions)),
        "skipped_source_shape_id_counts": {str(key): value for key, value in sorted(skipped_sources.items())},
        "skipped_target_shape_id_counts": {str(key): value for key, value in sorted(skipped_targets.items())},
        "source_nested_groups": sum(group_count(root) - 1 for root in source.roots),
        "output_nested_groups": sum(group_count(root) - 1 for root in roots),
        "collapsed_single_child_groups": collapsed_groups,
        "source_region_frames": {name: root.transform for name, root in zip(REGIONS, source.roots)},
        "source_shape_id_counts": {str(key): value for key, value in sorted(source_histogram.items())},
        "output_shape_id_counts": {str(key): value for key, value in sorted(histogram.items())},
        "unverified_vector_ids": unknown,
        "logo_placeholder_id": logo_placeholder,
        "logo_placeholder_name": "Turn 10" if logo_placeholder == DEFAULT_LOGO_PLACEHOLDER else None,
        "logo_placeholder_placements": sum(placeholder_uses.values()),
        "source_logo_ids_using_placeholder": placeholder_uses,
        "raster_ids_to_check": {str(key): value for key, value in sorted(manual_logos.items())},
        "shape_id_mapping": {str(key): value for key, value in sorted(id_map.items())},
        "color_conversion": "FM4 packed AARRGGBB big-endian to FH6 BGRA little-endian",
        "group_normalization": "empty groups omitted; one-child wrappers collapsed with transforms composed into their child",
        "rotation_conversion": "angles normalized to [0, 360) following the FH6 editor encoder",
        "metadata_and_paint": "preserved from FH6 template; FM4 paint tail retained below for reference",
        "source_creator_id": f"0x{source.creator_id:016X}",
        "source_header_base64": base64.b64encode(source.header).decode("ascii"),
        "source_paint_and_unknown_tail_base64": base64.b64encode(source.tail).decode("ascii"),
        "verification": "all emitted records decoded again; shapes, group tree, transforms, colors and counts match",
        "uncompressed_bytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return payload, report


def wrapped_payload(payload: bytes) -> bytes:
    compressed = zlib.compress(payload, 9)
    return struct.pack("<II", len(compressed), len(payload)) + compressed


def atomic_write(path: Path, data: bytes) -> None:
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".fm4_convert_", delete=False) as handle:
            temp_name = handle.name
            handle.write(data)
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def unsigned_car_id(value: str) -> int:
    try:
        result = int(value, 16 if value.lower().startswith("0x") else 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("car ID must be a decimal or hexadecimal integer") from exc
    if not 0 <= result <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("car ID must fit an unsigned 32-bit integer")
    return result


def logo_placeholder_id(value: str) -> int:
    try:
        result = numeric_id(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("logo ID must be a decimal or hexadecimal 16-bit integer") from exc
    if result & 0x7FFF == 0:
        raise argparse.ArgumentTypeError("logo ID must have a nonzero asset number")
    return result | 0x8000


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="decompressed FM4 livery (.bin, starting with crlv)")
    parser.add_argument("--template", required=True, type=Path, help="FH6 C_livery, decompressed payload, or folder containing C_livery")
    parser.add_argument("-o", "--output-dir", type=Path, help="output folder (default: INPUT_STEM_fh6 alongside input)")
    parser.add_argument("--car-id", type=unsigned_car_id, help="override target car ID; otherwise keep the template's car ID")
    parser.add_argument("--shape-map", type=Path, help="optional JSON mapping of FM4 shape/logo IDs to FH6 IDs")
    logo_options = parser.add_mutually_exclusive_group()
    logo_options.add_argument("--logo-placeholder", type=logo_placeholder_id,
                              default=DEFAULT_LOGO_PLACEHOLDER,
                              help="replacement for unmapped raster logos (default: 378, Turn 10); low asset ID or full wire ID")
    logo_options.add_argument("--preserve-logos", action="store_true", help="keep unmapped FM4 raster logo IDs")
    parser.add_argument("--strict-ids", action="store_true", help="stop if vector IDs are outside the supplied FH6 registry ranges")
    parser.add_argument("--skip-unmatched-shapes", "--skip-unmatched", action="store_true",
                        help="omit vector shapes whose resulting IDs are outside the FH6 registry ranges; mappings are applied first")
    parser.add_argument("--force", action="store_true", help="replace existing converter output files")
    args = parser.parse_args(argv)
    try:
        template_path = args.template / "C_livery" if args.template.is_dir() else args.template
        source_bytes = read_file(args.input)
        template_bytes = read_file(template_path)
        source = parse_fm4(source_bytes)
        template = parse_template(template_bytes)
        id_map = load_id_map(args.shape_map)
        placeholder = None if args.preserve_logos else args.logo_placeholder
        payload, report = convert_livery(source, template, id_map, args.car_id, placeholder,
                                         args.skip_unmatched_shapes)
        if args.strict_ids and report["unverified_vector_ids"]:
            raise ConversionError("Unverified vector IDs: " + ", ".join(report["unverified_vector_ids"]) +
                                  ". Supply --shape-map, use --skip-unmatched-shapes, or omit --strict-ids to retain them.")
        compressed = wrapped_payload(payload)
        if unwrap_fh6(compressed) != payload:
            raise ConversionError("Compressed output failed round-trip verification.")
        output_dir = args.output_dir or args.input.with_name(args.input.stem + "_fh6")
        outputs = (output_dir / "C_livery", output_dir / "C_livery.decompressed.bin", output_dir / "conversion_report.json")
        protected = {args.input.resolve(), template_path.resolve()}
        if args.shape_map:
            protected.add(args.shape_map.resolve())
        for path in outputs:
            if path.resolve() in protected:
                raise ConversionError(f"Output would overwrite an input file: {path}")
            if path.exists() and not args.force:
                raise ConversionError(f"Output already exists: {path}. Use another folder or --force.")
        report.update({"source_file": str(args.input.resolve()), "template_file": str(template_path.resolve()),
                       "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                       "template_sha256": hashlib.sha256(template_bytes).hexdigest(), "compressed_bytes": len(compressed)})
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(outputs[0], compressed)
        atomic_write(outputs[1], payload)
        atomic_write(outputs[2], (json.dumps(report, indent=2) + "\n").encode("utf-8"))
        print(f"Converted {report['total_shapes']} shapes and {report['output_nested_groups']} nested groups.")
        if report["collapsed_single_child_groups"]:
            print(f"Collapsed {report['collapsed_single_child_groups']} one-child group wrappers, retaining their transforms.")
        if report["skipped_shapes"]:
            print(f"Skipped {report['skipped_shapes']} of {report['source_total_shapes']} source shapes with unmatched IDs: " +
                  ", ".join(report["skipped_source_shape_id_counts"]) + ". Details are in conversion_report.json.")
        print("Regions: " + ", ".join(f"{name}={count}" for name, count in report["region_counts"].items() if count))
        print(f"Target car ID: {report['target_car_id']} (FM4 source ID: {source.car_id})")
        if report["logo_placeholder_placements"]:
            print(f"Generic logo placeholder: {placeholder} for {report['logo_placeholder_placements']} placements.")
        for path in outputs:
            print(path.resolve())
        if report["unverified_vector_ids"]:
            print("WARNING: FH6 availability is unverified for vector IDs: " + ", ".join(report["unverified_vector_ids"]) + ". All records are retained; details are in conversion_report.json.", file=sys.stderr)
        if report["raster_ids_to_check"]:
            print("Logo IDs retained or explicitly mapped: " + ", ".join(report["raster_ids_to_check"]) +
                  ". Their FH6 assets need checking.", file=sys.stderr)
        return 0
    except (ConversionError, OSError, ValueError, TypeError, struct.error) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
