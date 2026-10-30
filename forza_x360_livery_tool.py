#!/usr/bin/env python3
"""
--Forza Livery Swiss Knife Tool--
--by RokkuDayo--

Decryption and encryption of livery (.liv) files from XBOX 360 Forza titles.

Part of the source code for the "Horizon" modding software was referenced.

The script currently only handles (.liv) catalog files. Vinyl group (.lvg) files are not yet supported.
It also cannot update the profile's livery database, so you'll need to create new slots manually that you can then import over.

Examples:
  python forza_liv.py decrypt design.liv design.bin --game fm3 --profile-id 0123456789ABCDEF
  python forza_liv.py decrypt design.liv design.bin --game fm4 --profile-id 0123456789ABCDEF
  python forza_liv.py encrypt design.bin design.liv --game horizon --profile-id 0123456789ABCDEF

Forza 4 and Forza Horizon use built-in hexadecimal base keys by default.

Requires the third-party ``cryptography`` package. Install it with:
  python -m pip install cryptography
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Any


HEADER_SIZE = 20
FM3_DIGEST_SIZE = 20
FM4_DIGEST_SIZE = 32
BLOCK_SIZE = 16

# Forza3Class.cs, ForzaFileTypes.Liveries: catalog/design data.
FM3_CATALOG_AES = bytes.fromhex("13 B9 90 B1 33 60 AF 48 FA 0B 3E 1A 26 BF 42 36")
FM3_CATALOG_HMAC = bytes.fromhex("7F 7D 5E 05 98 64 C8 79 2E 20 6F FF DD 0A 42 91")
FM3_CATALOG_AES_START = 5
FM3_CATALOG_HMAC_START = 1

# Forza Motorsport 4 catalog keys recovered from the game's XEX
FM4_CATALOG_AES = bytes.fromhex("C49018BA8DA53B62D2A2EC691CE720E0")
FM4_CATALOG_HMAC = bytes.fromhex(
    "D7E857CD2877778E94D5D878D6C373499B9E32D4DB8A5EE2E8F83C99BC791D94"
)

# Forza Horizon catalog keys recovered from the game's XEX
HORIZON_CATALOG_AES = bytes.fromhex("65637420746F20746865206F74686572")
HORIZON_CATALOG_HMAC = bytes.fromhex(
    "65206E6574776F726B206973737565732E00556E61626C6520746F20636F6E6E"
)
BUILTIN_CATALOG_KEYS = {
    "fm4": (FM4_CATALOG_AES, FM4_CATALOG_HMAC),
    "horizon": (HORIZON_CATALOG_AES, HORIZON_CATALOG_HMAC),
}

# The session-key transforms used by the source's catalog livery handlers.
SESSION_SETTINGS = {
    "fm4": {"aes_start": 5, "sha_start": 1, "horizon_transform": False},
    "horizon": {"aes_start": 1, "sha_start": -3, "horizon_transform": True},
}


def parse_hex_bytes(value: str, label: str) -> bytes:
    text = re.sub(r"[\s:_-]", "", value)
    if text[:2].lower() == "0x":
        text = text[2:]
    if not text or len(text) % 2:
        raise ValueError(f"{label} must be a non-empty, even-length hexadecimal value")
    try:
        return bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"{label} contains a non-hexadecimal character") from exc


def parse_profile_id(value: str) -> bytes:
    creator = parse_hex_bytes(value, "Profile ID")
    if len(creator) != 8:
        raise ValueError("Profile ID must contain exactly 16 hexadecimal digits (8 bytes)")
    return creator


def transform_session_key(base_key: bytes, creator: bytes, start: int,
                          horizon_transform: bool) -> bytes:
    if not base_key or len(base_key) & (len(base_key) - 1):
        raise ValueError("Base key length must be a non-zero power of two")

    mask = len(base_key) - 1
    transformed = bytearray(len(base_key))
    increment = 0
    for index in range(len(base_key)):
        creator_index = (index + start) & 7
        value = base_key[index & mask] ^ creator[creator_index]
        if horizon_transform:
            value = (value + increment) & 0xFF
            increment += 2
        transformed[index] = value
    return bytes(transformed)


def load_user_keys(args: argparse.Namespace, game: str) -> tuple[bytes, bytes]:
    if game == "fm3":
        if args.keys_file or args.aes_key or args.sha_key:
            raise ValueError("Forza 3 uses keys embedded in this script; omit key arguments")
        return (
            transform_session_key(FM3_CATALOG_AES, args.profile_id,
                                  FM3_CATALOG_AES_START, False),
            transform_session_key(FM3_CATALOG_HMAC, args.profile_id,
                                  FM3_CATALOG_HMAC_START, False),
        )

    settings = SESSION_SETTINGS[game]
    if args.keys_file:
        if args.aes_key or args.sha_key:
            raise ValueError("Use either --keys-file or both --aes-key and --sha-key")
        try:
            key_data: Any = json.loads(Path(args.keys_file).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse key JSON: {exc}") from exc
        if not isinstance(key_data, dict):
            raise ValueError("Key JSON must be an object with aes_key and sha_key fields")
        aes_hex = key_data.get("aes_key")
        sha_hex = key_data.get("sha_key")
        if not isinstance(aes_hex, str) or not isinstance(sha_hex, str):
            raise ValueError("Key JSON must contain string fields aes_key and sha_key")
    elif args.aes_key and args.sha_key:
        aes_hex, sha_hex = args.aes_key, args.sha_key
    elif args.aes_key or args.sha_key:
        raise ValueError("Provide both --aes-key and --sha-key")
    elif game in BUILTIN_CATALOG_KEYS:
        base_aes, base_sha = BUILTIN_CATALOG_KEYS[game]
    else:
        raise ValueError(f"{game} requires base AES and SHA keys via --keys-file or key options")

    if args.keys_file or args.aes_key:
        base_aes = parse_hex_bytes(aes_hex, "AES base key")
        base_sha = parse_hex_bytes(sha_hex, "SHA base key")
    if len(base_aes) != 16:
        raise ValueError("AES base key must be 16 bytes (32 hexadecimal digits)")
    if len(base_sha) != 32:
        raise ValueError("SHA base key must be 32 bytes (64 hexadecimal digits)")

    aes_key = transform_session_key(
        base_aes, args.profile_id, settings["aes_start"], settings["horizon_transform"]
    )
    sha_key = transform_session_key(
        base_sha, args.profile_id, settings["sha_start"], settings["horizon_transform"]
    )
    return aes_key, sha_key


def aes_cbc(data: bytes, key: bytes, iv: bytes, encrypt: bool) -> bytes:
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError as exc:
        raise ValueError(
            "Missing dependency: install cryptography with "
            "'python -m pip install cryptography'"
        ) from exc

    if len(data) % BLOCK_SIZE:
        raise ValueError("AES-CBC input length must be a multiple of 16 bytes")
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    context = cipher.encryptor() if encrypt else cipher.decryptor()
    return context.update(data) + context.finalize()


def file_layout(game: str) -> tuple[int, int]:
    if game == "fm3":
        return FM3_DIGEST_SIZE, FM3_DIGEST_SIZE + HEADER_SIZE
    return FM4_DIGEST_SIZE, FM4_DIGEST_SIZE + HEADER_SIZE


def rounded_size(size: int) -> int:
    return (size + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1)


def decrypt_liv(data: bytes, game: str, aes_key: bytes, sha_key: bytes) -> bytes:
    digest_size, data_offset = file_layout(game)
    if len(data) < data_offset:
        raise ValueError("Input is too short to contain a Forza livery header")

    stored_digest = data[:digest_size]
    enc_header = data[digest_size:data_offset]
    iv = enc_header[4:20]
    encoded_size = int.from_bytes(enc_header[:4], "big")
    logical_size = encoded_size
    for offset in range(0, 16, 4):
        logical_size ^= int.from_bytes(iv[offset:offset + 4], "big")

    if logical_size <= 0 or logical_size > 0xFFFFFFFF:
        raise ValueError("Decoded file size in header is invalid")
    cipher_size = rounded_size(logical_size)
    ciphertext = data[data_offset:]
    if len(ciphertext) != cipher_size:
        raise ValueError(
            f"Header declares {cipher_size} encrypted bytes, but file has {len(ciphertext)}"
        )

    digest_name = "sha1" if game == "fm3" else "sha256"
    expected_digest = hmac.new(sha_key, enc_header + ciphertext, digest_name).digest()
    if not hmac.compare_digest(stored_digest, expected_digest):
        raise ValueError(
            "HMAC verification failed. Check the game, Profile ID, and base keys."
        )

    plaintext_padded = aes_cbc(ciphertext, aes_key, iv, encrypt=False)
    return plaintext_padded[:logical_size]


def encrypt_liv(plaintext: bytes, game: str, aes_key: bytes, sha_key: bytes) -> bytes:
    if not plaintext:
        raise ValueError("Cannot encrypt an empty livery file")
    if len(plaintext) > 0xFFFFFFFF:
        raise ValueError("Livery is too large for the Forza header")

    digest_size, _ = file_layout(game)
    iv = os.urandom(BLOCK_SIZE)
    pad_length = rounded_size(len(plaintext)) - len(plaintext)
    padded = plaintext + (os.urandom(pad_length) if pad_length else b"")
    ciphertext = aes_cbc(padded, aes_key, iv, encrypt=True)

    encoded_size = len(plaintext)
    for offset in range(0, 16, 4):
        encoded_size ^= int.from_bytes(iv[offset:offset + 4], "big")
    enc_header = encoded_size.to_bytes(4, "big") + iv

    digest_name = "sha1" if game == "fm3" else "sha256"
    digest = hmac.new(sha_key, enc_header + ciphertext, digest_name).digest()
    if len(digest) != digest_size:
        raise ValueError("Internal digest size does not match selected game")
    return digest + enc_header + ciphertext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decrypt or encrypt a Forza .liv livery catalog file."
    )
    parser.add_argument("action", choices=("decrypt", "encrypt"))
    parser.add_argument("input", type=Path, help="Input .liv or decrypted data file")
    parser.add_argument("output", type=Path, help="Output path")
    parser.add_argument("--game", choices=("fm3", "fm4", "horizon"), required=True)
    parser.add_argument("--profile-id", required=True,
                        help="16 hexadecimal digits, as displayed by the Xbox profile")
    keys = parser.add_mutually_exclusive_group()
    keys.add_argument("--keys-file", type=Path,
                      help="JSON file with aes_key and sha_key hexadecimal values")
    keys.add_argument("--aes-key", help="Base AES key as hexadecimal (use with --sha-key)")
    parser.add_argument("--sha-key", help="Base SHA key as hexadecimal (use with --aes-key)")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        creator = parse_profile_id(args.profile_id)
        args.profile_id = creator
        source_path = args.input.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        if source_path == output_path:
            raise ValueError("Input and output paths must be different")
        if output_path.exists() and not args.force:
            raise ValueError("Output already exists; choose another path or pass --force")

        aes_key, sha_key = load_user_keys(args, args.game)
        input_data = source_path.read_bytes()
        if args.action == "decrypt":
            output_data = decrypt_liv(input_data, args.game, aes_key, sha_key)
        else:
            output_data = encrypt_liv(input_data, args.game, aes_key, sha_key)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(output_data)
        print(f"{args.action.title()}ed {len(output_data)} bytes to {output_path}")
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
