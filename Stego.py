"""
Copyright 2026 MrMax

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# Native library
import argparse
import zlib
import struct
from pathlib import Path
# additional library from pip
import numpy as np
from PIL import Image
from io import BytesIO


class LSBSteganography:
    MAGIC = b"RSDF"
    TYPE_TEXT = b"T"
    TYPE_IMAGE = b"I"

    def __init__(self, lsb_per_channel: int = 1):
        if lsb_per_channel not in (1, 2):
            raise ValueError("lsb_per_channel must be 1 or 2")
        self.lsb = lsb_per_channel

    @staticmethod
    def _bytes_to_bits(data: bytes) -> str:
        return ''.join(f'{b:08b}' for b in data)

    @staticmethod
    def _bits_to_bytes(bits: str) -> bytes:
        return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))

    def _set_lsb(self, value, bits):
        mask = 0xFF ^ ((1 << self.lsb) - 1)
        return (value & mask) | int(bits, 2)

    def _get_lsb(self, value):
        return format(value & ((1 << self.lsb) - 1), f'0{self.lsb}b')

    def _build_payload(self, payload, payload_type: str) -> bytes:
        if payload_type == "text":
            raw = payload.encode("utf-8")
            ptype = self.TYPE_TEXT

        elif payload_type == "image":
            buf = BytesIO()
            payload.save(buf, format="PNG")
            raw = buf.getvalue()
            ptype = self.TYPE_IMAGE

        else:
            raise ValueError("Supported type : text | image")

        compressed = zlib.compress(raw, level=9)
        header = self.MAGIC + ptype + struct.pack(">I", len(compressed))
        return header + compressed

    def hide(self, cover_path: str, output_path: str, payload, payload_type: str):
        cover = np.array(Image.open(cover_path).convert("RGB"))
        payload_bytes = self._build_payload(payload, payload_type)

        bits = self._bytes_to_bits(payload_bytes)
        capacity = cover.shape[0] * cover.shape[1] * 3 * self.lsb

        if len(bits) > capacity:
            raise ValueError("Host image too small")

        idx = 0
        for y in range(cover.shape[0]):
            for x in range(cover.shape[1]):
                for c in range(3):
                    if idx < len(bits):
                        chunk = bits[idx:idx + self.lsb].ljust(self.lsb, "0")
                        cover[y, x, c] = self._set_lsb(cover[y, x, c], chunk)
                        idx += self.lsb

        Image.fromarray(cover).save(output_path)

    def extract(self, stego_path: str):
        data = np.array(Image.open(stego_path).convert("RGB"))

        bits = []
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                for c in range(3):
                    bits.append(self._get_lsb(data[y, x, c]))

        raw = self._bits_to_bytes(''.join(bits))

        if raw[:4] != self.MAGIC:
            raise ValueError("No payload detected")

        ptype = raw[4:5]
        size = struct.unpack(">I", raw[5:9])[0]
        compressed = raw[9:9 + size]

        payload = zlib.decompress(compressed)

        if ptype == self.TYPE_TEXT:
            return payload.decode("utf-8")

        if ptype == self.TYPE_IMAGE:
            return Image.open(BytesIO(payload))

        raise ValueError("Unrecognised type")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool for hide or extract data - based on LSB Stegano"
    )

    # Argument positionnel
    parser.add_argument(
        "host_file",
        help="Host image file"
    )

    # Modes exclusifs
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-c", "--cover",
        action="store_true",
        help="Hide data"
    )
    mode_group.add_argument(
        "-e", "--extract",
        action="store_true",
        help="Extract data"
    )

    # Donnée à cacher (texte OU chemin de fichier)
    parser.add_argument(
        "--data",
        help="Text or image file you want to hide"
    )

    args = parser.parse_args()

    # Validation conditionnelle
    if args.cover and not args.data:
        parser.error("--data is mandatory with option --hide")

    return args


if __name__ == "__main__":
    flag_source_type: str = "text"
    args = parse_args()
    pathlibed_host_filename = Path(args.host_file)
    fichier_de_sortie: Path = pathlibed_host_filename.parent / f"{pathlibed_host_filename.stem}_stego.png"
    secret = args.data
    steg = LSBSteganography(lsb_per_channel=1)

    if args.cover:
        if Path(args.data).is_file():
            flag_source_type = "image"
            secret = Image.open(args.data)

        print(f"Hide mode on {args.host_file}")

        steg.hide(
            cover_path=args.host_file,
            output_path=str(fichier_de_sortie),
            payload=secret,
            payload_type=flag_source_type
        )
    elif args.extract:
        print(f"Extract mode on {args.host_file} in progress ...")
        result = steg.extract(args.host_file)

        if isinstance(result, Image.Image):
            result.save("extracted.jpg")
            print("Extracted file is extracted.jpg")
        else:
            print("Extracted text :", result)






