"""Download and restore this release's shared base and both LoRA adapters.

Python 3.12+; standard library only. No model code is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'DialectS2S-original' / 'model_manifest.json'
BLOCK = 4 * 1024 * 1024


def file_sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def obtain(item, directory):
    path = directory / item['name']
    if path.exists():
        if path.stat().st_size == item['bytes'] and file_sha(path) == item['sha256']:
            print('Verified existing:', item['name'], flush=True)
            return path
        raise ValueError(f'Existing file differs; inspect or remove it before retry: {path}')
    temporary = path.with_name(path.name + '.partial')
    if temporary.exists():
        raise FileExistsError(f'Incomplete previous download; inspect or remove it before retry: {temporary}')
    request = urllib.request.Request(item['url'], headers={'User-Agent': 'AI-CantoTriage-model-download'})
    digest = hashlib.sha256()
    length = 0
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open('xb') as out:
        while block := response.read(BLOCK):
            length += len(block)
            if length > item['bytes']:
                raise ValueError('Download exceeds expected size: ' + item['name'])
            digest.update(block)
            out.write(block)
    if length != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise ValueError('Download checksum mismatch: ' + item['name'])
    temporary.rename(path)
    print('Downloaded and verified:', item['name'], flush=True)
    return path


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.stream = None

    def readable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            raise ValueError('Use bounded streaming reads')
        blocks = []
        while n:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.stream = path.open('rb')
            block = self.stream.read(n)
            if not block:
                self.stream.close()
                self.stream = None
                continue
            blocks.append(block)
            n -= len(block)
        return b''.join(blocks)

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        super().close()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or chr(92) in name or ':' in name:
        raise ValueError('Unsafe archive path: ' + name)
    result = root.joinpath(*relative.parts)
    result.resolve().relative_to(root.resolve())
    return result


def write_file(stream, destination, expected):
    if destination.exists():
        if (destination.is_file() and destination.stat().st_size == expected['bytes']
                and file_sha(destination) == expected['sha256']):
            return
        raise ValueError(f'Will not overwrite a different existing file: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    digest = hashlib.sha256()
    length = 0
    with partial.open('xb') as out:
        while block := stream.read(BLOCK):
            length += len(block)
            if length > expected['bytes']:
                raise ValueError('Archive member exceeds expected size: ' + str(destination))
            digest.update(block)
            out.write(block)
    if length != expected['bytes'] or digest.hexdigest() != expected['sha256']:
        raise ValueError('Restored file checksum mismatch: ' + str(destination))
    partial.rename(destination)


def restore_base(manifest, paths, output):
    archive = manifest['archive']
    # Verify the concatenated tar checksum before any extraction.
    digest = hashlib.sha256()
    length = 0
    for path in paths:
        with path.open('rb') as source:
            while block := source.read(BLOCK):
                digest.update(block)
                length += len(block)
    if length != archive['bytes'] or digest.hexdigest() != archive['sha256']:
        raise ValueError('Concatenated archive checksum mismatch')
    expected = {item['path']: item for item in archive['files']}
    seen = set()
    notice_names = {'PUBLICATION_NOTICE.md', 'snapshot_files.json'}
    with PartsReader(paths) as reader, tarfile.open(fileobj=reader, mode='r|') as tar:
        for member in tar:
            if not member.isfile() or member.name in seen:
                raise ValueError('Non-file or duplicate archive member: ' + member.name)
            seen.add(member.name)
            destination = safe_path(output, member.name)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError('Unreadable archive member')
            if member.name in expected:
                item = expected[member.name]
                if member.size != item['bytes']:
                    raise ValueError('Unexpected member size: ' + member.name)
                write_file(source, destination, item)
            elif member.name in notice_names and member.size < 1024*1024:
                data = source.read()
                item = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                write_file(io.BytesIO(data), destination, item)
            else:
                raise ValueError('Unexpected archive member: ' + member.name)
            source.close()
    if seen != set(expected) | notice_names:
        raise ValueError('Archive file inventory differs')
    print('Original model and decoder restored; all file hashes verified.', flush=True)


def restore_lora(manifest, archive_path, output):
    root = safe_path(output, 'glm-4-DialectS2S-CantoTriage-dual-LoRA')
    expected = {item['path']: item for item in manifest['lora']['files']}
    seen = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = member.filename
            if name in seen or member.is_dir():
                raise ValueError('Duplicate or directory adapter member: ' + name)
            seen.add(name)
            destination = safe_path(root, name)
            with archive.open(member) as source:
                if name in expected:
                    item = expected[name]
                    if member.file_size != item['bytes']:
                        raise ValueError('Adapter size mismatch: ' + name)
                    write_file(source, destination, item)
                elif name == 'PUBLICATION_NOTICE.md' and member.file_size < 1024*1024:
                    data = source.read()
                    write_file(io.BytesIO(data), destination,
                               {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
                else:
                    raise ValueError('Unexpected adapter member: ' + name)
    if seen != set(expected) | {'PUBLICATION_NOTICE.md'}:
        raise ValueError('Adapter file inventory differs')
    print('Both adapters restored and verified:', root / 'adapters', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['base','lora','both'], default='both')
    parser.add_argument('--archives-dir', type=Path, default=HERE / 'downloaded_assets')
    parser.add_argument('--output', type=Path, default=HERE / 'restored')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text('utf-8'))
    args.archives_dir.mkdir(parents=True, exist_ok=True)
    # lora downloads only the adapters; inference still requires the shared base.
    paths = []
    if args.variant in ('base','both'):
        paths = [obtain(x, args.archives_dir) for x in manifest['archive']['parts']]
    lora_path = None
    if args.variant in ('lora','both'):
        lora_path = obtain(manifest['lora'], args.archives_dir)
    if args.download_only:
        return
    if paths:
        restore_base(manifest, paths, args.output)
    if lora_path:
        restore_lora(manifest, lora_path, args.output)


if __name__ == '__main__':
    main()
