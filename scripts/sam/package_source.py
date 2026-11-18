"""Package experiment source for transfer to a GPU runtime; exclude datasets and weights."""
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ('src', 'scripts', 'configs', 'tests', 'docs', 'notebooks')
SOURCE_SUFFIXES = {'.py', '.sh', '.json', '.md', '.txt', '.ipynb', '.yaml', '.yml'}


def main():
    output = ROOT / 'data/transfer/sam_project_source.tar.gz'
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {ROOT / name for name in ('.gitignore', 'README.md', 'requirements.txt', 'requirements-sam.txt')}
    for directory in SOURCE_DIRS:
        files.update(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and not p.is_symlink()
                     and p.suffix in SOURCE_SUFFIXES
                     and not any(part.startswith('.') or part == '__pycache__'
                                 for part in p.relative_to(ROOT).parts))
    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(Path(ROOT.name) / path.relative_to(ROOT)), recursive=False)
    print(f'Packaged {len(files)} source files: {output}')


if __name__ == '__main__':
    main()
