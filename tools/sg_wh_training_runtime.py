"""Portable data-free kit utilities; training runs only through the train subcommand."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bootstrap():
    """@luna-comment
    Input: clean local process. Output: existing block model and all five architectures registered.
    Key Steps: reuse V3M bootstrap, then explicitly import the isolated teacher.
    Key Params: ROOT is this copied repository. State/Dependencies: process-local registries only.
    """
    from tools.train_v3m_block256 import bootstrap as original
    original(str(ROOT))
    import realesrgan.archs.basic_text_line_r4_teacher_y4uv_v3l_arch


def initialize(directory, seed):
    """@luna-comment
    Input: output directory and seed. Output: five reproducible fresh initialization checkpoints.
    Key Steps: register architectures, reset seed per architecture, create CPU models and save params.
    Key Params: canvas256; never overwrite existing checkpoint files.
    State/Dependencies: torch RNG and newly created output files; no learned weights or data loaded.
    """
    import torch
    bootstrap()
    from basicsr.archs import build_network
    torch.set_num_threads(1)
    directory = Path(directory)
    names = ['V3M'] + [f'V3L{ratio}' for ratio in (16, 32, 64, 128)]
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / f'{name}.pth').exists() for name in names):
        raise FileExistsError('Initialization directory already contains a requested model')
    for name in names:
        torch.manual_seed(seed)
        options = dict(type='BasicTextLineR4StudentY4UVV3M', canvas_hw=[256, 256])
        if name != 'V3M':
            options.update(type='BasicTextLineR4TeacherY4UVV3L', compute_ratio=int(name[3:]))
        model = build_network(options)
        torch.save({'params': model.state_dict()}, directory / f'{name}.pth')
    print('Fresh random weights generated; copy this same directory to both sites:', directory)


def bind(template, bindings, destination):
    """@luna-comment
    Input: YAML template, explicit JSON placeholder mapping and destination path.
    Output: resolved YAML with no unresolved placeholders; no data or training is accessed.
    Key Steps: load mappings, replace named tokens and validate YAML syntax before writing.
    Key Params: ${NAME} substitutions must be nonempty; destination must be new.
    State/Dependencies: PyYAML and caller-supplied path strings; creates only the output config.
    """
    import yaml
    values = json.loads(Path(bindings).read_text(encoding='utf-8-sig'))
    content = Path(template).read_text(encoding='utf-8-sig')

    def replace(match):
        """@luna-comment
        Input: placeholder regex match. Output: explicitly supplied nonempty value.
        Key Steps: reject absent/blank bindings. Key Params: exact uppercase placeholder key.
        State/Dependencies: caller mapping only; no default path guesses.
        """
        value = values.get(match.group(1))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Missing explicit binding: {match.group(1)}')
        return value.replace('\\', '/')

    def resolve(value):
        """@luna-comment
        Input: parsed YAML value. Output: recursively bound value preserving YAML types.
        Key Steps: substitute strings and traverse lists/dictionaries, then serialize safely.
        Key Params: explicit placeholder regex. State/Dependencies: caller mapping only.
        """
        if isinstance(value, str):
            return re.sub(r'\$\{([A-Z0-9_]+)\}', replace, value)
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        return value
    content = yaml.safe_dump(resolve(yaml.safe_load(content)), sort_keys=False)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        stream.write(content)
    print(target)


def check(config=None):
    """@luna-comment
    Input: optional fully bound supervised config. Output: readiness report or explicit failure.
    Key Steps: register models, verify torch/CUDA; optionally check architecture and required files.
    Key Params: no data arrays opened and no optimizer/forward started.
    State/Dependencies: torch, YAML, registry and existence-only data/checkpoint checks.
    """
    import torch
    import yaml
    bootstrap()
    from basicsr.archs import build_network
    result = dict(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                  repository=str(ROOT), trained=False)
    if config:
        content = Path(config).read_text(encoding='utf-8-sig')
        if '${' in content:
            raise ValueError('Bind all placeholders before checking or training')
        options = yaml.safe_load(content)
        if options.get('kind') == 'feature_distillation_component':
            raise ValueError('KD data-pipeline launcher is not integrated; this is a component specification')
        network = build_network(options['network_g'])
        required = [options['path']['pretrain_network_g']]
        for split in ('train', 'val'):
            data = options['datasets'][split]
            required += [data['manifest'], str(Path(data.get('root', Path(data['manifest']).parent)) / 'COMPLETE.json')]
        missing = [str(p) for p in required if not Path(p).is_file()]
        result.update(architecture=type(network).__name__, missing_files=missing)
        if missing:
            raise FileNotFoundError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    """@luna-comment
    Input: check/init/bind/train subcommand. Output: explicit preparation or supervised training.
    Key Steps: validate CLI, delegate selected operation; training uses unchanged BasicSR pipeline.
    Key Params: train accepts only a bound supervised YAML and requires available CUDA.
    State/Dependencies: copied source root; never executes legacy remote controllers.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    p = actions.add_parser('check'); p.add_argument('--config')
    p = actions.add_parser('init'); p.add_argument('--directory', required=True); p.add_argument('--seed', type=int, default=20260828)
    p = actions.add_parser('bind'); p.add_argument('--template', required=True); p.add_argument('--bindings', required=True); p.add_argument('--output', required=True)
    p = actions.add_parser('train'); p.add_argument('-opt', required=True)
    args = parser.parse_args()
    if args.action == 'init':
        initialize(args.directory, args.seed)
    elif args.action == 'bind':
        bind(args.template, args.bindings, args.output)
    elif args.action == 'check':
        check(args.config)
    else:
        check(args.opt)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Formal supervised training requires a verified CUDA environment')
        sys.argv = [sys.argv[0], '-opt', str(Path(args.opt).resolve())]
        from basicsr.train import train_pipeline
        train_pipeline(str(ROOT))


if __name__ == '__main__':
    main()
