import argparse
from utils import get_logger
from Attacks.AttackConfig import AttackConfig
import json
from collections import defaultdict

logger = get_logger(__name__)
DESCRIPTION = """Obtain evaluation results from CodeQL scan results (sarif-latest format)
This script outputs the number of hits for reach relevant rule according to an attack config."""


def get_hits(results: dict, rulenames: list[str]) -> list:
    r = results['runs'][0]['results']
    out = defaultdict(list)
    for hit in r:
        ruleid = hit['ruleId'].strip()
        if ruleid in rulenames:
            out[ruleid].append(hit)
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--attackconfig', type=str, required=True, help='AttackConfig file')
    parser.add_argument('-r', '--results', type=str, required=True, nargs='+', help='CodeQL results file (sarif-latest format)')
    parser.add_argument('-o', '--output', type=str, required=False, help='Write JSON output to this file')

    args = parser.parse_args()
    cfg = AttackConfig.load(args.attackconfig)
    assert cfg
    if not hasattr(cfg, 'codeqlrules'):
        logger.error('AttackConfig does not have codeqlrules field')
        return
    rulenames = cfg.codeqlrules
    if not rulenames:
        logger.error('codeqlrules is empty')
        return
    rulenames = [r.strip() for r in rulenames]

    hitsperfile = dict()
    for path in args.results:
        with open(path, 'r') as f:
            results = json.load(f)
        assert results
        hitsperfile[path] = get_hits(results, rulenames)

    for path, hits in hitsperfile.items():
        outString = f'File: {path}\n'
        for ruleid, rulehits in hits.items():
            outString += f'    {ruleid}: {len(rulehits)} hits'
            filehits = set((loc['physicalLocation']['artifactLocation']['uri'] for hit in rulehits for loc in hit['locations']))
            outString += f' in {len(filehits)} files\n'
        logger.info('\n' + outString)

    if args.output:
        logger.info(f'Writing output to {args.output}')
        with open(args.output, 'w') as f:
            json.dump(hitsperfile, f, indent=2)


if __name__ == '__main__':
    main()
