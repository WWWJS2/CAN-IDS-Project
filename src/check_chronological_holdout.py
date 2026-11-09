
import argparse
import csv
import json
from pathlib import Path


def check(raw,audit_path):
    audit=json.loads(Path(audit_path).read_text())
    if audit['dataset']!='car_hacking':raise ValueError('This diagnostic expects Car-Hacking CSVs')
    findings=[]
    for source in audit['sources']:
        n=source['valid_rows'];cuts=[int(n*.70),int(n*.70)+int(n*.15)]
        counts=[{'BENIGN':0,'ATTACK':0} for _ in range(3)]
        first=last=None;total=0
        with (Path(raw)/source['file']).open(newline='') as f:
            for position,row in enumerate(csv.reader(f)):
                flag=row[-1].strip()
                if flag not in {'R','T'}:raise ValueError('Unknown flag')
                part=0 if position<cuts[0] else (1 if position<cuts[1] else 2)
                counts[part]['ATTACK' if flag=='T' else 'BENIGN']+=1;total+=1
                if flag=='T':
                    if first is None:first=position+1
                    last=position+1
        if total!=n:raise ValueError('Source row count changed')
        findings.append({'file':source['file'],'rows':n,'first_attack_row':first,'last_attack_row':last,
                         'chronological_parts':dict(zip(['train','validation','test'],counts))})
    return findings


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--audit',required=True)
    a=p.parse_args();print(json.dumps(check(a.raw,a.audit),indent=2))
