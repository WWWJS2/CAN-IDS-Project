




import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    'ciciov2024': 'https://www.kaggle.com/api/v1/datasets/download/pushpakattarde/ciciov2024decimalcsv',
    'car_hacking': 'https://www.dropbox.com/scl/fo/9rwsf9pclhvv9xxloojom/AF7JeRW893grZkigkulkAHk?rlkey=3h6zamu3kc262lrnipu5qden8&dl=1'
}
FILES = {
    'ciciov2024': {'decimal_DoS.csv','decimal_benign.csv','decimal_spoofing-GAS.csv',
                   'decimal_spoofing-RPM.csv','decimal_spoofing-SPEED.csv','decimal_spoofing-STEERING_WHEEL.csv'},
    'car_hacking': {'DoS_dataset.csv','Fuzzy_dataset.csv','gear_dataset.csv','RPM_dataset.csv'}
}


def download(dataset, root):
    destination = Path(root)/dataset
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'Refusing to overwrite existing raw files: {destination}')
    destination.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        archive=Path(temp)/'source.zip'
        req=urllib.request.Request(SOURCES[dataset],headers={'User-Agent':'Phase1AcademicDataPreparation/1.0'})
        with urllib.request.urlopen(req,timeout=180) as src,archive.open('wb') as out:
            shutil.copyfileobj(src,out)
        found=set()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                name=Path(member.filename).name
                if name not in FILES[dataset]:
                    continue
                if name in found:
                    raise ValueError(f'Duplicate archive member {name}')
                found.add(name)
                
                with z.open(member) as src,(destination/name).open('xb') as out:
                    shutil.copyfileobj(src,out)
        if found != FILES[dataset]:
            raise ValueError(f'Missing expected CSVs: {FILES[dataset]-found}')
    print(f'Downloaded {len(found)} CSV files to {destination}. Compare SHA256 values with the delivered audit.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=SOURCES)
    p.add_argument('--root',default='data/raw')
    a=p.parse_args();download(a.dataset,a.root)
