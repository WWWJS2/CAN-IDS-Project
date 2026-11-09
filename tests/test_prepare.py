
import csv
import json
import sys
import tempfile
import unittest
import subprocess
import shutil
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from prepare import FEATURES, integer, load_car, load_cic, prepare, fingerprint, split_groups
from prepare_streaming import run, records
from dataset_io import reference_assignment


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_csv(self, rows, name='input.csv'):
        p = self.root / name
        with p.open('w', newline='') as f:
            csv.writer(f).writerows(rows)
        return p

    def cic_fixture(self):
        rows = [FEATURES + ['label','category','specific_class']]
        for c, category in [('BENIGN','BENIGN'),('DOS','DOS'),('GAS','SPOOFING')]:
            for n in range(30):
                row = [100+n, n, 0,0,0,0,0,0,0, 'BENIGN' if c=='BENIGN' else 'ATTACK',category,c]
                rows.extend([row, row])  
        return self.write_csv(rows)

    def test_numeric_validation(self):
        self.assertEqual(integer('ff',16,255),255)
        for v in ['NaN','1.5','-1','256','inf','']:
            with self.assertRaises(ValueError): integer(v,10,255)

    def test_reference_assignment_is_reproducible_and_independent_of_storage_order(self):
        f=pd.DataFrame({'source_id':[0]*1000,'source_row':list(range(1,1001))})
        a={'dataset':'car_hacking','sources':[{'source_id':0,'valid_rows':1000}]}
        result=dict(zip(f.source_row,reference_assignment(f,a)))
        shuffled=f.sample(frac=1,random_state=4)
        self.assertEqual(result,dict(zip(shuffled.source_row,reference_assignment(shuffled,a))))
        self.assertEqual(set(result.values()),{'train','validation','test'})

    def test_cic_mapping(self):
        rejects=[]
        frame,n=load_cic(self.cic_fixture(),rejects)
        self.assertEqual(n,180)
        self.assertEqual(rejects,[])
        self.assertIn('SPOOFING_GAS',set(frame.y_multiclass))
        self.assertEqual(frame.y_binary.sum(),120)

    def test_unknown_label_rejected(self):
        p=self.write_csv([FEATURES+['label','category','specific_class'],[1,0,0,0,0,0,0,0,0,'mystery','DOS','DOS']])
        r=[]; frame,n=load_cic(p,r)
        self.assertTrue(frame.empty); self.assertEqual(len(r),1)

    def test_inconsistent_hierarchy_rejected(self):
        p=self.write_csv([FEATURES+['label','category','specific_class'],[1,0,0,0,0,0,0,0,0,'BENIGN','DOS','DOS']])
        r=[]; load_cic(p,r); self.assertIn('hierarchy',r[0]['reason'])

    def test_car_flag_and_variable_dlc(self):
        p=self.write_csv([[1,'043f',2,'01','ff','R'],[2,'0000',0,'T'],[3,'0100',1,'aa','','','','','','','','T']])
        r=[]; frame,n=load_car(p,'DOS',r)
        self.assertEqual(r,[]); self.assertEqual(n,3)
        self.assertEqual(frame.y_multiclass.tolist(),['BENIGN','DOS','DOS'])
        self.assertEqual(frame.loc[0,'DATA_1'],255)
        self.assertEqual(frame.loc[1,'DATA_0'],0)

    def test_car_bad_flag_is_not_guessed(self):
        p=self.write_csv([[1,'043f',1,'ff','?']]); r=[]
        frame,n=load_car(p,'RPM',r)
        self.assertTrue(frame.empty); self.assertEqual(len(r),1)

    def test_group_isolation_and_reproducibility(self):
        f,_=load_cic(self.cic_fixture(),[])
        f['feature_group']=[fingerprint(x) for x in f[FEATURES].itertuples(index=False,name=None)]
        a=split_groups(f,42); b=split_groups(f,42)
        self.assertTrue(a.split.equals(b.split))
        self.assertEqual(a.groupby('feature_group')['split'].nunique().max(),1)
        for part in ['train','validation','test']:
            self.assertEqual(a[a.split==part].y_multiclass.nunique(),3)

    def test_rare_groups_fail_closed(self):
        f=pd.DataFrame({'feature_group':['one','two'],'y_multiclass':['DOS','DOS'],'y_binary':[1,1]})
        with self.assertRaises(ValueError): split_groups(f,42)

    def test_explicit_rare_policy_reports_missing_validation_support(self):
        f=pd.DataFrame({'feature_group':['one','two'],'y_multiclass':['DOS','DOS'],'y_binary':[1,1]})
        a=split_groups(f,42,'train_test')
        self.assertEqual(set(a.split),{'train','test'})
        self.assertEqual(a.feature_group.nunique(),2)

    def test_singleton_is_training_only(self):
        f=pd.DataFrame({'feature_group':['only'],'y_multiclass':['DOS'],'y_binary':[1]})
        a=split_groups(f,42,'train_test')
        self.assertEqual(a.split.tolist(),['train'])

    def test_end_to_end_and_feature_whitelist(self):
        raw=self.cic_fixture(); manifest=self.root/'manifest.json'
        manifest.write_text(json.dumps({'dataset':'ciciov2024','files':[{'path':raw.name}]}))
        out=self.root/'run'
        audit=prepare(manifest,out)
        self.assertEqual(audit['rows_valid'],180)
        self.assertGreater(audit['ambiguous_input_groups'],0)
        self.assertEqual(list(pd.read_csv(out/'X_train.csv').columns),FEATURES)
        for part in ['train','validation','test']:
            self.assertEqual(len(pd.read_csv(out/f'X_{part}.csv')),len(pd.read_csv(out/f'y_{part}.csv')))
        self.assertFalse(audit['benchmark_completion_claimed'])
        self.assertTrue((out/'class_distribution.png').is_file())
        with self.assertRaises(ValueError): prepare(manifest,out)

    def test_missing_raw_file_stops(self):
        manifest=self.root/'manifest.json'
        manifest.write_text(json.dumps({'dataset':'ciciov2024','files':[{'path':'absent.csv'}]}))
        with self.assertRaises(FileNotFoundError): prepare(manifest,self.root/'run')

    def test_streaming_preserves_every_row(self):
        raw=self.cic_fixture(); manifest=self.root/'manifest.json'
        manifest.write_text(json.dumps({'dataset':'ciciov2024','files':[{'path':raw.name}]}))
        out=self.root/'stream'
        audit=run(manifest,out)
        frames=[pd.read_csv(out/f'{part}.csv.gz') for part in ['train','validation','test']]
        self.assertEqual(sum(map(len,frames)),180)
        self.assertEqual(audit['rows_valid'],180)
        self.assertTrue((out/'SUCCESS.json').exists())
        sets=[set(f.feature_group) for f in frames]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        unique=pd.read_csv(out/'unique_inputs_with_counts.csv.gz')
        self.assertEqual(unique.message_count.sum(),180)

    def test_streaming_invalid_input_has_no_success_marker(self):
        raw=self.write_csv([FEATURES+['label','category','specific_class'],[1,999,0,0,0,0,0,0,0,'BENIGN','BENIGN','BENIGN']])
        m=self.root/'manifest.json'
        m.write_text(json.dumps({'dataset':'ciciov2024','files':[{'path':raw.name}]}))
        out=self.root/'stream'
        with self.assertRaises(ValueError): run(m,out)
        self.assertFalse((out/'SUCCESS.json').exists())

    @unittest.skipUnless(shutil.which('g++'), 'Optional native compiler unavailable')
    def test_native_parser_matches_portable_parser(self):
        executable=self.root/'can_accel'
        source=Path(__file__).resolve().parents[1]/'src/can_accel.cpp'
        subprocess.run(['g++','-O2','-std=c++17',str(source),'-lz','-o',str(executable)],check=True,capture_output=True)
        raw=self.root/'raw';raw.mkdir();out=self.root/'out';out.mkdir()
        expected=0
        for attack,name in zip(['DOS','FUZZY','GEAR','RPM'],['DoS_dataset.csv','Fuzzy_dataset.csv','gear_dataset.csv','RPM_dataset.csv']):
            rows=[[1,'043f',2,'01','ff','R'],[2,'0000',0,'T'],[3,'0100',1,'aa','T']]
            with (raw/name).open('w',newline='') as f:csv.writer(f).writerows(rows)
            portable=list(records(raw/name,'car_hacking',attack))
            self.assertTrue(all(r[-1] is None for r in portable));expected+=len(portable)
        subprocess.run([str(executable),'scan',str(raw),str(out)],check=True,capture_output=True)
        g=pd.read_csv(out/'group_counts.csv')
        self.assertEqual(g[['BENIGN','DOS','FUZZY','GEAR','RPM']].to_numpy().sum(),expected)
        normal=g[g.ID==int('043f',16)].iloc[0]
        self.assertEqual(normal.DATA_1,255);self.assertEqual(normal.BENIGN,4)
        assignments=g[FEATURES+['DLC']].copy();assignments['partition']=0
        assignments['feature_group']=[fingerprint(r) for r in assignments[FEATURES+['DLC']].itertuples(index=False,name=None)]
        assignments.to_csv(out/'assignments.csv',index=False)
        subprocess.run([str(executable),'export',str(raw),str(out)],check=True,capture_output=True)
        exported=pd.read_csv(out/'train.csv.gz')
        expected_rows=[]
        for attack,name in zip(['DOS','FUZZY','GEAR','RPM'],['DoS_dataset.csv','Fuzzy_dataset.csv','gear_dataset.csv','RPM_dataset.csv']):
            for num,values,target,timestamp,error in records(raw/name,'car_hacking',attack):
                expected_rows.append((*values,target))
        actual=list(exported[FEATURES+['DLC','y_multiclass']].itertuples(index=False,name=None))
        self.assertEqual(actual,expected_rows)
        reference_source=source.with_name('reference_accel.cpp')
        reference_exe=self.root/'reference_accel'
        subprocess.run(['g++','-O2','-std=c++17',str(reference_source),'-lz','-o',str(reference_exe)],check=True,capture_output=True)
        subprocess.run([str(reference_exe),'car_hacking',str(raw),str(out)],check=True,capture_output=True)
        reference=pd.read_csv(out/'reference_group_counts.csv')
        assignment=reference_assignment(exported,{'dataset':'car_hacking'})
        for part in ['train','validation','test']:
            for label in ['BENIGN','DOS','FUZZY','GEAR','RPM']:
                expected=int(((exported.y_multiclass==label)&(assignment==part)).sum())
                self.assertEqual(int(reference[label+'_'+part].sum()),expected)



if __name__=='__main__':
    unittest.main(verbosity=2)
