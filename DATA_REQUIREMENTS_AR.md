# متطلبات البيانات الخارجية

لا تتضمن الحزمة المضغوطة ملفات CSV الخام أو ملفات `runs/*.csv.gz` المجهزة بسبب حجمها الكبير وارتباط إعادة توزيعها بشروط ناشري مجموعتي البيانات. النتائج العلمية المرفقة لم تُنتج من بيانات صناعية أو من ملفات الاختبار الصغيرة.

## البيانات التي استُخدمت فعليًا

- CICIoV2024: ستة ملفات CSV، بإجمالي 1,408,219 رسالة.
- Car-Hacking: أربعة ملفات CSV، بإجمالي 16,569,475 رسالة.
- تطابقت بصمات SHA-256 للملفات العشرة مع القيم المسجلة في `metadata/datasets/*/audit.json`.

## استعادة البيانات وتشغيل المشروع

نفّذي الأوامر الآتية من جذر المشروع:

```powershell
python src/download_sources.py ciciov2024 --root data/raw
python src/download_sources.py car_hacking --root data/raw
python src/prepare_streaming.py --manifest config/ciciov2024.json --output runs/ciciov2024 --seed 42
python src/prepare_streaming.py --manifest config/car_hacking.json --output runs/car_hacking --seed 42
python src/verify_prepared.py runs/ciciov2024
python src/verify_prepared.py runs/car_hacking
python run_stage2.py --skip-training
python run_xai.py
python -m pytest -q
```

تُحفظ روابط المصدر المستخدمة في `src/download_sources.py`، وتوجد بيانات المصدر والبصمات والتعدادات في `stage1/evidence/acquisition.json` و`metadata/datasets`.

النماذج الثمانية والنتائج الرقمية والرسومات وإعدادات XAI وسجلات التنفيذ مضمّنة في الحزمة، ولذلك يمكن فحص النتائج الحالية من دون تنزيل البيانات. يلزم تنزيل البيانات وتجهيزها فقط لإعادة حساب التنبؤات والتفسيرات من البداية.
