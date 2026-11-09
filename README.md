# رصد | CAN Intrusion Detection Dashboard

واجهة React عربية مع FastAPI متصلة بالنماذج الثمانية المدربة في مشروع كشف التسلل في شبكات السيارات.

## التشغيل المحلي

يتطلب Python 3.11 وNode.js 22.

```bash
python -m venv .venv
```

على Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements-api.txt
uvicorn api:app --reload --port 8000
```

في نافذة طرفية ثانية:

```powershell
cd frontend
npm install --no-audit --no-fund
npm run dev
```

افتح http://localhost:5173. توثيق API متاح على http://localhost:8000/docs.

## تنسيق ملفات CSV

- CICIoV2024: `ID,DATA_0,DATA_1,DATA_2,DATA_3,DATA_4,DATA_5,DATA_6,DATA_7`
- Car-Hacking: الحقول السابقة إضافة إلى `DLC`.
- تقبل `ID` العشري أو الشكل الست عشري المسبوق بـ `0x`، وحقول DATA كأعداد صحيحة من 0 إلى 255، وDLC من 0 إلى 8.
- ملفات الاختبار المناسبة موجودة داخل `tests/fixtures/`.
- الحد لكل طلب 12 MB و50,000 سجل. لا يتطلب الكشف عبر الواجهة تحميل مجموعة البيانات الكاملة.

## الاختبارات

```bash
python -m pytest tests/test_dashboard_api.py -q
cd frontend && npm run build
```

## GitHub Pages

1. ارفع مجلد المشروع إلى مستودع GitHub، واجعل الفرع الرئيسي `main`.
2. انشر واجهة API على خادم Python يدعم تشغيل `uvicorn api:app --host 0.0.0.0 --port 8000`، وتأكّد من توافق إصدارات المكتبات مع ملفات النماذج.
3. في إعدادات مستودع GitHub أضف Repository variable باسم `VITE_API_URL` ويكون رابط API المنشور باستخدام HTTPS.
4. على استضافة API ضع `CORS_ORIGINS` مساويًا لرابط GitHub Pages الخاص بالمستودع، مثل `https://USERNAME.github.io`.
5. فعّل Pages باستخدام GitHub Actions، ثم شغّل workflow المسمى Deploy frontend to GitHub Pages.

GitHub Pages يستضيف الواجهة فقط، ولا يشغّل خادم FastAPI. ينبغي عدم استخدام النظام كوسيلة كشف تشغيلية ذات حساسية للسلامة دون معايرة وتحقق مستقل من بيانات حقيقية وجديدة.

## ملفات المشروع

- `api.py`: نقاط API وتحميل النماذج وتشغيل الاستدلال.
- `frontend/`: واجهة React كاملة بالعربية.
- `models/`: النماذج الفعلية والبيانات الوصفية.
- `results/`: نتائج التجارب التاريخية.
- `src/`: الشيفرة العلمية الأصلية.
- `README_AR.md` و`DATA_REQUIREMENTS_AR.md`: وصف تجارب البحث ومتطلبات البيانات.

## توافق النماذج

تم حفظ نماذج Random Forest باستخدام scikit-learn 1.9.1؛ استخدم بيئة `requirements-api.txt` نفسها لتجنب تحذيرات فك الحفظ. قد تعمل مع نسخ أخرى لكن النتائج غير مضمونة. لا ترفع بيانات أشخاص أو تسجيلات مركبات خاصة إلى مستودع عام.
