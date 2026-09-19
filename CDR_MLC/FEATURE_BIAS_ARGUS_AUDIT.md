# تحلیل سوگیری ویژگی‌ها و مشکلات Argus روی کل مجموعه‌داده

این تحلیل هر ۱۵ capture شامل پنج application و سه سطح ازدحام را بررسی می‌کند.
هدف، پیدا کردن ویژگی‌های نامعتبر، شناسه‌ای، تکراری، وابسته به capture و ویژگی‌هایی
است که نوع application را بسیار قوی تشخیص می‌دهند ولی نسبت به ازدحام حساس نیستند.

اسکریپت فایل‌های `.flow` را تغییر یا حذف نمی‌کند. خروجی آن یک پیشنهاد برای بازبینی
ویژگی‌هاست؛ ویژگی صرفاً به‌دلیل قدرت زیاد برای تشخیص application حذف نمی‌شود.

## اجرا

```powershell
python CDR_MLC/feature_bias_argus_audit.py
```

خروجی پیش‌فرض:

```text
CDR_MLC/outputs/feature_bias_argus_audit/
├── REPORT.md
├── input_audit.csv
├── numeric_feature_audit.csv
├── categorical_feature_audit.csv
├── per_capture_feature_quality.csv
├── near_duplicate_features.csv
└── recommended_feature_policy.json
```

## معیارها

- نرخ missing، صفر، مقدار منفی، cardinality و ثابت‌بودن هر ویژگی؛
- اثر application با Kruskal–Wallis epsilon-squared؛
- اثر ازدحام درون هر application و میانگین وزن‌دار آن؛
- یکسان‌بودن جهت تغییر از Low به High میان applicationها؛
- NMI ویژگی با application، congestion و capture؛
- همبستگی Spearman برای تشخیص ویژگی‌های تقریباً تکراری؛
- بررسی کیفیت هر ویژگی به‌صورت جداگانه در هر capture.

## قواعد تصمیم

موارد زیر مستقیماً از ورودی مدل کنار گذاشته می‌شوند:

- label و metadata؛
- آدرس‌ها، پورت‌ها، timestamp، نام capture و شماره ردیف؛
- ستون‌های خالی، ثابت یا دارای حداقل ۹۵٪ missing.

موارد زیر فقط برای بررسی علامت‌گذاری می‌شوند و خودکار حذف نمی‌شوند:

- ویژگی با قدرت زیاد برای تشخیص application و حساسیت کم به ازدحام؛
- cardinality بسیار زیاد؛
- مقادیر منفی در فیلدهای ذاتاً نامنفی Argus؛
- وابستگی بسیار زیاد به capture؛
- زوج ویژگی‌های تقریباً تکراری.

## محدودیت مهم

برای هر ترکیب application و congestion فقط یک capture وجود دارد. در نتیجه، اثر
واقعی ازدحام، شرایط خاص اجرای آزمایش و artifact مربوط به Argus کاملاً از یکدیگر
قابل تفکیک نیستند. برای اثبات قطعی artifact بودن یک ویژگی باید capture مستقل
دیگری از همان application و همان سطح ازدحام جمع‌آوری شود.

همچنین چون این تحلیل کل داده را می‌بیند، امتیازهای وابسته به label فقط برای
پاک‌سازی و گزارش dataset هستند. استفاده از آن‌ها برای انتخاب ویژگی و سپس ادعای
آزمون کاملاً untouched روی همین داده مجاز نیست. مقایسه بعدی RF و CDR-MLC باید
هم baseline کامل و هم ablation مبتنی بر سیاست پاک‌سازی ثابت را گزارش کند.

## فایل‌هایی که برای تحلیل نتیجه لازم‌اند

پس از اجرا این فایل‌ها را ارسال کنید:

1. `REPORT.md`
2. `recommended_feature_policy.json`
3. `numeric_feature_audit.csv`
4. `categorical_feature_audit.csv`
5. `near_duplicate_features.csv`
6. `per_capture_feature_quality.csv`

