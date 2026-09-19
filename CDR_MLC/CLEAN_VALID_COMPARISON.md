# ساخت Clean-Valid و مقایسه سه مدل

این pipeline ابتدا از ۱۵ فایل بازبینی‌شده یک نسخه‌ی تمیز و قابل بازتولید می‌سازد
و سپس سه سناریوی cross-congestion را برای مدل‌های زیر روی رکوردهای آزمون یکسان
مقایسه می‌کند:

- CDR-MLC ثابت مقاله با `TcpRtt`، `SynAck` و `AckDat` و پنجره ۳؛
- Random Forest روی تمام ویژگی‌های معتبر Clean-Valid؛
- Random Forest روی ورودی expertها بدون سه ویژگی router، برای ablation؛
- Adaptive CDR-MLC با انتخاب ویژگی و پنجره فقط از سطح مبدأ.

## مرحله ۱: ساخت مجموعه‌داده Clean-Valid

```powershell
python CDR_MLC/build_clean_valid.py
```

خروجی در مسیر زیر ساخته می‌شود:

```text
CDR_MLC/DATASETS/CDR-MLC/Clean_Valid/
```

فایل‌های اصلی تغییر نمی‌کنند. برای بازسازی صریح خروجی موجود:

```powershell
python CDR_MLC/build_clean_valid.py --overwrite
```

در Clean-Valid موارد زیر حذف می‌شوند:

- `IdleTime` به‌عنوان fingerprint مربوط به capture؛
- ستون‌های کاملاً missing یا ثابت؛
- `SrcGap` و `DstGap`؛
- aliasهای تکراری duration و نگه‌داشتن فقط `Dur`؛
- label، Cause و export metadata غیرضروری.

آدرس، پورت، زمان و پروتکل فقط برای کنترل جهت سرویس در فایل نگه داشته می‌شوند و
در هیچ مدل وارد نمی‌شوند. manifest شامل schema، علت حذف‌ها، تعداد رکورد و SHA-256
ورودی و خروجی است.

## مرحله ۲: مقایسه هر سه سناریو

```powershell
python CDR_MLC/compare_clean_valid.py --scenarios 1 2 3 --fixed-window 3 --windows 3 10 20 --ranking-top-k 5 --selection-seeds 42 --gating soft
```

تعریف سناریوها:

| سناریو | آموزش | آزمون |
|---|---|---|
| ۱ | Low | Medium |
| ۲ | Low | High |
| ۳ | Medium | High |

مدل Low برای سناریوهای ۱ و ۲ فقط یک بار fit می‌شود. سطح مقصد نه در آموزش و نه
در انتخاب تنظیمات Adaptive استفاده نمی‌شود. در هر سناریو، معیارهای هر چهار خروجی
روی intersection کاملاً یکسان رکوردهای قابل پیش‌بینی محاسبه می‌شوند.

## خروجی اصلی

```text
CDR_MLC/outputs/clean_valid_comparison/comparison_summary.csv
```

خروجی‌های تکمیلی شامل موارد زیر است:

- `run_manifest.json`؛
- تنظیمات انتخاب‌شده Adaptive برای sourceهای Low و Medium؛
- `metrics.json` برای هر سناریو؛
- `predictions_common_rows.csv` برای تحلیل خطای جفت‌شده؛
- شمارش کلاس‌ها در سه cluster مدل ثابت؛
- گزارش ویژگی‌های استفاده‌شده در هر مدل.

برای بررسی پایداری Adaptive می‌توان سه seed انتخاب کرد:

```powershell
python CDR_MLC/compare_clean_valid.py --scenarios 1 2 3 --fixed-window 3 --windows 3 10 20 --ranking-top-k 5 --selection-seeds 21 42 84 --gating soft --output CDR_MLC/outputs/clean_valid_comparison_3seeds
```

## تفسیر علمی

`RF_all_clean_valid` baseline اصلی است. `RF_expert_inputs` فقط نشان می‌دهد حذف
ویژگی‌های timing مورد استفاده router چه اثری دارد و جای baseline اصلی را نمی‌گیرد.
سیاست پاک‌سازی با مشاهده کل dataset تعیین شده است؛ بنابراین این آزمایش یک ablation
شفاف برای revision است و مجموعه آزمون مستقل و کاملاً pristine ایجاد نمی‌کند.

کد طبق درخواست کاربر در محیط توسعه اجرا یا تست نشده است.
