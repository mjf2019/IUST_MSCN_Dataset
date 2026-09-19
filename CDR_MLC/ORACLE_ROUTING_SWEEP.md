# Oracle routing upper bound

این آزمایش مشخص می‌کند اگر Router برای هر نمونه همیشه بهترین Expert را انتخاب کند،
حداکثر عملکرد قابل دستیابی CDR-MLC چقدر است. برای هر نمونه آزمون، خروجی هر سه
Expert محاسبه می‌شود. Oracle با مشاهده برچسب واقعی application، Expert درست را
انتخاب می‌کند؛ اگر هیچ Expert پاسخ درست نداشته باشد، خطا حفظ می‌شود.

این نتیجه عمداً دارای inference leakage است، قابل استقرار نیست و فقط باید به‌عنوان
upper-bound و تحلیل ضعف Router استفاده شود.

## اجرای هم‌زمان حالت صفر و بیست درصد

```powershell
python CDR_MLC/oracle_cdr_mlc_sweep.py --fractions 0 0.20 --scenarios 1 2 3 --window 3 --output CDR_MLC/outputs/oracle_routing_sweep
```

در حالت ۰٪، هیچ prefix از سطح غیرمبدأ وارد train نمی‌شود. در حالت ۲۰٪، prefix
زمانی Medium و High مطابق پروتکل calibration وارد development می‌شود و tail
هشتاددرصدی برای آزمون می‌ماند.

برای هر سناریو سه خروجی مقایسه می‌شوند:

- `CDR_MLC_actual_router`: مسیریابی واقعی KMeans؛
- `CDR_MLC_oracle_router`: بهترین انتخاب Expert با برچسب واقعی؛
- `RF_expert_inputs`: RF بدون سه ویژگی timing.

فایل اصلی `oracle_summary.csv` است. اختلاف Oracle با Actual ظرفیت بهبود Router را
نشان می‌دهد. اگر Oracle نیز از RF ضعیف‌تر باشد، مشکل اصلی Router نیست و خود
Expertها یا ویژگی‌های classifier محدودکننده‌اند.

`oracle_predictions.csv` شامل پیش‌بینی هر سه Expert، مسیر واقعی و Oracle و تعداد
Expertهای درست برای هر نمونه است. کد طبق درخواست کاربر در محیط توسعه اجرا نشده است.
