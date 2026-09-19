# Selective correction router

این نسخه تصمیم KMeans را پیش‌فرض نگه می‌دارد. ابتدا یک Error Detector احتمال
اشتباه‌بودن مسیر KMeans را تخمین می‌زند. فقط در صورت اطمینان بالا و برتری confidence
یک Expert جایگزین، مسیر تغییر می‌کند.

هر capture توسعه به ۶۰٪ آموزش Expert اولیه، ۲۰٪ آموزش correction و ۲۰٪ انتخاب
threshold تقسیم می‌شود. Scaler و KMeans بدون استفاده از label روی کل source fit
می‌شوند تا baseline اصلی تغییر نکند. Expertهای نهایی نیز روی کل development fit
می‌شوند.

## اجرا

```powershell
python CDR_MLC/selective_router_sweep.py --fractions 0 0.20 --scenarios 1 2 3 --window 3 --output CDR_MLC/outputs/selective_router_sweep
```

خروجی اصلی `selective_router_summary.csv`، audit انتخاب threshold و predictionهای
سطح رکورد است. Oracle فقط upper bound است. کد اجرا نشده است.
