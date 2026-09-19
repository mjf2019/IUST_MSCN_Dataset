# Minimal learned router for CDR-MLC

این نسخه KMeans، سه ویژگی timing، پنجره و سه RF Expert را حفظ می‌کند و فقط
تصمیم Router را با یک RF کوچک اصلاح می‌کند.

## کنترل نشت

هر capture توسعه به ۷۰٪ ابتدایی برای KMeans و Expertهای اولیه و ۳۰٪ انتهایی
برای آموزش Gate تقسیم می‌شود. بهترین Expert برای بخش Gate از پیش‌بینی Expertهایی
ساخته می‌شود که آن رکوردها را در آموزش ندیده‌اند. سپس KMeans ثابت می‌ماند و
Expertها با حفظ شناسه خوشه روی کل development refit می‌شوند.

در inference، Gate از فاصله سه centroid، احتمال کلاس‌های هر سه Expert، confidence،
margin، entropy و مسیر KMeans استفاده می‌کند. اگر confidence Gate کمتر از ۰٫۴۵
باشد، مسیر KMeans حفظ می‌شود.

## اجرا

```powershell
python CDR_MLC/learned_router_sweep.py --fractions 0 0.20 --scenarios 1 2 3 --window 3 --output CDR_MLC/outputs/learned_router_sweep
```

خروجی اصلی `learned_router_summary.csv` شامل Router واقعی، Learned Router،
Oracle Router و RF expert-inputs است. Oracle فقط upper bound است؛ Learned Router
در آزمون هیچ برچسب واقعی application یا congestion را نمی‌بیند.

کد طبق درخواست کاربر اجرا نشده است.
