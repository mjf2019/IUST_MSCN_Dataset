# Adaptive CDR-MLC: انتخاب پویای ویژگی و پنجره

فایل `adaptive_cdr_mlc.py` یک نسخهٔ پژوهشی توسعه‌یافته از CDR-MLC است. نسخهٔ
ثابت مقاله در `CDR-MLC.ipynb` تغییر نکرده است.

## رفتار الگوریتم

برای هر برچسب `HTTP/SFTP/SMTP/SSH/Video` به‌صورت جداگانه:

1. ویژگی‌های کاندید فقط با بخش train و معیار Kruskal–Wallis رتبه‌بندی می‌شوند.
2. تمام ترکیب‌های سه‌تایی از `ranking_top_k` ویژگی اول، همراه اندازه‌های پنجرهٔ
   خواسته‌شده، روی validation مقایسه می‌شوند.
3. معیار انتخاب میانگین ARI روی seedهای تعیین‌شده است. در تساوی، تنظیم پایدارتر
   و سپس پنجرهٔ کوچک‌تر انتخاب می‌شود.
4. پس از freeze شدن تنظیمات، برای هر برچسب یک StandardScaler و MBK سه‌خوشه‌ای
   روی رکوردهای همان برچسب در train+validation آموزش داده می‌شود.
5. برای هر روتر سه متخصص RF چندکلاسه آموزش داده می‌شود؛ در مجموع ۱۵ متخصص.

در inference برچسب واقعی خوانده نمی‌شود. یک RF اولیه احتمال برچسب‌ها را تولید
می‌کند. در حالت `soft`، خروجی بانک‌های مخصوص برچسب با این احتمال‌ها ترکیب می‌شود:

```text
final_probability = Σ gate_probability(label) × expert_probability(label-router)
```

در حالت `hard` فقط بانک مربوط به محتمل‌ترین برچسب وزن ۱ می‌گیرد. حالت soft
پیش‌فرض است، چون خطای مدل اولیه را کمتر به یک مسیر اشتباه قطعی تبدیل می‌کند.

## جلوگیری از leakage

- بخش هر capture به ترتیب زمان: ۶۰٪ train، ۲۰٪ validation و ۲۰٪ test رزروشده.
- رتبه‌بندی و انتخاب ویژگی/پنجره فقط با train و validation انجام می‌شود.
- test در انتخاب تنظیمات، scaler، MBK، gate یا متخصص‌ها شرکت نمی‌کند.
- `traffic_label` و `congestion_level` ورودی inference نیستند.
- مدل gate از ویژگی‌های انتخاب‌شده برای routing و تمام متادیتاها محروم است.
- خروجی test فقط یک بار پس از نهایی‌شدن مدل محاسبه می‌شود.

این جداسازی در سطح رکوردهای زمانی هر capture است، نه capture مستقل. چون برای هر
ترکیب application/level فقط یک capture موجود است، نتیجهٔ test هنوز می‌تواند به
شرایط همان run وابسته باشد. برای ادعای تعمیم، capture مستقل دوم لازم است.

## اجرای پیشنهادی

از ریشهٔ مخزن:

```bash
python -m pip install -r CDR_MLC/requirements-adaptive.txt
python CDR_MLC/adaptive_cdr_mlc.py \
  --windows 3 5 10 20 50 100 \
  --ranking-top-k 6 \
  --selection-seeds 21 42 84 \
  --gating soft
```

برای اجرای سریع اولیه:

```bash
python CDR_MLC/adaptive_cdr_mlc.py \
  --windows 3 10 20 \
  --ranking-top-k 5 \
  --selection-seeds 42
```

هزینهٔ جست‌وجوی هر برچسب تقریباً برابر است با:

```text
Combination(ranking_top_k, 3) × number_of_windows × number_of_seeds
```

بنابراین تنظیم پیش‌فرض `top_k=6`، شش پنجره و سه seed برای هر برچسب ۳۶۰ fit
انتخابی و در مجموع برای پنج برچسب ۱۸۰۰ fit انجام می‌دهد. برای بررسی اولیه از
دستور سریع استفاده کنید و پس از اطمینان، جست‌وجوی کامل را اجرا کنید.

## ویژگی‌های کاندید

لیست پیش‌فرض در `DEFAULT_CANDIDATES` قرار دارد و از CLI نیز قابل تغییر است:

```bash
python CDR_MLC/adaptive_cdr_mlc.py \
  --candidates TcpRtt SynAck AckDat SrcLoad DstLoad Load SrcRate DstRate Rate \
  --windows 3 10 20 50
```

حداقل سه ویژگی عددی معتبر لازم است. شناسه‌ها، آدرس‌ها، زمان، label، level و
`IdleTime` اجازهٔ ورود به routing را ندارند.

## خروجی‌هایی که برای تحلیل بفرستید

مسیر پیش‌فرض: `CDR_MLC/outputs/adaptive_cdr_mlc/`

- `selected_configurations.json`: سه ویژگی و پنجرهٔ منتخب هر برچسب.
- `selection_trials.csv`: نتیجهٔ تمام ترکیب‌ها و seedها.
- `training_feature_ranking.csv`: رتبه‌بندی train-only.
- `reserved_metrics.json`: نتیجهٔ نهایی بخش رزروشده.
- `reserved_predictions.csv`: احتمال gate، خوشهٔ هر بانک و پیش‌بینی هر ردیف.
- `run_manifest.json`: تنظیمات کامل اجرا.
- `adaptive_cdr_mlc.joblib`: مدل نهایی؛ فقط از فایل مورداعتماد load شود.

برای تحلیل بعدی، پنج فایل متنی/CSV اول را ارسال کنید؛ فایل مدل لازم نیست.

## وضعیت اعتبارسنجی

این کد طبق درخواست کاربر در محیط توسعه اجرا یا تست نشده است. ابتدا دستور سریع را
روی سیستم خود اجرا کنید. اگر خطا رخ داد، traceback کامل و نسخه‌های Python، pandas،
scipy و scikit-learn را ارسال کنید.
