# Adaptive CDR-MLC برای سناریوهای ۱ تا ۳

این اجرا با آزمایش mixed-level قبلی تفاوت دارد:

| سناریو | انتخاب و آموزش | آزمون کاملاً دیده‌نشده |
|---|---|---|
| ۱ | Low | Medium |
| ۲ | Low | High |
| ۳ | Medium | High |

در سناریوهای ۱ و ۲، هیچ رکورد Medium یا High در انتخاب ویژگی، انتخاب پنجره،
StandardScaler، MBK، Gate یا Expert شرکت نمی‌کند. سناریوی ۳ نیز High را تا
ارزیابی نهایی نمی‌بیند.

## چرا معیار انتخاب ARI نیست؟

دادهٔ مبدأ هر سناریو فقط یک سطح ازدحام دارد. بنابراین محاسبهٔ ARI سه‌سطحی در
Low یا Medium به‌تنهایی از نظر مفهومی ممکن نیست. اگر برای انتخاب تنظیمات از
سطح مقصد استفاده شود، test وارد model selection شده و نشت ایجاد می‌شود.

در این کد:

- هر capture مبدأ به ۷۵٪ train و ۲۵٪ validation زمانی تقسیم می‌شود.
- کاندیداها و ترکیب «سه ویژگی + پنجره» فقط با مبدأ بررسی می‌شوند.
- معیار اصلی، Silhouette خوشه‌های validation مبدأ است.
- پایداری بین seedها با AMI و حداقل سهم خوشه به‌عنوان کنترل گزارش می‌شود.
- بعد از انتخاب، مدل روی کل مبدأ refit و یک بار روی سطح مقصد ارزیابی می‌شود.

Silhouette بالا ثابت نمی‌کند خوشه‌ها Low/Medium/High واقعی هستند. این روش صرفاً
یک model-selection بدون نشت برای سناریوی cross-congestion فراهم می‌کند. نتایج
آن باید کنار CDR-MLC ثابت و RF baseline روی همان رکوردها گزارش شوند.

## اجرای هر سه سناریو

اجرای سریع با یک seed:

```bash
python CDR_MLC/adaptive_cdr_mlc_scenarios.py --scenarios 1 2 3 --windows 3 10 20 --ranking-top-k 5 --selection-seeds 42 --gating soft
```

اجرای بررسی پایداری:

```bash
python CDR_MLC/adaptive_cdr_mlc_scenarios.py --scenarios 1 2 3 --windows 3 10 20 --ranking-top-k 5 --selection-seeds 21 42 84 --gating soft
```

برای Hard Gating و خروجی جدا:

```bash
python CDR_MLC/adaptive_cdr_mlc_scenarios.py --scenarios 1 2 3 --windows 3 10 20 --ranking-top-k 5 --selection-seeds 42 --gating hard --output CDR_MLC/outputs/adaptive_scenarios_hard
```

سناریوهای ۱ و ۲ یک source مشترک (`Low`) دارند؛ انتخاب تنظیمات و آموزش مدل Low
فقط یک بار انجام می‌شود و همان مدل بدون تغییر روی Medium و High ارزیابی می‌شود.

## خروجی‌ها

```text
CDR_MLC/outputs/adaptive_scenarios/
├── scenario_summary.csv
├── source_low/
│   ├── selected_source_configurations.json
│   ├── source_feature_ranking.csv
│   └── source_selection_trials.csv
├── source_medium/
│   └── ...
├── scenario_1_low_to_medium/
│   ├── target_metrics.json
│   ├── target_predictions.csv
│   └── run_manifest.json
├── scenario_2_low_to_high/
│   └── ...
└── scenario_3_medium_to_high/
    └── ...
```

برای تحلیل، `scenario_summary.csv`، دو فایل تنظیمات منتخب و سه فایل
`target_metrics.json` را ارسال کنید. در صورت نیاز به تحلیل خطا، سه فایل
`target_predictions.csv` را نیز بفرستید.

## نکات اجرایی

- اجرای دوباره در همان `--output` فایل‌های قبلی را جایگزین می‌کند.
- برای مقایسه Soft و Hard حتماً مسیر خروجی جدا بدهید.
- `--save-model` مدل مبدأ را ذخیره می‌کند؛ فایل مدل برای تحلیل نتایج لازم نیست.
- اجرای کامل می‌تواند زمان‌بر باشد. ابتدا دستور سریع را اجرا کنید.
- کد طبق درخواست کاربر در محیط توسعه اجرا یا تست نشده است.
