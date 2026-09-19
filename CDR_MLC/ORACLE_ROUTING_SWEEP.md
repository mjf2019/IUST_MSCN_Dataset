# Oracle congestion-level experts

داده آموزشی به سه بخش سطحی تقسیم می‌شود:

- مدل Low با ۱۰۰٪ داده Low؛
- مدل Medium با prefix زمانی داده Medium؛
- مدل High با prefix زمانی داده High.

Router فرضی سطح واقعی ازدحام را می‌داند. tail آزمون Medium فقط به مدل Medium و
tail آزمون High فقط به مدل High فرستاده می‌شود. این Router قابل استقرار نیست و
فقط upper bound اثر مسیریابی صحیح را اندازه می‌گیرد.

## اجرای صفر و بیست درصد

```powershell
python CDR_MLC/oracle_cdr_mlc_sweep.py --fractions 0 0.20 --output CDR_MLC/outputs/oracle_level_experts
```

در حالت صفر درصد، مدل Medium و High داده‌ای ندارند و
`Oracle_Level_Expert` صریحاً به مدل Low fallback می‌کند. در حالت بیست درصد،
مدل Medium و High هرکدام با ۲۰٪ ابتدایی captureهای سطح خود آموزش می‌بینند و
۸۰٪ tail سطح خود را طبقه‌بندی می‌کنند.

سه روش گزارش می‌شود:

- `Oracle_Level_Expert`: مدل متناظر با سطح واقعی؛
- `Low_Expert`: مدل آموزش‌دیده با کل Low؛
- `Pooled_RF`: یک مدل واحد روی Low و prefixهای calibration.

خروجی اصلی `oracle_level_expert_summary.csv` است. ویژگی‌های `TcpRtt`،
`SynAck` و `AckDat` از classifierهای سطحی حذف شده‌اند تا ورودی با Expertهای
CDR-MLC یکسان باشد. این آزمایش clustering واقعی را اجرا نمی‌کند؛ حالت ایده‌آل
«خوشه دقیقاً برابر سطح ازدحام» را شبیه‌سازی می‌کند.

کد طبق درخواست کاربر اجرا نشده است.
