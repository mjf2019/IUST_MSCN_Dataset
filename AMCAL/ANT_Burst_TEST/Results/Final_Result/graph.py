import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# خواندن فایل CSV و نادیده گرفتن سطر اول (هدر)
data = pd.read_csv("C:\\Users\\lenovo\\Desktop\\Final Results\\Final_Result\\ARFTotalAccuracyResult.csv", skiprows=1, header=None)

# استخراج داده‌ها
sizes = data.iloc[:, 0]  # ستون سایزها
columns = data.columns[1:]  # ستون‌های داده به جز سایز

# تعیین نقاط محور افقی (0 تا 1073 با فاصله‌ی 100)
x_ticks = np.arange(0, 100, 1)

# تنظیم گراف
plt.figure(figsize=(12, 8))

# انتخاب رنگ‌ها برای نمودارها
colors = [
    'blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray'
]


# رسم نمودار برای هر سایز
for i, size in enumerate(sizes):
    accuracy = data.iloc[i, 1:]  # داده‌های مربوط به هر سایز
    plt.plot(x_ticks, accuracy[:len(x_ticks)], label=f'Size {size}', color=colors[i % len(colors)], marker='o')

# # رسم نمودار برای هر سایز
# for i, size in enumerate(sizes):
#     if size in (0,20,17,15,3):
#         accuracy = data.iloc[i, 1:]  # داده‌های مربوط به هر سایز
#         plt.plot(x_ticks, accuracy[:len(x_ticks)], label=f'Size {size}', color=colors[i % len(colors)], marker='o')

# تنظیمات گراف
plt.xlabel('Data Count', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Accuracy vs. Data Count for Different Sizes', fontsize=16)
plt.xticks(x_ticks, fontsize=5)
plt.yticks(fontsize=12)
plt.legend(title='Sizes', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# نمایش گراف
plt.show()
