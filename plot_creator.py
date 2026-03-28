import pandas as pd
import matplotlib.pyplot as plt
import os

base_dir = '/media/user/Data/IndustrialSafety/Models/Hard Hats.v6-resized6'

models = ['yolov8n', 'yolov8m', 'yolov8l', 'yolo11n']

map_column = 'metrics/mAP50-95(B)' 

plt.figure(figsize=(12, 7))

data_plotted = False

for model in models:
    csv_path = os.path.join(base_dir, model, 'train', 'results.csv')
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            df.columns = df.columns.str.strip()
            
            if 'epoch' in df.columns and map_column in df.columns:
                plt.plot(df['epoch'], df[map_column], label=model, linewidth=2)
                data_plotted = True
            else:
                print(f"[Внимание] В файле {csv_path} нет колонки '{map_column}'.")
                print(f"Доступные колонки: {df.columns.tolist()}")
                
        except Exception as e:
            print(f"[Ошибка] Не удалось прочитать {csv_path}: {e}")
    else:
        print(f"[Внимание] Файл не найден: {csv_path}")

if data_plotted:
    plt.title('Зависимость mAP от эпохи обучения для разных моделей YOLO', fontsize=16, pad=15)
    plt.xlabel('Эпоха (Epoch)', fontsize=14)
    plt.ylabel(f'Точность ({map_column})', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Модели', fontsize=12, title_fontsize=13)
    
    plt.tight_layout()
    
    save_path = os.path.join(base_dir, 'map_comparison_plot.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ График успешно сохранен: {save_path}")
    
    plt.show()
else:
    print("\n❌ Не удалось найти данные для построения графика. Проверьте пути и содержимое CSV-файлов.")