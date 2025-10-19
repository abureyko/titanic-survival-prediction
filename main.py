import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('train.csv')
test_df = test_df = pd.read_csv('test.csv')



# --------------------------- Подготовка данных ------------------------------

# print('=== АНАЛИЗ ВЫЖИВАЕМОСТИ ===')
# print('Распределение выживших:')
# print(train_df['Survived'].value_counts())
# print('\nДоля выживших:', train_df['Survived'].mean())

# print("\n=== ВЫЖИВАЕМОСТЬ ПО ПОЛУ ===")
# print(train_df.groupby('Sex')['Survived'].mean())

# print("\n=== ВЫЖИВАЕМОСТЬ ПО КЛАССУ ===")
# print(train_df.groupby('Pclass')['Survived'].mean())

# print("\n=== ВЫЖИВАЕМОСТЬ ПО ПОРТУ ПОСАДКИ ===")
# print(train_df.groupby('Embarked')['Survived'].mean())

# Заполняем пропуски
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna('C')

# Преобразуем категориальные признаки в числа
train_df['Sex'] = train_df['Sex'].map({'male' : 0, 'female' : 1})
train_df['Embarked'] = train_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Создаем новые признаки
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)

# Создаём возрастные категории
def get_age_category(age):
    if age < 12: return 0  # child
    elif age < 18: return 1  # teen  
    elif age < 30: return 2  # young adult
    elif age < 50: return 3  # adult
    else: return 4  # senior

train_df['AgeCategory'] = train_df['Age'].apply(get_age_category)
# print()
# print(train_df.groupby('AgeCategory')['Survived'].mean())



# -------------------------- Обучение -----------------------------

# Выбираем фичи для модели
features = ['Pclass', 'Sex', 'AgeCategory', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
X = train_df[features]
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Тренировочная выборка: {X_train.shape[0]} примеров")
print(f"Валидационная выборка: {X_val.shape[0]} примеров")

model = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Предсказания и оценка
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"\n=== РЕЗУЛЬТАТЫ ===")
print(f"Accuracy: {accuracy:.3f}")
print(f"Точно предсказано: {accuracy_score(y_val, y_pred, normalize=False)} из {len(y_val)}")


for C_value in [0.01, 0.1, 1.0, 10.0]:
    model_test = LogisticRegression(C=C_value, penalty='l2', max_iter=1000, random_state=42)
    model_test.fit(X_train, y_train)
    acc = model_test.score(X_val, y_val)
    print(f"C={C_value}: Accuracy = {acc:.3f}")
