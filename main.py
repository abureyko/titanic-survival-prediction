import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Загрузка данных
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("=" * 60)
print("АНАЛИЗ ДАННЫХ ТИТАНИКА")
print("=" * 60)

# --------------------------- ВИЗУАЛИЗАЦИЯ ----------------------------
plt.figure(figsize=(15, 10))

# 1. Выживаемость по классу и полу
plt.subplot(2, 3, 1)
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_df)
plt.title('Выживаемость по классу и полу')

# 2. Распределение возраста выживших/погибших
plt.subplot(2, 3, 2)
sns.histplot(data=train_df, x='Age', hue='Survived', bins=30, alpha=0.6)
plt.title('Распределение возраста')

# 3. Выживаемость по порту посадки
plt.subplot(2, 3, 3)
sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.title('Выживаемость по порту посадки')

# 4. Корреляционная матрица
plt.subplot(2, 3, 4)
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived']
sns.heatmap(train_df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица')

# 5. Распределение стоимости билета
plt.subplot(2, 3, 5)
sns.histplot(data=train_df, x='Fare', hue='Survived', bins=30, alpha=0.6)
plt.title('Распределение стоимости билета')

plt.tight_layout()
plt.savefig('titanic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# --------------------------- ПОДГОТОВКА ФИЧ -----------------------------
# Функция для извлечения титула из имени 
def extract_title(name):
    try:
        return name.split(', ')[1].split('.')[0]
    except:
        return 'Unknown'

train_df['Title'] = train_df['Name'].apply(extract_title)
test_df['Title'] = test_df['Name'].apply(extract_title)

# Группируем титулы
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
train_df['Title'] = train_df['Title'].replace(rare_titles, 'Rare')

test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')
test_df['Title'] = test_df['Title'].replace(rare_titles, 'Rare')

# Преобразуем категориальные признаки для title
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
train_df['Title'] = train_df['Title'].map(title_mapping).fillna(0)
test_df['Title'] = test_df['Title'].map(title_mapping).fillna(0)

# Заполняем пропуски
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna('C')

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna('C')

# Преобразуем категориальные признаки для sex и embarked
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df['Embarked'] = test_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Создаем фичи
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
train_df['IsLargeFamily'] = (train_df['FamilySize'] > 4).astype(int)

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)
test_df['IsLargeFamily'] = (test_df['FamilySize'] > 4).astype(int)


# --------------------------- ФИНАЛЬНЫЙ ВЫБОР ФИЧ --------------------------------
feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                   'Title', 'FamilySize', 'IsAlone', 'IsLargeFamily']

print(f"📊 Используемые фичи ({len(feature_columns)}): {feature_columns}")

X = train_df[feature_columns]
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --------------------------- СРАВНЕНИЕ МОДЕЛЕЙ -----------------------------
models = {
    'Logistic Regression': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

print("\n" + "="*50)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*50)

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    # Кросс-валидация для надежности
    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"\n{name}:")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"📊 Cross-val: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_model_name = name

# --------------------------- ФИНАЛЬНАЯ МОДЕЛЬ -----------------------------
print(f"\n🎯 ЛУЧШАЯ МОДЕЛЬ: {best_model_name} ({best_score:.4f})")

# Детальная оценка лучшей модели
y_pred = best_model.predict(X_val)
print("\n📈 ДЕТАЛЬНАЯ ОЦЕНКА:")
print(classification_report(y_val, y_pred))

# Матрица ошибок
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Погиб', 'Выжил'], 
            yticklabels=['Погиб', 'Выжил'])
plt.title('Матрица ошибок (Confusion Matrix)')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance для Random Forest
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    feature_imp = pd.Series(best_model.feature_importances_, index=feature_columns)
    feature_imp.sort_values().plot(kind='barh')
    plt.title('Важность признаков (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------- САБМИТ НА KAGGLE -----------------------------
X_test = test_df[feature_columns]
test_predictions = best_model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

submission.to_csv('titanic_submission.csv', index=False)
print(f"\n✅ САБМИТ СОЗДАН: titanic_submission.csv")
print(f"📊 Предсказано выживших: {test_predictions.sum()} из {len(test_predictions)}")
print(f"🎯 Процент выживших: {test_predictions.sum()/len(test_predictions)*100:.1f}%")

# Итоговая статистика
print(f"\n{'='*50}")
print("ИТОГОВАЯ СТАТИСТИКА")
print(f"{'='*50}")
print(f"Всего пассажиров в train: {len(train_df)}")
print(f"Выживших в train: {train_df['Survived'].sum()} ({train_df['Survived'].mean()*100:.1f}%)")
print(f"Лучшая модель: {best_model_name}")
print(f"Лучшая точность: {best_score:.4f}")