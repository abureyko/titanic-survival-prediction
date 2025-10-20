import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('train.csv')
test_df = test_df = pd.read_csv('test.csv')


# --------------------------- Подготовка данных ------------------------------

# Заполняем пропуски
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna('C')

# Преобразуем категориальные признаки в числа
train_df['Sex'] = train_df['Sex'].map({'male' : 0, 'female' : 1})
train_df['Embarked'] = train_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Создаем новые признаки
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)

# Создаём возрастные категории (не используются в финальной модели, тк нет пользы для модели)
def get_age_category(age):
    if age < 12: return 0  # child
    elif age < 18: return 1  # teen  
    elif age < 30: return 2  # young adult
    elif age < 50: return 3  # adult
    else: return 4  # senior

train_df['AgeCategory'] = train_df['Age'].apply(get_age_category)

# Создаем family survival feature
train_df['FamilyId'] = train_df['Name'].str.split(',').str[0] + '_' + train_df['FamilySize'].astype(str)
def get_family_survival(row, df):
    family_members = df[(df['FamilyId']==row['FamilyId'])&(df['PassengerId']!=row['PassengerId'])]
    if len(family_members) > 0:
        return family_members['Survived'].mean()
    return 0.5
train_df['FamilySurvivalRate'] = train_df.apply(lambda row: get_family_survival(row, train_df), axis=1)


# Иногда простые признаки работают лучше сложных
train_df['IsLargeFamily'] = (train_df['FamilySize'] > 4).astype(int)

# --------------------------- Feature Engineering ----------------------------
# (не используются в финальной модели, тк нет пользы для модели)

# Попытка 1: Age categories - не дали прироста качества
# train_df['AgeCategory'] = train_df['Age'].apply(get_age_category)

# Попытка 2: Family Survival Rate - переобучался и ухудшал результаты  
# train_df['FamilyId'] = train_df['Name'].str.split(',').str[0] + '_' + train_df['FamilySize'].astype(str)
# train_df['FamilySurvivalRate'] = train_df.apply(lambda row: get_family_survival(row, train_df), axis=1)

# --------------------------- ФИНАЛЬНЫЕ ФИЧИ --------------------------------
# Отобраны по результатам валидации - дают максимальный accuracy 
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
           'IsAlone', 'IsLargeFamily', 'Title']  

# -------------------------- Обучение -----------------------------

X = train_df[features]
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ФИНАЛЬНАЯ МОДЕЛЬ
model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
model.fit(X_train, y_train)

print(f"ACCURACY: {model.score(X_val, y_val):.3f}")




# Важность признаков (для портфолио)
feature_importance = pd.DataFrame({
    'feature': features,
    'weight': model.coef_[0]
}).sort_values('weight', key=abs, ascending=False)

print("\nВАЖНОСТЬ ПРИЗНАКОВ:")
print(feature_importance)



# --------------------------- САБМИТ НА KAGGLE -----------------------------

# Обработка test данных 
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna('C')
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df['Embarked'] = test_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)
test_df['IsLargeFamily'] = (test_df['FamilySize'] > 4).astype(int)


X_test = test_df[features]
test_predictions = model.predict(X_test)

# Создание файла для сабмита
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

# Сохраняем в CSV
submission.to_csv('titanic_submission.csv', index=False)
print("\nСабмит файл создан: titanic_submission.csv")
print(f"Предсказано выживших: {test_predictions.sum()} из {len(test_predictions)}")