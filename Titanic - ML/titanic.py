import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#filling missing age 
def fill_age(row, median_age):
   if pd.isnull(row['Age']):
       return median_age[row['Pclass']]
   return row['Age']

#replace gender with 1 or 0
def replace_gender(gender):
    if gender == "male":
        return 1
    return 0

def cleaning_data(file):
    titanic_df = pd.read_csv(file)
    titanic_df.drop(["PassengerId", "Name", "Parch", "Ticket", "Cabin"], axis = 1, inplace = True)

    titanic_df["Embarked"].fillna("S", inplace = True)
    titanic_df.drop("Embarked", axis = 1, inplace= True)

    median_age = titanic_df.groupby(by = "Pclass")["Age"].median()
    titanic_df['Age'] = titanic_df.apply(fill_age, median_age = median_age, axis = 1)
    titanic_df["Sex"] = titanic_df["Sex"].apply(replace_gender)
    return titanic_df

#Maching learning
def create_model(titanic_df):
    x = titanic_df.drop("Survived", axis = 1)
    y = titanic_df["Survived"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    model = KNeighborsClassifier(n_neighbors = 5)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    cm = confusion_matrix(y_test, prediction)
    accuracy = accuracy_score(y_test, prediction)*100

    return y_test, prediction, cm, accuracy


def plot_data(accuracy, titanic_df, cm):
    fig, axs = plt.subplots(1, 3, figsize = (13, 6))
    axs[0].bar(["Accuracy"], [accuracy], color = "purple")
    axs[0].set_ylim(0, 100)
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy of the Model")

    # Plotting confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[1], xticklabels=["Not survived", "Survived"], yticklabels=["Not survived", "Survived"])
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("Actual")
    axs[1].set_title("Confusion Matrix")

    # Plotting age distribution by survival
    sns.violinplot(x="Survived", y="Age", data=titanic_df, palette="muted", ax=axs[2])
    axs[2].set_xlabel("Survived")
    axs[2].set_ylabel("Age")
    axs[2].set_title("Age Distribution by Survival")
    axs[2].set_xticks([0, 1])
    axs[2].set_xticklabels(["Not survived", "Survived"])

    plt.show()


if __name__ in "__main__":
    titanic_df = cleaning_data("titanic.csv")
    y_test, prediction, cm, accuracy = create_model(titanic_df)
    plot_data(accuracy, titanic_df, cm)

