import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from lab_s01_utils import print_function_name


def todo_2():
    print_function_name()

    digits = datasets.load_digits()

    print(digits.DESCR)
    print(f'digits data:\n {digits.data},\n len: {len(digits.data)}')
    print(f'digits target:\n {digits.target},\n len: {len(digits.target)}')
    print(f'digits target_names:\n {digits.target_names}')
    print(f'digits images:\n {digits.images},\n len: {len(digits.images)}')

    print(digits.data[-1])
    print(digits.images[-1])

    index = -1
    print(digits.target[index])
    plt.imshow(digits.images[index], cmap=plt.cm.gray_r)
    plt.show()

    plt.imshow([digits.data[index]], cmap=plt.cm.gray_r)
    plt.show()
    ## ADD
    print(type(digits))
    print(type(digits['data']))
    print(type(digits['images']))

    print(digits['images'])

    print(digits['DESCR'])

    axs, fig = plt.subplots(len(digits['target_names']), 5)

    for class_n in digits['target_names']:
        print(class_n)
        for col in range(5):
            axs[class_n][col].imshow(digits['images'][digits['target']==class_n][col])

    plt.show()


def todo_3():
    print_function_name()

    digits = datasets.load_digits()

    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.33, random_state=42
    )

    print(f'{len(digits.data)=}')
    print(f'{len(X_train.data)=}')
    print(f'{len(X_test.data)=}')


def todo_4():
    print_function_name()

    faces = datasets.fetch_olivetti_faces()
    # alternative:
    #  X, y = datasets.fetch_olivetti_faces(return_X_y=True)

    print(faces.DESCR)
    print(f'{faces.data=}')
    print(f'{faces.target=}')

    image_shape = (64, 64)
    n_row, n_col = 2, 3
    n_components = n_row * n_col

    def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
        plt.figure(figsize=(2. * n_col, 2.26 * n_row))
        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=cmap,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

    print(f'target: {faces.target[:n_components]}')

    plot_gallery('Olivetti faces', faces.images[:n_components])
    plt.show()
    ## ADD3
    X, Y = [faces['data'], faces['target']]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y)
    print(f'target: {Y_test[:n_components]}')
    plot_gallery('Olivetti faces test', X_test[:n_components])
    plt.show()


def todo_5():
    print_function_name()

    diabetes = datasets.load_diabetes(as_frame=True)
    print(diabetes.DESCR)
    print(f'diabetes data:\n {diabetes.data},\n len: {len(diabetes.data)}')
    print(f'diabetes target:\n {diabetes.target},\n len: {len(diabetes.target)}')
    print(f'diabetes feature_names:\n {diabetes.feature_names}')
    print(diabetes.frame.head(5))
    print(diabetes.frame.info())


def todo_6():
    print_function_name()

    X, y = datasets.make_classification(
        n_features=3, n_redundant=0, n_repeated=0, n_informative=3,
        n_classes=3,
        class_sep=3.0,
        random_state=1,
        n_clusters_per_class=2
    )
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
    plt.show()

    print(f'{y=}')
    print("now")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def todo_6_2():
    print_function_name()

    x, y = datasets.make_classification(
        n_samples=100,
        n_features=3,
        n_informative=3, n_redundant=0, n_repeated=0,
        n_classes=6,
        n_clusters_per_class=1,
        class_sep=5.0,
        flip_y=0.0
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def todo_7():
    print_function_name()

    openml_dataset = datasets.fetch_openml(data_id=40536, as_frame=True)
    print(f'{type(openml_dataset)=}')
    print(f'{openml_dataset=}')


def todo_8():
    print_function_name()

    battery_problem_data = np.loadtxt(fname='./../data/battery_problem_data.csv', delimiter=',')
    print(f'{battery_problem_data=}')
    # ADD
    import pandas as pd
    df = pd.read_csv('./../data/battery_problem_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(df['charging'], df['watching'],random_state=42, test_size=0.3)



def todo_9_10():
    print_function_name()

    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree

    # OR gate
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    print(f'{clf.predict([[0, 0]])=}')
    print(f'{clf.predict([[0, 1]])=}')
    print(f'{clf.predict([[1, 0]])=}')
    print(f'{clf.predict([[1, 1]])=}')

    tree.plot_tree(clf, feature_names=['X1', 'X2'], filled=True, class_names=['0', '1'])
    plt.show()

    # AND gate

    X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
    y = [0, 0, 0, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict([[1, 1]]))   # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.
    print(clf.predict([[0, 1]]))

    tree.plot_tree(clf, feature_names=['X1', 'X2'], filled=True, class_names=['0', '1'])
    plt.show()

def main():
    # todo_2()
    # todo_3()
    # todo_4()
    # todo_5()
    # todo_6()
    # todo_6_2()
    # todo_7()
    # todo_8()
    todo_9_10()


if __name__ == '__main__':
    main()
