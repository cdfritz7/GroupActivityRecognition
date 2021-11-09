import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig
import scipy.stats as stats
from statistics import mean, variance

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

from joblib import dump

def plot_feature_space(df, activities, features):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)

    fig.add_axes(ax)

    for activity in activities:
        ax.scatter(df.loc[df['label'] == activity, features[0]], df.loc[df['label'] == activity, features[1]], df.loc[df['label'] == activity, features[2]], s=10)

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])

    plt.title('3D Feature Space Comprised of ' + features[0] + ', ' + features[1] + ', and ' + features[2])
    plt.savefig('./figs/feature_space.png', dpi=300)
    plt.show()

def plot_segments(frames):
    for frame in frames:
        plt.plot(range(0, len(frame)), frame)
        plt.show()

def segment(data, frameSize, overlap=0):
    """ Accepts a one-dimensional array and returns an array of arrays, where each array 
    is a segment of the input data of size frameSize. Optional overlap parameter specifies 
    the amount of overlap between adjacent frames.
    """
    ary = []

    for i in range(0, len(data), frameSize-overlap):
        ary.append(data[i:i+frameSize])

    return ary

def feature_extract(frames, features, label):
    """ Accepts a tuple of three arrays representing the segmentations of x, y, 
    and z raw accelerometer data as well as the label for the data (NOTE: this means
    that this function assumes all frames in the input data represent an activity of 
    the same single classification) and a tuple of functions to calculate features. Returns a 
    dataframe containing feature vectors, where each row is associated with a feature 
    vector for each segment of the input data. Input arrays must be of the same length.
    """
    if not (len(frames[0]) == len(frames[1]) and len(frames[1]) == len(frames[2])):
        raise ValueError('Input arrays of segments are not the same length.')

    fs = {}
    labels = []
    
    for i in range(0, len(frames[0])):
        for func in features:
            xkey = func.__name__ + '_x'            
            ykey = func.__name__ + '_y'
            zkey = func.__name__ + '_z'

            if xkey not in fs:
                fs[xkey] = []
                fs[ykey] = []
                fs[zkey] = []

            fs[xkey].append(func(frames[0][i]))
            fs[ykey].append(func(frames[1][i]))
            fs[zkey].append(func(frames[2][i]))

        labels.append(label)

    fs['label'] = labels 

    return pandas.DataFrame(data=fs)

def build_training_data(activities, features, frameSize, overlap=0):
    """ Accepts a list of activities (corresponding with a named data set in /data/), 
    a tuple of functions representing each desired feature, the desired frame size, and
    an optional frame overlap parameter.
    Returns a data frame of labeled instances.
    """
    df = pandas.DataFrame()

    for activity in activities:
        tmp = pandas.read_csv('./data/activities/' + activity + '.csv')

        # filter noise with median filtering and segment
        xs = segment(sig.medfilt(tmp['accelerometerAccelerationX(G)']), frameSize, overlap)
        ys = segment(sig.medfilt(tmp['accelerometerAccelerationY(G)']), frameSize, overlap)
        zs = segment(sig.medfilt(tmp['accelerometerAccelerationZ(G)']), frameSize, overlap)

        df = df.append(feature_extract((xs, ys, zs), features, activity))

    return df

def fit_random_forest(df):
    """ Part A: fits the given training data to a discriminative learning algorithm (Random Forest)
    and tests accuracy using 10-fold cross validation and split train/test.
    """
    x = df.drop('label', axis=1)
    y = df.label

    print(x)
    print('Fitting random forest model to data with 10-fold cross validation...')

    rf = RandomForestClassifier()
    scores = cross_val_score(rf, x, y, cv=10)
    accuracy = (sum(scores)/len(scores)) * 100

    print('  Result: ' + str(accuracy) + '% accuracy (10-fold cross validation)')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    print('  Result: ' + str(rf.score(x_test, y_test)*100) + '% accuracy (split train/test)')

    dump(rf, './data/activities/rf_model.joblib')


def main():
    activities = ['walking', 'running', 'sitting']
    features = (mean, variance)
    frameSize = 100 # 1s for a 100Hz sample rate
    overlap = 75 # %

    df = build_training_data(activities, features, frameSize, overlap)
    print(df)
    fit_random_forest(df)

if __name__ == '__main__':
    main()