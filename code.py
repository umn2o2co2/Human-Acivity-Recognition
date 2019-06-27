from matplotlib import pyplot as plt



import numpy as np

import pandas as pd

import seaborn as sns

#import coremltools

from scipy import stats

#from IPython.display import display, HTML



from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn import preprocessing



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Reshape

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils


# Set some standard parameters upfront

pd.options.display.float_format = '{:.1f}'.format

#sns.set() # Default seaborn look and feel
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

plt.style.use('ggplot')

print('keras version ', keras.__version__)

# Same labels will be reused throughout the program

LABELS = ['Downstairs',

          'Jogging',

          'Sitting',

          'Standing',

          'Upstairs',

          'Walking']

# The number of steps within one time segment

TIME_PERIODS = 80

# The steps to take from one segment to the next; if this value is equal to

# TIME_PERIODS, then there is no overlap between the segments

STEP_DISTANCE = 40
#######
def read_data(file_path):



    column_names = ['user-id',

                    'activity',

                    'timestamp',

                    'x-axis',

                    'y-axis',

                    'z-axis']

    df = pd.read_csv(file_path,

                     header=None,

                     names=column_names)

    # Last column has a ";" character which must be removed ...

    df['z-axis'].replace(regex=True,

      inplace=True,

      to_replace=r';',

      value=r'')

    # ... and then this column must be transformed to float explicitly

    df['z-axis'] = df['z-axis'].apply(convert_to_float)

    # This is very important otherwise the model will not fit and loss

    # will show up as NAN

    df.dropna(axis=0, how='any', inplace=True)



    return df



def convert_to_float(x):



    try:

        return np.float(x)

    except:

        return np.nan

 

def show_basic_dataframe_info(dataframe):



    # Shape and how many rows and columns

    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))

    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))



# Load data set containing all the data from csv

df = read_data('WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')

# Describe the data

show_basic_dataframe_info(df)

print(df.head(20))

# Show how many training examples exist for each of the six activities

df['activity'].value_counts().plot(kind='bar',

                                   title='Training Examples by Activity Type')

#plt.show()

# Better understand how the recordings are spread across the different

# users who participated in the study

df['user-id'].value_counts().plot(kind='bar',

                                  title='Training Examples by User')

#plt.show()

#########

def plot_activity(activity, data):



    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,

         figsize=(15, 10),

         sharex=True)

    plot_axis(ax0, data['timestamp'], data['x-axis'], 'X-Axis')

    plot_axis(ax1, data['timestamp'], data['y-axis'], 'Y-Axis')

    plot_axis(ax2, data['timestamp'], data['z-axis'], 'Z-Axis')

    plt.subplots_adjust(hspace=0.2)

    fig.suptitle(activity)

    plt.subplots_adjust(top=0.90)

    plt.show()



def plot_axis(ax, x, y, title):



    ax.plot(x, y, 'r')

    ax.set_title(title)

    ax.xaxis.set_visible(False)

    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])

    ax.set_xlim([min(x), max(x)])

    ax.grid(True)



for activity in np.unique(df['activity']):

    subset = df[df['activity'] == activity][:180]

    plot_activity(activity, subset)     

#######################
# Define column name of the label vector

LABEL = 'ActivityEncoded'

# Transform the labels from String to Integer via LabelEncoder

le = preprocessing.LabelEncoder()

# Add a new column to the existing DataFrame with the encoded values

df[LABEL] = le.fit_transform(df['activity'].values.ravel())

###########

# Differentiate between test set and training set

df_test = df[df['user-id'] > 28]

df_train = df[df['user-id'] <= 28]

# Normalize features for training data set (values between 0 and 1)

# Surpress warning for next 3 operation

pd.options.mode.chained_assignment = None  # default='warn'

df_train['x-axis'] = df_train['x-axis'] / df_train['x-axis'].max()

df_train['y-axis'] = df_train['y-axis'] / df_train['y-axis'].max()

df_train['z-axis'] = df_train['z-axis'] / df_train['z-axis'].max()

# Round numbers

df_train = df_train.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})

#####for test

# Normalize features for training data set (values between 0 and 1)

# Surpress warning for next 3 operation

pd.options.mode.chained_assignment = None  # default='warn'

df_test['x-axis'] = df_test['x-axis'] / df_test['x-axis'].max()

df_test['y-axis'] = df_test['y-axis'] / df_test['y-axis'].max()

df_test['z-axis'] = df_test['z-axis'] / df_test['z-axis'].max()

# Round numbers

df_test = df_test.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
#####
def create_segments_and_labels(df, time_steps, step, label_name):



    # x, y, z acceleration as features

    N_FEATURES = 3

    # Number of steps to advance in each iteration (for me, it should always

    # be equal to the time_steps in order to have no overlap between segments)

    # step = time_steps

    segments = []

    labels = []

    for i in range(0, len(df) - time_steps, step):

        xs = df['x-axis'].values[i: i + time_steps]

        ys = df['y-axis'].values[i: i + time_steps]

        zs = df['z-axis'].values[i: i + time_steps]

        # Retrieve the most often used label in this segment

        label = stats.mode(df[label_name][i: i + time_steps])[0][0]

        segments.append([xs, ys, zs])

        labels.append(label)



    # Bring the segments into a better shape

    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)

    labels = np.asarray(labels)



    return reshaped_segments, labels



x_train, y_train = create_segments_and_labels(df_train,

                                              TIME_PERIODS,

                                              STEP_DISTANCE,

                                              LABEL)
x_test, y_test = create_segments_and_labels(df_test,

                                              TIME_PERIODS,

                                              STEP_DISTANCE,

                                              LABEL)
                                        
print('x_train shape: ', x_train.shape)

print(x_train.shape[0], 'training samples')

print('y_train shape: ', y_train.shape)


# Set input & output dimensions

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]

num_classes = le.classes_.size

print(list(le.classes_))

input_shape = (num_time_periods*num_sensors)

x_train = x_train.reshape(x_train.shape[0], input_shape)

print('x_train shape:', x_train.shape)

print('input_shape:', input_shape)


x_train = x_train.astype('float32')

y_train = y_train.astype('float32')


y_train_hot = np_utils.to_categorical(y_train, num_classes)

print('New y_train shape: ', y_train_hot.shape)

################
num_time_periods_test, num_sensors_test = x_test.shape[1], x_test.shape[2]

num_classes_test = le.classes_.size

print(list(le.classes_))

input_shape_test = (num_time_periods_test*num_sensors_test)

x_test = x_test.reshape(x_test.shape[0], input_shape_test)

print('x_test shape:', x_test.shape)

print('input_shape:', input_shape_test)


x_test = x_test.astype('float32')

y_test = y_test.astype('float32')


y_test_hot = np_utils.to_categorical(y_test, num_classes_test)

print('New y_test shape: ', y_test_hot.shape)
############
from sklearn.svm import SVC
model_svm = SVC(kernel='rbf', probability=True) 
model_svm.fit(x_train, y_train_hot) 

y_pred = model_svm.predict(x_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.6657047387606319




#############################
#DNN
model_dnn = Sequential()
# Remark: since coreml cannot accept vector shapes of complex shape like
# [80,3] this workaround is used in order to reshape the vector internally
# prior feeding it into the network
model_dnn.add(Reshape((TIME_PERIODS, 3), input_shape=(input_shape,)))
model_dnn.add(Dense(100, activation='relu'))
model_dnn.add(Dense(100, activation='relu'))
model_dnn.add(Dense(100, activation='relu'))
model_dnn.add(Flatten())
model_dnn.add(Dense(num_classes, activation='softmax'))
print(model_dnn.summary())

#
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}dnn.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

#load weights
model_dnn.load_weights('best_model.05-0.64dnn.h5')

model_dnn.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
BATCH_SIZE = 400
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_dnn.fit(x_train,
                      y_train_hot,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

#16694/16694 [==============================] - 1s 67us/step - loss: 0.3414 - acc: 0.8720 - val_loss: 0.8046 - val_acc: 0.7329
#
plt.figure(figsize=(6, 4))
plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# Print confusion matrix for training data
y_pred_train = model_dnn.predict(x_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(y_train, max_y_pred_train))
#
y_pred_test = model_dnn.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
print("Accuracy:",metrics.accuracy_score(y_test, max_y_pred_test))
#Accuracy: 0.7252430133657352
#max_y_test = np.argmax(y_test, axis=0)

def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

show_confusion_matrix(y_test, max_y_pred_test)

print(classification_report(y_test, max_y_pred_test))