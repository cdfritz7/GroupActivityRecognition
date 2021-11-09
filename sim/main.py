import simpy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statistics import mean, variance

import pandas as pd

from joblib import load

# based on activities detected by Android Activity Recognition API
# activities = ['IN_VEHICLE', 'ON_BICYCLE', 'ON_FOOT', 'RUNNING', 'STILL', 'TILTING', 'WALKING', 'UNKNOWN']

VERBOSE = False

class ActivityChangedEvent:
    def __init__(self, origin_node, new_activity):
        self.origin_node = origin_node
        self.new_activity = new_activity

class SensorEvent:
    def __init__(self, origin_node, sample):
        self.origin_node = origin_node
        self.sample = sample

class Node:
    def __init__(self, scene, id, position, p=None, stream=None, frame_size=None):
        self.scene = scene
        self.id = id
        self.position = position
        self.stream = stream

        env.process(self.sensor_stream_schedule())

    def random_schedule(self, p):
        """ Randomly change this node's activity with probability p
        """
        while True:
            if p >= random.uniform(0, 1):
                activity_change = simpy.events.Timeout(
                    self.scene.get_environment(), 
                    delay=random.randint(0, 10), 
                    value=ActivityChangedEvent(self, activities[random.randint(0, len(activities)-1)])
                )
                yield activity_change
                self.scene.process_event(activity_change)
            
    def sensor_stream_schedule(self):
        """ Change this node's activity based on a replayed sensor stream with a 
        classifier making inferences on it
        """
        # load the classifier. the classifier has been trained in classifier.py and serialized using joblib.dump
        classifier = load('./data/activities/rf_model.joblib') 
        
        # initialize an empty array to batch samples into frames
        # once our batch is long enough, we will extract the features and try to infer the activity
        current_frame = []

        for index, sample in self.stream.iterrows():
            # raise a sensor event for this sample
            sensor_event = simpy.events.Timeout(
                self.scene.get_environment(), 
                delay=1, 
                value=SensorEvent(self, sample)
            )
            current_frame.append(sample)
            yield sensor_event
            self.scene.process_event(sensor_event)
            
            # we have a whole frame, time to classify it
            if len(current_frame) == frame_size:
                frame = pd.DataFrame().append(current_frame)
                current_frame = [] # reset the current batch of samples to empty

                # compute features from the current frame
                # dataframe has to look EXACTLY like a row of the dataframe we used to train the model
                fs = pd.DataFrame().from_dict({
                    'mean_x': [mean(frame['accelerometerAccelerationX(G)'])],
                    'mean_y': [mean(frame['accelerometerAccelerationY(G)'])],
                    'mean_z': [mean(frame['accelerometerAccelerationZ(G)'])],
                    'variance_x': [variance(frame['accelerometerAccelerationX(G)'])],
                    'variance_y': [variance(frame['accelerometerAccelerationY(G)'])],
                    'variance_z': [variance(frame['accelerometerAccelerationZ(G)'])]
                })

                # emit an activity event for the predicted activity
                activity_change = simpy.events.Timeout(
                    self.scene.get_environment(), 
                    delay=0, 
                    value=ActivityChangedEvent(self, classifier.predict(fs))
                )
                yield activity_change                
                self.scene.process_event(activity_change)

    def waypoint_schedule(self):
        """ Change this node's activity based on a schedule which
        provides specific activites to change to at certain time slices (waypoints)
        """
        for waypoint in self.stream:
            activity_change = simpy.events.Timeout(
                self.scene.get_environment(), 
                delay=waypoint, 
                value=ActivityChangedEvent(self, self.stream[waypoint])
            )
            yield activity_change                
            self.scene.process_event(activity_change)

class Scene:
    def __init__(self, env):
        print('initializing scene')
        print('')

        self.env = env
        self.event_batch = []
        self.time = env.now

    def get_environment(self):
        return self.env

    def process_event(self, event):
        """ Batches events for a time slice, processing them when the next time slice is reached
        """
        self.event_batch.append(event.value)

        # we're on the next time slice. time to process all of the events from the previous time slice
        if env.now > self.time:
            if VERBOSE:
                print('[Scene][%d] processing event batch' % self.time)

            for event in self.event_batch:
                if isinstance(event, ActivityChangedEvent):
                    print('[Scene][' + str(self.time) + '] ' + event.origin_node.id + ' is ' + str(event.new_activity))

                # elif isinstance(event, SensorEvent):
                #     print('[Scene][' + str(self.time) + '] ' + event.origin_node.id + ' sample ' + str(event.sample))

            self.event_batch = []
            self.time = env.now

def plot_nodes(nodes):
    for node in nodes:
        plt.scatter(node.position[0], node.position[1])
    plt.show()

if __name__ == '__main__':
    # sample rate of the replayed sensor data
    sample_rate = 100 # Hz
    frame_size = 100 # 1s

    # factor is the length of a step in the simulation, which should correspond to the
    # sample rate of the sensor stream
    # TODO strict is set to false since computing each sample of the 100Hz stream takes too long to do
    #      i.e. we can't simulate the sensor sample in 0.01s (takes closer to ~0.03s)
    env = simpy.rt.RealtimeEnvironment(factor=1/sample_rate, strict=False)
    scene = Scene(env)

    # n1script = {
    #     0: 'WALKING',
    #     1: 'STILL',
    #     10: 'WALKING'
    # }

    # n2script = {
    #     3: 'RUNNING',
    #     10: 'STILL',
    #     4: 'WALKING'
    # }

    df1 = pd.read_csv('./data/streams/01.csv') # 100Hz sample rate
    df2 = pd.read_csv('./data/streams/00.csv')
    # print(df)

    Node(scene, 'node1', (0, 0), stream=df1)
    Node(scene, 'node2', (0, 0), stream=df2)
    # nodes = [
    #     Node(scene, 'node1', (0, 0)),
    #     Node(scene, 'node2', (50, 50))
    # ]

    env.run()