# main.py
#
# Implements a near-real-time discrete event simulator that replays smartphone accelerometer 
# sensor streams associated with nodes. These sensor streams are classified using the 
# classifier in classifier.py with a fixed frame size.

import simpy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statistics import mean, variance, mode
import pandas as pd
from joblib import load
import requests

USE_API = False
GROUP_ACCURACY = True

steps = 335

class ActivityChangedEvent:
    """ Raised when an activity inference has been made on a sensor stream
    """
    def __init__(self, origin_node, new_activity, ground_truth='unknown'):
        self.origin_node = origin_node
        self.new_activity = new_activity
        self.ground_truth = ground_truth

class SensorEvent:
    """ Raised when a single sample of sensor data has been processed from a stream
    """
    def __init__(self, origin_node, sample):
        self.origin_node = origin_node
        self.sample = sample

class Node:
    """ Defines a single actor in the simulation with its own sensor stream. Each Node is,
    for example, a single individual who might be performing an activity that has an associated
    sensor stream that we use to make inferences 
    """
    def __init__(self, scene, id, position, p=None, stream=None, frame_size=None, frame_overlap=0):
        self.scene = scene
        self.id = id
        self.position = position # TODO position is currently unused
        self.stream = stream
        self.frame_size = frame_size
        self.frame_overlap = frame_overlap
        self.p = p

        if stream is not None and self.frame_size is not None:
            self.scene.get_environment().process(self.sensor_stream_schedule())
        elif p is not None:
            self.scene.get_environment().process(self.controlled_accuracy_schedule(p))

    def controlled_accuracy_schedule(self, p):
        """ Generate activity changes that are correct with given probability p
        """
        for _ in range(0, steps):
            activities = ['sitting', 'standing', 'walking']

            # randomly choose a ground_truth activity
            ground_truth = [activities[random.randint(0, len(activities)-1)]]

            if p >= random.uniform(0, 1): # emit a correct prediction
                prediction = ground_truth
            else: # emit an incorrect prediction
                activities.remove(ground_truth[0])
                prediction = [activities[random.randint(0, len(activities)-1)]]
                
            activity_change = simpy.events.Timeout(
                self.scene.get_environment(), 
                delay=1, 
                value=ActivityChangedEvent(self, prediction, ground_truth=ground_truth)
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
            if len(current_frame) == self.frame_size:
                frame = pd.DataFrame().append(current_frame)
                ground_truth = frame['groundTruthLabel'].tolist()
                
                if self.frame_overlap > 0:
                    for i in range(0, self.frame_size - self.frame_overlap):
                        current_frame.pop(0)
                else:
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
                    value=ActivityChangedEvent(self, classifier.predict(fs), ground_truth=ground_truth)
                )
                yield activity_change                
                self.scene.process_event(activity_change)

class Scene:
    def __init__(self, env):
        self.env = env
        self.event_batch = []
        self.time = env.now
        
    def get_environment(self):
        return self.env

    def post_api(payload):
        # make a POST request to the API with the updated activity
        r = requests.post('http://localhost:8080/sendActivityFromUser', data=payload)

    def process_event(self, event):
        """ Batches events for a time slice, processing them when the next time slice is reached
        """
        self.event_batch.append(event.value)

        # we're on the next time slice. time to process all of the events from the previous time slice
        if self.env.now > self.time:
            # aggregate ground truths, inferences, and correct count (for analyzing group accuracy)
            if GROUP_ACCURACY:
                ground_truths = []
                inferences = []
                iscorrect_inferences = 0

            for i, event in enumerate(self.event_batch):
                if isinstance(event, ActivityChangedEvent):
                    # print the inferred activity
                    activity = str(event.new_activity[0])
                    
                    # determine the ground truthiness of the frame:
                    # how many samples in the frame represent which labels
                    truthiness = {}
                    for label in event.ground_truth:
                        if label not in truthiness:
                            truthiness[label] = 1
                        else:
                            truthiness[label] += 1

                    # ground truth is whichever label is most represented by the samples in the frame
                    frame_ground_truth = max(truthiness, key=truthiness.get)

                    # log if the inferred activity is correct/incorrect for the node
                    if not GROUP_ACCURACY:
                        print(event.origin_node.id + ',' + frame_ground_truth + ',' + activity + ',' + ('1' if frame_ground_truth == activity else '0') + ',' + str(event.origin_node.p))
                    else:
                        ground_truths.append(frame_ground_truth)
                        inferences.append(activity)
                        iscorrect_inferences += 1 if frame_ground_truth == activity else 0

                        # if event.origin_node.id == 'node5': # TODO this doesn't scale past lab_scenario since it assumes node5 is the last one changing
                        if i == len(self.event_batch)-1: # use with controlled_accuracy
                            actual_mode = mode(ground_truths)
                            predicted_mode = mode(inferences)
                            
                            print(actual_mode + ',' + predicted_mode + ',' + ('1' if actual_mode == predicted_mode else '0') + ',' + str(iscorrect_inferences) + ',' + str(len(inferences)) + ',' + str(event.origin_node.p))
                    
                            ground_truths = []
                            inferences = []
                            iscorrect_inferences = 0

                    if USE_API:
                        post_api({
                            'lat': event.origin_node.position[0], 
                            'lng': event.origin_node.position[1],
                            'userId': event.origin_node.id,
                            'label': activity
                        })

            self.event_batch = []
            self.time = self.env.now

def lab_scenario_sim():
    """ Run the simulator with 5 different replayed sensor streams gathered in the MPC lab.
    Nodes 1-4 walk briefly and then sit
    Node 5 alternates between walking and standing
    """
    # sample rate of the replayed sensor data
    sample_rate = 100 # Hz
    frame_size = 100 # 1s

    # factor is the length of a step in the simulation, which should correspond to the
    # sample rate of the sensor stream
    # TODO strict is set to false since computing each sample of the 100Hz stream takes too long to do
    #      i.e. we can't simulate the sensor sample in 0.01s (takes closer to ~0.03s)
    # env = simpy.rt.RealtimeEnvironment(factor=1/sample_rate, strict=False)
    
    env = simpy.Environment()
    scene = Scene(env)

    # read recorded sensor streams into dataframes. these are the sensor streams that will be 
    # replayed when the simulation is run
    df1 = pd.read_csv('./data/streams/lab_scenario/labeled/walk-to-sit-01.csv') # 100Hz sample rate
    df2 = pd.read_csv('./data/streams/lab_scenario/labeled/walk-to-sit-02.csv')
    df3 = pd.read_csv('./data/streams/lab_scenario/labeled/walk-to-sit-03.csv')
    df4 = pd.read_csv('./data/streams/lab_scenario/labeled/walk-to-sit-04.csv')
    df5 = pd.read_csv('./data/streams/lab_scenario/labeled/alternate-walking-standing.csv')

    overlap = 25 # %

    Node(scene, 'node1', (0.5, 0.5), stream=df1, frame_size=100, frame_overlap=overlap)
    Node(scene, 'node2', (0.5, 0.5), stream=df2, frame_size=100, frame_overlap=overlap)
    Node(scene, 'node3', (0.5, 0.5), stream=df3, frame_size=100, frame_overlap=overlap)
    Node(scene, 'node4', (0.5, 0.5), stream=df4, frame_size=100, frame_overlap=overlap)
    Node(scene, 'node5', (0.5, 0.5), stream=df5, frame_size=100, frame_overlap=overlap)

    env.run()

def controlled_accuracy_sim(ps):
    """ Run the simulator with multiple nodes that have probabilistic accuracy – that is,
    we have the ability to set their accuracy and observe the effect on the group accuracy.
    Creates one node for each p value in ps, where p represents the probability that any 
    inference that node's classifer makes is correct.
    """
    env = simpy.Environment()
    scene = Scene(env)

    for i, p in enumerate(ps):
        Node(scene, 'node' + str(i+1), (0.5, 0.5), p=p)

    env.run()

def mult_controlled_accuracy_sim(n):
    for p in np.linspace(0, 1, 101):
        controlled_accuracy_sim([p] * n)

if __name__ == '__main__':
    # print headers
    if GROUP_ACCURACY:
        # actual_mode: what is the mode of the ground truth labels for the frame
        # predicted_mode: what is the mode of the inferred labels
        # iscorrect_mode: 1 if the mode is correct, 0 if not
        # iscorrect_inferences: number of correct inferences
        # total_nodes: number of nodes
        print('actual_mode,predicted_mode,iscorrect_mode,iscorrect_inferences,total_nodes,p')
    else:    
        print('node,ground_truth,inference,iscorrect,p')

    # lab_scenario_sim()
    mult_controlled_accuracy_sim(1000)