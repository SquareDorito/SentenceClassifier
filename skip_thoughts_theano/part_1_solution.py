"""
An example solution written by Peter Frisch
Created on Tue Aug 29 09:57:34 2017
****** Not yet verified ******
"""


class HMM:
    def __init__(self, sensor_model, transition_model, num_states):
        self.sensor_model = sensor_model
        self.transition_model = transition_model
        self.num_states = num_states
        self.distribution = [1] * num_states
        self.distribution = [x / float(sum(self.distribution)) for x in self.distribution]
        self.time = 0

    def tell(self, observation):
        """
        Takes in an observation and records it. A good implementation will keep track of the current
        time-step and increment it for each observation.

        observation: The observation for the current time-step

        returns: None
        """
        self.time += 1
        new_distribution = [0] * self.num_states
        for s1 in range(self.num_states):
            sensor_term = self.sensor_model(observation, s1)
            prior = 0
            for s0 in range(self.num_states):
                prior += self.transition_model(s0, s1) * self.distribution[s0]
            new_distribution[s1] = sensor_term * prior

        self.distribution = new_distribution
        self.distribution = [x / float(sum(self.distribution)) for x in self.distribution]

    def ask(self, time):
        """
        Takes in a time-step that is greater than or equal to the current time-step and outputs a
        probability distribution (represented as a list) over states for that time-step. The index
        of the probability is the observation it corresponds to.

        time: the time-step to get the observation distribution for

        returns: A list of probabilities for the given time-step
        """
        if time < self.time:
            print('Error: give a time in the present or future')
            return
        prediction = list(self.distribution)
        for t in range(time - self.time):
            new_distribution = [0] * self.num_states
            for s1 in range(self.num_states):
                for s0 in range(self.num_states):
                    new_distribution[s1] += self.transition_model(s0, s1) * prediction[s0]
            prediction = new_distribution

        return prediction
